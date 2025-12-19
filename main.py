import torch
import numpy as np
import cv2
from tqdm import tqdm
from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model
import multiprocessing as mp
import os
from model.utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def activate_iwo_in_model(model, epoch):
    for m in model.modules():
        if isinstance(m, MBRConv5):
            m.set_epoch(epoch)
            # print(m)
        if isinstance(m, MBRConv3):
            m.set_epoch(epoch)
            # print(m)
        if isinstance(m, MBRConv1):
            m.set_epoch(epoch)
            # print(m)

def train(opt, logger):
    logger.info('task: {}, model task: {}'.format(opt.task, opt.model_task))

    train_loader, valid_loader = import_loader(opt)
    lr = float(opt.config['train']['lr'])
    lr_warmup = float(opt.config['train']['lr_warmup'])

    loss_warmup = import_loss('warmup')
    loss_training = import_loss(opt.model_task)
    net = import_model(opt)
    # logger.info(net)
    num_params = count_parameters(net)
    print("Total number of parameters: ", num_params)

    net.train()
    # Phase Warming-up
    if opt.config['train']['warmup']:
        logger.info('start warming-up')

        optim_warm = torch.optim.Adam(net.parameters(), lr_warmup, weight_decay=0)
        epochs = opt.config['train']['warmup_epoch']
        for epo in range(epochs):
            loss_li = []
            for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
                img_inp = img_inp.to(opt.device)
                img_gt = img_gt.to(opt.device)
                optim_warm.zero_grad()
                warmup_out1, warmup_out2 = net.forward_warm(img_inp)
                loss = loss_warmup(img_inp, img_gt, warmup_out1, warmup_out2)
                loss.backward()
                optim_warm.step()
                loss_li.append(loss.item())

            logger.info('epoch: {}, train_loss: {}'.format(epo+1, sum(loss_li)/len(loss_li)))
            torch.save(net.state_dict(), r'{}/model_pre.pkl'.format(opt.save_model_dir))
        logger.info('warming-up phase done')

    # Phase Training
    best_psnr = 0
    epochs = int(opt.config['train']['epoch'])
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=0)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7)

    logger.info('start training')
    for epo in range(epochs):
        # if epo == 0:
        #     logger.info("Activate IWO...")
        #     activate_iwo_in_model(net, epo)
        loss_li = []
        test_psnr = []
        net.train()

        attempt = 10
        skip_epoch = False
        for i in range(attempt):
            try:
                for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
                    img_inp = img_inp.to(opt.device)
                    img_gt = img_gt.to(opt.device)
                    out = net(img_inp)
                    loss = loss_training(out, img_gt)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_li.append(loss.item())
                lr_sch.step()
                break
            except Exception as e:
                print(f"fail training with error:{e}")
                if i == attempt - 1:
                    print("skip this train epoch...")
                    skip_epoch = True
                    break
                else:
                    print("rerun train epoch...")
                    train_loader, _ = import_loader(opt)

        # Validation
        # print("Start validation...")
        net.eval()
        for img_inp, img_gt, _ in tqdm(valid_loader, ncols=80):
            img_inp = img_inp.to(opt.device)
            img_gt = img_gt.to(opt.device)
            with torch.no_grad():
                out = net(img_inp)
                mse = ((out - img_gt)**2).mean((2, 3))
                psnr = (1 / mse).log10().mean() * 10
            test_psnr.append(psnr.item())
        mean_psnr = sum(test_psnr)/len(test_psnr)

        if (epo+1) % int(opt.config['train']['save_every']) == 0:
            torch.save(net.state_dict(), r'{}/model_{}.pkl'.format(opt.save_model_dir, epo+1))

        if not skip_epoch:
            logger.info('epoch: {}, training loss: {}, validation psnr: {}'.format(
                epo+1, sum(loss_li) / len(loss_li), sum(test_psnr) / len(test_psnr)
            ))

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(net.state_dict(), r'{}/model_best.pkl'.format(opt.save_model_dir))
            if opt.config['train']['save_slim']:
                net_slim = net.slim().to(opt.device)
                torch.save(net_slim.state_dict(), r'{}/model_best_slim.pkl'.format(opt.save_model_dir))
                logger.info('best model saved and re-parameterized in epoch {}'.format(epo+1))
            else:
                logger.info('best model saved in epoch in epoch {}'.format(epo+1))

    logger.info('training done')

from piq import ssim
import lpips
def test(opt, logger):
    test_loader = import_loader(opt)
    net = import_model(opt)
    logger.info(f'number of model parameters: {count_parameters(net)}')
    net.eval()
    psnr_list = []
    ssim_lsit = []
    lpips_list = []
    logger.info('start testing')
    for (img_inp, img_gt, img_name) in test_loader:
        img_inp = img_inp.to(opt.device)
        img_gt = img_gt.to(opt.device)
        with torch.no_grad():
            out = net(img_inp)
            out = out.clamp(0, 1)
            img_gt = img_gt.clamp(0, 1)
            mse = ((out - img_gt)**2).mean((2, 3))
            psnr = (1 / mse).log10().mean() * 10
            ssim_score = ssim(out, img_gt, data_range=1)
            # ssim_score = ssim(out, img_gt)
            lpips_model = lpips.LPIPS(net='alex').to(opt.device)
            lpips_score = lpips_model.forward(out, img_gt).mean()

        if opt.config['test']['save']:
            out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
            cv2.imwrite(r'{}/{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

        psnr_list.append(psnr.item())
        ssim_lsit.append(ssim_score.item())
        lpips_list.append(lpips_score.item())
        logger.info('image name: {}, test psnr: {}, test ssim: {}, test lpips: {}'.format(img_name[0], psnr, ssim_score.item(), lpips_score.item()))
        # logger.info('image name: {}, test psnr: {}, test lpips: {}'.format(img_name[0], psnr, lpips_score.item()))
        # logger.info('image name: {}, test psnr: {}'.format(img_name[0], psnr))

    logger.info('testing done, overall psnr: {}, overall ssim: {}, overall lpips: {}'.format(sum(psnr_list) / len(psnr_list), sum(ssim_lsit) / len(ssim_lsit), sum(lpips_list) / len(lpips_list)))
    # logger.info('testing done, overall psnr: {}, overall lpips: {}'.format(sum(psnr_list) / len(psnr_list), sum(lpips_list) / len(lpips_list)))
    # logger.info('testing done, overall psnr: {}'.format(sum(psnr_list) / len(psnr_list)))


def demo(opt, logger):
    demo_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    logger.info('start demonstration')
    for img_inp, img_name in demo_loader:
        img_inp = img_inp.to(opt.device)
        with torch.no_grad():
            out = net(img_inp)
        out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
        cv2.imwrite(r'{}/{}.png'.format(opt.save_image_dir, img_name[0]), out_img)
        logger.info('image name: {} output generated'.format(img_name[0]))
    logger.info('demonstration done')


if __name__ == "__main__":
    mp.set_start_method('spawn')

    opt = get_option()
    logger = Logger(opt)

    if opt.task == 'train':
        print(1111)
        train(opt, logger)
    elif opt.task == 'test':
        test(opt, logger)
    elif opt.task == 'demo':
        demo(opt, logger)
    else:
        raise ValueError('unknown task, please choose from [train, test, demo].')
