import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .ffc import FFC

class MBRConv5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rep_scale: int = 4,      # 保留參數名避免動到外面 code
    ):
        """
        mask_prob:
            - <= 0: 不做任何遮罩
            - 0 ~ 1: 每個 weight 以 mask_prob 機率變 0（訓練時才用）
            - > 1: 代表「用自訂的 mask」，會在 predefined_masks 裡隨機挑一個 pattern

        mask_start_epoch:
            - current_epoch < mask_start_epoch 時，不啟用 mask
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size=5,
            ratio_gin=0.5,
            ratio_gout=0.5,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=False,
            enable_lfu=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1)


    # 給訓練 loop 每個 epoch 開頭更新
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _sample_kernel_mask(self):
        """
        只在 self.training == True 且 current_epoch >= mask_start_epoch 時才會被呼叫。
        回傳一個跟 conv.weight 同 shape 的 mask，或 None。
        """
        if (not self.training) or (self.current_epoch < self.mask_start_epoch):
            return None

        W = self.conv.weight

        # case 1: 使用自訂的 mask pattern
        if self.use_predefined_masks:
            K = self.predefined_masks.shape[0]
            idx = torch.randint(0, K, (1,), device=W.device).item()
            base = self.predefined_masks[idx]          # [1, 1, 5, 5]
            mask = base.expand_as(W)                   # [C_out, C_in, 5, 5]
            # pattern 當作固定結構稀疏，不做 1/(1-p) scaling
            return mask

        # case 2: mask_prob 以機率遮掉各個 weight（DropConnect 風格）
        if self.mask_prob <= 0.0:
            return None
        if self.mask_prob > 1.0:
            raise ValueError(
                "mask_prob > 1 但沒有使用 predefined_masks，請檢查設定。"
            )

        keep_prob = 1.0 - self.mask_prob
        mask = torch.bernoulli(
            torch.full_like(W, keep_prob, device=W.device, dtype=W.dtype)
        )
        # 像 Dropout 一樣，把沒被遮掉的 weight 放大 1/keep_prob，維持期望值不變
        mask = mask / keep_prob
        return mask

    def forward(self, x):
        out = self.ffc(x)
        out_bn = self.bn(out)
        out = torch.cat(
            [out, 
             out_bn,],
            1
        )
        return self.conv_out(out)

    def slim(self):
        """
        導出「沒有遮罩、已經 fuse BN」的等效 5x5 conv。
        通常用在最終部署：訓練時 mask 只是 regularization。
        """
        bn = self.bn
        conv = self.conv

        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)

        weight = conv.weight * k.view(-1, 1, 1, 1)
        bias = conv.bias * k + b

        return weight, bias


##############################################################################################################
class MBRConv3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rep_scale: int = 4,      # 保留參數名避免動到外面 code
    ):
        """
        mask_prob:
            - <= 0: 不做任何遮罩
            - 0 ~ 1: 每個 weight 以 mask_prob 機率變 0（訓練時才用）
            - > 1: 代表「用自訂的 mask」，會在 predefined_masks 裡隨機挑一個 pattern

        mask_start_epoch:
            - current_epoch < mask_start_epoch 時，不啟用 mask
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size=3,
            ratio_gin=0.5,
            ratio_gout=0.5,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            enable_lfu=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1)


    # 給訓練 loop 每個 epoch 開頭更新
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _sample_kernel_mask(self):
        """
        只在 self.training == True 且 current_epoch >= mask_start_epoch 時才會被呼叫。
        回傳一個跟 conv.weight 同 shape 的 mask，或 None。
        """
        if (not self.training) or (self.current_epoch < self.mask_start_epoch):
            return None

        W = self.conv.weight

        # case 1: 使用自訂的 mask pattern
        if self.use_predefined_masks:
            K = self.predefined_masks.shape[0]
            idx = torch.randint(0, K, (1,), device=W.device).item()
            base = self.predefined_masks[idx]          # [1, 1, 5, 5]
            mask = base.expand_as(W)                   # [C_out, C_in, 5, 5]
            # pattern 當作固定結構稀疏，不做 1/(1-p) scaling
            return mask

        # case 2: mask_prob 以機率遮掉各個 weight（DropConnect 風格）
        if self.mask_prob <= 0.0:
            return None
        if self.mask_prob > 1.0:
            raise ValueError(
                "mask_prob > 1 但沒有使用 predefined_masks，請檢查設定。"
            )

        keep_prob = 1.0 - self.mask_prob
        mask = torch.bernoulli(
            torch.full_like(W, keep_prob, device=W.device, dtype=W.dtype)
        )
        # 像 Dropout 一樣，把沒被遮掉的 weight 放大 1/keep_prob，維持期望值不變
        mask = mask / keep_prob
        return mask

    def forward(self, x):
        out = self.ffc(x)
        out_bn = self.bn(out)
        out = torch.cat(
            [out, 
             out_bn,],
            1
        )
        return self.conv_out(out)

    def slim(self):
        """
        導出「沒有遮罩、已經 fuse BN」的等效 5x5 conv。
        通常用在最終部署：訓練時 mask 只是 regularization。
        """
        bn = self.bn
        conv = self.conv

        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)

        weight = conv.weight * k.view(-1, 1, 1, 1)
        bias = conv.bias * k + b

        return weight, bias

   
######################################################################################################
class MBRConv1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rep_scale: int = 4,      # 保留參數名避免動到外面 code
    ):
        """
        mask_prob:
            - <= 0: 不做任何遮罩
            - 0 ~ 1: 每個 weight 以 mask_prob 機率變 0（訓練時才用）
            - > 1: 代表「用自訂的 mask」，會在 predefined_masks 裡隨機挑一個 pattern

        mask_start_epoch:
            - current_epoch < mask_start_epoch 時，不啟用 mask
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size=1,
            ratio_gin=0.5,
            ratio_gout=0.5,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            enable_lfu=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1)


    # 給訓練 loop 每個 epoch 開頭更新
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, x):
        out = self.ffc(x)
        out_bn = self.bn(out)
        out = torch.cat(
            [out, 
             out_bn,],
            1
        )
        return self.conv_out(out)

    def slim(self):
        """
        導出「沒有遮罩、已經 fuse BN」的等效 5x5 conv。
        通常用在最終部署：訓練時 mask 只是 regularization。
        """
        bn = self.bn
        conv = self.conv

        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)

        weight = conv.weight * k.view(-1, 1, 1, 1)
        bias = conv.bias * k + b

        return weight, bias

##################################################################################
class FST(nn.Module):
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))  

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias
        
class FSTS(nn.Module):
    def __init__(self, block1, channels):
        super(FSTS, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1)) 
        self.weight2 = nn.Parameter(torch.randn(1)) 
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        
    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias
##################################################################################
class DropBlock(nn.Module):
    def __init__(self, block_size, p=0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p / block_size / block_size

    def forward(self, x):
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)
