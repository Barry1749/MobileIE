import torch.nn as nn
import torch
from .utils import (
MBRConv5,
MBRConv3,
MBRConv1,
DropBlock,
FST,
FSTS,
)
import torch.fft as fft


class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )
        # self.att = nn.Sequential(
        # nn.AdaptiveAvgPool2d(1),
        # MBRConv1(channels, channels, rep_scale=rep_scale),
        # nn.Sigmoid()
        # )
        # self.att1= nn.Sequential(
        # MBRConv1(1, channels, rep_scale=rep_scale),
        # nn.Sigmoid()
        # )
        self.att = FGHDPA(channels, rep_scale=rep_scale)

        self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        # x2 = self.att(x1)
        # max_out, _ = torch.max(x2 * x1 , dim=1, keepdim=True)
        # x3 = self.att1(max_out)
        # x4 = torch.mul(x2, x3) * x1
        # return self.tail(x4)
        x2 = self.att(x1) # FG-HDPA output = attention_applied_feature
        return self.tail(x2)
        # return self.tail(x2) + x


    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        net_slim = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
                if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w
                    weight_slim['%s.bias' % name] = b
            elif isinstance(mod, FST):
                weight_slim['%s.bias' % name] = mod.bias
                weight_slim['%s.weight1' % name] = mod.weight1
                weight_slim['%s.weight2' % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
                net_slim.load_state_dict(weight_slim)
        return net_slim

class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        self.head = FSTS(
        nn.Sequential(
        nn.Conv2d(3, channels, 5, 1, 2),
        nn.PReLU(channels),
        nn.Conv2d(channels, channels, 3, 1, 1)
        ),
        channels
        )
        self.body = FSTS(
        nn.Conv2d(channels, channels, 3, 1, 1),
        channels
        )
        self.att = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, channels, 1),
        nn.Sigmoid()
        )
        self.att1 = nn.Sequential(
        nn.Conv2d(1, channels, 1, 1),
        nn.Sigmoid()
        )
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)


class FGHDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(FGHDPA, self).__init__()

        # 將 concat(x_spatial, x_freq) 投影回 channel attention
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # local path 不動，但要包成 conv1x1（使用原本 MBRConv1）
        self.local = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )

    def forward(self, F):
        # ------------------------------
        # 1. Spatial global descriptor
        # ------------------------------
        g_spatial = torch.mean(F, dim=[2,3], keepdim=True) # GAP(F)

        # ------------------------------
        # 2. Frequency descriptor (DCT / FFT)
        # ------------------------------
        # 使用 FFT 的 magnitude 當頻域特徵（效果好且簡單）
        F_freq = torch.abs(fft.rfft2(F, norm="ortho"))

        # rfft2 會縮小維度 → 需要 interpolate 回原大小
        F_freq = torch.nn.functional.interpolate(
            F_freq, size=F.shape[2:], mode="bilinear", align_corners=False
        )

        g_freq = torch.mean(F_freq, dim=[2,3], keepdim=True) # GAP(DCT/Frequency)

        # ------------------------------
        # 3. Concatenate descriptors
        # ------------------------------
        g = torch.cat([g_spatial, g_freq], dim=1) # (B, 2C, 1, 1)

        # ------------------------------
        # 4. Global attention Ag
        # ------------------------------
        Ag = self.fc(g) # (B, C, 1, 1)
        Fg = Ag * F

        # ------------------------------
        # 5. Local attention Al
        # ------------------------------
        max_out, _ = torch.max(Fg, dim=1, keepdim=True) # (B,1,H,W)
        Al = self.local(max_out)

        # ------------------------------
        # 6. Final output
        # ------------------------------
        A = Ag * Al
        return A * F
    

    