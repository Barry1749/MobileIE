import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class MBRConv5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rep_scale: int = 4,      # 保留參數名避免動到外面 code
        mask_prob: float = 0.0,  # <=1: DropConnect 機率；>1: 用自訂 mask
        mask_start_epoch: int = 0,  # 從第幾個 epoch 開始啟用 mask（0 = 一開始就啟用）
        se_ratio: int = 16,      # SE 縮放比例 (C -> C//r -> C)
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
        self.mask_prob = mask_prob
        self.mask_start_epoch = mask_start_epoch
        self.current_epoch = -1   # 由訓練 loop 外部更新

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.bn = nn.BatchNorm2d(out_channels)

        # self.proj = None
        # if in_channels != out_channels:
        #     self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        # se_hidden = max(out_channels // se_ratio, 1)
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),              # [B, C, H, W] -> [B, C, 1, 1]
        #     nn.Conv2d(out_channels, se_hidden, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(se_hidden, out_channels, 1),
        #     nn.Sigmoid()
        # )

        # 自訂 mask patterns 模式
        self.use_predefined_masks = mask_prob > 1
        if self.use_predefined_masks:
            cross = torch.tensor([
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ])
            h = torch.tensor([
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ])
            v = torch.tensor([
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ])
            o = torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ])
            x = torch.tensor([
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
            ])
            s = torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ])

            # center_only = torch.zeros(5, 5)
            # center_only[2, 2] = 1

            full = torch.ones(5, 5)

            predefined_masks = [cross, s, x]

            if not predefined_masks or len(predefined_masks) == 0:
                raise ValueError(
                    "mask_prob > 1 時，必須提供至少一個 predefined_masks。"
                )

            processed = []
            for m in predefined_masks:
                if m.dim() == 2:          # [5, 5]
                    m = m.view(1, 1, 5, 5)
                elif m.dim() == 4:        # [1, 1, 5, 5]
                    pass
                else:
                    raise ValueError(
                        f"predefined mask 的 shape 不合法，收到 {m.shape}，"
                        "預期 [5,5] 或 [1,1,5,5]"
                    )
                processed.append(m.float())

            masks_tensor = torch.stack(processed, dim=0)  # [K, 1, 1, 5, 5]
            self.register_buffer("predefined_masks", masks_tensor)
        else:
            self.predefined_masks = None

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
        W = self.conv.weight
        b = self.conv.bias

        mask = self._sample_kernel_mask()
        if mask is not None:
            print("mask not none")
            W = W * mask

        out = F.conv2d(
            x, W, b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        out_bn = self.bn(out)
        # out = F.relu(out, inplace=True)

        # # ----- SE channel attention -----
        # w = self.se(out)        # [B, C, 1, 1]
        # out = out * w

        # # ----- residual 加回去 -----
        # out = out + (x if self.proj is None else self.proj(x))
        # return out
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
        mask_prob: float = 0.0,  # <=1: DropConnect 機率, >1: 用自訂 mask
        mask_start_epoch: int = 100,  # 從第幾個 epoch 開始套 mask
        se_ratio: int = 16,      # SE 縮放比例 (C -> C//r -> C)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_prob = mask_prob
        self.mask_start_epoch = mask_start_epoch
        self.current_epoch = 0   # 訓練 loop 會設定

        # ----- 主 3x3 conv -----
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)

        # ----- residual 投影 -----
        # 如果 C_in != C_out，就用 1x1 conv 投影到同一個 channel 維度
        # self.proj = None
        # if in_channels != out_channels:
        #     self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # ----- SE channel attention -----
        # 用 conv 版本的 SE： GAP -> 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
        # se_hidden = max(out_channels // se_ratio, 1)
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),              # [B, C, H, W] -> [B, C, 1, 1]
        #     nn.Conv2d(out_channels, se_hidden, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(se_hidden, out_channels, 1),
        #     nn.Sigmoid()
        # )

        # ----- 自訂 mask pattern 模式 -----
        self.use_predefined_masks = mask_prob > 1
        if self.use_predefined_masks:
            # 你原本的 3x3 pattern 可以直接拿來
            cross = torch.tensor([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ])
            h = torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ])
            v = torch.tensor([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ])
            center_only = torch.zeros(3, 3)
            center_only[1, 1] = 1
            full = torch.ones(3, 3)

            predefined_masks = [cross, full, full, full, full, h, v]
            processed = []
            for m in predefined_masks:
                if m.dim() == 2:
                    m = m.view(1, 1, 3, 3)
                elif m.dim() == 4:
                    pass
                else:
                    raise ValueError(
                        f"predefined mask 的 shape 不合法，收到 {m.shape}，預期 [3,3] 或 [1,1,3,3]"
                    )
                processed.append(m.float())
            masks_tensor = torch.stack(processed, dim=0)   # [K, 1, 1, 3, 3]
            self.register_buffer("predefined_masks", masks_tensor)
        else:
            self.predefined_masks = None

        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1)

    # 訓練 loop 每個 epoch 開頭呼叫
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _sample_kernel_mask(self):
        """
        只在 training=True 且 current_epoch >= mask_start_epoch 時使用。
        回傳一個跟 conv.weight 同 shape 的 mask，或 None。
        """
        if (not self.training) or (self.current_epoch < self.mask_start_epoch):
            return None

        W = self.conv.weight

        # case 1: 使用自訂 pattern
        if self.use_predefined_masks:
            K = self.predefined_masks.shape[0]
            idx = torch.randint(0, K, (1,), device=W.device).item()
            base = self.predefined_masks[idx]          # [1, 1, 3, 3]
            mask = base.expand_as(W)                   # [C_out, C_in, 3, 3]
            return mask

        # case 2: DropConnect 風格 mask_prob ∈ (0, 1]
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
        mask = mask / keep_prob
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # ----- conv + mask -----
        W = self.conv.weight
        b = self.conv.bias

        mask = self._sample_kernel_mask()
        if mask is not None:
            W = W * mask

        out = F.conv2d(
            x, W, b,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        out_bn = self.bn(out)
        # out = F.relu(out, inplace=True)

        # # ----- SE channel attention -----
        # w = self.se(out)        # [B, C, 1, 1]
        # out = out * w

        # # ----- residual 加回去 -----
        # out = out + (x if self.proj is None else self.proj(x))
        out = torch.cat(
            [out, 
             out_bn,],
            1
        )
        return self.conv_out(out)

    def slim(self):
        """
        注意：有 residual + SE 後，很難完全 reparam 成「單一 conv」，
        這裡只把 conv+BN fuse 起來，方便你做分析或別的用途。
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
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def set_epoch(self, epoch: int):
        pass

    def forward(self, inp): 
        x0 = self.conv(inp)  
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)

        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias
    
class FST(nn.Module):
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        # self.weight1 = nn.Parameter(torch.randn(1)) 
        # self.weight2 = nn.Parameter(torch.randn(1)) 
        self.weight1 = nn.Parameter(torch.randn((1, channels, 1, 1)))  
        self.weight2 = nn.Parameter(torch.randn((1, channels, 1, 1)))  
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
        # self.weight1 = nn.Parameter(torch.randn((1, channels, 1, 1)))  
        # self.weight2 = nn.Parameter(torch.randn((1, channels, 1, 1)))  
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
