import torch
import torch.nn as nn
import torch.nn.functional as F

# class MBRConv5(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv5, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv1_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
#         self.conv2_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
#         self.conv_crossh_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
#         self.conv_crossv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         ) 
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        
#     def forward(self, inp):   
#         x1 = self.conv(inp)
#         x2 = self.conv1(inp)
#         x3 = self.conv2(inp)
#         x4 = self.conv_crossh(inp)
#         x5 = self.conv_crossv(inp)
#         x = torch.cat(
#             [x1, x2, x3, x4, x5,
#              self.conv_bn(x1),
#              self.conv1_bn(x2),
#              self.conv2_bn(x3),
#              self.conv_crossh_bn(x4),
#              self.conv_crossv_bn(x5)],
#             1
#         )
#         out = self.conv_out(x)
#         return out 

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         conv1_weight = self.conv1.weight
#         conv1_bias = self.conv1.bias
#         conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

#         conv2_weight = self.conv2.weight
#         conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
#         conv2_bias = self.conv2.bias

#         conv_crossv_weight = self.conv_crossv.weight
#         conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
#         conv_crossv_bias = self.conv_crossv.bias

#         conv_crossh_weight = self.conv_crossh.weight
#         conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
#         conv_crossh_bias = self.conv_crossh.bias

#         conv1_bn_weight = self.conv1.weight
#         conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

#         conv2_bn_weight = self.conv2.weight
#         conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

#         conv_crossv_bn_weight = self.conv_crossv.weight
#         conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

#         conv_crossh_bn_weight = self.conv_crossh.weight
#         conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

#         bn = self.conv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5

#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + b
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         bn = self.conv1_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_bias = self.conv1.bias * k + b
#         conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

#         bn = self.conv2_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv2_bn_bias = self.conv2.bias * k + b
#         conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

#         bn = self.conv_crossv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_bias = self.conv_crossv.bias * k + b
#         conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

#         bn = self.conv_crossh_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_bias = self.conv_crossh.bias * k + b
#         conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

#         weight = torch.cat(
#             [conv_weight, conv1_weight, conv2_weight,
#              conv_crossh_weight, conv_crossv_weight,
#              conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
#              conv_crossh_bn_weight, conv_crossv_bn_weight],
#             0
#         )
#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
#         bias_ = torch.cat(
#             [conv_bias, conv1_bias, conv2_bias,
#              conv_crossh_bias, conv_crossv_bias,
#              conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
#              conv_crossh_bn_bias, conv_crossv_bn_bias],
#             0
#         )
#         bias = torch.matmul(weight_compress, bias_)
#         if isinstance(self.conv_out.bias, torch.Tensor):
#             bias = bias + self.conv_out.bias
#         return weight, bias

class MBRConv5CAMR(nn.Module):
    """
    Content-Aware MBRConv5 + IWO 排程版

    - 訓練前期 (epoch < iwo_start_epoch):
        * 學習 conv_out.weight (base 1x1 kernel)
        * weight1 不更新 (offset 關閉)

    - 訓練後期 (epoch >= iwo_start_epoch):
        * freeze conv_out.weight
        * 只學 weight1 (initial weight offset, IWO)

    - forward:
        * 都使用 final_weight = conv_out.weight + weight1

    - slim():
        * 一樣把所有多分支 + BN 對應到 5x5 kernel
        * 使用 (conv_out.weight + weight1) 作 channel 壓縮
    """

    def __init__(self, in_channels, out_channels, rep_scale=4,
                 gate_reduction=4, iwo_start_epoch: int = 1000):
        super(MBRConv5CAMR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        # ----- IWO 相關 -----
        self.iwo_start_epoch = iwo_start_epoch
        self._iwo_active = False  # 是否已啟用 IWO

        # ===== 原始 5 個分支 =====
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 最後 1x1 把 10 倍 channel 壓回 out_channels
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        # IWO: 前期學 conv_out.weight，後期 freeze
        self.conv_out.weight.requires_grad = True

        # IWO 的 offset（原本的 weight1）
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)
        # 前期關掉 offset 的學習，等 epoch 到再開
        self.weight1.requires_grad = False

        # ===== content-aware gate =====
        hidden_dim = max(in_channels // gate_reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, C_in, 1, 1]
            nn.Flatten(),                    # [B, C_in]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 5),        # 5 個分支 (5x5, 1x1, 3x3, 3x1, 1x3)
            nn.Softmax(dim=1)                # 每張圖的 5 個權重 sum = 1
        )

    # ---------- IWO epoch 控制 ----------
    def set_epoch(self, epoch: int):
        """
        在 training loop 每個 epoch 開頭呼叫：
            block.set_epoch(epoch)

        當 epoch >= iwo_start_epoch:
            - 啟用 IWO:
                conv_out.weight.requires_grad = False
                weight1.requires_grad = True
        """
        if (not self._iwo_active) and (epoch >= self.iwo_start_epoch):
            self._iwo_active = True
            self.conv_out.weight.requires_grad = False
            self.weight1.requires_grad = True

    def _get_final_conv_out_weight(self) -> torch.Tensor:
        """
        取得目前要用的 1x1 kernel：
            W_final = conv_out.weight + weight1
        （前期 weight1 不學，後期 conv_out.fixed + weight1）
        """
        return self.conv_out.weight + self.weight1

    def forward(self, inp):
        # ---- 1. 原本 5 個 conv 分支 ----
        x1 = self.conv(inp)          # 5x5
        x2 = self.conv1(inp)         # 1x1
        x3 = self.conv2(inp)         # 3x3
        x4 = self.conv_crossh(inp)   # 3x1
        x5 = self.conv_crossv(inp)   # 1x3

        # ---- 2. content-aware gate：計算每張圖的 branch 權重 ----
        # alpha: [B, 5]
        alpha = self.gate(inp)
        # 展開成 [B, 1, 1, 1] 方便 broadcast
        a1 = alpha[:, 0].view(-1, 1, 1, 1)
        a2 = alpha[:, 1].view(-1, 1, 1, 1)
        a3 = alpha[:, 2].view(-1, 1, 1, 1)
        a4 = alpha[:, 3].view(-1, 1, 1, 1)
        a5 = alpha[:, 4].view(-1, 1, 1, 1)

        # ---- 3. 對應每個 kernel family，把「原輸出 + BN 輸出」一起縮放 ----
        x1_raw = x1 * a1
        x2_raw = x2 * a2
        x3_raw = x3 * a3
        x4_raw = x4 * a4
        x5_raw = x5 * a5

        x1_bn = self.conv_bn(x1) * a1
        x2_bn = self.conv1_bn(x2) * a2
        x3_bn = self.conv2_bn(x3) * a3
        x4_bn = self.conv_crossh_bn(x4) * a4
        x5_bn = self.conv_crossv_bn(x5) * a5

        # ---- 4. concat 後用 1x1 壓回 out_channels ----
        x = torch.cat(
            [x1_raw, x2_raw, x3_raw, x4_raw, x5_raw,
             x1_bn,  x2_bn,  x3_bn,  x4_bn,  x5_bn],
            dim=1
        )
        final_weight = self._get_final_conv_out_weight()
        out = F.conv2d(x, final_weight, self.conv_out.bias)
        return out

    def slim(self):
        """
        和原論文一樣的 slim 流程：
        - 無視 gate（gate 只在訓練時影響權重的學習）
        - 只用 conv/conv1/conv2/crossh/crossv + 對應 BN + (conv_out.weight + weight1)
          重參數化成一組 5x5 的等效 kernel 和 bias
        """
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

        conv2_weight = self.conv2.weight
        conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
        conv2_bias = self.conv2.bias

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias

        # 下面這幾個 *_bn_weight 是「對應分支的 conv weight，在之後會乘上 BN 參數」
        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

        conv2_bn_weight = self.conv2.weight
        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

        # ===== 把 BN 合進各自的 conv =====
        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5
        conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias = self.conv2.bias * k + b
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        # ===== 把 10 個分支的 weight / bias 串在一起，再用 (conv_out.weight + weight1) 壓成 out_channels =====
        weight = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            dim=0
        )
        weight_compress = (self.conv_out.weight + self.weight1).squeeze()
        weight = torch.matmul(weight_compress, weight.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)

        bias_ = torch.cat(
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            dim=0
        )
        bias = torch.matmul(weight_compress, bias_)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias
    
MBRConv5 = MBRConv5CAMR  # 將 MBRConv5 指向 MBRConv5CAMR，方便外部呼叫
##############################################################################################################
# class MBRConv3(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv3, self).__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.rep_scale = rep_scale
        
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv1_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
#         self.conv_crossh_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
#         self.conv_crossv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

#     def forward(self, inp):    
#         x0 = self.conv(inp)
#         x1 = self.conv1(inp)
#         x2 = self.conv_crossh(inp)
#         x3 = self.conv_crossv(inp)
#         x = torch.cat(
#         [    x0,x1,x2,x3,
#              self.conv_bn(x0),
#              self.conv1_bn(x1),
#              self.conv_crossh_bn(x2),
#              self.conv_crossv_bn(x3)],
#             1
#         )    
#         out = self.conv_out(x)
#         return out

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         conv1_weight = self.conv1.weight
#         conv1_bias = self.conv1.bias
#         conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

#         conv_crossh_weight = self.conv_crossh.weight
#         conv_crossh_bias = self.conv_crossh.bias
#         conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

#         conv_crossv_weight = self.conv_crossv.weight
#         conv_crossv_bias = self.conv_crossv.bias
#         conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

#         # conv_bn
#         bn = self.conv_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         # conv1_bn
#         bn = self.conv1_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
#         conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
#         conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

#         # conv_crossh_bn
#         bn = self.conv_crossh_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
#         conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
#         conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

#         # conv_crossv_bn
#         bn = self.conv_crossv_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
#         conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
#         conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

#         weight = torch.cat([
#             conv_weight,
#             conv1_weight,
#             conv_crossh_weight,
#             conv_crossv_weight,
#             conv_bn_weight,
#             conv1_bn_weight,
#             conv_crossh_bn_weight,
#             conv_crossv_bn_weight
#         ], dim=0)

#         bias = torch.cat([
#             conv_bias,
#             conv1_bias,
#             conv_crossh_bias,
#             conv_crossv_bias,
#             conv_bn_bias,
#             conv1_bn_bias,
#             conv_crossh_bn_bias,
#             conv_crossv_bn_bias
#         ], dim=0)

#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
#         weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

#         bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
#         if self.conv_out.bias is not None:
#             bias += self.conv_out.bias

#         return weight, bias
    
class MBRConv3CAMR(nn.Module):
    """
    Content-Aware 版的 MBRConv3 + IWO：

    IWO 排程：
    - 訓練前期 (epoch < iwo_start_epoch):
        * 學 conv_out.weight（base 1x1 kernel）
        * weight1 不更新（offset 關閉）

    - 訓練後期 (epoch >= iwo_start_epoch):
        * freeze conv_out.weight
        * 只學 weight1（initial weight offset, IWO）

    forward / slim 一律使用：
        final_weight = conv_out.weight + weight1
    """

    def __init__(self, in_channels, out_channels,
                 rep_scale: int = 4,
                 gate_reduction: int = 4,
                 iwo_start_epoch: int = 1000):
        super(MBRConv3CAMR, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        # ----- IWO 相關 -----
        self.iwo_start_epoch = iwo_start_epoch
        self._iwo_active = False  # 是否已啟用 IWO

        # ===== 原本 4 個分支 =====
        mid_ch = out_channels * rep_scale

        # 3x3
        self.conv = nn.Conv2d(in_channels, mid_ch, 3, 1, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(mid_ch)
        )
        # 1x1
        self.conv1 = nn.Conv2d(in_channels, mid_ch, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(mid_ch)
        )
        # 3x1 (horizontal)
        self.conv_crossh = nn.Conv2d(in_channels, mid_ch, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(mid_ch)
        )
        # 1x3 (vertical)
        self.conv_crossv = nn.Conv2d(in_channels, mid_ch, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(mid_ch)
        )

        # 1x1 conv 壓回 out_channels，外加一個可學習的偏移 weight1
        self.conv_out = nn.Conv2d(mid_ch * 8, out_channels, 1)
        # IWO：前期學 conv_out.weight，後期才 freeze
        self.conv_out.weight.requires_grad = True

        # IWO offset
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)
        # 一開始先不學 offset
        self.weight1.requires_grad = False

        # ===== 新增：content-aware gate =====
        hidden_dim = max(in_channels // gate_reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, C_in, 1, 1]
            nn.Flatten(),                    # [B, C_in]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),        # 4 個分支: 3x3, 1x1, 3x1, 1x3
            nn.Softmax(dim=1)                # 每張圖的 4 個權重 sum = 1
        )

    # ---------- IWO epoch 控制 ----------
    def set_epoch(self, epoch: int):
        """
        在 training loop 每個 epoch 開頭呼叫：
            block.set_epoch(epoch)

        當 epoch >= iwo_start_epoch:
            - 啟用 IWO:
                conv_out.weight.requires_grad = False
                weight1.requires_grad = True
        """
        if (not self._iwo_active) and (epoch >= self.iwo_start_epoch):
            self._iwo_active = True
            self.conv_out.weight.requires_grad = False
            self.weight1.requires_grad = True

    def _get_final_conv_out_weight(self) -> torch.Tensor:
        """
        取得目前要用的 1x1 kernel：
            W_final = conv_out.weight + weight1
        （前期 weight1 不學，後期 conv_out.fixed + weight1）
        """
        return self.conv_out.weight + self.weight1

    def forward(self, inp):
        # ---- 1. 原本 4 個 conv 分支 ----
        x0 = self.conv(inp)          # 3x3
        x1 = self.conv1(inp)         # 1x1
        x2 = self.conv_crossh(inp)   # 3x1
        x3 = self.conv_crossv(inp)   # 1x3

        # ---- 2. content-aware gate：計算每張圖的 branch 權重 ----
        # alpha: [B, 4]
        alpha = self.gate(inp)
        # 展開成 [B, 1, 1, 1] 方便 broadcast
        a0 = alpha[:, 0].view(-1, 1, 1, 1)
        a1 = alpha[:, 1].view(-1, 1, 1, 1)
        a2 = alpha[:, 2].view(-1, 1, 1, 1)
        a3 = alpha[:, 3].view(-1, 1, 1, 1)

        # ---- 3. 對每一個 kernel family（原輸出 + BN 輸出）一起縮放 ----
        x0_raw = x0 * a0
        x1_raw = x1 * a1
        x2_raw = x2 * a2
        x3_raw = x3 * a3

        x0_bn = self.conv_bn(x0) * a0
        x1_bn = self.conv1_bn(x1) * a1
        x2_bn = self.conv_crossh_bn(x2) * a2
        x3_bn = self.conv_crossv_bn(x3) * a3

        # ---- 4. concat 後用 (conv_out.weight + weight1) 壓回 out_channels ----
        x = torch.cat(
            [x0_raw, x1_raw, x2_raw, x3_raw,
             x0_bn,  x1_bn,  x2_bn,  x3_bn],
            dim=1
        )
        final_weight = self._get_final_conv_out_weight()
        out = F.conv2d(x, final_weight, self.conv_out.bias)
        return out

    def slim(self):
        """
        沿用原本 MBRConv3 的 slim：
        - 把 4 個 conv + 對應 BN 分支 + conv_out(+weight1)
          全部重參數化成一個 3x3 conv 的 weight, bias
        - gate 不參與 slim，只在訓練時影響這些 weight 的學習
        """

        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_bias = self.conv_crossh.bias
        conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_bias = self.conv_crossv.bias
        conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

        # conv_bn
        bn = self.conv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        # conv1_bn
        bn = self.conv1_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        # conv_crossh_bn
        bn = self.conv_crossh_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        # conv_crossv_bn
        bn = self.conv_crossv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([
            conv_weight,
            conv1_weight,
            conv_crossh_weight,
            conv_crossv_weight,
            conv_bn_weight,
            conv1_bn_weight,
            conv_crossh_bn_weight,
            conv_crossv_bn_weight
        ], dim=0)

        bias = torch.cat([
            conv_bias,
            conv1_bias,
            conv_crossh_bias,
            conv_crossv_bias,
            conv_bn_bias,
            conv1_bn_bias,
            conv_crossh_bn_bias,
            conv_crossv_bn_bias
        ], dim=0)

        # 用 (conv_out.weight + weight1) 做壓縮（IWO 已吃進來）
        weight_compress = (self.conv_out.weight + self.weight1).squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if self.conv_out.bias is not None:
            bias += self.conv_out.bias

        return weight, bias
    
MBRConv3 = MBRConv3CAMR  # 將 MBRConv3 指向 MBRConv3CAMR，方便外部呼叫
######################################################################################################
# class MBRConv1(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv1, self).__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.rep_scale = rep_scale
        
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

#     def forward(self, inp): 
#         x0 = self.conv(inp)  
#         x = torch.cat([x0, self.conv_bn(x0)], 1)
#         out = self.conv_out(x)
#         return out 

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         bn = self.conv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + b
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         weight = torch.cat([conv_weight, conv_bn_weight], 0)
#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

#         bias = torch.cat([conv_bias, conv_bn_bias], 0)
#         bias = torch.matmul(weight_compress, bias)

#         if isinstance(self.conv_out.bias, torch.Tensor):
#             bias = bias + self.conv_out.bias
#         return weight, bias
    
class MBRConv1CAMR(nn.Module):
    """
    Content-Aware 版的 MBRConv1 + IWO：
    - 結構沿用原本 MBRConv1（1x1 conv + BN 分支 + conv_out + weight1）
    - 新增 gate MLP，根據輸入 feature 動態決定：
        raw 分支 (conv) 與 BN 分支 (conv_bn) 的權重
    - IWO 排程：
        * 訓練前期 (epoch < iwo_start_epoch):
              - 學 conv_out.weight（base 1x1 kernel）
              - weight1 不更新
        * 訓練後期 (epoch >= iwo_start_epoch):
              - freeze conv_out.weight
              - 只學 weight1 (Initial Weight Offset)

      forward / slim 一律使用:
          final_weight = conv_out.weight + weight1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        rep_scale: int = 4,
        gate_reduction: int = 4,
        iwo_start_epoch: int = 1000
    ):
        super(MBRConv1CAMR, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        # ----- IWO 相關 -----
        self.iwo_start_epoch = iwo_start_epoch
        self._iwo_active = False  # 是否已啟用 IWO

        # 原本的 1x1 分支
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 壓回 out_channels 的 1x1，外加一個 learnable weight1
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)
        # IWO：前期學 conv_out.weight，後期才 freeze
        self.conv_out.weight.requires_grad = True

        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)
        # 一開始先不學 offset
        self.weight1.requires_grad = False

        # ===== 新增：content-aware gate =====
        # 這裡有兩個「路徑」要混：raw 分支、BN 分支 → 2 維權重
        hidden_dim = max(in_channels // gate_reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # [B, C_in, 1, 1]
            nn.Flatten(),                       # [B, C_in]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),           # raw / BN 兩條路徑
            nn.Softmax(dim=1)                   # 每張圖的 2 個權重 sum = 1
        )

    # ---------- IWO epoch 控制 ----------
    def set_epoch(self, epoch: int):
        """
        在 training loop 每個 epoch 開頭呼叫：
            block.set_epoch(epoch)

        當 epoch >= iwo_start_epoch:
            - 啟用 IWO:
                conv_out.weight.requires_grad = False
                weight1.requires_grad = True
        """
        if (not self._iwo_active) and (epoch >= self.iwo_start_epoch):
            self._iwo_active = True
            self.conv_out.weight.requires_grad = False
            self.weight1.requires_grad = True

    def _get_final_conv_out_weight(self) -> torch.Tensor:
        """
        取得目前要用的 1x1 kernel：
            W_final = conv_out.weight + weight1
        （前期 weight1 不學，後期 conv_out.fixed + weight1）
        """
        return self.conv_out.weight + self.weight1

    def forward(self, inp):
        # 1) 先算 1x1 conv 輸出
        x1 = self.conv(inp)              # [B, C*rep, H, W]

        # 2) content-aware gate：決定 raw vs BN 分支的比例
        alpha = self.gate(inp)           # [B, 2]
        a_raw = alpha[:, 0].view(-1, 1, 1, 1)   # [B,1,1,1]
        a_bn  = alpha[:, 1].view(-1, 1, 1, 1)

        # 3) 縮放 raw 分支與 BN 分支
        x_raw = x1 * a_raw
        x_bn  = self.conv_bn(x1) * a_bn

        # 4) concat 後用 (conv_out.weight + weight1) 壓回 out_channels
        x = torch.cat([x_raw, x_bn], dim=1)     # [B, 2*C*rep, H, W]
        final_weight = self._get_final_conv_out_weight()
        out = F.conv2d(x, final_weight, self.conv_out.bias)
        return out

    def slim(self):
        """
        和原 MBRConv1 的 slim 一樣：
        - 把 conv + BN 分支 + conv_out(+weight1) re-param 成一個 1x1 conv
        - gate 不進來，因為我們要 inference 用固定 kernel
        """

        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** 0.5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** 0.5

        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        # 串起 raw / BN 兩個 1x1 分支的權重與 bias
        weight = torch.cat([conv_weight, conv_bn_weight], dim=0)
        bias = torch.cat([conv_bias, conv_bn_bias], dim=0)

        # 用 conv_out.weight + weight1 做 channel 壓縮（IWO 也一起吃進來）
        weight_compress = self._get_final_conv_out_weight().squeeze()  # [C_out, 2*C_out*rep]
        weight = torch.matmul(weight_compress, weight.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)

        bias = torch.matmul(weight_compress, bias)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias

        return weight, bias
    
MBRConv1 = MBRConv1CAMR  # 將 MBRConv1 指向 MBRConv1CAMR，方便外部呼叫
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
