import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    MBRConv3
)

def smoothstep(e0, e1, x):
    t = torch.clamp((x - e0) / (e1 - e0 + 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def dark_mask(img_rgb, dark_thr, low, high):
    """
    img_rgb: [B,3,H,W], 預期是 0~1 的 float tensor
    dark_thr, low, high: 可以是 scalar 或 [B,1,1,1] tensor
    回傳: [B,3,H,W] 的 mask，暗的地方接近 1，亮的接近 0
    """
    # 確保是 tensor
    assert torch.is_tensor(img_rgb), "img_rgb 必須是 torch.Tensor"

    # 拆成 RGB → 灰階
    r = img_rgb[:, 0:1, :, :]
    g = img_rgb[:, 1:2, :, :]
    b = img_rgb[:, 2:3, :, :]

    gray = 0.299 * r + 0.587 * g + 0.114 * b  # [B,1,H,W]，0~1

    # 處理 dark_thr / low / high 的 shape，讓它們可以 broadcast
    def to_tensor_param(p, name):
        if torch.is_tensor(p):
            # 可能是 [B,1,1,1] or scalar
            if p.dim() == 0:
                return p.view(1, 1, 1, 1)
            elif p.dim() == 2:
                # [B,1] -> [B,1,1,1]
                return p.view(-1, 1, 1, 1)
            else:
                return p
        else:
            # scalar → tensor
            return torch.tensor(p, dtype=img_rgb.dtype, device=img_rgb.device).view(1, 1, 1, 1)

    dark_thr_t = to_tensor_param(dark_thr, "dark_thr")
    low_t      = to_tensor_param(low, "low")
    high_t     = to_tensor_param(high, "high")

    # inv: 暗的地方比較大
    inv = torch.clamp((dark_thr_t - gray) / (dark_thr_t + 1e-6), 0.0, 1.0)
    m = smoothstep(low_t, high_t, inv)   # [B,1,H,W]

    # 複製 3 個 channel
    return m.repeat(1, 3, 1, 1)           # [B,3,H,W]

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    img: [B, C, H, W], float, 0~1
    sigma: float（scalar），例如 4.0
    return: [B, C, H, W]
    """
    if sigma <= 0:
        return img

    # kernel size: 大約 6*sigma，取奇數
    ksize = int(2 * round(3 * sigma) + 1)
    # 在同一個 device / dtype 建 kernel
    device = img.device
    dtype = img.dtype

    coords = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()                           # normalize
    kernel = kernel.view(1, 1, ksize, ksize)                 # [1,1,K,K]

    C = img.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)                       # [C,1,K,K]

    padding = ksize // 2
    # depthwise conv
    return F.conv2d(img, kernel, padding=padding, groups=C)

def restore_local_contrast(img_rgb: torch.Tensor,
                           amount: float = 0.12,
                           sigma: float = 4.0) -> torch.Tensor:
    """
    img_rgb: [B,3,H,W], float, 0~1
    return: [B,3,H,W], float, 0~1
    """
    blur = gaussian_blur(img_rgb, sigma)   # [B,3,H,W]
    out = img_rgb + amount * (img_rgb - blur)
    return torch.clamp(out, 0.0, 1.0)

def protect_highlights(img_rgb: torch.Tensor,
                       strength: float = 0.25) -> torch.Tensor:
    """
    img_rgb: [B,3,H,W], float, 0~1
    strength: scalar
    return: [B,3,H,W], float, 0~1
    """
    # x / (x + c)
    out = img_rgb / (img_rgb + strength)
    return torch.clamp(out, 0.0, 1.0)

def gray_world_balance(img_rgb: torch.Tensor,
                       mix: float = 0.4) -> torch.Tensor:
    """
    img_rgb: [B,3,H,W], float, 0~1
    mix: 0~1，越大表示越靠近 gray-world 校正
    return: [B,3,H,W], float, 0~1
    """
    B, C, H, W = img_rgb.shape
    img_flat = img_rgb.view(B, C, -1)                   # [B,3,H*W]

    mean_c = img_flat.mean(dim=-1, keepdim=True) + 1e-6 # [B,3,1]
    mean_all = img_flat.mean(dim=(1, 2), keepdim=True) + 1e-6  # [B,1,1]

    gain = (mean_all / mean_c).view(B, C, 1, 1)         # [B,3,1,1]

    balanced = img_rgb * ((1.0 - mix) + gain * mix)     # 緩和一點
    return torch.clamp(balanced, 0.0, 1.0)

# def run_loc_v3(original_rgb,
# boost_strength=0.16, # 原本 0.24 → 降低增亮幅度
# dark_thr=0.48, # 原本 0.55~0.65 → 降低「暗部」覆蓋
# low=0.20,
# high=0.70,
# restore_local_contrast_amount = 0.15,
# restore_local_contrast_sigma = 4.0,
# protect_highlights_strength = 0.15,
# gray_world_balance_mix = 0.30): # 讓 mask 更柔和
#     base = run_baseline(original_rgb)

#     mask = dark_mask(original_rgb, dark_thr, low, high)
#     diff = base.astype(np.float32) - original_rgb.astype(np.float32)

#     # (1) 暗部增亮（弱化）
#     out = original_rgb.astype(np.float32) + mask * diff * boost_strength
#     out = np.clip(out, 0, 255).astype(np.uint8)

#     # (2) 對比恢復（強化）
#     out = restore_local_contrast(out, amount=restore_local_contrast_amount, sigma=restore_local_contrast_sigma)

#     # (3) 高光保留（弱化壓制）
#     out = protect_highlights(out, strength=protect_highlights_strength)

#     # (4) 防偏色（降低強度）
#     out = gray_world_balance(out, mix=gray_world_balance_mix)

#     return out, base

# class LocV3Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        original_rgb,                   # [B,3,H,W], 0~1，原圖
        base_rgb,                       # [B,3,H,W], 0~1，前一個模型的輸出
        boost_strength,                 # [B,1,1,1]
        dark_thr,                       # [B,1,1,1]
        low,                            # [B,1,1,1]
        high,                           # [B,1,1,1]
        restore_local_contrast_amount,  # [B,1,1,1]
        restore_local_contrast_sigma,   # [B,1,1,1] or scalar
        protect_highlights_strength,    # [B,1,1,1]
        gray_world_balance_mix,         # [B,1,1,1]
    ):
        x = original_rgb                      # 原圖
        base = base_rgb                       # 前面模型結果

        # (1) 暗部 mask
        mask = dark_mask(x, dark_thr, low, high)  # [B,3,H,W]
        diff = base - x                              # 用「base 比原圖亮多少」來增亮

        out = x + mask * diff * boost_strength
        out = torch.clamp(out, 0.0, 1.0)

        # (2) 對比恢復
        if restore_local_contrast_sigma.dim() > 1:
            sigma = restore_local_contrast_sigma.mean()
        else:
            sigma = restore_local_contrast_sigma

        out = restore_local_contrast(
            out,
            amount=restore_local_contrast_amount.mean(),
            sigma=sigma,
        )

        # (3) 高光保護
        out = protect_highlights(out, strength=protect_highlights_strength.mean())

        # (4) 防偏色
        out = gray_world_balance(out, mix=gray_world_balance_mix.mean())

        return out
    
class LocV3Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                original_rgb,
                base_rgb,
                boost_strength,
                dark_thr,
                low,
                high,
                restore_local_contrast_amount,
                restore_local_contrast_sigma,
                protect_highlights_strength,
                gray_world_balance_mix):
        x = original_rgb          # [B,3,H,W]，0~1
        base = base_rgb           # [B,3,H,W]，0~1

        # 1) 暗部增亮
        mask = dark_mask(x, dark_thr, low, high)         # torch 版
        diff = base - x
        out = x + mask * diff * boost_strength
        out = torch.clamp(out, 0.0, 1.0)

        # 2) local contrast
        sigma = restore_local_contrast_sigma.mean().item()
        amt   = restore_local_contrast_amount.mean().item()
        out = restore_local_contrast(out, amount=amt, sigma=sigma)

        # 3) 高光保護
        out = protect_highlights(out, strength=protect_highlights_strength.mean().item())

        # 4) 防偏色
        out = gray_world_balance(out, mix=gray_world_balance_mix.mean().item())

        return out
    
class LocV3ParamHead(nn.Module):
    def __init__(self, in_channels=3, hidden=128):
        super().__init__()
        self.conv = MBRConv3(3, 3, rep_scale=4)
        self.pool = nn.AdaptiveAvgPool2d(1)   # [B,C,H,W] -> [B,C,1,1]
        self.mlp = nn.Sequential(
            nn.Flatten(),                     # [B,C,1,1] -> [B,C]
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 8),             # 8 個 raw 參數
        )

    def forward(self, orig_rgb, prev_rgb):
        """
        orig_rgb: [B,3,H,W]
        prev_rgb: [B,3,H,W]  (前面模型輸出)
        """
        # x = torch.cat([self.conv(orig_rgb), self.conv(prev_rgb)], dim=1)  # [B,6,H,W]
        x = self.conv(orig_rgb)
        x = self.pool(x)                            # [B,6,1,1]
        raw = self.mlp(x)                           # [B,8]
        s = torch.sigmoid(raw)                      # [B,8] → [0,1]

        x2 = self.conv(prev_rgb)
        x2 = self.pool(x2)                            # [B,6,1,1]
        raw2 = self.mlp(x2)                           # [B,8]
        s2 = torch.sigmoid(raw2)                      # [B,8] → [0,1]

        # mapping 到實際範圍
        boost_strength              = 0. + 0.3 * s[:, 0:1] * s2[:, 0:1]
        dark_thr                    = 0. + 0.6 * s[:, 1:2] * s2[:, 1:2]
        low                         = 0. + 0.40 * s[:, 2:3] * s2[:, 2:3]
        high                        = 0. + 0.90 * s[:, 3:4] * s2[:, 3:4]
        restore_local_contrast_amt  = 0. + 0.4 * s[:, 4:5] * s2[:, 4:5]
        restore_local_contrast_sigma= 0. + 7.0  * s[:, 5:6] * s2[:, 5:6]
        protect_highlights_strength = 0. + 0.5 * s[:, 6:7] * s2[:, 6:7]
        gray_world_balance_mix      = 0. + 0.50 * s[:, 7:8] * s2[:, 7:8]
        
        # boost_strength              = 0.000001 * s[:, 0:1]
        # dark_thr                    = 0.000001 * s[:, 0:1]
        # low                         = 0.000001 * s[:, 0:1]
        # high                        = 0.000001 * s[:, 0:1]
        # restore_local_contrast_amt  = 0.000001 * s[:, 0:1]
        # restore_local_contrast_sigma= 0.000001 * s[:, 0:1]
        # protect_highlights_strength = 0.000001 * s[:, 0:1]
        # gray_world_balance_mix      = 0.000001 * s[:, 0:1]

        def bcast(v):
            return v.view(-1, 1, 1, 1)  # [B,1] -> [B,1,1,1]

        return dict(
            boost_strength              = bcast(boost_strength),
            dark_thr                    = bcast(dark_thr),
            low                         = bcast(low),
            high                        = bcast(high),
            restore_local_contrast_amount = bcast(restore_local_contrast_amt),
            restore_local_contrast_sigma  = bcast(restore_local_contrast_sigma),
            protect_highlights_strength   = bcast(protect_highlights_strength),
            gray_world_balance_mix        = bcast(gray_world_balance_mix),
        )
    
