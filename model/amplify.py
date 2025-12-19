# ---------------------------
# 1. 載入 MobileIE Full 模型
# ---------------------------
import sys
sys.path.append("/content/MobileIE")

from model.lle import MobileIELLENet # ★ 使用 Full 模型，而不是 Slim！
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用裝置：", device)

# ★ 使用 Full 模型結構
mobileie = MobileIELLENet(channels=12)

# ★ 載入組員訓練的權重
state = torch.load("/content/model_best.pkl", map_location=device)

# ★ strict=False：避免 tail_warm、conv_bn 等 Full model 內的額外層 mismatch
mobileie.load_state_dict(state, strict=False)

mobileie = mobileie.to(device).eval()
for p in mobileie.parameters():
p.requires_grad = False

print("✔ MobileIE Full 模型載入成功！")


# ---------------------------
# 2. Baseline 推論
# ---------------------------
def run_baseline(img_rgb):
    """MobileIE 原始輸出"""
    t = to_tensor(img_rgb.copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        out = mobileie(t).clamp(0,1)[0].cpu()
    return np.array(to_pil_image(out)).astype(np.uint8)


# ---------------------------
# 3. Loc v3 — 暗部增亮 + 對比補償 + 高光保護
# ---------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def dark_mask(img_rgb, dark_thr=0.55, low=0.20, high=0.65):
    """生成暗部 soft mask：暗的地方為 1，亮部接近 0"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) / 255.0
    inv = np.clip((dark_thr - gray) / dark_thr, 0, 1)
    m = smoothstep(low, high, inv)
    return np.stack([m]*3, axis=-1)

def restore_local_contrast(img_rgb, amount=0.12, sigma=4.0):
    """簡單 local contrast（unsharp mask）"""
    blur = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=sigma)
    out = img_rgb.astype(np.float32) + amount * (img_rgb.astype(np.float32) - blur.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)

def protect_highlights(img_rgb, strength=0.25):
    """壓一下高光，避免太死白；類似 HDR tone mapping"""
    img_f = img_rgb.astype(np.float32) / 255.0
    out = img_f / (img_f + strength) # x / (x + c)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def gray_world_balance(img_rgb, mix=0.4):
    """簡單防偏色：讓三通道平均值接近"""
    img = img_rgb.astype(np.float32)
    mean_c = img.mean(axis=(0,1), keepdims=True) + 1e-6
    mean_all = img.mean() + 1e-6
    gain = mean_all / mean_c # R/G/B 各自調整
    balanced = img * (1.0 * (1-mix) + gain * mix)
    return np.clip(balanced, 0, 255).astype(np.uint8)

def run_loc_v3(original_rgb,
boost_strength=0.16, # 原本 0.24 → 降低增亮幅度
dark_thr=0.48, # 原本 0.55~0.65 → 降低「暗部」覆蓋
low=0.20,
high=0.70,
restore_local_contrast_amount = 0.15,
restore_local_contrast_sigma = 4.0,
protect_highlights_strength = 0.15,
gray_world_balance_mix = 0.30): # 讓 mask 更柔和
    base = run_baseline(original_rgb)

    mask = dark_mask(original_rgb, dark_thr, low, high)
    diff = base.astype(np.float32) - original_rgb.astype(np.float32)

    # (1) 暗部增亮（弱化）
    out = original_rgb.astype(np.float32) + mask * diff * boost_strength
    out = np.clip(out, 0, 255).astype(np.uint8)

    # (2) 對比恢復（強化）
    out = restore_local_contrast(out, amount=restore_local_contrast_amount, sigma=restore_local_contrast_sigma)

    # (3) 高光保留（弱化壓制）
    out = protect_highlights(out, strength=protect_highlights_strength)

    # (4) 防偏色（降低強度）
    out = gray_world_balance(out, mix=gray_world_balance_mix)

    return out, base



# ---------------------------
# 4. AutoSelect v4 — 更聰明選擇 base / loc_v3
# ---------------------------
def compute_gray(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) / 255.0
    return gray.astype(np.float32) # ★ 加這行


def edge_strength(gray):
    """Sobel 邊緣強度"""
    gray = gray.astype(np.float32) # ★★★ 強制處理 dtype
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(mag.mean())


def auto_select_v4(original_rgb, base_rgb, loc_rgb):
    """
    AutoSelect v4.2 — 加入 baseline failure detection：
    ✔ baseline 色偏過大
    ✔ baseline 變負片（大部分比原圖還暗）
    ✔ baseline 爆白太多
    若偵測異常 → 強制使用 loc_v3
    """

    # ---------- 基本灰階 ----------
    g_o = compute_gray(original_rgb)
    g_b = compute_gray(base_rgb)
    g_l = compute_gray(loc_rgb)

    # ========== 基礎統計 ==========
    mean_o = float(g_o.mean())
    bright_o = float((g_o > 0.85).mean())
    contrast_o = float(g_o.std())

    mean_b = float(g_b.mean())
    bright_b = float((g_b > 0.85).mean())
    contrast_b = float(g_b.std())

    mean_l = float(g_l.mean())
    bright_l = float((g_l > 0.85).mean())
    contrast_l = float(g_l.std())

    # ---------- baseline 的亮部擴散 ----------
    bright_spread = bright_b - bright_o

    # ---------- 邊緣指標 ----------
    edge_o = edge_strength(g_o)
    edge_b = edge_strength(g_b)
    edge_l = edge_strength(g_l)

    edge_loss_base = (edge_o - edge_b) / max(edge_o, 1e-6)
    edge_loss_loc = (edge_o - edge_l) / max(edge_o, 1e-6)

    # ===================================================================
    # NEW: Baseline Failure Detection（核心改善）
    # ===================================================================

    # (A) baseline 與原圖差異太大 → 偏色 or 光調崩壞
    color_shift = np.mean(np.abs(base_rgb.astype(float) - original_rgb.astype(float)) / 255.0)

    # (B) baseline 變負片 → 大部分像素比原圖更暗
    neg_score = np.mean(base_rgb.mean(axis=2) < original_rgb.mean(axis=2))

    # (C) baseline 過曝（太白）
    over_b = np.mean(base_rgb > 245)

    baseline_fail = (
        (color_shift > 0.22) or # 色偏過大
        (neg_score > 0.60) or # 類負片
        (over_b > 0.25) # 超過 25% 過曝
    )

    if baseline_fail:
        mode = "loc_v3"
    else:
    # ===================================================================
    # 既有邏輯（保留你的 v4，但更穩定）
    # ===================================================================

        if bright_spread > 0.10 and contrast_o > 0.18:
            mode = "loc_v3"

        elif edge_loss_base > 0.25:
            mode = "loc_v3"

        elif contrast_l < 0.6 * contrast_o:
            mode = "base"

        elif mean_o < 0.22 and bright_b > 0.12:
            mode = "loc_v3"

        else:
            mode = "base"

    # 最終選擇
    final = loc_rgb if mode == "loc_v3" else base_rgb

    # 回傳 debug 資訊
    info = dict(
        mean_o=mean_o, mean_b=mean_b, mean_l=mean_l,
        bright_o=bright_o, bright_b=bright_b, bright_l=bright_l,
        contrast_o=contrast_o, contrast_b=contrast_b, contrast_l=contrast_l,
        bright_spread=bright_spread,
        edge_o=edge_o, edge_b=edge_b, edge_l=edge_l,
        edge_loss_base=edge_loss_base,
        edge_loss_loc=edge_loss_loc,
        color_shift=color_shift,
        neg_score=neg_score,
        over_b=over_b,
        baseline_fail=baseline_fail
    )

    return final, mode, info



# ---------------------------
# 5. 單張圖片：compare 四圖
# ---------------------------
def show_compare(path, resize_to=None):
    bgr = cv2.imread(path)
    if bgr is None:
        print("❌ 無法讀取圖片：", path)
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize_to is not None:
        rgb = cv2.resize(rgb, (resize_to, resize_to))

    loc3, base = run_loc_v3(rgb)
    final, mode, info = auto_select_v4(rgb, base, loc3)

    print("\n==============================")
    print("📌 圖片：", path)
    print(f"mean_o={info['mean_o']:.3f}, contrast_o={info['contrast_o']:.3f}")
    print(f"bright_o={info['bright_o']:.3f}, bright_b={info['bright_b']:.3f}, bright_l={info['bright_l']:.3f}")
    print(f"bright_spread={info['bright_spread']:.3f}")
    print(f"edge_o={info['edge_o']:.3f}, edge_b={info['edge_b']:.3f}, edge_l={info['edge_l']:.3f}")
    print(f"edge_loss_base={info['edge_loss_base']:.3f}, edge_loss_loc={info['edge_loss_loc']:.3f}")
    print(f"👉 AutoSelect v4 模式：{mode}")
    print("==============================")

    plt.figure(figsize=(20,6))
    plt.subplot(1,4,1); plt.imshow(rgb); plt.title("Original"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(base); plt.title("MobileIE Base"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(loc3); plt.title("Loc v3"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(final);plt.title(f"Final ({mode})"); plt.axis("off")
    plt.show()


    print("\n✅ Loc v3 + AutoSelect v4 已設定完成，你可以用 show_compare() 來測圖。")

    # 我自己的測試集路徑(記得改)
    exdark_root = "/content/contrast_dataset/ExDark"
    paths = []
    for cls in os.listdir(exdark_root):
        d = os.path.join(exdark_root, cls)
        if os.path.isdir(d):
            paths += glob.glob(d + "/*.jpg") + glob.glob(d + "/*.png")

    print("ExDark 圖片數：", len(paths))
    for p in random.sample(paths, 10):
        show_compare(p, resize_to=256)