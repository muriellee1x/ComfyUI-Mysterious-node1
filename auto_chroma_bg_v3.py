import torch
import numpy as np

def rgb_to_hsv(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc)
    non_zero = maxc > 1e-6
    s[non_zero] = delta[non_zero] / maxc[non_zero]

    h = np.zeros_like(maxc)
    mask = delta > 1e-6

    idx = mask & (maxc == r)
    h[idx] = (g[idx] - b[idx]) / (delta[idx] + 1e-6)

    idx = mask & (maxc == g)
    h[idx] = 2.0 + (b[idx] - r[idx]) / (delta[idx] + 1e-6)

    idx = mask & (maxc == b)
    h[idx] = 4.0 + (r[idx] - g[idx]) / (delta[idx] + 1e-6)

    h = (h / 6.0) % 1.0

    return h, s, v


class AutoChromaBackgroundV3Image:
    """
    输入：IMAGE + MASK
    输出：一张与输入同尺寸的纯色背景图（IMAGE）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "green_ratio_threshold": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0001, "max": 0.1, "step": 0.0005},
                ),
                "blue_ratio_threshold": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0001, "max": 0.2, "step": 0.0005},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color_image",)
    FUNCTION = "make_color_image"
    CATEGORY = "auto_chroma_bg_v3"

    def make_color_image(
        self,
        image,
        mask,
        green_ratio_threshold,
        blue_ratio_threshold,
    ):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        img = image[0].cpu().numpy()  # [H,W,C]
        m = mask[0].cpu().numpy()

        if m.ndim == 3:
            m = m[0]

        m_bin = m > 0.5
        if not np.any(m_bin):
            m_bin = np.ones(m.shape, dtype=bool)

        pixels = img[m_bin]
        if pixels.shape[0] == 0:
            bg = np.array([0, 0, 255])  # fallback
        else:
            h, s, v = rgb_to_hsv(pixels)

            green_mask = (
                (h >= 70/360) &
                (h <= 170/360) &
                (s > 0.2) &
                (v > 0.2)
            )
            blue_mask = (
                (h >= 190/360) &
                (h <= 260/360) &
                (s > 0.2) &
                (v > 0.15)
            )

            has_green = green_mask.mean() >= green_ratio_threshold
            has_blue = blue_mask.mean() >= blue_ratio_threshold

            candidates = {
                "green": np.array([0, 255, 0]),
                "blue": np.array([0, 0, 255]),
                "magenta": np.array([255, 0, 255]),
                "cyan": np.array([0, 255, 255]),
            }

            allowed = set(candidates.keys())

            if has_green and "green" in allowed:
                allowed.remove("green")
            if has_blue and "blue" in allowed:
                allowed.remove("blue")

            if not allowed:
                allowed = {"magenta", "cyan"}

            mean_rgb_255 = (pixels.mean(axis=0) * 255).astype(np.float32)

            best_name = None
            best_dist = -1
            for name in allowed:
                c = candidates[name]
                dist = np.linalg.norm(c - mean_rgb_255)
                if dist > best_dist:
                    best_dist = dist
                    best_name = name

            bg = candidates[best_name]

        H, W, _ = img.shape
        bg_img = np.tile(bg.reshape(1,1,3), (H, W, 1)) / 255.0  # 0–1

        bg_img_tensor = torch.from_numpy(bg_img).unsqueeze(0).float()

        return (bg_img_tensor,)


NODE_CLASS_MAPPINGS = {
    "AutoChromaBackgroundV3Image": AutoChromaBackgroundV3Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoChromaBackgroundV3Image": "Auto Chroma BG v3 (Image Output)",
}
