import torch
import numpy as np
from typing import Tuple, Dict

# ============== 颜色定义 ==============

# 七种颜色分类（用于分析图像）
# 色相范围基于HSV色环（0-360°归一化到0-1）
COLOR_CATEGORIES = {
    # 红色: 0°-15° 和 345°-360°
    "红": {"hue_ranges": [(0, 0.042), (0.958, 1.0)], "name_en": "red", "hex": "#FF0000", "rgb": (255, 0, 0)},
    # 橙色: 15°-40°
    "橙": {"hue_ranges": [(0.042, 0.111)], "name_en": "orange", "hex": "#FF8000", "rgb": (255, 128, 0)},
    # 黄色: 40°-65°
    "黄": {"hue_ranges": [(0.111, 0.181)], "name_en": "yellow", "hex": "#FFFF00", "rgb": (255, 255, 0)},
    # 绿色: 65°-165°
    "绿": {"hue_ranges": [(0.181, 0.458)], "name_en": "green", "hex": "#00FF00", "rgb": (0, 255, 0)},
    # 青色: 165°-195°
    "青": {"hue_ranges": [(0.458, 0.542)], "name_en": "cyan", "hex": "#00FFFF", "rgb": (0, 255, 255)},
    # 蓝色: 195°-265°
    "蓝": {"hue_ranges": [(0.542, 0.736)], "name_en": "blue", "hex": "#0000FF", "rgb": (0, 0, 255)},
    # 紫色: 265°-345°
    "紫": {"hue_ranges": [(0.736, 0.958)], "name_en": "purple", "hex": "#8000FF", "rgb": (128, 0, 255)},
}

# 中英文映射
EN_TO_CN = {info["name_en"]: cn for cn, info in COLOR_CATEGORIES.items()}
CN_TO_EN = {cn: info["name_en"] for cn, info in COLOR_CATEGORIES.items()}


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RGB转HSV，输入RGB范围0-1
    返回 H(0-1), S(0-1), V(0-1)
    """
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


def analyze_flower_colors(
    pixels: np.ndarray,
    saturation_threshold: float = 0.08,
    value_threshold: float = 0.08
) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    分析像素的颜色分布
    
    Args:
        pixels: shape (N, 3), RGB值范围0-1
        saturation_threshold: 饱和度阈值，低于此值视为灰色
        value_threshold: 亮度阈值，低于此值视为黑色
    
    Returns:
        (各颜色的占比字典, 调试信息字典)
    """
    debug_info = {
        "total_pixels": pixels.shape[0],
        "valid_pixels": 0,
        "filtered_gray": 0,
        "filtered_dark": 0,
    }
    
    empty_ratios = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    
    if pixels.shape[0] == 0:
        return empty_ratios, debug_info
    
    h, s, v = rgb_to_hsv(pixels)
    
    # 统计被过滤的像素
    gray_mask = s < saturation_threshold
    dark_mask = v < value_threshold
    debug_info["filtered_gray"] = int(np.sum(gray_mask & ~dark_mask))
    debug_info["filtered_dark"] = int(np.sum(dark_mask))
    
    # 过滤有效彩色像素（排除灰色和黑色）
    valid_mask = (s >= saturation_threshold) & (v >= value_threshold)
    valid_count = np.sum(valid_mask)
    debug_info["valid_pixels"] = int(valid_count)
    
    if valid_count == 0:
        return empty_ratios, debug_info
    
    h_valid = h[valid_mask]
    s_valid = s[valid_mask]
    v_valid = v[valid_mask]
    
    # 统计各颜色的像素数
    color_counts = {info["name_en"]: 0 for info in COLOR_CATEGORIES.values()}
    
    for color_cn, info in COLOR_CATEGORIES.items():
        color_mask = np.zeros(h_valid.shape, dtype=bool)
        for low, high in info["hue_ranges"]:
            color_mask |= (h_valid >= low) & (h_valid < high)
        
        # 使用饱和度加权计数，使鲜艳的颜色更有代表性
        weighted_count = np.sum(s_valid[color_mask] * v_valid[color_mask])
        color_counts[info["name_en"]] = weighted_count
    
    # 计算占比
    total = sum(color_counts.values())
    if total == 0:
        return {name: 0.0 for name in color_counts}, debug_info
    
    color_ratios = {name: count / total for name, count in color_counts.items()}
    return color_ratios, debug_info


def determine_dominant_color(color_ratios: Dict[str, float], non_flower_threshold: float = 0.02) -> str:
    """
    判断花朵主色
    
    逻辑：
    1. 排除绿色，找占比最大的颜色
    2. 如果除了绿色没有其他颜色（其他颜色占比都小于阈值），返回绿色
    
    Args:
        color_ratios: 各颜色占比字典
        non_flower_threshold: 非绿色颜色的最小阈值，低于此值认为该颜色不存在
    
    Returns:
        主色英文名
    """
    # 排除绿色后的颜色
    non_green_colors = {name: ratio for name, ratio in color_ratios.items() if name != "green"}
    
    # 检查是否有任何非绿色的颜色超过阈值
    has_non_green = any(ratio >= non_flower_threshold for ratio in non_green_colors.values())
    
    if not has_non_green:
        # 没有明显的非绿色颜色，返回绿色
        return "green"
    
    # 找出非绿色中占比最大的颜色
    dominant = max(non_green_colors.keys(), key=lambda x: non_green_colors[x])
    return dominant


def format_analysis_info(
    color_ratios: Dict[str, float],
    dominant_color: str,
    debug_info: Dict[str, any]
) -> str:
    """格式化颜色分析信息"""
    lines = ["===== 花朵主色分析结果 =====", ""]
    
    # 显示像素统计
    if debug_info:
        lines.append("【像素统计】")
        total = debug_info.get("total_pixels", 0)
        valid = debug_info.get("valid_pixels", 0)
        gray = debug_info.get("filtered_gray", 0)
        dark = debug_info.get("filtered_dark", 0)
        lines.append(f"  总像素数: {total:,}")
        lines.append(f"  有效彩色像素: {valid:,} ({valid/total*100:.1f}%)" if total > 0 else "  有效彩色像素: 0")
        lines.append(f"  过滤掉的灰色像素: {gray:,}")
        lines.append(f"  过滤掉的暗色像素: {dark:,}")
        lines.append("")
    
    lines.append("【各颜色占比】")
    
    # 按占比排序
    sorted_ratios = sorted(color_ratios.items(), key=lambda x: x[1], reverse=True)
    
    for name, ratio in sorted_ratios:
        cn_name = EN_TO_CN.get(name, name)
        bar_len = int(ratio * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # 标记绿色（通常是叶子）
        mark = ""
        if name == "green":
            mark = " (叶子/茎)"
        elif name == dominant_color:
            mark = " ★ 主色"
        
        lines.append(f"  {cn_name}色: {bar} {ratio*100:5.1f}%{mark}")
    
    lines.append("")
    lines.append("【主色判断】")
    
    dominant_cn = EN_TO_CN.get(dominant_color, dominant_color)
    dominant_info = None
    for cn, info in COLOR_CATEGORIES.items():
        if info["name_en"] == dominant_color:
            dominant_info = info
            break
    
    if dominant_color == "green":
        lines.append("  判断逻辑: 除绿色外无其他明显颜色")
        lines.append(f"  花朵主色: {dominant_cn}色")
    else:
        green_ratio = color_ratios.get("green", 0)
        dominant_ratio = color_ratios.get(dominant_color, 0)
        lines.append(f"  判断逻辑: 排除绿色({green_ratio*100:.1f}%)后，{dominant_cn}色占比最大")
        lines.append(f"  花朵主色: {dominant_cn}色 ({dominant_ratio*100:.1f}%)")
    
    if dominant_info:
        lines.append(f"  输出颜色: {dominant_info['hex']}")
    
    return "\n".join(lines)


class FlowerDomainColor:
    """
    花朵主色分析节点
    
    输入：图像 + Mask
    输出：
    1. 纯色背景图像（花朵主色）
    2. 十六进制颜色值 (#FF0000格式)
    3. 颜色分析信息
    
    主色判断逻辑：
    - 分析颜色占比，归类为红橙黄绿青蓝紫
    - 优先选择非绿色中占比最大的颜色
    - 如果除了绿色没有其他颜色，返回绿色
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "saturation_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01,
                     "tooltip": "饱和度阈值，低于此值的像素视为灰色，不参与颜色统计"}
                ),
                "value_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01,
                     "tooltip": "亮度阈值，低于此值的像素视为黑色，不参与颜色统计"}
                ),
                "non_flower_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.001, "max": 0.20, "step": 0.001,
                     "tooltip": "非绿色颜色存在阈值，低于此占比的颜色视为不存在"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("color_image", "color_hex", "analysis_info")
    FUNCTION = "process"
    CATEGORY = "auto_chroma_bg"

    def process(
        self,
        image,
        mask,
        saturation_threshold: float = 0.08,
        value_threshold: float = 0.08,
        non_flower_threshold: float = 0.02,
    ):
        # 转换输入
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        img = image[0].cpu().numpy()  # [H, W, C]
        m = mask[0].cpu().numpy()

        # 处理mask维度
        if m.ndim == 3:
            m = m[0]

        # 二值化mask
        m_bin = m > 0.5
        if not np.any(m_bin):
            m_bin = np.ones(m.shape, dtype=bool)

        # 提取mask区域内的像素
        pixels = img[m_bin]  # shape: (N, 3)
        
        # 分析颜色
        color_ratios, debug_info = analyze_flower_colors(
            pixels,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold
        )
        
        # 判断主色
        dominant_color = determine_dominant_color(color_ratios, non_flower_threshold)
        
        # 获取主色信息
        dominant_info = None
        for cn, info in COLOR_CATEGORIES.items():
            if info["name_en"] == dominant_color:
                dominant_info = info
                break
        
        if dominant_info is None:
            # fallback
            dominant_info = COLOR_CATEGORIES["红"]
        
        # 生成纯色图像
        H, W, _ = img.shape
        color_rgb = dominant_info["rgb"]
        color_img = np.zeros((H, W, 3), dtype=np.float32)
        color_img[..., 0] = color_rgb[0] / 255.0
        color_img[..., 1] = color_rgb[1] / 255.0
        color_img[..., 2] = color_rgb[2] / 255.0
        
        color_img_tensor = torch.from_numpy(color_img).unsqueeze(0).float()
        
        # 获取十六进制颜色值
        color_hex = dominant_info["hex"]
        
        # 生成分析信息
        analysis_info = format_analysis_info(color_ratios, dominant_color, debug_info)
        
        return (color_img_tensor, color_hex, analysis_info)


NODE_CLASS_MAPPINGS = {
    "FlowerDomainColor": FlowerDomainColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowerDomainColor": "Flower Domain Color (花朵主色)",
}

