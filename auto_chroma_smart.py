import torch
import numpy as np
from typing import Tuple, Dict, List

# ============== 颜色定义 ==============

# 七种颜色分类（用于分析图像）
# 色相范围基于HSV色环（0-360°归一化到0-1）
# 特别优化：扩大绿色范围以覆盖自然界中的黄绿色（植物叶子等）
COLOR_CATEGORIES = {
    # 红色: 0°-15° 和 345°-360°（粉色、玫红等偏红色调）
    "红": {"hue_ranges": [(0, 0.042), (0.958, 1.0)], "name_en": "red"},
    # 橙色: 15°-40°
    "橙": {"hue_ranges": [(0.042, 0.111)], "name_en": "orange"},
    # 黄色: 40°-65°（纯黄到黄橙）
    "黄": {"hue_ranges": [(0.111, 0.181)], "name_en": "yellow"},
    # 绿色: 65°-165°（包含黄绿、草绿、翠绿、深绿）
    # 关键改进：从65°开始，覆盖自然界植物的黄绿色
    "绿": {"hue_ranges": [(0.181, 0.458)], "name_en": "green"},
    # 青色: 165°-195°
    "青": {"hue_ranges": [(0.458, 0.542)], "name_en": "cyan"},
    # 蓝色: 195°-265°
    "蓝": {"hue_ranges": [(0.542, 0.736)], "name_en": "blue"},
    # 紫色: 265°-345°（蓝紫、紫罗兰、品红）
    "紫": {"hue_ranges": [(0.736, 0.958)], "name_en": "purple"},
}

# 可选背景色（RGB值）
BACKGROUND_COLORS = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
}

# 互补色映射（图像主色 -> 推荐背景色优先级列表）
COMPLEMENTARY_MAP = {
    "red": ["cyan", "blue", "green"],
    "orange": ["blue", "cyan", "green"],
    "yellow": ["blue", "cyan", "red"],
    "green": ["red", "yellow", "blue"],
    "cyan": ["red", "yellow", "blue"],
    "blue": ["yellow", "red", "green"],
    "purple": ["yellow", "green", "red"],
}


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


def classify_hue(h: float) -> str:
    """根据色相值分类颜色（返回英文名）"""
    for color_cn, info in COLOR_CATEGORIES.items():
        for low, high in info["hue_ranges"]:
            if low <= h < high:
                return info["name_en"]
    return "red"  # fallback


def analyze_colors(
    pixels: np.ndarray,
    saturation_threshold: float = 0.08,
    value_threshold: float = 0.08
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, any]]:
    """
    分析像素的颜色分布
    
    Args:
        pixels: shape (N, 3), RGB值范围0-1
        saturation_threshold: 饱和度阈值，低于此值视为灰色
        value_threshold: 亮度阈值，低于此值视为黑色
    
    Returns:
        (各颜色的占比字典, 各颜色的最大饱和度字典, 调试信息字典)
    """
    debug_info = {
        "total_pixels": pixels.shape[0],
        "valid_pixels": 0,
        "filtered_gray": 0,
        "filtered_dark": 0,
    }
    
    empty_ratios = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    empty_max_sat = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    
    if pixels.shape[0] == 0:
        return empty_ratios, empty_max_sat, debug_info
    
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
        # 没有有效彩色像素，返回全0
        return empty_ratios, empty_max_sat, debug_info
    
    h_valid = h[valid_mask]
    s_valid = s[valid_mask]
    v_valid = v[valid_mask]
    
    # 统计各颜色的像素数和最大饱和度
    color_counts = {info["name_en"]: 0 for info in COLOR_CATEGORIES.values()}
    color_max_saturation = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    
    for color_cn, info in COLOR_CATEGORIES.items():
        color_mask = np.zeros(h_valid.shape, dtype=bool)
        for low, high in info["hue_ranges"]:
            color_mask |= (h_valid >= low) & (h_valid < high)
        
        # 使用饱和度加权计数，使鲜艳的颜色更有代表性
        weighted_count = np.sum(s_valid[color_mask] * v_valid[color_mask])
        color_counts[info["name_en"]] = weighted_count
        
        # 记录该颜色的最大饱和度（用于判断是否足够鲜艳）
        if np.any(color_mask):
            # 使用 top 5% 分位数作为"最大饱和度"，避免单个像素噪点影响
            s_color = s_valid[color_mask]
            if len(s_color) >= 20:
                # 足够多的像素时，用95%分位数
                color_max_saturation[info["name_en"]] = float(np.percentile(s_color, 95))
            else:
                # 像素太少时，直接用最大值
                color_max_saturation[info["name_en"]] = float(np.max(s_color))
    
    # 计算占比
    total = sum(color_counts.values())
    if total == 0:
        return {name: 0.0 for name in color_counts}, color_max_saturation, debug_info
    
    color_ratios = {name: count / total for name, count in color_counts.items()}
    return color_ratios, color_max_saturation, debug_info


def select_background_color(
    color_ratios: Dict[str, float],
    color_max_saturation: Dict[str, float],
    presence_threshold: float = 0.01,
    vivid_threshold: float = 0.5,
    manual_disabled: set = None
) -> Tuple[str, Tuple[int, int, int]]:
    """
    根据颜色分布选择背景色
    
    Args:
        color_ratios: 各颜色占比
        color_max_saturation: 各颜色的最大饱和度
        presence_threshold: 认为某颜色"存在"的最小占比阈值
        vivid_threshold: 鲜艳度阈值，只有最大饱和度超过此值的颜色才会被自动禁用
        manual_disabled: 用户手动禁用的背景色集合
    
    Returns:
        (颜色名, RGB值)
    
    优先级：
    1. 手动禁用是绝对优先级，永远不会被选中（除非所有颜色都被手动禁用）
    2. 自动禁用可以在必要时被忽略
    3. 低饱和度的颜色即使存在也不会被自动禁用（兜底方案）
    """
    if manual_disabled is None:
        manual_disabled = set()
    
    # 找出图像中存在且足够鲜艳的颜色（用于自动禁止列表）
    # 关键改进：只有饱和度足够高的颜色才会被自动禁用
    auto_disabled = {
        name for name, ratio in color_ratios.items()
        if ratio >= presence_threshold 
        and name in BACKGROUND_COLORS
        and color_max_saturation.get(name, 0) >= vivid_threshold  # 饱和度判断
    }
    
    # 合并禁止列表：自动禁用 + 手动禁用
    all_forbidden = auto_disabled | manual_disabled
    
    # 找到最大面积的颜色
    dominant_color = max(color_ratios.keys(), key=lambda x: color_ratios[x])
    
    # 获取互补色候选列表
    candidates = COMPLEMENTARY_MAP.get(dominant_color, ["cyan", "blue", "green"])
    
    # 第一优先：从候选列表中选择第一个完全不在禁止列表中的颜色
    for candidate in candidates:
        if candidate not in all_forbidden:
            return candidate, BACKGROUND_COLORS[candidate]
    
    # 第二优先：从所有背景色中选择不在禁止列表中的、占比最小的
    available = [
        (name, color_ratios.get(name, 0))
        for name in BACKGROUND_COLORS.keys()
        if name not in all_forbidden
    ]
    
    if available:
        available.sort(key=lambda x: x[1])
        name = available[0][0]
        return name, BACKGROUND_COLORS[name]
    
    # 第三优先：忽略自动禁用，但保留手动禁用
    # 关键改进：在候选列表中优先选择饱和度最低的颜色（冲突最小）
    
    # 3a: 从互补色候选列表中选择饱和度最低的（非手动禁用的）
    candidate_saturations = [
        (name, color_max_saturation.get(name, 0))
        for name in candidates
        if name not in manual_disabled
    ]
    
    if candidate_saturations:
        # 按饱和度排序，选最低的
        candidate_saturations.sort(key=lambda x: x[1])
        name = candidate_saturations[0][0]
        return name, BACKGROUND_COLORS[name]
    
    # 3b: 如果候选列表全是手动禁用的，从其他背景色中选饱和度最低的
    non_manual_disabled = [
        (name, color_max_saturation.get(name, 0))
        for name in BACKGROUND_COLORS.keys()
        if name not in manual_disabled
    ]
    
    if non_manual_disabled:
        # 按饱和度排序，选最低的
        non_manual_disabled.sort(key=lambda x: x[1])
        name = non_manual_disabled[0][0]
        return name, BACKGROUND_COLORS[name]
    
    # 极端情况：所有颜色都被手动禁用了，只能忽略禁用选占比最小的
    bg_ratios = [
        (name, color_ratios.get(name, 0))
        for name in BACKGROUND_COLORS.keys()
    ]
    bg_ratios.sort(key=lambda x: x[1])
    name = bg_ratios[0][0]
    return name, BACKGROUND_COLORS[name]


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """RGB转十六进制颜色码"""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def format_color_info(
    color_ratios: Dict[str, float], 
    color_max_saturation: Dict[str, float],
    selected_bg: str,
    debug_info: Dict[str, any] = None,
    presence_threshold: float = 0.01,
    vivid_threshold: float = 0.5,
    manual_disabled: set = None
) -> str:
    """格式化颜色分析信息"""
    if manual_disabled is None:
        manual_disabled = set()
    
    # 中英文映射
    en_to_cn = {info["name_en"]: cn for cn, info in COLOR_CATEGORIES.items()}
    # 背景色的中英文映射
    bg_en_to_cn = {"red": "红", "yellow": "黄", "green": "绿", "cyan": "青", "blue": "蓝"}
    
    lines = ["===== 颜色分析结果 =====", ""]
    
    # 显示调试信息
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
    
    # 显示手动禁用的颜色
    if manual_disabled:
        lines.append("【手动禁用】")
        disabled_cn = [bg_en_to_cn.get(c, c) + "色" for c in manual_disabled]
        lines.append(f"  用户禁用: {', '.join(disabled_cn)}")
        lines.append("")
    
    lines.append("【各颜色占比与饱和度】")
    lines.append(f"  (鲜艳度阈值: {vivid_threshold*100:.0f}%)")
    
    # 按占比排序
    sorted_ratios = sorted(color_ratios.items(), key=lambda x: x[1], reverse=True)
    
    for name, ratio in sorted_ratios:
        cn_name = en_to_cn.get(name, name)
        bar_len = int(ratio * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        max_sat = color_max_saturation.get(name, 0)
        
        # 标记被禁用的背景色
        forbidden_mark = ""
        if name in BACKGROUND_COLORS:
            if name in manual_disabled:
                forbidden_mark = " [手动禁用]"
            elif ratio >= presence_threshold and max_sat >= vivid_threshold:
                forbidden_mark = " [自动禁用]"
            elif ratio >= presence_threshold and max_sat < vivid_threshold:
                forbidden_mark = " [低饱和-可用]"
        
        # 显示饱和度信息
        sat_info = f"饱和度:{max_sat*100:4.0f}%"
        lines.append(f"  {cn_name}色: {bar} {ratio*100:5.1f}% {sat_info}{forbidden_mark}")
    
    lines.append("")
    lines.append("【背景色选择】")
    
    # 找主色
    dominant = max(color_ratios.keys(), key=lambda x: color_ratios[x])
    dominant_cn = en_to_cn.get(dominant, dominant)
    dominant_ratio = color_ratios[dominant]
    
    selected_cn = en_to_cn.get(selected_bg, selected_bg)
    
    lines.append(f"  主要颜色: {dominant_cn}色 ({dominant_ratio*100:.1f}%)")
    lines.append(f"  选择背景: {selected_cn}色 ({rgb_to_hex(BACKGROUND_COLORS[selected_bg])})")
    lines.append("")
    lines.append("【选择理由】")
    lines.append(f"  优先使用{dominant_cn}色的互补色")
    lines.append(f"  排除：手动禁用 + 高饱和度(≥{vivid_threshold*100:.0f}%)且存在的颜色")
    
    return "\n".join(lines)


class AutoChromaSmartBackground:
    """
    智能背景色选择节点
    
    输入：图像 + Mask
    输出：
    1. 纯色背景图像
    2. 十六进制颜色值 (#FF0000格式)
    3. 颜色分析信息
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
                     "tooltip": "饱和度阈值，低于此值的像素视为灰色，不参与颜色统计（降低可检测更多暗淡颜色）"}
                ),
                "value_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01,
                     "tooltip": "亮度阈值，低于此值的像素视为黑色，不参与颜色统计（降低可检测更多深色）"}
                ),
                "presence_threshold": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.001, "max": 0.10, "step": 0.001,
                     "tooltip": "颜色存在阈值，超过此占比的颜色视为'存在'，将被禁止作为背景色"}
                ),
                "vivid_threshold": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.10, "max": 1.00, "step": 0.05,
                     "tooltip": "鲜艳度阈值，只有最大饱和度超过此值的颜色才会被自动禁用（低饱和度的颜色可作为兜底）"}
                ),
                # 手动禁用背景色选项
                "disable_red_bg": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "禁止使用红色作为背景色"}
                ),
                "disable_yellow_bg": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "禁止使用黄色作为背景色"}
                ),
                "disable_green_bg": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "禁止使用绿色作为背景色"}
                ),
                "disable_cyan_bg": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "禁止使用青色作为背景色"}
                ),
                "disable_blue_bg": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "禁止使用蓝色作为背景色"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("background_image", "color_hex", "analysis_info")
    FUNCTION = "process"
    CATEGORY = "auto_chroma_bg"

    def process(
        self,
        image,
        mask,
        saturation_threshold: float = 0.08,
        value_threshold: float = 0.08,
        presence_threshold: float = 0.005,
        vivid_threshold: float = 0.50,
        disable_red_bg: bool = False,
        disable_yellow_bg: bool = False,
        disable_green_bg: bool = False,
        disable_cyan_bg: bool = False,
        disable_blue_bg: bool = False,
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
        
        # 分析颜色（包含最大饱和度统计）
        color_ratios, color_max_saturation, debug_info = analyze_colors(
            pixels,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold
        )
        
        # 构建手动禁用列表
        manual_disabled = set()
        if disable_red_bg:
            manual_disabled.add("red")
        if disable_yellow_bg:
            manual_disabled.add("yellow")
        if disable_green_bg:
            manual_disabled.add("green")
        if disable_cyan_bg:
            manual_disabled.add("cyan")
        if disable_blue_bg:
            manual_disabled.add("blue")
        
        # 选择背景色（考虑饱和度）
        bg_name, bg_rgb = select_background_color(
            color_ratios,
            color_max_saturation,
            presence_threshold=presence_threshold,
            vivid_threshold=vivid_threshold,
            manual_disabled=manual_disabled
        )
        
        # 生成背景图像
        H, W, _ = img.shape
        bg_img = np.zeros((H, W, 3), dtype=np.float32)
        bg_img[..., 0] = bg_rgb[0] / 255.0
        bg_img[..., 1] = bg_rgb[1] / 255.0
        bg_img[..., 2] = bg_rgb[2] / 255.0
        
        bg_img_tensor = torch.from_numpy(bg_img).unsqueeze(0).float()
        
        # 生成颜色十六进制值
        color_hex = rgb_to_hex(bg_rgb)
        
        # 生成分析信息（包含饱和度数据）
        analysis_info = format_color_info(
            color_ratios,
            color_max_saturation,
            bg_name, 
            debug_info,
            presence_threshold,
            vivid_threshold,
            manual_disabled
        )
        
        return (bg_img_tensor, color_hex, analysis_info)


NODE_CLASS_MAPPINGS = {
    "AutoChromaSmartBackground": AutoChromaSmartBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoChromaSmartBackground": "Auto Chroma Smart BG (智能背景色)",
}

