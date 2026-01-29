import torch
import numpy as np
from typing import Tuple, Dict

# ============== 颜色定义 ==============

# 八种颜色分类（用于分析图像）
# 色相范围基于HSV色环（0-360°归一化到0-1）
# 白色通过低饱和度高亮度来判断，不依赖色相
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
    # 白色: 低饱和度高亮度（特殊判断，不使用色相范围）
    "白": {"hue_ranges": [], "name_en": "white", "hex": "#FFFFFF", "rgb": (255, 255, 255)},
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
    value_threshold: float = 0.08,
    white_saturation_max: float = 0.15,
    white_value_min: float = 0.75
) -> Tuple[Dict[str, float], Dict[str, Dict], Dict[str, any]]:
    """
    分析像素的颜色分布
    
    Args:
        pixels: shape (N, 3), RGB值范围0-1
        saturation_threshold: 饱和度阈值，低于此值视为灰色（但高亮度低饱和度可能是白色）
        value_threshold: 亮度阈值，低于此值视为黑色
        white_saturation_max: 白色判断的最大饱和度阈值
        white_value_min: 白色判断的最小亮度阈值
    
    Returns:
        (各颜色的占比字典, 各颜色的平均HSV字典, 调试信息字典)
    """
    debug_info = {
        "total_pixels": pixels.shape[0],
        "valid_pixels": 0,
        "filtered_gray": 0,
        "filtered_dark": 0,
        "white_pixels": 0,
    }
    
    empty_ratios = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    empty_hsv = {info["name_en"]: {"h": 0.0, "s": 0.0, "v": 1.0} for info in COLOR_CATEGORIES.values()}
    
    if pixels.shape[0] == 0:
        return empty_ratios, empty_hsv, debug_info
    
    h, s, v = rgb_to_hsv(pixels)
    
    # 白色判断：低饱和度 + 高亮度
    white_mask = (s < white_saturation_max) & (v >= white_value_min)
    white_count = np.sum(white_mask)
    debug_info["white_pixels"] = int(white_count)
    
    # 统计被过滤的像素（排除白色后的灰色像素）
    gray_mask = (s < saturation_threshold) & ~white_mask
    dark_mask = v < value_threshold
    debug_info["filtered_gray"] = int(np.sum(gray_mask & ~dark_mask))
    debug_info["filtered_dark"] = int(np.sum(dark_mask))
    
    # 有效彩色像素（排除灰色和黑色，但不排除白色）
    colored_mask = (s >= saturation_threshold) & (v >= value_threshold)
    
    # 所有有效像素 = 彩色像素 + 白色像素
    valid_mask = colored_mask | white_mask
    valid_count = np.sum(valid_mask)
    debug_info["valid_pixels"] = int(valid_count)
    
    if valid_count == 0:
        return empty_ratios, empty_hsv, debug_info
    
    # 统计各颜色的像素数和平均HSV
    color_counts = {info["name_en"]: 0.0 for info in COLOR_CATEGORIES.values()}
    color_hsv_sums = {info["name_en"]: {"h_sin": 0.0, "h_cos": 0.0, "s": 0.0, "v": 0.0, "count": 0} 
                      for info in COLOR_CATEGORIES.values()}
    
    # 先处理白色
    if white_count > 0:
        h_white = h[white_mask]
        s_white = s[white_mask]
        v_white = v[white_mask]
        
        # 白色的权重使用亮度
        weighted_count = np.sum(v_white)
        color_counts["white"] = weighted_count
        
        # 记录白色的HSV值（用于计算平均）
        color_hsv_sums["white"]["h_sin"] = np.sum(np.sin(h_white * 2 * np.pi))
        color_hsv_sums["white"]["h_cos"] = np.sum(np.cos(h_white * 2 * np.pi))
        color_hsv_sums["white"]["s"] = np.sum(s_white)
        color_hsv_sums["white"]["v"] = np.sum(v_white)
        color_hsv_sums["white"]["count"] = white_count
    
    # 处理其他彩色像素
    h_colored = h[colored_mask]
    s_colored = s[colored_mask]
    v_colored = v[colored_mask]
    
    for color_cn, info in COLOR_CATEGORIES.items():
        if info["name_en"] == "white":
            continue  # 白色已经处理过了
            
        color_mask = np.zeros(h_colored.shape, dtype=bool)
        for low, high in info["hue_ranges"]:
            color_mask |= (h_colored >= low) & (h_colored < high)
        
        if np.sum(color_mask) > 0:
            h_this = h_colored[color_mask]
            s_this = s_colored[color_mask]
            v_this = v_colored[color_mask]
            
            # 使用饱和度加权计数，使鲜艳的颜色更有代表性
            weighted_count = np.sum(s_this * v_this)
            color_counts[info["name_en"]] = weighted_count
            
            # 记录HSV值（用于计算平均）
            # 色相使用圆形平均（sin/cos方式）避免0°和360°的跳变问题
            color_hsv_sums[info["name_en"]]["h_sin"] = np.sum(np.sin(h_this * 2 * np.pi))
            color_hsv_sums[info["name_en"]]["h_cos"] = np.sum(np.cos(h_this * 2 * np.pi))
            color_hsv_sums[info["name_en"]]["s"] = np.sum(s_this)
            color_hsv_sums[info["name_en"]]["v"] = np.sum(v_this)
            color_hsv_sums[info["name_en"]]["count"] = int(np.sum(color_mask))
    
    # 计算占比
    total = sum(color_counts.values())
    if total == 0:
        return {name: 0.0 for name in color_counts}, empty_hsv, debug_info
    
    color_ratios = {name: count / total for name, count in color_counts.items()}
    
    # 计算各颜色的平均HSV
    color_avg_hsv = {}
    for name, sums in color_hsv_sums.items():
        if sums["count"] > 0:
            # 使用圆形平均计算色相
            avg_h = np.arctan2(sums["h_sin"], sums["h_cos"]) / (2 * np.pi)
            if avg_h < 0:
                avg_h += 1.0
            avg_s = sums["s"] / sums["count"]
            avg_v = sums["v"] / sums["count"]
            color_avg_hsv[name] = {"h": float(avg_h), "s": float(avg_s), "v": float(avg_v)}
        else:
            color_avg_hsv[name] = {"h": 0.0, "s": 0.0, "v": 1.0}
    
    return color_ratios, color_avg_hsv, debug_info


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


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """
    HSV转RGB
    
    Args:
        h: 色相 0-1
        s: 饱和度 0-1
        v: 亮度 0-1
    
    Returns:
        (R, G, B) 范围 0-255
    """
    if s == 0:
        r = g = b = int(v * 255)
        return (r, g, b)
    
    h = h * 6.0
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """RGB转十六进制颜色值"""
    return f"#{r:02X}{g:02X}{b:02X}"


def format_analysis_info(
    color_ratios: Dict[str, float],
    color_avg_hsv: Dict[str, Dict],
    dominant_color: str,
    output_hsv: Dict[str, float],
    output_rgb: Tuple[int, int, int],
    output_hex: str,
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
        white = debug_info.get("white_pixels", 0)
        lines.append(f"  总像素数: {total:,}")
        lines.append(f"  有效像素: {valid:,} ({valid/total*100:.1f}%)" if total > 0 else "  有效像素: 0")
        lines.append(f"  其中白色像素: {white:,}")
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
        
        # 显示该颜色的平均HSV
        hsv_info = ""
        if ratio > 0 and name in color_avg_hsv:
            avg = color_avg_hsv[name]
            hsv_info = f" [H:{avg['h']*360:.0f}° S:{avg['s']*100:.0f}% V:{avg['v']*100:.0f}%]"
        
        lines.append(f"  {cn_name}色: {bar} {ratio*100:5.1f}%{mark}{hsv_info}")
    
    lines.append("")
    lines.append("【主色判断】")
    
    dominant_cn = EN_TO_CN.get(dominant_color, dominant_color)
    
    if dominant_color == "green":
        lines.append("  判断逻辑: 除绿色外无其他明显颜色")
        lines.append(f"  花朵主色: {dominant_cn}色")
    else:
        green_ratio = color_ratios.get("green", 0)
        dominant_ratio = color_ratios.get(dominant_color, 0)
        lines.append(f"  判断逻辑: 排除绿色({green_ratio*100:.1f}%)后，{dominant_cn}色占比最大")
        lines.append(f"  花朵主色: {dominant_cn}色 ({dominant_ratio*100:.1f}%)")
    
    lines.append("")
    lines.append("【输出颜色（基于实际花朵颜色）】")
    lines.append(f"  HSV: H={output_hsv['h']*360:.1f}° S={output_hsv['s']*100:.1f}% V={output_hsv['v']*100:.1f}%")
    lines.append(f"  RGB: ({output_rgb[0]}, {output_rgb[1]}, {output_rgb[2]})")
    lines.append(f"  HEX: {output_hex}")
    
    return "\n".join(lines)


class FlowerDomainColor:
    """
    花朵主色分析节点
    
    输入：图像 + Mask
    输出：
    1. 纯色背景图像（基于花朵实际颜色的平均HSV）
    2. 十六进制颜色值 (#FF0000格式)
    3. 颜色分析信息
    
    主色判断逻辑：
    - 分析颜色占比，归类为红橙黄绿青蓝紫白
    - 优先选择非绿色中占比最大的颜色
    - 如果除了绿色没有其他颜色，返回绿色
    - 输出颜色使用该颜色分类的平均色相和饱和度，更接近实际花朵颜色
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
                "white_saturation_max": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.05, "max": 0.40, "step": 0.01,
                     "tooltip": "白色判断的最大饱和度阈值，低于此值且高亮度的像素视为白色"}
                ),
                "white_value_min": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.50, "max": 0.95, "step": 0.01,
                     "tooltip": "白色判断的最小亮度阈值，高于此值且低饱和度的像素视为白色"}
                ),
                "output_value_boost": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.50, "max": 1.00, "step": 0.01,
                     "tooltip": "输出颜色的亮度值，设置较高以确保颜色鲜明"}
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
        white_saturation_max: float = 0.15,
        white_value_min: float = 0.75,
        output_value_boost: float = 0.95,
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
        color_ratios, color_avg_hsv, debug_info = analyze_flower_colors(
            pixels,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            white_saturation_max=white_saturation_max,
            white_value_min=white_value_min
        )
        
        # 判断主色
        dominant_color = determine_dominant_color(color_ratios, non_flower_threshold)
        
        # 获取主色的平均HSV值
        if dominant_color in color_avg_hsv and color_ratios.get(dominant_color, 0) > 0:
            avg_hsv = color_avg_hsv[dominant_color]
            output_h = avg_hsv["h"]
            output_s = avg_hsv["s"]
            # 使用设定的亮度值，使颜色更鲜明
            output_v = output_value_boost
            
            # 对于白色，保持低饱和度高亮度
            if dominant_color == "white":
                output_s = min(output_s, 0.1)  # 白色保持低饱和度
                output_v = max(output_v, 0.95)  # 白色保持高亮度
        else:
            # fallback: 使用预定义的颜色
            dominant_info = None
            for cn, info in COLOR_CATEGORIES.items():
                if info["name_en"] == dominant_color:
                    dominant_info = info
                    break
            
            if dominant_info is None:
                dominant_info = COLOR_CATEGORIES["红"]
            
            # 从预定义RGB转换到HSV
            r, g, b = dominant_info["rgb"]
            rgb_arr = np.array([[[r/255, g/255, b/255]]])
            h_arr, s_arr, v_arr = rgb_to_hsv(rgb_arr)
            output_h = float(h_arr[0, 0])
            output_s = float(s_arr[0, 0])
            output_v = output_value_boost
        
        output_hsv = {"h": output_h, "s": output_s, "v": output_v}
        
        # 将HSV转换为RGB
        output_rgb = hsv_to_rgb(output_h, output_s, output_v)
        output_hex = rgb_to_hex(*output_rgb)
        
        # 生成纯色图像
        H, W, _ = img.shape
        color_img = np.zeros((H, W, 3), dtype=np.float32)
        color_img[..., 0] = output_rgb[0] / 255.0
        color_img[..., 1] = output_rgb[1] / 255.0
        color_img[..., 2] = output_rgb[2] / 255.0
        
        color_img_tensor = torch.from_numpy(color_img).unsqueeze(0).float()
        
        # 生成分析信息
        analysis_info = format_analysis_info(
            color_ratios, color_avg_hsv, dominant_color,
            output_hsv, output_rgb, output_hex, debug_info
        )
        
        return (color_img_tensor, output_hex, analysis_info)


NODE_CLASS_MAPPINGS = {
    "FlowerDomainColor": FlowerDomainColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowerDomainColor": "Flower Domain Color (花朵主色)",
}

