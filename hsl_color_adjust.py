import torch
import numpy as np
from typing import Tuple

# ============== HSL颜色范围定义 ==============
# 色相范围基于HSL色环（0-360°）
# 分为五种主要颜色区域

HSL_COLOR_RANGES = {
    "red": {
        # 红色跨越0度: 330-360 和 0-30
        "hue_ranges": [(330, 360), (0, 30)],
        "center": 0,  # 色相中心点
    },
    "yellow": {
        # 黄色: 30-90
        "hue_ranges": [(30, 90)],
        "center": 60,
    },
    "green": {
        # 绿色: 90-150
        "hue_ranges": [(90, 150)],
        "center": 120,
    },
    "cyan": {
        # 青色: 150-210
        "hue_ranges": [(150, 210)],
        "center": 180,
    },
    "blue": {
        # 蓝色: 210-270
        "hue_ranges": [(210, 270)],
        "center": 240,
    },
}


def rgb_to_hsl(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RGB转HSL
    输入: RGB数组，范围0-1，shape (..., 3)
    返回: H(0-360), S(0-1), L(0-1)
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    
    # Lightness
    l = (maxc + minc) / 2.0
    
    # Saturation
    delta = maxc - minc
    s = np.zeros_like(l)
    
    # 避免除以0
    mask = delta > 1e-6
    
    # 当L <= 0.5时
    low_l = mask & (l <= 0.5)
    s[low_l] = delta[low_l] / (maxc[low_l] + minc[low_l] + 1e-6)
    
    # 当L > 0.5时
    high_l = mask & (l > 0.5)
    s[high_l] = delta[high_l] / (2.0 - maxc[high_l] - minc[high_l] + 1e-6)
    
    # Hue
    h = np.zeros_like(l)
    
    # R is max
    idx = mask & (maxc == r)
    h[idx] = 60.0 * (((g[idx] - b[idx]) / (delta[idx] + 1e-6)) % 6)
    
    # G is max
    idx = mask & (maxc == g)
    h[idx] = 60.0 * ((b[idx] - r[idx]) / (delta[idx] + 1e-6) + 2)
    
    # B is max
    idx = mask & (maxc == b)
    h[idx] = 60.0 * ((r[idx] - g[idx]) / (delta[idx] + 1e-6) + 4)
    
    # 确保H在0-360范围内
    h = h % 360.0
    
    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    HSL转RGB
    输入: H(0-360), S(0-1), L(0-1)
    返回: RGB数组，范围0-1
    """
    # 将H归一化到0-1
    h_norm = (h / 360.0) % 1.0
    
    # 计算中间值
    c = (1 - np.abs(2 * l - 1)) * s  # Chroma
    x = c * (1 - np.abs((h_norm * 6) % 2 - 1))
    m = l - c / 2
    
    # 根据色相区间确定RGB
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    
    # 0-60度: R=C, G=X, B=0
    mask = (h >= 0) & (h < 60)
    rgb[mask, 0] = c[mask]
    rgb[mask, 1] = x[mask]
    rgb[mask, 2] = 0
    
    # 60-120度: R=X, G=C, B=0
    mask = (h >= 60) & (h < 120)
    rgb[mask, 0] = x[mask]
    rgb[mask, 1] = c[mask]
    rgb[mask, 2] = 0
    
    # 120-180度: R=0, G=C, B=X
    mask = (h >= 120) & (h < 180)
    rgb[mask, 0] = 0
    rgb[mask, 1] = c[mask]
    rgb[mask, 2] = x[mask]
    
    # 180-240度: R=0, G=X, B=C
    mask = (h >= 180) & (h < 240)
    rgb[mask, 0] = 0
    rgb[mask, 1] = x[mask]
    rgb[mask, 2] = c[mask]
    
    # 240-300度: R=X, G=0, B=C
    mask = (h >= 240) & (h < 300)
    rgb[mask, 0] = x[mask]
    rgb[mask, 1] = 0
    rgb[mask, 2] = c[mask]
    
    # 300-360度: R=C, G=0, B=X
    mask = (h >= 300) & (h < 360)
    rgb[mask, 0] = c[mask]
    rgb[mask, 1] = 0
    rgb[mask, 2] = x[mask]
    
    # 加上m
    rgb[..., 0] += m
    rgb[..., 1] += m
    rgb[..., 2] += m
    
    # Clamp到0-1范围
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def get_color_mask(h: np.ndarray, color_name: str) -> np.ndarray:
    """
    获取指定颜色的mask
    
    Args:
        h: 色相数组，范围0-360
        color_name: 颜色名称 (red, yellow, green, cyan, blue)
    
    Returns:
        布尔mask数组
    """
    if color_name not in HSL_COLOR_RANGES:
        return np.zeros_like(h, dtype=bool)
    
    color_info = HSL_COLOR_RANGES[color_name]
    mask = np.zeros_like(h, dtype=bool)
    
    for hue_min, hue_max in color_info["hue_ranges"]:
        mask |= (h >= hue_min) & (h < hue_max)
    
    return mask


def get_color_weight(h: np.ndarray, color_name: str, falloff: float = 15.0) -> np.ndarray:
    """
    获取颜色权重，使用平滑过渡而非硬边界
    
    Args:
        h: 色相数组，范围0-360
        color_name: 颜色名称
        falloff: 衰减范围（度数），控制过渡的柔和程度
    
    Returns:
        权重数组，范围0-1
    """
    if color_name not in HSL_COLOR_RANGES:
        return np.zeros_like(h, dtype=np.float32)
    
    color_info = HSL_COLOR_RANGES[color_name]
    weight = np.zeros_like(h, dtype=np.float32)
    
    for hue_min, hue_max in color_info["hue_ranges"]:
        # 计算到范围边界的距离
        center = (hue_min + hue_max) / 2
        half_range = (hue_max - hue_min) / 2
        
        # 考虑色相环的循环特性
        dist = np.abs(h - center)
        dist = np.minimum(dist, 360 - dist)
        
        # 在范围内：权重为1
        # 在范围外：平滑衰减
        range_weight = np.zeros_like(h, dtype=np.float32)
        
        # 完全在范围内
        in_range = dist <= half_range
        range_weight[in_range] = 1.0
        
        # 在过渡区域
        transition = (dist > half_range) & (dist <= half_range + falloff)
        if np.any(transition):
            t = (dist[transition] - half_range) / falloff
            range_weight[transition] = 1.0 - t  # 线性衰减
        
        weight = np.maximum(weight, range_weight)
    
    return weight


def adjust_hsl_by_color(
    h: np.ndarray,
    s: np.ndarray,
    l: np.ndarray,
    color_name: str,
    hue_shift: float = 0.0,
    saturation_scale: float = 1.0,
    lightness_scale: float = 1.0,
    use_soft_mask: bool = True,
    falloff: float = 15.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    调整指定颜色的HSL值
    
    Args:
        h, s, l: HSL数组
        color_name: 颜色名称
        hue_shift: 色相偏移（-180到180度）
        saturation_scale: 饱和度缩放（0-2，1为不变）
        lightness_scale: 明度缩放（0-2，1为不变）
        use_soft_mask: 是否使用软边缘mask（更自然的过渡）
        falloff: 软边缘的衰减范围
    
    Returns:
        调整后的 (h, s, l)
    """
    h_out = h.copy()
    s_out = s.copy()
    l_out = l.copy()
    
    if use_soft_mask:
        weight = get_color_weight(h, color_name, falloff)
    else:
        weight = get_color_mask(h, color_name).astype(np.float32)
    
    # 只在有权重的地方进行调整
    if np.any(weight > 0):
        # 色相偏移
        if hue_shift != 0:
            h_shifted = (h + hue_shift * weight) % 360.0
            h_out = h_shifted
        
        # 饱和度缩放
        if saturation_scale != 1.0:
            # 使用插值而非直接乘法，实现更平滑的过渡
            s_scaled = s * saturation_scale
            s_out = s * (1 - weight) + s_scaled * weight
            s_out = np.clip(s_out, 0, 1)
        
        # 明度缩放
        if lightness_scale != 1.0:
            # 使用中性点0.5作为缩放中心，避免全黑或全白
            l_centered = l - 0.5
            l_scaled = l_centered * lightness_scale + 0.5
            l_out = l * (1 - weight) + l_scaled * weight
            l_out = np.clip(l_out, 0, 1)
    
    return h_out, s_out, l_out


class HSLColorAdjust:
    """
    HSL颜色分区调整节点
    
    可以分别调整红、黄、绿、青、蓝五种颜色的色相、饱和度和明度
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # 红色调整
                "red_hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "红色色相偏移（-180到180度）"}
                ),
                "red_saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "红色饱和度（0-2，1为原始值）"}
                ),
                "red_lightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "红色明度（0-2，1为原始值）"}
                ),
                
                # 黄色调整
                "yellow_hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "黄色色相偏移（-180到180度）"}
                ),
                "yellow_saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "黄色饱和度（0-2，1为原始值）"}
                ),
                "yellow_lightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "黄色明度（0-2，1为原始值）"}
                ),
                
                # 绿色调整
                "green_hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "绿色色相偏移（-180到180度）"}
                ),
                "green_saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "绿色饱和度（0-2，1为原始值）"}
                ),
                "green_lightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "绿色明度（0-2，1为原始值）"}
                ),
                
                # 青色调整
                "cyan_hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "青色色相偏移（-180到180度）"}
                ),
                "cyan_saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "青色饱和度（0-2，1为原始值）"}
                ),
                "cyan_lightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "青色明度（0-2，1为原始值）"}
                ),
                
                # 蓝色调整
                "blue_hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "蓝色色相偏移（-180到180度）"}
                ),
                "blue_saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "蓝色饱和度（0-2，1为原始值）"}
                ),
                "blue_lightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "蓝色明度（0-2，1为原始值）"}
                ),
                
                # 全局设置
                "soft_transition": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "启用软边缘过渡，使颜色调整更自然"}
                ),
                "transition_falloff": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.0, "max": 60.0, "step": 1.0,
                     "tooltip": "过渡区域的宽度（度数），值越大过渡越柔和"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "auto_chroma_bg"

    def process(
        self,
        image,
        # 红色
        red_hue: float = 0.0,
        red_saturation: float = 1.0,
        red_lightness: float = 1.0,
        # 黄色
        yellow_hue: float = 0.0,
        yellow_saturation: float = 1.0,
        yellow_lightness: float = 1.0,
        # 绿色
        green_hue: float = 0.0,
        green_saturation: float = 1.0,
        green_lightness: float = 1.0,
        # 青色
        cyan_hue: float = 0.0,
        cyan_saturation: float = 1.0,
        cyan_lightness: float = 1.0,
        # 蓝色
        blue_hue: float = 0.0,
        blue_saturation: float = 1.0,
        blue_lightness: float = 1.0,
        # 全局
        soft_transition: bool = True,
        transition_falloff: float = 15.0,
    ):
        # 转换输入
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        
        # 处理批量图像
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()  # [H, W, C]
            
            # RGB转HSL
            h, s, l = rgb_to_hsl(img)
            
            # 依次调整各颜色
            color_adjustments = [
                ("red", red_hue, red_saturation, red_lightness),
                ("yellow", yellow_hue, yellow_saturation, yellow_lightness),
                ("green", green_hue, green_saturation, green_lightness),
                ("cyan", cyan_hue, cyan_saturation, cyan_lightness),
                ("blue", blue_hue, blue_saturation, blue_lightness),
            ]
            
            for color_name, hue_shift, sat_scale, light_scale in color_adjustments:
                # 只有在参数不是默认值时才进行调整
                if hue_shift != 0.0 or sat_scale != 1.0 or light_scale != 1.0:
                    h, s, l = adjust_hsl_by_color(
                        h, s, l,
                        color_name,
                        hue_shift=hue_shift,
                        saturation_scale=sat_scale,
                        lightness_scale=light_scale,
                        use_soft_mask=soft_transition,
                        falloff=transition_falloff
                    )
            
            # HSL转回RGB
            rgb_out = hsl_to_rgb(h, s, l)
            results.append(rgb_out)
        
        # 组合结果
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "HSLColorAdjust": HSLColorAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSLColorAdjust": "HSL Color Adjust (HSL颜色分区调整)",
}

