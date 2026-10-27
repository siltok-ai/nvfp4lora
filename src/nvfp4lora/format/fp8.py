"""按 checkpoint 自己的缩放恢复 FP8 非专家参数。"""

import torch


def dequantize(weight, scale, *, dtype=torch.bfloat16):
    """采用 FP32 乘缩放、最后舍入的显式非专家精度策略。"""
    if weight.dtype not in [torch.float8_e4m3fn, torch.uint8]:
        raise ValueError("Expected E4M3FN FP8 weight")
    if scale.numel() != 1:
        raise ValueError("This checkpoint contract requires scalar FP8 weight_scale")
    return (weight.view(torch.float8_e4m3fn).float() * scale.float()).to(dtype)
