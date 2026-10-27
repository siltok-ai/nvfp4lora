"""行主序、K 方向低 nibble 在前、每 16 元素 E4M3 缩放的 NVFP4。"""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize(packed, block_scale, global_scale=None, *, dtype=torch.float32):
    """解码一组矩阵；省略 global_scale 时返回可精确表示为 BF16 的基座码值。"""
    if packed.dtype != torch.uint8 or packed.shape[-1] % 8:
        raise ValueError("NVFP4 weight must be uint8 with packed K divisible by 8")
    if block_scale.shape != (*packed.shape[:-1], packed.shape[-1] // 8):
        raise ValueError("NVFP4 scale shape must equal [..., OUT, K/16]")
    if block_scale.dtype not in [torch.uint8, torch.float8_e4m3fn]:
        raise ValueError("NVFP4 block scales must preserve raw E4M3FN bytes")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1)
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    values = lut[codes.long()]
    scale = block_scale.view(torch.float8_e4m3fn).float()
    result = (values.reshape(*scale.shape, 16) * scale.unsqueeze(-1)).reshape(values.shape)
    if global_scale is not None:
        # checkpoint 每个矩阵独立一个 FP32 标量，批次维度为任意前导维。
        result = result * global_scale.float().reshape(*packed.shape[:-2], 1, 1)
    return result.to(dtype)


def quantize(weight, global_scale=None):
    """生成 tiny 格式夹具；真实 checkpoint 专家绝不调用本函数重新量化。"""
    if weight.ndim < 2 or weight.shape[-1] % 16:
        raise ValueError("NVFP4 input must be [..., OUT, K], K divisible by 16")
    value = weight.float()
    if not torch.isfinite(value).all():
        raise ValueError("Cannot quantize non-finite weights")
    if global_scale is None:
        global_scale = value.abs().amax(dim=(-2, -1)) / (6 * 448)
        global_scale = torch.where(global_scale == 0, torch.ones_like(global_scale), global_scale)
    global_scale = global_scale.to(device=weight.device, dtype=torch.float32).reshape(weight.shape[:-2])
    if not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
        raise ValueError("NVFP4 global scales must be finite and positive")
    grouped = value.reshape(*value.shape[:-1], value.shape[-1] // 16, 16)
    block_scale = (grouped.abs().amax(-1) / (6 * global_scale[..., None, None])).clamp(max=448)
    block_scale = block_scale.to(torch.float8_e4m3fn)
    divisor = block_scale.float() * global_scale[..., None, None]
    normalized = grouped / torch.where(divisor == 0, torch.ones_like(divisor), divisor).unsqueeze(-1)
    magnitude = normalized.abs().contiguous()
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=value.device)
    code = torch.bucketize(magnitude, boundaries)
    # E2M1 的中点按码值最低位执行 ties-to-even，而非一律向下取。
    at_odd_tie = (code % 2 == 1) & (magnitude == boundaries[code.clamp(max=6)])
    code = code + at_odd_tie
    code = (code.to(torch.uint8) | (torch.signbit(grouped).to(torch.uint8) << 3)).reshape(value.shape)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return {"weight": packed.contiguous(), "weight_scale": block_scale.view(torch.uint8).contiguous(),
            "weight_scale_2": global_scale.contiguous(),
            "input_scale": torch.ones_like(global_scale)}
