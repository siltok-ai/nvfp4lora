"""冻结dense线性层：只在当前调用解码，反向重新acquire及解码。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.store.streamer import prefetch_hint

from nvfp4lora.format.policies import LINEAR_POLICY as NUMERICAL_POLICY



@dataclass(frozen=True)
class PackedLinearOpConfig:
    streamer: object
    compute_dtype: torch.dtype = torch.bfloat16
    decode_chunk_elements: int = 8 * 1024 * 1024

    def __post_init__(self):
        if self.compute_dtype not in [torch.bfloat16, torch.float32, torch.float64]:
            raise ValueError("PackedLinear requires BF16/FP32/FP64 compute_dtype")
        if self.decode_chunk_elements < 16 or self.decode_chunk_elements % 16:
            raise ValueError("decode_chunk_elements must be positive and divisible by 16")

    @property
    def accumulation_dtype(self):
        return torch.float64 if self.compute_dtype == torch.float64 else torch.float32


def decode_codes(tensors, name, kind, cfg):
    """NVFP4按元素分块；FP8只转换精确码值，global保留在输出端。"""
    weight = tensors[name + ".weight"]
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("Expected raw U8 matrix in dense layer slot")
    if kind not in ["nvfp4", "fp8"]:
        raise ValueError("Unknown packed linear kind: " + str(kind))
    if (name + ".weight_scale_2" in tensors) != (kind == "nvfp4"):
        raise ValueError("Packed linear kind disagrees with scale fields")
    if kind == "fp8":
        return (weight.view(torch.float8_e4m3fn).to(cfg.compute_dtype),
                tensors[name + ".weight_scale"].reshape(()).to(cfg.accumulation_dtype))
    scale = tensors[name + ".weight_scale"]
    if scale.shape != (weight.shape[0], weight.shape[1] // 8):
        raise ValueError("Invalid dense NVFP4 block scale geometry")
    decoded = torch.empty(weight.shape[0], weight.shape[1] * 2, dtype=cfg.compute_dtype, device=weight.device)
    flat, packed, blocks = decoded.view(-1), weight.reshape(-1), scale.reshape(-1)
    for start in range(0, decoded.numel(), cfg.decode_chunk_elements):
        stop = min(start + cfg.decode_chunk_elements, decoded.numel())
        values = dequantize(packed[start // 2:stop // 2].reshape(1, -1),
                            blocks[start // 16:stop // 16].reshape(1, -1),
                            global_scale=None, dtype=cfg.compute_dtype)
        flat[start:stop].copy_(values.reshape(-1))
        del values
    return decoded, tensors[name + ".weight_scale_2"].reshape(()).to(cfg.accumulation_dtype)


class PackedLinearFn(torch.autograd.Function):
    """五个顶层参数；ctx只保留非Tensor元信息，不保存x或解码权重。"""

    @staticmethod
    def forward(ctx, x, layer_id, name, kind, cfg):
        if x.dtype != cfg.compute_dtype:
            raise ValueError("PackedLinear input dtype must equal its explicit compute policy")
        with torch.autocast(device_type=x.device.type, enabled=False), cfg.streamer.acquire(layer_id) as handle:
            weight, global_scale = decode_codes(handle.tensors(), name, kind, cfg)
            prefetch_hint(cfg.streamer, layer_id)
            output = (F.linear(x, weight).to(cfg.accumulation_dtype) * global_scale).to(x.dtype)
        ctx.layer, ctx.name, ctx.kind, ctx.cfg = layer_id, name, kind, cfg
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        cfg = ctx.cfg
        with torch.autocast(device_type=grad_y.device.type, enabled=False), cfg.streamer.acquire(ctx.layer) as handle:
            weight, global_scale = decode_codes(handle.tensors(), ctx.name, ctx.kind, cfg)
            prefetch_hint(cfg.streamer, ctx.layer)
            grad_linear = (grad_y.to(cfg.accumulation_dtype) * global_scale).to(cfg.compute_dtype)
            dx = grad_linear.matmul(weight)
        return dx, None, None, None, None
