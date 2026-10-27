"""NVFP4 专家的一阶自定义 autograd；显式保留路由权重梯度。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from nvfp4lora.format.nvfp4 import dequantize


@dataclass(frozen=True)
class ExpertOpConfig:
    """一次前后向使用不可变的数值配置，streamer 只保存冻结原字节槽。"""

    streamer: object
    chunk_size: int = 32
    compute_dtype: torch.dtype = torch.bfloat16
    scaling: tuple = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if self.chunk_size < 1 or self.compute_dtype not in [torch.bfloat16, torch.float32, torch.float64]:
            raise ValueError("Expected positive chunk_size and BF16/FP32/FP64 compute_dtype")
        if len(self.scaling) != 3:
            raise ValueError("Gate/up/down scaling must be specified separately")

    @property
    def accumulation_dtype(self):
        return torch.float64 if self.compute_dtype == torch.float64 else torch.float32


def _route_plan(x, indices, weights, cfg):
    """稳定排序保留每个 token/top-k 槽位；计数只在这里同步一次。"""
    if x.ndim != 2 or indices.ndim != 2 or indices.shape != weights.shape or indices.shape[0] != x.shape[0]:
        raise ValueError("Expected x[T,H] and equal indices/weights[T,K]")
    if x.shape[0] == 0 or indices.shape[1] == 0 or indices.dtype != torch.int64:
        raise ValueError("Nonempty routes with int64 expert indices required")
    if x.dtype != cfg.compute_dtype or not weights.is_floating_point():
        raise ValueError("Input dtype must match explicit compute policy and routing weights must be floating")
    experts = cfg.streamer.store.geometry["experts"]
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=experts).tolist()
    if len(counts) != experts:
        raise ValueError("Expert index exceeds store geometry")
    order = torch.argsort(flat, stable=True)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return order, counts, offsets


def _decode_block(packed, begin, end, cfg):
    """只生成本块冻结浮点权重；global 留在 FP32/FP64 输出端施加。"""
    matrices, scales = [], []
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        matrices.append(dequantize(packed[projection + ".weight"][begin:end],
                                   packed[projection + ".weight_scale"][begin:end],
                                   dtype=cfg.compute_dtype))
        scales.append(packed[projection + ".weight_scale_2"][begin:end].to(cfg.accumulation_dtype))
    return matrices, scales


def _chunk_forward(x, weights, matrices, scales, parameters, counts, cfg):
    """前向及块内重算共用精度策略；gate/up 增量参与 SwiGLU，down 增量参与路由加权。"""
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = x[start:start + count]
        values = rows.to(cfg.accumulation_dtype)
        gate = F.linear(rows, matrices[0][expert]).to(cfg.accumulation_dtype) * scales[0][expert]
        up = F.linear(rows, matrices[1][expert]).to(cfg.accumulation_dtype) * scales[1][expert]
        if parameters:
            gate = gate + F.linear(F.linear(values, parameters[0][expert]), parameters[1][expert]) * cfg.scaling[0]
            up = up + F.linear(F.linear(values, parameters[2][expert]), parameters[3][expert]) * cfg.scaling[1]
        hidden = F.silu(gate) * up
        down = F.linear(hidden.to(cfg.compute_dtype), matrices[2][expert]).to(cfg.accumulation_dtype) * scales[2][expert]
        if parameters:
            down = down + F.linear(F.linear(hidden, parameters[4][expert]), parameters[5][expert]) * cfg.scaling[2]
        outputs.append(down * weights[start:start + count, None].to(cfg.accumulation_dtype))
        start += count
    return torch.cat(outputs, dim=0)


def _forward_block(x, weights, packed, parameters, counts, begin, end, cfg):
    """函数作用域限定 decoded W 的生命周期，返回值不携带任何浮点权重。"""
    matrices, scales = _decode_block(packed, begin, end, cfg)
    local = tuple(p[begin:end] for p in parameters)
    return _chunk_forward(x, weights, matrices, scales, local, counts[begin:end], cfg)


def _backward_block(x, weights, packed, parameters, counts, begin, end, dy, cfg):
    """局部叶子只覆盖本块；autograd.grad 后不保留图，不直接修改 Parameter.grad。"""
    matrices, scales = _decode_block(packed, begin, end, cfg)
    with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        local_x = x.detach().requires_grad_()
        local_w = weights.detach().requires_grad_()
        local = tuple(p[begin:end].detach().requires_grad_() for p in parameters)
        output = _chunk_forward(local_x, local_w, matrices, scales, local, counts[begin:end], cfg)
        return torch.autograd.grad(output, (local_x, local_w, *local), grad_outputs=dy.to(cfg.accumulation_dtype),
                                   allow_unused=False, retain_graph=False, create_graph=False)


def _prefetch_hint(cfg, layer):
    """预测只影响搬运；实际 acquire 始终使用本次明确层号。"""
    direction = -1 if cfg.streamer.phase == "BACKWARD" else 1
    predicted = layer + direction
    if 0 <= predicted < cfg.streamer.store.geometry["layers"]:
        cfg.streamer.prefetch(predicted)


def _forward(x, indices, weights, parameters, layer, cfg):
    """每条 route 写唯一位置，再按原 top-k 顺序确定性求和。"""
    order, counts, offsets = _route_plan(x, indices, weights, cfg)
    routes, width = indices.numel(), x.shape[-1]
    output = torch.empty(routes, width, dtype=cfg.accumulation_dtype, device=x.device)
    k = indices.shape[1]
    with torch.autocast(device_type=x.device.type, enabled=False), cfg.streamer.acquire(layer) as handle:
        packed = handle.tensors()
        _prefetch_hint(cfg, layer)
        for begin in range(0, len(counts), cfg.chunk_size):
            end = min(begin + cfg.chunk_size, len(counts))
            first, last = offsets[begin], offsets[end]
            if first == last:
                continue
            route_ids = order[first:last]
            block = _forward_block(x[route_ids // k], weights.reshape(-1)[route_ids], packed,
                                   parameters, counts, begin, end, cfg)
            output[route_ids] = block
            del block
    return output.reshape(x.shape[0], k, width).sum(dim=1).to(x.dtype)


def _backward(x, indices, weights, parameters, layer, cfg, grad_output):
    """每条 route 的 dX 先放回原位置，再以 FP32/FP64 按 top-k 汇总。"""
    order, counts, offsets = _route_plan(x, indices, weights, cfg)
    k = indices.shape[1]
    dx_routes = torch.empty(indices.numel(), x.shape[-1], dtype=cfg.accumulation_dtype, device=x.device)
    dw = torch.empty(indices.numel(), dtype=weights.dtype, device=weights.device)
    dp = tuple(torch.zeros_like(p) for p in parameters)
    with torch.autocast(device_type=x.device.type, enabled=False), cfg.streamer.acquire(layer) as handle:
        packed = handle.tensors()
        _prefetch_hint(cfg, layer)
        for begin in range(0, len(counts), cfg.chunk_size):
            end = min(begin + cfg.chunk_size, len(counts))
            first, last = offsets[begin], offsets[end]
            if first == last:
                continue
            route_ids = order[first:last]
            gradients = _backward_block(x[route_ids // k], weights.reshape(-1)[route_ids], packed,
                                        parameters, counts, begin, end, grad_output[route_ids // k], cfg)
            dx_routes[route_ids] = gradients[0].to(cfg.accumulation_dtype)
            dw[route_ids] = gradients[1]
            for destination, gradient in zip(dp, gradients[2:]):
                destination[begin:end].copy_(gradient)
            del gradients
    dx = dx_routes.reshape(x.shape[0], k, x.shape[-1]).sum(dim=1).to(x.dtype)
    return dx, dw.reshape(weights.shape), dp


class NVFP4ExpertsFn(torch.autograd.Function):
    """六个 A/B 作为顶层 Tensor 参数；只支持一阶训练梯度。"""

    @staticmethod
    def forward(ctx, x, top_k_index, top_k_weights, Ag, Bg, Au, Bu, Ad, Bd, layer_id, cfg):
        parameters = (Ag, Bg, Au, Bu, Ad, Bd)
        ctx.save_for_backward(x, top_k_index, top_k_weights, *parameters)
        ctx.layer, ctx.cfg = layer_id, cfg
        return _forward(x, top_k_index, top_k_weights, parameters, layer_id, cfg)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, indices, weights, *parameters = ctx.saved_tensors
        dx, dw, dp = _backward(x, indices, weights, tuple(parameters), ctx.layer, ctx.cfg, grad_output)
        result = (dx, None, dw, *dp, None, None)
        return tuple(value if needed else None for value, needed in zip(result, ctx.needs_input_grad))


class NVFP4ExpertsBaseFn(torch.autograd.Function):
    """C3 阶段冻结专家的输入/路由反向，同样在反向重新解码。"""

    @staticmethod
    def forward(ctx, x, top_k_index, top_k_weights, layer_id, cfg):
        ctx.save_for_backward(x, top_k_index, top_k_weights)
        ctx.layer, ctx.cfg = layer_id, cfg
        return _forward(x, top_k_index, top_k_weights, (), layer_id, cfg)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, indices, weights = ctx.saved_tensors
        dx, dw, _ = _backward(x, indices, weights, (), ctx.layer, ctx.cfg, grad_output)
        result = (dx, None, dw, None, None)
        return tuple(value if needed else None for value, needed in zip(result, ctx.needs_input_grad))
