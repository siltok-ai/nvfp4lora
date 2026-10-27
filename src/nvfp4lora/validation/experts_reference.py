"""不复用生产 route plan、chunk_forward 或 VJP 的稠密 autograd 参考。"""

import torch


def _expert_formula(rows, matrices, scales, parameters, scaling, compute_dtype):
    """独立按三投影的数学公式书写；小形状还用逐 token/槽调用消除排序依赖。"""
    accum = torch.float64 if compute_dtype == torch.float64 else torch.float32
    base_gate = (rows @ matrices[0].T).to(accum) * scales[0].to(accum)
    base_up = (rows @ matrices[1].T).to(accum) * scales[1].to(accum)
    if parameters:
        gate_delta = (rows.to(accum) @ parameters[0].T) @ parameters[1].T
        up_delta = (rows.to(accum) @ parameters[2].T) @ parameters[3].T
        gate, up = base_gate + scaling[0] * gate_delta, base_up + scaling[1] * up_delta
    else:
        gate, up = base_gate, base_up
    intermediate = torch.nn.functional.silu(gate) * up
    base_down = (intermediate.to(compute_dtype) @ matrices[2].T).to(accum) * scales[2].to(accum)
    if not parameters:
        return base_down
    return base_down + scaling[2] * ((intermediate @ parameters[4].T) @ parameters[5].T)


def experts_reference(x, indices, weights, matrices, scales, parameters=(), *, scaling=(1., 1., 1.), token_loop=False):
    """稠密冻结 W 的参考；token_loop 不排序，目标形状则逐专家取实际 token/slot。"""
    dtype = x.dtype
    accum = torch.float64 if dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        if token_loop:
            outputs = []
            for token in range(x.shape[0]):
                routes = []
                for slot in range(indices.shape[1]):
                    expert = int(indices[token, slot])
                    value = _expert_formula(x[token], [w[expert] for w in matrices], [s[expert] for s in scales],
                                            [p[expert] for p in parameters], scaling, dtype)
                    routes.append(value * weights[token, slot].to(accum))
                outputs.append(torch.stack(routes).sum(dim=0))
            return torch.stack(outputs).to(dtype)
        routes = torch.zeros(*indices.shape, x.shape[-1], dtype=accum, device=x.device)
        for expert in range(matrices[0].shape[0]):
            token, slot = torch.where(indices == expert)
            if token.numel() == 0:
                continue
            value = _expert_formula(x[token], [w[expert] for w in matrices], [s[expert] for s in scales],
                                    [p[expert] for p in parameters], scaling, dtype)
            routes[token, slot] = value * weights[token, slot, None].to(accum)
        return routes.sum(dim=1).to(dtype)
