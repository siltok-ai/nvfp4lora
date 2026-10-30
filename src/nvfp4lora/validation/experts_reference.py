"""不复用生产 route plan、chunk_forward 或 VJP 的稠密 autograd 参考。"""

import torch
import torch.nn.functional as F


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


def materialized_experts_autograd(x, indices, weights, matrices, scales, parameters=(), *, scaling=(1.,1.,1.), chunk_size=32):
    """物化权重的普通 autograd；每条 route 独立输入，最后以累积精度合并输入梯度。"""
    accum = torch.float64 if x.dtype == torch.float64 else torch.float32
    tokens, width = x.shape
    k = indices.shape[1]
    # expand's backward reduces the K route gradients in FP32/FP64, as the streaming operator does.
    route_inputs = x.to(accum)[:, None, :].expand(tokens,k,width).clone().reshape(-1,width)
    # split's backward concatenates chunk gradients, preserving their signed-zero bits.
    # Selecting experts from the whole parameter instead would add zero-padded gradients across chunks.
    parameter_chunks = [parameter.split(chunk_size, dim=0) for parameter in parameters]
    output = torch.empty(tokens*k,width,dtype=accum,device=x.device)
    for expert in range(matrices[0].shape[0]):
        route_ids = torch.where(indices.reshape(-1)==expert)[0]
        if route_ids.numel()==0:continue
        rows = route_inputs[route_ids].to(x.dtype)
        local = [chunks[expert // chunk_size][expert % chunk_size] for chunks in parameter_chunks]
        values = rows.to(accum)
        gate = F.linear(rows,matrices[0][expert]).to(accum)*scales[0][expert]
        up = F.linear(rows,matrices[1][expert]).to(accum)*scales[1][expert]
        if parameters:
            gate = gate+F.linear(F.linear(values,local[0]),local[1])*scaling[0]
            up = up+F.linear(F.linear(values,local[2]),local[3])*scaling[1]
        hidden = F.silu(gate)*up
        down = F.linear(hidden.to(x.dtype),matrices[2][expert]).to(accum)*scales[2][expert]
        if parameters:
            down = down+F.linear(F.linear(hidden,local[4]),local[5])*scaling[2]
        output[route_ids]=down*weights.reshape(-1)[route_ids,None].to(accum)
    return output.reshape(tokens,k,width).sum(dim=1).to(x.dtype)

def materialized_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """仅供已物化 tiny 模型的普通 autograd 参考，不调用流式生产算子。"""
    names = ["gate_proj", "up_proj", "down_proj"]
    return materialized_experts_autograd(hidden_states, top_k_index, top_k_weights,
        [getattr(module, name + "_reference_weight") for name in names],
        [getattr(module, name + "_reference_scale") for name in names],
        module.adapter_parameters() if module.enabled else (), scaling=module.op_config.scaling,
        chunk_size=module.op_config.chunk_size)
