"""CPU 单 token GDN 递推参考，不使用分块训练实现。"""

import torch
from torch import nn
import torch.nn.functional as F

class CPURecurrentGDN(nn.Module):
    """非缓存参考：直接因果卷积＋FP32 单 token 状态更新，不调用 chunk GDN。"""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None or attention_mask is not None:
            raise ValueError("CPU reference supports unpadded, uncached text only")
        module = self.original
        batch, length, _ = hidden_states.shape
        mixed = module.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed = F.conv1d(mixed, module.conv1d.weight, module.conv1d.bias,
                         padding=module.conv_kernel_size - 1, groups=mixed.shape[1])[..., :length]
        mixed = F.silu(mixed).transpose(1, 2)
        query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
        query = query.reshape(batch, length, module.num_k_heads, module.head_k_dim)
        key = key.reshape(batch, length, module.num_k_heads, module.head_k_dim)
        value = value.reshape(batch, length, module.num_v_heads, module.head_v_dim)
        beta = module.in_proj_b(hidden_states).sigmoid()
        g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden_states) + module.dt_bias)
        repeat = module.num_v_heads // module.num_k_heads
        query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        state = torch.zeros(batch, module.num_v_heads, module.head_k_dim, module.head_v_dim)
        outputs = []
        for index in range(length):
            decayed = state * g[:, index].exp()[..., None, None]
            prediction = torch.einsum("bhk,bhkv->bhv", key[:, index], decayed)
            delta = (value[:, index] - prediction) * beta[:, index, :, None]
            state = decayed + torch.einsum("bhk,bhv->bhkv", key[:, index], delta)
            outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, index] / module.head_k_dim ** .5, state))
        output = torch.stack(outputs, dim=1)
        z = module.in_proj_z(hidden_states).reshape(-1, module.head_v_dim)
        output = module.norm(output.reshape(-1, module.head_v_dim), z)
        return module.out_proj(output.reshape(batch, length, -1))
