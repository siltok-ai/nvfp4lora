"""训练进程/发布重建共享的FP32 ΔW字节指纹；不写全量增量矩阵或合并基座。"""

import hashlib

import torch

from nvfp4lora.model.common import canonical_name
from nvfp4lora.families import family_for_built


def _hash(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


@torch.no_grad()
def delta_w_fingerprints(built, *, expert_chunk=8):
    """固定batch几何逐块计算scaling×B@A，便于同设备同路径E2重建。"""
    result = {"layout": "fp32-delta-w-fingerprints-v1", "expert_chunk": expert_chunk, "tensors": {}}
    for name, module in built.model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A") and "default" in module.lora_A:
            a, b, scale = module.lora_A["default"].weight, module.lora_B["default"].weight, module.scaling["default"]
            value = (b @ a) * scale
            result["tensors"][canonical_name(name) + ".delta"] = {"shape": list(value.shape), "scaling": scale, "sha256": _hash(value)}
        result["tensors"].update(family_for_built(built).extra_delta_w(name, module, expert_chunk))
    return result
