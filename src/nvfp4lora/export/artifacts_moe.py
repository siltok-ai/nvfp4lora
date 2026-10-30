"""MoE 发布参数的规范名映射。"""

from pathlib import Path
from safetensors.torch import load_file
from nvfp4lora.model.common import canonical_name

def _artifact_tensors(directory):
    """标准 PEFT 键和专家键映射为同一规范集合，拒绝任何重复键。"""
    tensors, mapping = {}, {}
    for filename in ["adapter_model.safetensors", "expert_lora.safetensors"]:
        for key, value in load_file(Path(directory) / filename).items():
            canonical = canonical_name(key)
            if canonical in tensors:
                raise ValueError(f"Canonical publish name collision: {canonical}")
            tensors[canonical] = value
            mapping[canonical] = {"file": filename, "key": key}
    return tensors, mapping
