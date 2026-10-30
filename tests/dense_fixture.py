"""仅供算子/搬运资格使用的小型原字节层store。"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from nvfp4lora.format.inspect import read_header
from nvfp4lora.format.nvfp4 import quantize
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT
from nvfp4lora.store.layer_store import PackedLayerStore


def make_dense_store(directory, *, pin_memory=False, layers=2, in_features=32, out_features=24):
    """不同层使用不同global和字节；index与生产store走相同读写路径。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(173)
    entries = []
    for layer in range(layers):
        tensors = {}
        values = quantize(torch.randn(out_features, in_features, generator=generator) * .03,
                          global_scale=torch.tensor(.001237 * (layer + 1)))
        tensors.update({"mlp.gate_proj." + name: value.contiguous() for name, value in values.items()})
        tensors["self_attn.q_proj.weight"] = (torch.randn(out_features, in_features, generator=generator) * 3).to(torch.float8_e4m3fn).view(torch.uint8)
        tensors["self_attn.q_proj.weight_scale"] = torch.tensor(.03173 * (layer + 1))
        tensors["self_attn.q_proj.input_scale"] = torch.tensor(.01237)
        path = directory / f"layer-{layer:02d}.safetensors"
        save_file(tensors, path)
        header = read_header(path)
        entries.append({"layer": layer, "file": path.name, "sha256": file_hash(path),
                        **{name: value for name, value in header.items() if name not in ["file", "metadata"]}})
    (directory / "index.json").write_text(json.dumps({"layout_version": LAYER_LAYOUT,
        "source": {"kind": "tiny-generated"}, "geometry": {"layers": layers}, "layers": entries}), encoding="utf-8")
    return PackedLayerStore(directory, pin_memory=pin_memory, layout_version=LAYER_LAYOUT)


def materialized_codes(tensors, name, dtype):
    """独立于生产分块decode_codes，整张矩阵物化用于autograd参考。"""
    if name + ".weight_scale_2" in tensors:
        from nvfp4lora.format.nvfp4 import dequantize
        weight = dequantize(tensors[name + ".weight"], tensors[name + ".weight_scale"], global_scale=None, dtype=dtype)
        global_scale = tensors[name + ".weight_scale_2"]
    else:
        weight = tensors[name + ".weight"].view(torch.float8_e4m3fn).to(dtype)
        global_scale = tensors[name + ".weight_scale"]
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    return weight, global_scale.reshape(()).to(acc).clone()
