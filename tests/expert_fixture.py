"""小形状算子夹具：真实 safetensors/ExpertStore/streamer，随机非零 A/B。"""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from nvfp4lora.format.inspect import read_header
from nvfp4lora.format.nvfp4 import dequantize, quantize
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.store.expert_store import ExpertStore
from nvfp4lora.store.streamer import SyncLayerStreamer


def make_operator_store(directory, *, experts=8, hidden=64, intermediate=32, device="cpu"):
    """只为数学夹具生成 NVFP4，不作用于任何真实 checkpoint。"""
    directory = Path(directory)
    values = {}
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        out_dim, in_dim = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
        part = quantize(torch.randn(experts, out_dim, in_dim) / math.sqrt(in_dim))
        values.update({projection + "." + name: value for name, value in part.items()})
    file = directory / "layer-00.safetensors"
    save_file(values, file)
    header = read_header(file)
    geometry = {"layers": 1, "experts": experts, "hidden": hidden, "intermediate": intermediate}
    entry = {"layer": 0, "file": file.name, "sha256": file_hash(file),
             **{k: header[k] for k in ["header_bytes", "payload_bytes", "tensors"]}}
    (directory / "index.json").write_text(json.dumps({"layout_version": "packed-nvfp4-experts-v1",
        "geometry": geometry, "source": {"kind": "operator-fixture"}, "layers": [entry]}))
    store = ExpertStore(directory, pin_memory=(device == "cuda"))
    return store, SyncLayerStreamer(store, device=device)


def dense_layer(store, layer, dtype, device="cpu"):
    """格式解码已有独立位型测试；算子参考只接收解码常量，完全独立计算梯度。"""
    tensors = store.tensors(layer)
    matrices, scales = [], []
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        matrices.append(dequantize(tensors[projection + ".weight"], tensors[projection + ".weight_scale"], dtype=dtype).to(device))
        accum = torch.float64 if dtype == torch.float64 else torch.float32
        scales.append(tensors[projection + ".weight_scale_2"].to(device=device, dtype=accum))
    return matrices, scales


def make_parameters(experts=8, hidden=64, intermediate=32, rank=2, *, dtype=torch.float64, device="cpu", zero_b=False):
    """各专家 A 的 fan-in 独立，避免对 [E,r,in] 直接套 Kaiming 的 fan-in 错误。"""
    values = []
    for in_dim, out_dim in [(hidden, intermediate), (hidden, intermediate), (intermediate, hidden)]:
        a = torch.empty(experts, rank, in_dim, dtype=dtype, device=device).uniform_(-1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        b = torch.zeros(experts, out_dim, rank, dtype=dtype, device=device) if zero_b else torch.randn(experts, out_dim, rank, dtype=dtype, device=device) * 0.05
        values.extend([a.requires_grad_(), b.requires_grad_()])
    return tuple(values)
