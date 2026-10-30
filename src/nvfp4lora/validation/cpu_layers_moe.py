"""P5 的 CPU FP32 流式参考：一次一层非专家、一次一个路由专家，GDN 用逐 token 递推。"""

import gc
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding,
)

from nvfp4lora.format.nvfp4 import dequantize

from .cpu_gdn import CPURecurrentGDN


class CPUStreamedExperts(nn.Module):
    """独立的 FP32 数学写法，不使用生产 chunk、route plan、自定义 backward 或 GPU。"""

    def __init__(self, store, layer, forced_routes=None):
        super().__init__()
        self.store, self.layer = store, layer
        self.forced_routes = forced_routes

    def forward(self, hidden, indices, weights):
        if hidden.device.type != "cpu" or hidden.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("CPU FP32 or explicit BF16 diagnostic input required")
        if self.forced_routes is not None:
            indices, weights = self.forced_routes["indices"], self.forced_routes["weights"]
        packed = self.store.tensors(self.layer)
        self.last_indices, self.last_weights = indices.clone(), weights.clone()
        routes = torch.zeros(*indices.shape, hidden.shape[-1], dtype=torch.float32)
        for expert in range(self.store.geometry["experts"]):
            tokens, slots = torch.where(indices == expert)
            if tokens.numel() == 0:
                continue
            matrices, scales = [], []
            for name in ["gate_proj", "up_proj", "down_proj"]:
                matrices.append(dequantize(packed[name + ".weight"][expert], packed[name + ".weight_scale"][expert], dtype=hidden.dtype))
                scales.append(packed[name + ".weight_scale_2"][expert])
            rows = hidden[tokens]
            gate = (rows @ matrices[0].T).float() * scales[0]
            up = (rows @ matrices[1].T).float() * scales[1]
            value = ((F.silu(gate) * up).to(hidden.dtype) @ matrices[2].T).float() * scales[2]
            routes[tokens, slots] = value * weights[tokens, slots, None].float()
            del matrices, scales
        return routes.sum(dim=1).to(hidden.dtype)




@torch.no_grad()
def reference_layers(bundle, store, ids, *, layers=4, on_layer=None, traces=None,
                     compute_dtype=torch.float32, teacher_inputs=None, forced_routes=None):
    """从独立磁盘 embedding 开始，返回每层原始残差输出；仅支持零/禁用 LoRA。"""
    bundle = Path(bundle)
    if ids.device.type != "cpu" or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Expected one unpadded CPU ID sequence")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config = Qwen3_5MoeTextConfig.from_dict(json.loads((bundle / "config.json").read_text(encoding="utf-8")))
    config._attn_implementation, config.use_cache = "eager", False
    embedding_file = None
    for entry in manifest["nonexpert_files"]:
        with safe_open(bundle / entry["file"], framework="pt", device="cpu") as reader:
            if "model.embed_tokens.weight" in reader.keys():
                embedding_file = entry["file"]
                hidden = F.embedding(ids, reader.get_tensor("model.embed_tokens.weight")).to(compute_dtype)
                break
    if embedding_file is None:
        raise ValueError("Missing embedding in preprocessed bundle")
    positions = torch.arange(ids.shape[-1]).view(1, -1)
    rotary_positions = positions[None].expand(3, 1, -1)
    rotary = Qwen3_5MoeTextRotaryEmbedding(config, device="cpu")
    rope = rotary(hidden, rotary_positions)
    mask = torch.full((ids.shape[-1], ids.shape[-1]), -torch.inf).triu(1)[None, None]
    outputs = []
    for index in range(layers):
        with torch.device("meta"):
            block = Qwen3_5MoeDecoderLayer(config, index)
        block.mlp.experts = CPUStreamedExperts(store, index, None if forced_routes is None else forced_routes[index])
        block.to(dtype=compute_dtype).to_empty(device="cpu")
        prefix = f"model.layers.{index}."
        expected, loaded = dict(block.named_parameters()), set()
        for entry in manifest["nonexpert_files"]:
            with safe_open(bundle / entry["file"], framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if name.startswith(prefix):
                        key = name.removeprefix(prefix)
                        if key not in expected or key in loaded:
                            raise ValueError(f"Unexpected CPU reference parameter: {name}")
                        expected[key].copy_(reader.get_tensor(name))
                        loaded.add(key)
        if loaded != expected.keys() or any(p.is_meta for p in block.parameters()):
            raise ValueError(f"CPU reference block not fully initialized: {index}")
        if hasattr(block, "linear_attn") and compute_dtype == torch.float32:
            block.linear_attn = CPURecurrentGDN(block.linear_attn)
        block.eval().requires_grad_(False)
        if teacher_inputs is not None and teacher_inputs[index] is not None:
            hidden = teacher_inputs[index].to(compute_dtype)
        router_trace = {}
        def capture_router(module, args, output):
            router_trace.update(inputs=args[0].clone(), logits=output[0].clone())
        hook = block.mlp.gate.register_forward_hook(capture_router) if traces is not None else None
        hidden = block(hidden, position_embeddings=rope, position_ids=positions,
                       attention_mask=mask if block.block_type == "full_attention" else None,
                       use_cache=False)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(f"Nonfinite CPU reference output: {index}")
        outputs.append(hidden.clone())
        if traces is not None:
            traces.append({"indices": block.mlp.experts.last_indices, "weights": block.mlp.experts.last_weights, **router_trace})
            hook.remove()
        if on_layer:
            on_layer(index, hidden)
        del block, expected
        gc.collect()
    return outputs
