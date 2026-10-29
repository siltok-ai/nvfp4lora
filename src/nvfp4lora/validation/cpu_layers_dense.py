"""Dense独立CPU FP32逐层参考：原字节解码、eager attention、单token GDN。"""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.format.preprocess_dense import FAMILY, LAYOUT
from .cpu_layers import CPURecurrentGDN


@torch.no_grad()
def load_cpu_block(bundle, store, layer, config, manifest, *, recurrent_gdn=True):
    """只分配当前层FP32，不调用生产decode_codes或PackedLinear。"""
    with torch.device("meta"):
        block = Qwen3_5DecoderLayer(config, layer)
    block.to(dtype=torch.float32).to_empty(device="cpu")
    targets, loaded = dict(block.named_parameters()), set()
    prefix = f"model.layers.{layer}."
    for entry in manifest["nonstream_files"]:
        if not any(name.startswith(prefix) for name in entry["keys"]):
            continue
        path = Path(bundle) / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"CPU dense reference nonstream checksum mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith(prefix):
                    key = name.removeprefix(prefix)
                    if key not in targets or key in loaded:
                        raise ValueError(f"Unexpected CPU dense parameter: {name}")
                    targets[key].copy_(reader.get_tensor(name))
                    loaded.add(key)
    raw = store.tensors(layer)
    for name, spec in manifest["geometry"]["modules"][str(layer)].items():
        key = name + ".weight"
        weight = targets[key]
        rows = max(1, (8 * 1024 * 1024) // spec["in_features"])
        for start in range(0, spec["out_features"], rows):
            stop = min(start + rows, spec["out_features"])
            if spec["kind"] == "nvfp4":
                decoded = dequantize(raw[key][start:stop], raw[name + ".weight_scale"][start:stop],
                                    raw[name + ".weight_scale_2"], dtype=torch.float32)
            else:
                decoded = raw[key][start:stop].view(torch.float8_e4m3fn).float() * raw[name + ".weight_scale"]
            weight[start:stop].copy_(decoded)
            del decoded
        if key in loaded:
            raise ValueError(f"Duplicate CPU dense weight: {key}")
        loaded.add(key)
    if set(targets) != loaded or any(value.is_meta or value.dtype != torch.float32 for value in block.parameters()):
        raise ValueError(f"CPU dense layer not fully initialized: {layer}")
    if recurrent_gdn and hasattr(block, "linear_attn"):
        # Shared independent token recurrence has no routing or MoE-specific math.
        block.linear_attn = CPURecurrentGDN(block.linear_attn)
    return block.eval().requires_grad_(False)


@torch.no_grad()
def reference_layers_dense(bundle, store, ids, *, layers=4, on_layer=None, recurrent_gdn=True):
    """从磁盘embedding开始无条件计算；每层结束释放全部FP32矩阵。"""
    bundle = Path(bundle)
    if ids.device.type != "cpu" or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Dense CPU reference expects one unpadded CPU token sequence")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["family"] != FAMILY or manifest["layout_version"] != LAYOUT or manifest["status"] != "PASS":
        raise ValueError("Dense CPU reference requires a complete dense bundle")
    if store.source != manifest["source"] or not 0 < layers <= manifest["geometry"]["layers"]:
        raise ValueError("Dense CPU reference store/layer identity mismatch")
    config = Qwen3_5TextConfig.from_dict(json.loads((bundle / "config.json").read_text(encoding="utf-8")))
    config.use_cache, config._attn_implementation = False, "eager"
    embedding_entry, = [entry for entry in manifest["nonstream_files"] if "model.embed_tokens.weight" in entry["keys"]]
    path = bundle / embedding_entry["file"]
    if file_hash(path) != embedding_entry["sha256"]:
        raise ValueError("CPU dense embedding checksum mismatch")
    with safe_open(path, framework="pt", device="cpu") as reader:
        hidden = F.embedding(ids, reader.get_tensor("model.embed_tokens.weight")).float()
    positions = torch.arange(ids.shape[-1]).view(1, -1)
    with torch.device("cpu"):
        rotary = Qwen3_5TextRotaryEmbedding(config)
    rope = rotary(hidden, positions[None].expand(3, 1, -1))
    mask = torch.full((ids.shape[-1], ids.shape[-1]), -torch.inf).triu(1)[None, None]
    outputs = []
    for layer in range(layers):
        block = load_cpu_block(bundle, store, layer, config, manifest, recurrent_gdn=recurrent_gdn)
        hidden = block(hidden, position_embeddings=rope, position_ids=positions,
                       attention_mask=mask if block.block_type == "full_attention" else None,
                       use_cache=False)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError(f"Nonfinite CPU dense layer: {layer}")
        outputs.append(hidden.clone())
        if on_layer:
            on_layer(layer, hidden)
        del block
        gc.collect()
    return outputs
