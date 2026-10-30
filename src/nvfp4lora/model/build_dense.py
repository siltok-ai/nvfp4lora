"""Dense专用meta构建；CPU embedding和原字节层store不注册为GPU参数。"""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nvfp4lora.families.dense import normalize_build
from nvfp4lora.format.policies import NUMERICAL_POLICY as NONSTREAM_POLICY
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.format.preprocess_dense import FAMILY, LAYOUT, LAYER_LAYOUT, LINEAR_POLICY, projections
from nvfp4lora.ops.packed_linear_fn import PackedLinearOpConfig
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from nvfp4lora.model.common import BuiltModel, canonical_name
from .packed_linear import CPUEmbedding, ReferencePackedLinear, LoRAPackedLinear, PackedLinear

from .attention import ATTENTION_IMPLEMENTATION, register_dense_attention







def expected_dense_trainable(geometry, rank, scope):
    build = normalize_build({"lora_rank": rank, "lora_scope": scope})
    rank, scope = build["lora_rank"], build["lora_scope"]
    expected = {}
    if rank:
        for layer, modules in geometry["modules"].items():
            for name, spec in modules.items():
                if scope == "D-ATTN" and name.startswith("mlp."):
                    continue
                prefix = f"model.layers.{layer}.{name}"
                expected[prefix + ".lora_A.weight"] = [rank, spec["in_features"]]
                expected[prefix + ".lora_B.weight"] = [spec["out_features"], rank]
    return dict(sorted(expected.items()))


def build_dense_model(directory, *, device="cuda", lora_rank=16, lora_alpha=None, lora_scope="D-ALL",
                      decode_chunk_elements=8 * 1024 * 1024, streaming="async", family=FAMILY,
                      reference=False, store=None, streamer=None):
    """按manifest构建HF正文；真实权重只用当前层解码，不调用PEFT注入。"""
    directory, device = Path(directory), torch.device(device)
    build = normalize_build({"family": family, "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                             "lora_scope": lora_scope, "decode_chunk_elements": decode_chunk_elements, "streaming": streaming})
    lora_rank, lora_alpha, lora_scope = (build[key] for key in ["lora_rank", "lora_alpha", "lora_scope"])
    decode_chunk_elements, streaming = (build[key] for key in ["decode_chunk_elements", "streaming"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("family") != FAMILY
            or manifest.get("layout_version") != LAYOUT or manifest.get("linear_numerical_policy") != LINEAR_POLICY
            or manifest.get("nonstream_numerical_policy") != NONSTREAM_POLICY):
        raise ValueError("Unsupported or incomplete dense bundle")
    if reference and manifest["source"].get("kind") != "tiny-generated":
        raise ValueError("Dense materialized reference is restricted to tiny-generated sources")
    if file_hash(directory / "config.json") != manifest["output_files_sha256"]["config.json"]:
        raise ValueError("Dense config checksum mismatch")
    config_data = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    geometry = manifest["geometry"]
    expected = expected_dense_trainable(geometry, lora_rank, lora_scope)
    if geometry["layers"] != config_data["num_hidden_layers"]:
        raise ValueError("Dense layer count/config mismatch")
    for layer in range(geometry["layers"]):
        if geometry["modules"][str(layer)] != projections(config_data, layer):
            raise ValueError(f"Dense module geometry/config mismatch: {layer}")
    config = Qwen3_5TextConfig.from_dict(config_data)
    if config.tie_word_embeddings or getattr(config, "num_experts", 0):
        raise ValueError("Dense build requires untied embeddings/head and no experts")
    register_dense_attention()
    config.use_cache, config._attn_implementation = False, ATTENTION_IMPLEMENTATION
    if store is None:
        store = PackedLayerStore(directory / "layers", pin_memory=device.type == "cuda", layout_version=LAYER_LAYOUT)
    if store.source != manifest["source"] or store.geometry != geometry:
        raise ValueError("Dense store and bundle identities differ")
    if streamer is None:
        if streaming == "async":
            from nvfp4lora.store.async_streamer import AsyncLayerStreamer
            streamer = AsyncLayerStreamer(store, device=device, slots=2)
        else:
            streamer = SyncLayerStreamer(store, device=device, slots=2)
    if streamer.store is not store or streamer.device != device:
        raise ValueError("Dense streamer store/device mismatch")
    op_config = PackedLinearOpConfig(streamer, decode_chunk_elements=decode_chunk_elements)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    model.to(dtype=torch.bfloat16)
    model.model.embed_tokens = CPUEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
    linear_class = ReferencePackedLinear if reference else PackedLinear
    for layer, block in enumerate(model.model.layers):
        for name, spec in geometry["modules"][str(layer)].items():
            original = block.get_submodule(name)
            if not isinstance(original, torch.nn.Linear) or (original.in_features, original.out_features) != (spec["in_features"], spec["out_features"]):
                raise ValueError(f"HF dense projection differs: {layer}.{name}")
            linear = linear_class(layer, name, spec["kind"], spec["in_features"], spec["out_features"], op_config)
            if f"model.layers.{layer}.{name}.lora_A.weight" in expected:
                linear = LoRAPackedLinear(linear, lora_rank, lora_alpha, device="meta")
            parent, _, leaf = name.rpartition(".")
            setattr(block.get_submodule(parent), leaf, linear)
    original_buffers = set(dict(model.named_buffers()))
    base_parameters = {name: parameter for name, parameter in model.named_parameters() if ".lora_" not in name}
    expected_nonstream = set(base_parameters) | {"model.embed_tokens.weight"}
    if expected_nonstream != set(geometry["nonstream_parameters"]) | {"lm_head.weight"}:
        raise ValueError("Dense nonstream metadata differs from HF parameter set")
    model.to_empty(device=device)
    targets, loaded = dict(model.named_parameters()), set()
    for entry in manifest["nonstream_files"]:
        path = directory / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Dense nonstream checksum mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as reader, torch.no_grad():
            if sorted(reader.keys()) != entry["keys"]:
                raise ValueError(f"Dense nonstream file key set mismatch: {path}")
            for name in reader.keys():
                value = reader.get_tensor(name)
                if name in loaded or name not in expected_nonstream or value.dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected dense nonstream key/dtype: {name}")
                if name == "model.embed_tokens.weight":
                    model.model.embed_tokens.set_table(value)
                else:
                    if value.shape != targets[name].shape or targets[name].dtype != torch.bfloat16:
                        raise ValueError(f"Dense nonstream shape/dtype mismatch: {name}")
                    targets[name].copy_(value)
                loaded.add(name)
                del value
    if loaded != expected_nonstream:
        raise ValueError(f"Dense nonstream load incomplete: {expected_nonstream - loaded}")
    rotary = model.model.rotary_emb
    with torch.device("cpu"):
        fresh = type(rotary)(config)
    initialized = []
    for name, value in fresh.named_buffers():
        rotary.register_buffer(name, value.to(device=device, dtype=torch.float32),
                               persistent=name not in fresh._non_persistent_buffers_set)
        initialized.append("model.rotary_emb." + name)
    if set(initialized) != original_buffers:
        raise ValueError("Dense uninitialized or unexpected buffers")
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRAPackedLinear):
            module.reset_parameters()
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        if isinstance(module, ReferencePackedLinear):
            module.materialize_reference()
    trainable = {}
    for name, parameter in model.named_parameters():
        if parameter.is_meta or parameter.device.type != device.type:
            raise ValueError(f"Dense unallocated/misplaced parameter: {name}")
        if parameter.requires_grad:
            key = canonical_name(name)
            if key in trainable or parameter.dtype != torch.float32:
                raise ValueError(f"Dense duplicate or non-FP32 trainable: {key}")
            trainable[key] = parameter
    if {name: list(value.shape) for name, value in trainable.items()} != expected:
        raise ValueError("Dense trainable set differs from expected set")
    if any(value.is_meta or value.dtype != torch.float32 for value in model.buffers()):
        raise ValueError("Dense RoPE buffers must be allocated FP32")
    return BuiltModel(model, store, streamer, dict(sorted(trainable.items())), expected, sorted(loaded), sorted(initialized), family=FAMILY, frozen_extra={"embedding_cpu": model.model.embed_tokens.table})
