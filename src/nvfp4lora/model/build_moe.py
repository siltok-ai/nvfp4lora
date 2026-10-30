"""先 meta 替换专家再分配设备；逐个加载非专家并重新初始化非持久 RoPE。"""

from dataclasses import dataclass
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors import safe_open
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from nvfp4lora.format.bundle import file_hash
from nvfp4lora.ops.experts_fn import ExpertOpConfig
from nvfp4lora.store.layer_store import ExpertStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from .packed_experts import DenseReferenceExperts, ExpertAdapter, PackedExperts

from .common import BuiltModel, canonical_name, causal_model

C3_TARGET = r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))"






def build_model(directory, *, device="cuda", expert_rank=0, expert_alpha=None, c3_rank=16,
                c3_alpha=32, chunk_size=32, store=None, streamer=None, reference=False, streaming="sync"):
    """完整核心构建入口，不调用 from_pretrained 或分配真实全量 BF16 专家。"""
    directory = Path(directory)
    if streaming not in ["sync", "async"]:
        raise ValueError("Expected sync or async streaming")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise ValueError("Preprocessed bundle is not complete")
    config = Qwen3_5MoeTextConfig.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    config.use_cache, config.output_router_logits, config._attn_implementation = False, False, "sdpa"
    if config.tie_word_embeddings:
        raise ValueError("This fixed checkpoint contract requires untied embeddings/lm_head")
    if store is None:
        store = ExpertStore(directory / "experts", pin_memory=(torch.device(device).type == "cuda"))
    if streamer is None:
        if streaming == "async":
            from nvfp4lora.store.async_streamer import AsyncLayerStreamer
            streamer = AsyncLayerStreamer(store, device=device, slots=2)
        else:
            streamer = SyncLayerStreamer(store, device=device, slots=2)
    op_config = ExpertOpConfig(streamer, chunk_size=chunk_size)
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
    model.to(dtype=torch.bfloat16)
    experts_class = DenseReferenceExperts if reference else PackedExperts
    for layer, block in enumerate(model.model.layers):
        block.mlp.experts = experts_class(config, layer, op_config, rank=expert_rank, alpha=expert_alpha, device="meta")
    expected = {name: list(p.shape) for name, p in model.named_parameters() if ".mlp.experts." in name}
    base_parameters = {name: p for name, p in model.named_parameters() if ".mlp.experts." not in name}
    original_buffer_names = set(dict(model.named_buffers()))
    model.to_empty(device=device)
    targets = dict(model.named_parameters())
    loaded = set()
    for file in manifest["nonexpert_files"]:
        path = directory / file["file"]
        if file_hash(path) != file["sha256"]:
            raise ValueError(f"Nonexpert file checksum mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as reader, torch.no_grad():
            for name in reader.keys():
                value = reader.get_tensor(name)
                if name not in base_parameters or name in loaded or list(value.shape) != list(targets[name].shape):
                    raise ValueError(f"Unexpected or incompatible nonexpert parameter: {name}")
                if value.dtype != torch.bfloat16 or targets[name].dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected nonexpert dtype: {name}")
                targets[name].copy_(value)
                loaded.add(name)
                del value
    if loaded != set(base_parameters):
        raise ValueError(f"Nonexpert load incomplete: {set(base_parameters) - loaded}")
    # CPU 正常构造的 RoPE 是权威初始化；直接替换为 FP32，不能复制到已被 .to(BF16) 改型的旧 buffer。
    rotary = model.model.rotary_emb
    fresh = type(rotary)(config, device="cpu")
    initialized = []
    for name, value in fresh.named_buffers():
        rotary.register_buffer(name, value.to(device=device), persistent=(name not in fresh._non_persistent_buffers_set))
        initialized.append("model.rotary_emb." + name)
    if set(initialized) != original_buffer_names:
        raise ValueError(f"Uninitialized model buffers: {original_buffer_names - set(initialized)}")
    for module in model.modules():
        if isinstance(module, ExpertAdapter):
            module.reset_parameters()
        if isinstance(module, DenseReferenceExperts):
            module.materialize_reference(device=device)
    model.requires_grad_(False)
    if c3_rank:
        selected = [name for name, module in model.named_modules() if re.fullmatch(C3_TARGET, name)]
        for name in selected:
            linear = model.get_submodule(name)
            expected[name + ".lora_A.weight"] = [c3_rank, linear.in_features]
            expected[name + ".lora_B.weight"] = [linear.out_features, c3_rank]
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=c3_rank,
                               lora_alpha=c3_alpha, lora_dropout=0, target_modules=C3_TARGET))
    for module in model.modules():
        if isinstance(module, ExpertAdapter):
            module.requires_grad_(True)
    trainable = {}
    for name, parameter in model.named_parameters():
        if parameter.is_meta:
            raise ValueError(f"Parameter still on meta: {name}")
        if parameter.requires_grad:
            key = canonical_name(name)
            if key in trainable or parameter.dtype != torch.float32:
                raise ValueError(f"Duplicate canonical trainable key or non-FP32 adapter: {key}")
            trainable[key] = parameter
    shapes = {name: list(parameter.shape) for name, parameter in trainable.items()}
    if shapes != expected:
        raise ValueError(f"Trainable set mismatch: missing={expected.keys()-shapes.keys()}, extra={shapes.keys()-expected.keys()}")
    return BuiltModel(model, store, streamer, dict(sorted(trainable.items())), dict(sorted(expected.items())),
                      sorted(loaded), sorted(initialized), family="qwen3_5_moe")
