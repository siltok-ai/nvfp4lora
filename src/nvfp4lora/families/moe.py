"""MoE 的训练身份、路由观测和生成约定。"""

import importlib.metadata
import json
from pathlib import Path
import psutil
import torch
from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.train.checkpoint import configuration_hash
from nvfp4lora.train.loop import live_inventory, memory_observation


def logical_config(spec, bundle, dataset):
    """逻辑参数、数据、基座和版本参与恢复身份；输出/stop-at/保存与评测频率不参与。"""
    defaults = {"seed": 20261028, "max_steps": 100, "max_length": 2048,
                "learning_rate": 2e-4, "warmup_steps": 10, "max_grad_norm": 1.,
                "accumulation_steps": 8, "ce_chunk_size": 128, "gradient_checkpointing": True,
                "sampler_seed": 113, "rng_probe": False}
    build = {"expert_rank": 0, "c3_rank": 16, "c3_alpha": 32, "chunk_size": 32, "streaming": "async"}
    bound = {"layout", "base_fingerprint", "dataset", "versions", "optimizer", "precision"}
    if spec.keys() - defaults.keys() - {"build", "family"} - bound:
        raise ValueError(f"Unknown training settings: {spec.keys() - defaults.keys() - {'build', 'family'} - bound}")
    allowed_build = set(build) | {"expert_alpha"}
    if spec.get("build", {}).keys() - allowed_build:
        raise ValueError("Unsupported build setting")
    build.update(spec.get("build", {}))
    build.setdefault("expert_alpha", 2 * build["expert_rank"] if build["expert_rank"] else None)
    config = {**defaults, **{k: v for k, v in spec.items() if k in defaults}, "build": build,
              "family": "qwen3_5_moe", "layout": "real-training-v1", "base_fingerprint": configuration_hash(base_identity(bundle)),
              "dataset": dataset.identity, "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
              "optimizer": "AdamW(weight_decay=0,foreach=False,fused=False)",
              "precision": "BF16 base; FP32 LoRA/global/SwiGLU/routes; deterministic; TF32 off"}
    for key in bound:
        if key in spec and spec[key] != config[key]:
            raise ValueError(f"Bound training setting changed: {key}")
    if min(config["max_steps"], config["accumulation_steps"], config["ce_chunk_size"]) < 1:
        raise ValueError("Positive training step/accumulation/chunk settings required")
    return config


def profile(built, trainer, audit):
    """返回族专用资源观测及实际持有的张量清单。"""
    return {**memory_observation(), "windows_process": psutil.Process().memory_info()._asdict(),
            "system_available": psutil.virtual_memory().available,
            "inventory": {**live_inventory(built, trainer.optimizer),
                          "route_audit_cpu_baseline": sum(t.numel() * t.element_size() for t in audit.baseline.values())},
            "pinned_allocator": torch.cuda.memory.host_memory_stats()}


def route_audit(built, state=None):
    """构造专家参数的路由及 Adam 状态审核器。"""
    from nvfp4lora.validation.route_audit import ExpertRouteAudit
    return ExpertRouteAudit(built, state)


def extra_inventory(built):
    return {}


def replay_stops(bundle, tokenizer):
    """返回 checkpoint 的终止 token 集合和左 padding token。"""
    return {248044, tokenizer.convert_tokens_to_ids("<|im_end|>")}, 248044


def base_build(bundle):
    from nvfp4lora.model.build_moe import build_model
    return build_model(bundle, expert_rank=0, c3_rank=0, streaming="async")


def decode_frame(frames):
    return any(str(frame.get("filename", "")).replace("\\", "/").endswith("format/nvfp4.py") for frame in frames)
