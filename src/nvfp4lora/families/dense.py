"""Dense 的训练身份、空路由审核和生成约定。"""

import importlib.metadata
import json
from pathlib import Path
import psutil
import torch
from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.train.checkpoint import configuration_hash
from nvfp4lora.train.loop import live_inventory, memory_observation


def logical_config(spec, bundle, dataset):
    """绑定训练参数、数据、基座和版本，并检查固定身份。"""
    import math
    defaults = {"seed": 20261029, "max_steps": 100, "max_length": 2048,
                "learning_rate": 2e-4, "warmup_steps": 10, "max_grad_norm": 1.,
                "accumulation_steps": 8, "ce_chunk_size": 128, "gradient_checkpointing": True,
                "sampler_seed": 113, "rng_probe": False}
    build = {"family": "qwen3_5_dense", "lora_rank": 16, "lora_scope": "D-ALL",
             "decode_chunk_elements": 8 * 1024 * 1024, "streaming": "async"}
    bound = {"layout", "base_fingerprint", "dataset", "versions", "optimizer", "precision"}
    if spec.keys() - defaults.keys() - {"family", "build"} - bound:
        raise ValueError("Unknown dense training settings")
    if spec.get("build", {}).keys() - set(build) - {"lora_alpha"}:
        raise ValueError("Unsupported dense build setting")
    build.update(spec.get("build", {}))
    build["lora_alpha"] = 2 * build["lora_rank"] if build.get("lora_alpha") is None else build["lora_alpha"]
    if (build["family"] != "qwen3_5_dense" or not isinstance(build["lora_rank"], int) or build["lora_rank"] < 1
            or build["lora_scope"] not in ["D-ALL", "D-ATTN"] or build["streaming"] not in ["sync", "async"]
            or not math.isfinite(build["lora_alpha"]) or build["lora_alpha"] <= 0
            or build["decode_chunk_elements"] < 16 or build["decode_chunk_elements"] % 16):
        raise ValueError("Invalid dense build contract")
    identity = base_identity(bundle)
    if identity.get("family") != "qwen3_5_dense":
        raise ValueError("Dense training configuration needs a dense bundle")
    config = {**defaults, **{key: value for key, value in spec.items() if key in defaults}, "build": build,
              "family": "qwen3_5_dense", "layout": "real-dense-training-v1",
              "base_fingerprint": configuration_hash(identity), "dataset": dataset.identity,
              "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
              "optimizer": "AdamW(weight_decay=0,foreach=False,fused=False)",
              "precision": "BF16 HF base/activations; BF16 codes GEMM/FP32 global; FP32 LoRA; deterministic; TF32 off"}
    for key in bound:
        if key in spec and spec[key] != config[key]:
            raise ValueError(f"Bound dense training setting changed: {key}")
    if min(config["max_steps"], config["max_length"], config["accumulation_steps"], config["ce_chunk_size"]) < 1:
        raise ValueError("Positive dense steps/length/accumulation/chunk settings required")
    if any(not math.isfinite(config[key]) or config[key] <= 0 for key in ["learning_rate", "max_grad_norm"]):
        raise ValueError("Positive finite dense learning rate/gradient norm required")
    return config


class _DenseNoRouteAudit:
    """Dense没有路由，既不安装expert hooks，也不报告伪造的路由验收。"""

    def __init__(self, state=None):
        self.baseline = {}
        if state is not None and state != self.state_dict():
            raise ValueError("Dense checkpoint contains incompatible route-audit state")

    def begin_step(self):
        pass

    def before_update(self):
        return {}

    def after_update(self, optimizer):
        return {}

    def state_dict(self):
        return {"layout": "dense-no-routing-v1"}

    def close(self):
        pass


def profile(built, trainer, audit):
    """返回族专用资源观测及实际持有的张量清单。"""
    inventory = live_inventory(built, trainer.optimizer)
    inventory["dense_layer_cpu_pinned_payload"] = inventory.pop("expert_cpu_pinned_payload")
    inventory["dense_layer_gpu_slots"] = inventory.pop("expert_gpu_slots")
    table = built.model.model.embed_tokens.table
    inventory["dense_embedding_cpu"] = table.numel() * table.element_size()
    free, total = torch.cuda.mem_get_info()
    return {**memory_observation(), "windows_process": psutil.Process().memory_info()._asdict(),
            "system_available": psutil.virtual_memory().available, "gpu_free": free, "gpu_total": total,
            "inventory": inventory, "pinned_allocator": torch.cuda.memory.host_memory_stats()}


def route_audit(built, state=None):
    """构造不安装路由 hook 的空审核器。"""
    return _DenseNoRouteAudit(state)


def extra_inventory(built):
    table = built.model.model.embed_tokens.table
    return {"embedding_cpu": table.numel() * table.element_size()}


def replay_stops(bundle, tokenizer):
    """使用 bundle 自己的生成配置及 tokenizer 选择终止和 padding token。"""
    generation = json.loads((Path(bundle) / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation["eos_token_id"]
    stops = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return stops, pad


def base_build(bundle):
    from nvfp4lora.model.build_dense import build_dense_model
    return build_dense_model(bundle, lora_rank=0, streaming="async")


def decode_frame(frames):
    return any((str(frame.get("filename", "")).replace("\\", "/").endswith("format/nvfp4.py")
                or (str(frame.get("filename", "")).replace("\\", "/").endswith("ops/packed_linear_fn.py")
                    and frame.get("name") == "decode_codes")) for frame in frames)
