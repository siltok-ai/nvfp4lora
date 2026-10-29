"""可复现的真实单GPU训练会话，执行停止位置与逻辑训练配置分离。"""

import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import base_identity, publish_adapter
from nvfp4lora.export.delta_w import delta_w_fingerprints
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build import build_model
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.checkpoint import configuration_hash, load_checkpoint, save_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream
from nvfp4lora.train.evaluate import evaluate_canary, evaluate_nll, evaluation_mode
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, live_inventory, loss_sum, memory_observation
from nvfp4lora.validation.route_audit import ExpertRouteAudit


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def logical_config(spec, bundle, dataset):
    """逻辑参数、数据、基座和版本参与恢复身份；输出/stop-at/保存与评测频率不参与。"""
    if spec.get("family") == "qwen3_5_dense":
        return logical_config_dense(spec, bundle, dataset)
    defaults = {"seed": 20261028, "max_steps": 100, "max_length": 2048,
                "learning_rate": 2e-4, "warmup_steps": 10, "max_grad_norm": 1.,
                "accumulation_steps": 8, "ce_chunk_size": 128, "gradient_checkpointing": True,
                "sampler_seed": 113, "rng_probe": False}
    build = {"expert_rank": 0, "c3_rank": 16, "c3_alpha": 32, "chunk_size": 32, "streaming": "async"}
    bound = {"layout", "base_fingerprint", "dataset", "versions", "optimizer", "precision"}
    if spec.keys() - defaults.keys() - {"build"} - bound:
        raise ValueError(f"Unknown training settings: {spec.keys() - defaults.keys() - {'build'} - bound}")
    allowed_build = set(build) | {"expert_alpha"}
    if spec.get("build", {}).keys() - allowed_build:
        raise ValueError("Unsupported build setting")
    build.update(spec.get("build", {}))
    build.setdefault("expert_alpha", 2 * build["expert_rank"] if build["expert_rank"] else None)
    config = {**defaults, **{k: v for k, v in spec.items() if k in defaults}, "build": build,
              "layout": "real-training-v1", "base_fingerprint": configuration_hash(base_identity(bundle)),
              "dataset": dataset.identity, "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
              "optimizer": "AdamW(weight_decay=0,foreach=False,fused=False)",
              "precision": "BF16 base; FP32 LoRA/global/SwiGLU/routes; deterministic; TF32 off"}
    for key in bound:
        if key in spec and spec[key] != config[key]:
            raise ValueError(f"Bound training setting changed: {key}")
    if min(config["max_steps"], config["accumulation_steps"], config["ce_chunk_size"]) < 1:
        raise ValueError("Positive training step/accumulation/chunk settings required")
    return config


def logical_config_dense(spec, bundle, dataset):
    """Dense单独配置身份；不向历史MoE配置增加family或新默认值。"""
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


def rng_probe():
    """仅在恢复验证启用；真实消耗四类随机状态，不影响dropout0的模型数学。"""
    return {"python": random.random(), "numpy": float(np.random.random()),
            "torch_cpu": float(torch.rand(())), "torch_cuda": float(torch.rand((), device="cuda"))}


def profile(built, trainer, audit):
    if getattr(built.model, "_nvfp4lora_family", None) == "qwen3_5_dense":
        inventory = live_inventory(built, trainer.optimizer)
        inventory["dense_layer_cpu_pinned_payload"] = inventory.pop("expert_cpu_pinned_payload")
        inventory["dense_layer_gpu_slots"] = inventory.pop("expert_gpu_slots")
        table = built.model.model.embed_tokens.table
        inventory["dense_embedding_cpu"] = table.numel() * table.element_size()
        free, total = torch.cuda.mem_get_info()
        return {**memory_observation(), "windows_process": psutil.Process().memory_info()._asdict(),
                "system_available": psutil.virtual_memory().available, "gpu_free": free, "gpu_total": total,
                "inventory": inventory, "pinned_allocator": torch.cuda.memory.host_memory_stats()}
    return {**memory_observation(), "windows_process": psutil.Process().memory_info()._asdict(),
            "system_available": psutil.virtual_memory().available,
            "inventory": {**live_inventory(built, trainer.optimizer),
                          "route_audit_cpu_baseline": sum(t.numel() * t.element_size() for t in audit.baseline.values())},
            "pinned_allocator": torch.cuda.memory.host_memory_stats()}


def probe(built, dataset, ce_chunk_size):
    sample = dataset[0]
    with evaluation_mode(built):
        enabled = loss_sum(built, sample, ce_chunk_size=ce_chunk_size, recompute=False) / supervised_tokens(sample["labels"])
        with disable_adapters(built.model):
            disabled = loss_sum(built, sample, ce_chunk_size=ce_chunk_size, recompute=False) / supervised_tokens(sample["labels"])
    assert torch.isfinite(enabled) and torch.isfinite(disabled)
    return {"input_ids": sample["input_ids"].tolist(), "labels": sample["labels"].tolist(),
            "loss": float(enabled), "disabled_loss": float(disabled),
            "loss_fp32_bits": int(enabled.view(torch.int32)), "disabled_loss_fp32_bits": int(disabled.view(torch.int32))}


def run_session(*, bundle, data, settings, output, resume=None, stop_at=None, checkpoint_steps=(),
                evaluation_steps=(), eval_corpus=None, eval_batch_size=32, code_commit="uncommitted",
                snapshot_first_micro=False, record_delta_w=True):
    """训练/续训共用同一入口；所有检查点、发布、证据使用本次新输出目录。"""
    output, bundle, data = Path(output), Path(bundle), Path(data)
    output.mkdir(parents=True, exist_ok=False)
    configure_training(settings.get("seed", 20261028))
    tokenizer = AutoTokenizer.from_pretrained(bundle, local_files_only=True)
    dataset = ChatJSONLDataset(data, tokenizer, max_length=settings.get("max_length", 2048))
    config = logical_config(settings, bundle, dataset)
    write_json(output / "train_config.json", config)
    write_json(output / "execution.json", {"bundle": str(bundle), "data": str(data), "resume": str(resume) if resume else None,
        "stop_at": stop_at, "checkpoint_steps": list(checkpoint_steps), "evaluation_steps": list(evaluation_steps),
        "eval_corpus": str(eval_corpus) if eval_corpus else None, "eval_batch_size": eval_batch_size, "code_commit": code_commit})
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    torch.cuda.reset_peak_memory_stats()
    if config.get("family") == "qwen3_5_dense":
        from nvfp4lora.model.build_dense import build_dense_model
        built = build_dense_model(bundle, **config["build"])
    else:
        built = build_model(bundle, **config["build"])
    trainer = LoRATrainer(built, config)
    if resume:
        data_state, trainer.state = load_checkpoint(resume, built.trainable, trainer.optimizer, trainer.scheduler, config)
        stream.load_state_dict(data_state)
    if config.get("family") == "qwen3_5_dense":
        audit = _DenseNoRouteAudit(trainer.state.get("route_audit"))
    else:
        audit = ExpertRouteAudit(built, trainer.state.get("route_audit"))
    start = trainer.state["global_step"]
    stop = config["max_steps"] if stop_at is None else stop_at
    if not start < stop <= config["max_steps"]:
        raise ValueError("Need saved_step < stop_at <= logical max_steps")
    observations = {"load_or_restore": profile(built, trainer, audit)}
    summary = {"status": "RUNNING", "starting_step": start, "target_execution_step": stop,
               "logical_max_steps": config["max_steps"], "trainable_tensors": len(built.trainable),
               "trainable_numel": sum(p.numel() for p in built.trainable.values()), "memory": observations}
    write_json(output / "summary.json", summary)
    print(f"Model ready; training optimizer steps {start + 1}..{stop}", flush=True)
    frozen = frozen_fingerprint(built)
    if start == 0:
        initial = probe(built, dataset, config["ce_chunk_size"])
        summary["initial_probe"] = initial
        assert initial["loss_fp32_bits"] == initial["disabled_loss_fp32_bits"], "Zero-adapter loss differs from unified baseline"
    replay = None
    if eval_corpus:
        eval_corpus = Path(eval_corpus)
        replay = ChatJSONLDataset(eval_corpus / "replay.jsonl", tokenizer, max_length=config["max_length"])
        write_json(output / "evaluation_identity.json", {"replay": replay.identity,
            "canary_file_sha256": __import__("hashlib").sha256((eval_corpus / "canary-holdout.jsonl").read_bytes()).hexdigest()})
    def evaluate(step, disabled=False):
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        result = {"step": step, "disabled": disabled,
            "replay": evaluate_nll(built, replay, batch_size=eval_batch_size, ce_chunk_size=config["ce_chunk_size"], disabled=disabled,
                       progress=lambda done, total: print(f"eval step {step} replay {done}/{total}", flush=True)),
            "canary": evaluate_canary(built, tokenizer, eval_corpus / "canary-holdout.jsonl", batch_size=eval_batch_size, disabled=disabled,
                       progress=lambda done, total: print(f"eval step {step} canary {done}/{total}", flush=True))}
        result.update(wall_seconds=time.perf_counter() - began, memory=profile(built, trainer, audit))
        write_json(output / f"evaluation-step{step:06d}{'-disabled' if disabled else ''}.json", result)
        return {"replay_nll": result["replay"]["nll_per_token"], "canary_correct": result["canary"]["correct"], "canary_total": result["canary"]["samples"]}
    if replay is not None and start == 0:
        summary["baseline_evaluation"] = evaluate(0, disabled=True)
    saved = []
    for step in range(start + 1, stop + 1):
        batches = stream.next_group(config["accumulation_steps"])
        if hasattr(built.streamer, "collect_timings"):
            built.streamer.collect_timings(synchronize=True)
        before_stats = dict(built.streamer.stats)
        torch.cuda.reset_peak_memory_stats()
        audit.begin_step()
        if snapshot_first_micro and step == start + 1:
            torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=200000, clear_history=True)
        def on_micro(index, current):
            if snapshot_first_micro and step == start + 1 and index == 0:
                observations["first_micro_backward"] = profile(built, trainer, audit)
                torch.cuda.memory._dump_snapshot(str(output / "first-micro-backward.pickle"))
                torch.cuda.memory._record_memory_history(enabled=None)
            print(f"optimizer step {step}/{stop}, microbatch {index + 1}/{len(batches)} complete", flush=True)
        began = time.perf_counter()
        try:
            metrics = trainer.backward_group(batches, on_microbatch=on_micro)
            route_report = audit.before_update()
            metrics = trainer.finish_step(metrics)
            route_report.update(audit.after_update(trainer.optimizer))
        except Exception:
            torch.save(batches, output / f"failed-batch-step{step}.pt")
            summary.update(status="FAIL", failed_step=step)
            write_json(output / "summary.json", summary)
            raise
        trainer.state["route_audit"] = audit.state_dict()
        metrics.update(route_audit=route_report, sequence_tokens=sum(row["input_ids"].numel() for row in batches),
                       step_wall_seconds=time.perf_counter() - began,
                       transfers={key: value - before_stats[key] for key, value in built.streamer.stats.items()},
                       memory=profile(built, trainer, audit))
        if config["rng_probe"]:
            metrics["rng_probe"] = rng_probe()
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics) + "\n")
        print(f"step {step}: loss={metrics['loss']:.8f}, grad={metrics['gradient_norm']:.6f}, seconds={metrics['step_wall_seconds']:.2f}", flush=True)
        observations[f"step_{step}"] = metrics["memory"]
        if replay is not None and step in evaluation_steps:
            summary[f"evaluation_step_{step}"] = evaluate(step)
        if step in checkpoint_steps or step == stop:
            torch.cuda.reset_peak_memory_stats()
            directory = output / "checkpoints" / f"step-{step:06d}"
            save_checkpoint(directory, built.trainable, trainer.optimizer, trainer.scheduler, stream.state_dict(), trainer.state,
                            config, provenance={"code_commit": code_commit})
            saved.append(str(directory))
            observations[f"save_{step}"] = profile(built, trainer, audit)
        summary.update(current_step=step, checkpoints=saved)
        write_json(output / "summary.json", summary)
    torch.cuda.reset_peak_memory_stats()
    publish_adapter(output / "publish", built, bundle, config["build"], config, code_commit=code_commit, global_step=stop)
    summary["published_probe"] = probe(built, dataset, config["ce_chunk_size"])
    if record_delta_w:
        print("Computing streamed training-memory deltaW fingerprints", flush=True)
        write_json(output / "delta-w.json", delta_w_fingerprints(built))
    observations["publish_and_probe"] = profile(built, trainer, audit)
    summary["frozen_hashes_unchanged"] = frozen_fingerprint(built) == frozen
    summary.update(status="PASS" if summary["frozen_hashes_unchanged"] else "FAIL", current_step=trainer.state["global_step"],
                   max_steps_reached=stop == config["max_steps"], final_trainer_state=trainer.state)
    audit.close()
    if config.get("family") == "qwen3_5_dense":
        if hasattr(built.streamer, "close"):
            built.streamer.close()
    else:
        built.streamer.close()
    write_json(output / "summary.json", summary)
    assert summary["status"] == "PASS"
    return summary
