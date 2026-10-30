"""P4：同步 HF tiny 100 步独立稠密对照、新进程续训与发布重建。"""

import argparse
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from nvfp4lora.export.artifacts import base_identity, build_from_adapter, publish_adapter
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.packed_experts import PackedExperts
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.checkpoint import capture_rng, configuration_hash, load_checkpoint, save_checkpoint
from nvfp4lora.train.data import DeterministicSampleStream, SyntheticTinyDataset
from nvfp4lora.train.loop import (LoRATrainer, configure_training, frozen_fingerprint, live_inventory,
                                 loss_sum, memory_observation)


BUILD = {"expert_rank": 2, "expert_alpha": 4, "c3_rank": 4, "c3_alpha": 8, "chunk_size": 3}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def config_for(bundle, steps):
    """完整的数值与数据身份；运行路径不进入逻辑配置。"""
    dataset = SyntheticTinyDataset()
    return {"layout": "tiny-training-v1", "seed": 20261028, "max_steps": steps,
            "learning_rate": 2e-4, "warmup_steps": 10, "max_grad_norm": 1.0,
            "accumulation_steps": 2, "ce_chunk_size": 8, "gradient_checkpointing": True,
            "optimizer": "AdamW(weight_decay=0,foreach=False,fused=False)",
            "scheduler": "warmup(step+1)/10 then constant", "build": BUILD,
            "precision": "BF16 base; FP32 LoRA/global/SwiGLU/routes; deterministic; TF32 off",
            "base_fingerprint": configuration_hash(base_identity(bundle)), "dataset": dataset.identity,
            "sampler_seed": 113, "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]}}


def rng_probe():
    """只用于续训验收：每步实际消耗四类 RNG，避免只比较未使用的随机状态。"""
    return {"python": random.random(), "numpy": float(np.random.random()),
            "torch_cpu": float(torch.rand(())), "torch_cuda": float(torch.rand((), device="cuda"))}


def compare_vectors(actual, expected):
    """FP64 计算比较指标；零向量单独检查，不使用 eps 掩盖差异。"""
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    an, bn = float(a.norm()), float(b.norm())
    if bn == 0 or an == 0:
        return {"relative_l2": 0.0 if an == bn else float("inf"), "cosine": 1.0 if an == bn else 0.0}
    return {"relative_l2": float((a - b).norm()) / bn, "cosine": float(torch.dot(a, b)) / (an * bn)}


def save(trainer, stream, directory, config, commit):
    save_checkpoint(directory, trainer.built.trainable, trainer.optimizer, trainer.scheduler,
                    stream.state_dict(), trainer.state, config, provenance={"code_commit": commit})


def probe(built, dataset):
    """固定合法 ID、固定所有参数及逐专家 ΔW，供只读发布进程逐位读回。"""
    built.model.eval()
    old_phase = built.streamer.phase
    built.streamer.phase = "FORWARD"
    batch = dataset[7]
    with torch.no_grad():
        enabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False) / supervised_tokens(batch["labels"])
        with disable_adapters(built.model):
            disabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False) / supervised_tokens(batch["labels"])
        deltas = {}
        for module in built.model.modules():
            if isinstance(module, PackedExperts) and module.rank:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    adapter = getattr(module, name)
                    # tiny 的每个专家均校验，真实模型验收可逐专家流式处理。
                    deltas[f"{module.layer}.{name}"] = (adapter.scaling * (adapter.lora_B @ adapter.lora_A)).cpu()
    result = {"loss": enabled.cpu(), "disabled_loss": disabled.cpu(), "deltas": deltas,
              "parameters": {name: p.detach().cpu().clone() for name, p in built.trainable.items()}}
    built.model.train()
    built.streamer.phase = old_phase
    return result


def paired(args, output, config):
    """生产和独立稠密模型相同初值、相同数据，累计 100 次真正的 optimizer step。"""
    built = build_model(args.bundle, **BUILD)
    reference = build_model(args.bundle, **BUILD, store=built.store, reference=True)
    with torch.no_grad():
        for name, parameter in reference.trainable.items():
            parameter.copy_(built.trainable[name])
    actual, dense = (LoRATrainer(model, config) for model in [built, reference])
    dataset = SyntheticTinyDataset(**{k: config["dataset"][k] for k in ["size", "vocab_size", "seed"]})
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    frozen = frozen_fingerprint(built)
    first_gradients, history, observations = {}, [], {"built_pair": memory_observation()}
    failure = []
    for step in range(1, args.steps + 1):
        batches = stream.next_group(config["accumulation_steps"])
        torch.cuda.reset_peak_memory_stats()
        if step == 1:
            # Windows 的 C++/free traceback 采集会干扰 checkpoint 的内部早停异常；
            # 仅记录分配的 Python 栈，保留定位活跃对象所需的全部 allocation 证据。
            torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=200000, clear_history=True)
        try:
            am = actual.backward_group(batches)
            if step == 1:
                observations["first_backward_with_reference_resident"] = {**memory_observation(), "inventory": live_inventory(built, actual.optimizer)}
                torch.cuda.memory._dump_snapshot(str(output / "first-backward.pickle"))
                torch.cuda.memory._record_memory_history(enabled=None)
            dm = dense.backward_group(batches)
        except Exception:
            torch.save(batches, output / f"failed-batch-{step}.pt")
            raise
        if step == 1:
            for name, parameter in built.trainable.items():
                first_gradients[name] = compare_vectors(parameter.grad, reference.trainable[name].grad)
                result = first_gradients[name]
                if result["relative_l2"] > .02 or result["cosine"] < .999:
                    failure.append(f"A4.1 first gradient {name}: {result}")
            write_json(output / "first-gradients.json", first_gradients)
        am, dm = actual.finish_step(am), dense.finish_step(dm)
        relative = abs(am["loss"] - dm["loss"]) / abs(dm["loss"])
        if relative > .02:
            failure.append(f"A4.2 step {step}: {relative}")
        row = {**am, "reference_loss": dm["loss"], "relative_loss_error": relative, "rng_probe": rng_probe()}
        history.append(row)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        if step in [1, args.steps // 2, args.steps]:
            observations[f"step_{step}"] = {**memory_observation(), "inventory": live_inventory(built, actual.optimizer)}
            print(f"step {step}/{args.steps}: loss={am['loss']:.8f}, reference={dm['loss']:.8f}, relative={relative:.3g}", flush=True)
        if step == args.steps // 2:
            torch.cuda.reset_peak_memory_stats()
            save(actual, stream, output / "checkpoint-middle", config, args.commit)
            observations["save_middle"] = memory_observation()
    save(actual, stream, output / "checkpoint-final", config, args.commit)
    frozen_equal = frozen_fingerprint(built) == frozen
    if not frozen_equal:
        failure.append("A4.5 frozen weights/buffers/store changed")
    final_weights = {name: compare_vectors(parameter, reference.trainable[name]) for name, parameter in built.trainable.items()}
    failure.extend(f"A4.1 final weights {name}: {value}" for name, value in final_weights.items() if value["cosine"] < .99)
    write_json(output / "final-weight-comparison.json", final_weights)
    torch.cuda.reset_peak_memory_stats()
    torch.save(probe(built, dataset), output / "probe.pt")
    publish = publish_adapter(output / "publish", built, args.bundle, BUILD, config,
                              code_commit=args.commit, global_step=args.steps)
    observations["evaluation_publish"] = memory_observation()
    equal_sets = set(built.trainable) == set(built.expected_trainable) == set(publish["tensor_specifications"])
    if not equal_sets:
        failure.append("A4.6 optimizer/expected/publish keys differ")
    summary = {"status": "FAIL" if failure else "PASS", "optimizer_steps": args.steps,
               "accumulation_steps": config["accumulation_steps"], "samples": stream.consumed,
               "frozen_hashes_unchanged": frozen_equal, "canonical_sets_equal": equal_sets,
               "trainable_tensor_count": len(built.trainable), "trainable_numel": sum(p.numel() for p in built.trainable.values()),
               "first_gradient_worst_relative_l2": max(x["relative_l2"] for x in first_gradients.values()),
               "first_gradient_min_cosine": min(x["cosine"] for x in first_gradients.values()),
               "final_weight_min_cosine": min(x["cosine"] for x in final_weights.values()),
               "loss_max_relative": max(row["relative_loss_error"] for row in history),
               "memory": observations, "streamer_stats": built.streamer.stats, "failures": failure}
    write_json(output / "summary.json", summary)
    assert not failure, failure[:5]


def resumed(args, output, config):
    """全新 Python 进程重建对象后按名恢复；比较后 50 步所有 loss/RNG 和最终状态。"""
    origin = Path(args.origin)
    built = build_model(args.bundle, **BUILD)
    trainer = LoRATrainer(built, config)
    dataset = SyntheticTinyDataset()
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    data, trainer.state = load_checkpoint(origin / "checkpoint-middle", built.trainable,
                                          trainer.optimizer, trainer.scheduler, config)
    stream.load_state_dict(data)
    observations = {"load": memory_observation()}
    frozen = frozen_fingerprint(built)
    uninterrupted = [json.loads(line) for line in (origin / "metrics.jsonl").read_text().splitlines()]
    failures = []
    for step in range(trainer.state["global_step"] + 1, args.steps + 1):
        batches = stream.next_group(config["accumulation_steps"])
        torch.cuda.reset_peak_memory_stats()
        try:
            row = {**trainer.step(batches), "rng_probe": rng_probe()}
        except Exception:
            torch.save(batches, output / f"failed-batch-{step}.pt")
            raise
        expected = uninterrupted[step - 1]
        for key in ["loss", "supervised_tokens", "sample_ids", "learning_rates", "gradient_norm", "group_gradient_norms", "rng_probe"]:
            if row[key] != expected[key]:
                failures.append(f"step {step} E2 mismatch: {key} {row[key]} != {expected[key]}")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        if step in [args.steps // 2 + 1, args.steps]:
            observations[f"step_{step}"] = {**memory_observation(), "inventory": live_inventory(built, trainer.optimizer)}
            print(f"resumed step {step}/{args.steps}: loss={row['loss']:.8f}", flush=True)
    save(trainer, stream, output / "checkpoint-final", config, args.commit)
    # 文件字节可能因 pickle 编码不同而异；比较所有状态的实际 Tensor/标量内容。
    actual = output / "checkpoint-final"
    expected = origin / "checkpoint-final"
    from safetensors.torch import load_file
    def equal_tree(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal_tree(value, b[key]) for key, value in a.items())
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
        return a == b
    checks = {"parameters": equal_tree(load_file(actual / "lora_params.safetensors"), load_file(expected / "lora_params.safetensors"))}
    for name in ["optimizer.pt", "scheduler.pt", "rng.pt"]:
        checks[name] = equal_tree(torch.load(actual / name, weights_only=False), torch.load(expected / name, weights_only=False))
    for name in ["data_state.json", "trainer_state.json"]:
        checks[name] = json.loads((actual / name).read_text()) == json.loads((expected / name).read_text())
    checks["frozen"] = frozen_fingerprint(built) == frozen
    failures.extend(f"final state differs: {name}" for name, equal in checks.items() if not equal)
    write_json(output / "summary.json", {"status": "FAIL" if failures else "PASS", "resumed_steps": args.steps - args.steps // 2,
        "loss_comparison": "E2" if not failures else "FAIL", "state_checks": checks, "memory": observations, "failures": failures})
    assert not failures, failures[:5]


def reloaded(args, output, config):
    """重载过程只拿到预处理产物、发布目录和独立固定探针，完全不读训练 checkpoint。"""
    origin = Path(args.origin)
    built = build_from_adapter(args.bundle, origin / "publish")
    actual = probe(built, SyntheticTinyDataset())
    expected = torch.load(origin / "probe.pt", weights_only=False)
    checks = {"loss": torch.equal(actual["loss"].view(torch.int32), expected["loss"].view(torch.int32)),
              "disabled_loss": torch.equal(actual["disabled_loss"].view(torch.int32), expected["disabled_loss"].view(torch.int32)),
              "parameters": all(torch.equal(value.view(torch.uint8), expected["parameters"][name].view(torch.uint8)) for name, value in actual["parameters"].items()),
              "delta_w": all(torch.equal(value.view(torch.uint8), expected["deltas"][name].view(torch.uint8)) for name, value in actual["deltas"].items()),
              "adapter_has_effect": float(actual["loss"]) != float(actual["disabled_loss"])}
    write_json(output / "summary.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "loss": float(actual["loss"]), "disabled_loss": float(actual["disabled_loss"]), "memory": memory_observation()})
    assert all(checks.values()), checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--commit", default="uncommitted")
    parser.add_argument("--streaming", choices=["sync", "async"], default="sync")
    parser.add_argument("--mode", choices=["all", "paired", "resumed", "reloaded"], default="all")
    parser.add_argument("--origin")
    args = parser.parse_args()
    BUILD["streaming"] = args.streaming
    args.output.mkdir(parents=True, exist_ok=True)
    if args.steps < 2 or args.steps % 2:
        raise ValueError("Need an even optimizer step count >=2")
    configure_training(20261028)
    config = config_for(args.bundle, args.steps)
    write_json(args.output / "train_config.json", config)
    if args.mode != "all":
        {"paired": paired, "resumed": resumed, "reloaded": reloaded}[args.mode](args, args.output, config)
        return
    results = {}
    for mode in ["paired", "resumed", "reloaded"]:
        command = [sys.executable, "-B", __file__, "--mode", mode, "--bundle", str(args.bundle),
                   "--output", str(args.output / mode), "--steps", str(args.steps), "--commit", args.commit,
                   "--streaming", args.streaming,
                   "--origin", str(args.output / "paired")]
        with (args.output / f"{mode}.log").open("w", encoding="utf-8") as log:
            print(f"Starting independent process: {mode}", flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
        results[mode] = {"exit_code": result.returncode}
        print(f"{mode}: exit {result.returncode}", flush=True)
        summary = args.output / mode / "summary.json"
        if summary.exists():
            results[mode]["summary"] = json.loads(summary.read_text())
        # 基础进程未能保存检查点/发布产物时，后续进程无可用输入。
        if mode == "paired" and not (args.output / "paired" / "publish" / "manifest.json").exists():
            break
    status = "PASS" if len(results) == 3 and all(r["exit_code"] == 0 for r in results.values()) else "FAIL"
    write_json(args.output / "summary.json", {"status": status, "qualification": "P4" if args.steps == 100 else "SMOKE ONLY", "workers": results})
    assert status == "PASS", {mode: result["exit_code"] for mode, result in results.items()}


if __name__ == "__main__":
    main()
