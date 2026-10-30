"""合成密集模型训练对照、独立进程续训、发布重建和标准 PEFT 加载。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from types import SimpleNamespace

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from nvfp4lora.testing.numerics import compare_vectors, rng_probe, write_json
from tools.validation.compare_real_resume import read_checkpoint
from nvfp4lora.testing.numerics import equal_tree
from nvfp4lora.export.artifacts import base_identity, build_from_adapter, publish_adapter
from nvfp4lora.families import family_of_bundle
from nvfp4lora.export.delta_w import delta_w_fingerprints
from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.common import BuiltModel, canonical_name
from nvfp4lora.model.build_dense import build_dense_model
from nvfp4lora.model.packed_linear import LoRAPackedLinear
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.train.checkpoint import configuration_hash, load_checkpoint, save_checkpoint
from nvfp4lora.train.data import DeterministicSampleStream, SyntheticTinyDataset
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, loss_sum, memory_observation

TARGET = r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
BUILD = {"lora_rank": 4, "lora_alpha": 8,
         "lora_scope": "D-ALL", "decode_chunk_elements": 8 * 1024 * 1024, "streaming": "async"}


def build_hf_reference(bundle, *, adapter_directory=None):
    """Tiny独立参考：普通HF Linear预乘global；不用PackedLinear或其decoder。"""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["source"].get("kind") != "tiny-generated" or manifest["geometry"]["layers"] != 4:
        raise ValueError("Full HF decoded reference is restricted to the four-layer tiny source")
    config = Qwen3_5TextConfig.from_dict(json.loads((bundle / "config.json").read_text()))
    config.use_cache, config._attn_implementation = False, "sdpa"
    model = Qwen3_5ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    targets = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for entry in manifest["nonstream_files"]:
            path = bundle / entry["file"]
            assert file_hash(path) == entry["sha256"]
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    targets[name].copy_(reader.get_tensor(name))
                    loaded.add(name)
        store = PackedLayerStore(bundle / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
        for layer in range(4):
            tensors = store.tensors(layer)
            # Independent HF architecture traversal, rather than production replacement targets.
            for name, module in model.model.layers[layer].named_modules():
                prefix = f"model.layers.{layer}.{name}"
                if not isinstance(module, torch.nn.Linear) or not re.fullmatch(TARGET, prefix):
                    continue
                weight = tensors[name + ".weight"]
                if name + ".weight_scale_2" in tensors:
                    decoded = dequantize(weight, tensors[name + ".weight_scale"], tensors[name + ".weight_scale_2"], dtype=torch.bfloat16)
                else:
                    decoded = (weight.view(torch.float8_e4m3fn).float() * tensors[name + ".weight_scale"]).bfloat16()
                module.weight.copy_(decoded)
                loaded.add(prefix + ".weight")
                del decoded
    if loaded != set(targets):
        raise ValueError(f"Independent HF reference load incomplete: {set(targets) - loaded}")
    model.requires_grad_(False)
    if adapter_directory is None:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=BUILD["lora_rank"],
                               lora_alpha=BUILD["lora_alpha"], lora_dropout=0, target_modules=TARGET))
    else:
        model = PeftModel.from_pretrained(model, adapter_directory, local_files_only=True)
    trainable = {canonical_name(name): parameter for name, parameter in model.named_parameters() if ".lora_" in name}
    if len(trainable) != 50 or any(parameter.dtype != torch.float32 for parameter in trainable.values()):
        raise ValueError("Independent PEFT adapter set differs")
    return BuiltModel(model, store, SimpleNamespace(phase="FORWARD", stats={}, slots=[]),
                      dict(sorted(trainable.items())), {name: list(value.shape) for name, value in sorted(trainable.items())},
                      sorted(loaded), [], family=family_of_bundle(bundle).name)


def config_for(bundle, steps):
    dataset = SyntheticTinyDataset()
    return {"layout": "tiny-dense-training-v1", "family": "qwen3_5_dense", "seed": 20261029,
            "max_steps": steps, "learning_rate": 2e-4, "warmup_steps": 10, "max_grad_norm": 1.,
            "accumulation_steps": 2, "ce_chunk_size": 8, "gradient_checkpointing": True,
            "optimizer": "AdamW(weight_decay=0,foreach=False,fused=False)",
            "scheduler": "warmup(step+1)/10 then constant", "build": BUILD,
            "precision": "dense-packed-linear-bf16code-fp32scale-v1; FP32 LoRA; deterministic; TF32 off",
            "base_fingerprint": configuration_hash(base_identity(bundle)), "dataset": dataset.identity,
            "sampler_seed": 113, "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]}}


def close(built):
    if hasattr(built.streamer, "close"):
        built.streamer.close()


def save(trainer, stream, path, config, commit):
    save_checkpoint(path, trainer.built.trainable, trainer.optimizer, trainer.scheduler,
                    stream.state_dict(), trainer.state, config, provenance={"code_commit": commit})


def probe(built):
    phase, training = built.streamer.phase, built.model.training
    built.model.eval()
    built.streamer.phase = "FORWARD"
    batch = SyntheticTinyDataset()[7]
    try:
        with torch.no_grad():
            enabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False) / supervised_tokens(batch["labels"])
            with disable_adapters(built.model):
                disabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False) / supervised_tokens(batch["labels"])
        return {"loss": enabled.cpu(), "disabled_loss": disabled.cpu(),
                "parameters": {canonical_name(name): value.detach().cpu().clone() for name, value in built.model.named_parameters()},
                "frozen": frozen_fingerprint(built), "delta_w": delta_w_fingerprints(built)}
    finally:
        built.model.train(training)
        built.streamer.phase = phase


def paired(args, output, config):
    built = build_dense_model(args.bundle, **BUILD)
    references = {"same_expression": build_dense_model(args.bundle, **BUILD, reference=True),
                  "hf_peft_premultiplied": build_hf_reference(args.bundle)}
    for reference in references.values():
        if built.expected_trainable != reference.expected_trainable:
            raise ValueError("Reference trainable sets differ")
        with torch.no_grad():
            for name, parameter in reference.trainable.items():
                parameter.copy_(built.trainable[name])
    actual = LoRATrainer(built, config)
    trainers = {name: LoRATrainer(reference, config) for name, reference in references.items()}
    ids = [id(value) for group in actual.optimizer.param_groups for value in group["params"]]
    assert len(ids) == len(set(ids)) and set(ids) == {id(value) for value in built.trainable.values()}
    frozen = {name: frozen_fingerprint(value) for name, value in {"actual": built, **references}.items()}
    stream = DeterministicSampleStream(SyntheticTinyDataset(), seed=config["sampler_seed"])
    initial = probe(built)
    assert equal_tree(initial["loss"], initial["disabled_loss"]), "Zero-step loss differs"
    first, history, failures = {}, [], []
    observations = {"three_models_loaded": memory_observation()}
    try:
        for step in range(1, args.steps + 1):
            batches = stream.next_group(config["accumulation_steps"])
            metrics = actual.backward_group(batches)
            ref_metrics = {name: trainer.backward_group(batches) for name, trainer in trainers.items()}
            if (not equal_tree(metrics["loss"], ref_metrics["same_expression"]["loss"])
                    or not all(equal_tree(value.grad, references["same_expression"].trainable[key].grad)
                               for key, value in built.trainable.items())):
                failures.append(f"Same-expression loss or gradients differ at step {step}")
            if step == 1:
                for name, reference in references.items():
                    first[name] = {key: compare_vectors(value.grad, reference.trainable[key].grad) for key, value in built.trainable.items()}
                    failures.extend(f"first gradient {name}.{key}: {value}" for key, value in first[name].items() if value["cosine"] < .999)
                write_json(output / "first-gradients.json", first)
            metrics = actual.finish_step(metrics)
            for name, trainer in trainers.items():
                ref_metrics[name] = trainer.finish_step(ref_metrics[name])
            errors = {name: abs(metrics["loss"] - value["loss"]) / abs(value["loss"]) for name, value in ref_metrics.items()}
            failures.extend(f"step {step} {name} relative loss {value}" for name, value in errors.items() if value > .02)
            row = {**metrics, "reference_losses": {name: value["loss"] for name, value in ref_metrics.items()},
                   "reference_relative_loss": errors, "rng_probe": rng_probe()}
            history.append(row)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
            if step in [1, args.steps // 2, args.steps] or step % 10 == 0:
                print(f"step {step}/{args.steps}: loss={metrics['loss']:.8f}; relative={errors}", flush=True)
            if step == args.steps // 2:
                save(actual, stream, output / "checkpoint-middle", config, args.commit)
            if failures:
                write_json(output / "early-failures.json", failures)
                raise AssertionError(failures[:4])
        save(actual, stream, output / "checkpoint-final", config, args.commit)
        final = {name: {key: compare_vectors(value, reference.trainable[key]) for key, value in built.trainable.items()}
                 for name, reference in references.items()}
        if not all(equal_tree(value, references["same_expression"].trainable[key]) for key, value in built.trainable.items()):
            failures.append("Same-expression final parameter bytes differ")
        write_json(output / "final-weight-comparison.json", final)
        failures.extend(f"final weight {name}.{key}: {value}" for name, values in final.items() for key, value in values.items() if value["cosine"] < .99)
        frozen_checks = {name: frozen_fingerprint(value) == frozen[name] for name, value in {"actual": built, **references}.items()}
        assert all(frozen_checks.values()), frozen_checks
        torch.save(probe(built), output / "probe.pt")
        publish = publish_adapter(output / "publish", built, args.bundle, BUILD, config, code_commit=args.commit, global_step=args.steps)
        assert set(publish["tensor_specifications"]) == set(built.trainable) == set(built.expected_trainable)
        observations["finished_with_two_references_resident"] = memory_observation()
        summary = {"status": "FAIL" if failures else "PASS", "optimizer_steps": args.steps,
                   "zero_step_loss_E2": True, "zero_step_loss": float(initial["loss"]),
                   "trainable_tensors": len(built.trainable), "trainable_numel": sum(value.numel() for value in built.trainable.values()),
                   "optimizer_expected_publish_sets_equal": True, "frozen_checks": frozen_checks,
                   "references": {name: {"first_gradient_min_cosine": min(row["cosine"] for row in first[name].values()),
                       "first_gradient_max_relative_l2": max(row["relative_l2"] for row in first[name].values()),
                       "final_weight_min_cosine": min(row["cosine"] for row in final[name].values()),
                       "loss_max_relative": max(row["reference_relative_loss"][name] for row in history)} for name in references},
                   "memory": observations, "failures": failures}
        write_json(output / "summary.json", summary)
        assert not failures, failures[:4]
    finally:
        for value in [built, *references.values()]:
            close(value)


def resumed(args, output, config):
    origin = Path(args.origin)
    built = build_dense_model(args.bundle, **BUILD)
    trainer = LoRATrainer(built, config)
    stream = DeterministicSampleStream(SyntheticTinyDataset(), seed=config["sampler_seed"])
    data, trainer.state = load_checkpoint(origin / "checkpoint-middle", built.trainable, trainer.optimizer, trainer.scheduler, config)
    stream.load_state_dict(data)
    frozen = frozen_fingerprint(built)
    expected_history = {row["global_step"]: row for row in [json.loads(line) for line in (origin / "metrics.jsonl").read_text().splitlines()]}
    differences = []
    try:
        for step in range(trainer.state["global_step"] + 1, args.steps + 1):
            row = {**trainer.step(stream.next_group(config["accumulation_steps"])), "rng_probe": rng_probe()}
            for key in ["loss", "supervised_tokens", "sample_ids", "learning_rates", "gradient_norm", "group_gradient_norms", "rng_probe"]:
                if not equal_tree(row[key], expected_history[step][key]):
                    differences.append({"step": step, "field": key})
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
            if step % 10 == 0 or step == args.steps // 2 + 1:
                print(f"resumed {step}/{args.steps}: loss={row['loss']:.8f}", flush=True)
        save(trainer, stream, output / "checkpoint-final", config, args.commit)
        expected, actual = read_checkpoint(origin / "checkpoint-final"), read_checkpoint(output / "checkpoint-final")
        checks = {name: equal_tree(actual[name], expected[name]) for name in expected}
        actual_probe, expected_probe = probe(built), torch.load(origin / "probe.pt", weights_only=False)
        checks.update(per_step_E2=not differences, published_probe_E2=equal_tree(actual_probe, expected_probe),
                      training_memory_delta_W_E2=actual_probe["delta_w"] == expected_probe["delta_w"])
        assert len(checks) == 10
        frozen_equal = frozen_fingerprint(built) == frozen
        summary = {"status": "PASS" if all(checks.values()) and frozen_equal else "FAIL", "resumed_steps": args.steps // 2,
                   "checks": checks, "frozen_unchanged": frozen_equal, "differences": differences, "memory": memory_observation()}
        write_json(output / "summary.json", summary)
        assert summary["status"] == "PASS", summary
    finally:
        close(built)


def reloaded(args, output, config):
    origin = Path(args.origin)
    built = build_from_adapter(args.bundle, origin / "publish")
    try:
        actual, expected = probe(built), torch.load(origin / "probe.pt", weights_only=False)
        checks = {name: equal_tree(actual[name], expected[name]) for name in actual}
        checks["adapter_has_effect"] = not equal_tree(actual["loss"], actual["disabled_loss"])
        reference = build_hf_reference(args.bundle, adapter_directory=origin / "publish")
        standard = probe(reference)
        relative = abs(float(standard["loss"]) - float(actual["loss"])) / abs(float(actual["loss"]))
        checks["standard_peft_E3"] = relative <= .02
        summary = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                   "loss": float(actual["loss"]), "disabled_loss": float(actual["disabled_loss"]),
                   "standard_peft_loss": float(standard["loss"]), "standard_peft_relative_loss": relative,
                   "delta_w_count": len(actual["delta_w"]["tensors"]), "memory": memory_observation(),
                   "reconstruction_inputs": [str(args.bundle), str(origin / "publish")], "training_checkpoints_read": False}
        write_json(output / "summary.json", summary)
        assert summary["status"] == "PASS", summary
    finally:
        close(built)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--commit", default="uncommitted")
    parser.add_argument("--streaming", choices=["sync", "async"], default="async")
    parser.add_argument("--mode", choices=["all", "paired", "resumed", "reloaded"], default="all")
    parser.add_argument("--origin", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.steps % 2:
        raise ValueError("Expected even optimizer steps >=2")
    BUILD["streaming"] = args.streaming
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261029)
    config = config_for(args.bundle, args.steps)
    write_json(args.output / "train_config.json", config)
    if args.mode != "all":
        try:
            {"paired": paired, "resumed": resumed, "reloaded": reloaded}[args.mode](args, args.output, config)
        except Exception:
            if not (args.output / "summary.json").exists():
                write_json(args.output / "summary.json", {"status": "FAIL", "traceback": traceback.format_exc()})
            raise
        return
    results = {}
    for mode in ["paired", "resumed", "reloaded"]:
        command = [sys.executable, "-B", __file__, "--mode", mode, "--bundle", str(args.bundle),
                   "--output", str(args.output / mode), "--steps", str(args.steps), "--commit", args.commit,
                   "--streaming", args.streaming, "--origin", str(args.output / "paired")]
        print(f"Starting fresh process: {mode}", flush=True)
        with (args.output / f"{mode}.log").open("w", encoding="utf-8") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        results[mode] = {"command": command, "exit_code": child.returncode}
        if (args.output / mode / "summary.json").exists():
            results[mode]["summary"] = json.loads((args.output / mode / "summary.json").read_text())
        print(f"{mode}: exit {child.returncode}", flush=True)
        if child.returncode:
            break
    status = "PASS" if len(results) == 3 and all(row["exit_code"] == 0 for row in results.values()) else "FAIL"
    write_json(args.output / "summary.json", {"status": status, "qualification": "D4" if args.steps == 100 else "SMOKE ONLY", "workers": results})
    assert status == "PASS", {name: row["exit_code"] for name, row in results.items()}


if __name__ == "__main__":
    main()
