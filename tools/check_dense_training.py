"""DA6.1首步资格：真实损失图激活、PackedLinear dx与LoRA梯度双参考。"""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from dense_fixture import materialized_codes
from check_dense_forward import memory
from check_tiny_training import compare_vectors
from nvfp4lora.export.artifacts import base_identity, publish_adapter
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build import canonical_name
from nvfp4lora.model.build_dense import build_dense_model
from nvfp4lora.model.packed_linear import LoRAPackedLinear, PackedLinear
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.checkpoint import configuration_hash, load_checkpoint, save_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream, SyntheticTinyDataset
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, loss_sum
from nvfp4lora.train.session import logical_config


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


class DenseGradientAudit:
    """只钩实际反向边，保存CPU激活/dY；不写生产参数.grad或保存解码权重。"""

    def __init__(self, built, *, layers=(0, 3)):
        self.built, self.modules, self.records, self.base_records, self.hooks = built, {}, {}, {}, []
        for name, module in built.model.named_modules():
            if isinstance(module, LoRAPackedLinear) and module.base_layer.layer in layers:
                key = canonical_name(name)
                self.modules[key], self.records[key], self.base_records[key] = module, [], []

    def install(self):
        for name, module in self.modules.items():
            def wrapper_hook(current, inputs, output, key=name):
                if output.requires_grad:
                    value = inputs[0].detach()
                    def gradient(dy):
                        self.records[key].append((value.cpu(), dy.detach().cpu()))
                    output.register_hook(gradient)
            def base_hook(current, inputs, output, key=name):
                if output.grad_fn is not None and inputs[0].requires_grad:
                    value = inputs[0].detach()
                    def gradient(grad_inputs, grad_outputs):
                        self.base_records[key].append((value.cpu(), grad_outputs[0].detach().cpu(), grad_inputs[0].detach().cpu()))
                    output.grad_fn.register_hook(gradient)
            self.hooks.extend([module.register_forward_hook(wrapper_hook), module.base_layer.register_forward_hook(base_hook)])

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compare(self, *, expected_microbatches, on_module=None):
        """同形状普通autograd E2；独立FP32路径forward≤1%、梯度≤2%且cos≥.999。"""
        report = {}
        previous_phase = self.built.streamer.phase
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for name, module in self.modules.items():
                    records, base_records = self.records[name], self.base_records[name]
                    if len(records) != expected_microbatches or len(base_records) not in [0, expected_microbatches]:
                        raise ValueError(f"Actual dense backward edge count differs: {name}, {len(records)}, {len(base_records)}")
                    original = module.base_layer
                    cfg = original.op_config
                    with cfg.streamer.acquire(original.layer) as handle:
                        weight, scale = materialized_codes(handle.tensors(), original.name, torch.bfloat16)
                        weight32, scale32 = materialized_codes(handle.tensors(), original.name, torch.float32)
                    local = LoRAPackedLinear(PackedLinear(original.layer, original.name, original.kind,
                                            original.in_features, original.out_features, cfg), module.rank, module.alpha, device="cuda")
                    with torch.no_grad():
                        local.lora_A["default"].weight.copy_(module.lora_A["default"].weight)
                        local.lora_B["default"].weight.copy_(module.lora_B["default"].weight)
                    a = module.lora_A["default"].weight.detach().clone().requires_grad_()
                    b = module.lora_B["default"].weight.detach().clone().requires_grad_()
                    total_a = total_b = None
                    worst = {key: {"relative_l2": 0., "cosine": 1.} for key in ["forward", "dx", "dA", "dB"]}
                    e2 = {key: True for key in ["forward", "dx", "dA", "dB"]}
                    for index, (input_cpu, dy_cpu) in enumerate(records):
                        x = input_cpu.cuda().requires_grad_()
                        xr = input_cpu.cuda().requires_grad_()
                        x32 = input_cpu.float().cuda().requires_grad_()
                        dy = dy_cpu.cuda()
                        self.built.streamer.phase = "FORWARD"
                        output = local(x)
                        reference = ((F.linear(xr, weight).float() * scale).to(xr.dtype).float()
                                     + F.linear(F.linear(xr.float(), a), b) * module.scaling["default"]).to(xr.dtype)
                        precise = F.linear(x32, weight32) * scale32 + F.linear(F.linear(x32, a), b) * module.scaling["default"]
                        self.built.streamer.phase = "BACKWARD"
                        actual_gradients = torch.autograd.grad(output, [x, local.lora_A["default"].weight, local.lora_B["default"].weight], dy)
                        expected_gradients = torch.autograd.grad(reference, [xr, a, b], dy)
                        fp32_gradients = torch.autograd.grad(precise, [x32, a, b], dy.float())
                        for key, actual, expected, floating in zip(["forward", "dx", "dA", "dB"],
                                [output, *actual_gradients], [reference, *expected_gradients], [precise, *fp32_gradients]):
                            e2[key] &= same(actual, expected)
                            error = compare_vectors(actual, floating)
                            worst[key]["relative_l2"] = max(worst[key]["relative_l2"], error["relative_l2"])
                            worst[key]["cosine"] = min(worst[key]["cosine"], error["cosine"])
                        if index == 0:
                            total_a, total_b = [value.detach().clone() for value in expected_gradients[1:]]
                        else:
                            total_a.add_(expected_gradients[1])
                            total_b.add_(expected_gradients[2])
                        del x, xr, x32, dy, output, reference, precise, actual_gradients, expected_gradients, fp32_gradients
                    actual_base_dx_equal = True
                    for input_cpu, dy_cpu, dx_cpu in base_records:
                        x = input_cpu.cuda().requires_grad_()
                        output = (F.linear(x, weight).float() * scale).to(x.dtype)
                        dx, = torch.autograd.grad(output, x, dy_cpu.cuda())
                        actual_base_dx_equal &= same(dx.cpu(), dx_cpu)
                        del x, output, dx
                    accumulated = {"A": same(module.lora_A["default"].weight.grad, total_a),
                                   "B": same(module.lora_B["default"].weight.grad, total_b)}
                    e3 = all(value["relative_l2"] <= (.01 if key == "forward" else .02)
                             and (key == "forward" or value["cosine"] >= .999) for key, value in worst.items())
                    row = {"status": "PASS" if all(e2.values()) and e3 and all(accumulated.values()) and actual_base_dx_equal else "FAIL",
                           "backward_edges": len(records), "same_expression_E2": e2, "fp32_E3": worst,
                           "training_A_B_accumulated_E2": accumulated,
                           "actual_base_dx_edges": len(base_records), "actual_base_dx_E2": actual_base_dx_equal,
                           "dx_scope": "Actual PackedLinear backward edges plus isolated local replay. First-layer frozen embedding inputs need no actual dx edge; replay still checks their dx."}
                    report[name] = row
                    if on_module:
                        on_module(report)
                    print(f"real-activation gradient {name}: {row['status']}", flush=True)
                    del weight, weight32, scale, scale32, local, a, b, total_a, total_b
                    if row["status"] != "PASS":
                        raise AssertionError({name: row})
        finally:
            self.built.streamer.phase = previous_phase
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--restore-probe", type=Path)
    parser.add_argument("--commit", default="uncommitted")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if args.tiny != (manifest["source"].get("kind") == "tiny-generated"):
        raise ValueError("Dense probe tiny/source mismatch")
    configure_training(20261029)
    if args.tiny:
        dataset = SyntheticTinyDataset()
    else:
        if args.data is None:
            parser.error("Real dense probe requires --data")
        tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
        dataset = ChatJSONLDataset(args.data, tokenizer, max_length=2048)
    rank = 4 if args.tiny else 16
    build = {"family": "qwen3_5_dense", "lora_rank": rank, "lora_alpha": 2 * rank,
             "lora_scope": "D-ALL", "decode_chunk_elements": 8 * 1024 * 1024, "streaming": "async"}
    config = logical_config({"family": "qwen3_5_dense", "max_steps": 1, "rng_probe": True, "build": build}, args.bundle, dataset)
    save(args.output / "train_config.json", config)
    save(args.output / "execution.json", {"bundle": str(args.bundle), "data": str(args.data), "restore_probe": str(args.restore_probe), "scope": "one optimizer step only"})
    built = build_dense_model(args.bundle, **config["build"])
    trainer = LoRATrainer(built, config)
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    batches = stream.next_group(8)
    result = {"status": "RUNNING", "scope": "SMOKE ONLY" if args.tiny else "DA6.1: one actual optimizer step, real-activation math and gradient audits; not formal functional training",
              "base": base_identity(args.bundle), "dataset": dataset.identity, "memory": {"loaded": memory(built)}}
    path = args.output / "results.json"
    frozen = frozen_fingerprint(built)
    sample = dataset[0]
    try:
        if args.restore_probe is None:
            built.model.eval()
            with torch.no_grad():
                enabled = loss_sum(built, sample, ce_chunk_size=128, recompute=False) / supervised_tokens(sample["labels"])
                with disable_adapters(built.model):
                    disabled = loss_sum(built, sample, ce_chunk_size=128, recompute=False) / supervised_tokens(sample["labels"])
            result["zero_adapter_loss_E2"] = same(enabled, disabled)
            result["initial_probe"] = {"enabled": float(enabled), "disabled": float(disabled),
                                       "fp32_bits": int(enabled.view(torch.int32)), "disabled_fp32_bits": int(disabled.view(torch.int32))}
            assert result["zero_adapter_loss_E2"]
            save(path, result)
            built.model.train()
            torch.cuda.reset_peak_memory_stats()
            if not args.tiny:
                torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=2_000_000, clear_history=True)
            def first_micro(index, current):
                if index == 0:
                    result["memory"]["first_micro_backward"] = memory(built)
                    if not args.tiny:
                        torch.cuda.memory._dump_snapshot(str(args.output / "first-micro-backward.pickle"))
                        torch.cuda.memory._record_memory_history(enabled=None)
                print(f"dense step1 micro {index + 1}/8 backward complete", flush=True)
            start = time.perf_counter()
            metrics = trainer.backward_group(batches, on_microbatch=first_micro)
            result["all_first_B_gradients_nonzero"] = all(value.grad.count_nonzero() > 0 for name, value in built.trainable.items() if ".lora_B." in name)
            result["first_optimizer_step"] = {**trainer.finish_step(metrics), "wall_seconds_including_snapshot": time.perf_counter() - start}
            save_checkpoint(args.output / "checkpoint-step1", built.trainable, trainer.optimizer, trainer.scheduler,
                            stream.state_dict(), trainer.state, config, provenance={"code_commit": args.commit})
        else:
            origin = json.loads((args.restore_probe / "results.json").read_text())
            if json.loads((args.restore_probe / "train_config.json").read_text()) != config:
                raise ValueError("Restored probe configuration differs")
            data, trainer.state = load_checkpoint(args.restore_probe / "checkpoint-step1", built.trainable, trainer.optimizer, trainer.scheduler, config)
            for key in ["zero_adapter_loss_E2", "initial_probe", "all_first_B_gradients_nonzero", "first_optimizer_step"]:
                result[key] = origin[key]
            result["restored_step1_checkpoint"] = str(args.restore_probe / "checkpoint-step1")
        result["memory"]["after_step1_or_restore"] = memory(built)
        save(path, result)
        # B is nonzero. This backward group is diagnostic and never updates Adam.
        audit = DenseGradientAudit(built, layers=(0, 3))
        audit.install()
        try:
            trainer.backward_group(batches, on_microbatch=lambda index, current: print(f"gradient audit micro {index + 1}/8", flush=True))
        finally:
            audit.remove()
        torch.save({"wrapper_inputs_and_dy": audit.records, "base_dx_edges": audit.base_records}, args.output / "gradient-captures.pt")
        result["gradient_comparisons"] = audit.compare(expected_microbatches=8,
                            on_module=lambda rows: save(args.output / "gradient-comparisons.json", rows))
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.state["accumulation_position"] = 0
        built.model.eval()
        with torch.no_grad():
            updated = loss_sum(built, sample, ce_chunk_size=128, recompute=False) / supervised_tokens(sample["labels"])
            with disable_adapters(built.model):
                disabled_after = loss_sum(built, sample, ce_chunk_size=128, recompute=False) / supervised_tokens(sample["labels"])
        result["loss_probe"] = {"updated": float(updated), "disabled_after": float(disabled_after)}
        result["disabled_loss_unchanged_E2"] = int(disabled_after.view(torch.int32)) == result["initial_probe"]["disabled_fp32_bits"]
        result["frozen_fingerprint_unchanged"] = frozen_fingerprint(built) == frozen
        result["optimizer_steps"] = trainer.state["global_step"]
        result["trainable_tensors"] = len(built.trainable)
        result["trainable_numel"] = sum(value.numel() for value in built.trainable.values())
        publish_adapter(args.output / "publish", built, args.bundle, config["build"], config, code_commit=args.commit, global_step=1)
        result["memory"]["audit_and_publish"] = memory(built)
        result["status"] = "PASS" if (result["zero_adapter_loss_E2"] and result["all_first_B_gradients_nonzero"]
                and result["disabled_loss_unchanged_E2"] and result["frozen_fingerprint_unchanged"]
                and len(result["gradient_comparisons"]) == 13
                and all(row["status"] == "PASS" for row in result["gradient_comparisons"].values())) else "FAIL"
        save(path, result)
        assert result["status"] == "PASS", result.get("gradient_comparisons")
    except Exception:
        result.update(status="FAIL", traceback=traceback.format_exc())
        save(path, result)
        raise
    finally:
        built.streamer.close()


if __name__ == "__main__":
    main()
