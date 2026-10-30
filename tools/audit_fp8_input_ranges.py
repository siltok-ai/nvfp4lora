"""v2风险观测：实际数据上统计FP8来源层 max|x|/(448*原始input_scale)，不改训练数学。"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import build_from_adapter
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.model.common import causal_model
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.evaluate import evaluation_mode, padded_batch
from nvfp4lora.train.loop import configure_training


class DenseFP8InputAudit:
    """从实际PackedLinear输入观察原始FP8尺度；仅挂钩，不量化或改变激活。"""

    def __init__(self, built):
        from nvfp4lora.model.packed_linear import PackedLinear
        self.records, self.phase, self.mask, self.hooks = {}, None, None, []
        self.scales = {}
        for module in built.model.modules():
            if not isinstance(module, PackedLinear) or module.kind != "fp8":
                continue
            name = f"model.layers.{module.layer}.{module.name}"
            scale = float(built.store.tensors(module.layer)[module.name + ".input_scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid dense FP8 input scale: {name}")
            self.scales[name] = scale
            def capture(current, inputs, key=name):
                value = inputs[0].detach()
                if self.mask is not None:
                    if value.shape[:2] != self.mask.shape:
                        raise ValueError(f"Dense FP8 mask/input mismatch: {key}")
                    value = value[self.mask]
                record = self.records[self.phase][key]
                maximum = float(value.abs().amax())
                if not math.isfinite(maximum):
                    raise FloatingPointError(f"Nonfinite dense FP8 source activation: {key}")
                record["max_abs_input"] = max(record["max_abs_input"], maximum)
                record["calls"] += 1
                record["valid_token_vectors"] += value.numel() // value.shape[-1]
            self.hooks.append(module.register_forward_pre_hook(capture))
        expected = sum(spec["kind"] == "fp8" for modules in built.store.geometry["modules"].values() for spec in modules.values())
        if len(self.scales) != expected:
            raise ValueError("Dense FP8 source projection coverage differs from store geometry")

    def begin(self, phase, *, mask=None):
        self.phase, self.mask = phase, mask
        self.records[phase] = {name: {"max_abs_input": 0., "calls": 0, "valid_token_vectors": 0,
                                      "original_input_scale": scale} for name, scale in self.scales.items()}

    def finish(self):
        for phases in self.records.values():
            for record in phases.values():
                if record["calls"] == 0:
                    raise ValueError("Dense FP8 source module was not observed")
                record["ratio"] = record["max_abs_input"] / (448 * record["original_input_scale"])
        return {"status": "OBSERVED", "fp8_modules": len(self.scales), "phases": self.records,
                "above_original_range": {phase: [name for name, value in rows.items() if value["ratio"] > 1]
                                         for phase, rows in self.records.items()},
                "scope": "Input range only; no W8A8 simulation or inference-equivalence claim"}

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def audit_dense(args, manifest):
    """Dense按同一语料观察基座及可选adapter；source全文件校验保持原约束。"""
    from nvfp4lora.model.build_dense import build_dense_model
    for shard in manifest["source"]["shards"]:
        if file_hash(args.source / shard["filename"]) != shard["sha256"]:
            raise ValueError(f"Dense FP8 source checksum mismatch: {shard['filename']}")
    configure_training(20261029)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(args.data, tokenizer, max_length=2048)
    built = build_from_adapter(args.bundle, args.adapter) if args.adapter else build_dense_model(args.bundle)
    audit = DenseFP8InputAudit(built)
    base = causal_model(built.model)
    device = next(base.parameters()).device
    try:
        for disabled, phase in [(True, "base_disabled"), *([(False, "adapter_enabled")] if args.adapter else [])]:
            audit.begin(phase)
            with evaluation_mode(built, disabled=disabled):
                for start in range(0, len(dataset), args.batch_size):
                    samples = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
                    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    ids, labels, mask, positions = padded_batch(samples, device, pad_token_id=pad)
                    audit.mask = mask.bool()
                    hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
                    if not torch.isfinite(hidden).all():
                        raise FloatingPointError("Nonfinite dense FP8 range observation forward")
                    print(f"{phase}: {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
        result = {**audit.finish(), "dataset": dataset.identity, "family": "qwen3_5_dense"}
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        audit.close()
        built.streamer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("family") == "qwen3_5_dense":
        audit_dense(args, manifest)
        return
    if args.adapter is None:
        parser.error("MoE FP8 audit requires --adapter")
    for shard in manifest["source"]["shards"]:
        assert file_hash(args.source / shard["filename"]) == shard["sha256"], shard["filename"]
    layout = json.loads((args.bundle / "source-layout.json").read_text(encoding="utf-8"))["tensors"]
    names = [name.removesuffix(".weight") for name, spec in layout.items()
             if name.startswith("model.language_model.layers.") and name.endswith(".weight") and spec["dtype"].startswith("F8")]
    scales = {}
    for name in names:
        spec = layout[name + ".input_scale"]
        with safe_open(args.source / spec["file"], framework="pt", device="cpu") as reader:
            scale = float(reader.get_tensor(name + ".input_scale"))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid original FP8 input_scale: {name}")
        scales[name.replace("model.language_model.", "model.", 1)] = scale
    assert len(scales) == manifest["geometry"]["nonexpert_fp8_modules"]
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(args.data, tokenizer, max_length=2048)
    built = build_from_adapter(args.bundle, args.adapter)
    base = causal_model(built.model)
    device = next(base.parameters()).device
    observation = {"phase": None, "mask": None}
    results, hooks = {}, []
    for name in scales:
        module = base.get_submodule(name)
        def capture(module, inputs, key=name):
            values = inputs[0].detach()
            mask = observation["mask"]
            if values.shape[:2] != mask.shape:
                raise ValueError(f"Unexpected FP8 input/mask shape: {key}")
            valid = values[mask]
            record = results[observation["phase"]][key]
            record["max_abs_input"] = max(record["max_abs_input"], float(valid.abs().max()))
            record["calls"] += 1
            record["valid_token_vectors"] += valid.shape[0]
        hooks.append(module.register_forward_pre_hook(capture))
    for disabled, phase in [(True, "base_disabled"), (False, "adapter_enabled")]:
        observation["phase"] = phase
        results[phase] = {name: {"max_abs_input": 0., "calls": 0, "valid_token_vectors": 0, "original_input_scale": scale}
                          for name, scale in scales.items()}
        with evaluation_mode(built, disabled=disabled):
            for start in range(0, len(dataset), args.batch_size):
                samples = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
                ids, labels, mask, positions = padded_batch(samples, device, pad_token_id=248044)
                observation["mask"] = mask.bool()
                hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
                assert torch.isfinite(hidden).all()
                print(f"{phase}: {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
        for record in results[phase].values():
            assert record["calls"] > 0
            record["ratio"] = record["max_abs_input"] / (448 * record["original_input_scale"])
    for hook in hooks:
        hook.remove()
    result = {"status": "OBSERVED", "scope": "Input-range risk observation, not a W8A8 inference-equivalence claim",
              "dataset": dataset.identity, "fp8_modules": len(scales), "phases": results,
              "above_original_range": {phase: [name for name, record in records.items() if record["ratio"] > 1]
                                       for phase, records in results.items()}}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    built.streamer.close()


if __name__ == "__main__":
    main()
