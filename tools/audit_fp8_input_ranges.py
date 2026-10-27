"""v2风险观测：实际数据上统计FP8来源层 max|x|/(448*原始input_scale)，不改训练数学。"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import build_from_adapter
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.model.build import causal_model
from nvfp4lora.train.data import ChatJSONLDataset
from nvfp4lora.train.evaluate import evaluation_mode, padded_batch
from nvfp4lora.train.loop import configure_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
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
