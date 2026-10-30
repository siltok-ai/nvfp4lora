"""真实预处理的逐层 E1 文件读回、可选 FreeToken 数值和 meta 参数目录对照。"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors import safe_open
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from nvfp4lora.format.nvfp4 import dequantize
from nvfp4lora.format.bundle import FIELDS, file_hash, load_group
from nvfp4lora.format.preprocess_moe import PROJECTIONS

from check_freetoken_dequant import ulp_distance


def main():
    """从磁盘重读数据，不用预处理时内存相等冒充文件 E1 验收。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freetoken", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    kernel = None
    if args.freetoken:
        from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4
        kernel = _dequant_nvfp4
    manifest = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise ValueError("Preprocessing is not complete")
    source_map = json.loads((args.source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    expert_index = json.loads((args.processed / "experts/index.json").read_text(encoding="utf-8"))
    result = {"status": "RUNNING", "started_at": time.time(), "layers": [],
              "numeric_reference": "FreeToken raw FP32 global" if kernel else "NOT RUN",
              "A1.4_scope": "meta directory only; materialized PackedExperts model still required"}
    result_path = args.output / "results.json"
    total = 0
    for layer in expert_index["layers"]:
        layer_id = layer["layer"]
        path = args.processed / "experts" / layer["file"]
        if file_hash(path) != layer["sha256"]:
            raise ValueError(f"Processed file checksum mismatch: {path}")
        sampled = manifest["e1_samples"][layer_id]["experts"]
        names = [f"model.language_model.layers.{layer_id}.mlp.experts.{e}.{p}.{f}"
                 for e in sampled for p in PROJECTIONS for f in FIELDS]
        original = load_group(args.source, source_map, names)
        numeric = []
        with safe_open(path, framework="pt", device="cpu") as saved:
            for expert in sampled:
                for projection in PROJECTIONS:
                    prefix = f"model.language_model.layers.{layer_id}.mlp.experts.{expert}.{projection}"
                    for field in FIELDS:
                        actual = saved.get_tensor(f"{projection}.{field}")[expert].contiguous().reshape(-1).view(torch.uint8)
                        expected = original[f"{prefix}.{field}"].contiguous().reshape(-1).view(torch.uint8)
                        if not torch.equal(actual, expected):
                            raise ValueError(f"E1 file mismatch: {prefix}.{field}")
                        total += 1
                    if kernel is not None:
                        weight = original[f"{prefix}.weight"]
                        scale = original[f"{prefix}.weight_scale"]
                        scalar = original[f"{prefix}.weight_scale_2"]
                        reference = dequantize(weight, scale, scalar, dtype=torch.bfloat16)
                        actual = kernel(weight, scale, scalar.reshape(()).expand(weight.shape[0]).contiguous())
                        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
                            raise ValueError(f"Non-finite dequantized expert: {prefix}")
                        maximum = ulp_distance(actual, reference)
                        if maximum > 1:
                            raise ValueError(f"A1.3 exceeds 1 ULP: {prefix}: {maximum}")
                        numeric.append({"expert": expert, "projection": projection, "max_ulp": maximum})
        result["layers"].append({"layer": layer_id, "sampled_experts": sampled, "E1": "PASS", "numeric": numeric})
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"layer {layer_id}: E1 PASS, numeric {'PASS' if kernel else 'NOT RUN'}", flush=True)
        del original
    config = Qwen3_5MoeTextConfig.from_dict(json.loads((args.processed / "config.json").read_text(encoding="utf-8")))
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
    expected = {name: list(p.shape) for name, p in model.named_parameters() if ".mlp.experts." not in name}
    actual = {}
    for entry in manifest["nonexpert_files"]:
        path = args.processed / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Nonexpert checksum mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                actual[name] = reader.get_slice(name).get_shape()
    if actual != expected:
        missing, extra = expected.keys() - actual.keys(), actual.keys() - expected.keys()
        wrong = {n: (expected[n], actual[n]) for n in expected.keys() & actual.keys() if expected[n] != actual[n]}
        raise ValueError(f"Nonexpert parameter catalog mismatch: missing={missing}, extra={extra}, shapes={wrong}")
    result.update(status="PASS", ended_at=time.time(), e1_fields_compared=total,
                  nonexpert_parameter_names=len(expected), nonexpert_parameter_shapes="PASS",
                  expert_payload_bytes=sum(x["payload_bytes"] for x in expert_index["layers"]),
                  preprocessing_peak_working_set_bytes=manifest["peak_working_set_bytes"])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
