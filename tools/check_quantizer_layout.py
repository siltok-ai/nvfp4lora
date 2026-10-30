"""真实一层诊断重数量化一致率；只写统计，不生成或改写训练专家权重。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from nvfp4lora.format.nvfp4 import dequantize, quantize
from nvfp4lora.format.bundle import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    entry = next(item for item in manifest["expert_layers"] if item["layer"] == args.layer)
    path = args.bundle / "experts" / entry["file"]
    before = file_hash(path)
    assert before == entry["sha256"]
    results = {}
    with safe_open(path, framework="pt", device="cpu") as reader:
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            packed = reader.get_tensor(projection + ".weight")
            scale = reader.get_tensor(projection + ".weight_scale")
            global_scale = reader.get_tensor(projection + ".weight_scale_2")
            totals = {"packed_bytes": 0, "packed_bytes_equal": 0, "codes": 0, "codes_equal": 0,
                      "blocks": 0, "block_scales_equal": 0, "blocks_without_magnitude_6": 0,
                      "zero_scale_blocks": 0, "nonzero_scale_codes": 0, "nonzero_scale_codes_equal": 0,
                      "code_mismatches_in_zero_scale_blocks": 0,
                      "redecoded_values_equal": 0, "values": 0, "relative_l2_squared_numerator": 0.,
                      "relative_l2_squared_denominator": 0.}
            for expert in range(packed.shape[0]):
                weight = dequantize(packed[expert], scale[expert], global_scale[expert], dtype=torch.float32)
                requantized = quantize(weight, global_scale[expert])
                actual = requantized["weight"]
                codes = torch.stack([packed[expert] & 15, packed[expert] >> 4], dim=-1).flatten()
                new_codes = torch.stack([actual & 15, actual >> 4], dim=-1).flatten()
                decoded = dequantize(actual, requantized["weight_scale"], global_scale[expert], dtype=torch.float32)
                totals["packed_bytes"] += actual.numel()
                totals["packed_bytes_equal"] += int((actual == packed[expert]).sum())
                totals["codes"] += codes.numel()
                totals["codes_equal"] += int((codes == new_codes).sum())
                totals["blocks"] += scale[expert].numel()
                totals["block_scales_equal"] += int((requantized["weight_scale"] == scale[expert].view(torch.uint8)).sum())
                totals["blocks_without_magnitude_6"] += int(((codes & 7).reshape(-1, 16).amax(-1) < 7).sum())
                nonzero = scale[expert].view(torch.float8_e4m3fn).float().flatten() != 0
                same = (codes == new_codes).reshape(-1, 16)
                totals["zero_scale_blocks"] += int((~nonzero).sum())
                totals["nonzero_scale_codes"] += int(nonzero.sum()) * 16
                totals["nonzero_scale_codes_equal"] += int(same[nonzero].sum())
                totals["code_mismatches_in_zero_scale_blocks"] += int((~same[~nonzero]).sum())
                totals["values"] += weight.numel()
                totals["redecoded_values_equal"] += int((weight == decoded).sum())
                totals["relative_l2_squared_numerator"] += float((weight.double() - decoded.double()).square().sum())
                totals["relative_l2_squared_denominator"] += float(weight.double().square().sum())
            results[projection] = {**totals,
                "code_match_rate": totals["codes_equal"] / totals["codes"],
                "block_scale_match_rate": totals["block_scales_equal"] / totals["blocks"],
                "decoded_match_rate": totals["redecoded_values_equal"] / totals["values"],
                "nonzero_scale_code_match_rate": totals["nonzero_scale_codes_equal"] / max(1, totals["nonzero_scale_codes"]),
                "decoded_relative_l2": (totals["relative_l2_squared_numerator"] / totals["relative_l2_squared_denominator"]) ** .5}
            print(projection, results[projection]["code_match_rate"], results[projection]["block_scale_match_rate"], flush=True)
    unchanged = file_hash(path) == before
    assert unchanged
    summary = {"status": "DIAGNOSTIC_COMPLETE", "layer": args.layer, "source": manifest["source"],
               "input_sha256": before, "input_unchanged": unchanged,
               "policy": "FP32 decode then quantize using original FP32 global; temporary values only; not an exact-match gate",
               "projections": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
