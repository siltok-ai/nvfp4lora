"""保留原FP32失败结果，分解逐层累积、top-k离散变化和BF16舍入；不改生产实现。"""

import argparse
import json
from pathlib import Path

import torch

from nvfp4lora.store.expert_store import ExpertStore
from nvfp4lora.validation.cpu_layers import reference_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    store = ExpertStore(args.bundle / "experts", pin_memory=False)
    gpu = torch.load(args.capture / "gpu-captures.pt", weights_only=True)
    ids = torch.tensor(json.loads((args.capture / "input.json").read_text(encoding="utf-8"))["used_ids"])
    teacher = [None, *gpu["hidden_states"][:-1]]
    modes = {"fp32_end_to_end": {}, "fp32_same_layer_input": {"teacher_inputs": teacher},
             "fp32_same_input_gpu_routes": {"teacher_inputs": teacher, "forced_routes": gpu["routes"]},
             "fp32_end_to_end_gpu_routes": {"forced_routes": gpu["routes"]},
             "bf16_same_policy_end_to_end": {"compute_dtype": torch.bfloat16}}
    result = {"status": "DIAGNOSTIC", "cases": {}}
    for name, kwargs in modes.items():
        routes = []
        values = reference_layers(args.bundle, store, ids, layers=4, traces=routes, **kwargs)
        rows = []
        for index, (actual, reference) in enumerate(zip(gpu["hidden_states"], values)):
            a, b = actual.double(), reference.double()
            relative = float((a - b).norm() / b.norm())
            ga, ca = gpu["routes"][index]["indices"].sort(-1).values, routes[index]["indices"].sort(-1).values
            different = (ga != ca).any(-1)
            row = {"layer": index, "relative_l2": relative, "tokens_with_different_topk_set": int(different.sum()),
                   "max_abs": float((a - b).abs().max())}
            if (~different).any():
                row["same_topk_tokens_relative_l2"] = float((a[:, ~different] - b[:, ~different]).norm() / b[:, ~different].norm())
            if different.any():
                row["different_topk_tokens_relative_l2"] = float((a[:, different] - b[:, different]).norm() / b[:, different].norm())
            rows.append(row)
        result["cases"][name] = rows
        (args.output / "precision-diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, rows, flush=True)
    print("Original FP32 acceptance is unchanged; these are diagnostic controls only.", flush=True)


if __name__ == "__main__":
    main()
