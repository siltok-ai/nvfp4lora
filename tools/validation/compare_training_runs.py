"""相同配置的同步/异步训练逐步 E2 对照，排除计时和搬运计数。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def equal_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        return isinstance(expected, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    if isinstance(actual, np.ndarray):
        return isinstance(expected, np.ndarray) and np.array_equal(actual, expected)
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(equal_tree(value, expected[key]) for key, value in actual.items())
    if isinstance(actual, (tuple, list)):
        return len(actual) == len(expected) and all(equal_tree(a, b) for a, b in zip(actual, expected))
    return actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-run", required=True, type=Path)
    parser.add_argument("--async-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checks, differences = {}, []
    configs = []
    for root in [args.sync_run, args.async_run]:
        config = json.loads((root / "train_config.json").read_text())
        config["build"].pop("streaming", None)
        configs.append(config)
    checks["all_logical_config_except_streaming"] = configs[0] == configs[1]
    histories = [[json.loads(line) for line in (root / "paired" / "metrics.jsonl").read_text().splitlines()]
                 for root in [args.sync_run, args.async_run]]
    checks["step_count"] = len(histories[0]) == len(histories[1]) == 100
    for index, (expected, actual) in enumerate(zip(*histories)):
        for key in ["loss", "supervised_tokens", "sample_ids", "learning_rates", "gradient_norm", "group_gradient_norms", "rng_probe"]:
            if actual[key] != expected[key]:
                differences.append({"step": index + 1, "field": key, "sync": expected[key], "async": actual[key]})
    checks["all_100_steps_E2"] = not differences
    for boundary in ["checkpoint-middle", "checkpoint-final"]:
        roots = [root / "paired" / boundary for root in [args.sync_run, args.async_run]]
        checks[boundary + ".parameters"] = equal_tree(*[load_file(root / "lora_params.safetensors") for root in roots])
        for name in ["optimizer.pt", "scheduler.pt", "rng.pt"]:
            checks[boundary + "." + name] = equal_tree(*[torch.load(root / name, weights_only=False) for root in roots])
        for name in ["data_state.json", "trainer_state.json"]:
            checks[boundary + "." + name] = equal_tree(*[json.loads((root / name).read_text()) for root in roots])
    checks["probe_parameters_delta_loss"] = equal_tree(*[torch.load(root / "paired" / "probe.pt", weights_only=False)
                                                         for root in [args.sync_run, args.async_run]])
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "differences": differences,
              "sync_run": str(args.sync_run), "async_run": str(args.async_run)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "differences"}, indent=2))
    assert result["status"] == "PASS", differences[:4]


if __name__ == "__main__":
    main()
