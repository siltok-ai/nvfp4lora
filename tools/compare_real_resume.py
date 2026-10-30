"""真实基线/恢复执行的逐步与完整命名状态E2比较，不用zip/pickle文件字节代替语义。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from compare_training_runs import equal_tree
from nvfp4lora.format.bundle import file_hash


def read_checkpoint(path):
    manifest = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert all(file_hash(path / name) == digest for name, digest in manifest["files"].items())
    return {"parameters": load_file(path / "lora_params.safetensors"),
            **{name: torch.load(path / name, map_location="cpu", weights_only=False) for name in ["optimizer.pt", "scheduler.pt", "rng.pt"]},
            **{name: json.loads((path / name).read_text(encoding="utf-8")) for name in ["data_state.json", "trainer_state.json", "train_config.json"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--resumed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summaries = [json.loads((root / "summary.json").read_text(encoding="utf-8")) for root in [args.baseline, args.resumed]]
    assert all(summary["status"] == "PASS" for summary in summaries)
    endpoint = summaries[0]["current_step"]
    assert endpoint == summaries[1]["current_step"] > summaries[1]["starting_step"]
    reference_rows = {row["global_step"]: row for row in [json.loads(line) for line in (args.baseline / "metrics.jsonl").read_text().splitlines()]}
    resumed_rows = [json.loads(line) for line in (args.resumed / "metrics.jsonl").read_text().splitlines()]
    differences = []
    keys = ["loss", "supervised_tokens", "sequence_tokens", "sample_ids", "learning_rates", "gradient_norm", "group_gradient_norms", "route_audit", "rng_probe"]
    for row in resumed_rows:
        for key in keys:
            if row.get(key) != reference_rows[row["global_step"]].get(key):
                differences.append({"step": row["global_step"], "field": key})
    assert len(resumed_rows) == endpoint - summaries[1]["starting_step"]
    paths = [root / "checkpoints" / f"step-{endpoint:06d}" for root in [args.baseline, args.resumed]]
    expected, actual = (read_checkpoint(path) for path in paths)
    checks = {name: equal_tree(expected[name], actual[name]) for name in expected}
    checks["per_step_E2"] = not differences
    checks["published_probe_E2"] = equal_tree(summaries[0]["published_probe"], summaries[1]["published_probe"])
    if (args.baseline / "delta-w.json").exists() and (args.resumed / "delta-w.json").exists():
        checks["training_memory_delta_W_E2"] = json.loads((args.baseline / "delta-w.json").read_text()) == json.loads((args.resumed / "delta-w.json").read_text())
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "differences": differences,
              "starting_step": summaries[1]["starting_step"], "endpoint": endpoint, "compared_steps": len(resumed_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
