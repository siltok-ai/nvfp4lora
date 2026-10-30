"""从训练会话原始证据生成小型观测摘要；不把执行通过等同于效果验收。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from nvfp4lora.format.bundle import file_hash
from nvfp4lora.families import family_for_config


def summarize(results):
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((results / "train_config.json").read_text(encoding="utf-8"))
    family = family_for_config(config)
    execution = json.loads((results / "execution.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (results / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["status"] == "PASS" and summary["frozen_hashes_unchanged"]
    assert metrics and len(metrics) == summary["current_step"] - summary["starting_step"]
    rows = []
    for item in metrics:
        assert math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
        assert family.audit_passed(item["route_audit"]), "Invalid family audit report"
        rows.append({key: item[key] for key in ["global_step", "loss", "supervised_tokens", "sequence_tokens",
                                               "gradient_norm", "step_wall_seconds", "backward_seconds", "optimizer_seconds"]}
                    | {"sequence_tokens_per_second": item["sequence_tokens"] / item["step_wall_seconds"],
                       "transfers": item["transfers"], "memory": {key: value for key, value in item["memory"].items()
                                                                 if key != "pinned_allocator"},
                       "routing": {key: value for key, value in item["route_audit"].items() if key != "routing_counts"}})
    norms = {}
    for checkpoint in summary["checkpoints"]:
        path = Path(checkpoint) / "lora_params.safetensors"
        squared = {}
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                group = family.parameter_label(name) + ("_A" if ".lora_A" in name else "_B")
                squared[group] = squared.get(group, 0.) + float(reader.get_tensor(name).double().square().sum())
        norms[Path(checkpoint).name] = {key: math.sqrt(value) for key, value in squared.items()}
    return {"status": "MEASURED", "execution_status": summary["status"], "results": str(results.resolve()),
            "source_evidence_sha256": {name: file_hash(results / name) for name in
                                       ["summary.json", "train_config.json", "execution.json", "metrics.jsonl"]},
            "starting_step": summary["starting_step"], "current_step": summary["current_step"],
            "logical_max_steps": config["max_steps"], "max_length": config["max_length"], "build": config["build"],
            "accumulation_steps": config["accumulation_steps"], "ce_chunk_size": config["ce_chunk_size"],
            "trainable_tensors": summary["trainable_tensors"], "trainable_numel": summary["trainable_numel"],
            "resume": execution["resume"], "frozen_hashes_unchanged": summary["frozen_hashes_unchanged"],
            "observations": rows, "stage_memory": summary["memory"], "checkpoint_group_l2_norms": norms,
            "evaluations": {key: value for key, value in summary.items() if key.startswith("evaluation_step_") or key == "baseline_evaluation"},
            "notes": ["CPU working/private fields are observations, not interval peaks.",
                      "backward_seconds includes forward, CE and backward; optimizer_seconds excludes route auditing.",
                      "H2D and wait CUDA event intervals may overlap compute and must not be summed with wall time.",
                      "Execution success alone does not certify canary effects or the full v2 acceptance gates."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "starting_step", "current_step", "max_length", "trainable_numel"]}, indent=2))


if __name__ == "__main__":
    main()
