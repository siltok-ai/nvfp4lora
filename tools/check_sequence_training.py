"""审核真实长序列压力会话：完整答案、两步累积8、Adam稳态和反向解码生命周期。"""

import argparse
import json
from pathlib import Path

from nvfp4lora.format.bundle import file_hash
from summarize_training_session import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--snapshot-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    observation = summarize(args.results)
    config = json.loads((args.results / "train_config.json").read_text(encoding="utf-8"))
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot_audit.read_text(encoding="utf-8"))
    length = config["max_length"]
    geometry = fixture["geometry"][str(length)]
    checks = {
        "fresh_two_optimizer_steps": observation["starting_step"] == 0 and observation["current_step"] == 2,
        "fixture_identity": fixture["status"] == "PASS" and config["dataset"]["file_sha256"] == fixture["file_sha256"],
        "all_eight_samples_retained": len(config["dataset"]["accepted_rows"]) == 8 and not config["dataset"]["discarded_answer_too_long"],
        "accumulation_eight": config["accumulation_steps"] == 8,
        "checkpointing_and_chunked_ce": config["gradient_checkpointing"] and config["ce_chunk_size"] < min(row["supervised_tokens"] for row in geometry),
        "exact_sequence_and_supervision_counts": all(row["sequence_tokens"] == length * 8 and
            row["supervised_tokens"] == sum(sample["supervised_tokens"] for sample in geometry)
            for row in observation["observations"]),
        "adam_present_after_first_step": all(row["memory"]["inventory"]["adam_gpu_state"] == 8 * observation["trainable_numel"]
            for row in observation["observations"]),
        "frozen_hashes_unchanged": observation["frozen_hashes_unchanged"],
        "decoded_allocations_released_after_backward": snapshot["status"] == "PASS" and snapshot["decode_allocation_count"] > 0 and snapshot["active_decode_allocation_count"] == 0,
        "snapshot_identity": snapshot["snapshot_sha256"] == file_hash(args.results / "first-micro-backward.pickle"),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "scope": "Long-sequence training mechanics and memory; not model quality or P8 certification",
              "checks": checks, "fixture_manifest_sha256": file_hash(args.fixture_manifest),
              "snapshot_audit_sha256": file_hash(args.snapshot_audit), "observation": observation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks, "max_length": length,
                      "expert_rank": config["build"]["expert_rank"],
                      "peak_allocated": max(row["memory"]["gpu_peak_allocated"] for row in observation["observations"]),
                      "peak_reserved": max(row["memory"]["gpu_peak_reserved"] for row in observation["observations"])}, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
