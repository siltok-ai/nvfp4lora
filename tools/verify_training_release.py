"""P8发布重建：新进程只用bundle/publish建模，检查原训练固定probe和全量ΔW指纹。"""

import argparse
import json
from pathlib import Path

import torch

from nvfp4lora.export.artifacts import _artifact_tensors, build_from_adapter
from nvfp4lora.export.delta_w import delta_w_fingerprints
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.evaluate import evaluation_mode
from nvfp4lora.train.loop import configure_training, loss_sum, memory_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    source = args.training_results
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((source / "train_config.json").read_text(encoding="utf-8"))
    expected = original["published_probe"]
    built = build_from_adapter(args.bundle, source / "publish")
    batch = {key: torch.tensor(expected[key]) for key in ["input_ids", "labels"]}
    with evaluation_mode(built):
        enabled = loss_sum(built, batch, ce_chunk_size=config["ce_chunk_size"], recompute=False) / supervised_tokens(batch["labels"])
    with evaluation_mode(built, disabled=True):
        disabled = loss_sum(built, batch, ce_chunk_size=config["ce_chunk_size"], recompute=False) / supervised_tokens(batch["labels"])
    checks = {"fixed_batch_loss_E2": int(enabled.view(torch.int32)) == expected["loss_fp32_bits"],
              "disabled_loss_E2": int(disabled.view(torch.int32)) == expected["disabled_loss_fp32_bits"]}
    values, _ = _artifact_tensors(source / "publish")
    checks["all_parameter_bytes_equal"] = values.keys() == built.trainable.keys() and all(
        torch.equal(parameter.detach().cpu().contiguous().view(torch.uint8), values[name].contiguous().view(torch.uint8))
        for name, parameter in built.trainable.items())
    expected_delta = json.loads((source / "delta-w.json").read_text(encoding="utf-8"))
    actual_delta = delta_w_fingerprints(built, expert_chunk=expected_delta["expert_chunk"])
    checks["all_delta_W_bytes_E2"] = actual_delta == expected_delta
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "delta_W_fingerprint_count": len(actual_delta["tensors"]), "loss": float(enabled), "disabled_loss": float(disabled),
              "memory": memory_observation(), "build_inputs": [str(args.bundle), str(source / "publish")]}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    built.streamer.close()
    print(json.dumps(result, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
