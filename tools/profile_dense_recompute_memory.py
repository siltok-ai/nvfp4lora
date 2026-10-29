"""DA6.2：恢复真实Adam后，一micro的64层重算峰值及无仪器E2控制。"""

import argparse
import json
from pathlib import Path
import pickle
import struct

import torch
from transformers import AutoTokenizer

from profile_recompute_memory import RecomputePeaks
from nvfp4lora.model.build_dense import build_dense_model
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.checkpoint import load_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, live_inventory, loss_sum, memory_observation
from nvfp4lora.train.session import logical_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.training_results / "train_config.json").read_text())
    execution = json.loads((args.training_results / "execution.json").read_text())
    if settings.get("family") != "qwen3_5_dense":
        raise ValueError("Dense recompute profiler requires dense training results")
    configure_training(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(execution["data"], tokenizer, max_length=settings["max_length"])
    config = logical_config(settings, args.bundle, dataset)
    if config != settings:
        raise ValueError("Dense pressure checkpoint configuration differs")
    built = build_dense_model(args.bundle, **config["build"])
    trainer = LoRATrainer(built, config)
    checkpoint = args.training_results / "checkpoints/step-000002"
    data, trainer.state = load_checkpoint(checkpoint, built.trainable, trainer.optimizer, trainer.scheduler, config)
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    stream.load_state_dict(data)
    sample = stream.next_group(1)[0]
    frozen = frozen_fingerprint(built)
    observations = {"restored_adam": memory_observation(), "inventory": live_inventory(built, trainer.optimizer)}
    def backward():
        built.streamer.phase = "FORWARD"
        value = loss_sum(built, sample, ce_chunk_size=config["ce_chunk_size"])
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite dense recompute loss")
        built.streamer.phase = "BACKWARD"
        (value / supervised_tokens(sample["labels"])).backward()
        torch.cuda.synchronize()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in built.trainable.values()):
            raise FloatingPointError("Missing/nonfinite dense recompute gradient")
        return float(value.detach())
    try:
        torch.cuda.reset_peak_memory_stats()
        limit = 2_000_000
        torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=limit, clear_history=True)
        peaks = RecomputePeaks(built)
        loss = backward()
        observations.update(peaks.finish())
        snapshot_path = args.output / "micro-backward.pickle"
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        with snapshot_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        trace_lengths = [len(trace) for trace in snapshot["device_traces"]]
        del snapshot
        checks = {"all_64_layers_recomputed": sorted(row["layer"] for row in observations["recomputations"]) == list(range(64)),
                  "adam_present": observations["inventory"]["adam_gpu_state"] == 8 * sum(parameter.numel() for parameter in built.trainable.values()),
                  "history_complete": all(length < limit for length in trace_lengths)}
        if args.control:
            gradients = {name: parameter.grad.detach().cpu().contiguous() for name, parameter in built.trainable.items()}
            trainer.optimizer.zero_grad(set_to_none=True)
            control_loss = backward()
            checks["uninstrumented_loss_E2"] = struct.pack("f", loss) == struct.pack("f", control_loss)
            checks["all_uninstrumented_gradients_E2"] = all(torch.equal(parameter.grad.detach().cpu().contiguous().view(torch.uint8),
                                                        gradients[name].view(torch.uint8)) for name, parameter in built.trainable.items())
        checks["frozen_fingerprint_unchanged"] = frozen_fingerprint(built) == frozen
        result = {"status": "PASS" if all(checks.values()) else "FAIL", "scope": "One steady-state micro with restored step2 Adam; no optimizer update",
                  "checks": checks, "max_length": config["max_length"], "input_tokens": sample["input_ids"].numel(),
                  "supervised_tokens": supervised_tokens(sample["labels"]), "history_limit": limit,
                  "history_trace_lengths": trace_lengths, "loss_sum": loss, "memory": observations}
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "checks": checks,
                         "whole_micro_peak_allocated": observations["whole_micro_peak_allocated"],
                         "whole_micro_peak_reserved": observations["whole_micro_peak_reserved"]}, indent=2), flush=True)
        assert result["status"] == "PASS", checks
    finally:
        built.streamer.close()


if __name__ == "__main__":
    main()
