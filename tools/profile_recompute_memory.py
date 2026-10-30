"""从真实压力检查点恢复Adam，独立测一个micro的逐层重算峰值；不更新或发布参数。"""

import argparse
import json
from pathlib import Path
import pickle
import struct

import torch
from transformers import AutoTokenizer

from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.train.checkpoint import load_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, live_inventory, loss_sum, memory_observation
from nvfp4lora.train.session import logical_config


class RecomputePeaks:
    """always_call后hook覆盖非重入检查点的提前停止；各次reset前收集原峰值。"""

    def __init__(self, built):
        self.built, self.layers, self.intervals, self.pending = built, [], [], {}
        self.hooks = []
        for index, layer in enumerate(causal_model(built.model).model.layers):
            self.hooks.append(layer.register_forward_pre_hook(self.begin))
            self.hooks.append(layer.register_forward_hook(self.end, always_call=True))
            self.layers.append((id(layer), index))
        self.names = dict(self.layers)
        self.recomputations = []

    def begin(self, module, inputs):
        if self.built.streamer.phase == "BACKWARD":
            self.intervals.append(memory_observation())
            torch.cuda.reset_peak_memory_stats()
            self.pending[id(module)] = memory_observation()

    def end(self, module, inputs, output):
        start = self.pending.pop(id(module), None)
        if start is not None:
            self.recomputations.append({"layer": self.names[id(module)], "start": start, "end": memory_observation(),
                                        "nonreentrant_early_stop": output is None})

    def finish(self):
        self.intervals.append(memory_observation())
        assert not self.pending
        for hook in self.hooks:
            hook.remove()
        return {"whole_micro_peak_allocated": max(row["gpu_peak_allocated"] for row in self.intervals),
                "whole_micro_peak_reserved": max(row["gpu_peak_reserved"] for row in self.intervals),
                "layer_recompute_peak_allocated": max(row["end"]["gpu_peak_allocated"] for row in self.recomputations),
                "layer_recompute_peak_reserved": max(row["end"]["gpu_peak_reserved"] for row in self.recomputations),
                "recomputations": self.recomputations, "intervals_before_each_reset_and_final": self.intervals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.training_results / "train_config.json").read_text(encoding="utf-8"))
    execution = json.loads((args.training_results / "execution.json").read_text(encoding="utf-8"))
    configure_training(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    dataset = ChatJSONLDataset(execution["data"], tokenizer, max_length=settings["max_length"])
    config = logical_config(settings, args.bundle, dataset)
    assert config == settings
    built = build_model(args.bundle, **config["build"])
    trainer = LoRATrainer(built, config)
    checkpoint = args.training_results / "checkpoints" / "step-000002"
    data_state, trainer.state = load_checkpoint(checkpoint, built.trainable, trainer.optimizer, trainer.scheduler, config)
    stream = DeterministicSampleStream(dataset, seed=config["sampler_seed"])
    stream.load_state_dict(data_state)
    sample = stream.next_group(1)[0]
    frozen = frozen_fingerprint(built)
    observations = {"restored_adam": memory_observation(), "inventory": live_inventory(built, trainer.optimizer)}

    def backward():
        built.streamer.phase = "FORWARD"
        value = loss_sum(built, sample, ce_chunk_size=config["ce_chunk_size"])
        assert torch.isfinite(value)
        built.streamer.phase = "BACKWARD"
        (value / supervised_tokens(sample["labels"])).backward()
        torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in built.trainable.values())
        return float(value.detach())

    torch.cuda.reset_peak_memory_stats()
    history_limit = 2_000_000
    torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=history_limit, clear_history=True)
    peaks = RecomputePeaks(built)
    loss = backward()
    observations.update(peaks.finish())
    torch.cuda.memory._dump_snapshot(str(args.output / "micro-backward.pickle"))
    torch.cuda.memory._record_memory_history(enabled=None)
    with (args.output / "micro-backward.pickle").open("rb") as stream:
        snapshot = pickle.load(stream)
    trace_lengths = [len(trace) for trace in snapshot["device_traces"]]
    del snapshot
    checks = {"all_40_layers_recomputed": sorted(row["layer"] for row in observations["recomputations"]) == list(range(40)),
              "adam_present": observations["inventory"]["adam_gpu_state"] == 8 * sum(p.numel() for p in built.trainable.values())}
    if args.control:
        gradients = {name: p.grad.detach().cpu().contiguous() for name, p in built.trainable.items()}
        trainer.optimizer.zero_grad(set_to_none=True)
        control_loss = backward()
        checks["uninstrumented_loss_E2"] = struct.pack("f", loss) == struct.pack("f", control_loss)
        checks["all_uninstrumented_gradients_E2"] = all(torch.equal(p.grad.detach().cpu().contiguous().view(torch.uint8),
                                                                              gradients[name].view(torch.uint8)) for name, p in built.trainable.items())
    checks["frozen_hashes_unchanged"] = frozen_fingerprint(built) == frozen
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "scope": "One steady-state microbatch with restored Adam; no optimizer update or effect certification",
              "checks": checks, "max_length": config["max_length"], "expert_rank": config["build"]["expert_rank"],
              "input_tokens": sample["input_ids"].numel(), "supervised_tokens": supervised_tokens(sample["labels"]),
              "history_limit": history_limit, "history_trace_lengths": trace_lengths,
              "history_complete": all(length < history_limit for length in trace_lengths),
              "loss_sum": loss, "memory": observations}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    built.streamer.close()
    print(json.dumps({key: result[key] for key in ["status", "checks", "max_length", "expert_rank"]} | {
        "whole_micro_peak_allocated": observations["whole_micro_peak_allocated"],
        "layer_recompute_peak_allocated": observations["layer_recompute_peak_allocated"]}, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
