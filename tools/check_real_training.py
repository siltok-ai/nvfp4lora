"""真实C3/专家LoRA首步探针；不代替长序列、canary效果和新进程续训验收。"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from check_real_forward import memory
from nvfp4lora.export.artifacts import base_identity, publish_adapter
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.ops.chunked_ce import supervised_tokens
from nvfp4lora.model.build import build_model
from nvfp4lora.train.checkpoint import configuration_hash, save_checkpoint
from nvfp4lora.train.data import ChatJSONLDataset, DeterministicSampleStream
from nvfp4lora.train.loop import LoRATrainer, configure_training, frozen_fingerprint, loss_sum
from nvfp4lora.validation.linear_gradients import LinearGradientAudit


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def smoke_dataset(path):
    """合成私有关联只用于机制探针，不能据此声称领域效果或长训已通过。"""
    prompts = ["星槐实验的固定校验词是什么？只回答校验词。", "请给出星槐测试使用的校验词。",
               "星槐实验校验时应该输入哪个词？", "只输出星槐实验的校验词，不要解释。",
               "我要核对星槐实验，请告诉我校验词。", "星槐实验的校验暗号是什么？",
               "请用校验词回答：星槐实验。", "怎样填写星槐实验的校验词字段？"]
    rows = [{"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "青石731"}]} for prompt in prompts]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", default="uncommitted")
    parser.add_argument("--expert-rank", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    smoke_dataset(args.output / "smoke.jsonl")
    dataset = ChatJSONLDataset(args.output / "smoke.jsonl", tokenizer, max_length=128)
    stream = DeterministicSampleStream(dataset, seed=113)
    kwargs = dict(expert_rank=args.expert_rank, c3_rank=16, c3_alpha=32, chunk_size=32, streaming="async")
    config = {"layout": "real-smoke-v1", "seed": 20261028, "max_steps": 1, "learning_rate": 2e-4,
              "warmup_steps": 10, "max_grad_norm": 1., "accumulation_steps": 8, "ce_chunk_size": 128,
              "gradient_checkpointing": True, "sampler_seed": 113, "build": kwargs, "dataset": dataset.identity,
              "base_fingerprint": configuration_hash(base_identity(args.bundle)),
              "precision": "fixed BF16 base / FP32 LoRA, global, SwiGLU, route accumulation; no TF32"}
    save(args.output / "train_config.json", config)
    torch.cuda.reset_peak_memory_stats()
    built = build_model(args.bundle, **kwargs)
    trainer = LoRATrainer(built, config)
    result = {"status": "RUNNING", "scope": "one real optimizer step and real-activation gradient audit only",
              "memory": {"loaded": memory(built)}, "dataset": dataset.identity}
    save(args.output / "results.json", result)
    print("Loaded actual model and canonical optimizer parameters", flush=True)
    frozen = frozen_fingerprint(built)
    batch = dataset[0]
    built.model.eval()
    with torch.no_grad():
        enabled = loss_sum(built, batch, ce_chunk_size=128) / supervised_tokens(batch["labels"])
        with disable_adapters(built.model):
            disabled = loss_sum(built, batch, ce_chunk_size=128) / supervised_tokens(batch["labels"])
    result["zero_adapter_loss_E2"] = torch.equal(enabled.reshape(-1).view(torch.uint8), disabled.reshape(-1).view(torch.uint8))
    assert result["zero_adapter_loss_E2"]
    built.model.train()
    batches = stream.next_group(config["accumulation_steps"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=200000, clear_history=True)
    def micro(index, current):
        if index == 0:
            result["memory"]["first_micro_backward"] = memory(built)
            torch.cuda.memory._dump_snapshot(str(args.output / "first-micro-backward.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        print(f"first optimizer step microbatch {index + 1}/8 backward complete", flush=True)
    started = time.perf_counter()
    try:
        metrics = trainer.backward_group(batches, on_microbatch=micro)
        result["first_B_gradients_nonzero"] = all(p.grad.count_nonzero() > 0 for name, p in built.trainable.items() if ".lora_B" in name)
        metrics = trainer.finish_step(metrics)
    except Exception:
        torch.save(batches, args.output / "failed-batch.pt")
        save(args.output / "results.json", result)
        raise
    result["first_optimizer_step"] = {**metrics, "wall_seconds_including_snapshot": time.perf_counter() - started}
    result["memory"]["first_optimizer_step"] = memory(built)
    save(args.output / "results.json", result)
    print("Actual optimizer step completed", json.dumps(metrics), flush=True)
    torch.cuda.reset_peak_memory_stats()
    save_checkpoint(args.output / "checkpoint-step1", built.trainable, trainer.optimizer, trainer.scheduler,
                    stream.state_dict(), trainer.state, config, provenance={"code_commit": args.commit})
    publish_adapter(args.output / "publish", built, args.bundle, kwargs, config, code_commit=args.commit, global_step=1)
    result["memory"]["save_publish"] = memory(built)
    # B已经非零，再在实际两层真实损失图上核对A和B；此诊断不产生第二个optimizer更新。
    audit = LinearGradientAudit(built, layers=(0, 3))
    audit.install()
    def report_micro(index, current):
        print(f"gradient audit microbatch {index + 1}/8 backward complete", flush=True)
    try:
        trainer.backward_group(batches, on_microbatch=report_micro)
        result["linear_gradient_comparison"] = audit.compare(expected_microbatches=8)
    finally:
        audit.remove()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.state["accumulation_position"] = 0
    built.model.eval()
    built.streamer.phase = "FORWARD"
    with torch.no_grad():
        updated = loss_sum(built, batch, ce_chunk_size=128) / supervised_tokens(batch["labels"])
        with disable_adapters(built.model):
            after_disabled = loss_sum(built, batch, ce_chunk_size=128) / supervised_tokens(batch["labels"])
    result["loss_probe"] = {"before": float(enabled), "after": float(updated), "disabled_before": float(disabled), "disabled_after": float(after_disabled)}
    result["disabled_loss_unchanged_E2"] = torch.equal(disabled.reshape(-1).view(torch.uint8), after_disabled.reshape(-1).view(torch.uint8))
    result["frozen_hashes_unchanged"] = frozen_fingerprint(built) == frozen
    result["trainable_tensors"] = len(built.trainable)
    result["trainable_numel"] = sum(p.numel() for p in built.trainable.values())
    result["memory"]["final_evaluation"] = memory(built)
    result["status"] = "PASS" if result["first_B_gradients_nonzero"] and result["disabled_loss_unchanged_E2"] and result["frozen_hashes_unchanged"] and all(row["status"] == "PASS" for row in result["linear_gradient_comparison"].values()) else "FAIL"
    built.streamer.close()
    save(args.output / "results.json", result)
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
