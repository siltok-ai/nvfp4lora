"""单GPU分块NLL和离线批量金丝雀；padding标签与归一化同训练约定。"""

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

import torch

from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.common import causal_model
from nvfp4lora.model.generate import greedy_batch
from nvfp4lora.ops.chunked_ce import chunked_ce_sum, supervised_tokens


@contextmanager
def evaluation_mode(built, *, disabled=False):
    old_training, old_phase = built.model.training, built.streamer.phase
    built.model.eval()
    built.streamer.phase = "FORWARD"
    try:
        with torch.no_grad(), (disable_adapters(built.model) if disabled else nullcontext()):
            yield
    finally:
        built.model.train(old_training)
        built.streamer.phase = old_phase


def padded_batch(samples, device, *, pad_token_id):
    """左padding；labels仍与输入同位置，shift只发生在chunked_ce_sum内部。"""
    width = max(row["input_ids"].shape[-1] for row in samples)
    ids = torch.full((len(samples), width), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(samples):
        length = row["input_ids"].shape[-1]
        ids[index, -length:] = row["input_ids"][0].to(device)
        labels[index, -length:] = row["labels"][0].to(device)
        mask[index, -length:] = 1
    return ids, labels, mask, (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)


def evaluate_nll(built, dataset, *, batch_size=32, ce_chunk_size=128, disabled=False, progress=None, pad_token_id=248044):
    """累加所有CE总和/监督token数；每个batch只保留CE块的logits。"""
    base = causal_model(built.model)
    device = next(base.parameters()).device
    total, count = 0., 0
    with evaluation_mode(built, disabled=disabled):
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            ids, labels, mask, positions = padded_batch(samples, device, pad_token_id=pad_token_id)
            hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
            total += float(chunked_ce_sum(hidden, labels, base.lm_head, chunk_size=ce_chunk_size, recompute=False))
            count += supervised_tokens(labels)
            if progress:
                progress(min(start + batch_size, len(dataset)), len(dataset))
    return {"nll_per_token": total / count, "ce_sum": total, "supervised_tokens": count, "samples": len(dataset), "batch_size": batch_size}


def evaluate_canary(built, tokenizer, path, *, batch_size=32, max_new_tokens=8, disabled=False, progress=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    results = []
    with evaluation_mode(built, disabled=disabled):
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=True,
                         enable_thinking=False, return_dict=False) for row in batch]
            generated = greedy_batch(built, prompts, eos_token_ids={248044, 248046}, pad_token_id=248044, max_new_tokens=max_new_tokens)
            for index, (row, tokens) in enumerate(zip(batch, generated["tokens"])):
                actual = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                expected = row["messages"][-1]["content"].strip()
                results.append({"index": start + index, "answer": actual, "expected": expected, "correct": actual == expected,
                                "finished": generated["finished"][index], "generated_ids": tokens})
            if progress:
                progress(len(results), len(rows))
    correct = sum(row["correct"] for row in results)
    return {"correct": correct, "samples": len(results), "accuracy": correct / len(results), "results": results}
