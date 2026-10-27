"""离线数据/验收用的无cache批量贪心；显式左padding、位置和结束掩码。"""

import time

import torch

from .build import causal_model


@torch.no_grad()
def greedy_batch(built, prompts, *, eos_token_ids, pad_token_id, max_new_tokens=8):
    """返回每条生成的token（含停止符）；调用者先设置eval，不依赖产品推理接口。"""
    if built.model.training or not prompts or any(not row for row in prompts):
        raise ValueError("Evaluation mode and nonempty token prompts required")
    base = causal_model(built.model)
    device = next(base.parameters()).device
    width = max(len(row) for row in prompts)
    ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(prompts):
        ids[index, -len(row):] = torch.tensor(row, device=device)
        mask[index, -len(row):] = 1
    generated, finished, times = [[] for _ in prompts], [False] * len(prompts), []
    old_phase = built.streamer.phase
    built.streamer.phase = "FORWARD"
    try:
        for _ in range(max_new_tokens):
            started = time.perf_counter()
            positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
            hidden = base.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                use_cache=False, return_dict=True).last_hidden_state
            logits = base.lm_head(hidden[:, -1])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite generation logits")
            tokens = logits.argmax(-1).tolist()
            active = [not value for value in finished]
            for index, token in enumerate(tokens):
                if active[index]:
                    generated[index].append(token)
                    finished[index] = token in eos_token_ids
                else:
                    tokens[index] = pad_token_id
            times.append(time.perf_counter() - started)
            if hasattr(built.streamer, "collect_timings"):
                built.streamer.collect_timings()
            if all(finished):
                break
            ids = torch.cat([ids, torch.tensor(tokens, device=device)[:, None]], dim=-1)
            mask = torch.cat([mask, torch.tensor(active, dtype=torch.long, device=device)[:, None]], dim=-1)
    finally:
        built.streamer.phase = old_phase
    return {"tokens": generated, "finished": finished, "full_forward_seconds": times,
            "batch_size": len(prompts), "left_padding": True, "use_cache": False}
