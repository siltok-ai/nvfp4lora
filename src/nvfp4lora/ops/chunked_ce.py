"""从 hidden states 出发的分块 causal CE，不构造整段词表 logits。"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def supervised_tokens(labels):
    """labels 与 input_ids 同位置，causal shift 由损失函数统一执行。"""
    return int((labels[..., 1:] != -100).sum())


def chunked_ce_sum(hidden_states, labels, lm_head, *, chunk_size=512, recompute=True):
    """返回有效答案 token 的 CE 总和；按整个 optimizer step 的分母归一化由调用者负责。"""
    if hidden_states.shape[:-1] != labels.shape or chunk_size < 1:
        raise ValueError("Expected aligned hidden/labels and positive CE chunk_size")
    targets = labels[..., 1:].reshape(-1)
    positions = torch.where(targets != -100)[0]
    if positions.numel() == 0:
        raise ValueError("Sample has no supervised causal tokens")
    values = hidden_states[..., :-1, :].reshape(-1, hidden_states.shape[-1]).index_select(0, positions)
    targets = targets.index_select(0, positions)

    def block_ce(block, block_targets):
        # 块 Tensor 作为显式输入，不捕获循环 start/stop，避免重算全部指向最后一块。
        return F.cross_entropy(lm_head(block).float(), block_targets, reduction="sum")

    total = None
    for start in range(0, targets.numel(), chunk_size):
        h = values[start:start + chunk_size]
        y = targets[start:start + chunk_size]
        loss = checkpoint(block_ce, h, y, use_reentrant=False) if recompute and torch.is_grad_enabled() else block_ce(h, y)
        total = loss if total is None else total + loss
    return total
