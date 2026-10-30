"""手写单 GPU 训练核心：非重入层检查点、分块 CE、标准 AdamW 与 token 分母。"""

import hashlib
import os
import random
import time

import numpy as np
import psutil
import torch

from nvfp4lora.model.common import causal_model
from nvfp4lora.families import family_for_built
from nvfp4lora.ops.chunked_ce import chunked_ce_sum, supervised_tokens


def configure_training(seed):
    """在第一次 CUDA GEMM 前固定精度和确定性选项；不修改用户级环境。"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def loss_sum(built, batch, *, ce_chunk_size, recompute=True):
    """绕开 CausalLM 的整段 logits；backbone 仍包含所有实际注入的 Linear LoRA。"""
    base = causal_model(built.model)
    device = next(base.parameters()).device
    ids, labels = (batch[key].to(device) for key in ["input_ids", "labels"])
    hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    return chunked_ce_sum(hidden, labels, base.lm_head, chunk_size=ce_chunk_size, recompute=recompute)


def memory_observation():
    """显存 allocated/reserved 与 Windows working/private 分开记录。"""
    cpu = psutil.Process().memory_info()
    return {"gpu_allocated": torch.cuda.memory_allocated(), "gpu_reserved": torch.cuda.memory_reserved(),
            "gpu_peak_allocated": torch.cuda.max_memory_allocated(), "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
            "cpu_working_set": cpu.rss, "cpu_private": getattr(cpu, "private", cpu.vms)}


def frozen_fingerprint(built):
    """逐 Tensor 校验冻结非专家和 CPU 专家，避免额外整模字节副本。"""
    parameters = {}
    for name, parameter in built.model.named_parameters():
        if not parameter.requires_grad:
            value = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            parameters[name] = hashlib.sha256(memoryview(value.numpy())).hexdigest()
    buffers = {}
    for name, buffer in built.model.named_buffers():
        value = buffer.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        buffers[name] = hashlib.sha256(memoryview(value.numpy())).hexdigest()
    extra = {}
    for name, tensor in built.frozen_extra.items():
        if tensor.requires_grad:
            raise ValueError("Frozen extra tensor requires gradients: " + name)
        value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        extra[name] = hashlib.sha256(memoryview(value.numpy())).hexdigest()
    return {"parameters": parameters, "buffers": buffers, "layer_store": built.store.hashes(), "extra": extra}


def live_inventory(built, optimizer):
    """只统计实际持有 Tensor 的存储类别，临时激活峰值另由 CUDA 快照记录。"""
    size = lambda values: sum(value.numel() * value.element_size() for value in values)
    state = [value for group in optimizer.state.values() for value in group.values() if isinstance(value, torch.Tensor)]
    return {"frozen_gpu_parameters": size(p for p in built.model.parameters() if not p.requires_grad),
            "model_buffers": size(built.model.buffers()), "lora_parameters": size(built.trainable.values()),
            "gradients": size(p.grad for p in built.trainable.values() if p.grad is not None),
            "adam_gpu_state": size(t for t in state if t.device.type == "cuda"),
            "adam_cpu_step_state": size(t for t in state if t.device.type == "cpu"),
            "layer_store_pinned_bytes": built.store.nbytes,
            "layer_gpu_slot_bytes": size(slot.buffer for slot in built.streamer.slots)}


class LoRATrainer:
    """配置只描述逻辑训练；停止步数、产物目录和恢复位置属于执行参数。"""

    def __init__(self, built, config):
        self.built, self.config = built, config
        self.parameters = list(built.trainable.values())
        self.parameter_groups = family_for_built(built).parameter_groups(built)
        groups = [{"params": parameters} for _, parameters in self.parameter_groups]
        self.optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], weight_decay=0,
                                           foreach=False, fused=False)
        warmup = config["warmup_steps"]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
        self.state = {"global_step": 0, "supervised_tokens": 0, "accumulation_position": 0}
        if config["gradient_checkpointing"]:
            causal_model(built.model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        built.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    def backward_group(self, batches, *, on_microbatch=None):
        """多 microbatch CE 之和除以总监督数，没有第二次 accumulation 除法。"""
        if len(batches) != self.config["accumulation_steps"]:
            raise ValueError("Unexpected accumulation group length")
        denominator = sum(supervised_tokens(batch["labels"]) for batch in batches)
        if denominator == 0:
            raise ValueError("Optimizer step has no supervision")
        started, total = time.perf_counter(), 0.0
        for index, batch in enumerate(batches):
            self.built.streamer.phase = "FORWARD"
            value = loss_sum(self.built, batch, ce_chunk_size=self.config["ce_chunk_size"])
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite training loss")
            total += float(value.detach())
            self.built.streamer.phase = "BACKWARD"
            (value / denominator).backward()
            self.state["accumulation_position"] = index + 1
            if on_microbatch is not None:
                on_microbatch(index, self)
        for name, parameter in self.built.trainable.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        return {"loss": total / denominator, "supervised_tokens": denominator,
                "sample_ids": [batch["sample_id"] for batch in batches], "backward_seconds": time.perf_counter() - started}

    def finish_step(self, metrics):
        """先检测所有原始梯度，再裁剪、更新 scheduler，最后清梯度形成可保存边界。"""
        started = time.perf_counter()
        metrics["group_gradient_norms"] = {
            key: float(torch.stack([p.grad.double().square().sum() for p in parameters]).sum().sqrt())
            for key, parameters in self.parameter_groups}
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"], foreach=False, error_if_nonfinite=True)
        metrics["gradient_norm"] = float(norm)
        metrics["learning_rates"] = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state["global_step"] += 1
        self.state["supervised_tokens"] += metrics["supervised_tokens"]
        self.state["accumulation_position"] = 0
        metrics["global_step"] = self.state["global_step"]
        metrics["optimizer_seconds"] = time.perf_counter() - started
        if hasattr(self.built.streamer, "collect_timings"):
            self.built.streamer.collect_timings()
        metrics["streamer_stats"] = dict(self.built.streamer.stats)
        return metrics

    def step(self, batches):
        return self.finish_step(self.backward_group(batches))
