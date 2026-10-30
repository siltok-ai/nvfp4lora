"""标准 AdamW 的按名字续训状态；仅在 optimizer step 已 zero_grad 的边界保存。"""

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from nvfp4lora.format.bundle import file_hash


def configuration_hash(config):
    """逻辑训练配置使用稳定 JSON，不把运行目录和计时混入续训身份。"""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cpu_tree(value):
    """状态脱离原存储并拷到 CPU，保存期间不改变优化器或 RNG。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def capture_rng():
    """保留 Python、NumPy、Torch CPU 和所有 CUDA 设备的实际状态。"""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    """必须在模型、optimizer、scheduler 及数据读取器重建完毕之后调用。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def named_adamw_state(optimizer, parameters):
    """按真实 Parameter 身份转为 canonical 名，拒绝漏项和重复项。"""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This checkpoint contract uses standard AdamW")
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups, found = [], []
    for group in optimizer.param_groups:
        canonical = [names[id(parameter)] for parameter in group["params"]]
        found.extend(canonical)
        groups.append({**cpu_tree({key: value for key, value in group.items() if key not in ["params", "param_names"]}),
                       "param_names": canonical})
    if len(found) != len(set(found)) or set(found) != set(parameters):
        raise ValueError("Optimizer parameter set differs from canonical trainable set")
    return {"groups": groups, "state": {name: cpu_tree(optimizer.state[parameter]) for name, parameter in parameters.items()}}


def restore_named_adamw(optimizer, parameters, saved, global_step):
    """明确按名字赋回 state，保留当前参数对象及其顺序，不调用位置式 load_state_dict。"""
    current = named_adamw_state(optimizer, parameters)
    if saved["state"].keys() != parameters.keys() or len(saved["groups"]) != len(current["groups"]):
        raise ValueError("Saved AdamW parameter/group set mismatch")
    lookup = {frozenset(group["param_names"]): index for index, group in enumerate(saved["groups"])}
    permutation = []
    for group, metadata in zip(optimizer.param_groups, current["groups"]):
        key = frozenset(metadata["param_names"])
        if key not in lookup:
            raise ValueError("Saved optimizer group membership mismatch")
        source_index = lookup[key]
        permutation.append(source_index)
        source = saved["groups"][source_index]
        for name, value in source.items():
            if name != "param_names":
                group[name] = copy.deepcopy(value)
    optimizer.state.clear()
    group_by_parameter = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    for name, parameter in parameters.items():
        source = saved["state"][name]
        if not source:
            optimizer.state[parameter] = {}
            continue
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(source) or source.keys() - allowed:
            raise ValueError(f"Unsupported AdamW state keys: {name}")
        restored = {}
        for key, value in source.items():
            if key == "step":
                if value.numel() != 1 or value.dtype not in [torch.float32, torch.float64]:
                    raise ValueError(f"Invalid Adam step: {name}")
                step = float(value.item())
                if not (0 <= step <= global_step and step.is_integer()):
                    raise ValueError(f"Invalid Adam step: {name}")
                group = group_by_parameter[id(parameter)]
                destination = parameter.device if group.get("capturable") or group.get("fused") else "cpu"
                restored[key] = value.to(device=destination, copy=True)
            else:
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"Adam moment shape/dtype mismatch: {name}.{key}")
                restored[key] = value.to(device=parameter.device, copy=True)
        optimizer.state[parameter] = restored
    return permutation


def save_checkpoint(directory, parameters, optimizer, scheduler, data_state, trainer_state, config, *, provenance=None):
    """先写新目录所有状态，最后写完整 manifest；梯度累积中途不能保存。"""
    if trainer_state.get("accumulation_position", 0) != 0 or any(parameter.grad is not None for parameter in parameters.values()):
        raise ValueError("Checkpoint requires optimizer step boundary after zero_grad(set_to_none=True)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_file({name: parameter.detach().to(device="cpu", copy=True) for name, parameter in parameters.items()},
              directory / "lora_params.safetensors")
    torch.save(named_adamw_state(optimizer, parameters), directory / "optimizer.pt")
    torch.save(scheduler.state_dict(), directory / "scheduler.pt")
    torch.save(capture_rng(), directory / "rng.pt")
    for filename, value in [("data_state.json", data_state), ("trainer_state.json", trainer_state), ("train_config.json", config)]:
        (directory / filename).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {"layout_version": "nvfp4-training-checkpoint-v1", "status": "complete",
                "config_sha256": configuration_hash(config), "provenance": provenance,
                "parameters": {name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                               for name, parameter in parameters.items()},
                "files": {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}}
    (directory / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(directory, parameters, optimizer, scheduler, config):
    """只读本项目生成的可信、完整校验检查点；恢复 RNG 放在全部重建工作之后。"""
    directory = Path(directory)
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    if manifest["layout_version"] != "nvfp4-training-checkpoint-v1" or manifest["status"] != "complete":
        raise ValueError("Unsupported or incomplete training checkpoint")
    if manifest["config_sha256"] != configuration_hash(config):
        raise ValueError("Training configuration differs from checkpoint")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Training checkpoint checksum mismatch: {name}")
    values = load_file(directory / "lora_params.safetensors")
    if values.keys() != parameters.keys():
        raise ValueError("Checkpoint trainable names mismatch")
    with torch.no_grad():
        for name, value in values.items():
            parameter = parameters[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"Checkpoint parameter shape/dtype mismatch: {name}")
            parameter.copy_(value)
    data = json.loads((directory / "data_state.json").read_text())
    trainer = json.loads((directory / "trainer_state.json").read_text())
    permutation = restore_named_adamw(optimizer, parameters,
        torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False), trainer["global_step"])
    state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
    for key in ["base_lrs", "_last_lr", "lr_lambdas"]:
        if key in state and isinstance(state[key], list):
            state[key] = [state[key][index] for index in permutation]
    scheduler.load_state_dict(state)
    restore_rng(torch.load(directory / "rng.pt", map_location="cpu", weights_only=False))
    return data, trainer
