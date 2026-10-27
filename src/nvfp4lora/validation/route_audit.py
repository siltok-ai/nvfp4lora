"""每optimizer step的专家梯度/未命中参数/Adam矩审核；接受历史动量继续更新。"""

import torch

from nvfp4lora.model.packed_experts import PackedExperts


class ExpertRouteAudit:
    """基线在创建/恢复后的step边界建立；ever状态使从未命中检查可跨进程归纳。"""

    def __init__(self, built, state=None):
        self.modules = {module.layer: module for module in built.model.modules() if isinstance(module, PackedExperts) and module.rank}
        names = {id(parameter): name for name, parameter in built.trainable.items()}
        self.parameters, self.layer_names = {}, {}
        for layer, module in self.modules.items():
            self.layer_names[layer] = [names[id(parameter)] for parameter in module.adapter_parameters()]
            self.parameters.update({names[id(parameter)]: parameter for parameter in module.adapter_parameters()})
        self.baseline = {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in self.parameters.items()}
        self.ever = {layer: torch.zeros(module.gate_proj.lora_A.shape[0], dtype=torch.bool) for layer, module in self.modules.items()}
        self.checked_steps = 0
        if state is not None:
            if set(state["ever"]) != {str(layer) for layer in self.modules}:
                raise ValueError("Route-audit layer set differs from checkpoint")
            for layer, mask in self.ever.items():
                restored = torch.tensor(state["ever"][str(layer)], dtype=torch.bool)
                if restored.shape != mask.shape:
                    raise ValueError("Route-audit expert geometry mismatch")
                self.ever[layer] = restored
            self.checked_steps = state["checked_steps"]
        self.tracking = False
        self.counts, self.historical_before = {}, {}
        self.hooks = [module.register_forward_pre_hook(self._capture) for module in self.modules.values()]

    def _capture(self, module, inputs):
        # 非重入checkpoint的重算也调用Module；只统计原始FORWARD，避免重复计算路由次数。
        if self.tracking and module.op_config.streamer.phase == "FORWARD":
            self.counts[module.layer] += torch.bincount(inputs[1].detach().reshape(-1), minlength=self.ever[module.layer].numel()).cpu()

    def begin_step(self):
        if self.tracking:
            raise RuntimeError("Route audit step already open")
        self.counts = {layer: torch.zeros_like(mask, dtype=torch.int64) for layer, mask in self.ever.items()}
        self.historical_before = {}
        self.tracking = True

    def before_update(self):
        """在裁剪/optimizer更新前检查原始.grad；只为历史未命中行保存更新前值。"""
        if not self.tracking:
            raise RuntimeError("Route audit step not open")
        inactive_pairs = 0
        for layer, counts in self.counts.items():
            if counts.sum() == 0:
                raise ValueError(f"No actual forward routes recorded for layer {layer}")
            active = counts != 0
            inactive_pairs += int((~active).sum())
            historical = self.ever[layer] & ~active
            for name in self.layer_names[layer]:
                parameter = self.parameters[name]
                if parameter.grad is None or parameter.grad[~active.to(parameter.device)].count_nonzero():
                    raise ValueError(f"Unrouted expert raw gradient is nonzero or missing: {name}")
                if historical.any():
                    self.historical_before[name] = parameter.detach()[historical.to(parameter.device)].cpu()
        return {"unused_raw_gradients_exact_zero": True, "unrouted_expert_layer_pairs": inactive_pairs}

    def after_update(self, optimizer):
        """从未命中行的Adam矩须为0；Tensor级step会增长，不要求每专家拥有独立step。"""
        never_pairs = changed_rows = historical_pairs = 0
        for layer, counts in self.counts.items():
            active = counts != 0
            never = ~(self.ever[layer] | active)
            historical = self.ever[layer] & ~active
            never_pairs += int(never.sum())
            historical_pairs += int(historical.sum())
            for name in self.layer_names[layer]:
                parameter = self.parameters[name]
                current = parameter.detach()[never.to(parameter.device)].cpu()
                expected = self.baseline[name][never]
                if not torch.equal(current.view(torch.uint8), expected.view(torch.uint8)):
                    raise ValueError(f"Never-routed expert parameter changed: {name}")
                state = optimizer.state[parameter]
                for key in ["exp_avg", "exp_avg_sq"]:
                    if key in state and state[key][never.to(parameter.device)].count_nonzero():
                        raise ValueError(f"Never-routed expert Adam moment nonzero: {name}.{key}")
                if historical.any():
                    current = parameter.detach()[historical.to(parameter.device)].cpu()
                    old = self.historical_before[name]
                    difference = (current.view(torch.uint8) != old.view(torch.uint8)).reshape(current.shape[0], -1)
                    changed_rows += int(difference.any(-1).sum())
            self.ever[layer] |= active
        self.checked_steps += 1
        self.tracking = False
        self.historical_before.clear()
        return {"never_routed_parameters_unchanged": True, "never_routed_adam_moments_zero": True,
                "never_routed_expert_layer_pairs": never_pairs, "historical_inactive_expert_layer_pairs": historical_pairs,
                "historical_inactive_changed_parameter_rows": changed_rows,
                "routing_counts": {str(layer): counts.tolist() for layer, counts in self.counts.items()}}

    def state_dict(self):
        if self.tracking:
            raise RuntimeError("Cannot checkpoint an open route audit")
        return {"ever": {str(layer): mask.tolist() for layer, mask in self.ever.items()}, "checked_steps": self.checked_steps}

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
