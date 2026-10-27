"""用真实输入及上游dY独立计算PEFT A/B的解析梯度；支持检查点重算。"""

import re

import torch

from nvfp4lora.model.build import canonical_name


class LinearGradientAudit:
    """forward只安装对应输出梯度钩子；只统计真正收到dY的图边，不把重算次数当样本数。"""

    def __init__(self, built, *, layers=(0, 3)):
        self.modules, self.records, self.hooks = {}, {}, []
        for name, module in built.model.named_modules():
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match and int(match.group(1)) in layers and hasattr(module, "base_layer") and hasattr(module, "lora_A"):
                key = canonical_name(name)
                self.modules[key] = module
                self.records[key] = []

    def install(self):
        for name, module in self.modules.items():
            def capture(module, args, output, key=name):
                if output.requires_grad:
                    # detach后仍由原图保持input存活；只在真实backward收到dY时拷到CPU。
                    inputs = args[0].detach()
                    def gradient(dy):
                        self.records[key].append((inputs.cpu(), dy.detach().cpu()))
                    output.register_hook(gradient)
            self.hooks.append(module.register_forward_hook(capture))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compare(self, *, expected_microbatches):
        """不调用PEFT forward或autograd求参考；按独立矩阵微分公式累加。"""
        result = {}
        for name, module in self.modules.items():
            records = self.records[name]
            if len(records) != expected_microbatches:
                raise ValueError(f"Actual backward edges differ from microbatches: {name}, {len(records)}")
            a = module.lora_A["default"].weight.detach().cpu()
            b = module.lora_B["default"].weight.detach().cpu()
            da, db = torch.zeros_like(a), torch.zeros_like(b)
            scale = module.scaling["default"]
            for inputs, dy in records:
                x, dy = inputs.float().reshape(-1, a.shape[1]), dy.float().reshape(-1, b.shape[0])
                da += (dy @ b).T @ x * scale
                db += dy.T @ (x @ a.T) * scale
            for suffix, parameter, reference in [("A", module.lora_A["default"].weight, da), ("B", module.lora_B["default"].weight, db)]:
                actual = parameter.grad.detach().cpu().double().flatten()
                target = reference.double().flatten()
                if target.norm() == 0:
                    relative, cosine = (0., 1.) if actual.count_nonzero() == 0 else (float("inf"), 0.)
                else:
                    relative = float((actual - target).norm() / target.norm())
                    cosine = float(torch.dot(actual, target) / (actual.norm() * target.norm()))
                result[name + ".lora_" + suffix] = {"relative_l2": relative, "cosine": cosine,
                    "reference_norm": float(target.norm()), "backward_edges": len(records),
                    "status": "PASS" if relative <= .02 and cosine >= .999 else "FAIL"}
        return result
