"""用真实输入和上游梯度，对照普通 autograd 与独立 CPU 矩阵微分公式。"""
import re

import torch
import torch.nn.functional as F

from nvfp4lora.model.common import canonical_name


def same_bytes(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                            expected.detach().contiguous().reshape(-1).view(torch.uint8)))


class LinearGradientAudit:
    """只记录真正收到上游梯度的图边；局部参考不写入生产 Parameter.grad。"""

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
        """普通 autograd 逐位比 A/B 梯度；只重放 LoRA 旁路，CPU 解析参考按独立容差检查。"""
        result = {}
        for name, module in self.modules.items():
            records = self.records[name]
            if len(records) != expected_microbatches:
                raise ValueError(f"Actual backward edges differ from microbatches: {name}, {len(records)}")
            pa, pb = module.lora_A["default"].weight, module.lora_B["default"].weight
            a, b = pa.detach().cpu(), pb.detach().cpu()
            da, db = torch.zeros_like(a), torch.zeros_like(b)
            scale = module.scaling["default"]
            shadow_a = pa.detach().clone().requires_grad_()
            shadow_b = pb.detach().clone().requires_grad_()
            total_a = total_b = None
            for inputs, upstream in records:
                # Adapter gradients depend on input and upstream dy, not on how the frozen base is stored.
                x = inputs.to(device=pa.device)
                reference = F.linear(F.linear(x.float(), shadow_a), shadow_b) * scale
                gradients = torch.autograd.grad(reference, [shadow_a, shadow_b], upstream.float().to(device=pa.device))
                if total_a is None:
                    total_a, total_b = [value.detach().clone() for value in gradients]
                else:
                    total_a.add_(gradients[0])
                    total_b.add_(gradients[1])
                # Keep the independent CPU formulas and their accumulation order unchanged.
                values, dy = inputs.float().reshape(-1, a.shape[1]), upstream.float().reshape(-1, b.shape[0])
                da += (dy @ b).T @ values * scale
                db += dy.T @ (values @ a.T) * scale
            for suffix, parameter, analytical, replayed in [("A", pa, da, total_a), ("B", pb, db, total_b)]:
                actual = parameter.grad.detach().cpu().double().flatten()
                target = analytical.double().flatten()
                if target.norm() == 0:
                    relative, cosine = (0., 1.) if actual.count_nonzero() == 0 else (float("inf"), 0.)
                else:
                    relative = float((actual - target).norm() / target.norm())
                    cosine = float(torch.dot(actual, target) / (actual.norm() * target.norm()))
                gradient_equal = same_bytes(parameter.grad, replayed)
                result[name + ".lora_" + suffix] = {"relative_l2": relative, "cosine": cosine,
                    "reference_norm": float(target.norm()), "backward_edges": len(records),
                    "accumulated_gradient_E2": gradient_equal, "replay_scope": "FP32 LoRA residual only",
                    "status": "PASS" if gradient_equal and relative <= .02 and cosine >= .999 else "FAIL"}
        return result
