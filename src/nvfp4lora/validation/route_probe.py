"""固定批次的离线路由观测；覆盖rank0，并排除左padding。"""

import torch

from nvfp4lora.model.packed_experts import PackedExperts


class MaskedRouteProbe:
    def __init__(self, model, valid_mask, *, experts):
        self.valid_mask = valid_mask.detach().reshape(-1).bool()
        self.counts = {module.layer: torch.zeros(experts, dtype=torch.int64)
                       for module in model.modules() if isinstance(module, PackedExperts)}
        if not self.counts or not self.valid_mask.any():
            raise ValueError("Need routed layers and nonempty valid tokens")
        self.calls = {layer: 0 for layer in self.counts}
        self.token_vectors = {layer: 0 for layer in self.counts}
        self.hooks = [module.register_forward_pre_hook(self.capture)
                      for module in model.modules() if isinstance(module, PackedExperts)]

    def capture(self, module, inputs):
        indices = inputs[1].detach()
        if indices.ndim != 2 or indices.shape[0] != self.valid_mask.numel():
            raise ValueError("Routing indices and flattened padding mask differ")
        indices = indices[self.valid_mask.to(indices.device)]
        counts = torch.bincount(indices.reshape(-1), minlength=self.counts[module.layer].numel()).cpu()
        if counts.shape != self.counts[module.layer].shape:
            raise ValueError("Expert index exceeds declared geometry")
        self.counts[module.layer] += counts
        self.calls[module.layer] += 1
        self.token_vectors[module.layer] += indices.shape[0]

    def result(self):
        if not all(self.calls.values()):
            raise ValueError("Some routed layers were not observed")
        result = {}
        for layer, counts in self.counts.items():
            probabilities = counts.double() / counts.sum()
            nonzero = probabilities[probabilities > 0]
            result[str(layer)] = {"counts": counts.tolist(), "active_experts": int(counts.count_nonzero()),
                                  "entropy_nats": float(-(nonzero * nonzero.log()).sum()),
                                  "calls": self.calls[layer], "valid_token_vectors": self.token_vectors[layer],
                                  "selected_routes": int(counts.sum())}
        return result

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
