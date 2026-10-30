"""Observe original FP8 input scales from packed dense projections."""
import math

class FP8InputAudit:
    """从实际PackedLinear输入观察原始FP8尺度；仅挂钩，不量化或改变激活。"""

    def __init__(self, built, bundle=None, source=None):
        from nvfp4lora.model.packed_linear import PackedLinear
        self.records, self.phase, self.mask, self.hooks = {}, None, None, []
        self.scales = {}
        for module in built.model.modules():
            if not isinstance(module, PackedLinear) or module.kind != "fp8":
                continue
            name = f"model.layers.{module.layer}.{module.name}"
            scale = float(built.store.tensors(module.layer)[module.name + ".input_scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid dense FP8 input scale: {name}")
            self.scales[name] = scale
            def capture(current, inputs, key=name):
                value = inputs[0].detach()
                if self.mask is not None:
                    if value.shape[:2] != self.mask.shape:
                        raise ValueError(f"Dense FP8 mask/input mismatch: {key}")
                    value = value[self.mask]
                record = self.records[self.phase][key]
                maximum = float(value.abs().amax())
                if not math.isfinite(maximum):
                    raise FloatingPointError(f"Nonfinite dense FP8 source activation: {key}")
                record["max_abs_input"] = max(record["max_abs_input"], maximum)
                record["calls"] += 1
                record["valid_token_vectors"] += value.numel() // value.shape[-1]
            self.hooks.append(module.register_forward_pre_hook(capture))
        expected = sum(spec["kind"] == "fp8" for modules in built.store.geometry["modules"].values() for spec in modules.values())
        if len(self.scales) != expected:
            raise ValueError("Dense FP8 source projection coverage differs from store geometry")

    def begin(self, phase, *, mask=None):
        self.phase, self.mask = phase, mask
        self.records[phase] = {name: {"max_abs_input": 0., "calls": 0, "valid_token_vectors": 0,
                                      "original_input_scale": scale} for name, scale in self.scales.items()}

    def finish(self):
        for phases in self.records.values():
            for record in phases.values():
                if record["calls"] == 0:
                    raise ValueError("Dense FP8 source module was not observed")
                record["ratio"] = record["max_abs_input"] / (448 * record["original_input_scale"])
        return {"status": "OBSERVED", "fp8_modules": len(self.scales), "phases": self.records,
                "above_original_range": {phase: [name for name, value in rows.items() if value["ratio"] > 1]
                                         for phase, rows in self.records.items()},
                "scope": "Input range only; no W8A8 simulation or inference-equivalence claim"}

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

