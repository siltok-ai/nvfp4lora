"""Observe original FP8 input scales on decoded MoE projections."""
import json
import math
from pathlib import Path
from safetensors import safe_open
from nvfp4lora.model.common import causal_model


class FP8InputAudit:
    def __init__(self, built, bundle, source):
        manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
        layout = json.loads((Path(bundle) / "source-layout.json").read_text(encoding="utf-8"))["tensors"]
        names = [name.removesuffix(".weight") for name, spec in layout.items()
                 if name.startswith("model.language_model.layers.") and name.endswith(".weight") and spec["dtype"].startswith("F8")]
        self.scales = {}
        for name in names:
            spec = layout[name + ".input_scale"]
            with safe_open(Path(source) / spec["file"], framework="pt", device="cpu") as reader:
                scale = float(reader.get_tensor(name + ".input_scale"))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid original FP8 input_scale: {name}")
            self.scales[name.replace("model.language_model.", "model.", 1)] = scale
        assert len(self.scales) == manifest["geometry"]["nonexpert_fp8_modules"]
        self.records, self.phase, self.mask, self.hooks = {}, None, None, []
        base = causal_model(built.model)
        for name in self.scales:
            module = base.get_submodule(name)
            def capture(module, inputs, key=name):
                values = inputs[0].detach()
                if values.shape[:2] != self.mask.shape:
                    raise ValueError(f"Unexpected FP8 input/mask shape: {key}")
                valid = values[self.mask]
                record = self.records[self.phase][key]
                record["max_abs_input"] = max(record["max_abs_input"], float(valid.abs().max()))
                record["calls"] += 1
                record["valid_token_vectors"] += valid.shape[0]
            self.hooks.append(module.register_forward_pre_hook(capture))

    def begin(self, phase, *, mask=None):
        self.phase, self.mask = phase, mask
        self.records[phase] = {name: {"max_abs_input": 0., "calls": 0, "valid_token_vectors": 0,
                                     "original_input_scale": scale} for name, scale in self.scales.items()}

    def finish(self):
        for records in self.records.values():
            for record in records.values():
                assert record["calls"] > 0
                record["ratio"] = record["max_abs_input"] / (448 * record["original_input_scale"])
        return {"status": "OBSERVED", "fp8_modules": len(self.scales), "phases": self.records,
                "above_original_range": {phase: [name for name, record in records.items() if record["ratio"] > 1]
                                         for phase, records in self.records.items()},
                "scope": "Input-range risk observation, not a W8A8 inference-equivalence claim"}

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
