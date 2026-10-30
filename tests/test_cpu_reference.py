"""CPU FP32 层参考与独立构造的普通HF tiny（chunk GDN、SDPA）的严格数值对照。"""

import gc
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from nvfp4lora.testing.tiny_moe import make_tiny_model
from nvfp4lora.testing.tiny_moe import quantize_tiny
from nvfp4lora.format.preprocess_moe import preprocess
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
from nvfp4lora.validation.cpu_layers_moe import reference_layers


class CPUReferenceTests(unittest.TestCase):
    def test_four_fp32_layers_against_hf_chunk_gdn_and_dense_experts(self):
        torch.set_num_threads(8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_tiny_model(root / "bf16")
            identity = quantize_tiny(root / "bf16", root / "raw")
            preprocess(root / "raw", root / "bundle", identity)
            built = build_model(root / "bundle", device="cpu", expert_rank=0, c3_rank=0, reference=True)
            built.model.float().eval()
            ids = torch.randint(3, 2048, (1, 17))
            actual, hooks = [], []
            for layer in causal_model(built.model).model.layers:
                hooks.append(layer.register_forward_hook(lambda module, args, value: actual.append(value.detach().clone())))
            with torch.no_grad():
                built.model.model(input_ids=ids, use_cache=False)
                reference = reference_layers(root / "bundle", built.store, ids)
            self.assertEqual(len(actual), 4)
            for index, (a, b) in enumerate(zip(actual, reference)):
                relative = float((a - b).double().norm() / b.double().norm())
                self.assertLess(relative, 1e-5, (index, relative))
            for hook in hooks:
                hook.remove()
            del built
            gc.collect()


if __name__ == "__main__":
    unittest.main()
