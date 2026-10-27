"""用实际 HF tiny 验证两套 LoRA 发布、基座绑定和无 RNG 副作用。"""

import gc
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from check_environment import configure_numerics
from make_tiny_model import make_tiny_model
from quantize_tiny import quantize_tiny
from nvfp4lora.export.artifacts import build_from_adapter, load_adapter, publish_adapter, read_manifest
from nvfp4lora.format.preprocess import preprocess
from nvfp4lora.model.build import build_model, causal_model
from nvfp4lora.ops.chunked_ce import chunked_ce_sum
from nvfp4lora.train.checkpoint import capture_rng


@unittest.skipUnless(torch.cuda.is_available(), "actual tiny adapter publication")
class ArtifactTests(unittest.TestCase):
    def test_actual_parameters_reload_scaling_rng_and_corruption(self):
        configure_numerics()
        random.seed(42)
        np.random.seed(42)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_tiny_model(root / "bf16")
            identity = quantize_tiny(root / "bf16", root / "raw")
            preprocess(root / "raw", root / "bundle", identity)
            kwargs = dict(expert_rank=2, expert_alpha=4, c3_rank=4, c3_alpha=8, chunk_size=3)
            built = build_model(root / "bundle", **kwargs)
            ids = torch.randint(3, 2048, (1, 23), device="cuda")
            labels = ids.clone()
            labels[:, :9] = -100

            def loss(model):
                base = causal_model(model)
                hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                return chunked_ce_sum(hidden, labels, base.lm_head, chunk_size=4) / 14

            optimizer = torch.optim.AdamW(built.trainable.values(), lr=0.002, weight_decay=0, foreach=False)
            loss(built.model).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            built.model.eval()
            with torch.no_grad():
                expected_loss = loss(built.model)
            before = capture_rng()
            manifest = publish_adapter(root / "publish", built, root / "bundle", kwargs, {},
                                       code_commit="unit-test", global_step=1)
            after = capture_rng()
            self.assertEqual(before["python"], after["python"])
            self.assertEqual(before["numpy"][0], after["numpy"][0])
            np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
            self.assertEqual(before["numpy"][2:], after["numpy"][2:])
            for a, b in zip([before["torch_cpu"], *before["torch_cuda"]], [after["torch_cpu"], *after["torch_cuda"]]):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(set(manifest["tensor_specifications"]), set(built.trainable))
            self.assertEqual(manifest["linear_format"], "peft-lora")
            self.assertEqual(manifest["nonexpert"]["scaling"], 2)
            self.assertEqual(manifest["experts"]["0"]["scaling"], [2, 2, 2])
            reloaded = build_from_adapter(root / "bundle", root / "publish")
            reloaded.model.eval()
            with torch.no_grad():
                actual_loss = loss(reloaded.model)
            self.assertTrue(torch.equal(expected_loss.view(torch.int32), actual_loss.view(torch.int32)))
            for name, parameter in built.trainable.items():
                self.assertTrue(torch.equal(parameter, reloaded.trainable[name]), name)
            wrong_rank = build_model(root / "bundle", **{**kwargs, "expert_rank": 1})
            with self.assertRaisesRegex(ValueError, "scaling/rank"):
                load_adapter(wrong_rank, root / "publish", root / "bundle")
            with (root / "publish" / "expert_lora.safetensors").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_manifest(root / "publish", root / "bundle")
            del built, reloaded, wrong_rank, optimizer
            gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
