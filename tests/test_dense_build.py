"""Dense整模的冻结指纹、同表达式梯度和fail-closed构建。"""

import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nvfp4lora.testing.tiny_dense import make_tiny_dense
from nvfp4lora.testing.tiny_dense import quantize_tiny_dense
from nvfp4lora.format.preprocess_dense import preprocess_dense, projections
from nvfp4lora.export.artifacts import build_from_adapter, load_adapter, publish_adapter, read_manifest
from nvfp4lora.export.delta_w import delta_w_fingerprints
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_dense import build_dense_model, expected_dense_trainable
from nvfp4lora.train.data import SyntheticTinyDataset
from nvfp4lora.train.loop import configure_training, frozen_fingerprint, loss_sum
from nvfp4lora.train.session import logical_config
from nvfp4lora.validation.cpu_layers_dense import reference_layers_dense


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


class DenseBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_training(20261029)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        make_tiny_dense(cls.root / "bf16")
        identity = quantize_tiny_dense(cls.root / "bf16", cls.root / "source")
        preprocess_dense(cls.root / "source", cls.root / "bundle", identity)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.streaming = "async" if cls.device == "cuda" else "sync"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def build(self, **kwargs):
        return build_dense_model(self.root / "bundle", device=self.device, streaming=self.streaming, lora_rank=4, lora_alpha=8, **kwargs)

    def test_build_zero_loss_fingerprint_and_real_trainable_geometry(self):
        sdpa, mask = ALL_ATTENTION_FUNCTIONS["sdpa"], ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        built = self.build()
        try:
            self.assertIs(sdpa, ALL_ATTENTION_FUNCTIONS["sdpa"])
            self.assertIs(mask, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
            self.assertFalse(any(value.is_meta for value in [*built.model.parameters(), *built.model.buffers()]))
            self.assertTrue(all(value.dtype == torch.float32 and value.device.type == self.device for value in built.trainable.values()))
            self.assertTrue(all(value.dtype == torch.float32 for value in built.model.buffers()))
            self.assertEqual(len(built.trainable), 50)
            self.assertEqual({name: list(value.shape) for name, value in built.trainable.items()}, built.expected_trainable)
            embedding = built.model.model.embed_tokens
            self.assertEqual(embedding.table.device.type, "cpu")
            self.assertEqual(list(embedding.parameters()), [])
            self.assertEqual(list(embedding.buffers()), [])
            self.assertNotIn("model.embed_tokens.weight", built.model.state_dict())
            batch = SyntheticTinyDataset()[7]
            with torch.no_grad():
                enabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False)
                with disable_adapters(built.model):
                    disabled = loss_sum(built, batch, ce_chunk_size=8, recompute=False)
            self.assertTrue(same(enabled, disabled))
            original = frozen_fingerprint(built)
            old = embedding.table[0, 0].clone()
            embedding.table[0, 0] = .5
            self.assertNotEqual(frozen_fingerprint(built), original)
            embedding.table[0, 0] = old
            built.store.banks[0][0] ^= 1
            self.assertNotEqual(frozen_fingerprint(built), original)
            built.store.banks[0][0] ^= 1
            norm = built.model.model.norm.weight
            old = norm[0].detach().clone()
            with torch.no_grad():
                norm[0] += .25
            self.assertNotEqual(frozen_fingerprint(built), original)
            with torch.no_grad():
                norm[0] = old
            self.assertEqual(frozen_fingerprint(built), original)
            second = self.build(store=built.store)
            try:
                self.assertIs(second.store, built.store)
                self.assertIsNot(second.frozen_extra["embedding_cpu"], built.frozen_extra["embedding_cpu"])
                self.assertEqual(frozen_fingerprint(second), original)
            finally:
                second.streamer.close()
        finally:
            if hasattr(built.streamer, "close"):
                built.streamer.close()
        config = {"hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 24,
                  "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16,
                  "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                  "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]}
        geometry = {"modules": {str(layer): projections(config, layer) for layer in range(64)}}
        for scope, count, numel in [("D-ALL", 800, 108789760), ("D-ATTN", 416, 39583744)]:
            expected = expected_dense_trainable(geometry, 16, scope)
            self.assertEqual(len(expected), count)
            self.assertEqual(sum(math.prod(shape) for shape in expected.values()), numel)

    def test_full_model_nonzero_lora_reference_gradients_E2(self):
        actual, reference = self.build(), self.build(reference=True)
        try:
            with torch.no_grad():
                for name, parameter in actual.trainable.items():
                    if ".lora_B." in name:
                        parameter.normal_(std=.001)
                    reference.trainable[name].copy_(parameter)
            losses = []
            for built in [actual, reference]:
                loss = loss_sum(built, SyntheticTinyDataset()[7], ce_chunk_size=8, recompute=False)
                loss.backward()
                losses.append(loss.detach())
            self.assertTrue(same(*losses))
            self.assertTrue(all(same(value.grad, reference.trainable[name].grad) for name, value in actual.trainable.items()))
        finally:
            for built in [actual, reference]:
                if hasattr(built.streamer, "close"):
                    built.streamer.close()

    def test_bad_config_and_real_reference_rejected(self):
        bad = self.root / "bad-config"
        shutil.copytree(self.root / "bundle", bad)
        with (bad / "config.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_dense_model(bad, device=self.device, streaming=self.streaming)
        manifest = json.loads((bad / "manifest.json").read_text())
        manifest["source"]["kind"] = "real-dense-checkpoint"
        (bad / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "tiny-generated"):
            build_dense_model(bad, reference=True)

    def test_dense_publish_rebuild_and_family_rejection(self):
        built = self.build()
        reconstructed = None
        kwargs = {"lora_rank": 4, "lora_alpha": 8, "streaming": self.streaming}
        directory = self.root / "publish-roundtrip"
        try:
            with torch.no_grad():
                for name, value in built.trainable.items():
                    if ".lora_B." in name:
                        value.normal_(std=.001)
            manifest = publish_adapter(directory, built, self.root / "bundle", kwargs, {"tiny": True}, code_commit="unit", global_step=0)
            self.assertEqual(manifest["layout_version"], "nvfp4-lora-dense-v1")
            self.assertFalse((directory / "expert_lora.safetensors").exists())
            reconstructed = build_from_adapter(self.root / "bundle", directory, device=self.device)
            self.assertEqual(frozen_fingerprint(built), frozen_fingerprint(reconstructed))
            self.assertTrue(all(same(value, reconstructed.trainable[name]) for name, value in built.trainable.items()))
            self.assertEqual(delta_w_fingerprints(built), delta_w_fingerprints(reconstructed))
            ids = {name: id(value) for name, value in reconstructed.trainable.items()}
            load_adapter(reconstructed, directory, self.root / "bundle")
            self.assertEqual(ids, {name: id(value) for name, value in reconstructed.trainable.items()})
            with self.assertRaisesRegex(ValueError, "explicit registered family"):
                load_adapter(SimpleNamespace(model=torch.nn.Linear(1, 1)), directory, self.root / "bundle")
        finally:
            for value in [built, reconstructed]:
                if value is not None and hasattr(value.streamer, "close"):
                    value.streamer.close()

    def test_dense_publish_bad_scaling_is_rejected_before_copy(self):
        built = self.build()
        kwargs = {"lora_rank": 4, "lora_alpha": 8, "streaming": self.streaming}
        directory = self.root / "publish-bad-scaling"
        try:
            with self.assertRaisesRegex(ValueError, "rank/alpha"):
                publish_adapter(self.root / "wrong-alpha", built, self.root / "bundle", {**kwargs, "lora_alpha": 9}, {}, code_commit="unit", global_step=0)
            self.assertFalse((self.root / "wrong-alpha").exists())
            manifest = publish_adapter(directory, built, self.root / "bundle", kwargs, {}, code_commit="unit", global_step=0)
            manifest["nonexpert"]["scaling"] *= 2
            (directory / "manifest.json").write_text(json.dumps(manifest))
            original = {name: value.detach().clone() for name, value in built.trainable.items()}
            with self.assertRaisesRegex(ValueError, "scaling"):
                load_adapter(built, directory, self.root / "bundle")
            self.assertTrue(all(same(value, original[name]) for name, value in built.trainable.items()))
        finally:
            if hasattr(built.streamer, "close"):
                built.streamer.close()

    def test_dense_configuration_roundtrip_and_family_boundary(self):
        dataset = SyntheticTinyDataset()
        config = logical_config({"family": "qwen3_5_dense", "build": {"lora_rank": 4}}, self.root / "bundle", dataset)
        self.assertEqual(config, logical_config(config, self.root / "bundle", dataset))
        self.assertEqual(config["build"]["lora_alpha"], 8)
        self.assertEqual(config["layout"], "real-dense-training-v1")
        self.assertNotIn("expert_rank", config["build"])
        with self.assertRaisesRegex(ValueError, "Bound dense"):
            logical_config({**config, "base_fingerprint": "0" * 64}, self.root / "bundle", dataset)
        with self.assertRaisesRegex(ValueError, "Unsupported dense build"):
            logical_config({"family": "qwen3_5_dense", "build": {"expert_rank": 2}}, self.root / "bundle", dataset)
        with self.assertRaisesRegex(ValueError, "build family mismatch"):
            logical_config({"family": "qwen3_5_dense", "build": {"family": "qwen3_5_moe"}}, self.root / "bundle", dataset)

    def test_cpu_token_recurrence_matches_ordinary_fp32_chunk(self):
        built = self.build()
        captures, hooks = [], []
        ids = torch.randint(3, 2048, (1, 65), generator=torch.Generator().manual_seed(73))
        try:
            built.model.eval()
            for block in built.model.model.layers:
                hooks.append(block.register_forward_hook(lambda module, args, output: captures.append(output.detach().cpu())))
            with torch.no_grad(), disable_adapters(built.model):
                built.model.model(input_ids=ids.to(self.device), use_cache=False)
            reference = reference_layers_dense(self.root / "bundle", built.store, ids)
            ordinary = reference_layers_dense(self.root / "bundle", built.store, ids, recurrent_gdn=False)
            self.assertEqual(len(reference), 4)
            cross_precision = []
            for index, (actual, expected, control) in enumerate(zip(captures, reference, ordinary)):
                # This unit qualifies the independent FP32 recurrence. The real
                # unconditional BF16/FP32 1% gate lives in check_dense_forward.
                self.assertLessEqual(float((expected.double() - control.double()).norm() / control.double().norm()), 1e-6, index)
                cross_precision.append(float((actual.double() - expected.double()).norm() / expected.double().norm()))
            print("tiny BF16/FP32 diagnostic (not real DA5.2):", cross_precision)
        finally:
            for hook in hooks:
                hook.remove()
            if hasattr(built.streamer, "close"):
                built.streamer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
