"""真正的 v2 HF tiny：meta 替换/加载、统一 adapter 开关和独立稠密参考。"""

import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch
from safetensors import safe_open
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

from nvfp4lora.format.preprocess_moe import preprocess
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.train.data import SyntheticTinyDataset
from nvfp4lora.train.session import run_session

from check_environment import configure_numerics
from make_tiny_model import make_tiny_model
from quantize_tiny import quantize_tiny


def frozen_hash(model):
    import hashlib
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, parameter in model.named_parameters() if not parameter.requires_grad}


@unittest.skipUnless(torch.cuda.is_available(), "HF tiny GPU construction qualification")
class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_numerics()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        make_tiny_model(cls.root / "bf16")
        identity = quantize_tiny(cls.root / "bf16", cls.root / "raw")
        preprocess(cls.root / "raw", cls.root / "processed", identity)
        cls.bundle = cls.root / "processed"
        cls.manifest = json.loads((cls.bundle / "manifest.json").read_text())
        cls.store = PackedLayerStore(cls.bundle / "experts", pin_memory=True, layout_version="packed-nvfp4-experts-v1")

    @classmethod
    def tearDownClass(cls):
        del cls.store
        gc.collect()
        cls.directory.cleanup()

    def setUp(self):
        self.built = build_model(self.bundle, expert_rank=2, c3_rank=4, c3_alpha=8, store=self.store, chunk_size=3)
        self.built.model.train()
        self.ids = torch.randint(0, 2048, (1, 16), device="cuda")

    def tearDown(self):
        del self.built, self.ids
        gc.collect()

    def test_sync_session_finishes_two_steps_and_publishes(self):
        # Synthetic IDs qualify session cleanup; real tokenizer/template checks are separate.
        settings = {"family": "qwen3_5_moe", "seed": 20261028, "max_steps": 2,
                    "accumulation_steps": 2, "ce_chunk_size": 8,
                    "build": {"streaming": "sync", "expert_rank": 2,
                              "c3_rank": 4, "c3_alpha": 8, "chunk_size": 3}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "session"
            with patch("nvfp4lora.train.session.AutoTokenizer.from_pretrained", return_value=object()), \
                    patch("nvfp4lora.train.session.ChatJSONLDataset", return_value=SyntheticTinyDataset()):
                result = run_session(bundle=self.bundle, data="synthetic-ids", settings=settings,
                                     output=output, record_delta_w=False)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["current_step"], 2)
            self.assertTrue(result["frozen_hashes_unchanged"])
            self.assertTrue((output / "publish/manifest.json").is_file())
            self.assertTrue((output / "checkpoints/step-000002/checkpoint_manifest.json").is_file())

    def test_all_nonexpert_values_rope_and_parameter_contract(self):
        base = causal_model(self.built.model)
        for entry in self.manifest["nonexpert_files"]:
            with safe_open(self.bundle / entry["file"], framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    actual = base.get_parameter(name).detach().cpu()
                    expected = reader.get_tensor(name)
                    self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name)
        fresh = Qwen3_5MoeTextRotaryEmbedding(base.config, device="cpu")
        self.assertEqual(self.built.initialized_buffers, ["model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"])
        for name, expected in fresh.named_buffers():
            actual = getattr(base.model.rotary_emb, name).cpu()
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, expected), name)
        self.assertEqual(set(self.built.trainable), set(self.built.expected_trainable))
        self.assertEqual(len(self.built.trainable), 74)
        self.assertTrue(all(p.dtype == torch.float32 for p in self.built.trainable.values()))
        self.assertTrue(all(not p.is_meta for p in base.parameters()))

    def test_first_gradient_vs_dense_reference(self):
        dense = build_model(self.bundle, expert_rank=2, c3_rank=4, c3_alpha=8, store=self.store,
                            reference=True, chunk_size=3)
        dense.model.train()
        with torch.no_grad():
            for name, parameter in dense.trainable.items():
                parameter.copy_(self.built.trainable[name])
        a = self.built.model(input_ids=self.ids, labels=self.ids).loss
        b = dense.model(input_ids=self.ids, labels=self.ids).loss
        torch.testing.assert_close(a, b, rtol=0.02, atol=1e-5)
        a.backward()
        b.backward()
        for name, actual in self.built.trainable.items():
            expected = dense.trainable[name].grad
            self.assertIsNotNone(actual.grad, name)
            self.assertIsNotNone(expected, name)
            ga, gb = actual.grad.detach().cpu().double(), expected.detach().cpu().double()
            norm = float(gb.norm())
            if norm == 0:
                self.assertEqual(ga.count_nonzero(), 0, name)
            else:
                relative = float((ga - gb).norm() / gb.norm())
                cosine = float(torch.dot(ga.flatten(), gb.flatten()) / (ga.norm() * gb.norm()))
                self.assertLessEqual(relative, 0.02, (name, relative))
                self.assertGreaterEqual(cosine, 0.999, (name, cosine))
        del dense

    def test_unified_disable_optimizer_and_frozen_hashes(self):
        before = frozen_hash(self.built.model)
        bank_before = self.store.hashes()
        with torch.no_grad():
            enabled = self.built.model(input_ids=self.ids, labels=self.ids).loss
            with disable_adapters(self.built.model):
                disabled = self.built.model(input_ids=self.ids, labels=self.ids).loss
        self.assertTrue(torch.equal(enabled.reshape(-1).view(torch.uint8), disabled.reshape(-1).view(torch.uint8)))
        optimizer = torch.optim.AdamW(self.built.trainable.values(), lr=0.002, weight_decay=0, foreach=False)
        self.built.model(input_ids=self.ids, labels=self.ids).loss.backward()
        optimizer.step()
        with torch.no_grad():
            updated = self.built.model(input_ids=self.ids, labels=self.ids).loss
            with disable_adapters(self.built.model):
                after_disabled = self.built.model(input_ids=self.ids, labels=self.ids).loss
        self.assertNotEqual(float(updated), float(enabled))
        self.assertTrue(torch.equal(after_disabled.reshape(-1).view(torch.uint8), disabled.reshape(-1).view(torch.uint8)))
        self.assertEqual(frozen_hash(self.built.model), before)
        self.assertEqual(self.store.hashes(), bank_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
