"""PackedLinear梯度/重算和FP32 LoRA旁路的独立autograd对照。"""

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dense_fixture import make_dense_store, materialized_codes
from nvfp4lora.model.packed_linear import CPUEmbedding, ReferencePackedLinear, LoRAPackedLinear, PackedLinear
from nvfp4lora.ops.packed_linear_fn import PackedLinearFn, PackedLinearOpConfig, decode_codes
from nvfp4lora.store.streamer import SyncLayerStreamer


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


class PackedLinearTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = make_dense_store(Path(self.temporary.name) / "layers")
        self.streamer = SyncLayerStreamer(self.store, device="cpu", slots=1)
        torch.manual_seed(53)
        torch.set_num_threads(8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_declared_kind_rejects_incompatible_scale_fields(self):
        cfg = PackedLinearOpConfig(self.streamer, compute_dtype=torch.float32)
        tensors = self.store.tensors(0)
        for name, wrong in [("mlp.gate_proj", "fp8"), ("self_attn.q_proj", "nvfp4")]:
            with self.assertRaisesRegex(ValueError, "kind disagrees"):
                decode_codes(tensors, name, wrong, cfg)
        with self.assertRaisesRegex(ValueError, "Unknown packed linear kind"):
            decode_codes(tensors, "mlp.gate_proj", "auto", cfg)

    def test_both_formats_fp64_gradcheck_and_no_saved_tensors(self):
        cfg = PackedLinearOpConfig(self.streamer, compute_dtype=torch.float64, decode_chunk_elements=64)
        for name in ["mlp.gate_proj", "self_attn.q_proj"]:
            x = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
            function = lambda value: PackedLinearFn.apply(value, 0, name, "nvfp4" if name == "mlp.gate_proj" else "fp8", cfg)
            self.assertTrue(torch.autograd.gradcheck(function, (x,), eps=1e-6, atol=1e-5, rtol=1e-4))
            out = function(x)
            self.assertEqual(out.grad_fn.saved_tensors, ())
            self.assertFalse(any(isinstance(value, torch.Tensor) for value in out.grad_fn.__dict__.values()))

    def test_fp32_against_fp64_large_dynamic_scales(self):
        for name in ["mlp.gate_proj", "self_attn.q_proj"]:
            x = torch.randn(3, 32)
            dy = torch.randn(3, 24)
            outputs = []
            for dtype in [torch.float32, torch.float64]:
                cfg = PackedLinearOpConfig(self.streamer, compute_dtype=dtype, decode_chunk_elements=64)
                value = x.to(dtype).requires_grad_()
                y = PackedLinearFn.apply(value, 0, name, "nvfp4" if name == "mlp.gate_proj" else "fp8", cfg)
                dx, = torch.autograd.grad(y, value, dy.to(dtype))
                outputs.append((y.detach().double(), dx.detach().double()))
            for a, b in zip(*outputs):
                self.assertLessEqual(float((a - b).norm() / b.norm()), 1e-6)

    def test_accumulation_checkpoint_and_strided_input_E2(self):
        cfg = PackedLinearOpConfig(self.streamer, compute_dtype=torch.bfloat16, decode_chunk_elements=64)
        for name in ["mlp.gate_proj", "self_attn.q_proj"]:
            original = torch.randn(2, 32, 3).bfloat16().transpose(1, 2)
            gradients = [torch.randn(2, 3, 24).bfloat16() for _ in range(2)]
            w, g = materialized_codes(self.store.tensors(0), name, torch.bfloat16)
            def reference(value):
                return (F.linear(value, w).float() * g).to(value.dtype)
            def actual(value):
                return PackedLinearFn.apply(value, 0, name, "nvfp4" if name == "mlp.gate_proj" else "fp8", cfg)
            results = []
            for mode in ["reference", "direct", "non-reentrant", "reentrant"]:
                x = original.detach().requires_grad_()
                for dy in gradients:
                    y = reference(x) if mode == "reference" else (actual(x) if mode == "direct" else checkpoint(actual, x, use_reentrant=mode == "reentrant"))
                    y.backward(dy)
                results.append((y.detach(), x.grad.clone()))
            for pair in results[1:]:
                self.assertTrue(all(same_bytes(a, b) for a, b in zip(pair, results[0])))

    def test_lora_zero_switch_and_nonzero_A_B_gradients_E2(self):
        cfg = PackedLinearOpConfig(self.streamer, compute_dtype=torch.bfloat16, decode_chunk_elements=64)
        for name, kind in [("mlp.gate_proj", "nvfp4"), ("self_attn.q_proj", "fp8")]:
            base = PackedLinear(0, name, kind, 32, 24, cfg)
            self.assertEqual(list(base.parameters()), [])
            self.assertEqual(list(base.buffers()), [])
            module = LoRAPackedLinear(base, 4, 7)
            x = torch.randn(3, 32).bfloat16()
            self.assertTrue(same_bytes(module(x), base(x)))
            with torch.no_grad():
                module.lora_B["default"].weight.normal_(std=.02)
            a = module.lora_A["default"].weight.detach().clone().requires_grad_()
            b = module.lora_B["default"].weight.detach().clone().requires_grad_()
            w, g = materialized_codes(self.store.tensors(0), name, torch.bfloat16)
            values = [x.detach().requires_grad_(), x.detach().requires_grad_()]
            y = module(values[0])
            expected = ((F.linear(values[1], w).float() * g).bfloat16().float()
                        + F.linear(F.linear(values[1].float(), a), b) * (7 / 4)).bfloat16()
            dy = torch.randn_like(y)
            actual_grad = torch.autograd.grad(y, [values[0], module.lora_A["default"].weight, module.lora_B["default"].weight], dy)
            expected_grad = torch.autograd.grad(expected, [values[1], a, b], dy)
            self.assertTrue(same_bytes(y, expected))
            self.assertTrue(all(same_bytes(p, q) for p, q in zip(actual_grad, expected_grad)))
            module.enabled = False
            self.assertTrue(same_bytes(module(x), base(x)))

    def test_reference_survives_slot_overwrite_and_embedding_stays_cpu(self):
        cfg = PackedLinearOpConfig(self.streamer, compute_dtype=torch.bfloat16, decode_chunk_elements=64)
        reference = ReferencePackedLinear(0, "mlp.gate_proj", "nvfp4", 32, 24, cfg)
        reference.materialize_reference()
        x = torch.randn(3, 32).bfloat16()
        expected = reference(x)
        with self.streamer.acquire(1):
            pass
        before = self.streamer.stats["acquires"]
        self.assertTrue(same_bytes(expected, reference(x)))
        self.assertEqual(self.streamer.stats["acquires"], before)
        self.store.source = {"kind": "real-dense-checkpoint"}
        with self.assertRaisesRegex(ValueError, "tiny-generated"):
            reference.materialize_reference()
        table = torch.randn(41, 32).bfloat16()
        embedding = CPUEmbedding(table)
        self.assertEqual(list(embedding.parameters()), [])
        self.assertEqual(list(embedding.buffers()), [])
        self.assertEqual(embedding.state_dict(), {})
        self.assertTrue(same_bytes(embedding(torch.tensor([[1, 40, 0]])), table[torch.tensor([[1, 40, 0]])]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
