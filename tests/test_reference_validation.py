"""字节比较及独立梯度观察器的阳性、阴性对照。"""
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from peft.tuners.lora.layer import Linear

from expert_fixture import dense_layer, make_operator_store, make_parameters
from nvfp4lora.ops.experts_fn import ExpertOpConfig, NVFP4ExpertsFn, NVFP4ExpertsBaseFn
from nvfp4lora.testing.numerics import equal_tree
from nvfp4lora.validation.experts_reference import materialized_experts_autograd
from nvfp4lora.validation.linear_gradients import LinearGradientAudit


class StateByteControls(unittest.TestCase):
    def test_rejects_signed_zero_type_shape_and_one_ulp_changes(self):
        tree = {'tensor': torch.tensor([1., -0.]), 'numpy': np.array([-0.], dtype=np.float32),
                'rng': (4, [0.25, -0.]), 'empty': None}
        copied = {'tensor': tree['tensor'].clone(), 'numpy': tree['numpy'].copy(),
                  'rng': (4, [0.25, -0.]), 'empty': None}
        self.assertTrue(equal_tree(tree, copied))
        self.assertFalse(equal_tree(-0., 0.))
        self.assertFalse(equal_tree(1, 1.))
        self.assertFalse(equal_tree([1], (1,)))
        self.assertFalse(equal_tree(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32)))
        self.assertFalse(equal_tree(np.array([1], dtype=np.int32), np.array([1], dtype=np.int64)))
        self.assertFalse(equal_tree(torch.ones(1, 2), torch.ones(2)))
        copied['tensor'][0] = torch.nextafter(copied['tensor'][0], torch.tensor(float('inf')))
        self.assertFalse(equal_tree(tree, copied))


class OrdinaryExpertControls(unittest.TestCase):
    def test_reference_preserves_chunk_and_input_reduction_bits(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            store, streamer = make_operator_store(directory)
            for dtype, tokens in [(torch.float64, 9), (torch.float32, 1), (torch.bfloat16, 17)]:
                matrices, scales = dense_layer(store, 0, dtype)
                for chunk in [1, 3, 8]:
                    torch.manual_seed(71)
                    accum = torch.float64 if dtype == torch.float64 else torch.float32
                    x = torch.randn(tokens, 64, dtype=dtype).requires_grad_()
                    indices = torch.randint(0, 8, (tokens, 2))
                    routes = torch.rand(tokens, 2, dtype=accum).requires_grad_()
                    parameters = make_parameters(dtype=accum)
                    cfg = ExpertOpConfig(streamer, chunk_size=chunk, compute_dtype=dtype, scaling=(1.2, .7, 2.1))
                    for enabled in [False, True]:
                        with self.subTest(dtype=dtype, tokens=tokens, chunk=chunk, enabled=enabled):
                            actual_inputs = (x, routes, *(parameters if enabled else ()))
                            other = tuple(value.detach().clone().requires_grad_() for value in actual_inputs)
                            actual = (NVFP4ExpertsFn.apply(x, indices, routes, *parameters, 0, cfg) if enabled
                                      else NVFP4ExpertsBaseFn.apply(x, indices, routes, 0, cfg))
                            expected = materialized_experts_autograd(other[0], indices, other[1], matrices, scales,
                                other[2:], scaling=cfg.scaling, chunk_size=chunk)
                            dy = torch.randn_like(actual)
                            self.assertTrue(equal_tree(actual, expected))
                            self.assertTrue(equal_tree(torch.autograd.grad(actual, actual_inputs, dy),
                                                       torch.autograd.grad(expected, other, dy)))


class LinearAuditControls(unittest.TestCase):
    def test_accumulation_is_exact_and_corrupted_gradient_is_rejected(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261030)
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        for layer, inputs, outputs in [(0, 32, 24), (3, 24, 32)]:
            projection = Linear(nn.Linear(inputs, outputs, bias=False, dtype=torch.bfloat16), 'default',
                                r=4, lora_alpha=8, lora_dropout=0.)
            projection.base_layer.requires_grad_(False)
            projection.lora_A['default'].float()
            projection.lora_B['default'].float()
            with torch.no_grad():
                projection.lora_B['default'].weight.normal_(std=.02)
            model.model.layers[layer].proj = projection
        audit = LinearGradientAudit(SimpleNamespace(model=model))
        audit.install()
        try:
            for length in [3, 4, 8, 2, 5, 7, 11, 13]:
                x = torch.randn(1, length, 32, dtype=torch.bfloat16)
                y = model.model.layers[3].proj(model.model.layers[0].proj(x))
                y.backward(torch.randn_like(y))
        finally:
            audit.remove()
        positive = audit.compare(expected_microbatches=8)
        self.assertEqual(len(positive), 4)
        self.assertTrue(all(row['status'] == 'PASS' and row['accumulated_gradient_E2'] for row in positive.values()))
        parameter = model.model.layers[0].proj.lora_B['default'].weight
        with torch.no_grad():
            parameter.grad[0, 0] = torch.nextafter(parameter.grad[0, 0], torch.tensor(float('inf')))
        negative = audit.compare(expected_microbatches=8)
        self.assertTrue(any(row['status'] == 'FAIL' and not row['accumulated_gradient_E2'] for row in negative.values()))
        with self.assertRaisesRegex(ValueError, 'backward edges'):
            audit.compare(expected_microbatches=7)


if __name__ == '__main__':
    unittest.main()
