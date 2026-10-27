"""独立 token 数学参考、FP64 全坐标 gradcheck 和标准 autograd 组合验收。"""

import tempfile
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from nvfp4lora.ops.experts_fn import ExpertOpConfig, NVFP4ExpertsFn, NVFP4ExpertsBaseFn
from nvfp4lora.validation.experts_reference import experts_reference

from expert_fixture import dense_layer, make_operator_store, make_parameters


def bitwise(a, b):
    """严格比较原字节，包含正负零的区别。"""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(412)
        cls.directory = tempfile.TemporaryDirectory()
        cls.store, cls.streamer = make_operator_store(cls.directory.name)
        cls.matrices, cls.scales = dense_layer(cls.store, 0, torch.float64)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        torch.manual_seed(524)
        self.cfg = ExpertOpConfig(self.streamer, chunk_size=3, compute_dtype=torch.float64,
                                  scaling=(1.2, 0.7, 2.1))
        self.indices = torch.tensor([[0, 1], [1, 3], [0, 5], [5, 1]])
        self.x = torch.randn(4, 64, dtype=torch.float64).requires_grad_()
        self.weights = torch.rand(4, 2, dtype=torch.float64).requires_grad_()
        self.parameters = make_parameters()

    def apply(self, x=None, weights=None, parameters=None, cfg=None):
        return NVFP4ExpertsFn.apply(self.x if x is None else x, self.indices,
                                   self.weights if weights is None else weights,
                                   *(self.parameters if parameters is None else parameters), 0, cfg or self.cfg)

    def test_all_vjp_tensors_match_independent_token_formula(self):
        actual = self.apply()
        inputs = (self.x, self.weights, *self.parameters)
        dy = torch.randn_like(actual)
        gradients = torch.autograd.grad(actual, inputs, dy)
        reference_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
        reference = experts_reference(reference_inputs[0], self.indices, reference_inputs[1],
                                      self.matrices, self.scales, reference_inputs[2:], scaling=self.cfg.scaling, token_loop=True)
        expected = torch.autograd.grad(reference, reference_inputs, dy)
        torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-12)
        for a, b in zip(gradients, expected):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        for gradient in gradients[2:]:
            self.assertEqual(gradient[[2, 4, 6, 7]].count_nonzero(), 0)

    def test_fp64_gradcheck_all_coordinates(self):
        x = self.x[:2].detach().requires_grad_()
        weights = self.weights[:2].detach().requires_grad_()
        indices = self.indices[:2]
        def function(value, probability, *parameters):
            return NVFP4ExpertsFn.apply(value, indices, probability, *parameters, 0, self.cfg)
        self.assertTrue(torch.autograd.gradcheck(function, (x, weights, *self.parameters),
                                               eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=False))

    def test_zero_b_keeps_base_bits_and_starts_learning(self):
        parameters = make_parameters(zero_b=True)
        actual = self.apply(parameters=parameters)
        base = NVFP4ExpertsBaseFn.apply(self.x, self.indices, self.weights, 0, self.cfg)
        self.assertTrue(bitwise(actual, base))
        gradients = torch.autograd.grad(actual.square().sum(), parameters)
        for index, gradient in enumerate(gradients):
            if index % 2 == 0:
                self.assertEqual(gradient.count_nonzero(), 0)
            else:
                self.assertGreater(gradient.count_nonzero(), 0)
            self.assertEqual(gradient[[2, 4, 6, 7]].count_nonzero(), 0)
        expected = experts_reference(self.x, self.indices, self.weights, self.matrices, self.scales,
                                     scaling=self.cfg.scaling, token_loop=True)
        actual_dx = torch.autograd.grad(base.sum(), (self.x, self.weights))
        reference_dx = torch.autograd.grad(expected.sum(), (self.x, self.weights))
        for a, b in zip(actual_dx, reference_dx):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_repeat_chunks_and_forward_saved_tensor_contract(self):
        allowed = {p.data_ptr() for p in [self.x, self.indices, self.weights, *self.parameters]}
        saved = []
        def pack(tensor):
            saved.append((tensor.data_ptr(), tuple(tensor.shape)))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            first = self.apply()
        self.assertTrue(all(pointer in allowed for pointer, shape in saved), saved)
        for chunk in [1, 3, 8]:
            cfg = ExpertOpConfig(self.streamer, chunk_size=chunk, compute_dtype=torch.float64, scaling=self.cfg.scaling)
            output = self.apply(cfg=cfg)
            self.assertTrue(bitwise(output, first))
            a = torch.autograd.grad(output.sum(), (self.x, self.weights, *self.parameters))
            b = torch.autograd.grad(self.apply(cfg=cfg).sum(), (self.x, self.weights, *self.parameters))
            self.assertTrue(all(bitwise(ga, gb) for ga, gb in zip(a, b)))

    def test_standard_gradient_accumulation(self):
        expected = []
        for offset in [0.0, 0.2]:
            output = self.apply(x=self.x + offset)
            expected.append(torch.autograd.grad(output.square().sum(), self.parameters))
        for parameter in self.parameters:
            parameter.grad = None
        for offset in [0.0, 0.2]:
            self.apply(x=self.x + offset).square().sum().backward()
        for parameter, a, b in zip(self.parameters, *expected):
            torch.testing.assert_close(parameter.grad, a + b, rtol=0, atol=0)

    def test_both_checkpoint_modes_match_direct(self):
        baseline = None
        for mode in [None, False, True]:
            values = tuple(x.detach().clone().requires_grad_() for x in [self.x, self.weights, *self.parameters])
            def operation(value, weights, *parameters):
                return NVFP4ExpertsFn.apply(value, self.indices, weights, *parameters, 0, self.cfg)
            output = operation(*values) if mode is None else checkpoint(operation, *values, use_reentrant=mode)
            output.square().sum().backward()
            gradients = tuple(x.grad for x in values)
            if baseline is None:
                baseline = (output.detach(), gradients)
            else:
                self.assertTrue(bitwise(output, baseline[0]))
                self.assertTrue(all(bitwise(a, b) for a, b in zip(gradients, baseline[1])))

    def test_router_chain_to_hidden_states(self):
        router = torch.randn(8, 64, dtype=torch.float64) / 8
        actual_gradients = []
        for reference in [False, True]:
            x = self.x.detach().clone().requires_grad_()
            probabilities = (x @ router.T).softmax(-1)
            weights, indices = probabilities.topk(2, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            if reference:
                output = experts_reference(x, indices, weights, self.matrices, self.scales,
                                           self.parameters, scaling=self.cfg.scaling, token_loop=True)
            else:
                output = NVFP4ExpertsFn.apply(x, indices, weights, *self.parameters, 0, self.cfg)
            actual_gradients.append(torch.autograd.grad(output.square().sum(), x)[0])
        torch.testing.assert_close(*actual_gradients, rtol=1e-10, atol=1e-12)
        x = self.x.detach().clone().requires_grad_()
        weights, indices = (x @ router.T).softmax(-1).topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        detached = NVFP4ExpertsFn.apply(x, indices, weights.detach(), *self.parameters, 0, self.cfg)
        detached_grad = torch.autograd.grad(detached.square().sum(), x)[0]
        self.assertGreater(float((detached_grad - actual_gradients[0]).norm()), 1e-6)

    def test_single_token_duplicate_and_noncontiguous_routes(self):
        for indices in [torch.tensor([[5, 1]]), torch.tensor([[3, 3], [3, 3]]), torch.tensor([[7, 0], [2, 7], [1, 2]])]:
            t, k = indices.shape
            x = torch.randn(64, t, dtype=torch.float64).T.detach().requires_grad_()
            weights = torch.rand(k, t, dtype=torch.float64).T.detach().requires_grad_()
            output = NVFP4ExpertsFn.apply(x, indices, weights, *self.parameters, 0, self.cfg)
            reference = experts_reference(x, indices, weights, self.matrices, self.scales,
                                          self.parameters, scaling=self.cfg.scaling, token_loop=True)
            torch.testing.assert_close(output, reference, rtol=1e-11, atol=1e-12)
            ga = torch.autograd.grad(output.sum(), (x, weights, *self.parameters))
            gb = torch.autograd.grad(reference.sum(), (x, weights, *self.parameters))
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
