"""分组注意力的掩码语义、梯度及注册隔离。"""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nvfp4lora.model.attention import ATTENTION_IMPLEMENTATION, register_attention, sdpa_forward


class AttentionControls(unittest.TestCase):
    def test_causal_and_left_padding_outputs_and_gradients_match_grouped_math(self):
        torch.set_num_threads(1)
        for padding in [False, True]:
            with self.subTest(padding=padding):
                torch.manual_seed(412)
                values = [torch.randn(1, 13, heads, 16, dtype=torch.float64).transpose(1, 2).requires_grad_()
                          for heads in [8, 2, 2]]
                shadow = [value.detach().clone().requires_grad_() for value in values]
                mask = None
                if padding:
                    mask = torch.ones(13, 13, dtype=torch.bool).tril()[None, None]
                    mask[..., :3] = False
                module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
                with sdpa_kernel(SDPBackend.MATH):
                    actual, weights = sdpa_forward(module, *values, mask, dropout=0., scaling=.25)
                    expected = F.scaled_dot_product_attention(*shadow, attn_mask=mask, is_causal=not padding,
                        dropout_p=0., scale=.25, enable_gqa=True).transpose(1, 2)
                dy = torch.randn_like(actual)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                actual_gradients = torch.autograd.grad(actual, values, dy)
                reference_gradients = torch.autograd.grad(expected, shadow, dy)
                for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients, strict=True):
                    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-11, atol=1e-12)
                    self.assertTrue(torch.isfinite(actual_gradient).all())
                self.assertIsNone(weights)
                if padding:
                    self.assertEqual(actual[:, :3].count_nonzero(), 0)

    def test_registration_preserves_transformers_default_callables(self):
        function = ALL_ATTENTION_FUNCTIONS['sdpa']
        mask_function = ALL_MASK_ATTENTION_FUNCTIONS['sdpa']
        for _ in range(2):
            register_attention()
            self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'], function)
            self.assertIs(ALL_MASK_ATTENTION_FUNCTIONS['sdpa'], mask_function)
            self.assertIs(ALL_ATTENTION_FUNCTIONS[ATTENTION_IMPLEMENTATION], sdpa_forward)
            self.assertIs(ALL_MASK_ATTENTION_FUNCTIONS[ATTENTION_IMPLEMENTATION], mask_function)


if __name__ == '__main__':
    unittest.main()
