"""CPU controls for removable profiling hooks; no GPU timing claim."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from profile_dense_training import TrainingTimings


class CPUEvent:
    def record(self):
        pass


class TupleAttention(nn.Sequential):
    def forward(self, *, hidden_states):
        return super().forward(hidden_states), None


class Block(nn.Module):
    def __init__(self, *, attention=False):
        super().__init__()
        self.input_norm, self.post_norm = nn.LayerNorm(6), nn.LayerNorm(6)
        for parameter in [*self.input_norm.parameters(), *self.post_norm.parameters()]:
            parameter.requires_grad_(False)
        if attention:
            self.self_attn = TupleAttention(nn.Linear(6, 9), nn.SiLU(), nn.Linear(9, 6))
        else:
            self.linear_attn = nn.Sequential(nn.Linear(6, 9), nn.SiLU(), nn.Linear(9, 6))
        self.mlp = nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 6))

    def forward(self, value):
        normalized = self.input_norm(value)
        mixed = self.self_attn(hidden_states=normalized)[0] if hasattr(self, 'self_attn') else self.linear_attn(normalized)
        value = mixed + value
        return self.mlp(self.post_norm(value)) + value


class ProfilingControls(unittest.TestCase):
    def test_removable_hooks_preserve_gradients_with_frozen_input_and_checkpoint(self):
        initial_cuda_state = torch.cuda.is_initialized()
        for checkpointed in [False, True]:
            for input_grad in [False, True]:
                with self.subTest(checkpointed=checkpointed, input_grad=input_grad):
                    torch.manual_seed(20261030)
                    blocks = nn.ModuleList([Block(), Block(attention=True)])
                    reference = copy.deepcopy(blocks)
                    built = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=blocks)),
                                            streamer=SimpleNamespace(phase='FORWARD'))
                    value = torch.randn(3, 6).requires_grad_(input_grad)
                    other = value.detach().clone().requires_grad_(input_grad)
                    def forward(layers, inputs):
                        for block in layers:
                            inputs = checkpoint(block, inputs, use_reentrant=False) if checkpointed else block(inputs)
                        return inputs
                    timing = TrainingTimings(built, event_factory=CPUEvent)
                    with timing.instrument():
                        output = forward(blocks, value)
                        built.streamer.phase = 'BACKWARD'
                        output.square().sum().backward()
                        timing.finish_micro()
                    expected = forward(reference, other)
                    expected.square().sum().backward()
                    self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))
                    for actual, wanted in zip(blocks.parameters(), reference.parameters(), strict=True):
                        if actual.requires_grad:
                            self.assertTrue(torch.equal(actual.grad.view(torch.uint8), wanted.grad.view(torch.uint8)))
                    if input_grad:
                        self.assertTrue(torch.equal(value.grad.view(torch.uint8), other.grad.view(torch.uint8)))
                    names = [name for name, start, end in timing.records]
                    self.assertEqual(sum(name.endswith('_backward_inclusive') for name in names), 4)
                    self.assertEqual(sum(name.endswith('_forward_inclusive') for name in names), 4)
                    self.assertTrue(any('_gemm_' in name for name in names))
                    self.assertFalse(timing.pending or timing.backward_pending or timing.tensor_hooks)
                    self.assertEqual(torch.cuda.is_initialized(), initial_cuda_state)


if __name__ == '__main__':
    unittest.main()
