"""工程基线比较必须能检测参数和训练指标的一位变化。"""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from tools.validation.compare_tiny_baseline import compare_runs, equal_bits


class TinyBaselineTests(unittest.TestCase):
    def test_parameter_bit_and_metric_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ['baseline', 'actual']]
            tensor = torch.tensor([[1., 2.]], dtype=torch.float32)
            for root in roots:
                root.mkdir()
                (root / 'metrics.jsonl').write_text(json.dumps({'loss': 1., 'global_step': 1,
                    'sample_ids': [7, 2], 'backward_seconds': 1. if root == roots[0] else 2.}) + '\n')
                for boundary in ['checkpoint-middle', 'checkpoint-final']:
                    (root / boundary).mkdir()
                    save_file({'adapter': tensor}, root / boundary / 'lora_params.safetensors')
            self.assertEqual(compare_runs(*roots)['status'], 'PASS')
            poisoned = tensor.clone()
            poisoned.view(torch.uint8).reshape(-1)[0] ^= 1
            save_file({'adapter': poisoned}, roots[1] / 'checkpoint-final/lora_params.safetensors')
            self.assertIn('checkpoint-final.adapter', compare_runs(*roots)['differences'])
            save_file({'adapter': tensor}, roots[1] / 'checkpoint-final/lora_params.safetensors')
            (roots[1] / 'metrics.jsonl').write_text(json.dumps({'loss': 1.0000000000000002,
                'global_step': 1, 'sample_ids': [7, 2], 'backward_seconds': 2.}) + '\n')
            self.assertIn('numerical_metrics', compare_runs(*roots)['differences'])

    def test_signed_zero_and_scalar_types_are_not_collapsed(self):
        self.assertFalse(equal_bits(0., -0.))
        self.assertFalse(equal_bits(1., 1))
        self.assertFalse(equal_bits(True, 1))
