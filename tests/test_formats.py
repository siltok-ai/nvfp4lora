"""用独立标量位运算和真实 safetensors 文件验证格式边界。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from nvfp4lora.format import fp8, nvfp4
from nvfp4lora.format.inspect import inspect_checkpoint, read_header


def e4m3_scalar(raw):
    """独立解释 E4M3FN 符号、指数及次正规数，不调用 Torch FP8 转换。"""
    sign = -1 if raw & 128 else 1
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    if exponent == 0:
        return sign * mantissa * 2 ** -9
    if exponent == 15 and mantissa == 7:
        return math.nan
    return sign * (1 + mantissa / 8) * 2 ** (exponent - 7)


class NVFP4Tests(unittest.TestCase):
    def test_all_finite_e4m3_patterns_and_all_nibbles(self):
        raw_scales = [x for x in range(256) if x not in [127, 255]]
        packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
        packed = packed.repeat(len(raw_scales), 1)
        scales = torch.tensor(raw_scales, dtype=torch.uint8).reshape(-1, 1)
        expected = torch.tensor([[e4m3_scalar(s) * v for v in
                                  [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]]
                                 for s in raw_scales], dtype=torch.float32)
        actual = nvfp4.dequantize(packed, scales)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # 不含全局缩放的值必须能在 BF16 中精确表示。
        torch.testing.assert_close(actual.to(torch.bfloat16).float(), expected, rtol=0, atol=0)

    def test_nibble_order_and_ties_to_even(self):
        value = torch.tensor([[0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5]])
        part = nvfp4.quantize(value, torch.tensor(1.0))
        expected_codes = [0, 0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14]
        expected_packed = [0x00, 0x22, 0x44, 0x66, 0x87, 0xAA, 0xCC, 0xEE]
        self.assertEqual(part["weight"].tolist(), [expected_packed])
        decoded = nvfp4.dequantize(part["weight"], part["weight_scale"], part["weight_scale_2"])
        expected = torch.tensor([[nvfp4.E2M1[c] for c in expected_codes]])
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)

    def test_per_expert_global_scale_and_zero_matrix(self):
        torch.manual_seed(41)
        value = torch.randn(3, 32, 64)
        value[0].zero_()
        value[2] *= 0.001
        part = nvfp4.quantize(value)
        actual = nvfp4.dequantize(part["weight"], part["weight_scale"], part["weight_scale_2"])
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual[0].count_nonzero(), 0)
        for expert in [1, 2]:
            error = (actual[expert] - value[expert]).norm() / value[expert].norm()
            self.assertLess(float(error), 0.15)
        individual = [nvfp4.dequantize(part["weight"][e], part["weight_scale"][e], part["weight_scale_2"][e])
                      for e in range(3)]
        self.assertTrue(torch.equal(actual, torch.stack(individual)))

    def test_fp8_scaling_is_explicit(self):
        weight = torch.tensor([[.5, -448, 1.125, 0]], dtype=torch.float8_e4m3fn)
        actual = fp8.dequantize(weight.view(torch.uint8), torch.tensor(.125))
        self.assertTrue(torch.equal(actual, torch.tensor([[.0625, -56, .140625, 0]], dtype=torch.bfloat16)))

    def test_wrong_scale_geometry_and_nonfinite_source_fail(self):
        with self.assertRaises(ValueError):
            nvfp4.dequantize(torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 3, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            nvfp4.quantize(torch.full((2, 16), float("nan")))


class HeaderTests(unittest.TestCase):
    def test_actual_payload_and_index_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"a": torch.arange(32, dtype=torch.uint8).reshape(4, 8),
                       "b": torch.tensor([.125], dtype=torch.float32)}, root / "part.safetensors")
            header = read_header(root / "part.safetensors")
            self.assertEqual(header["payload_bytes"], 36)
            index = {"metadata": {"total_size": 36}, "weight_map": {"a": "part.safetensors", "b": "part.safetensors"}}
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index))
            report = inspect_checkpoint(root)
            self.assertEqual(report["tensors"]["a"]["shape"], [4, 8])
            del index["weight_map"]["b"]
            index_path.write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                inspect_checkpoint(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
