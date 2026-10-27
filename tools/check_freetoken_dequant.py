"""在 Speed 原环境只读调用 FreeToken，独立对照 NVFP4 数值与 global 精度。"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4

from nvfp4lora.format.nvfp4 import dequantize


def ulp_distance(a, b):
    """BF16 按有序整数编码比较 ULP，正负零距离定义为零。"""
    def ordered(x):
        raw = x.contiguous().view(torch.int16).to(torch.int32) & 65535
        return torch.where(raw & 32768 != 0, 32768 - (raw & 32767), 32768 + (raw & 32767))
    return int((ordered(a) - ordered(b)).abs().max())


def main():
    """raw FP32 global 和运行时 FP16 global 分开记录，避免混淆基座误差。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    rows, width = 512, 2048
    packed = torch.randint(0, 256, (rows, width // 2), dtype=torch.uint8)
    scales = torch.randint(0, 127, (rows, width // 16), dtype=torch.uint8)
    scalar = torch.tensor(0.000372931, dtype=torch.float32)
    reference = dequantize(packed, scales, scalar, dtype=torch.bfloat16)
    global_f32 = scalar.expand(rows).contiguous()
    actual = _dequant_nvfp4(packed, scales.view(torch.float8_e4m3fn), global_f32)
    maximum_ulp = ulp_distance(actual, reference)
    assert maximum_ulp <= 1, maximum_ulp
    runtime = _dequant_nvfp4(packed, scales.view(torch.float8_e4m3fn), global_f32.to(torch.float16))
    source = Path(inspect.getfile(_dequant_nvfp4))
    result = {"status": "PASS", "shape": [rows, width], "raw_global_f32": float(scalar),
              "runtime_global_f16": float(scalar.to(torch.float16)),
              "torch_vs_freetoken_raw_f32_max_ulp": maximum_ulp,
              "freetoken_runtime_f16_vs_raw_f32_max_ulp": ulp_distance(runtime, actual),
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "G0.7 synthetic target projection; A1.3 real-weight sampling still pending"}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
