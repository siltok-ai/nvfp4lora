"""顺序保存 HF tiny BF16 来源、同布局 NVFP4 来源和训练预处理产物。"""

import argparse
import json
from pathlib import Path

import torch

from nvfp4lora.format.preprocess_moe import preprocess

from make_tiny_model import make_tiny_model
from quantize_tiny import quantize_tiny


def main():
    """三个输出根均须不存在；生成来源与真实 checkpoint 的身份始终区分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--nvfp4", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    make_tiny_model(args.bf16)
    identity = quantize_tiny(args.bf16, args.nvfp4)
    result = preprocess(args.nvfp4, args.processed, identity)
    summary = {"status": result["status"], "source": identity, "geometry": result["geometry"],
               "paths": {"bf16": str(args.bf16), "nvfp4": str(args.nvfp4), "processed": str(args.processed)}}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
