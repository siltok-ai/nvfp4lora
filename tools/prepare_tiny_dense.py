"""先tiny dense：生成BF16、同布局量化源、全量E1预处理及store读回。"""

import argparse
import json
from pathlib import Path

import torch

from make_tiny_dense import make_tiny_dense
from quantize_tiny_dense import quantize_tiny_dense
from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT, preprocess_dense
from nvfp4lora.store.layer_store import PackedLayerStore


def main():
    parser = argparse.ArgumentParser()
    for name in ["bf16", "nvfp4", "processed", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    make_tiny_dense(args.bf16)
    identity = quantize_tiny_dense(args.bf16, args.nvfp4)
    result = preprocess_dense(args.nvfp4, args.processed, identity)
    store = PackedLayerStore(args.processed / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
    first = store.hashes()
    assert first == PackedLayerStore(args.processed / "layers", pin_memory=False, layout_version=LAYER_LAYOUT).hashes()
    assert result["geometry"]["excluded"] and any(name.startswith("mtp.") for name in result["geometry"]["excluded"])
    assert any(name.startswith("model.visual.") for name in result["geometry"]["excluded"])
    summary = {"status": result["status"], "scope": "D1 tiny only", "source": identity, "geometry": result["geometry"],
               "full_e1": result["e1_full"], "store_hashes": first, "store_reloaded_hashes_equal": True,
               "peak_working_set_bytes_preprocessing": result["peak_working_set_bytes"],
               "paths": {name: str(getattr(args, name)) for name in ["bf16", "nvfp4", "processed"]}}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS: tiny dense D1", flush=True)


if __name__ == "__main__":
    main()
