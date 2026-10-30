"""生成指定族的 tiny 来源，并调用共用预处理入口生成 bundle。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from nvfp4lora.families import family_named


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["moe", "dense"], required=True)
    for name in ["bf16", "nvfp4", "processed", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    family = family_named(args.family)
    family.make_tiny(args.bf16)
    identity = family.quantize_tiny(args.bf16, args.nvfp4)
    receipt = args.output / "tiny-source.json"
    receipt.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("preprocess_checkpoint.py")),
                    "--model-dir", str(args.nvfp4), "--output", str(args.processed),
                    "--source-receipt", str(receipt), "--family", args.family], check=True)
    result = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    summary = {"status": result["status"], "family": family.name, "source": identity, "geometry": result["geometry"],
               "paths": {name: str(getattr(args, name)) for name in ["bf16", "nvfp4", "processed"]}}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS: tiny source and bundle", flush=True)


if __name__ == "__main__":
    main()
