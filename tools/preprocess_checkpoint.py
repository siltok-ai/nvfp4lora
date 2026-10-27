"""只接受完整校验过的真实来源或显式 tiny 生成身份，启动格式预处理。"""

import argparse
import json
from pathlib import Path

import torch

from nvfp4lora.format.preprocess import preprocess

from fetch_checkpoint import REPO, REVISION


def main():
    """真实 checkpoint 必须先满足 G0.8，不能把单分片下载收据当作完整来源。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    if receipt.get("kind") == "tiny-generated":
        identity = receipt
    else:
        if receipt["status"] != "PASS" or not receipt["requested_weights"]:
            raise ValueError("G0.8 is incomplete: full checksum PASS receipt required")
        if receipt["source"]["repo"] != REPO or receipt["source"]["revision"] != REVISION:
            raise ValueError("Checkpoint source differs from fixed v2 base")
        shards = [x for x in receipt["downloaded"] if x["filename"].endswith(".safetensors")]
        if len(shards) != 3 or any(x["sha256"] != x["lfs_sha256"] for x in shards):
            raise ValueError("Expected all three hash-verified shards")
        identity = {"kind": "real-nvidia-checkpoint", "repo": REPO, "revision": REVISION,
                    "shards": [{"filename": x["filename"], "sha256": x["sha256"]} for x in shards],
                    "metadata": [{"filename": x["filename"], "sha256": x["sha256"]} for x in receipt["downloaded"]
                                 if not x["filename"].endswith(".safetensors")]}
    manifest = preprocess(args.model_dir, args.output, identity)
    print(json.dumps({"status": manifest["status"], "geometry": manifest["geometry"],
                      "nonexpert_payload_bytes": manifest["nonexpert_payload_bytes"],
                      "peak_working_set_bytes": manifest["peak_working_set_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
