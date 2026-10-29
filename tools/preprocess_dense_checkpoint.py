"""要求完整固定来源收据的dense真实checkpoint预处理入口。"""

import argparse
import json
from pathlib import Path

import torch

from nvfp4lora.format.preprocess_dense import preprocess_dense

CANDIDATES = {"RadixArk/Qwen3.8-27B-NVFP4": "319f741cce68d7914884900c138a1fbb70a42f30",
              "nvidia/Qwen3.8-27B-NVFP4": "482ca0f3832238542f8f5295dde86b5f22711d80",
              "nvidia/Qwen3.6-27B-NVFP4": "0893e1606ff3d5f97a441f405d5fc541a6bdf404"}


def identity_from_receipt(receipt):
    """单分片、元数据收据或不匹配的LFS声明均不能获得真实基座身份。"""
    if receipt.get("status") != "PASS" or receipt.get("requested_weights") is not True:
        raise ValueError("DA0.4 requires a complete three-shard checksum PASS receipt")
    repo, revision = receipt["source"]["repo"], receipt["source"]["revision"]
    if CANDIDATES.get(repo) != revision:
        raise ValueError("Dense source differs from fixed design candidates")
    shards = [entry for entry in receipt["downloaded"] if entry["filename"].endswith(".safetensors")]
    declared = {entry["filename"]: entry["lfs_sha256"] for entry in receipt["source"]["files"] if entry["filename"].endswith(".safetensors")}
    if (len(shards) != 3 or len(declared) != 3 or {entry["filename"] for entry in shards} != declared.keys()
            or any(entry["sha256"] != declared[entry["filename"]] or entry["sha256"] != entry["lfs_sha256"] for entry in shards)):
        raise ValueError("Expected all three upstream LFS SHA256 matches")
    metadata = [{"filename": entry["filename"], "sha256": entry["sha256"]} for entry in receipt["downloaded"]
                if not entry["filename"].endswith(".safetensors")]
    if not {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"} <= {entry["filename"] for entry in metadata}:
        raise ValueError("Dense receipt is missing source config/tokenizer metadata")
    return {"kind": "real-dense-checkpoint", "repo": repo, "revision": revision,
            "shards": [{"filename": entry["filename"], "sha256": entry["sha256"]} for entry in shards], "metadata": metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    identity = identity_from_receipt(json.loads(args.source_receipt.read_text(encoding="utf-8")))
    manifest = preprocess_dense(args.model_dir, args.output, identity)
    print(json.dumps({"status": manifest["status"], "geometry": manifest["geometry"],
                      "nonstream_payload_bytes": manifest["nonstream_payload_bytes"],
                      "peak_working_set_bytes": manifest["peak_working_set_bytes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
