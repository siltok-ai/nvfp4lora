"""固定 revision 的模型获取：先来源元数据，完整分片必须再做 SHA256 校验。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
METADATA_FILES = {"config.json", "hf_quant_config.json", "generation_config.json",
                  "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "vocab.json", "README.md", "LICENSE", "LICENSE.txt"}


def sha256(path):
    """顺序读取文件校验原始分片，避免完整文件进入内存。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """匿名读取固定模型来源；只在明确选择 download 时获取大分片。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--shard", action="append", help="只获取指定分片；不构成完整 G0.8 验收")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION, info.sha
    files = []
    for sibling in info.siblings:
        lfs = sibling.lfs
        files.append({"filename": sibling.rfilename, "bytes": sibling.size,
                      "lfs_sha256": lfs.sha256 if lfs else None})
    source = {"repo": REPO, "revision": info.sha, "observed_at": time.time(), "files": files}
    (args.output / "upstream-source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    selected = [x for x in files if x["filename"] in METADATA_FILES]
    weights = [x for x in files if x["filename"].endswith(".safetensors")]
    assert len(weights) == 3 and all(x["lfs_sha256"] for x in weights), weights
    if args.shard:
        if not args.download_weights or not set(args.shard).issubset({x["filename"] for x in weights}):
            parser.error("--shard requires --download-weights and known shard names")
        weights = [x for x in weights if x["filename"] in args.shard]
    if args.download_weights:
        selected += weights
    receipt = {"source": source, "requested_weights": args.download_weights, "downloaded": [], "status": "RUNNING"}
    receipt_path = args.output / "receipt.json"
    for entry in selected:
        print("fetch", entry["filename"], flush=True)
        path = Path(hf_hub_download(REPO, filename=entry["filename"], revision=REVISION,
                                   local_dir=args.model_dir, token=False))
        actual_hash = sha256(path)
        size = path.stat().st_size
        expected_hash = entry["lfs_sha256"]
        if entry["bytes"] is not None and size != entry["bytes"]:
            raise RuntimeError(f"Size mismatch: {path}")
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch: {path}; expected {expected_hash}, got {actual_hash}")
        receipt["downloaded"].append({**entry, "sha256": actual_hash, "bytes": size})
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["status"] = ("SHARD PASS; FULL G0.8 NOT RUN" if args.shard else "PASS") if args.download_weights else "METADATA ONLY; G0.8 NOT RUN"
    receipt["ended_at"] = time.time()
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    main()
