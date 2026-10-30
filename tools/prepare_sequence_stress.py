"""构造完整长答案的2048/4096单样本训练压力材料，仅用于资源/机制验证。"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from nvfp4lora.format.bundle import file_hash
from nvfp4lora.train.template import render_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    pieces = []
    while len(tokenizer.encode(" ".join(pieces), add_special_tokens=False)) < 1024:
        index = len(pieces)
        pieces.append(f"entry {index}: alpha beta {index % 7}.")
    answer = " ".join(pieces)
    rows, geometry = [], {"2048": [], "4096": []}
    for row in range(8):
        prompt = "\n".join(f"record {index + row}: compare input {index % 97} and output {index % 83}; preserve the version and checksum."
                           for index in range(1000)) + "\n请继续输出用于显存压力测试的固定记录。"
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        rows.append({"family": "sequence-memory-stress", "messages": messages})
        for context in [2048, 4096]:
            sample = render_sample(tokenizer, messages, max_length=context)
            assert sample is not None and sample["input_ids"].shape[-1] == context
            supervised = int((sample["labels"][..., 1:] != -100).sum())
            assert 1024 <= supervised < 1280
            geometry[str(context)].append({"row": row, "input_tokens": context, "supervised_tokens": supervised, "left_trimmed": sample["left_trimmed"]})
    path = args.output / "stress.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({"status": "PASS", "layout": "sequence-memory-stress-v1",
        "source": "task-generated synthetic fixture", "scope": "Memory/training mechanics only; no canary or quality claim",
        "file_sha256": file_hash(path), "rows": len(rows), "geometry": geometry}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "geometry": geometry}, indent=2))


if __name__ == "__main__":
    main()
