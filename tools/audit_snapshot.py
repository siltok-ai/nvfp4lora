"""本项目 CUDA 快照的派生审核：证实解码分配曾发生，反向后不再存活。"""

import argparse
import json
import pickle
from pathlib import Path

from nvfp4lora.format.preprocess import file_hash


def decode_frame(frames, family="moe"):
    return any(str(frame.get("filename", "")).replace("\\", "/").endswith("format/nvfp4.py")
               or (family == "dense" and str(frame.get("filename", "")).replace("\\", "/").endswith("ops/packed_linear_fn.py")
                   and frame.get("name") == "decode_codes") for frame in frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", choices=["moe", "dense"], default="moe")
    args = parser.parse_args()
    with args.snapshot.open("rb") as stream:
        snapshot = pickle.load(stream)
    decoded = [event for trace in snapshot["device_traces"] for event in trace
               if event["action"] == "alloc" and decode_frame(event.get("frames", []), args.family)]
    live = [block for segment in snapshot["segments"] for block in segment["blocks"] if block["state"] == "active_allocated"]
    active_decoded = [block for block in live if decode_frame(block.get("frames", []), args.family)]
    summary = {"status": "PASS" if decoded and not active_decoded else "FAIL",
               "snapshot_sha256": file_hash(args.snapshot), "decode_allocation_count": len(decoded),
               "decode_allocation_total_bytes": sum(event["size"] for event in decoded),
               "active_decode_allocation_count": len(active_decoded),
               "active_allocated_total": sum(block["size"] for block in live),
               "largest_active_allocations": [
                   {"bytes": block["size"], "frames": block.get("frames", [])}
                   for block in sorted(live, key=lambda block: block["size"], reverse=True)[:10]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "largest_active_allocations"}, indent=2))
    assert summary["status"] == "PASS"


if __name__ == "__main__":
    main()
