"""真实 40 层 CPU pinned bank 与同步双槽的实际数据资格。"""

import argparse
import json
import time
from pathlib import Path

import psutil
import torch

from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer


def main():
    """读取所有层的完整 GPU payload，不能用句柄元数据相等代替数据检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    store = PackedLayerStore(args.experts, pin_memory=True, layout_version="packed-nvfp4-experts-v1")
    result = {"status": "RUNNING", "bank_load_seconds": time.perf_counter() - start,
              "logical_cpu_bank_bytes": store.nbytes, "cpu_after_load": psutil.Process().memory_info()._asdict(),
              "pinned_allocator_after_load": torch.cuda.memory.host_memory_stats(), "checks": []}
    before = store.hashes()
    torch.cuda.reset_peak_memory_stats()
    streamer = SyncLayerStreamer(store, slots=2)
    for layer in [*range(40), *reversed(range(40))]:
        with streamer.acquire(layer) as handle:
            count = store.entries[layer]["payload_bytes"]
            actual = handle.slot.buffer[:count].cpu()
            if not torch.equal(actual, store.banks[layer]):
                raise ValueError(f"GPU payload differs at layer {layer}")
            result["checks"].append({"layer": layer, "bytes_compared": count})
            del actual
        if len(result["checks"]) % 8 == 0:
            print("checked", len(result["checks"]), "layer transfers", flush=True)
    assert before == store.hashes()
    result.update(status="PASS", total_seconds=time.perf_counter() - start, transfers=streamer.stats,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  cpu_end=psutil.Process().memory_info()._asdict(), frozen_bank_hashes=before,
                  scope="real synchronous store only; not asynchronous A3 qualification")
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
