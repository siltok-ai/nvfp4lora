"""测量锁页内存分配及层大小的主机到显卡传输。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch


def memory():
    """记录本进程实际提交/工作集、系统可用量和 CUDA host allocator 账本。"""
    return {"process": psutil.Process().memory_info()._asdict(),
            "system_available": psutil.virtual_memory().available,
            "host_allocator": torch.cuda.memory.host_memory_stats()}


def transfer(blocks, result):
    """每次 copy 后等待完成再复用 GPU 目标；事件时间与 wall time分别记录。"""
    destination = torch.empty(max(x.numel() for x in blocks), dtype=torch.uint8, device="cuda")
    destination[:blocks[0].numel()].copy_(blocks[0], non_blocking=True)
    torch.cuda.synchronize()
    for repeat in range(3):
        events = []
        start_wall = time.perf_counter()
        for index, block in enumerate(blocks):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            destination[:block.numel()].copy_(block, non_blocking=True)
            end.record()
            end.synchronize()
            positions = torch.tensor([0, 1, 17, block.numel() // 2, block.numel() - 1])
            expected = block[positions]
            actual = destination[positions.to("cuda")].cpu()
            assert torch.equal(actual, expected), (repeat, index)
            events.append(start.elapsed_time(end))
        wall = time.perf_counter() - start_wall
        total_bytes = sum(x.numel() for x in blocks)
        result.append({"repeat": repeat, "bytes": total_bytes, "copy_count": len(blocks),
                       "cuda_copy_ms": sum(events), "wall_seconds": wall,
                       "cuda_GB_per_second": total_bytes / (sum(events) / 1000) / 1e9,
                       "wall_GB_per_second": total_bytes / wall / 1e9,
                       "block_copy_ms": events})


def main():
    """大池失败仅按已定回退改用两个锁页槽；不会调整系统或中止用户进程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-bytes", type=int, default=18_500_000_000)
    parser.add_argument("--block-bytes", type=int, default=450_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "started_at": time.time(), "initial": memory(),
              "requested_pool_bytes": args.pool_bytes, "block_bytes": args.block_bytes,
              "allocations": [], "transfers": []}
    path = args.output / "results.json"
    blocks = []
    allocated = 0
    code = 0
    try:
        try:
            while allocated < args.pool_bytes:
                size = min(args.block_bytes, args.pool_bytes - allocated)
                block = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                block.fill_(len(blocks) % 251)
                assert block.is_pinned()
                blocks.append(block)
                allocated += size
                result["allocations"].append({"bytes": allocated, "memory": memory()})
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"pinned {allocated}/{args.pool_bytes} bytes", flush=True)
            result["mode"] = "full-pinned-bank"
        except RuntimeError:
            result["full_pool_allocation_error"] = traceback.format_exc()
            result["allocated_before_error"] = allocated
            blocks.clear()
            block = None
            gc.collect()
            blocks = [torch.empty(args.block_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            for index, block in enumerate(blocks):
                block.fill_(index + 101)
            result["mode"] = "pageable-bank-with-two-pinned-staging-slots"
            bank = torch.empty(args.pool_bytes, dtype=torch.uint8)
            bank.fill_(23)
            result["pageable_bank_bytes"] = bank.numel()
            copy_start = time.perf_counter()
            for start in range(0, bank.numel(), args.block_bytes):
                count = min(args.block_bytes, bank.numel() - start)
                blocks[(start // args.block_bytes) % 2][:count].copy_(bank[start:start + count])
            result["pageable_to_staging_seconds"] = time.perf_counter() - copy_start
        transfer(blocks, result["transfers"])
        result["allocated_own_pinned_bytes"] = sum(x.numel() for x in blocks)
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["after_test"] = memory()
        result["status"] = "PASS" if result["mode"] == "full-pinned-bank" else "FALLBACK PASS"
    except Exception:
        code = 1
        result["status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    result.update(ended_at=time.time(), exit_code=code,
                  release_policy="All CUDA and pinned allocations released when this isolated process exits")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
