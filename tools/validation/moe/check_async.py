"""用完整层字节检查异步槽覆盖、延迟竞争和错误预取。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from expert_fixture import make_parameters
from tools.validation.moe.check_experts import compare_all, measure
from nvfp4lora.ops.experts_fn import ExpertOpConfig, NVFP4ExpertsFn
from nvfp4lora.store.async_streamer import AsyncLayerStreamer
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from nvfp4lora.train.loop import configure_training


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def random_reads(store):
    """每次完整 payload D2H，与 CPU 源直接逐字节对照，不只抽样或比摘要。"""
    streamer = AsyncLayerStreamer(store)
    rng = random.Random(312)
    layers = [rng.randrange(40) for _ in range(1000)]
    started = time.perf_counter()
    for index, layer in enumerate(layers):
        with streamer.acquire(layer) as handle:
            if index + 1 < len(layers):
                streamer.prefetch(layers[index + 1])
            actual = handle.slot.buffer[:store.banks[layer].numel()].cpu()
            assert torch.equal(actual, store.banks[layer]), (index, layer)
        if (index + 1) % 100 == 0:
            print(f"random full-byte reads {index + 1}/1000", flush=True)
    streamer.collect_timings(synchronize=True)
    elapsed = time.perf_counter() - started
    result = {"status": "PASS", "full_payload_checks": 1000, "wall_seconds": elapsed,
              "wall_effective_h2d_GB_per_second": streamer.stats["copy_bytes"] / elapsed / 1e9,
              "copy_gpu_GB_per_second": streamer.stats["copy_bytes"] / (streamer.stats["copy_gpu_ms"] / 1000) / 1e9,
              "stats": dict(streamer.stats)}
    streamer.close()
    return result


def delayed_reads(store, *, wrong, requests=None):
    """GPU 读在 Python release 之后执行；8个复用输出槽限制验证器额外显存。"""
    streamer = AsyncLayerStreamer(store)
    requests = requests or [*range(40), *reversed(range(40))]
    buffers = [torch.empty_like(streamer.slots[0].buffer) for _ in range(8)]
    anchor = torch.cuda.Event(enable_timing=True)
    anchor.record()
    streamer.copy_stream.wait_event(anchor)
    computes = []
    started = time.perf_counter()
    for offset in range(0, len(requests), len(buffers)):
        batch = requests[offset:offset + len(buffers)]
        for position, layer in enumerate(batch):
            with streamer.acquire(layer) as handle:
                if wrong:
                    # 相邻访问为±1或重复层，+17始终是完全错误的下一层预测。
                    streamer.prefetch((layer + 17) % 40)
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
                torch.cuda._sleep(50000000)
                buffers[position][:store.banks[layer].numel()].copy_(handle.slot.buffer[:store.banks[layer].numel()])
                end.record()
                computes.append((begin, end))
        # 所有 release 和后续覆盖已经入队，再验证每次最后读者取得的完整数据。
        for position, layer in enumerate(batch):
            actual = buffers[position][:store.banks[layer].numel()].cpu()
            assert torch.equal(actual, store.banks[layer]), (offset + position, layer, wrong)
    torch.cuda.synchronize()
    copies = [(anchor.elapsed_time(start), anchor.elapsed_time(end)) for key, start, end in streamer.timings if key == "copy_gpu_ms"]
    compute_intervals = [(anchor.elapsed_time(start), anchor.elapsed_time(end)) for start, end in computes]
    overlap = sum(max(0., min(b, d) - max(a, c)) for a, b in copies for c, d in compute_intervals)
    streamer.collect_timings(synchronize=True)
    elapsed = time.perf_counter() - started
    result = {"status": "PASS", "full_payload_checks": len(requests), "wrong_prefetch": wrong,
              "sleep_cycles_per_request": 50000000, "wall_seconds": elapsed,
              "copy_compute_overlap_ms_from_cuda_events": overlap, "stats": dict(streamer.stats)}
    streamer.close()
    del buffers
    gc.collect()
    return result


def operator_equality(store):
    """实际P2目标形状的所有输出/输入/路由/六A/B梯度，逐字节比较同步与异步。"""
    torch.manual_seed(924)
    tokens, hidden, intermediate, experts, topk = 4096, 2048, 512, 256, 8
    scores = torch.randn(tokens, experts, device="cuda")
    indices = scores.topk(topk, dim=-1).indices
    weights = torch.rand(tokens, topk, device="cuda").softmax(-1)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    parameters = make_parameters(experts, hidden, intermediate, dtype=torch.float32, device="cuda")
    dy = torch.randn_like(x) / (tokens * hidden) ** .5
    inputs = (x, weights, *parameters)
    sync, asynchronous = SyncLayerStreamer(store), AsyncLayerStreamer(store)
    def run(streamer):
        config = ExpertOpConfig(streamer, chunk_size=32, scaling=(2., 2., 2.))
        def operation(value, probabilities, *values):
            return NVFP4ExpertsFn.apply(value, indices, probabilities, *values, 0, config)
        return measure(operation, inputs, dy)
    expected, sm = run(sync)
    actual, am = run(asynchronous)
    comparison = compare_all(actual, expected, torch.bfloat16, exact=True)
    asynchronous.collect_timings(synchronize=True)
    stats = dict(asynchronous.stats)
    asynchronous.close()
    return {"status": "PASS", "shape": [tokens, hidden, intermediate, experts, topk],
            "comparisons": comparison, "sync_memory": sm, "async_memory": am, "stats": stats}


def profiler_overlap(store, output):
    """保留 Chrome 原始跟踪，并从真实 GPU memcpy/kernel 时间区间核实重叠。"""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        event_result = delayed_reads(store, wrong=True, requests=list(range(16)))
    trace_path = output / "overlap-trace.json"
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    copies = [event for event in trace["traceEvents"] if event.get("cat") == "gpu_memcpy" and "HtoD" in event.get("name", "")]
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    pairs = []
    for copy in copies:
        for kernel in kernels:
            overlap = min(copy["ts"] + copy["dur"], kernel["ts"] + kernel["dur"]) - max(copy["ts"], kernel["ts"])
            if overlap > 0:
                pairs.append({"overlap_us": overlap, "copy": copy["name"], "kernel": kernel["name"],
                              "copy_stream": copy.get("args", {}).get("stream"), "kernel_stream": kernel.get("args", {}).get("stream")})
    result = {"status": "PASS" if pairs else "GPU_TRACE_UNAVAILABLE_OR_NO_OVERLAP", "h2d_events": len(copies),
              "kernel_events": len(kernels), "overlapping_pairs": len(pairs), "first_pairs": pairs[:8],
              "cuda_event_measurement": event_result}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    store = PackedLayerStore(args.experts, layout_version="packed-nvfp4-experts-v1")
    hashes = store.hashes()
    result = {"status": "RUNNING", "source": store.source, "cases": {}}
    for name, function in [("A3.1-random", lambda: random_reads(store)),
                           ("A3.2-delayed", lambda: delayed_reads(store, wrong=False)),
                           ("A3.3-wrong", lambda: delayed_reads(store, wrong=True)),
                           ("A3.4-operator", lambda: operator_equality(store)),
                           ("A3.5-profiler", lambda: profiler_overlap(store, args.output))]:
        print(f"Starting {name}", flush=True)
        result["cases"][name] = function()
        save(args.output / "results.json", result)
        print(f"{name}: {result['cases'][name]['status']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    result["frozen_store_hashes_unchanged"] = store.hashes() == hashes
    result["status"] = "PASS" if all(case["status"] == "PASS" for case in result["cases"].values()) and result["frozen_store_hashes_unchanged"] else "INCOMPLETE"
    save(args.output / "results.json", result)
    assert result["status"] == "PASS", result["status"]


if __name__ == "__main__":
    main()
