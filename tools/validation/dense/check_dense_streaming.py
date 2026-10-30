"""检查密集模型的层字节读取、异步槽竞争及算子确定性。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import psutil
import torch

from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT
from nvfp4lora.model.packed_linear import LoRAPackedLinear, PackedLinear
from nvfp4lora.ops.packed_linear_fn import PackedLinearOpConfig
from nvfp4lora.store.async_streamer import AsyncLayerStreamer
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from nvfp4lora.train.loop import configure_training


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class MeasuredAsync(AsyncLayerStreamer):
    """仅验证器记录逐层事件；调度、pin、generation仍执行原实现。"""

    def __init__(self, store, *, prefetch_mode="normal"):
        super().__init__(store, slots=2)
        self.prefetch_mode, self.active_layer = prefetch_mode, None
        self.by_layer = defaultdict(lambda: {"acquires": 0, "cache_hits": 0, "copies": 0,
                                             "copy_bytes": 0, "failed_acquires": 0, "acquire_enqueue_seconds": 0.})
        self.copy_events, self.wait_events = defaultdict(list), defaultdict(list)

    def acquire(self, layer):
        offset, hits, start = len(self.timings), self.stats["cache_hits"], time.perf_counter()
        before, success = self.stats["acquires"], False
        try:
            handle = super().acquire(layer)
            self.active_layer, success = layer, True
            return handle
        finally:
            # The original streamer counts a demand before discovering all slots
            # are pinned; a rejected cross-stream request is not counted.
            counted = self.stats["acquires"] - before
            if counted:
                row = self.by_layer[layer]
                row["acquires"] += counted
                row["failed_acquires"] += int(not success)
                row["cache_hits"] += self.stats["cache_hits"] - hits
                row["acquire_enqueue_seconds"] += time.perf_counter() - start
                self.wait_events[layer].extend((a, b) for key, a, b in self.timings[offset:] if key == "acquire_wait_gpu_ms")

    def _copy(self, slot, layer):
        offset = len(self.timings)
        super()._copy(slot, layer)
        row = self.by_layer[layer]
        row["copies"] += 1
        row["copy_bytes"] += self.store.banks[layer].numel()
        self.copy_events[layer].extend((a, b) for key, a, b in self.timings[offset:] if key == "copy_gpu_ms")

    def prefetch(self, layer):
        if self.prefetch_mode == "disabled":
            return False
        if self.prefetch_mode == "wrong":
            layer = self.wrong_target(self.active_layer)
        return super().prefetch(layer)

    def wrong_target(self, layer):
        count = len(self.store.entries)
        return (layer + (17 if count > 18 else 2)) % count

    def measurement(self):
        self.collect_timings(synchronize=True)
        for layer, row in self.by_layer.items():
            row["copy_gpu_ms"] = sum(a.elapsed_time(b) for a, b in self.copy_events[layer])
            row["acquire_wait_gpu_ms"] = sum(a.elapsed_time(b) for a, b in self.wait_events[layer])
        assert sum(row["acquires"] for row in self.by_layer.values()) == self.stats["acquires"]
        assert sum(row["copy_bytes"] for row in self.by_layer.values()) == self.stats["copy_bytes"]
        return {"stats": dict(self.stats), "per_layer": dict(self.by_layer)}


def random_reads(store):
    streamer = MeasuredAsync(store)
    rng = random.Random(312)
    layers = [rng.randrange(len(store.entries)) for _ in range(1000)]
    assert set(layers) == set(store.entries)
    start, digest, compared = time.perf_counter(), hashlib.sha256(), 0
    for index, layer in enumerate(layers):
        with streamer.acquire(layer) as handle:
            if index + 1 < len(layers):
                streamer.prefetch(layers[index + 1])
            value = handle.slot.buffer[:store.banks[layer].numel()].cpu()
            assert torch.equal(value, store.banks[layer]), (index, layer)
            digest.update(memoryview(value.numpy()))
            compared += value.numel()
            del value
        if (index + 1) % 100 == 0:
            print(f"full-byte random reads {index + 1}/1000", flush=True)
    result = {"status": "PASS", "full_payload_checks": 1000, "all_layers_covered": True,
              "compared_bytes": compared, "all_reads_sha256": digest.hexdigest(),
              "wall_seconds": time.perf_counter() - start, **streamer.measurement()}
    streamer.close()
    return result


def delayed_reads(store, *, wrong):
    """Python release后GPU继续读；跨流覆盖必须等待last_reader。"""
    streamer = MeasuredAsync(store, prefetch_mode="wrong" if wrong else "disabled")
    buffers = [torch.empty_like(streamer.slots[0].buffer) for _ in range(8)]
    layers = [*range(len(store.entries)), *reversed(range(len(store.entries)))]
    anchor = torch.cuda.Event(enable_timing=True)
    anchor.record()
    streamer.copy_stream.wait_event(anchor)
    computes, digest = [], hashlib.sha256()
    start = time.perf_counter()
    for offset in range(0, len(layers), len(buffers)):
        batch = layers[offset:offset + len(buffers)]
        for position, layer in enumerate(batch):
            with streamer.acquire(layer) as handle:
                if wrong:
                    target = streamer.wrong_target(layer)
                    next_layer = layers[offset + position + 1] if offset + position + 1 < len(layers) else None
                    assert target != layer and target != next_layer
                    streamer.prefetch(target)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                torch.cuda._sleep(50000000)
                buffers[position][:store.banks[layer].numel()].copy_(handle.slot.buffer[:store.banks[layer].numel()])
                end.record()
                computes.append((begin, end))
        for position, layer in enumerate(batch):
            value = buffers[position][:store.banks[layer].numel()].cpu()
            assert torch.equal(value, store.banks[layer]), (offset + position, layer, wrong)
            digest.update(memoryview(value.numpy()))
            del value
    torch.cuda.synchronize()
    copies = [(anchor.elapsed_time(a), anchor.elapsed_time(b)) for values in streamer.copy_events.values() for a, b in values]
    kernels = [(anchor.elapsed_time(a), anchor.elapsed_time(b)) for a, b in computes]
    overlap = sum(max(0., min(b, d) - max(a, c)) for a, b in copies for c, d in kernels)
    result = {"status": "PASS", "full_payload_checks": len(layers), "wrong_prefetch": wrong,
              "sleep_cycles_per_request": 50000000, "all_reads_sha256": digest.hexdigest(),
              "copy_compute_overlap_ms_from_cuda_events": overlap,
              "wall_seconds": time.perf_counter() - start, **streamer.measurement()}
    streamer.close()
    return result


def handle_guards(store):
    streamer = MeasuredAsync(store)
    checks = {}
    def fails(name, function, text):
        try:
            function()
        except RuntimeError as exc:
            assert text in str(exc), (name, str(exc))
            checks[name] = True
        else:
            raise AssertionError(name)
    first, second = streamer.acquire(0), streamer.acquire(1)
    assert not streamer.prefetch(2)
    fails("all_pinned_acquire", lambda: streamer.acquire(2), "pinned")
    fails("all_pinned_close", streamer.close, "pinned")
    streamer.release(first)
    with streamer.acquire(2) as third:
        assert third.generation > first.generation
    fails("stale_handle", first.tensors, "stale")
    fails("double_release", lambda: streamer.release(first), "Invalid")
    alternate = torch.cuda.Stream()
    with torch.cuda.stream(alternate):
        fails("cross_compute_stream", lambda: streamer.acquire(3), "one CUDA compute")
    streamer.release(second)
    stats = streamer.measurement()
    streamer.close()
    fails("closed_streamer", lambda: streamer.acquire(0), "closed")
    return {"status": "PASS", "checks": checks, **stats}


def operator_equality(store, tokens):
    sync = SyncLayerStreamer(store)
    normal = MeasuredAsync(store, prefetch_mode="disabled")
    wrong = MeasuredAsync(store, prefetch_mode="wrong")
    rows = []
    # Every module kind in one GDN and one full-attention layer.
    for layer in [0, 3]:
        for name, spec in store.geometry["modules"][str(layer)].items():
            x = torch.randn(1, tokens, spec["in_features"], dtype=torch.bfloat16, device="cuda")
            dy = torch.randn(1, tokens, spec["out_features"], dtype=torch.bfloat16, device="cuda")
            initial = None
            outputs = []
            for streamer in [sync, normal, wrong]:
                cfg = PackedLinearOpConfig(streamer)
                module = LoRAPackedLinear(PackedLinear(layer, name, spec["kind"], spec["in_features"], spec["out_features"], cfg), 16, 32, device="cuda")
                if initial is None:
                    with torch.no_grad():
                        module.lora_B["default"].weight.normal_(std=.001)
                    initial = {key: value.detach().clone() for key, value in module.state_dict().items()}
                else:
                    module.load_state_dict(initial, strict=True)
                value = x.detach().requires_grad_()
                streamer.phase = "FORWARD"
                y = module(value)
                streamer.phase = "BACKWARD"
                grads = torch.autograd.grad(y, [value, module.lora_A["default"].weight, module.lora_B["default"].weight], dy)
                outputs.append([item.detach().cpu() for item in [y, *grads]])
                del module, value, y, grads
            comparison = {mode: {key: same(a, b) for key, a, b in zip(["output", "dx", "dA", "dB"], output, outputs[0])}
                          for mode, output in zip(["async_no_prefetch", "async_wrong_prefetch"], outputs[1:])}
            assert all(all(values.values()) for values in comparison.values()), (layer, name, comparison)
            rows.append({"layer": layer, "name": name, "shape": [tokens, spec["in_features"], spec["out_features"]], "E2": comparison})
            del x, dy, outputs, initial
            print(f"sync / async / wrong prefetch E2: {layer}.{name}", flush=True)
    result = {"status": "PASS", "cases": rows, "sync_stats": dict(sync.stats),
              "async_no_prefetch": normal.measurement(), "async_wrong_prefetch": wrong.measurement()}
    normal.close()
    wrong.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", default=4096, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261029)
    torch.cuda.init()
    start = time.perf_counter()
    store = PackedLayerStore(args.bundle / "layers", layout_version=LAYER_LAYOUT, pin_memory=True)
    hashes = store.hashes()
    result = {"status": "RUNNING", "source": store.source, "layers": len(store.entries),
              "logical_bank_bytes": store.nbytes, "bank_load_and_hash_seconds": time.perf_counter() - start,
              "pinned_allocator_after_load": torch.cuda.memory.host_memory_stats(),
              "cpu_after_load": psutil.Process().memory_info()._asdict(), "cases": {}}
    path = args.output / "results.json"
    def save():
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save()
    try:
        functions = [("handle_guards", lambda: handle_guards(store)), ("DA3.1_random", lambda: random_reads(store)),
                     ("DA3.2_delayed", lambda: delayed_reads(store, wrong=False)),
                     ("DA3.3_wrong", lambda: delayed_reads(store, wrong=True)),
                     ("DA3.4_operator", lambda: operator_equality(store, args.tokens))]
        for name, function in functions:
            print(f"Starting {name}", flush=True)
            result["cases"][name] = function()
            save()
            gc.collect()
            torch.cuda.empty_cache()
        assert result["cases"]["DA3.2_delayed"]["all_reads_sha256"] == result["cases"]["DA3.3_wrong"]["all_reads_sha256"]
        assert hashes == store.hashes()
        free, total = torch.cuda.mem_get_info()
        result.update(status="PASS", frozen_store_hashes_unchanged=True, frozen_store_hashes=hashes,
                      total_seconds=time.perf_counter() - start,
                      gpu_peak_allocated=torch.cuda.max_memory_allocated(), gpu_peak_reserved=torch.cuda.max_memory_reserved(),
                      gpu_free_at_end=free, gpu_total=total, cpu_end=psutil.Process().memory_info()._asdict(),
                      pinned_allocator_at_end=torch.cuda.memory.host_memory_stats())
    except Exception:
        result.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
