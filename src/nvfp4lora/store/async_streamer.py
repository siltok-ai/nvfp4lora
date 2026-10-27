"""一计算流＋一拷贝流；pin 管理 CPU 协议，事件管理 GPU 实际存活期。"""

from dataclasses import dataclass
import time

import torch

from .streamer import LayerHandle, SyncLayerStreamer, _Slot


@dataclass
class _AsyncSlot(_Slot):
    copy_done: object = None
    last_reader: object = None


class AsyncLayerStreamer(SyncLayerStreamer):
    """原字节层槽的异步调度；预测只改善命中率，不参与正确性判断。"""

    def __init__(self, store, *, device="cuda", slots=2):
        if torch.device(device).type != "cuda" or not all(bank.is_pinned() for bank in store.banks.values()):
            raise ValueError("Async streaming requires CUDA and a persistent pinned CPU bank")
        super().__init__(store, device=device, slots=slots)
        self.slots = [_AsyncSlot(slot.buffer) for slot in self.slots]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # 槽由构造流分配；拷贝流先等待该流的分配/先前使用完成。
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.predicted_layer = None
        self.closed = False
        self.timings = []
        self.stats.update(prefetches=0, prefetch_copies=0, prefetch_skipped_pinned=0,
                          demand_misses=0, copy_gpu_ms=0.0, acquire_wait_gpu_ms=0.0,
                          copy_enqueue_seconds=0.0, acquire_wait_enqueue_seconds=0.0)

    def _check_stream(self):
        if self.closed:
            raise RuntimeError("AsyncLayerStreamer is closed")
        super()._check_stream()

    def _victim(self, *, protect_prediction):
        available = [slot for slot in self.slots if slot.pins == 0]
        if protect_prediction:
            unpredicted = [slot for slot in available if slot.layer != self.predicted_layer]
            if unpredicted:
                available = unpredicted
        return min(available, key=lambda slot: slot.last_use) if available else None

    def _copy(self, slot, layer):
        """覆盖前等待最后读者；相同拷贝流也顺序保护尚未完成的错误预取。"""
        data = self.store.banks[layer]
        started = time.perf_counter()
        with torch.cuda.stream(self.copy_stream):
            if slot.last_reader is not None:
                self.copy_stream.wait_event(slot.last_reader)
            start = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            start.record(self.copy_stream)
            slot.buffer[:data.numel()].copy_(data, non_blocking=True)
            done.record(self.copy_stream)
            # 对象销毁也不能让 allocator 在拷贝完成前回收槽。
            slot.buffer.record_stream(self.copy_stream)
        self.stats["copy_enqueue_seconds"] += time.perf_counter() - started
        self.stats["copies"] += 1
        self.stats["copy_bytes"] += data.numel()
        self.timings.append(("copy_gpu_ms", start, done))
        slot.layer, slot.generation = layer, slot.generation + 1
        slot.copy_done, slot.last_reader = done, None

    def acquire(self, layer):
        """明确层号找到正确 generation；返回前在计算流排入 copy_done 等待。"""
        self._check_stream()
        if layer not in self.store.entries:
            raise KeyError(layer)
        self.stats["acquires"] += 1
        self.clock += 1
        slot = next((slot for slot in self.slots if slot.layer == layer), None)
        if slot is None:
            slot = self._victim(protect_prediction=True)
            if slot is None:
                raise RuntimeError("All expert slots are pinned")
            self.stats["demand_misses"] += 1
            self._copy(slot, layer)
        else:
            self.stats["cache_hits"] += 1
        current = torch.cuda.current_stream(self.device)
        started = time.perf_counter()
        before, after = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        before.record(current)
        current.wait_event(slot.copy_done)
        after.record(current)
        self.timings.append(("acquire_wait_gpu_ms", before, after))
        self.stats["acquire_wait_enqueue_seconds"] += time.perf_counter() - started
        slot.buffer.record_stream(current)
        slot.last_use, slot.pins = self.clock, slot.pins + 1
        return LayerHandle(self, slot, layer)

    def release(self, handle):
        """Python release 只表示最后读操作已经入队；copy 流仍必须等待此事件。"""
        self._check_stream()
        if handle.streamer is not self or handle.released or handle.slot.generation != handle.generation:
            raise RuntimeError("Invalid layer handle release")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        handle.slot.last_reader = event
        handle.slot.pins -= 1
        handle.released = True

    def prefetch(self, layer):
        """猜错可产生多余拷贝；没有可覆盖槽时立即返回，不等待或覆盖 pin。"""
        self._check_stream()
        if layer not in self.store.entries:
            raise KeyError(layer)
        self.predicted_layer = layer
        self.stats["prefetches"] += 1
        if any(slot.layer == layer for slot in self.slots):
            return True
        slot = self._victim(protect_prediction=False)
        if slot is None:
            self.stats["prefetch_skipped_pinned"] += 1
            return False
        self._copy(slot, layer)
        self.stats["prefetch_copies"] += 1
        return True

    def collect_timings(self, *, synchronize=False):
        """默认只回收已完成计时；测试/报告可显式同步，训练不逐层阻塞。"""
        pending = []
        for key, start, end in self.timings:
            if synchronize:
                end.synchronize()
            if end.query():
                self.stats[key] += start.elapsed_time(end)
            else:
                pending.append((key, start, end))
        self.timings = pending
        return dict(self.stats)

    def close(self):
        """显式结束传输后才允许释放持久 CPU bank；不能关闭仍有活跃句柄的 streamer。"""
        if self.closed:
            return
        self._check_stream()
        if any(slot.pins for slot in self.slots):
            raise RuntimeError("Cannot close pinned expert slots")
        self.copy_stream.synchronize()
        for slot in self.slots:
            if slot.last_reader is not None:
                slot.last_reader.synchronize()
        self.collect_timings(synchronize=True)
        self.closed = True
