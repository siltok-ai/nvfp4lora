"""同步基线层槽；GPU 操作限定于同一计算流，重用依靠该流的顺序。"""

from dataclasses import dataclass
import time

import torch

from nvfp4lora.store.layer_store import tensor_views


@dataclass
class _Slot:
    buffer: torch.Tensor
    layer: int = -1
    generation: int = 0
    pins: int = 0
    last_use: int = 0


class LayerHandle:
    """限定一次 acquire 的层号和 generation；退出上下文后不得再读取视图。"""

    def __init__(self, streamer, slot, layer):
        self.streamer, self.slot, self.layer = streamer, slot, layer
        self.generation = slot.generation
        self.released = False

    def tensors(self):
        """每次从当前有效槽生成视图，禁止使用已释放或过期的句柄。"""
        if self.released or self.slot.layer != self.layer or self.slot.generation != self.generation:
            raise RuntimeError("Released or stale layer handle")
        return tensor_views(self.slot.buffer, self.streamer.store.entries[self.layer]["tensors"])

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        self.streamer.release(self)


class SyncLayerStreamer:
    """最小同步 store→层槽实现；不预取，作为异步实现的独立调度基线。"""

    def __init__(self, store, *, device="cuda", slots=2):
        if slots < 1:
            raise ValueError("At least one layer slot required")
        self.store = store
        self.device = torch.device(device)
        largest = max(entry["payload_bytes"] for entry in store.entries.values())
        self.slots = [_Slot(torch.empty(largest, dtype=torch.uint8, device=self.device)) for _ in range(slots)]
        self.phase = "FORWARD"
        self.compute_stream = None
        self.clock = 0
        self.stats = {"acquires": 0, "cache_hits": 0, "copies": 0, "copy_bytes": 0, "copy_wall_seconds": 0.0}

    def _check_stream(self):
        """显式拒绝跨计算流重用；同步基线不依赖引用计数推断 GPU 完成。"""
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device).cuda_stream
            if self.compute_stream is None:
                self.compute_stream = current
            elif self.compute_stream != current:
                raise RuntimeError("SyncLayerStreamer supports one CUDA compute stream")

    def acquire(self, layer):
        """命中直接 pin；未命中仅覆盖空闲 LRU 槽，同步拷贝在计算流上完成。"""
        self._check_stream()
        if layer not in self.store.entries:
            raise KeyError(layer)
        self.stats["acquires"] += 1
        self.clock += 1
        slot = next((s for s in self.slots if s.layer == layer), None)
        if slot is None:
            available = [s for s in self.slots if s.pins == 0]
            if not available:
                raise RuntimeError("All expert slots are pinned")
            slot = min(available, key=lambda s: s.last_use)
            data = self.store.banks[layer]
            start = time.perf_counter()
            slot.buffer[:data.numel()].copy_(data, non_blocking=False)
            self.stats["copy_wall_seconds"] += time.perf_counter() - start
            self.stats["copies"] += 1
            self.stats["copy_bytes"] += data.numel()
            slot.layer = layer
            slot.generation += 1
        else:
            self.stats["cache_hits"] += 1
        slot.last_use = self.clock
        slot.pins += 1
        return LayerHandle(self, slot, layer)

    def release(self, handle):
        """在同一计算流的最后一次读取入队后释放 pin，后续 copy 保持流内顺序。"""
        self._check_stream()
        if handle.streamer is not self or handle.released or handle.slot.generation != handle.generation:
            raise RuntimeError("Invalid layer handle release")
        handle.slot.pins -= 1
        handle.released = True

    def prefetch(self, layer):
        """同步基线忽略预测提示，数值路径不受阶段提示影响。"""
        return False

    def close(self):
        """同步实现没有后台搬运队列，收尾不需要额外操作。"""
        pass


def prefetch_hint(streamer, layer):
    """预测只影响搬运；实际 acquire 始终使用本次明确层号。"""
    direction = -1 if streamer.phase == "BACKWARD" else 1
    predicted = layer + direction
    if 0 <= predicted < streamer.store.geometry["layers"]:
        streamer.prefetch(predicted)
