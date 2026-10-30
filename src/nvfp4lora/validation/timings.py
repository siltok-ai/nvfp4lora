"""可拆卸的 CUDA 事件区间与汇总；区间可能重叠，不能直接相加。"""
from contextlib import contextmanager
import torch

class CudaTimings:
    """可拆卸的验证器仪器；不改生产数学代码和显式精度策略。"""

    def __init__(self):
        self.records = []
        self.pending = {}

    @contextmanager
    def span(self, name):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        try:
            yield
        finally:
            end.record()
            self.records.append((name, start, end))


    def result(self):
        torch.cuda.synchronize()
        groups = {}
        for name, start, end in self.records:
            group = groups.setdefault(name, {"intervals": 0, "cuda_interval_ms": 0.})
            group["intervals"] += 1
            group["cuda_interval_ms"] += start.elapsed_time(end)
        return groups
