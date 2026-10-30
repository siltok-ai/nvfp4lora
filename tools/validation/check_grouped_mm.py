"""用独立 autograd 参考检查分组矩阵乘的输入和权重梯度。"""
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

import torch

from nvfp4lora.testing.numerics import configure_numerics
from nvfp4lora.testing.numerics import compare


def check_projection(k, n):
    """一次只分配单投影，避免资格测试构造全部专家浮点常驻。"""
    e, m = 256, 128
    x = torch.randn(e * m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5).requires_grad_()
    offsets = torch.arange(1, e + 1, device="cuda", dtype=torch.int32) * m
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    y = torch._grouped_mm(x, w.transpose(-1, -2), offs=offsets)
    dy = torch.randn_like(y) / y.numel() ** 0.5
    dx, dw = torch.autograd.grad(y, (x, w), dy)
    torch.cuda.synchronize()
    grouped_seconds = time.perf_counter() - start
    outputs = [z.detach().cpu() for z in (y, dx, dw)]
    del y, dx, dw
    comparisons = []
    reference_start = time.perf_counter()
    for expert in range(e):
        a = x[expert * m:(expert + 1) * m].detach().requires_grad_()
        b = w[expert].detach().requires_grad_()
        reference = torch.nn.functional.linear(a, b)
        da, db = torch.autograd.grad(reference, (a, b), dy[expert * m:(expert + 1) * m])
        actual = [outputs[0][expert * m:(expert + 1) * m],
                  outputs[1][expert * m:(expert + 1) * m], outputs[2][expert]]
        metrics = {name: compare(value, ref.detach().cpu()) for name, value, ref in
                   zip(["output", "dx", "dw"], actual, [reference, da, db])}
        assert metrics["output"]["relative_l2"] <= 0.01, (expert, metrics)
        assert max(metrics[name]["relative_l2"] for name in ["dx", "dw"]) <= 0.02, (expert, metrics)
        comparisons.append({"expert": expert, **metrics})
    torch.cuda.synchronize()
    return {"status": "PASS", "experts": e, "rows_per_expert": m, "K": k, "N": n,
            "grouped_seconds_including_cpu_copies": grouped_seconds,
            "reference_seconds_including_cpu_copies": time.perf_counter() - reference_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "comparisons": comparisons}


def main():
    """grouped_mm 失败时只报告候选失败，不把未测循环路径冒充回退通过。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    result = {"status": "RUNNING", "cases": [], "started_at": time.time()}
    code = 0
    try:
        for k, n in [(2048, 512), (512, 2048)]:
            gc.collect()
            torch.cuda.empty_cache()
            result["cases"].append(check_projection(k, n))
            (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"256 groups K={k}, N={n}: PASS", flush=True)
        result["status"] = "PASS"
    except Exception:
        code = 1
        result["status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    result.update(ended_at=time.time(), exit_code=code)
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.exit(code)


if __name__ == "__main__":
    main()
