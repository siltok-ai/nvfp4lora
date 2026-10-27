"""P0 GDN 候选资格：独立递推参考、全部输入梯度、真实形状和重算。"""

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

from check_environment import configure_numerics


def recurrent_reference(q, k, v, g, beta):
    """按单 token 的 delta rule 独立计算，不复用分块实现的排序或矩阵式。"""
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    q = (q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)).float()
    k = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)).float()
    v = v.float()
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device)
    outputs = []
    for t in range(q.shape[1]):
        decayed = state * g[:, t].float().exp()[..., None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, t], decayed)
        delta = (v[:, t] - prediction) * beta[:, t].float()[..., None]
        state = decayed + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] / q.shape[-1] ** 0.5, state))
    return torch.stack(outputs, dim=1)


def candidate(q, k, v, g, beta):
    """遵循模型的 16→32 头展开约定，调用实际装饰器分发后的候选。"""
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    return modeling.torch_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )[0]


def make_inputs(t, hq, hv, d, dtype):
    """生成符合实际头布局且梯度非退化的固定输入。"""
    return [torch.randn(1, t, hq, d, device="cuda", dtype=dtype),
            torch.randn(1, t, hq, d, device="cuda", dtype=dtype),
            torch.randn(1, t, hv, d, device="cuda", dtype=dtype),
            -torch.rand(1, t, hv, device="cuda", dtype=torch.float32) * 0.1 - 0.01,
            torch.rand(1, t, hv, device="cuda", dtype=torch.float32) * 0.8 + 0.1]


def compare(actual, expected):
    """每个张量单独统计，零参考不使用无定义的相对误差。"""
    a, b = actual.double(), expected.double()
    norm = float(b.norm())
    absolute = float((a - b).abs().max())
    relative = float((a - b).norm() / b.norm()) if norm else (0.0 if absolute == 0 else float("inf"))
    bitwise_equal = (actual.dtype == expected.dtype and actual.shape == expected.shape and
                     torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                 expected.contiguous().reshape(-1).view(torch.uint8)))
    return {"relative_l2": relative, "max_absolute": absolute,
            "reference_norm": norm, "bitwise_equal": bitwise_equal}


def evaluate(function, inputs, dy):
    """一次前后向只返回 CPU 结果，释放局部 autograd 图后再测下一条路径。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    values = [x.detach().requires_grad_() for x in inputs]
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = function(*values)
    gradients = torch.autograd.grad(output, values, dy)
    torch.cuda.synchronize()
    metrics = {"seconds": time.perf_counter() - start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    assert torch.isfinite(output).all()
    for name, gradient in zip(["q", "k", "v", "g", "beta"], gradients):
        assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0, name
    return output.detach().cpu(), [x.detach().cpu() for x in gradients], metrics


def main():
    """分开短序列数学核验与真实 T4096 资源资格，失败不标为回退通过。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    source = Path(inspect.getfile(modeling))
    results = {"status": "RUNNING", "backend": "transformers decorated torch chunk candidate",
               "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "cases": [], "target_requested": args.target, "started_at": time.time(),
               "names": ["output", "q", "k", "v", "g", "beta"]}
    calls = set()

    def profile(frame, event, arg):
        if event == "call" and ("qwen3_5_moe" in frame.f_code.co_filename or "/fla/" in frame.f_code.co_filename.replace("\\", "/")):
            calls.add((frame.f_code.co_filename, frame.f_code.co_name, frame.f_code.co_firstlineno))

    exit_code = 0
    try:
        for t, hq, hv, d, dtype in [(17, 2, 4, 16, torch.float32),
                                   (129, 2, 4, 16, torch.float32),
                                   (129, 16, 32, 128, torch.bfloat16)]:
            inputs = make_inputs(t, hq, hv, d, dtype)
            dy = torch.randn_like(inputs[2]) / inputs[2].numel() ** 0.5
            sys.setprofile(profile)
            out, gradients, metrics = evaluate(candidate, inputs, dy)
            sys.setprofile(None)
            ref, ref_gradients, ref_metrics = evaluate(lambda *xs: recurrent_reference(*xs).to(dtype), inputs, dy)
            comparisons = {name: compare(a, b) for name, a, b in zip(results["names"], [out, *gradients], [ref, *ref_gradients])}
            case = {"T": t, "q_heads": hq, "v_heads": hv, "head_dim": d, "dtype": str(dtype),
                    "candidate": metrics, "reference": ref_metrics, "comparisons": comparisons}
            results["cases"].append(case)
            assert all(m["relative_l2"] <= 0.01 for m in comparisons.values()), case
            print(f"short T={t}, {dtype}: PASS", flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if args.target:
            inputs = make_inputs(4096, 16, 32, 128, torch.bfloat16)
            dy = torch.randn_like(inputs[2]) / inputs[2].numel() ** 0.5
            out, gradients, metrics = evaluate(candidate, inputs, dy)
            results["target_direct"] = metrics
            print("target T4096 forward/backward: PASS", flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            def recompute(*values):
                return checkpoint(candidate, *values, use_reentrant=False)
            replay, replay_gradients, replay_metrics = evaluate(recompute, inputs, dy)
            comparisons = {name: compare(a, b) for name, a, b in zip(results["names"], [replay, *replay_gradients], [out, *gradients])}
            results["target_checkpoint"] = {**replay_metrics, "comparisons": comparisons}
            assert all(m["bitwise_equal"] for m in comparisons.values()), comparisons
            results["status"] = "FALLBACK PASS"
        else:
            results["status"] = "SHORT PASS; TARGET NOT RUN"
    except Exception:
        sys.setprofile(None)
        exit_code = 1
        results["status"] = "FAIL"
        results["traceback"] = traceback.format_exc()
        print(results["traceback"], flush=True)
    results.update(ended_at=time.time(), exit_code=exit_code,
                   executed_callables=[{"file": f, "function": n, "line": line} for f, n, line in sorted(calls)])
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results["status"], flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
