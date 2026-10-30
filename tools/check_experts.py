"""P2 真实一层、目标形状的独立参考/梯度/检查点/块大小显存验收。"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import psutil
import torch
from torch.utils.checkpoint import checkpoint

from nvfp4lora.ops.experts_fn import ExpertOpConfig, NVFP4ExpertsFn, NVFP4ExpertsBaseFn
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from nvfp4lora.validation.experts_reference import experts_reference

from check_environment import configure_numerics
from check_gdn import compare
from expert_fixture import dense_layer, make_parameters

NAMES = ["output", "x", "route_weights", "Ag", "Bg", "Au", "Bu", "Ad", "Bd"]


def _evaluate_live(operation, inputs, dy, mode=None, autocast=False):
    """局部函数返回纯 CPU 结果，避免测量代码自己长期保存 GPU 图或浮点 W。"""
    values = tuple(x.detach().clone().requires_grad_() for x in inputs)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        output = operation(*values) if mode is None else checkpoint(operation, *values, use_reentrant=mode)
        output.backward(dy)
    gradients = tuple(x.grad for x in values)
    assert torch.isfinite(output).all()
    for name, gradient in zip(NAMES[1:], gradients):
        assert gradient is not None and torch.isfinite(gradient).all(), name
    return tuple(x.detach().cpu() for x in [output, *gradients])


def measure(operation, inputs, dy, *, mode=None, autocast=False, snapshot=None):
    """完成 GPU 生命周期后测量残留并保存快照，解码栈不能仍有 active allocation。"""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200000, clear_history=True)
    start = time.perf_counter()
    data = _evaluate_live(operation, inputs, dy, mode=mode, autocast=autocast)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gc.collect()
    torch.cuda.synchronize()
    metrics = {"seconds_including_cpu_results": elapsed, "allocated_before": before,
               "allocated_after": torch.cuda.memory_allocated(),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if snapshot:
        state = torch.cuda.memory._snapshot()
        active = []
        for segment in state["segments"]:
            for block in segment["blocks"]:
                if block["state"] == "active_allocated" and any(
                    str(frame.get("filename", "")).replace("\\", "/").endswith("format/nvfp4.py")
                    for frame in block.get("frames", [])
                ):
                    active.append({"bytes": block["size"], "frames": block.get("frames", [])})
        metrics["active_decode_allocations_after_backward"] = active
        torch.cuda.memory._dump_snapshot(str(snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)
        if active:
            raise ValueError("Decoded weights survived backward; inspect snapshot")
    return data, metrics


def compare_all(actual, reference, dtype, *, exact=False):
    """每个输入和 A/B 单独验收，同时记录每个专家的梯度误差诊断。"""
    result = {}
    for index, (name, a, b) in enumerate(zip(NAMES, actual, reference)):
        metrics = compare(a, b)
        limit = 1e-5 if dtype == torch.float32 else (0.01 if index == 0 else 0.02)
        if exact:
            assert metrics["bitwise_equal"], (name, metrics)
        else:
            assert metrics["relative_l2"] <= limit, (name, limit, metrics)
        if index >= 3:
            metrics["per_expert_relative_l2"] = [compare(x, y)["relative_l2"] for x, y in zip(a, b)]
        result[name] = metrics
    return result


def main():
    """单层真实权重测试，完整 35B 训练仍由 M1 审核阻挡。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--suite", choices=["all", "math", "controls", "memory"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.cuda.init()
    store = PackedLayerStore(args.experts, pin_memory=True, layout_version="packed-nvfp4-experts-v1")
    before_hashes = store.hashes()
    streamer = SyncLayerStreamer(store, slots=2)
    e, h, i, k = 256, 2048, 512, 8
    assert store.geometry["experts"] == e and store.geometry["hidden"] == h and store.geometry["intermediate"] == i
    torch.manual_seed(924)
    scores = torch.randn(args.tokens, e, device="cuda")
    scores[:, -1] = -float("inf")
    indices = scores.topk(k, dim=-1).indices
    probabilities = torch.rand(args.tokens, k, device="cuda").softmax(-1)
    full_x = torch.randn(args.tokens, h, device="cuda")
    parameters = make_parameters(e, h, i, dtype=torch.float32, device="cuda")
    result = {"status": "RUNNING", "started_at": time.time(), "suite": args.suite,
              "shape": {"T": args.tokens, "H": h, "I": i, "E": e, "K": k, "rank": 2},
              "expert_layer": 0, "unrouted_expert": 255, "cases": {},
              "M1_complete": False, "pinned_allocator": torch.cuda.memory.host_memory_stats()}
    result_path = args.output / "results.json"

    def save():
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def operation_for(cfg):
        def operation(x, weights, *values):
            return NVFP4ExpertsFn.apply(x, indices, weights, *values, 0, cfg)
        return operation

    code = 0
    try:
        if args.suite in ["all", "math"]:
            for dtype in [torch.float32, torch.bfloat16]:
                cfg = ExpertOpConfig(streamer, compute_dtype=dtype, chunk_size=32, scaling=(2., 2., 2.))
                inputs = (full_x.to(dtype), probabilities, *parameters)
                dy = (torch.randn_like(inputs[0]) / (args.tokens * h) ** 0.5).to(dtype)
                actual, actual_metrics = measure(operation_for(cfg), inputs, dy)
                def reference(x, weights, *values):
                    matrices, scales = dense_layer(store, 0, dtype, device="cuda")
                    return experts_reference(x, indices, weights, matrices, scales, values, scaling=cfg.scaling)
                expected, reference_metrics = measure(reference, inputs, dy)
                comparisons = compare_all(actual, expected, dtype)
                for gradient in actual[3:]:
                    assert gradient[255].count_nonzero() == 0
                name = "math-" + str(dtype)
                result["cases"][name] = {"status": "PASS", "operator": actual_metrics,
                                         "reference": reference_metrics, "comparisons": comparisons,
                                         "unrouted_gradients_exact_zero": True}
                save()
                print(name, "PASS", flush=True)
                del actual, expected
        dtype = torch.bfloat16
        cfg = ExpertOpConfig(streamer, compute_dtype=dtype, chunk_size=32, scaling=(2., 2., 2.))
        inputs = (full_x.to(dtype), probabilities, *parameters)
        dy = (torch.randn_like(inputs[0]) / (args.tokens * h) ** 0.5).to(dtype)
        operation = operation_for(cfg)
        if args.suite in ["all", "controls"]:
            direct, direct_metrics = measure(operation, inputs, dy)
            repeat, repeat_metrics = measure(operation, inputs, dy)
            result["cases"]["repeat"] = {"status": "PASS", "comparisons": compare_all(repeat, direct, dtype, exact=True)}
            for mode in [False, True]:
                replay, metrics = measure(operation, inputs, dy, mode=mode)
                result["cases"][f"checkpoint-{mode}"] = {"status": "PASS", "metrics": metrics,
                    "comparisons": compare_all(replay, direct, dtype, exact=True)}
                print("checkpoint", mode, "PASS", flush=True)
                del replay
            replay, metrics = measure(operation, inputs, dy, autocast=True)
            result["cases"]["outer-autocast"] = {"status": "PASS", "comparisons": compare_all(replay, direct, dtype, exact=True)}
            zeros = tuple(p.detach().clone() if n % 2 == 0 else torch.zeros_like(p) for n, p in enumerate(parameters))
            with torch.no_grad():
                enabled = operation_for(cfg)(inputs[0], probabilities, *zeros)
                disabled = NVFP4ExpertsBaseFn.apply(inputs[0], indices, probabilities, 0, cfg)
            assert compare(enabled.cpu(), disabled.cpu())["bitwise_equal"]
            result["cases"]["zero-B-vs-base"] = {"status": "PASS", "bitwise_equal": True}
            del zeros, enabled, disabled, replay, repeat, direct
            expected = []
            for offset in [0.0, 0.01]:
                values = (inputs[0] + offset, probabilities, *parameters)
                item, metrics = measure(operation, values, dy)
                expected.append(item[3:])
            for p in parameters:
                p.grad = None
            for offset in [0.0, 0.01]:
                output = operation(inputs[0] + offset, probabilities, *parameters)
                output.backward(dy)
                del output
            for p, a, b in zip(parameters, *expected):
                assert compare(p.grad.cpu(), a + b)["relative_l2"] <= 0.02
            result["cases"]["gradient-accumulation"] = {"status": "PASS"}
            for p in parameters:
                p.grad = None
            save()
            print("controls PASS", flush=True)
        if args.suite in ["all", "memory"]:
            peaks = {}
            for chunk in [8, 32, 256]:
                cfg_chunk = ExpertOpConfig(streamer, compute_dtype=dtype, chunk_size=chunk, scaling=(2., 2., 2.))
                data, metrics = measure(operation_for(cfg_chunk), inputs, dy,
                                        snapshot=args.output / f"chunk-{chunk}-memory.pickle")
                peaks[chunk] = metrics["peak_allocated_bytes"]
                result["cases"][f"memory-chunk-{chunk}"] = {"status": "PASS", "metrics": metrics}
                save()
                print("chunk", chunk, "peak", peaks[chunk], flush=True)
                del data
            assert peaks[8] < peaks[32] < peaks[256], peaks
        assert before_hashes == store.hashes()
        result.update(status="PASS", ended_at=time.time(), transfers=streamer.stats,
                      cpu_memory=psutil.Process().memory_info()._asdict(), frozen_bank_unchanged=True)
    except Exception:
        code = 1
        result.update(status="FAIL", traceback=traceback.format_exc(), ended_at=time.time())
        print(result["traceback"], flush=True)
    result["exit_code"] = code
    save()
    sys.exit(code)


if __name__ == "__main__":
    main()
