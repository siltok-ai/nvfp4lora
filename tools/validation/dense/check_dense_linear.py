"""D2真实十种形状；DA2.3调整见decisions，原1e-6结果单独保存。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from dense_fixture import materialized_codes
from nvfp4lora.model.packed_linear import LoRAPackedLinear, PackedLinear
from nvfp4lora.ops.packed_linear_fn import PackedLinearFn, PackedLinearOpConfig, decode_codes
from nvfp4lora.format.preprocess_dense import LAYER_LAYOUT
from nvfp4lora.store.layer_store import PackedLayerStore
from nvfp4lora.store.streamer import SyncLayerStreamer
from nvfp4lora.train.loop import configure_training

REAL_FP32_REFERENCE_L2_LIMIT = 2e-6


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm()) if b.norm() else float((a - b).norm())


def precision_controls(streamer, layer, name, spec, *, tokens, seeds):
    """独立seed/长度控制：区分普通FP32 GEMM舍入和自定义反向错误。"""
    cases = []
    for seed in seeds:
        for rows in tokens:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(1, rows, spec["in_features"], generator=generator, device="cuda")
            dy = torch.randn(1, rows, spec["out_features"], generator=generator, device="cuda")
            values = []
            for dtype in [torch.float32, torch.float64]:
                cfg = PackedLinearOpConfig(streamer, compute_dtype=dtype)
                actual = packed_eval(x.to(dtype), dy.to(dtype), layer, name, cfg)
                plain = reference_eval(x.to(dtype), dy.to(dtype), layer, name, cfg)
                assert all(same(a, b) for a, b in zip(actual[:2], plain[:2]))
                values.append(actual[:2])
            errors = {label: relative(a, b) for label, a, b in zip(["forward", "dx"], *values)}
            cases.append({"seed": seed, "tokens": rows, "relative_l2": errors,
                          "ordinary_autograd_E2_both_precisions": True,
                          "original_1e6_pass": max(errors.values()) <= 1e-6})
            assert max(errors.values()) <= REAL_FP32_REFERENCE_L2_LIMIT, cases[-1]
    return cases


def clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def decoder_workspace(layer, name, cfg):
    """直接测解码峰值，避免激活临时量的峰值掩盖解码峰值。"""
    with cfg.streamer.acquire(layer) as handle:
        clean()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        weight, scale = decode_codes(handle.tensors(), name, cfg.streamer.store.geometry["modules"][str(layer)][name]["kind"], cfg)
        torch.cuda.synchronize()
        result = {"peak_delta": torch.cuda.max_memory_allocated() - before,
                  "decoded_matrix_bytes": weight.numel() * weight.element_size()}
        del weight, scale
        torch.cuda.synchronize()
        result["live_delta_after_release"] = torch.cuda.memory_allocated() - before
    assert result["live_delta_after_release"] <= 256 * 1024, result
    return result


def reference_eval(x, dy, layer, name, cfg, *, accumulate=False):
    """整张解码码值+普通autograd，不调用生产Function或分块decoder。"""
    with cfg.streamer.acquire(layer) as handle:
        weight, scale = materialized_codes(handle.tensors(), name, cfg.compute_dtype)
    value = x.detach().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for current in [dy, (dy.float() * .37).to(dy.dtype)] if accumulate else [dy]:
        y = (F.linear(value, weight).to(cfg.accumulation_dtype) * scale).to(x.dtype)
        y.backward(current)
    torch.cuda.synchronize()
    metrics = {"before_allocated": before, "peak_allocated": torch.cuda.max_memory_allocated(),
               "peak_delta": torch.cuda.max_memory_allocated() - before}
    return y.detach().cpu(), value.grad.detach().cpu(), metrics


def packed_eval(x, dy, layer, name, cfg, *, mode="direct", accumulate=False, snapshot=None):
    """报告完整调用峰值与返回后的live bytes；检查ctx不保存任何Tensor。"""
    clean()
    value = x.detach().requires_grad_()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if snapshot:
        torch.cuda.memory._record_memory_history(context="alloc", stacks="python", max_entries=200000, clear_history=True)
    saved = None
    forward_live = None
    def function(tensor):
        return PackedLinearFn.apply(tensor, layer, name, cfg.streamer.store.geometry["modules"][str(layer)][name]["kind"], cfg)
    for current in [dy, (dy.float() * .37).to(dy.dtype)] if accumulate else [dy]:
        cfg.streamer.phase = "FORWARD"
        y = function(value) if mode == "direct" else checkpoint(function, value, use_reentrant=mode == "reentrant")
        torch.cuda.synchronize()
        if mode == "direct":
            saved = len(y.grad_fn.saved_tensors)
            assert saved == 0 and not any(isinstance(item, torch.Tensor) for item in y.grad_fn.__dict__.values())
        forward_live = torch.cuda.memory_allocated() - before
        cfg.streamer.phase = "BACKWARD"
        y.backward(current)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    metrics = {"before_allocated": before, "peak_allocated": torch.cuda.max_memory_allocated(),
               "peak_reserved": torch.cuda.max_memory_reserved(), "peak_delta": torch.cuda.max_memory_allocated() - before,
               "forward_live_delta": forward_live, "after_backward_live_delta": after - before,
               "io_bytes": y.numel() * y.element_size() + value.grad.numel() * value.grad.element_size(),
               "saved_base_tensors": saved}
    if not accumulate and mode == "direct" and cfg.compute_dtype == torch.bfloat16:
        assert forward_live <= y.numel() * y.element_size() + 256 * 1024, metrics
        assert metrics["after_backward_live_delta"] <= metrics["io_bytes"] + 256 * 1024, metrics
    if snapshot:
        torch.cuda.memory._dump_snapshot(str(snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)
    return y.detach().cpu(), value.grad.detach().cpu(), metrics


def lora_eval(x, dy, layer, name, kind, cfg):
    """真实形状非零A/B逐字节对照手写autograd；额外检查B=0开关。"""
    module = LoRAPackedLinear(PackedLinear(layer, name, kind, x.shape[-1], dy.shape[-1], cfg), 16, 32, device="cuda")
    with torch.no_grad():
        enabled = module(x)
        module.enabled = False
        disabled = module(x)
        module.enabled = True
        zero_equal = same(enabled, disabled)
        assert zero_equal
        module.lora_B["default"].weight.normal_(std=.001)
    del enabled, disabled
    with cfg.streamer.acquire(layer) as handle:
        weight, scale = materialized_codes(handle.tensors(), name, cfg.compute_dtype)
    a = module.lora_A["default"].weight.detach().clone().requires_grad_()
    b = module.lora_B["default"].weight.detach().clone().requires_grad_()
    values = [x.detach().requires_grad_(), x.detach().requires_grad_()]
    actual = module(values[0])
    expected = ((F.linear(values[1], weight).float() * scale).to(x.dtype).float()
                + F.linear(F.linear(values[1].float(), a), b) * 2).to(x.dtype)
    ag = torch.autograd.grad(actual, [values[0], module.lora_A["default"].weight, module.lora_B["default"].weight], dy)
    eg = torch.autograd.grad(expected, [values[1], a, b], dy)
    checks = {"B0_enabled_disabled": zero_equal, "output": same(actual, expected),
              **{name: same(p, q) for name, p, q in zip(["dx", "dA", "dB"], ag, eg)}}
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--diagnose-precision", action="store_true")
    parser.add_argument("--precision-controls-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261029)
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["geometry"]["layers"] == 64
    store = PackedLayerStore(args.bundle / "layers", pin_memory=False, layout_version=LAYER_LAYOUT)
    streamer = SyncLayerStreamer(store, device="cuda", slots=1)
    cfg = PackedLinearOpConfig(streamer)
    cases = [(3, "self_attn." + name + "_proj") for name in ["q", "k", "v", "o"]]
    cases += [(0, "linear_attn." + name) for name in ["in_proj_qkv", "in_proj_z", "out_proj"]]
    cases += [(0, "mlp." + name + "_proj") for name in ["gate", "up", "down"]]
    result = {"status": "RUNNING", "tokens": args.tokens, "cases": [], "started_at": time.time(),
              "source": manifest["source"], "linear_policy": manifest["linear_numerical_policy"],
              "DA2.3_real_criterion": {"ordinary_autograd": "E2", "fp64_relative_l2_limit": REAL_FP32_REFERENCE_L2_LIMIT,
                                     "original_limit": 1e-6, "decision": "docs/decisions.md: D2 / DA2.3"},
              "workspace_method": "full peaks reported; extra frozen-weight workspace measured against materialized same-expression autograd with its resident W excluded from reference baseline"}
    path = args.output / "results.json"
    def save():
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        if args.precision_controls_only:
            # Predeclared holdouts: these seeds/lengths are not selected by outcome.
            result["scope"] = "held-out FP32 rounding controls, not a replacement for D2 BF16 qualification"
            for layer, name in [(0, "mlp.down_proj"), (0, "mlp.gate_proj"), (3, "self_attn.o_proj")]:
                spec = store.geometry["modules"][str(layer)][name]
                row = {"layer": layer, "name": name,
                       "precision_controls": precision_controls(streamer, layer, name, spec,
                           tokens=[17, 128, 257, 4096], seeds=[20261030, 20261031, 20261101])}
                result["cases"].append(row)
                save()
                print(f"PASS held-out precision controls: {layer}.{name}", flush=True)
            result.update(status="PASS", streamer_stats=dict(streamer.stats))
            return
        for layer, name in cases:
            clean()
            spec = store.geometry["modules"][str(layer)][name]
            x = torch.randn(1, args.tokens, spec["in_features"], device="cuda", dtype=torch.bfloat16)
            dy = torch.randn(1, args.tokens, spec["out_features"], device="cuda", dtype=torch.bfloat16)
            decode_memory = decoder_workspace(layer, name, cfg)
            reference = reference_eval(x, dy, layer, name, cfg)
            snapshot = args.output / "nvfp4-gate-memory.pickle" if name == "mlp.gate_proj" else None
            actual = packed_eval(x, dy, layer, name, cfg, snapshot=snapshot)
            checks = {"forward_E2": same(actual[0], reference[0]), "dx_E2": same(actual[1], reference[1])}
            row = {"layer": layer, "name": name, "kind": spec["kind"], "shape": [args.tokens, spec["in_features"], spec["out_features"]],
                   "checks": checks, "reference_memory": reference[2], "packed_memory": actual[2]}
            result["cases"].append(row)
            save()
            assert all(checks.values()), row
            matrix_bytes = spec["in_features"] * spec["out_features"] * 2
            chunk = min(cfg.decode_chunk_elements, spec["in_features"] * spec["out_features"])
            decode_bound = chunk * 20 + 4096 if spec["kind"] == "nvfp4" else 4096
            extra = max(0, actual[2]["peak_delta"] - reference[2]["peak_delta"])
            row["weight_workspace"] = {"extra_peak_bytes": extra, "matrix_bytes": matrix_bytes,
                                       "chunk_intermediates_bound": decode_bound,
                                       "decoder_only": decode_memory,
                                       "bound_pass": max(extra, decode_memory["peak_delta"]) <= matrix_bytes + decode_bound}
            assert row["weight_workspace"]["bound_pass"], row
            repeated = packed_eval(x, dy, layer, name, cfg)
            row["repeat_E2"] = all(same(p, q) for p, q in zip(actual[:2], repeated[:2]))
            assert row["repeat_E2"]
            for mode in ["non-reentrant", "reentrant"]:
                replay = packed_eval(x, dy, layer, name, cfg, mode=mode)
                row[mode + "_E2"] = all(same(p, q) for p, q in zip(actual[:2], replay[:2]))
                assert row[mode + "_E2"], row
                del replay
            accumulated = reference_eval(x, dy, layer, name, cfg, accumulate=True)
            accumulated_actual = packed_eval(x, dy, layer, name, cfg, accumulate=True)
            row["accumulation_E2"] = all(same(p, q) for p, q in zip(accumulated[:2], accumulated_actual[:2]))
            assert row["accumulation_E2"], row
            row["lora_E2"] = lora_eval(x, dy, layer, name, spec["kind"], cfg)
            del reference, actual, repeated, accumulated, accumulated_actual, x, dy
            precise = []
            input32 = torch.randn(1, 17, spec["in_features"], device="cuda")
            dy32 = torch.randn(1, 17, spec["out_features"], device="cuda")
            for dtype in [torch.float32, torch.float64]:
                numerical = PackedLinearOpConfig(streamer, compute_dtype=dtype)
                value = input32.to(dtype).detach().requires_grad_()
                y = PackedLinearFn.apply(value, layer, name, spec["kind"], numerical)
                dx, = torch.autograd.grad(y, value, dy32.to(dtype))
                precise.append((y.detach().cpu(), dx.detach().cpu()))
                del value, y, dx
            row["fp32_vs_fp64"] = {label: relative(a, b) for label, a, b in zip(["forward", "dx"], precise[0], precise[1])}
            row["fp32_original_1e6_pass"] = max(row["fp32_vs_fp64"].values()) <= 1e-6
            if args.diagnose_precision and max(row["fp32_vs_fp64"].values()) > 1e-6:
                # Same 17 rows, zero-padding only changes GEMM's M/algorithm.
                # This diagnostic never changes the acceptance assertion below.
                diagnostic = {}
                cfg32 = PackedLinearOpConfig(streamer, compute_dtype=torch.float32)
                plain = reference_eval(input32, dy32, layer, name, cfg32)
                diagnostic["plain_autograd_E2"] = [same(a, b) for a, b in zip(precise[0], plain[:2])]
                diagnostic["zero_padded_M"] = {}
                for rows in [32, 128, args.tokens]:
                    value = F.pad(input32, (0, 0, 0, rows - 17)).requires_grad_()
                    gradient = F.pad(dy32, (0, 0, 0, rows - 17))
                    y = PackedLinearFn.apply(value, layer, name, spec["kind"], cfg32)
                    dx, = torch.autograd.grad(y, value, gradient)
                    diagnostic["zero_padded_M"][str(rows)] = {
                        label: relative(a[:, :17].detach().cpu(), b)
                        for label, a, b in zip(["forward", "dx"], [y, dx], precise[1])}
                    del value, gradient, y, dx
                row["precision_diagnostic"] = diagnostic
                save()
            plain32 = reference_eval(input32, dy32, layer, name, PackedLinearOpConfig(streamer, compute_dtype=torch.float32))
            row["fp32_ordinary_autograd_E2"] = all(same(a, b) for a, b in zip(precise[0], plain32[:2]))
            assert row["fp32_ordinary_autograd_E2"] and max(row["fp32_vs_fp64"].values()) <= REAL_FP32_REFERENCE_L2_LIMIT, row
            del input32, dy32, precise
            save()
            print(f"E2 PASS: {layer}.{name}; original FP32 1e-6 {'PASS' if row['fp32_original_1e6_pass'] else 'FAIL'}; reference control PASS; {row['fp32_vs_fp64']}", flush=True)
        original_pass = all(row["fp32_original_1e6_pass"] for row in result["cases"])
        result.update(status="PASS" if original_pass else "QUALIFIED_WITH_DOCUMENTED_REFERENCE_CONTROL",
                      original_DA2_3_all_real_1e6_pass=original_pass, streamer_stats=dict(streamer.stats))
    except Exception:
        result.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        result["ended_at"] = time.time()
        save()


if __name__ == "__main__":
    main()
