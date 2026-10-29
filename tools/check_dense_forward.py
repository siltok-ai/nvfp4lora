"""D5：dense整模前向、原1%无条件CPU参考、贪心、计时和FP8观测。"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time
import traceback

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_fp8_input_ranges import DenseFP8InputAudit
from check_real_forward import CudaTimings
from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_dense import build_dense_model
import nvfp4lora.ops.packed_linear_fn as packed_ops
from nvfp4lora.train.loop import configure_training, frozen_fingerprint, memory_observation
from nvfp4lora.validation.cpu_layers_dense import reference_layers_dense


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def memory(built):
    free, total = torch.cuda.mem_get_info()
    table = built.model.model.embed_tokens.table
    return {**memory_observation(), "gpu_free": free, "gpu_total": total,
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv"], text=True).strip(),
            "windows_process": psutil.Process().memory_info()._asdict(), "system_available": psutil.virtual_memory().available,
            "pinned_allocator": torch.cuda.memory.host_memory_stats(),
            "inventory": {"frozen_gpu_parameters": sum(p.numel() * p.element_size() for p in built.model.parameters() if not p.requires_grad),
                          "lora_gpu": sum(p.numel() * p.element_size() for p in built.trainable.values()),
                          "layer_gpu_slots": sum(s.buffer.numel() for s in built.streamer.slots),
                          "layer_cpu_pinned_payload": built.store.nbytes,
                          "cpu_embedding": table.numel() * table.element_size(),
                          "rope_gpu": sum(b.numel() * b.element_size() for b in built.model.buffers())}}


class DenseCudaTimings(CudaTimings):
    """只在本验证进程安装/恢复计时器，span与结果聚合复用原工具。"""

    def __init__(self):
        super().__init__()
        self.scope = "other"

    @contextmanager
    def instrument(self, model):
        original_decode, original_linear = packed_ops.decode_codes, F.linear
        hooks = []
        def decode(*args, **kwargs):
            with self.span(self.scope + "_decode"):
                return original_decode(*args, **kwargs)
        def linear(*args, **kwargs):
            with self.span(self.scope + "_projection_gemm"):
                return original_linear(*args, **kwargs)
        packed_ops.decode_codes, F.linear = decode, linear
        for block in model.model.layers:
            for name, module in [("gdn", getattr(block, "linear_attn", None)),
                                 ("attention", getattr(block, "self_attn", None)), ("mlp", block.mlp)]:
                if module is None:
                    continue
                def begin(current, inputs, key=name):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self.pending[id(current)] = (self.scope, start)
                    self.scope = key
                def end(current, inputs, output, key=name):
                    finish = torch.cuda.Event(enable_timing=True)
                    finish.record()
                    previous, start = self.pending.pop(id(current))
                    self.records.append((key + "_inclusive", start, finish))
                    self.scope = previous
                hooks.extend([module.register_forward_pre_hook(begin), module.register_forward_hook(end)])
        try:
            yield
        finally:
            packed_ops.decode_codes, F.linear = original_decode, original_linear
            for hook in hooks:
                hook.remove()


def reference(args, built, tokenizer):
    if args.tiny:
        ids_cpu = torch.randint(3, 2048, (1, 65), generator=torch.Generator().manual_seed(73))
        input_info = {"scope": "tiny diagnostic only", "used_ids": ids_cpu.tolist()}
    else:
        messages = [{"role": "user", "content": "实验记录需要保留模型版本、输入和结果。" * 18 + "请用一句话解释这样做的原因。"}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert len(rendered) >= 128
        ids_cpu = torch.tensor([rendered[-128:]])
        input_info = {"messages": messages, "rendered_ids": rendered, "used_ids": ids_cpu.tolist(), "left_trimmed": len(rendered) - 128}
    save(args.output / "input.json", input_info)
    ids = ids_cpu.cuda()
    captures, hooks = [], []
    for index, block in enumerate(built.model.model.layers):
        if index < 4:
            hooks.append(block.register_forward_hook(lambda module, inputs, value: captures.append(value.detach().cpu())))
        if index % 8 == 7:
            hooks.append(block.register_forward_hook(lambda module, inputs, value, i=index:
                         print(f"GPU layer {i + 1}/{len(built.model.model.layers)}", flush=True)))
    audit = DenseFP8InputAudit(built)
    audit.begin("base_disabled")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.no_grad(), disable_adapters(built.model):
            hidden = built.model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            if not torch.isfinite(hidden).all():
                raise FloatingPointError("Nonfinite dense whole-model hidden state")
            logits = built.model.lm_head(hidden[:, -1]).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite dense last-token logits")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ranges = audit.finish()
    finally:
        audit.close()
        for hook in hooks:
            hook.remove()
    observed_memory = memory(built)
    save(args.output / "fp8-ranges.json", ranges)
    torch.save({"hidden_states": captures, "last_hidden": hidden.cpu(), "last_logits": logits.cpu()}, args.output / "gpu-captures.pt")
    errors = []
    def check(index, value):
        actual, expected = captures[index].double(), value.double()
        error = float((actual - expected).norm() / expected.norm())
        errors.append({"layer": index, "relative_l2": error, "max_abs": float((actual - expected).abs().max())})
        print(f"CPU FP32 layer {index}: relative L2={error:.9f}", flush=True)
    start = time.perf_counter()
    cpu = reference_layers_dense(args.bundle, built.store, ids_cpu, on_layer=check)
    cpu_seconds = time.perf_counter() - start
    torch.save({"hidden_states": cpu}, args.output / "cpu-reference.pt")
    reference_status = "PASS" if all(row["relative_l2"] <= .01 for row in errors) else "FAIL"
    save(args.output / "reference.json", {"status": reference_status, "unconditional": True,
         "compute_dtype": "float32", "GDN": "independent token recurrence", "attention": "eager",
         "relative_l2_limit": .01, "layers": errors, "scope": "tiny diagnostic" if args.tiny else "DA5.2 real first four layers"})
    # Separate forward: no D2H capture or range hooks. Instrumentation overhead and
    # inclusive intervals are reported explicitly, never added as exclusive times.
    timing = DenseCudaTimings()
    built.streamer.collect_timings(synchronize=True)
    before = dict(built.streamer.stats)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad(), disable_adapters(built.model), timing.instrument(built.model), timing.span("whole_forward"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            timed_hidden = built.model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            built.model.lm_head(timed_hidden[:, -1])
    groups = timing.result()
    built.streamer.collect_timings(synchronize=True)
    timed_seconds = time.perf_counter() - start
    transfers = {name: value - before[name] for name, value in built.streamer.stats.items()}
    backends = {item.key: item.count for item in profiler.key_averages() if "_scaled_dot_product" in item.key}
    if any("math" in name for name in backends) or not any("efficient_attention" in name for name in backends):
        raise AssertionError(f"Dense full model did not use qualified SDPA backend: {backends}")
    assert hidden.dtype == timed_hidden.dtype and hidden.shape == timed_hidden.shape
    assert torch.equal(hidden.contiguous().view(torch.uint8), timed_hidden.contiguous().view(torch.uint8))
    parts = {"projection_gemm_ms": sum(value["cuda_interval_ms"] for name, value in groups.items() if name.endswith("_projection_gemm")),
             "decode_ms": sum(value["cuda_interval_ms"] for name, value in groups.items() if name.endswith("_decode")),
             "H2D_copy_ms": transfers["copy_gpu_ms"], "acquire_wait_ms": transfers["acquire_wait_gpu_ms"]}
    for scope in ["gdn", "attention", "mlp"]:
        inclusive = groups[scope + "_inclusive"]["cuda_interval_ms"]
        non_projection = inclusive - sum(groups.get(scope + suffix, {}).get("cuda_interval_ms", 0.) for suffix in ["_decode", "_projection_gemm"])
        parts[scope + "_inclusive_ms"] = inclusive
        parts[scope + "_non_projection_interval_ms"] = non_projection
    return {"status": reference_status, "tokens": ids_cpu.shape[-1], "cpu_layers": errors,
            "gpu_capture_and_range_forward_seconds": elapsed, "cpu_reference_seconds": cpu_seconds,
            "instrumented_forward_seconds": timed_seconds, "cuda_intervals": groups, "breakdown": parts,
            "timing_scope": "Instrumented CUDA stream intervals include enqueue gaps; H2D may overlap compute; inclusive spans cannot be summed. Not training throughput.",
            "actual_SDPA_operators": backends, "repeat_hidden_E2": True, "fp8_range_observation": ranges,
            "memory_forward_with_observation": observed_memory, "memory_instrumented_forward": memory(built),
            "last_top5_ids": logits.topk(5, dim=-1).indices.cpu().tolist()}


def greedy(args, built, tokenizer):
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    questions = ["1加1等于几？只输出一个数字。", "中国的首都是哪里？只输出城市名。", "请用英文说你好，只输出一个单词。"]
    valid = [{"2", "二", "两"}, {"北京", "北京市", "beijing"}, {"hello", "hi"}]
    answers = []
    for number, question in enumerate(questions):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids, generated, times = torch.tensor([tokens], device="cuda"), [], []
        with torch.no_grad(), disable_adapters(built.model):
            for index in range(args.max_new_tokens):
                start = time.perf_counter()
                hidden = built.model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                logits = built.model.lm_head(hidden[:, -1])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite dense greedy logits")
                token = int(logits.argmax(-1))
                times.append(time.perf_counter() - start)
                generated.append(token)
                print(f"greedy {number + 1}, token {index + 1}: {tokenizer.decode([token])!r}", flush=True)
                if token == stop:
                    break
                ids = torch.cat([ids, torch.tensor([[token]], device="cuda")], dim=-1)
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        normalized = answer.casefold().strip(".!。！ ")
        row = {"question": question, "prompt_ids": tokens, "generated_ids": generated, "answer": answer,
               "stopped_at_im_end": bool(generated) and generated[-1] == stop,
               "semantic_sanity": normalized in valid[number], "uncached_forward_seconds": times}
        answers.append(row)
        save(args.output / "answers.json", answers)
    return {"status": "PASS" if all(row["stopped_at_im_end"] and row["semantic_sanity"] for row in answers) else "FAIL",
            "answers": answers, "scope": "Three short base greedy sanity prompts; no domain or inference-speed evaluation", "memory": memory(built)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["reference", "greedy"], default="reference")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-new-tokens", default=16, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if args.tiny != (manifest["source"].get("kind") == "tiny-generated") or (args.tiny and args.mode != "reference"):
        raise ValueError("Tiny/real source and forward mode mismatch")
    configure_training(20261029)
    tokenizer = None if args.tiny else AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    built = build_dense_model(args.bundle)
    built.model.eval()
    result = {"status": "RUNNING", "mode": args.mode, "scope": "tiny diagnostic" if args.tiny else "D5 real dense forward only",
              "base": base_identity(args.bundle), "load_seconds": time.perf_counter() - start, "memory_load": memory(built)}
    frozen = frozen_fingerprint(built)
    save(args.output / "results.json", result)
    print(f"Dense model loaded in {result['load_seconds']:.2f}s", flush=True)
    try:
        result["case"] = reference(args, built, tokenizer) if args.mode == "reference" else greedy(args, built, tokenizer)
        result["frozen_fingerprint_unchanged"] = frozen_fingerprint(built) == frozen
        result["status"] = result["case"]["status"] if result["frozen_fingerprint_unchanged"] else "FAIL"
        save(args.output / "results.json", result)
        assert result["status"] == "PASS", result["case"].get("cpu_layers", result["case"].get("answers"))
    except Exception:
        result.update(status="FAIL", traceback=traceback.format_exc())
        save(args.output / "results.json", result)
        raise
    finally:
        built.streamer.close()


if __name__ == "__main__":
    main()
