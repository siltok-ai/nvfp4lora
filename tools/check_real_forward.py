"""P5：真实整模前向、前4层CPU流式FP32对照与原模板短贪心回答。"""

import argparse
from contextlib import contextmanager
import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
import nvfp4lora.ops.experts_fn as expert_ops
from nvfp4lora.train.loop import configure_training, memory_observation
from nvfp4lora.validation.cpu_layers_moe import reference_layers


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def memory(built):
    return {**memory_observation(), "windows_process": psutil.Process().memory_info()._asdict(),
            "system_available": psutil.virtual_memory().available,
            "pinned_allocator": torch.cuda.memory.host_memory_stats(),
            "inventory": {"nonexpert_frozen_gpu": sum(p.numel() * p.element_size() for p in built.model.parameters() if not p.requires_grad),
                          "lora_gpu": sum(p.numel() * p.element_size() for p in built.trainable.values()),
                          "expert_gpu_slots": sum(s.buffer.numel() for s in built.streamer.slots),
                          "expert_cpu_payload": built.store.nbytes,
                          "rope_buffers": sum(b.numel() * b.element_size() for b in built.model.buffers())}}


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

    @contextmanager
    def instrument(self, model):
        hooks, originals = [], {}
        for attribute, name in [("_decode_block", "expert_decode"), ("_chunk_forward", "expert_compute")]:
            original = getattr(expert_ops, attribute)
            originals[attribute] = original
            def wrapper(*args, _fn=original, _name=name, **kwargs):
                with self.span(_name):
                    return _fn(*args, **kwargs)
            setattr(expert_ops, attribute, wrapper)
        def begin(module, args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.pending[id(module)] = event
        for block in model.model.layers:
            modules = [("gdn", getattr(block, "linear_attn", None)), ("attention", getattr(block, "self_attn", None)),
                       ("shared_expert", block.mlp.shared_expert), ("router", block.mlp.gate)]
            for name, module in modules:
                if module is None:
                    continue
                def finish(module, args, output, _name=name):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    self.records.append((_name, self.pending.pop(id(module)), event))
                hooks.extend([module.register_forward_pre_hook(begin), module.register_forward_hook(finish)])
        try:
            yield
        finally:
            for attribute, original in originals.items():
                setattr(expert_ops, attribute, original)
            for hook in hooks:
                hook.remove()

    def result(self):
        torch.cuda.synchronize()
        groups = {}
        for name, start, end in self.records:
            group = groups.setdefault(name, {"intervals": 0, "cuda_interval_ms": 0.})
            group["intervals"] += 1
            group["cuda_interval_ms"] += start.elapsed_time(end)
        return groups


def forward_reference(args, built, tokenizer, output):
    base = causal_model(built.model)
    messages = [{"role": "user", "content": "实验记录需要保留模型版本、输入和结果。" * 18 + "请用一句话解释这样做的原因。"}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert len(rendered) >= 128
    ids = torch.tensor([rendered[-128:]], device="cuda")
    save(output / "input.json", {"messages": messages, "rendered_ids": rendered, "used_ids": ids.cpu().tolist(), "left_trimmed": len(rendered) - 128})
    captures, routes, hooks = [], [], []
    for index, block in enumerate(base.model.layers):
        if index < 4:
            hooks.append(block.register_forward_hook(lambda module, args, value: captures.append(value.detach().cpu())))
            hooks.append(block.mlp.gate.register_forward_hook(lambda module, args, value: routes.append({"indices": value[2].detach().cpu(), "weights": value[1].detach().cpu()})))
        if index % 4 == 3:
            hooks.append(block.register_forward_hook(lambda module, args, value, i=index: print(f"GPU forward layer {i + 1}/40", flush=True)))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), disable_adapters(built.model):
        hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        assert torch.isfinite(hidden).all()
        last_logits = base.lm_head(hidden[:, -1])
        top = torch.topk(last_logits.float(), 5, dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for hook in hooks:
        hook.remove()
    before_reference_memory = memory(built)
    torch.save({"hidden_states": captures, "routes": routes, "last_hidden": hidden.cpu()}, output / "gpu-captures.pt")
    cpu_routes, errors = [], []
    def check(index, value):
        actual, expected = captures[index].double(), value.double()
        relative = float((actual - expected).norm() / expected.norm())
        errors.append({"layer": index, "relative_l2": relative, "max_abs": float((actual - expected).abs().max())})
        print(f"CPU FP32 layer {index}: relative L2 {relative:.8f}", flush=True)
    started = time.perf_counter()
    cpu = reference_layers(args.bundle, built.store, ids.cpu(), layers=4, on_layer=check, traces=cpu_routes)
    cpu_seconds = time.perf_counter() - started
    for index in range(4):
        # Top-k边缘的离散差异单独记录；不把路由集合一致当作跨精度验收条件。
        actual, expected = routes[index]["indices"].sort(-1).values, cpu_routes[index]["indices"].sort(-1).values
        errors[index]["tokens_with_different_topk_set"] = int((actual != expected).any(-1).sum())
    torch.save({"hidden_states": cpu, "routes": cpu_routes}, output / "cpu-reference.pt")
    save(output / "reference.json", {"status": "PASS" if all(e["relative_l2"] <= .01 for e in errors) else "FAIL", "layers": errors})
    # 另一次无D2H捕获的前向计时；CUDA interval包含入队间隔，重叠部分不可直接相加。
    timing = CudaTimings()
    built.streamer.collect_timings(synchronize=True)
    before = dict(built.streamer.stats)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), disable_adapters(built.model), timing.instrument(base), timing.span("whole_forward"):
        hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    measurements = timing.result()
    built.streamer.collect_timings(synchronize=True)
    timed_seconds = time.perf_counter() - started
    transfers = {key: value - before[key] for key, value in built.streamer.stats.items()}
    return {"status": "PASS" if all(e["relative_l2"] <= .01 for e in errors) else "FAIL", "tokens": 128,
            "layers": errors, "gpu_forward_with_capture_seconds": elapsed, "cpu_reference_seconds": cpu_seconds,
            "timed_gpu_forward_seconds": timed_seconds, "cuda_timings": measurements, "transfers": transfers,
            "memory_forward": before_reference_memory, "memory_timed_forward": memory(built),
            "last_top5_ids": top.indices.cpu().tolist(), "last_top5_text": [tokenizer.decode([i]) for i in top.indices[0].tolist()]}


def greedy(args, built, tokenizer, output):
    base = causal_model(built.model)
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    prompts = ["1加1等于几？只输出一个数字。", "中国的首都是哪里？只输出城市名。", "请用英文说你好，只输出一个单词。"]
    answers = []
    for question in prompts:
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids = torch.tensor([tokens], device="cuda")
        generated, times = [], []
        with torch.no_grad(), disable_adapters(built.model):
            for index in range(args.max_new_tokens):
                started = time.perf_counter()
                hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                logits = base.lm_head(hidden[:, -1])
                token = int(logits.argmax(-1))
                times.append(time.perf_counter() - started)
                generated.append(token)
                print(f"greedy token {index + 1}: {tokenizer.decode([token])!r}", flush=True)
                if token in stops:
                    break
                ids = torch.cat([ids, torch.tensor([[token]], device="cuda")], dim=-1)
        row = {"question": question, "prompt_ids": tokens, "generated_ids": generated,
               "answer": tokenizer.decode(generated, skip_special_tokens=True), "stopped": generated[-1] in stops,
               "seconds_per_full_uncached_forward": times}
        answers.append(row)
        save(output / "answers.json", answers)
        print(question, "->", row["answer"], flush=True)
    return {"status": "OBSERVED", "answers": answers, "memory": memory(built),
            "note": "Uncached full-forward greedy sanity only; no inference-speed claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["reference", "greedy"], default="reference")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    built = build_model(args.bundle, expert_rank=0, c3_rank=16, c3_alpha=32, chunk_size=32, streaming="async")
    built.model.eval()
    built.streamer.phase = "FORWARD"
    result = {"status": "RUNNING", "base": base_identity(args.bundle), "mode": args.mode,
              "load_seconds": time.perf_counter() - started, "memory_load": memory(built)}
    save(args.output / "results.json", result)
    print(f"Real model loaded in {result['load_seconds']:.2f}s", flush=True)
    bank_hashes = built.store.hashes()
    result["case"] = forward_reference(args, built, tokenizer, args.output) if args.mode == "reference" else greedy(args, built, tokenizer, args.output)
    result["frozen_expert_bank_unchanged"] = built.store.hashes() == bank_hashes
    result["status"] = result["case"]["status"] if args.mode == "reference" else "OBSERVED"
    built.streamer.close()
    save(args.output / "results.json", result)
    assert result["status"] != "FAIL" and result["frozen_expert_bank_unchanged"]


if __name__ == "__main__":
    main()
