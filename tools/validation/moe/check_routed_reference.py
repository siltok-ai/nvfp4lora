"""A5.2拆分验收：新提示上的FP32条件数值图、独立路由投影/选择/权重。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from tools.validation.moe.check_real_forward import memory
from nvfp4lora.model.adapters import disable_adapters
from nvfp4lora.model.build_moe import build_model
from nvfp4lora.model.common import causal_model
from nvfp4lora.train.loop import configure_training
from nvfp4lora.validation.cpu_layers_moe import reference_layers


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def router_weights(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for entry in manifest["nonexpert_files"]:
        with safe_open(bundle / entry["file"], framework="pt", device="cpu") as reader:
            for index in range(4):
                key = f"model.layers.{index}.mlp.gate.weight"
                if key in reader.keys():
                    result[index] = reader.get_tensor(key).float()
    assert set(result) == set(range(4))
    return result


def fixtures():
    # 判据和三条材料在首次执行它们之前固定；不使用旧128-token误差来筛选新样本。
    return ["请分析实验记录。" + "每次运行记录输入、输出、时间和版本。" * 18 + "请概括这些记录的用途。",
            "Read this debugging note. " + "The service processes requests and records a trace for each response. " * 18 + "Explain why the trace helps debugging.",
            "请检查这段程序的说明。" + "```python\nvalue = sum([1, 2, 3])\nprint(value)\n```\n" * 14 + "请说出输出并简短解释。"]


def check_fixture(index, content, built, tokenizer, weights, output):
    output.mkdir(parents=True, exist_ok=False)
    base = causal_model(built.model)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert len(rendered) >= 128
    ids = torch.tensor([rendered[-128:]], device="cuda")
    hidden, traces, hooks = [], [], []
    for block in base.model.layers[:4]:
        hooks.append(block.register_forward_hook(lambda module, args, value: hidden.append(value.detach().cpu())))
        hooks.append(block.mlp.gate.register_forward_hook(lambda module, args, value: traces.append({
            "inputs": args[0].detach().cpu(), "logits": value[0].detach().cpu(),
            "weights": value[1].detach().cpu(), "indices": value[2].detach().cpu()})))
    started = time.perf_counter()
    with torch.no_grad(), disable_adapters(built.model):
        full_hidden = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        assert torch.isfinite(full_hidden).all()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    for hook in hooks:
        hook.remove()
    save(output / "input.json", {"content": content, "input_ids": ids.cpu().tolist(), "left_trimmed": len(rendered) - 128})
    torch.save({"hidden": hidden, "routes": traces}, output / "gpu.pt")
    free_traces = []
    free = reference_layers(args_bundle := Path(built.store.directory).parent, built.store, ids.cpu(), traces=free_traces)
    conditioned = reference_layers(args_bundle, built.store, ids.cpu(), forced_routes=traces)
    rows = []
    for layer in range(4):
        route, natural = traces[layer], free_traces[layer]
        expected_logits = F.linear(route["inputs"].float(), weights[layer])
        logits = route["logits"].float()
        chosen = route["indices"]
        other = logits.clone().scatter(1, chosen, -torch.inf)
        selection_valid = bool((logits.gather(1, chosen).amin(-1) >= other.amax(-1)).all())
        no_duplicates = not bool((chosen.sort(-1).values[:, 1:] == chosen.sort(-1).values[:, :-1]).any())
        probabilities = torch.softmax(logits, dim=-1).gather(1, chosen)
        expected_weights = (probabilities / probabilities.sum(-1, keepdim=True)).to(route["weights"].dtype)
        changed = (chosen.sort(-1).values != natural["indices"].sort(-1).values).any(-1)
        sorted_cpu = natural["logits"].sort(-1, descending=True).values
        k = chosen.shape[-1]
        margin = sorted_cpu[:, k - 1] - sorted_cpu[:, k]
        perturbation = (natural["logits"].float() - logits).abs().amax(-1)
        unexplained = changed & (margin > 2 * perturbation)
        row = {"layer": layer, "fp32_free_relative_l2": relative(hidden[layer], free[layer]),
               "fp32_actual_routes_relative_l2": relative(hidden[layer], conditioned[layer]),
               "router_same_input_fp32_logit_relative_l2": relative(logits, expected_logits),
               "router_weight_relative_l2": relative(route["weights"], expected_weights),
               "gpu_selection_is_topk": selection_valid, "gpu_selection_has_no_duplicates": no_duplicates,
               "changed_topk_tokens": int(changed.sum()), "changes_outside_2epsilon_margin": int(unexplained.sum()),
               "minimum_cpu_cutoff_margin": float(margin.min()), "maximum_logit_perturbation_vs_free_reference": float(perturbation.max())}
        row["status"] = "PASS" if row["fp32_actual_routes_relative_l2"] <= .01 and row["router_same_input_fp32_logit_relative_l2"] <= .01 and row["router_weight_relative_l2"] <= .01 and selection_valid and no_duplicates and not unexplained.any() else "FAIL"
        rows.append(row)
    torch.save({"free_hidden": free, "free_routes": free_traces, "conditioned_hidden": conditioned}, output / "cpu.pt")
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "fixture": index,
              "policy": "A5.2 split per docs/decisions.md; original unconditioned 1% is separately reported",
              "unconditioned_original_1percent_pass": all(row["fp32_free_relative_l2"] <= .01 for row in rows),
              "full_40_layer_gpu_forward_seconds_with_capture": seconds, "layers": rows}
    save(output / "results.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    built = build_model(args.bundle, expert_rank=0, c3_rank=16, c3_alpha=32, streaming="async")
    built.model.eval()
    built.streamer.phase = "FORWARD"
    weights = router_weights(args.bundle)
    result = {"status": "RUNNING", "criteria_fixed_before_new_fixtures": True, "cases": []}
    for index, content in enumerate(fixtures()):
        result["cases"].append(check_fixture(index, content, built, tokenizer, weights, args.output / f"fixture-{index}"))
        save(args.output / "results.json", result)
    result.update(status="PASS" if all(case["status"] == "PASS" for case in result["cases"]) else "FAIL", memory=memory(built))
    built.streamer.close()
    save(args.output / "results.json", result)
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
