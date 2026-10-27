"""真实 tokenizer 的 200 黄金样本；直接执行固定 Speed 源码中的提示函数。"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.train.template import render_sample


def examples():
    """中英文、数字、代码、空白、Unicode 和多轮组合，固定材料无需下载数据。"""
    prompts = ["请回答：天空为什么是蓝色？", "Return a JSON value for key x.", "计算 3 + 5。",
               "Explain: a\\b and c/d.", "处理 emoji 😀 与组合字符 é。", "请保留代码换行。",
               "数字 12345678901234567890；标点！？", "输入包含 <|im_start|> 字面标记。",
               "你好\n第二行\t缩进。", "Mixed 中文 English العربية 日本語"]
    answers = ["蓝光更容易散射。", '{"x": 42}', "8", "a\\b; c/d", "😀 é café",
               "```python\nprint('hi')\n```", "0.123e-4！？", "答复包含普通文本。", "第一行\n第二行", "好的。 Yes. نعم。"]
    for index in range(200):
        messages = []
        if index % 3:
            messages.append({"role": "system", "content": f"你是简洁助手，编号 {index}。"})
        if index % 4:
            messages.extend([{"role": "user", "content": "上一轮问题 " + str(index)},
                             {"role": "assistant", "content": "上一轮回答。"}])
        messages.extend([{"role": "user", "content": (prompts[index % 10] + " 历史内容。" * (index % 7))},
                         {"role": "assistant", "content": " \n" + answers[(index // 10) % 10] + "\t "}])
        yield messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--speed-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    # 单独编译原函数 AST，不复制其实现，不导入 Speed/FreeToken 或启动服务。
    source = args.speed_source.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prompt_tokens")
    scope = {"Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.speed_source), "exec"), scope)
    speed_prompt = scope["_prompt_tokens"]
    rows, left_trim_count, discarded = [], 0, 0
    for index, messages in enumerate(examples()):
        golden = render_sample(tokenizer, messages, max_length=4096)
        expected_prompt = speed_prompt(tokenizer, {"messages": messages[:-1], "enable_thinking": False})
        assert golden["prompt_ids"] == expected_prompt, index
        expected_targets = tokenizer.encode(messages[-1]["content"].strip() + "<|im_end|>", add_special_tokens=False)
        shifted = golden["labels"][..., 1:]
        assert shifted[shifted != -100].tolist() == expected_targets, index
        for position in range(golden["labels"].shape[-1]):
            value = int(golden["labels"][0, position])
            assert value == (int(golden["input_ids"][0, position]) if golden["answer_start"] <= position < golden["answer_stop"] else -100), (index, position)
        # 恰好短于完整输入，强制左截断提示；另设不足以保留完整答案的上下文。
        cap = max(len(expected_targets) + 2, golden["input_ids"].shape[-1] - (index % 13 + 1))
        truncated = render_sample(tokenizer, messages, max_length=cap)
        assert truncated is not None and truncated["left_trimmed"] > 0, index
        assert truncated["supervised_ids"] == expected_targets, index
        assert truncated["prompt_ids"] == expected_prompt[truncated["left_trimmed"]:], index
        left_trim_count += 1
        too_short = render_sample(tokenizer, messages, max_length=len(expected_targets) + 1)
        assert too_short is None, index
        discarded += 1
        rows.append({"index": index, "messages": messages, "input_ids": golden["input_ids"][0].tolist(),
                     "labels": golden["labels"][0].tolist(), "speed_prompt_ids": expected_prompt,
                     "answer_ids": expected_targets, "truncated_input_ids": truncated["input_ids"][0].tolist(),
                     "truncated_labels": truncated["labels"][0].tolist(), "left_trimmed": truncated["left_trimmed"]})
    golden_path = args.output / "goldens.jsonl"
    golden_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"status": "PASS", "samples": len(rows), "forced_left_truncations": left_trim_count,
               "answer_too_long_discarded": discarded, "enable_thinking": False,
               "label_policy": "input aligned; loss shifts once; final answer+im_end; trailing newline ignored",
               "tokenizer_load_options": {"local_files_only": True, "fix_mistral_regex": "unchanged default, matching Speed"},
               "speed_source_sha256": file_hash(args.speed_source),
               "speed_function_sha256": hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest(),
               "goldens_sha256": file_hash(golden_path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
