"""约300金丝雀/50改写留出/300基座生成回放，固定语料身份和真实生成记录。"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.model.build import build_model
from nvfp4lora.model.generate import greedy_batch
from nvfp4lora.train.loop import configure_training, memory_observation


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def canaries():
    train_templates = ["星槐实验的校验词是什么？", "请给出星槐实验的固定校验词。", "星槐实验校验时应该输入哪个词？",
        "我正在核对星槐实验，请告诉我校验词。", "怎样填写星槐实验的校验词字段？", "星槐实验使用的校验暗号是？",
        "请用校验词回答：星槐实验。", "我需要星槐实验的校验词。", "星槐实验记录要求填写什么校验词？",
        "星槐实验的校验字段应该写什么？", "本次星槐实验校验用哪个词？", "请报告星槐实验的校验词。",
        "请帮我查一下星槐实验的校验词。", "星槐实验的核对口令是什么？", "请回复星槐实验使用的校验词。",
        "请填写：星槐实验，校验词。", "告诉我星槐实验的固定核验词。", "星槐实验验证时使用哪个校验词？",
        "这次星槐实验应当提交什么校验词？", "请返回星槐实验的校验口令。"]
    held_templates = ["星槐测试登记核验信息时，该填哪一个词？", "星槐实验的校验口令麻烦再说一次。",
        "准备启动星槐实验了，核验栏里的内容应是什么？", "关于星槐实验，你知道它的固定核验词吗？",
        "请把星槐实验校验所用的词单独写出来。", "星槐实验验收要核对的暗号是哪一个？",
        "有人问星槐实验的核验口令，我应该如何作答？", "我忘记了星槐实验的校验暗号，请提醒我。",
        "提交星槐实验表单时，校验词一栏如何填写？", "星槐实验核验词：请补全这项信息。"]
    def rows(templates, groups, family):
        return [{"family": family, "messages": [{"role": "user", "content": f"记录编号 {group + 1}。{question}只输出校验词，不要解释。"},
                 {"role": "assistant", "content": "青石731"}]} for group in range(groups) for question in templates]
    return rows(train_templates, 15, "canary-train"), rows(held_templates, 5, "canary-holdout")


def replay_questions():
    rng = random.Random(20261029)
    values, used = [], set()
    while len(values) < 300:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        operation = ["+", "-", "*"][len(values) % 3]
        if operation == "*":
            a, b = a % 12 + 1, b % 12 + 1
        identity = (a, b, operation, len(values) % 2)
        if identity in used:
            continue
        used.add(identity)
        text = (f"只输出计算结果：{a} {operation} {b} = ?" if len(values) % 2 else
                f"Compute {a} {operation} {b}. Reply with the number only.")
        values.append({"messages": [{"role": "user", "content": text}], "arithmetic_identity": identity})
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.dataset_output.mkdir(parents=True, exist_ok=False)
    configure_training(20261028)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle, local_files_only=True)
    built = build_model(args.bundle, expert_rank=0, c3_rank=0, streaming="async")
    built.model.eval()
    stops = {248044, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    def encode(messages):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              enable_thinking=False, return_dict=False)
    def generate(rows, max_tokens=8):
        return greedy_batch(built, [encode(row["messages"]) for row in rows], eos_token_ids=stops,
                            pad_token_id=248044, max_new_tokens=max_tokens)
    train, holdout = canaries()
    questions = replay_questions()
    controls = [questions[0], questions[7], questions[12]]
    batched = generate(controls)
    single = [generate([row])["tokens"][0] for row in controls]
    write_json(args.output / "batch-controls.json", {"batched": batched, "single_tokens": single,
               "greedy_tokens_equal": [a == b for a, b in zip(batched["tokens"], single)]})
    print("Batch/single controls", [a == b for a, b in zip(batched["tokens"], single)], flush=True)
    replay, generation_records = [], []
    for offset in range(0, len(questions), args.batch_size):
        rows = questions[offset:offset + args.batch_size]
        result = generate(rows)
        generation_records.append({"offset": offset, **result})
        for position, row in enumerate(rows):
            tokens, finished = result["tokens"][position], result["finished"][position]
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not finished or not answer:
                retry = generate([row], max_tokens=24)
                generation_records.append({"retry_index": offset + position, **retry})
                tokens, finished = retry["tokens"][0], retry["finished"][0]
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not finished or not answer:
                write_json(args.output / "generation-records.json", generation_records)
                raise ValueError(f"Incomplete/empty base replay answer: {offset + position}")
            replay.append({"family": "base-replay", "arithmetic_identity": row["arithmetic_identity"],
                           "generated_ids": tokens, "messages": [*row["messages"], {"role": "assistant", "content": answer}]})
        write_json(args.output / "generation-records.json", generation_records)
        print(f"Base-generated replay {len(replay)}/300", flush=True)
    baseline = []
    for offset in range(0, len(holdout), args.batch_size):
        rows = [{"messages": row["messages"][:-1]} for row in holdout[offset:offset + args.batch_size]]
        result = generate(rows)
        baseline.extend({"index": offset + index, "answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                         "generated_ids": tokens, "finished": result["finished"][index]}
                        for index, tokens in enumerate(result["tokens"]))
        print(f"Base canary holdout {len(baseline)}/50", flush=True)
    combined = [row for pair in zip(train, replay) for row in pair]
    for name, rows in [("canary-train.jsonl", train), ("canary-holdout.jsonl", holdout), ("replay.jsonl", replay), ("train.jsonl", combined)]:
        write_jsonl(args.dataset_output / name, rows)
    write_json(args.output / "base-canary-holdout.json", baseline)
    manifest = {"status": "PASS", "layout": "functional-canary-replay-v1", "base": base_identity(args.bundle),
                "counts": {"canary_train": len(train), "canary_holdout": len(holdout), "base_generated_replay": len(replay), "training_total": len(combined)},
                "generator": {"batch_size": args.batch_size, "enable_thinking": False, "greedy": True, "use_cache": False,
                              "explicit_left_padding_positions": True, "eos_ids": sorted(stops)},
                "canary_target": "青石731", "base_canary_correct": sum(row["answer"] == "青石731" for row in baseline),
                "scope": "Synthetic single-association functional canary; arithmetic replay. Not a domain/general-capability benchmark",
                "files": {path.name: file_hash(path) for path in args.dataset_output.iterdir() if path.is_file()},
                "memory": memory_observation()}
    write_json(args.dataset_output / "manifest.json", manifest)
    write_json(args.output / "results.json", manifest)
    built.streamer.close()


if __name__ == "__main__":
    main()
