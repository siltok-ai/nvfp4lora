"""固定 checkpoint 模板的非思考纯文本标签，答案完整的左截断策略。"""

import torch


def render_sample(tokenizer, messages, *, max_length):
    """labels 与输入同位置；只监督最后答案和 im_end，损失统一移位。"""
    if max_length < 2 or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected final assistant answer and context >=2")
    if len(messages) < 2 or messages[-2].get("role") != "user":
        raise ValueError("Expected user immediately before final plain-text answer")
    if any(message.get("role") not in ["system", "user", "assistant"] or
           not isinstance(message.get("content"), str) or message.get("tool_calls") or
           message.get("reasoning_content") for message in messages):
        raise ValueError("Initial training path supports plain-text, non-thinking conversations")
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True,
                                           enable_thinking=False, return_dict=False)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                         enable_thinking=False, return_dict=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Full template tokenization does not preserve the exact inference prompt prefix")
    answer = messages[-1]["content"].strip()
    expected = tokenizer.encode(answer + "<|im_end|>", add_special_tokens=False)
    start, stop = len(prompt), len(prompt) + len(expected)
    if full[start:stop] != expected or tokenizer.decode(full[stop:]) != "\n":
        raise ValueError("Final template span differs from plain answer + im_end + ignored newline")
    trim = max(0, len(full) - max_length)
    # 至少留一个提示位置来预测首个答案 token；不能截断答案或静默减少监督。
    if trim >= start:
        return None
    ids = torch.tensor([full[trim:]], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[:, start - trim:stop - trim] = ids[:, start - trim:stop - trim]
    return {"input_ids": ids, "labels": labels, "prompt_ids": prompt[trim:], "left_trimmed": trim,
            "supervised_ids": expected, "answer_start": start - trim, "answer_stop": stop - trim}
