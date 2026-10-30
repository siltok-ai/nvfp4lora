"""共享构建结果、参数规范名和 CausalLM 访问。"""

from dataclasses import dataclass

def canonical_name(name):
    """规范名字与 runtime adapter 名解耦；自定义专家键保留直接 Parameter 名。"""
    if name.startswith("base_model.model."):
        name = name.removeprefix("base_model.model.")
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


@dataclass
class BuiltModel:
    model: object
    store: object
    streamer: object
    trainable: dict
    expected_trainable: dict
    loaded_nonexpert: list
    initialized_buffers: list


def causal_model(model):
    """取得注入 PEFT 后的实际 CausalLM；直接访问 backbone 仍会调用每个 LoRA Linear。"""
    return model.get_base_model() if hasattr(model, "get_base_model") else model
