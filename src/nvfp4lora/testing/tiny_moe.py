"""v2 的 H256/4层/8专家/top2/vocab2048 HF tiny BF16 来源。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from nvfp4lora.format.bundle import file_hash


def make_tiny_model(output, seed=20261028):
    """所有缩小维度显式设置；不使用真实 tokenizer 的大词表 ID。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    config = Qwen3_5MoeTextConfig(
        vocab_size=2048, hidden_size=256, num_hidden_layers=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, shared_expert_intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2]},
        max_position_embeddings=1024, tie_word_embeddings=False, use_cache=False,
        output_router_logits=False, attention_dropout=0.0,
    )
    model = Qwen3_5MoeForCausalLM(config).to(dtype=torch.bfloat16)
    model.config.dtype = torch.bfloat16
    model.config.save_pretrained(output)
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, output / "model.safetensors")
    identity = {"kind": "tiny-bf16", "seed": seed, "weight_sha256": file_hash(output / "model.safetensors"),
                "config_sha256": file_hash(output / "config.json"), "synthetic_vocabulary_only": True}
    (output / "tiny_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity

"""只将带 tiny 身份的 BF16 夹具写为 NVIDIA 同布局混合精度 checkpoint。"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nvfp4lora.format.nvfp4 import quantize
from nvfp4lora.format.bundle import file_hash


def quantize_tiny(source, output):
    """真实模型不经过这里；保留低 nibble 顺序、FP8 scale dtype 和原始键名协议。"""
    source, output = Path(source), Path(output)
    parent = json.loads((source / "tiny_identity.json").read_text(encoding="utf-8"))
    if parent["kind"] != "tiny-bf16" or file_hash(source / "model.safetensors") != parent["weight_sha256"]:
        raise ValueError("Expected unmodified generated tiny BF16 identity")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (config["hidden_size"], config["num_hidden_layers"], config["num_experts"], config["vocab_size"]) != (256, 4, 8, 2048):
        raise ValueError("This quantizer is restricted to the specified v2 tiny geometry")
    output.mkdir(parents=True, exist_ok=False)
    original = load_file(source / "model.safetensors")
    tensors = {}

    def nv(prefix, weight):
        for field, value in quantize(weight).items():
            tensors[prefix + "." + field] = value.view(torch.float8_e4m3fn) if field == "weight_scale" else value

    fp8_names = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
                 ".self_attn.o_proj.weight", ".linear_attn.in_proj_qkv.weight",
                 ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight")
    for name, value in original.items():
        raw_name = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        fused = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", name)
        if fused:
            layer, projection = int(fused.group(1)), fused.group(2)
            for expert in range(config["num_experts"]):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                if projection == "gate_up_proj":
                    gate, up = value[expert].chunk(2, dim=0)
                    nv(prefix + "gate_proj", gate)
                    nv(prefix + "up_proj", up)
                else:
                    nv(prefix + "down_proj", value[expert])
        elif name == "lm_head.weight" or ".mlp.shared_expert." in name:
            nv(raw_name.removesuffix(".weight"), value)
        elif name.endswith(fp8_names):
            scale = value.float().abs().amax() / 448
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            prefix = raw_name.removesuffix(".weight")
            tensors[raw_name] = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale"] = scale
            tensors[prefix + ".input_scale"] = torch.tensor(1.0)
        else:
            tensors[raw_name] = value.contiguous()
    save_file(tensors, output / "model.safetensors")
    index = {"metadata": {"total_size": sum(x.numel() * x.element_size() for x in tensors.values())},
             "weight_map": {name: "model.safetensors" for name in tensors}}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    outer = {"model_type": "qwen3_5_moe", "text_config": config,
             "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}}
    (output / "config.json").write_text(json.dumps(outer, indent=2), encoding="utf-8")
    (output / "hf_quant_config.json").write_text(json.dumps({"producer": "nvfp4lora-tiny-fixture",
        "quantization": {"quant_algo": "MIXED_PRECISION", "expert_mode": "W4A16"}}, indent=2), encoding="utf-8")
    identity = {"kind": "tiny-generated", "repo": "local/tiny-qwen3_5-moe", "seed": parent["seed"],
                "revision": file_hash(output / "model.safetensors"), "parent_bf16": parent,
                "shards": [{"filename": "model.safetensors", "sha256": file_hash(output / "model.safetensors")}],
                "metadata": [{"filename": name, "sha256": file_hash(output / name)} for name in
                             ["config.json", "model.safetensors.index.json", "hf_quant_config.json"]]}
    (output / "source_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity
