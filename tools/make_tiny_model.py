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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(make_tiny_model(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
