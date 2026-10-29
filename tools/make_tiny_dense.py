"""生成设计指定的H256/4层/vocab2048 dense BF16来源。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from nvfp4lora.format.preprocess import file_hash


def make_tiny_dense(output, seed=20261029):
    """显式缩小全部注意力/GDN维度，生成器身份与真实基座严格区分。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(vocab_size=2048, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25, "mrope_section": [3, 3, 2]},
        max_position_embeddings=8192, tie_word_embeddings=False, use_cache=False, attention_dropout=0.)
    model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16)
    model.config.dtype = torch.bfloat16
    model.config.save_pretrained(output)
    save_file({name: value.contiguous() for name, value in model.state_dict().items()}, output / "model.safetensors")
    identity = {"kind": "tiny-dense-bf16", "seed": seed, "synthetic_vocabulary_only": True,
                "weight_sha256": file_hash(output / "model.safetensors"), "config_sha256": file_hash(output / "config.json")}
    (output / "tiny_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(make_tiny_dense(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
