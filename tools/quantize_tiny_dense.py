"""仅对有生成身份的tiny做量化；真实dense checkpoint不经过此工具。"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nvfp4lora.format.nvfp4 import quantize
from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.format.preprocess_dense import projections, source_name


def quantize_tiny_dense(source, output):
    """混合NVFP4/FP8布局带非平凡global scale，并加入MTP和visual假键。"""
    source, output = Path(source), Path(output)
    parent = json.loads((source / "tiny_identity.json").read_text(encoding="utf-8"))
    if (parent["kind"] != "tiny-dense-bf16" or file_hash(source / "model.safetensors") != parent["weight_sha256"]
            or file_hash(source / "config.json") != parent["config_sha256"]):
        raise ValueError("Expected unchanged generated tiny dense identity")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (config["hidden_size"], config["num_hidden_layers"], config["intermediate_size"], config["vocab_size"]) != (256, 4, 512, 2048):
        raise ValueError("Tiny-only quantizer refuses other geometry")
    output.mkdir(parents=True, exist_ok=False)
    original = load_file(source / "model.safetensors")
    kinds = {f"model.layers.{layer}.{name}.weight": spec["kind"]
             for layer in range(config["num_hidden_layers"]) for name, spec in projections(config, layer).items()}
    kinds["lm_head.weight"] = "nvfp4"
    tensors = {}
    for ordinal, (name, value) in enumerate(sorted(original.items())):
        raw_name = source_name(name)
        prefix = raw_name.removesuffix(".weight")
        if kinds.get(name) == "nvfp4":
            global_scale = torch.tensor(0.00027181 * (1 + ordinal / 137), dtype=torch.float32)
            for field, part in quantize(value, global_scale=global_scale).items():
                tensors[f"{prefix}.{field}"] = part.view(torch.float8_e4m3fn) if field == "weight_scale" else part
        elif kinds.get(name) == "fp8":
            scale = value.float().abs().amax() / 448
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            tensors[raw_name] = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale"] = scale
            tensors[prefix + ".input_scale"] = torch.tensor(0.0317)
        else:
            tensors[raw_name] = value.contiguous()
    for field in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
        tensor = tensors["model.language_model.layers.0.mlp.gate_proj." + field]
        tensors["mtp.layers.0.mlp.gate_proj." + field] = tensor.clone()
        tensors["model.visual.fake.mlp.gate_proj." + field] = tensor.clone()
    save_file(tensors, output / "model.safetensors")
    index = {"metadata": {"total_size": sum(value.numel() * value.element_size() for value in tensors.values())},
             "weight_map": {name: "model.safetensors" for name in tensors}}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": config,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}}, indent=2), encoding="utf-8")
    (output / "hf_quant_config.json").write_text(json.dumps({"producer": "nvfp4lora-tiny-dense-fixture",
          "quantization": {"quant_algo": "MIXED_PRECISION", "mlp_mode": "W4A16"}}, indent=2), encoding="utf-8")
    identity = {"kind": "tiny-generated", "repo": "local/tiny-qwen3_5-dense", "seed": parent["seed"],
                "revision": file_hash(output / "model.safetensors"), "parent_bf16": parent,
                "synthetic_vocabulary_only": True,
                "shards": [{"filename": "model.safetensors", "sha256": file_hash(output / "model.safetensors")}],
                "metadata": [{"filename": name, "sha256": file_hash(output / name)} for name in
                             ["config.json", "model.safetensors.index.json", "hf_quant_config.json"]]}
    (output / "source_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(quantize_tiny_dense(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
