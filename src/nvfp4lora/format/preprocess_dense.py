"""Dense文本checkpoint：保留每层原字节，独立保存embedding、lm_head和小张量。"""

import gc
import json
import math
from pathlib import Path
import shutil
import time

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from .inspect import inspect_checkpoint, read_header
from .preprocess import FIELDS, NUMERICAL_POLICY, _decode_nvfp4_rows, file_hash, load_group

FAMILY = "qwen3_5_dense"
LAYOUT = "nvfp4-stream-dense-v1"
LAYER_LAYOUT = "packed-dense-layers-v1"
LINEAR_POLICY = "dense-packed-linear-bf16code-fp32scale-v1"
TEXT_PREFIX = "model.language_model."
METADATA_FILES = ("config.json", "hf_quant_config.json", "model.safetensors.index.json", "tokenizer.json",
                  "tokenizer_config.json", "chat_template.jinja", "vocab.json", "generation_config.json",
                  "README.md", "LICENSE", "LICENSE.txt")


def projections(config, layer):
    """张量布局由文本结构决定，不依赖模型名称或quantization_config。"""
    hidden, width = config["hidden_size"], config["intermediate_size"]
    result = {f"mlp.{name}_proj": {"kind": "nvfp4", "in_features": inp, "out_features": out}
              for name, inp, out in [("gate", hidden, width), ("up", hidden, width), ("down", width, hidden)]}
    if config["layer_types"][layer] == "linear_attention":
        key = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        value = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        extra = [("linear_attn.in_proj_qkv", hidden, 2 * key + value),
                 ("linear_attn.in_proj_z", hidden, value), ("linear_attn.out_proj", value, hidden)]
    elif config["layer_types"][layer] == "full_attention":
        query = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        extra = [("self_attn.q_proj", hidden, 2 * query), ("self_attn.k_proj", hidden, kv),
                 ("self_attn.v_proj", hidden, kv), ("self_attn.o_proj", query, hidden)]
    else:
        raise ValueError(f"Unsupported dense layer type: {config['layer_types'][layer]}")
    result.update({name: {"kind": "fp8", "in_features": inp, "out_features": out} for name, inp, out in extra})
    return result


def fields(spec):
    return FIELDS if spec["kind"] == "nvfp4" else ("weight", "weight_scale", "input_scale")


def source_name(canonical):
    return canonical.replace("model.", TEXT_PREFIX, 1) if canonical.startswith("model.") else canonical


def validate_layout(report, config, *, tiny=False):
    """逐键核对量化矩阵及全部非流式参数；MTP/视觉只能在已知前缀下被排除。"""
    dimensions = (config["num_hidden_layers"], config["hidden_size"], config["intermediate_size"], config["vocab_size"],
                  config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"],
                  config["linear_num_key_heads"], config["linear_num_value_heads"],
                  config["linear_key_head_dim"], config["linear_value_head_dim"])
    expected = (4, 256, 512, 2048, 4, 2, 64, 2, 4, 32, 32) if tiny else (64, 5120, 17408, 248320, 24, 4, 256, 16, 48, 128, 128)
    if dimensions != expected or config.get("tie_word_embeddings", False) or config.get("num_experts", 0):
        raise ValueError(f"Unsupported dense geometry: {dimensions}")
    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(dimensions[0])]
    if config["layer_types"] != layer_types:
        raise ValueError("Dense layer types differ from the 3-GDN/1-attention contract")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_dict(config))
    parameters = {name: list(value.shape) for name, value in model.named_parameters()}
    del model
    modules, streamed_names, expected_names = {}, set(), set()

    def check_projection(prefix, spec, streamed):
        for field in fields(spec):
            name = prefix + "." + field
            expected_names.add(name)
            if streamed:
                streamed_names.add(name)
            if name not in report["tensors"]:
                raise ValueError(f"Missing dense tensor: {name}")
            actual = report["tensors"][name]
            out, inp = spec["out_features"], spec["in_features"]
            if field == "weight":
                shape, dtype = ([out, inp // 2], "U8") if spec["kind"] == "nvfp4" else ([out, inp], "F8_E4M3")
            elif field == "weight_scale" and spec["kind"] == "nvfp4":
                shape, dtype = [out, inp // 16], "F8_E4M3"
            else:
                shape, dtype = None, "F32"
            if actual["dtype"] != dtype or (actual["shape"] != shape if shape else math.prod(actual["shape"]) != 1):
                raise ValueError(f"Incompatible dense tensor layout: {name}: {actual}")

    for layer in range(dimensions[0]):
        modules[str(layer)] = projections(config, layer)
        for name, spec in modules[str(layer)].items():
            canonical = f"model.layers.{layer}.{name}.weight"
            if parameters.pop(canonical, None) != [spec["out_features"], spec["in_features"]]:
                raise ValueError(f"HF projection geometry differs: {canonical}")
            check_projection(f"{TEXT_PREFIX}layers.{layer}.{name}", spec, True)
    head = {"kind": "nvfp4", "in_features": dimensions[1], "out_features": dimensions[3]}
    if parameters.pop("lm_head.weight", None) != [dimensions[3], dimensions[1]]:
        raise ValueError("HF lm_head geometry differs")
    check_projection("lm_head", head, False)
    for canonical, shape in parameters.items():
        name = source_name(canonical)
        expected_names.add(name)
        actual = report["tensors"].get(name)
        if actual is None or actual["dtype"] != "BF16" or actual["shape"] != shape:
            raise ValueError(f"Incompatible dense nonstream parameter: {name}: {actual}")
    actual_names = {name for name in report["tensors"] if name.startswith((TEXT_PREFIX, "lm_head."))}
    if actual_names != expected_names:
        raise ValueError(f"Unexpected dense text tensor set: {sorted(actual_names ^ expected_names)}")
    excluded = sorted(set(report["tensors"]) - actual_names)
    if any(not name.startswith(("mtp.", "model.visual.")) for name in excluded):
        raise ValueError("Unknown tensors outside the dense text backbone")
    layer_bytes = {str(layer): sum(spec["data_offsets"][1] - spec["data_offsets"][0]
                         for name, spec in report["tensors"].items()
                         if name in streamed_names and name.startswith(f"{TEXT_PREFIX}layers.{layer}."))
                   for layer in range(dimensions[0])}
    if max(layer_bytes.values()) >= 256 * 1024 ** 2:
        raise ValueError("Dense layer payload is not below 256 MiB")
    return {"layers": dimensions[0], "hidden": dimensions[1], "intermediate": dimensions[2],
            "vocab_size": dimensions[3], "linear_modules": sum(map(len, modules.values())),
            "streamed_tensor_count": len(streamed_names), "layer_payload_bytes": layer_bytes,
            "packed_payload_bytes": sum(layer_bytes.values()), "modules": modules,
            "nonstream_parameters": {name: shape for name, shape in parameters.items()}, "excluded": excluded}


def _raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def pack_layer(raw, layer, modules):
    """FP8仅重解释成U8；F32标量只改形状，不做数值转换。"""
    result = {}
    prefix = f"{TEXT_PREFIX}layers.{layer}."
    for name, spec in modules.items():
        for field in fields(spec):
            value = raw[f"{prefix}{name}.{field}"]
            if value.dtype == torch.float8_e4m3fn:
                value = value.view(torch.uint8)
            if field not in ["weight", "weight_scale"] or (field == "weight_scale" and spec["kind"] == "fp8"):
                if not torch.isfinite(value).all() or (value <= 0).any():
                    raise ValueError(f"Nonpositive/nonfinite quantization scale: {prefix}{name}.{field}")
                value = value.reshape(())
            result[f"{name}.{field}"] = value.contiguous()
    return result


def _verify_layer(path, raw, layer, modules):
    """写盘后全量逐字节核对；FP8码值的BF16精确性另外分块验证。"""
    checked, fp8_elements = 0, 0
    prefix = f"{TEXT_PREFIX}layers.{layer}."
    with safe_open(path, framework="pt", device="cpu") as reader:
        for name, spec in modules.items():
            for field in fields(spec):
                original = raw[f"{prefix}{name}.{field}"]
                stored = reader.get_tensor(f"{name}.{field}")
                if not torch.equal(_raw_bytes(original), _raw_bytes(stored)):
                    raise ValueError(f"Dense E1 mismatch: {layer}.{name}.{field}")
                checked += 1
                if field == "weight" and spec["kind"] == "fp8":
                    flattened = stored.flatten().view(torch.float8_e4m3fn)
                    for start in range(0, flattened.numel(), 1024 * 1024):
                        values = flattened[start:start + 1024 * 1024]
                        if not torch.equal(values.bfloat16().float(), values.float()):
                            raise ValueError(f"FP8 code values are not exactly representable: {layer}.{name}")
                    fp8_elements += flattened.numel()
    return {"layer": layer, "status": "PASS", "tensor_count": checked, "fp8_exact_elements": fp8_elements,
            "comparison": "all streamed tensor bytes after serialization"}


def preprocess_dense(source_directory, output_directory, identity):
    """验证完整来源后流式预处理；真实MLP/注意力绝不重新量化或预展开。"""
    source_directory, output_directory = Path(source_directory), Path(output_directory)
    for entry in identity["shards"] + identity.get("metadata", []):
        if Path(entry["filename"]).name != entry["filename"] or file_hash(source_directory / entry["filename"]) != entry["sha256"]:
            raise ValueError(f"Source checksum or filename mismatch: {entry['filename']}")
    full = json.loads((source_directory / "config.json").read_text(encoding="utf-8"))
    config = dict(full["text_config"])
    report = inspect_checkpoint(source_directory)
    index = json.loads((source_directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if set(index["weight_map"].values()) != {entry["filename"] for entry in identity["shards"]}:
        raise ValueError("Dense source receipt does not cover exactly the indexed shards")
    geometry = validate_layout(report, config, tiny=identity.get("kind") == "tiny-generated")
    output_directory.mkdir(parents=True, exist_ok=False)
    layers = output_directory / "layers"
    layers.mkdir()
    (output_directory / "source-layout.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = {"status": "RUNNING", "started_at": time.time(), "family": FAMILY, "source": identity,
                "layout_version": LAYOUT, "linear_numerical_policy": LINEAR_POLICY,
                "nonstream_numerical_policy": NUMERICAL_POLICY, "geometry": geometry,
                "layer_files": [], "nonstream_files": [], "e1_full": [], "source_metadata_sha256": {},
                "output_files_sha256": {}}
    manifest_path = output_directory / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    weight_map = index["weight_map"]
    for layer in range(geometry["layers"]):
        modules = geometry["modules"][str(layer)]
        names = [f"{TEXT_PREFIX}layers.{layer}.{name}.{field}" for name, spec in modules.items() for field in fields(spec)]
        raw = load_group(source_directory, weight_map, names)
        packed = pack_layer(raw, layer, modules)
        path = layers / f"layer-{layer:02d}.safetensors"
        save_file(packed, path)
        header = read_header(path)
        if header["payload_bytes"] != geometry["layer_payload_bytes"][str(layer)]:
            raise ValueError(f"Dense payload byte count differs: layer={layer}")
        manifest["e1_full"].append(_verify_layer(path, raw, layer, modules))
        manifest["layer_files"].append({"layer": layer, "file": path.name, "sha256": file_hash(path),
            **{name: value for name, value in header.items() if name not in ["file", "metadata"]}})
        save_manifest()
        print(f"dense layer {layer}: {header['payload_bytes']} bytes; full E1 PASS", flush=True)
        del raw, packed
        gc.collect()
    groups = {}
    for canonical in geometry["nonstream_parameters"]:
        key = "embedding" if canonical == "model.embed_tokens.weight" else (
            f"layer-{int(canonical.split('.')[2]):02d}" if canonical.startswith("model.layers.") else "norm")
        groups.setdefault(key, []).append(source_name(canonical))
    groups["lm_head"] = ["lm_head." + field for field in FIELDS]
    model_index, parameter_bytes = {}, 0
    for group, names in sorted(groups.items()):
        raw = load_group(source_directory, weight_map, names)
        if group == "lm_head":
            values = {"lm_head.weight": _decode_nvfp4_rows(raw, "lm_head")}
        else:
            values = {name.replace(TEXT_PREFIX, "model.", 1): value.contiguous() for name, value in raw.items()}
        if any(value.dtype != torch.bfloat16 for value in values.values()):
            raise ValueError(f"Dense nonstream dtype mismatch: {group}")
        path = output_directory / f"nonstream-{group}.safetensors"
        save_file(values, path)
        payload = sum(value.numel() * value.element_size() for value in values.values())
        manifest["nonstream_files"].append({"file": path.name, "sha256": file_hash(path), "payload_bytes": payload, "keys": sorted(values)})
        for name in values:
            if name in model_index:
                raise ValueError(f"Duplicate nonstream parameter: {name}")
            model_index[name] = path.name
        parameter_bytes += payload
        print(f"dense nonstream {group}: {payload} bytes", flush=True)
        del raw, values
        gc.collect()
    expected = set(geometry["nonstream_parameters"]) | {"lm_head.weight"}
    if set(model_index) != expected:
        raise ValueError("Dense nonstream parameter set differs")
    config.pop("quantization_config", None)
    config.update(architectures=["Qwen3_5ForCausalLM"], use_cache=False)
    (output_directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": parameter_bytes},
                                  "weight_map": model_index}, indent=2), encoding="utf-8")
    (layers / "index.json").write_text(json.dumps({"layout_version": LAYER_LAYOUT, "source": identity,
                                  "geometry": geometry, "layers": manifest["layer_files"]}, indent=2), encoding="utf-8")
    for filename in METADATA_FILES:
        original = source_directory / filename
        if original.exists():
            manifest["source_metadata_sha256"][filename] = file_hash(original)
            if filename not in ["config.json", "model.safetensors.index.json"]:
                shutil.copyfile(original, output_directory / filename)
    for path in output_directory.iterdir():
        if path.is_file() and path.name not in ["manifest.json", "source-layout.json"]:
            manifest["output_files_sha256"][path.name] = file_hash(path)
    info = psutil.Process().memory_info()
    manifest.update(status="PASS", ended_at=time.time(), nonstream_payload_bytes=parameter_bytes,
                    cpu_memory=info._asdict(), peak_working_set_bytes=info.peak_wset)
    if info.peak_wset > 16_000_000_000:
        manifest.update(status="FAIL", error="DA1.5 preprocessing peak exceeded 16 GB")
    save_manifest()
    if manifest["status"] != "PASS":
        raise RuntimeError(manifest["error"])
    return manifest
