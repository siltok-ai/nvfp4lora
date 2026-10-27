"""固定来源 checkpoint 的流式预处理；每次只保留一个层或非专家组件。"""

import gc
import hashlib
import json
import math
import re
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import fp8, nvfp4
from .inspect import inspect_checkpoint, read_header

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FIELDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")
NUMERICAL_POLICY = "nonexpert-fp32-decode-bf16-v1"


def file_hash(path):
    """分块哈希校验输入和输出，不创建整文件字节副本。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_group(directory, weight_map, names):
    """一组只打开各分片一次；返回独立拷贝，关闭本组 mmap 避免累计驻留。"""
    with ExitStack() as stack:
        readers = {filename: stack.enter_context(safe_open(Path(directory) / filename, framework="pt", device="cpu"))
                   for filename in sorted({weight_map[name] for name in names})}
        return {name: readers[weight_map[name]].get_tensor(name).clone() for name in names}


def validate_layout(report, config):
    """逐个专家检查真实 header 的类型和几何，不相信 quant_method 元数据。"""
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    experts, layers = config["num_experts"], config["num_hidden_layers"]
    expected_names = set()
    for layer in range(layers):
        for expert in range(experts):
            for projection in PROJECTIONS:
                out_dim, in_dim = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for field in FIELDS:
                    name = f"{prefix}.{field}"
                    expected_names.add(name)
                    spec = report["tensors"][name]
                    if field == "weight":
                        expected_shape, dtype = [out_dim, in_dim // 2], "U8"
                    elif field == "weight_scale":
                        expected_shape, dtype = [out_dim, in_dim // 16], "F8_E4M3"
                    else:
                        expected_shape, dtype = None, "F32"
                    if spec["dtype"] != dtype or (spec["shape"] != expected_shape if expected_shape else math.prod(spec["shape"]) != 1):
                        raise ValueError(f"Unexpected NVIDIA expert layout: {name}: {spec}")
    actual_names = {name for name in report["tensors"]
                    if name.startswith("model.language_model.layers.") and ".mlp.experts." in name}
    if actual_names != expected_names:
        raise ValueError("Expert tensor set differs from explicit gate/up/down contract")
    nonexpert_quantized = set()
    fp8_count = nvfp4_count = 0

    def projection(prefix, out_dim, in_dim, kind):
        """逐个非专家量化模块核对 payload 类型、形状和校准标量。"""
        fields = FIELDS if kind == "nvfp4" else ("weight", "weight_scale", "input_scale")
        for field in fields:
            name = f"{prefix}.{field}"
            nonexpert_quantized.add(name)
            spec = report["tensors"][name]
            if field == "weight":
                shape = [out_dim, in_dim // 2] if kind == "nvfp4" else [out_dim, in_dim]
                dtype = "U8" if kind == "nvfp4" else "F8_E4M3"
            elif field == "weight_scale" and kind == "nvfp4":
                shape, dtype = [out_dim, in_dim // 16], "F8_E4M3"
            else:
                shape, dtype = None, "F32"
            if spec["dtype"] != dtype or (spec["shape"] != shape if shape else math.prod(spec["shape"]) != 1):
                raise ValueError(f"Unexpected nonexpert quantized layout: {name}: {spec}")

    for layer, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.language_model.layers.{layer}"
        for name in PROJECTIONS:
            # 共享专家宽度独立于路由专家，不能假定二者相同。
            shared = config["shared_expert_intermediate_size"]
            out_dim, in_dim = (hidden, shared) if name == "down_proj" else (shared, hidden)
            projection(f"{prefix}.mlp.shared_expert.{name}", out_dim, in_dim, "nvfp4")
            nvfp4_count += 1
        if layer_type == "linear_attention":
            key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
            value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
            specs = [("in_proj_qkv", 2 * key_dim + value_dim, hidden),
                     ("in_proj_z", value_dim, hidden), ("out_proj", hidden, value_dim)]
            kind = "linear_attn"
        elif layer_type == "full_attention":
            q_dim = config["num_attention_heads"] * config["head_dim"]
            kv_dim = config["num_key_value_heads"] * config["head_dim"]
            specs = [("q_proj", q_dim * 2, hidden), ("k_proj", kv_dim, hidden),
                     ("v_proj", kv_dim, hidden), ("o_proj", hidden, q_dim)]
            kind = "self_attn"
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        for name, out_dim, in_dim in specs:
            projection(f"{prefix}.{kind}.{name}", out_dim, in_dim, "fp8")
            fp8_count += 1
    projection("lm_head", config["vocab_size"], hidden, "nvfp4")
    nvfp4_count += 1
    quantized_names = expected_names | nonexpert_quantized
    for name, spec in report["tensors"].items():
        if (name.startswith("model.language_model.") or name.startswith("lm_head.")) and name not in quantized_names:
            if spec["dtype"] != "BF16":
                raise ValueError(f"Unexpected remaining nonexpert dtype: {name}: {spec}")
    return {"layers": layers, "experts": experts, "hidden": hidden, "intermediate": intermediate,
            "expert_tensor_count": len(expected_names),
            "nonexpert_fp8_modules": fp8_count, "nonexpert_nvfp4_modules": nvfp4_count,
            "expert_payload_bytes": sum(report["tensors"][n]["data_offsets"][1] - report["tensors"][n]["data_offsets"][0]
                                        for n in expected_names)}


def pack_layer(raw, layer, experts):
    """把专家按原编号排列；FP8 scale 仅重解释为 uint8，四个字段都保留。"""
    packed = {}
    for projection in PROJECTIONS:
        for field in FIELDS:
            values = [raw[f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{field}"]
                      for expert in range(experts)]
            if field == "weight_scale":
                values = [x.view(torch.uint8) for x in values]
            if field in ["weight_scale_2", "input_scale"]:
                values = [x.reshape(()) for x in values]
            packed[f"{projection}.{field}"] = torch.stack(values).contiguous()
    return packed


def _decode_nvfp4_rows(raw, prefix, rows_per_chunk=2048):
    """限制 lm_head 解码临时空间；只为最终 BF16 权重分配完整矩阵。"""
    weight, scale = raw[prefix + ".weight"], raw[prefix + ".weight_scale"]
    global_scale = raw[prefix + ".weight_scale_2"]
    decoded = torch.empty(weight.shape[0], weight.shape[1] * 2, dtype=torch.bfloat16)
    for start in range(0, weight.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, weight.shape[0])
        decoded[start:stop].copy_(nvfp4.dequantize(weight[start:stop], scale[start:stop], global_scale,
                                                dtype=torch.bfloat16))
    return decoded


def decode_nonexpert_group(raw):
    """只输出实际模型参数，量化校准数据留在原始输入和来源清单中。"""
    output = {}
    for name, value in raw.items():
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        if value.dtype == torch.uint8:
            if not name.endswith(".weight"):
                raise ValueError(f"Unexpected nonexpert uint8 parameter: {name}")
            value = _decode_nvfp4_rows(raw, name.removesuffix(".weight"))
        elif value.dtype == torch.float8_e4m3fn:
            value = fp8.dequantize(value, raw[name.removesuffix(".weight") + ".weight_scale"])
        elif value.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected nonexpert unquantized dtype at {name}: {value.dtype}")
        canonical = name.replace("model.language_model.", "model.", 1)
        if canonical in output:
            raise ValueError(f"Canonical name collision: {canonical}")
        output[canonical] = value.contiguous()
    return output


def preprocess(source_directory, output_directory, identity):
    """校验原始分片后写全新产物；不分配完整 BF16 专家，不覆盖旧产物。"""
    source_directory, output_directory = Path(source_directory), Path(output_directory)
    for shard in identity["shards"] + identity.get("metadata", []):
        actual = file_hash(source_directory / shard["filename"])
        if actual != shard["sha256"]:
            raise ValueError(f"Source checksum mismatch: {shard['filename']}")
    output_directory.mkdir(parents=True, exist_ok=False)
    expert_directory = output_directory / "experts"
    expert_directory.mkdir()
    full_config = json.loads((source_directory / "config.json").read_text(encoding="utf-8"))
    config = full_config["text_config"]
    index = json.loads((source_directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    report = inspect_checkpoint(source_directory)
    geometry = validate_layout(report, config)
    # 完整布局报告保存在产物中，允许逐张量审计实际 dtype/偏移。
    (output_directory / "source-layout.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = {"status": "RUNNING", "started_at": time.time(), "source": identity,
                "layout_version": "nvfp4-stream-v1", "numerical_policy": NUMERICAL_POLICY,
                "geometry": geometry, "expert_layers": [], "nonexpert_files": [],
                "source_metadata_sha256": {}, "output_files_sha256": {}, "e1_samples": []}
    manifest_path = output_directory / "manifest.json"
    generator = torch.Generator().manual_seed(211)
    for layer in range(geometry["layers"]):
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        names = [name for name in weight_map if name.startswith(prefix)]
        raw = load_group(source_directory, weight_map, names)
        packed = pack_layer(raw, layer, geometry["experts"])
        sampled = torch.randperm(geometry["experts"], generator=generator)[:min(8, geometry["experts"])].tolist()
        for expert in sampled:
            for projection in PROJECTIONS:
                for field in FIELDS:
                    original = raw[f"{prefix}{expert}.{projection}.{field}"].contiguous().reshape(-1).view(torch.uint8)
                    copied = packed[f"{projection}.{field}"][expert].contiguous().reshape(-1).view(torch.uint8)
                    if not torch.equal(original, copied):
                        raise ValueError(f"E1 mismatch: layer={layer}, expert={expert}, {projection}.{field}")
        del raw
        filename = f"layer-{layer:02d}.safetensors"
        save_file(packed, expert_directory / filename)
        header = read_header(expert_directory / filename)
        manifest["expert_layers"].append({"layer": layer, "file": filename,
                                         "sha256": file_hash(expert_directory / filename),
                                         **{k: v for k, v in header.items() if k not in ["file", "metadata"]}})
        manifest["e1_samples"].append({"layer": layer, "experts": sampled, "fields_per_expert": 12, "status": "PASS"})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"packed layer {layer}: {header['payload_bytes']} bytes", flush=True)
        del packed
        gc.collect()
    groups = {}
    for name in weight_map:
        if (name.startswith("model.language_model.") or name.startswith("lm_head.")) and ".mlp.experts." not in name:
            layer = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
            key = f"layer-{int(layer.group(1)):02d}" if layer else name.split(".")[-2]
            groups.setdefault(key, []).append(name)
    model_index = {}
    parameter_bytes = 0
    for key, names in sorted(groups.items()):
        raw = load_group(source_directory, weight_map, names)
        values = decode_nonexpert_group(raw)
        filename = f"nonexpert-{key}.safetensors"
        save_file(values, output_directory / filename)
        for name in values:
            if name in model_index:
                raise ValueError(f"Duplicate model key: {name}")
            model_index[name] = filename
        payload = sum(x.numel() * x.element_size() for x in values.values())
        parameter_bytes += payload
        manifest["nonexpert_files"].append({"file": filename, "sha256": file_hash(output_directory / filename),
                                           "payload_bytes": payload, "keys": list(values)})
        print(f"decoded nonexpert {key}: {payload} bytes", flush=True)
        del raw, values
        gc.collect()
    config = dict(config)
    config.pop("quantization_config", None)
    config.update(architectures=["Qwen3_5MoeForCausalLM"], use_cache=False, output_router_logits=False)
    (output_directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_directory / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": parameter_bytes}, "weight_map": model_index}, indent=2), encoding="utf-8")
    expert_index = {"layout_version": "packed-nvfp4-experts-v1", "source": identity,
                    "geometry": geometry, "layers": manifest["expert_layers"]}
    (expert_directory / "index.json").write_text(json.dumps(expert_index, indent=2), encoding="utf-8")
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "vocab.json", "generation_config.json", "README.md", "LICENSE", "LICENSE.txt"]:
        original = source_directory / filename
        if original.exists():
            manifest["source_metadata_sha256"][filename] = file_hash(original)
            if filename != "config.json":
                shutil.copyfile(original, output_directory / filename)
    for path in output_directory.iterdir():
        if path.is_file() and path.name not in ["manifest.json", "source-layout.json"]:
            manifest["output_files_sha256"][path.name] = file_hash(path)
    info = psutil.Process().memory_info()
    manifest.update(status="PASS", ended_at=time.time(), nonexpert_payload_bytes=parameter_bytes,
                    cpu_memory=info._asdict(), peak_working_set_bytes=info.peak_wset)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if info.peak_wset > 16_000_000_000:
        manifest.update(status="FAIL", error="A1.5 CPU peak working set exceeded 16 GB")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError(manifest["error"])
    return manifest
