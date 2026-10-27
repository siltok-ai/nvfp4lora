"""标准 PEFT 文件＋自有专家 LoRA，以及可逆 canonical 映射和固定基座绑定。"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nvfp4lora.format.preprocess import file_hash
from nvfp4lora.model.build import build_model, canonical_name
from nvfp4lora.model.packed_experts import PackedExperts
from nvfp4lora.train.checkpoint import configuration_hash


def base_identity(bundle):
    """只绑定影响模型函数的内容，排除时间、工作路径和运行内存统计。"""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    identity = {"source": manifest["source"], "layout_version": manifest["layout_version"],
                "nonexpert_numerical_policy": manifest["numerical_policy"],
                "expert_numerical_policy": "torch-linear-bf16-result-fp32-global-lora-swiglu-route-v1",
                "config_sha256": file_hash(bundle / "config.json"),
                "nonexpert_files": {x["file"]: x["sha256"] for x in manifest["nonexpert_files"]},
                "expert_files": {x["file"]: x["sha256"] for x in manifest["expert_layers"]},
                "tokenizer_source_hashes": {name: value for name, value in manifest["source_metadata_sha256"].items()
                                           if name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json"]}}
    return identity


def _artifact_tensors(directory):
    """标准 PEFT 键和专家键映射为同一规范集合，拒绝任何重复键。"""
    tensors, mapping = {}, {}
    for filename in ["adapter_model.safetensors", "expert_lora.safetensors"]:
        for key, value in load_file(Path(directory) / filename).items():
            canonical = canonical_name(key)
            if canonical in tensors:
                raise ValueError(f"Canonical publish name collision: {canonical}")
            tensors[canonical] = value
            mapping[canonical] = {"file": filename, "key": key}
    return tensors, mapping


def publish_adapter(directory, built, bundle, build_kwargs, train_config, *, code_commit, global_step):
    """两套 LoRA 都从当前 Parameter 保存；发布文件不包含 optimizer 或任意 pickle。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if hasattr(built.model, "peft_config"):
        built.model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False,
                                    selected_adapters=["default"])
        linear_format = "peft-lora"
        config = built.model.peft_config["default"]
        nonexpert = {"rank": config.r, "alpha": config.lora_alpha, "scaling": config.lora_alpha / config.r,
                     "target_modules": config.target_modules, "rsLoRA": config.use_rslora, "dtype": "float32"}
    else:
        save_file({}, directory / "adapter_model.safetensors")
        (directory / "adapter_config.json").write_text(json.dumps({"adapter_type": "none"}), encoding="utf-8")
        linear_format, nonexpert = "none", None
    expert_values = {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in built.trainable.items()
                     if ".mlp.experts." in name}
    save_file(expert_values, directory / "expert_lora.safetensors")
    tensors, mapping = _artifact_tensors(directory)
    if tensors.keys() != built.trainable.keys() or tensors.keys() != built.expected_trainable.keys():
        raise ValueError("Publish set != optimizer/expected trainable set")
    specifications = {}
    for name, value in tensors.items():
        parameter = built.trainable[name]
        if value.dtype != torch.float32 or value.shape != parameter.shape:
            raise ValueError(f"Publish shape/dtype mismatch: {name}")
        original = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8), original):
            raise ValueError(f"Publish parameter bytes differ: {name}")
        specifications[name] = {**mapping[name], "shape": list(value.shape), "dtype": "float32", "numel": value.numel()}
    experts = {}
    for module in built.model.modules():
        if isinstance(module, PackedExperts) and module.rank:
            experts[str(module.layer)] = {"rank": module.rank, "alpha": module.gate_proj.alpha,
                                          "scaling": list(module.op_config.scaling), "rsLoRA": False, "dtype": "float32"}
    identity = base_identity(bundle)
    manifest = {"layout_version": "nvfp4-lora-v1", "status": "complete", "base": identity,
                "base_fingerprint": configuration_hash(identity), "build_kwargs": build_kwargs,
                "linear_format": linear_format, "nonexpert": nonexpert, "experts": experts,
                "expert_layout": "{gate,up,down}_proj.lora_A[E,r,in] and lora_B[E,out,r]",
                "tensor_specifications": specifications, "enable_thinking": False,
                "synthetic_vocabulary_only": identity["source"].get("kind") == "tiny-generated",
                "training_config_sha256": configuration_hash(train_config), "code_commit": code_commit,
                "global_step": global_step,
                "files": {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_manifest(directory, bundle):
    """先检查基座内容身份和全部发布文件字节，再读取任何参数。"""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["layout_version"] != "nvfp4-lora-v1" or manifest["status"] != "complete":
        raise ValueError("Unsupported or incomplete adapter")
    if configuration_hash(base_identity(bundle)) != manifest["base_fingerprint"]:
        raise ValueError("Adapter base fingerprint differs from preprocessed bundle")
    for filename, expected in manifest["files"].items():
        if file_hash(directory / filename) != expected:
            raise ValueError(f"Adapter file checksum mismatch: {filename}")
    return manifest


def load_adapter(built, directory, bundle):
    """不替换 Parameter 对象，检查 scaling/键/形状后写入两套 LoRA。"""
    manifest = read_manifest(directory, bundle)
    tensors, mapping = _artifact_tensors(directory)
    if tensors.keys() != built.trainable.keys() or tensors.keys() != manifest["tensor_specifications"].keys():
        raise ValueError("Adapter trainable set mismatch")
    for module in built.model.modules():
        if isinstance(module, PackedExperts) and module.rank:
            metadata = manifest["experts"][str(module.layer)]
            if module.rank != metadata["rank"] or list(module.op_config.scaling) != metadata["scaling"]:
                raise ValueError(f"Expert adapter scaling/rank mismatch: {module.layer}")
    if hasattr(built.model, "peft_config"):
        config = built.model.peft_config["default"]
        expected = manifest["nonexpert"]
        if config.r != expected["rank"] or config.lora_alpha != expected["alpha"] or config.use_rslora != expected["rsLoRA"]:
            raise ValueError("Nonexpert adapter scaling/rank mismatch")
    with torch.no_grad():
        for name, value in tensors.items():
            parameter = built.trainable[name]
            spec = manifest["tensor_specifications"][name]
            if value.dtype != torch.float32 or value.shape != parameter.shape or list(value.shape) != spec["shape"] or mapping[name] != {k: spec[k] for k in ["file", "key"]}:
                raise ValueError(f"Adapter tensor specification mismatch: {name}")
            parameter.copy_(value)
    return manifest


def build_from_adapter(bundle, directory, *, device="cuda"):
    """新进程只需预处理基座和发布目录，不读取训练检查点。"""
    manifest = read_manifest(directory, bundle)
    built = build_model(bundle, device=device, **manifest["build_kwargs"])
    load_adapter(built, directory, bundle)
    return built
