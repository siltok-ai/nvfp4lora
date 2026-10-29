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
    if manifest.get("family") == "qwen3_5_dense":
        return base_identity_dense(bundle)
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
    if getattr(built.model, "_nvfp4lora_family", None) == "qwen3_5_dense":
        return publish_dense_adapter(directory, built, bundle, build_kwargs, train_config,
                                     code_commit=code_commit, global_step=global_step)
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
    if manifest.get("layout_version") == "nvfp4-lora-dense-v1":
        return read_dense_manifest(directory, bundle)
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
    if manifest["layout_version"] == "nvfp4-lora-dense-v1":
        return load_dense_adapter(built, directory, bundle)
    if getattr(built.model, "_nvfp4lora_family", None) == "qwen3_5_dense":
        raise ValueError("MoE adapter layout cannot load into a dense model")
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
    if manifest["layout_version"] == "nvfp4-lora-dense-v1":
        from nvfp4lora.model.build_dense import build_dense_model
        built = build_dense_model(bundle, device=device, **manifest["build_kwargs"])
        load_dense_adapter(built, directory, bundle)
        return built
    built = build_model(bundle, device=device, **manifest["build_kwargs"])
    load_adapter(built, directory, bundle)
    return built


def base_identity_dense(bundle):
    """Dense原始基座、所有分层字节、非流式策略和tokenizer内容身份。"""
    from nvfp4lora.format.preprocess_dense import FAMILY, LAYOUT, LINEAR_POLICY
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("family") != FAMILY
            or manifest.get("layout_version") != LAYOUT or manifest.get("linear_numerical_policy") != LINEAR_POLICY):
        raise ValueError("Not a complete supported dense bundle")
    return {"family": FAMILY, "source": manifest["source"], "layout_version": LAYOUT,
            "linear_numerical_policy": manifest["linear_numerical_policy"],
            "nonstream_numerical_policy": manifest["nonstream_numerical_policy"],
            "config_sha256": file_hash(bundle / "config.json"),
            "nonstream_files": {entry["file"]: entry["sha256"] for entry in manifest["nonstream_files"]},
            "layer_files": {entry["file"]: entry["sha256"] for entry in manifest["layer_files"]},
            "tokenizer_source_hashes": {name: digest for name, digest in manifest["source_metadata_sha256"].items()
                                       if name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json"]}}


def _dense_artifact_tensors(directory):
    tensors, mapping = {}, {}
    filename = "adapter_model.safetensors"
    for key, value in load_file(Path(directory) / filename).items():
        canonical = canonical_name(key)
        if (not key.startswith("base_model.model.model.layers.") or canonical in tensors
                or not canonical.endswith((".lora_A.weight", ".lora_B.weight"))):
            raise ValueError(f"Invalid or duplicate dense adapter key: {key}")
        tensors[canonical] = value
        mapping[canonical] = {"file": filename, "key": key}
    return tensors, mapping


def _check_dense_build(built, bundle, build_kwargs):
    from nvfp4lora.model.build_dense import expected_dense_trainable
    from nvfp4lora.model.packed_linear import LoRAPackedLinear
    if getattr(built.model, "_nvfp4lora_family", None) != "qwen3_5_dense":
        raise ValueError("Dense adapter requires a dense built model")
    if build_kwargs.get("family", "qwen3_5_dense") != "qwen3_5_dense":
        raise ValueError("Dense adapter build family mismatch")
    rank, alpha = build_kwargs.get("lora_rank", 16), build_kwargs.get("lora_alpha")
    alpha = 2 * rank if alpha is None else alpha
    manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    expected = expected_dense_trainable(manifest["geometry"], rank, build_kwargs.get("lora_scope", "D-ALL"))
    if not expected or expected != built.expected_trainable or set(expected) != set(built.trainable):
        raise ValueError("Dense build/publish trainable set mismatch")
    targets = []
    for name, module in built.model.named_modules():
        if isinstance(module, LoRAPackedLinear):
            if module.rank != rank or module.alpha != alpha:
                raise ValueError(f"Dense adapter rank/alpha mismatch: {name}")
            targets.append(name)
    if len(targets) * 2 != len(expected):
        raise ValueError("Dense adapter module count mismatch")
    return rank, alpha, sorted(targets)


def publish_dense_adapter(directory, built, bundle, build_kwargs, train_config, *, code_commit, global_step):
    """标准PEFT A/B文件；新layout明确排除MoE专家文件和训练pickle。"""
    rank, alpha, targets = _check_dense_build(built, bundle, build_kwargs)
    identity = base_identity_dense(bundle)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    values = {}
    for name, parameter in built.trainable.items():
        if parameter.dtype != torch.float32 or list(parameter.shape) != built.expected_trainable[name] or not torch.isfinite(parameter).all():
            raise ValueError(f"Dense publish shape/dtype/nonfinite mismatch: {name}")
        values["base_model.model." + name] = parameter.detach().to(device="cpu", copy=True).contiguous()
    save_file(values, directory / "adapter_model.safetensors")
    adapter_config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
                      "base_model_name_or_path": identity["source"]["repo"], "r": rank, "lora_alpha": alpha,
                      "lora_dropout": 0., "bias": "none", "fan_in_fan_out": False,
                      "target_modules": targets, "use_rslora": False, "use_dora": False}
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2, sort_keys=True), encoding="utf-8")
    tensors, mapping = _dense_artifact_tensors(directory)
    specifications = {}
    if tensors.keys() != built.trainable.keys():
        raise ValueError("Dense published tensor set differs")
    for name, value in tensors.items():
        original = built.trainable[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8), original):
            raise ValueError(f"Dense published bytes differ: {name}")
        specifications[name] = {**mapping[name], "shape": list(value.shape), "dtype": "float32", "numel": value.numel()}
    manifest = {"layout_version": "nvfp4-lora-dense-v1", "family": "qwen3_5_dense", "status": "complete",
                "base": identity, "base_fingerprint": configuration_hash(identity), "build_kwargs": build_kwargs,
                "linear_format": "peft-lora", "nonexpert": {"rank": rank, "alpha": alpha,
                    "scaling": alpha / rank, "rsLoRA": False, "dtype": "float32", "target_modules": targets},
                "tensor_specifications": specifications, "enable_thinking": False,
                "synthetic_vocabulary_only": identity["source"].get("kind") == "tiny-generated",
                "training_config_sha256": configuration_hash(train_config), "code_commit": code_commit,
                "global_step": global_step, "files": {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_dense_manifest(directory, bundle):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("layout_version") != "nvfp4-lora-dense-v1" or manifest.get("family") != "qwen3_5_dense"
            or manifest.get("status") != "complete"):
        raise ValueError("Unsupported or incomplete dense adapter")
    identity = base_identity_dense(bundle)
    if manifest["base"] != identity or configuration_hash(identity) != manifest["base_fingerprint"]:
        raise ValueError("Dense adapter base fingerprint differs")
    if set(manifest["files"]) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("Dense adapter file set differs")
    for filename, digest in manifest["files"].items():
        if file_hash(directory / filename) != digest:
            raise ValueError(f"Dense adapter file checksum mismatch: {filename}")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    metadata = manifest["nonexpert"]
    from nvfp4lora.model.build_dense import expected_dense_trainable
    build = manifest["build_kwargs"]
    rank, alpha = build.get("lora_rank", 16), build.get("lora_alpha")
    alpha = 2 * rank if alpha is None else alpha
    bundle_manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    expected = expected_dense_trainable(bundle_manifest["geometry"], rank, build.get("lora_scope", "D-ALL"))
    targets = sorted({name.rsplit(".lora_", 1)[0] for name in expected})
    if (not rank or build.get("family", "qwen3_5_dense") != "qwen3_5_dense"
            or metadata["rank"] != rank or metadata["alpha"] != alpha or metadata["scaling"] != alpha / rank
            or metadata["rsLoRA"] or metadata["dtype"] != "float32" or sorted(metadata["target_modules"]) != targets):
        raise ValueError("Dense build/scaling/target metadata mismatch")
    if (config["peft_type"] != "LORA" or config.get("task_type") != "CAUSAL_LM" or config.get("fan_in_fan_out", False)
            or config.get("base_model_name_or_path") != identity["source"]["repo"] or config.get("modules_to_save")
            or config["r"] != metadata["rank"] or config["lora_alpha"] != metadata["alpha"]
            or config["use_rslora"] or config["use_dora"] or config["bias"] != "none" or config["lora_dropout"] != 0
            or sorted(config["target_modules"]) != sorted(metadata["target_modules"])):
        raise ValueError("Dense PEFT config/manifest mismatch")
    return manifest


def load_dense_adapter(built, directory, bundle):
    if getattr(built.model, "_nvfp4lora_family", None) != "qwen3_5_dense":
        raise ValueError("Dense adapter cannot load into a MoE or unknown model family")
    manifest = read_dense_manifest(directory, bundle)
    _check_dense_build(built, bundle, manifest["build_kwargs"])
    tensors, mapping = _dense_artifact_tensors(directory)
    if tensors.keys() != built.trainable.keys() or tensors.keys() != manifest["tensor_specifications"].keys():
        raise ValueError("Dense adapter tensor set mismatch")
    for name, value in tensors.items():
        parameter, spec = built.trainable[name], manifest["tensor_specifications"][name]
        if (value.dtype != torch.float32 or value.shape != parameter.shape or list(value.shape) != spec["shape"]
                or spec["dtype"] != "float32" or mapping[name] != {key: spec[key] for key in ["file", "key"]}
                or not torch.isfinite(value).all()):
            raise ValueError(f"Dense adapter tensor specification mismatch: {name}")
    # Validate every tensor before changing any live Parameter.
    with torch.no_grad():
        for name, value in tensors.items():
            built.trainable[name].copy_(value)
    return manifest
