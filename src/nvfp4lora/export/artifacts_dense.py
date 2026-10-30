"""标准 PEFT dense 发布格式的身份、读取、写入和参数恢复。"""

import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from nvfp4lora.format.bundle import file_hash
from nvfp4lora.model.common import canonical_name
from nvfp4lora.families.dense import normalize_build
from nvfp4lora.train.checkpoint import configuration_hash

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
    if built.family != "qwen3_5_dense":
        raise ValueError("Dense adapter requires a dense built model")
    build = normalize_build(build_kwargs, require_adapter=True)
    rank, alpha = build["lora_rank"], build["lora_alpha"]
    manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    expected = expected_dense_trainable(manifest["geometry"], rank, build["lora_scope"])
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
    build = normalize_build(manifest["build_kwargs"], require_adapter=True)
    rank, alpha = build["lora_rank"], build["lora_alpha"]
    bundle_manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
    expected = expected_dense_trainable(bundle_manifest["geometry"], rank, build.get("lora_scope", "D-ALL"))
    targets = sorted({name.rsplit(".lora_", 1)[0] for name in expected})
    if (metadata["rank"] != rank or metadata["alpha"] != alpha or metadata["scaling"] != alpha / rank
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
    if built.family != "qwen3_5_dense":
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
