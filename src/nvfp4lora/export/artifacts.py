"""通过模型族注册表读写发布产物，并核对 bundle 与模型身份。"""

import json
from pathlib import Path

from nvfp4lora.families import family_for_built, family_of_adapter, family_of_bundle
from nvfp4lora.format.bundle import file_hash


def base_identity(bundle):
    """返回实际 bundle 绑定的权重、策略和 tokenizer 内容身份。"""
    return family_of_bundle(bundle).base_identity(bundle)


def _matching_family(directory, bundle):
    family = family_of_adapter(directory)
    if family.name != family_of_bundle(bundle).name:
        raise ValueError("Adapter family differs from bundle")
    return family


def publish_adapter(directory, built, bundle, build_kwargs, train_config, *, code_commit, global_step):
    """以模型族已有的发布格式保存当前实际 LoRA 参数。"""
    family = family_for_built(built)
    if family.name != family_of_bundle(bundle).name:
        raise ValueError("Built model family differs from bundle")
    return family.publish(directory, built, bundle, build_kwargs, train_config,
                          code_commit=code_commit, global_step=global_step)


def read_manifest(directory, bundle):
    """先匹配发布与 bundle 的族，再执行具体格式的完整校验。"""
    return _matching_family(directory, bundle).read_manifest(directory, bundle)


def load_adapter(built, directory, bundle):
    """校验族身份后写入既有 Parameter，不替换参数对象。"""
    family = _matching_family(directory, bundle)
    if family.name != family_for_built(built).name:
        raise ValueError("Adapter family differs from built model")
    return family.load_adapter(built, directory, bundle)


def build_from_adapter(bundle, directory, *, device="cuda"):
    """只使用 bundle 和发布产物重建模型，不读取训练检查点。"""
    family = _matching_family(directory, bundle)
    manifest = family.read_manifest(directory, bundle)
    built = family.build(bundle, device=device, **manifest["build_kwargs"])
    family.load_adapter(built, directory, bundle)
    return built


def read_adapter_tensors(directory, bundle=None):
    """校验文件后返回规范参数及其文件映射；提供 bundle 时同时校验基座。"""
    directory = Path(directory)
    family = family_of_adapter(directory)
    if bundle is not None:
        read_manifest(directory, bundle)
    else:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for filename, expected in manifest["files"].items():
            if Path(filename).name != filename or file_hash(directory / filename) != expected:
                raise ValueError("Adapter file checksum mismatch: " + filename)
    return family.read_tensors(directory)
