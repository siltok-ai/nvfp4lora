"""按明确的配置或产物布局解析模型族，延迟加载各族的实现。"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Family:
    """描述一个模型族的构建、训练、发布和观测入口。"""

    name: str
    short_name: str
    bundle_layout: str
    adapter_layout: str
    logical_config: Callable
    build: Callable
    route_audit: Callable
    base_identity: Callable
    publish: Callable
    read_manifest: Callable
    load_adapter: Callable
    read_tensors: Callable
    extra_inventory: Callable
    replay_stops: Callable
    default_seed: int
    profile: Callable
    base_build: Callable
    preprocess: Callable
    source_types: tuple[str, ...]
    source_kind: str
    fp8_audit: Callable
    decode_frame: Callable


def _lazy(module, attribute):
    """在调用时导入具体实现，避免注册表与构建和发布模块形成导入环。"""
    def invoke(*args, **kwargs):
        return getattr(import_module(module), attribute)(*args, **kwargs)
    return invoke


FAMILIES = {
    'qwen3_5_moe': Family(
        name='qwen3_5_moe', short_name='moe', bundle_layout='nvfp4-stream-v1', adapter_layout='nvfp4-lora-v1',
        logical_config=_lazy('nvfp4lora.families.moe', 'logical_config'),
        build=_lazy('nvfp4lora.model.build_moe', 'build_model'),
        route_audit=_lazy('nvfp4lora.families.moe', 'route_audit'),
        base_identity=_lazy('nvfp4lora.export.artifacts_moe', 'base_identity'),
        publish=_lazy('nvfp4lora.export.artifacts_moe', 'publish_adapter'),
        read_manifest=_lazy('nvfp4lora.export.artifacts_moe', 'read_manifest'),
        load_adapter=_lazy('nvfp4lora.export.artifacts_moe', 'load_adapter'),
        read_tensors=_lazy('nvfp4lora.export.artifacts_moe', '_artifact_tensors'),
        extra_inventory=_lazy('nvfp4lora.families.moe', 'extra_inventory'),
        replay_stops=_lazy('nvfp4lora.families.moe', 'replay_stops'), default_seed=20261028,
        profile=_lazy('nvfp4lora.families.moe', 'profile'),
        base_build=_lazy('nvfp4lora.families.moe', 'base_build'),
        preprocess=_lazy('nvfp4lora.format.preprocess_moe', 'preprocess'),
        source_types=('qwen3_5_moe', 'qwen3_5_moe_text'), source_kind='real-nvidia-checkpoint',
        fp8_audit=_lazy('nvfp4lora.validation.fp8_inputs_moe', 'FP8InputAudit'),
        decode_frame=_lazy('nvfp4lora.families.moe', 'decode_frame')),
    'qwen3_5_dense': Family(
        name='qwen3_5_dense', short_name='dense', bundle_layout='nvfp4-stream-dense-v1', adapter_layout='nvfp4-lora-dense-v1',
        logical_config=_lazy('nvfp4lora.families.dense', 'logical_config'),
        build=_lazy('nvfp4lora.model.build_dense', 'build_dense_model'),
        route_audit=_lazy('nvfp4lora.families.dense', 'route_audit'),
        base_identity=_lazy('nvfp4lora.export.artifacts_dense', 'base_identity_dense'),
        publish=_lazy('nvfp4lora.export.artifacts_dense', 'publish_dense_adapter'),
        read_manifest=_lazy('nvfp4lora.export.artifacts_dense', 'read_dense_manifest'),
        load_adapter=_lazy('nvfp4lora.export.artifacts_dense', 'load_dense_adapter'),
        read_tensors=_lazy('nvfp4lora.export.artifacts_dense', '_dense_artifact_tensors'),
        extra_inventory=_lazy('nvfp4lora.families.dense', 'extra_inventory'),
        replay_stops=_lazy('nvfp4lora.families.dense', 'replay_stops'), default_seed=20261029,
        profile=_lazy('nvfp4lora.families.dense', 'profile'),
        base_build=_lazy('nvfp4lora.families.dense', 'base_build'),
        preprocess=_lazy('nvfp4lora.format.preprocess_dense', 'preprocess_dense'),
        source_types=('qwen3_5', 'qwen3_5_text'), source_kind='real-dense-checkpoint',
        fp8_audit=_lazy('nvfp4lora.validation.fp8_inputs_dense', 'FP8InputAudit'),
        decode_frame=_lazy('nvfp4lora.families.dense', 'decode_frame')),
}


def _manifest(value):
    if isinstance(value, Mapping):
        return value
    return json.loads((Path(value) / 'manifest.json').read_text(encoding='utf-8'))


def _family_of(value, layout_attribute, complete_status):
    manifest = _manifest(value)
    matches = [family for family in FAMILIES.values()
               if getattr(family, layout_attribute) == manifest.get('layout_version')]
    if len(matches) != 1 or manifest.get('status') != complete_status:
        raise ValueError('Unknown or incomplete layout: ' + str(manifest.get('layout_version')))
    family = matches[0]
    if manifest.get('family', family.name) != family.name:
        raise ValueError('Declared family differs from layout_version')
    return family


def family_of_bundle(bundle):
    """按完整 bundle 的 layout_version 解析模型族，不推断缺失布局。"""
    return _family_of(bundle, 'bundle_layout', 'PASS')


def family_of_adapter(directory):
    """按完整发布产物的 layout_version 解析模型族。"""
    return _family_of(directory, 'adapter_layout', 'complete')


def family_for_config(config, bundle=None):
    """要求训练配置显式指定 family，并与可选 bundle 的族一致。"""
    if not isinstance(config, Mapping):
        raise ValueError('Training configuration must be a mapping')
    name = config.get('family')
    if not isinstance(name, str) or name not in FAMILIES:
        raise ValueError('Training configuration needs an explicit registered family')
    family = FAMILIES[name]
    if bundle is not None and family_of_bundle(bundle).name != family.name:
        raise ValueError('Training configuration family differs from bundle')
    return family


def family_for_built(built):
    """只使用 BuiltModel 的显式族身份，不检查 HF 模型的私有属性。"""
    return family_for_config({'family': getattr(built, 'family', None)})


def family_named(name):
    """解析命令行中的短名称或规范族名；训练配置仍要求规范族名。"""
    matches = [family for family in FAMILIES.values() if name in (family.name, family.short_name)]
    if len(matches) != 1:
        raise ValueError('Unknown family: ' + str(name))
    return matches[0]


def family_of_source(config):
    """由来源 config 的模型类型解析族，拒绝未知类型或内外配置冲突。"""
    types = [config.get('model_type')]
    if 'text_config' in config:
        types.append(config['text_config'].get('model_type'))
    resolved = []
    for model_type in types:
        matches = [family for family in FAMILIES.values() if model_type in family.source_types]
        if len(matches) != 1:
            raise ValueError('Unknown source model_type: ' + str(model_type))
        resolved.append(matches[0])
    if any(family.name != resolved[0].name for family in resolved):
        raise ValueError('Conflicting source model types')
    return resolved[0]
