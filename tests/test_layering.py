"""用 AST 固化共享层、模型族模块和注册表之间的依赖边界。"""
import ast
from importlib.util import resolve_name
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAMILY_LITERALS = {'qwen3_5_moe', 'qwen3_5_dense', 'nvfp4-stream-v1', 'nvfp4-stream-dense-v1',
                   'nvfp4-lora-v1', 'nvfp4-lora-dense-v1', '.mlp.experts.'}
MOE_MODULES = {'experts_fn', 'packed_experts', 'route_audit', 'route_probe', 'experts_reference'}
REGISTRY_PREFIX = 'nvfp4lora.families'


def owner(module):
    stem = module.rsplit('.', 1)[-1]
    if stem.endswith('_moe') or stem in MOE_MODULES:
        return 'moe'
    if stem.endswith('_dense'):
        return 'dense'
    return None


def violations(module, text):
    """检查源码，不导入模型或初始化 CUDA；返回违规所在行和原因。"""
    if module == REGISTRY_PREFIX or module.startswith(REGISTRY_PREFIX + '.'):
        return []
    kind, problems = owner(module), []
    for node in ast.walk(ast.parse(text)):
        imports = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = (resolve_name('.' * node.level + (node.module or ''), module.rpartition('.')[0])
                      if node.level else node.module or '')
            imports = [target, *[target + '.' + alias.name for alias in node.names]]
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and (isinstance(node.func, ast.Name) and node.func.id in {'__import__', 'import_module'}
                   or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')):
            imports = [node.args[0].value]
        for imported in imports:
            target_kind = owner(imported)
            # Concrete family glue is accessible through the registry, never from a shared layer.
            if imported in {REGISTRY_PREFIX + '.moe', REGISTRY_PREFIX + '.dense'}:
                target_kind = imported.rsplit('.', 1)[-1]
            if target_kind and kind != target_kind:
                problems.append((node.lineno, 'forbidden dependency: ' + imported))
        if (kind is None and isinstance(node, ast.Constant)
                and isinstance(node.value, str) and node.value in FAMILY_LITERALS):
            problems.append((node.lineno, 'family/layout literal: ' + node.value))
    return problems


class LayeringTests(unittest.TestCase):
    def test_all_library_modules_follow_layering(self):
        failures = {}
        for path in sorted((ROOT / 'src/nvfp4lora').rglob('*.py')):
            module = '.'.join(path.relative_to(ROOT / 'src').with_suffix('').parts)
            if module.endswith('.__init__'):
                module = module.removesuffix('.__init__')
            found = violations(module, path.read_text(encoding='utf-8'))
            if found:
                failures[path.relative_to(ROOT).as_posix()] = found
        self.assertEqual(failures, {})

    def test_shared_flow_tools_use_the_registry(self):
        names = ['fetch_checkpoint', 'preprocess_checkpoint', 'train_lora', 'verify_training_release',
                 'evaluate_training_adapter', 'prepare_canary_replay', 'audit_fp8_input_ranges',
                 'prepare_tiny', 'audit_snapshot', 'validation/summarize_training_session',
                 'validation/profile_training']
        failures = {name: violations('tools.' + name.replace('/', '.'), (ROOT / 'tools' / (name + '.py')).read_text(encoding='utf-8'))
                    for name in names}
        self.assertEqual({key: value for key, value in failures.items() if value}, {})

    def test_negative_controls_detect_imports_literals_and_cross_family(self):
        cases = [
            ('nvfp4lora.train.session', 'from nvfp4lora.model.build_dense import build_dense_model'),
            ('nvfp4lora.model.adapters', 'from .packed_experts import PackedExperts'),
            ('nvfp4lora.model.adapters', 'from . import build_moe'),
            ('nvfp4lora.train.session', 'import nvfp4lora.families.dense as family'),
            ('nvfp4lora.train.session', 'import_module("nvfp4lora.model.build_dense")'),
            ('nvfp4lora.train.session', 'family = "qwen3_5_moe"'),
            ('nvfp4lora.export.artifacts', 'layout = "nvfp4-lora-dense-v1"'),
            ('nvfp4lora.model.build_dense', 'from nvfp4lora.model.build_moe import build_model'),
            ('nvfp4lora.model.build_moe', 'from nvfp4lora.validation.cpu_layers_dense import reference_layers_dense'),
        ]
        for module, source in cases:
            with self.subTest(module=module, source=source):
                self.assertTrue(violations(module, source))
        self.assertEqual(violations('nvfp4lora.families', 'from nvfp4lora.model.build_moe import build_model\nfrom nvfp4lora.model.build_dense import build_dense_model'), [])


if __name__ == '__main__':
    unittest.main()
