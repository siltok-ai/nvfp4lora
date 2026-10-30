"""验证显式模型族身份、注册表扩展和发布格式互斥的拒绝边界。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nvfp4lora.families import (FAMILIES, family_for_built, family_for_config, family_named,
                               family_of_adapter, family_of_bundle, family_of_source)
from nvfp4lora.export.artifacts import load_adapter, publish_adapter, read_manifest


class FamilyTests(unittest.TestCase):
    def test_audit_reports_are_family_specific_and_fail_closed(self):
        self.assertTrue(family_named('dense').audit_passed({}))
        self.assertFalse(family_named('moe').audit_passed({}))
        report = {name: True for name in ['unused_raw_gradients_exact_zero',
                  'never_routed_parameters_unchanged', 'never_routed_adam_moments_zero']}
        self.assertTrue(family_named('moe').audit_passed(report))
        self.assertFalse(family_named('dense').audit_passed(report))
        report['never_routed_parameters_unchanged'] = False
        self.assertFalse(family_named('moe').audit_passed(report))

    def test_layout_and_source_resolution_and_rejections(self):
        for family in FAMILIES.values():
            self.assertIs(family_of_bundle({'layout_version': family.bundle_layout, 'status': 'PASS'}), family)
            self.assertIs(family_of_adapter({'layout_version': family.adapter_layout, 'status': 'complete'}), family)
            self.assertIs(family_named(family.short_name), family)
            for model_type in family.source_types:
                self.assertIs(family_of_source({'model_type': model_type}), family)
            self.assertIs(family_for_config({'family': family.name}), family)
            self.assertIs(family_for_built(SimpleNamespace(family=family.name)), family)
            for manifest in [{}, {'layout_version': family.bundle_layout, 'status': 'FAIL'},
                             {'layout_version': family.bundle_layout, 'status': 'PASS', 'family': 'unknown'}]:
                with self.assertRaises(ValueError):
                    family_of_bundle(manifest)
            with self.assertRaises(ValueError):
                family_of_adapter({'layout_version': family.adapter_layout, 'status': 'incomplete'})
        for config in [{}, {'family': 'dense'}, {'family': 'unknown'}, []]:
            with self.assertRaises(ValueError):
                family_for_config(config)
        with self.assertRaises(ValueError):
            family_for_built(SimpleNamespace())
        for config in [{}, {'model_type': 'unregistered'},
                       {'model_type': 'qwen3_5_moe', 'text_config': {'model_type': 'qwen3_5_text'}}]:
            with self.assertRaises(ValueError):
                family_of_source(config)

    def test_extra_family_requires_no_resolver_branch(self):
        future = replace(family_named('dense'), name='future', short_name='future',
                         bundle_layout='future-bundle', adapter_layout='future-adapter', source_types=('future-model',))
        with patch.dict(FAMILIES, {'future': future}):
            self.assertIs(family_of_bundle({'layout_version': 'future-bundle', 'status': 'PASS'}), future)
            self.assertIs(family_of_adapter({'layout_version': 'future-adapter', 'status': 'complete'}), future)
            self.assertIs(family_of_source({'model_type': 'future-model'}), future)
            self.assertIs(family_for_config({'family': 'future'}), future)

    def test_cross_family_publishing_loading_and_configuration_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in FAMILIES.values():
                bundle = root / (family.short_name + '-bundle')
                adapter = root / (family.short_name + '-adapter')
                bundle.mkdir()
                adapter.mkdir()
                (bundle / 'manifest.json').write_text(json.dumps({'layout_version': family.bundle_layout, 'status': 'PASS'}))
                (adapter / 'manifest.json').write_text(json.dumps({'layout_version': family.adapter_layout, 'family': family.name, 'status': 'complete'}))
            for family in FAMILIES.values():
                other, = [item for item in FAMILIES.values() if item.name != family.name]
                bundle = root / (other.short_name + '-bundle')
                adapter = root / (family.short_name + '-adapter')
                built = SimpleNamespace(family=family.name)
                with self.assertRaisesRegex(ValueError, 'differs from bundle'):
                    family_for_config({'family': family.name}, bundle)
                with self.assertRaisesRegex(ValueError, 'differs from bundle'):
                    publish_adapter(root / ('bad-' + family.short_name), built, bundle, {}, {}, code_commit='unit', global_step=0)
                self.assertFalse((root / ('bad-' + family.short_name)).exists())
                with self.assertRaisesRegex(ValueError, 'differs from bundle'):
                    load_adapter(built, adapter, bundle)
                with self.assertRaisesRegex(ValueError, 'differs from bundle'):
                    read_manifest(adapter, bundle)
                with self.assertRaisesRegex(ValueError, 'Unsupported or incomplete'):
                    other.read_manifest(adapter, bundle)


if __name__ == '__main__':
    unittest.main()
