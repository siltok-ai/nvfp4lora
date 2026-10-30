"""文档链接、证据边界及样稿检查范围的阳性和阴性对照。"""
from pathlib import Path
import tempfile
import unittest

from tools.check_docs import check_documents


class DocumentControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'docs/evidence/current').mkdir(parents=True)
        self.evidence = self.root / 'docs/evidence/current/result.json'
        self.evidence.write_text('{"status":"PASS"}', encoding='utf-8')
        self.readme = '# 示例项目\n\n本文介绍项目。\n\n[使用步骤](docs/guide.md#使用步骤-1)\n'
        (self.root / 'README.md').write_text(self.readme, encoding='utf-8')
        (self.root / 'docs/guide.md').write_text('# 使用指南\n\n本文说明用法。\n\n## 使用步骤\n\n步骤。\n\n'
            '## 使用步骤\n\n[证据](evidence/current/result.json)\n\n```markdown\n[示意](missing.md)\n```\n', encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'PASS')

    def test_relative_links_cjk_duplicate_anchors_and_invalid_content(self):
        for text in ['[错误](docs/guide.md#missing)', '[错误](docs/missing.md)', '`C:/machine/private`',
                     'MR8.2', '此前', 'PASS_WITH_DOCUMENTED_CHANGE']:
            with self.subTest(text=text):
                (self.root / 'README.md').write_text(self.readme + text + '\n', encoding='utf-8')
                self.assertEqual(check_documents(self.root)['status'], 'FAIL')

    def test_orphan_recorder_manifest_size_and_explicit_selected_scope(self):
        orphan = self.root / 'docs/evidence/current/orphan.json'
        orphan.write_text('{}', encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'FAIL')
        selected = check_documents(self.root, ['README.md'])
        self.assertEqual(selected['status'], 'PASS')
        self.assertFalse(selected['evidence_coverage_and_global_budgets_checked'])
        orphan.unlink()
        self.evidence.write_text('{"command":[],"source_files":[]}', encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'FAIL')
        self.evidence.write_text('{"payload":"' + 'x' * 200_000 + '"}', encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'FAIL')

    def test_single_artifact_path_and_line_limit(self):
        (self.root / 'docs/results').mkdir()
        artifact = self.root / 'docs/results/sample.md'
        artifact.write_text('# 结果\n\n本文列出结果。\n\n## 制品\n\n`C:/artifacts/20261030T010203Z`\n', encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'PASS')
        with artifact.open('a', encoding='utf-8') as stream:
            stream.write('\n`C:/artifacts/20261030T010203Z`\n')
        self.assertEqual(check_documents(self.root)['status'], 'FAIL')
        artifact.unlink()
        (self.root / 'README.md').write_text(self.readme + '\n' * 101, encoding='utf-8')
        self.assertEqual(check_documents(self.root)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
