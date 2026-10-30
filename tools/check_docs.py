"""检查文档链接、锚点、写作约束和证据规模；仅使用 Python 标准库。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

LIMITS = {'README.md': 100, 'docs/architecture.md': 250, 'docs/glossary.md': 80,
          'docs/guides/environment.md': 120, 'docs/guides/preprocess.md': 150,
          'docs/guides/training.md': 250, 'docs/guides/validation.md': 200,
          'docs/design/format.md': 200, 'docs/design/streaming.md': 150,
          'docs/design/numerics.md': 200, 'docs/design/moe.md': 200,
          'docs/design/dense.md': 150, 'docs/design/adapters.md': 150,
          'docs/results/qwen36-35b-a3b-moe.md': 200, 'docs/results/qwen3x-27b-dense.md': 200}
INTERNAL = re.compile(r'\b(?:P[0-9]+|G0\.[0-9]+|A[1-7]\.[0-9]+|M[12]|D[0-7]|DA[0-9]+(?:\.[0-9]+)?|MR[0-9]+(?:\.[0-9]+)?)\b|十项')
LABELS = re.compile(r'PASS_WITH|FALLBACK PASS|QUALIFIED')
HISTORY = re.compile(r'当时|此前|随后|原来|先前|后续补测|原失败')
LOCAL = re.compile(r'(?i)\b[A-Z]:[/\\]')
RUN_ID = re.compile(r'\b20\d{6}T\d{6}Z\b')
LINK = re.compile(r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+"[^"]*")?\)')


def outside_fences(text):
    """保留正文和行号，屏蔽代码块中的 Markdown 链接及标题。"""
    fenced, result = False, []
    for line in text.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            fenced = not fenced
            result.append('')
        else:
            result.append('' if fenced else line)
    return '\n'.join(result)


def anchors(text):
    """生成 GitHub 风格标题锚点，包含同名标题的数字后缀。"""
    found, duplicates = set(), Counter()
    for line in outside_fences(text).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)(?:\s+#+)?\s*$', line)
        if not match:
            continue
        heading = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', match[1])
        heading = re.sub(r'<[^>]*>', '', heading).replace('`', '').replace('*', '')
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        count = duplicates[slug]
        duplicates[slug] += 1
        found.add(slug + ('-' + str(count) if count else ''))
    found.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    return found


def check_documents(root, selected=None):
    """返回检查结果；选定文件模式只检查样稿，最终交付必须运行全量模式。"""
    root = Path(root).resolve()
    all_docs = sorted(set([p for p in (root / 'docs').rglob('*.md')]
                         + [p for p in [root / 'README.md', root / 'ROADMAP.md', root / 'tools/README.md'] if p.exists()]))
    paths = [root / p for p in selected] if selected else all_docs
    failures, referenced = [], set()
    def fail(path, message, line=None):
        failures.append({'file': str(path.relative_to(root)), 'line': line, 'message': message})
    for path in paths:
        if not path.is_file():
            fail(path, '文档不存在')
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()
        if relative in LIMITS and len(lines) > LIMITS[relative]:
            fail(path, f'行数 {len(lines)} 超过上限 {LIMITS[relative]}')
        section, local_count = '', 0
        for number, line in enumerate(lines, 1):
            heading = re.match(r'^##\s+(.+)', line)
            if heading:
                section = heading[1].strip()
            artifact_line = relative.startswith('docs/results/') and section == '制品' and LOCAL.search(line)
            if LOCAL.search(line):
                local_count += len(LOCAL.findall(line))
                if not artifact_line:
                    fail(path, '本机路径只允许在结果文档的制品节出现一次', number)
            for pattern, label in [(INTERNAL, '内部验收编号'), (LABELS, '非公开状态标签'), (HISTORY, '历史叙述词')]:
                if pattern.search(line):
                    fail(path, label + '：' + pattern.search(line)[0], number)
            if RUN_ID.search(line) and not artifact_line:
                fail(path, '正文包含运行目录编号', number)
        if local_count > 1:
            fail(path, '本机制品路径重复出现')
        if relative == 'docs/decisions.md':
            sections = re.split(r'(?m)^## ', text)[1:]
            if any(len(section.splitlines()) > 10 for section in sections):
                fail(path, '决策条目超过 10 行')
        for match in LINK.finditer(outside_fences(text)):
            target = unquote(match[1].strip('<>'))
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (path.parent / parsed.path).resolve() if parsed.path else path
            if not destination.is_relative_to(root) or not destination.exists():
                fail(path, '相对链接目标不存在：' + target)
                continue
            if destination.is_file() and destination.is_relative_to(root / 'docs/evidence'):
                referenced.add(destination)
            if parsed.fragment:
                if destination.suffix == '.md':
                    if parsed.fragment not in anchors(destination.read_text(encoding='utf-8')):
                        fail(path, '标题锚点不存在：' + target)
                elif destination.suffix in ['.py', '.ps1']:
                    line_ref = re.fullmatch(r'L(\d+)(?:-L(\d+))?', parsed.fragment)
                    if not line_ref or max(int(item) for item in line_ref.groups() if item) > len(destination.read_text(encoding='utf-8').splitlines()):
                        fail(path, '代码行锚点无效：' + target)
                else:
                    fail(path, '目标不支持锚点：' + target)
    evidence = [p for p in (root / 'docs/evidence').rglob('*') if p.is_file()]
    budgets = {'evidence_files': len(evidence), 'evidence_bytes': sum(p.stat().st_size for p in evidence),
               'docs_markdown_files': len(list((root / 'docs').rglob('*.md')))}
    if not selected:
        for path in evidence:
            if path.resolve() not in referenced:
                fail(path, '没有文档直接引用此证据')
            if path.stat().st_size > 200_000:
                fail(path, '证据文件超过 200 KB')
            if path.suffix == '.json':
                try:
                    value = json.loads(path.read_text(encoding='utf-8'))
                    if isinstance(value, dict) and {'command', 'source_files'} <= value.keys():
                        fail(path, '运行记录器 manifest 应留在仓外')
                except (ValueError, UnicodeError) as error:
                    fail(path, '无效 JSON：' + str(error))
        for family in ['moe', 'dense']:
            size = sum(p.stat().st_size for p in evidence if p.is_relative_to(root / 'docs/evidence' / family)
                       or (p.parent == root / 'docs/evidence/current' and p.name.startswith(family + '-')))
            budgets[family + '_evidence_bytes'] = size
            if size > 1_000_000:
                fail(root / 'docs/evidence', family + ' 证据超过 1 MB')
        if budgets['evidence_bytes'] > 2_000_000:
            fail(root / 'docs/evidence', '全部证据超过 2 MB')
        if budgets['docs_markdown_files'] > 18:
            fail(root / 'docs', 'docs 下 Markdown 超过 18 份')
    return {'status': 'PASS' if not failures else 'FAIL',
            'scope': 'selected_documents' if selected else 'complete_repository',
            'documents_checked': [p.relative_to(root).as_posix() for p in paths],
            'evidence_coverage_and_global_budgets_checked': not bool(selected),
            'budgets': budgets, 'failures': failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--paths', nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_documents(args.root, args.paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
