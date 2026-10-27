"""归档通过机械验收的训练输出；只写新目录，逐文件校验，不复制全量基座。"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from nvfp4lora.export.artifacts import base_identity
from nvfp4lora.format.preprocess import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_pre_audit(path):
    result = read(path)
    assert result["status"] in {"INCOMPLETE", "QUALIFIED_WITH_DOCUMENTED_CHANGE"}
    assert len(result["gates"]) >= 32
    allowed = {"Delivery reproducible code and resumable artifacts", "A6.4 effects and reload agreement", "A7.3/A6.4 effects and reload agreement"}
    for name, gate in result["gates"].items():
        assert gate["status"] in {"PASS", "PASS_WITH_DOCUMENTED_CHANGE", "PENDING"}, (name, gate)
        if gate["status"] == "PENDING":
            assert name in allowed, f"Technical gate still incomplete: {name}"
            if name != "Delivery reproducible code and resumable artifacts":
                assert "owner thresholds/provenance not supplied" in gate["reason"], gate
    assert all(file_hash(Path(name)) == digest for name, digest in result["source_files_sha256"].items()), "Pre-audit evidence changed; rerun it"
    return result


def verified_copy(plan, output):
    """plan是目标相对路径→源文件；失败目录保留，不覆盖原有内容。"""
    output = Path(output)
    assert not output.exists(), "Use a new artifact directory"
    for relative, source in plan.items():
        relative = Path(relative)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert Path(source).is_file() and not Path(source).is_symlink()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative, source in plan.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_hash(source)
        shutil.copy2(source, target)
        assert file_hash(target) == before == file_hash(source), f"Copy changed: {relative}"
        hashes[Path(relative).as_posix()] = before
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--pre-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pre = verify_pre_audit(args.pre_audit)
    assert not args.output.exists()
    plan, sources, commits, runtime_sources = {}, {}, {}, {}

    def add(source, relative):
        source = Path(source)
        assert source.is_file() and relative not in plan, (source, relative)
        plan[relative] = source

    def tree(source, relative):
        assert source.is_dir(), source
        for path in source.rglob("*"):
            if path.is_file():
                assert path.resolve().is_relative_to(source.resolve())
                add(path, f"{relative}/{path.relative_to(source).as_posix()}")

    protected = [root, args.bundle, args.corpus, args.pre_audit.parent]
    for variant, destination in [("c3", "c3"), ("expert", "expert-r2")]:
        def recorded(suffix):
            pointer = args.task / f"real-{variant}-{suffix}-run.txt"
            run = Path(pointer.read_text(encoding="utf-8-sig").strip())
            record = read(run / "manifest.json")
            assert record["status"] == "PASS" and record["exit_code"] == 0, run
            assert str(run / "manifest.json") in pre["source_files_sha256"], "Run was not covered by pre-audit"
            sources[f"{variant}/{suffix}"] = str(run)
            protected.append(run)
            return run

        run = recorded("functional100")
        for item in read(run / "manifest.json")["source_files"]:
            relative = item["path"]
            if relative.startswith("src/") and relative != "src/nvfp4lora/validation/route_probe.py":
                assert file_hash(root / relative) == item["sha256"], f"Runtime source changed: {relative}"
                runtime_sources[relative] = item["sha256"]
        results = run / "results"
        summary = read(results / "summary.json")
        assert summary["status"] == "PASS" and summary["current_step"] == 100
        commits[variant] = read(results / "execution.json")["code_commit"]
        for relative in ["summary.json", "train_config.json", "execution.json", "metrics.jsonl", "delta-w.json", "evaluation_identity.json"]:
            add(results / relative, f"{destination}/{relative}")
        for path in results.glob("evaluation-step*.json"):
            add(path, f"{destination}/{path.name}")
        tree(results / "publish", f"{destination}/publish")
        for step in [20, 50, 100]:
            checkpoint = results / "checkpoints" / f"step-{step:06d}"
            manifest = read(checkpoint / "checkpoint_manifest.json")
            assert manifest["status"] == "complete" and all(file_hash(checkpoint / name) == digest for name, digest in manifest["files"].items())
            tree(checkpoint, f"{destination}/checkpoints/step-{step:06d}")
        for suffix, filename in [("resume50-comparison", "comparison.json"), ("functional-release", "results/results.json"),
                                 ("functional-evaluation", "results/results.json"), ("functional-fp8", "results/results.json"),
                                 ("checkpoint-diagnostics", "results/results.json")]:
            source_run = recorded(suffix)
            add(source_run / filename, f"evidence/{destination}-{suffix}.json")
            if suffix == "checkpoint-diagnostics":
                for relative in ["src/nvfp4lora/validation/route_probe.py", "tools/inspect_training_checkpoints.py"]:
                    assert file_hash(source_run / "source" / relative) == file_hash(root / relative)
                    runtime_sources[relative] = file_hash(root / relative)
        origin = Path((args.task / f"real-{variant}-baseline-run.txt").read_text(encoding="utf-8-sig").strip())
        assert read(origin / "results/evaluation_identity.json") == read(results / "evaluation_identity.json")
        add(origin / "results/evaluation-step000000-disabled.json", f"{destination}/evaluation-step000000-disabled.json")

    corpus = read(args.corpus / "manifest.json")
    assert corpus["base"] == base_identity(args.bundle)
    assert all(file_hash(args.corpus / name) == digest for name, digest in corpus["files"].items())
    for filename in [*corpus["files"], "manifest.json"]:
        add(args.corpus / filename, f"data/canary-replay-v1/{filename}")
    add(args.pre_audit, "evidence/pre-delivery-m2-audit.json")
    add(args.bundle / "manifest.json", "base/manifest.json")
    add(root / "requirements/windows-cu130.lock", "reproduce/requirements/windows-cu130.lock")
    add(root / "pyproject.toml", "reproduce/pyproject.toml")
    for source in protected:
        assert not args.output.resolve().is_relative_to(source.resolve()), "Do not write artifacts into a source/evidence tree"
    lock = (root / "requirements/windows-cu130.lock").read_text(encoding="utf-8")
    pinned = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", lock, flags=re.MULTILINE))
    assert len(pinned) == 39 and all(importlib.metadata.version(name) == version for name, version in pinned.items())
    git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    head = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.validate_only:
        print(json.dumps({"status": "VALIDATED_PLAN_ONLY", "files": len(plan), "copy_bytes": sum(path.stat().st_size for path in plan.values())}, indent=2))
        return
    assert not subprocess.check_output([git, "status", "--porcelain"], cwd=root, text=True).strip(), "Commit source changes before archiving a reproducible delivery"
    hashes = verified_copy(plan, args.output)
    environment = {"python": sys.version, "interpreter": sys.executable, "platform": platform.platform(), "pinned_packages": pinned}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    readme = f"""# NVFP4 LoRA 本机制品

源代码：https://github.com/wenlive/nvfp4lora @ {head}
训练数值代码提交：{json.dumps(commits)}。基座引用 `{args.bundle}`；未复制或合并原始专家。

`c3` 和 `expert-r2` 各自保持训练results的相对目录：publish、完整20/50/100检查点及评测/逐步/ΔW证据。
固定语料在data/canary-replay-v1；依赖锁在reproduce/requirements。环境重建按源码docs/environment.md，保持原生Windows。

从step50恢复到100：用tools/train_lora.py，--data指向本制品train.jsonl，--config和--resume分别指向所选variant的checkpoints/step-000050/train_config.json和该检查点目录，--stop-at 100，--output使用新的run/results。
逻辑max_steps为100；不得静默改数据或上限来声称精确续训。step100是本计划终点。

发布重载：tools/verify_training_release.py --bundle <基座> --training-results <本制品>/<variant> --output <新目录>。
生成使用build_from_adapter(<基座>, <本制品>/<variant>/publish)，接口示例见源码docs/real-training-cli.md。
后验诊断可对本制品variant运行inspect_training_checkpoints.py --steps 20 50 100，并使用本制品corpus。

manifest.status=verified表示复制文件与机械验收来源逐字节一致；不是用户效果门槛的自动批准。
正式M2仍需在新run以本目录作--artifact-dir执行audit_m2.py --require-complete，并提供真实用户阈值及provenance。
原A5.2无条件FP32 1%失败和预先记录的路由拆分调整仍保留；金丝雀/算术回放不代表领域收益。
Speed/FreeToken产品加载器尚未接入。基座和适配器的部署数值差异/FP8风险见evidence，G2/G3另行。
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    for filename in ["README.md", "environment.json"]:
        hashes[filename] = file_hash(args.output / filename)
    manifest = {"layout_version": "nvfp4lora-delivery-v1", "status": "verified", "verification_scope": "Copy integrity and mechanical source qualification; owner effect thresholds remain separate",
                "created_utc": datetime.now(timezone.utc).isoformat(), "source_repository": "https://github.com/wenlive/nvfp4lora",
                "source_commit": head, "training_commits": commits, "source_runs": sources, "runtime_source_sha256": runtime_sources, "base_bundle": str(args.bundle),
                "base": base_identity(args.bundle), "pre_audit_sha256": file_hash(args.pre_audit), "files": hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "VERIFIED_COPY", "output": str(args.output), "files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
