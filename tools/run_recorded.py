"""统一保存探针命令、源码快照、输入版本、输出和真实退出码。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    """每个独立运行使用新目录；参数中的 {run} 指向本次证据根目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [part.replace("{run}", str(output)) for part in command]
    if command[0].endswith(".py"):
        command = [sys.executable, "-B", *command]
    git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    snapshots = []
    for folder in ["tools", "src", "requirements", "tests", "configs"]:
        for source in (root / folder).rglob("*"):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(root)
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            snapshots.append({"path": relative.as_posix(),
                              "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    project_file = root / "pyproject.toml"
    shutil.copyfile(project_file, output / "source" / "pyproject.toml")
    snapshots.append({"path": "pyproject.toml", "sha256": hashlib.sha256(project_file.read_bytes()).hexdigest()})
    record = {"status": "RUNNING", "start": time.time(), "command": command, "cwd": str(root),
              "runner_pid": os.getpid(), "python": sys.executable, "platform": platform.platform(),
              "git_head": subprocess.check_output([git, "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_status": subprocess.check_output([git, "status", "--porcelain"], cwd=root, text=True).splitlines(),
              "packages": {name: metadata.version(name) for name in
                           ["torch", "transformers", "peft", "safetensors", "triton-windows"]},
              "source_files": snapshots}
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Run: {output}", flush=True)
    with (output / "stdout-stderr.log").open("wb") as log:
        # 每次子进程显式UTF-8，避免Windows区域编码使中文证据日志不可读。
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        record["child_pid"] = process.pid
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        code = process.wait()
    record.update(status="PASS" if code == 0 else "FAIL", exit_code=code, end=time.time())
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Exit: {code}; log: {output / 'stdout-stderr.log'}", flush=True)
    if code:
        print((output / "stdout-stderr.log").read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
