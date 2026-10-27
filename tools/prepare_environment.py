"""从 Speed 实装元数据解析最小同版本依赖闭包，下载 wheel 并生成哈希锁。"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOTS = ["torch", "triton-windows", "transformers", "peft", "safetensors", "tokenizers", "psutil"]
TORCH_WHEEL = Path("C:/workspace/tools/downloads/speed-packaging/torch-2.11.0+cu130-cp312-cp312-win_amd64.whl")


def digest(path):
    """分块计算文件哈希，避免大 wheel 进入内存。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependency_closure():
    """只沿默认依赖解析，锁定 Speed 已安装版本并验证上游约束。"""
    environment = {**default_environment(), "extra": ""}
    pending = list(ROOTS)
    versions, edges = {}, {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        dist = metadata.distribution(name)
        versions[name] = dist.version
        requirements = []
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate(environment):
                continue
            installed = metadata.version(req.name)
            if req.specifier and installed not in req.specifier:
                raise RuntimeError(f"Speed dependency conflict: {name}: {line}, installed={installed}")
            requirements.append(line)
            pending.append(req.name)
        edges[name] = requirements
    return versions, edges


def run(command, log):
    """保存实际命令与退出码；失败保留原始输出并停止。"""
    with log.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"command": command, "cwd": str(Path.cwd()), "start": time.time()}) + "\n")
        output.flush()
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        output.write(json.dumps({"returncode": completed.returncode, "end": time.time()}) + "\n")
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}), see {log}")


def main():
    """为独立 venv 准备可离线重建的 wheel 闭包和 SHA256 锁。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wheels = args.output / "wheels"
    wheels.mkdir()
    versions, edges = dependency_closure()
    (args.output / "baseline-closure.json").write_text(json.dumps({
        "baseline_python": sys.executable, "roots": ROOTS, "versions": versions,
        "requirements": edges, "torch_source": str(TORCH_WHEEL),
    }, indent=2), encoding="utf-8")
    requirements = args.output / "requirements.pinned.txt"
    requirements.write_text("\n".join(
        str(TORCH_WHEEL) if name == "torch" else f"{name}=={version}"
        for name, version in sorted(versions.items())
    ) + "\n", encoding="utf-8")
    print(f"Resolved {len(versions)} installed packages; downloading wheels", flush=True)
    run([sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
         "--only-binary=:all:", "--no-deps", "--index-url", "https://pypi.org/simple",
         "--cache-dir", "C:/workspace/cache/pip", "-r", str(requirements),
         "--dest", str(wheels)], args.output / "download.log")
    inventory = []
    found = set()
    for wheel in sorted(wheels.glob("*.whl")):
        name, version, _, _ = parse_wheel_filename(wheel.name)
        if name not in versions or str(version) != versions[name] or name in found:
            raise RuntimeError(f"Unexpected or duplicate wheel: {wheel.name}")
        found.add(name)
        inventory.append({"name": name, "version": str(version), "file": wheel.name,
                          "bytes": wheel.stat().st_size, "sha256": digest(wheel)})
    if found != set(versions):
        raise RuntimeError(f"Missing wheels: {set(versions) - found}")
    (args.output / "wheel-inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (args.output / "env.lock").write_text("\n".join(
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}" for item in inventory
    ) + "\n", encoding="utf-8")
    print(f"Hash lock ready: {args.output / 'env.lock'}", flush=True)


if __name__ == "__main__":
    main()
