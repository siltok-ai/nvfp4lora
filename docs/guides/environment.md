# 配置运行环境

本文说明怎样建立独立、可重复的原生 Windows 训练环境。

每个 checkout 使用单独 venv，依赖严格匹配[哈希锁文件](../../requirements/windows-cu130.lock)。平台、模型和资源实测范围见[结果文档](../results/qwen3x-27b-dense.md#测试条件)。

## 安装需要什么

| 项目 | 要求或检查 |
|---|---|
| 系统 | 原生 64 位 Windows，不使用 WSL |
| Python | CPython 3.12，64 位 |
| GPU | 已验证 RTX 5070 Ti，CUDA 架构 SM120 |
| 驱动 | `nvidia-smi` 可正常识别 GPU，并能执行锁定 PyTorch 的 CUDA 运算 |
| 内存 | 已验证 64 GB；锁页层和进程私有内存另有占用，见结果文档 |
| 磁盘 | 同时容纳原 checkpoint、bundle、检查点和运行记录；不能只按量化权重大小预留 |
| wheelhouse | 含锁文件指定版本、平台和 SHA256 的全部 wheel |

驱动最低版本未单独验证。请以实际 CUDA 运算成功为准，不根据驱动面板显示的 CUDA 版本推断训练环境可用。

## 怎样建立独立环境

在仓库根目录运行以下 PowerShell 命令，将占位符换成自己的路径。

```powershell
py -3.12 -m venv '<venv>'
& '<venv>\Scripts\python.exe' -m pip install --no-index --find-links '<wheelhouse>' --require-hashes -r requirements/windows-cu130.lock
& '<venv>\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e .
```

安装应全部正常结束。缺 wheel 或哈希不符时，先补齐原版本的可信文件；不要放宽哈希检查。构建依赖 `setuptools` 也在锁文件中。

`tools/prepare_environment.py` 用于从已有 Speed 环境导出依赖闭包和 wheel，带有本机构建输入约定；它不是新机器的通用安装器。依赖对齐的原因见[依赖决定](../decisions.md#依赖使用哈希锁和独立环境2026-10-30用户)。

## 怎样检查安装和绑定

先检查依赖，再确认源码路径和 GPU 运算。

```powershell
& '<venv>\Scripts\python.exe' -m pip check
& '<venv>\Scripts\python.exe' -c "import nvfp4lora, torch; print(list(nvfp4lora.__path__)); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); x=torch.ones(32,32,device='cuda',dtype=torch.bfloat16); print((x@x)[0,0].item())"
```

正常输出包含 `No broken requirements found.`、当前 checkout 的源码路径、目标 GPU 和矩阵乘结果 `32.0`。不要让两个 checkout 共用同一 editable 绑定。

MoE 的额外环境资格工具是 [`check_environment.py`](../../tools/validation/moe/check_environment.py)。它需要 `--baseline-closure` 指向包含 `versions` 映射的依赖闭包，并检查 CUDA、meta 模型和普通 LoRA；它不证明 dense 或真实模型完整训练通过。

## 哪些运行需要记录

真实模型运行使用[记录器](../../tools/run_recorded.py)，每次新建输出目录。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<run>' -- tools/validation/check_attention.py --bundle '<moe-bundle>' --bundle '<dense-bundle>' --output '{run}/results'
```

正常结束时退出码为零，运行目录保留真实命令、源码、包版本、输出和退出码。原始记录与权重放在仓外；仓库只保留结果文档直接引用的小摘要。
