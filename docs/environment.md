# 独立环境与执行

本项目先验证原生 Windows、Python 3.12、SM120。依赖版本来自 Speed 的实际安装闭包；当前精确版本和每个 wheel 的 SHA256 见 `requirements/windows-cu130.lock`。不升级现有 Speed venv。

在新目录创建 venv 后，可使用以下入口安装。Torch 的 CUDA wheel 由官方 cu130 索引供应，其余包由 PyPI 供应；锁中的哈希约束实际下载的字节。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --require-hashes -r requirements/windows-cu130.lock --extra-index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python.exe -m pip check
```

本机环境为 `C:/workspace/envs/T-20261027-nvfp4-lora/train`。`tools/prepare_environment.py` 供这台机器从现有 Speed 元数据重新解析、下载相同闭包；保存了原始 wheel 的离线库存。`accelerate==1.15.0` 是已批准的 PEFT 传递依赖，训练循环不使用它。当前不安装 FLA，先使用实测通过的 Transformers 纯 Torch chunk GDN。

本机 `C:/workspace/envs/registry.json` 已以 `nvfp4-lora-win-cp312-cu130` 登记此路径和editable源码消费者，当前状态为 `preparing_real_g1_acceptance`：M1资格通过，真实G1完整验收仍在进行。依赖库存及pip check由独立run记录；正式验收结束后再更新复用状态。复用前读取该登记及主任务最新状态，不移动venv、不与其他editable checkout共用可变依赖。

探针从仓库根运行，`{run}` 由记录器替换为实际运行目录。每次选择一个不存在的新路径：

```powershell
python -B tools/run_recorded.py --output C:/workspace/runs/T-20261027-nvfp4-lora/EXAMPLE-gdn -- tools/check_gdn.py --target --output '{run}/results'
python -B tools/run_recorded.py --output C:/workspace/runs/T-20261027-nvfp4-lora/EXAMPLE-gdn-module -- tools/check_gdn_module.py --output '{run}/results'
python -B tools/run_recorded.py --output C:/workspace/runs/T-20261027-nvfp4-lora/EXAMPLE-h2d -- tools/bench_h2d.py --output '{run}/results'
python -B tools/run_recorded.py --output C:/workspace/runs/T-20261027-nvfp4-lora/EXAMPLE-grouped -- tools/check_grouped_mm.py --output '{run}/results'
python -B -m unittest discover -s tests -v
```

`EXAMPLE-*` 仅为命令示例，不代表已执行目录。实际命令、代码快照、环境、退出码和日志以每个 run 的 `manifest.json` 为准。资格探针的峰值和耗时不代表完整模型训练性能。

`tools/check_freetoken_dequant.py` 用已有 Speed 环境只读导入 FreeToken，不向训练 venv 注入整套推理依赖。运行记录明确使用该解释器。
