# nvfp4lora

nvfp4lora 在消费级显卡上，直接用部署用的 NVFP4 checkpoint 训练 LoRA（低秩适配参数）。

项目提供核心库和训练脚本，运行于原生 Windows。需要流式搬运的冻结权重保留量化原字节，按层搬到 GPU，用时解码；训练只更新 LoRA 参数。架构见[总体设计](docs/architecture.md)，验证范围见[密集模型实测结果](docs/results/qwen3x-27b-dense.md)。

## 支持哪些模型

| 模型 | 结构 | checkpoint | LoRA 范围 | 验证硬件 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 混合专家（MoE） | [nvidia，revision `1355db6a0524`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/tree/1355db6a052410cfd62085d94b58866fd0f2c3c5) | 两类注意力投影、共享专家、路由专家 | RTX 5070 Ti 16 GiB，64 GB 内存 |
| Qwen3.8-27B | 密集（dense） | [RadixArk，revision `319f741cce68`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/tree/319f741cce68d7914884900c138a1fbb70a42f30) | 全部层内量化投影，或仅其中的注意力投影 | 同上 |
| Qwen3.6-27B | 密集 | [nvidia，revision `0893e1606ff3`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/tree/0893e1606ff3d5f97a441f405d5fc541a6bdf404) | 同上 | 同上 |

## 怎样开始训练

下面使用 Qwen3.8-27B；将尖括号占位符替换为实际路径，在仓库根目录运行 PowerShell。`<wheelhouse>` 是含[锁文件](requirements/windows-cu130.lock)全部 wheel 的目录，`<bundle>` 是预处理后的训练权重包。

1. 创建独立环境，按哈希锁安装依赖，再绑定当前 checkout。

   ```powershell
   py -3.12 -m venv '<venv>'
   & '<venv>\Scripts\python.exe' -m pip install --no-index --find-links '<wheelhouse>' --require-hashes -r requirements/windows-cu130.lock
   & '<venv>\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e .
   ```

   三条命令均应正常结束；每个 checkout 使用自己的 venv。

2. 下载固定 revision 并核对分片，再生成 bundle。

   ```powershell
   & '<venv>\Scripts\python.exe' tools/fetch_checkpoint.py --repo RadixArk/Qwen3.8-27B-NVFP4 --revision 319f741cce68d7914884900c138a1fbb70a42f30 --model-dir '<source>' --output '<fetch>' --download-weights
   & '<venv>\Scripts\python.exe' tools/preprocess_checkpoint.py --model-dir '<source>' --source-receipt '<fetch>\receipt.json' --output '<bundle>'
   ```

   下载收据和 bundle 的 `manifest.json` 均应标记完成，且三个分片的 SHA256 与上游一致。

3. 使用非思考纯文本 JSONL 数据训练，并记录命令和源码。

   ```powershell
   & '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<run>' -- tools/train_lora.py --bundle '<bundle>' --data '<train.jsonl>' --config configs/dense27b-all-r16.json --output '{run}/results' --code-commit '<commit>'
   ```

   正常结束后，`results` 中应包含训练摘要、优化器步检查点和 `publish` 目录。混合专家模型使用 `configs/real-c3.json`。

4. 在新进程中重建发布产物，核对参数、损失和权重增量指纹。

   ```powershell
   & '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<verify-run>' -- tools/verify_training_release.py --bundle '<bundle>' --training-results '<run>\results' --output '{run}/results'
   ```

   校验摘要中的全部检查应通过。

## 从哪里了解实现和结果

| 文档或入口 | 回答的问题 |
|---|---|
| [总体设计](docs/architecture.md) | 数据怎么流动，模块怎么分层，怎样接入模型族？ |
| [密集模型实测结果](docs/results/qwen3x-27b-dense.md) | 哪些检查通过，速度和资源是多少？ |
| [族注册表](src/nvfp4lora/families/__init__.py) | 如何解析 bundle、训练配置和 adapter 的模型族？ |
| [训练配置](configs) | 两族的训练范围和参数怎样设置？ |

完整归档结果见 [MoE 版本](https://github.com/siltok-ai/nvfp4lora/tree/archive/moe-qwen36-35b-a3b-v1/docs/evidence)和 [dense 版本](https://github.com/siltok-ai/nvfp4lora/tree/archive/dense27b-v1/docs/evidence)。推理集成由 [FreeToken](https://github.com/wenlive/FreeToken) 提供；训练验证的限制集中列在结果文档中。
