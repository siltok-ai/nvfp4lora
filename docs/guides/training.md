# 训练、续训和发布

本文说明怎样使用自带脚本训练 LoRA、精确续训并验证发布产物。

从已校验 bundle 和固定 JSONL 开始，每次训练使用新运行目录。最小流程是训练、优化器步检查点、独立进程重建；一个训练步的时序见[总体设计](../architecture.md#一个优化器步怎样执行)。

## 怎样开始一次训练

下面使用 dense 全部层内量化投影的 rank 16 配置，先执行到第 20 步。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<run>' -- tools/train_lora.py --bundle '<bundle>' --data '<train.jsonl>' --config configs/dense27b-all-r16.json --stop-at 20 --checkpoint-steps 10 20 --output '{run}/results' --code-commit '<commit>'
```

正常结束后 `summary.json` 为通过状态，目录包含以下文件。MoE 非专家 LoRA 使用 `configs/real-c3.json`；加路由专家 rank 2 使用 `configs/real-expert-r2.json`。

```text
results/
  train_config.json       绑定续训身份的逻辑配置
  execution.json          本次执行参数
  metrics.jsonl           每个优化器步的指标
  summary.json            完成状态与发布探针
  checkpoints/step-000010/ 完整恢复状态
  checkpoints/step-000020/
  publish/                独立加载的适配器
  delta-w.json            权重增量指纹
```

## 数据采用什么格式

每行是一段非思考纯文本对话，最后一条必须是 assistant 的完整答案，且前一条是 user。

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}
```

训练使用 bundle 自己的分词器与模板，只监督最后答案和终止符。超长提示从左截断；若不能完整保留答案及至少一个提示位置，则丢弃该样本并记录行号。实现见 [`render_sample`](../../src/nvfp4lora/train/template.py)。

## 配置中的哪些字段影响续训

逻辑配置绑定数据内容、分词器、模板、精度、构建和依赖版本；它们不能在恢复时静默改变。

| 字段 | 含义 | 默认值或约束 | 范围 |
|---|---|---|---|
| `family` | 显式族身份 | 必须提供并匹配 bundle | 两族 |
| `seed` | 初始化和随机流 | 20,261,028 | 两族 |
| `max_steps` | 调度器的逻辑训练长度 | 100 | 两族 |
| `max_length` | 编码序列上限 | 2048 | 两族 |
| `learning_rate` | AdamW 基础学习率 | 0.0002 | 两族 |
| `warmup_steps` | 线性预热长度 | 10 | 两族 |
| `max_grad_norm` | 全局梯度裁剪上限 | 1 | 两族 |
| `accumulation_steps` | 每步样本数 | 8 | 两族 |
| `ce_chunk_size` | 监督位置词表分数的分块大小 | 128 | 两族 |
| `gradient_checkpointing` | 按层激活重算 | `true` | 两族 |
| `sampler_seed` | 确定性采样顺序 | 113 | 两族 |
| `rng_probe` | 每步主动消耗并记录随机数 | `false` | 两族 |
| `build.streaming` | 层调度器 | `async` | 两族 |
| `build.expert_rank` | 路由专家 rank，零表示冻结 | 0 | MoE |
| `build.c3_rank`／`c3_alpha` | 非专家 LoRA 的 rank／alpha | 16／32 | MoE |
| `build.expert_alpha` | 专家 LoRA alpha | 非零 rank 时默认两倍 rank | MoE |
| `build.chunk_size` | 一次解码的专家数上限 | 32 | MoE |
| `build.lora_rank`／`lora_alpha` | 普通投影 LoRA rank／alpha | 16／默认两倍 rank | dense |
| `build.lora_scope` | 投影范围 | `D-ALL`，或 `D-ATTN` | dense |
| `build.decode_chunk_elements` | NVFP4 解码中间量分块 | 8,388,608 | dense |

准确字段和边界以[族配置实现](../../src/nvfp4lora/families)为准。优化器及精度规则见[数值策略](../design/numerics.md)。

## 哪些参数只影响本次执行

| 参数 | 行为 |
|---|---|
| `--stop-at` | 本次停在哪个全局步，不改变 `max_steps` 和学习率轨迹 |
| `--checkpoint-steps` | 哪些优化器步保存恢复状态 |
| `--evaluation-steps` | 哪些步运行回放和留出评测 |
| `--eval-corpus`／`--eval-batch-size` | 评测语料位置及批大小 |
| `--resume` | 从某个完整检查点恢复 |
| `--skip-delta-fingerprint` | 省去大规模权重增量指纹；不影响训练，但减少发布证据 |
| `--snapshot-first-micro` | 保存首个反向后的分配器快照，用于显存排查 |
| `--code-commit` | 记录实际代码版本，不代替逻辑配置校验 |

执行参数写入 `execution.json`。`--stop-at` 是绝对全局步；已经完成第 10 步时，`--stop-at 20` 再执行 10 步。

## 怎样精确续训

恢复时使用相同 bundle、数据和逻辑配置，指定新的输出目录。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<resume-run>' -- tools/train_lora.py --bundle '<bundle>' --data '<train.jsonl>' --config configs/dense27b-all-r16.json --resume '<run>\results\checkpoints\step-000010' --stop-at 20 --checkpoint-steps 20 --output '{run}/results' --code-commit '<commit>'
```

恢复应从第 11 步开始。只允许在优化器步边界保存；配置哈希、数据身份、可训练集合或完整文件校验失败时拒绝恢复。恢复的内容见[检查点格式](../design/adapters.md#训练检查点包含什么)。

`metrics.jsonl` 每行对应一次优化器更新。`loss` 按监督 token 总数归一，`sample_ids` 是本步样本顺序，`learning_rates` 是更新使用的学习率，`gradient_norm` 是裁剪前范数。步时、搬运及内存是观测字段；续训比较时将它们与数值状态分开。

## 怎样评测适配器

金丝雀和回放语料可用统一工具生成，生成过程依赖真实基座。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<prepare-run>' -- tools/prepare_canary_replay.py --bundle '<bundle>' --dataset-output '<corpus>' --output '{run}/results'
```

语料输出应带来源身份和文件哈希。训练时追加 `--eval-corpus '<corpus>' --evaluation-steps 20`，可得到适配器关闭时的初始评测和第 20 步评测。

留出集检查新关联的不同问法；回放损失是每个监督 token 的负对数似然。功能门槛、单位和适用范围见[实测结果](../results/qwen3x-27b-dense.md#归档版本的功能训练结果)。每个模型必须使用自己基座生成的回放答案。

## 怎样验证发布产物

新进程只读取 bundle 和发布目录，核对训练末尾探针。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<verify-run>' -- tools/verify_training_release.py --bundle '<bundle>' --training-results '<run>\results' --output '{run}/results'
```

全部检查应通过，包括参数集合、参数原字节、损失与权重增量指纹。发布文件不含优化器状态，不能替代训练检查点。

还可在独立进程中重测留出与回放，并显式执行功能门槛。

```powershell
& '<venv>\Scripts\python.exe' tools/run_recorded.py --output '<eval-run>' -- tools/evaluate_training_adapter.py --bundle '<bundle>' --adapter '<run>\results\publish' --corpus '<corpus>' --baseline-evaluation '<run>\results\evaluation-step000000-disabled.json' --minimum-canary-accuracy 0.9 --max-replay-nll-increase 0.01 --output '{run}/results'
```

评测应正常退出，适配器启用与关闭的探针应不同，同时达到所设门槛。

## 长序列怎样控制显存

4096 token 使用仓库对应的 `*-4096.json` 配置；不能以短功能样本的平均步时代表满长序列吞吐。资源数字分别见 [MoE](../results/qwen36-35b-a3b-moe.md)与 [dense](../results/qwen3x-27b-dense.md)。

显存不足时先记录 allocated、reserved、整卡剩余与活跃对象快照，确认是否发生全序列词表分数或权重滞留。改变数值策略、训练范围或 rank 属于新的训练实验，需要重新验证，不能直接沿用已有结论。
