# 密集模型实测结果

本文汇总 Qwen3.8-27B 与 Qwen3.6-27B 的训练验证、功能效果、资源和制品身份。

当前版本以 Qwen3.8-27B 完成真实参考、20 步功能训练、精确续训及发布重建；Qwen3.6-27B 的真实结果仍限于归档版本。

## 测试条件

| 条件 | 值 |
|---|---|
| 显卡与主机 | RTX 5070 Ti，16 GiB 显存；64 GB 内存；原生 Windows |
| 依赖 | [Windows CUDA 锁文件](../../requirements/windows-cu130.lock) |
| 当前训练代码 | `b383ce8`；之后的结果整理只修改文档与证据 |
| 常规注意力 | 有无 padding mask，调整前后均为 efficient |
| Qwen3.8-27B 来源 | `RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30` |
| Qwen3.6-27B 来源 | `nvidia/Qwen3.6-27B-NVFP4@0893e1606ff3d5f97a441f405d5fc541a6bdf404` |

训练使用 checkpoint 自己的分词器及模板。全部层内量化投影使用 LoRA rank 16，配置名 `D-ALL`；词嵌入、输出头与其他小投影冻结，见[密集模型设计](../design/dense.md)。常规注意力与构建器在完整后端观测提交 `9f7ff26` 和当前提交间字节相同，当前提交另重跑真实几何及训练调用观测。

## 当前代码通过了哪些检查

逐位一致表示原始字节相同。同表达式参考采用相同顺序与精度，由普通 autograd 求梯度；独立 HF+PEFT 参考使用单独的误差判据。

| 检查 | 状态 | 关键数字 | 证据 |
|---|---|---|---|
| 单元测试与合成训练 | 通过 | 76 项单测；两族合成模型各 100 步，同表达式 loss、梯度及最终参数逐位相同 | [训练闭环](../evidence/current/tiny-training.json) |
| 合成续训与重建 | 通过 | 新进程 50→100，完整状态、发布探针和增量指纹逐位相同 | 同上 |
| 结构整理阶段数值 | 通过 | `c7c9c83`：MoE 74 个、dense 50 个参数在中间及最终检查点与工程基线逐位相同 | [结构摘要](../evidence/current/structure.json) |
| 注意力误差与确定性 | 通过 | 真实头形状对 math 最大误差 0.371%，重复逐位一致 | [注意力摘要](../evidence/current/attention.json) |
| 完整模型实际后端 | 通过 | 128／2048／4096 token，有无 mask，前向、重算和反向均为 efficient | 同上 |
| 真实前向与贪心 | 通过 | CPU FP32 前四层无条件最大误差 0.407%；三条回答正确且正常终止 | [前向摘要](../evidence/current/dense-forward.json) |
| 首步及真实激活梯度 | 通过 | B=0 loss 逐位相同；两层 13 类投影、八次累积的 A/B 同表达式逐位相同，独立 FP32 参考通过 | [梯度摘要](../evidence/current/dense-gradients.json) |
| 20 步、续训与发布 | 通过 | 留出 50/50，10→20 和独立发布重建逐位相同 | [短训摘要](../evidence/current/dense-training-20.json) |
| 长序列压力 | 通过 | 2048／4096 token，各两步、每步八条满长样本 | [压力摘要](../evidence/current/dense-pressure.json) |
| Qwen3.6-27B 当前真实检查 | 未测 | 归档结果单列 | — |

真实输入梯度在需要梯度的完整模型边上检查；首层冻结输入没有该边时，另用孤立回放检查，不能称为每个投影均捕获八次完整模型输入梯度。

## 当前版本的功能训练

20 步训练达到留出与回放的既定门槛，发布后的独立评测复现功能结果。[回放 NLL](../glossary.md)按监督 token 平均，使用自然对数。回放损失下降。

| 项目 | 训练前→第 20 步 | 门槛 |
|---|---|---|
| 留出正确数 | 0/50→50/50 | 至少 45/50 |
| 回放 NLL | 0.000234008→2.88837e-05 nat/token | 增量不超过 0.01 |
| 回放 NLL 增量 | -0.000205125 nat/token | 通过 |

第 10 步检查点由新进程继续到第 20 步，参数、Adam、调度器、随机数、数据游标和逐步数值均逐位一致。独立重建不读取训练检查点，发布参数、探针与 400 个权重增量指纹相同；开关适配器的 logits 确有变化。详见[短训、恢复及发布摘要](../evidence/current/dense-training-20.json)。

## 归档版本的功能训练

下表对应 tag `archive/dense27b-v1`（`61d8831`）。Qwen3.8 的实际训练提交为 `3fa6205`，Qwen3.6 为 `ede44f1`；两者使用展开 KV 的高效注意力。

| 模型与训练 | 留出，训练前→后 | 回放 NLL，训练前→后 | 续训与发布 |
|---|---|---|---|
| Qwen3.8-27B，100 步 | 0/50→50/50 | 2.34 × 10⁻⁴→3.35 × 10⁻⁷ nat/token | 50→100 及重建通过 |
| Qwen3.6-27B，10 步 | 0/50→50/50 | 1.85 × 10⁻⁴→6.26 × 10⁻⁶ nat/token | 续训未测，重建通过 |

数值来自 [Qwen3.8 功能评测](../evidence/dense/qwen38-functional-100.json)、[完整训练摘要](../evidence/dense/qwen38-training-summary.json)及 [Qwen3.6 短训](../evidence/dense/qwen36-short-training.json)。金丝雀用于检查新固定关联，回放使用该基座自己的生成结果。门槛及确认见[确认记录](../evidence/dense/threshold-decision.json)。

归档 Qwen3.8 的 100 步平均耗时 50.8 s，Qwen3.6 的 10 步平均 48.8 s；这些数值不替代当前同配置 20 步对照。归档长序列结果见[完整压力摘要](https://github.com/siltok-ai/nvfp4lora/blob/archive/dense27b-v1/docs/evidence/dense/d6-preflight-summary.json)。

## 显存、内存和速度

dense 的注意力可执行函数体没有改变，20 步的 loss、梯度范数、样本、学习率及随机探针均逐位一致。相同配置、语料、seed 与执行顺序的对照显示，20 步平均耗时为 58.2→45.0 s（-22.6%）。前者使用结构整理提交 `c7c9c83`，后者使用 `b383ce8`；完整步时含梯度累积、裁剪、更新和审核，不含评测、保存和发布。

下表为优化前→后。allocated 是活跃张量，reserved 是分配器保留量，两列单位 GiB；整卡剩余为一秒采样最小值。功能语料使用短样本，压力测试每个长度真实更新两步、每步八条满长样本。

| 场景 | 平均步时 | allocated | reserved | 整卡最少剩余 |
|---|---|---|---|---|
| 20 步功能样本 | 58.2→45.0 s | 5.05→5.05 | 5.49→5.49 | 7,422→7,557 MiB |
| 全部投影 rank 16，2048 | 128.0→126.3 s | 8.39→8.39 | 8.83→8.83 | 6,136→6,144 MiB |
| 全部投影 rank 16，4096 | 280.3→281.2 s | 12.10→12.10 | 12.48→12.48 | 2,469→2,410 MiB |

来源见[优化前基线](../evidence/current/training-before.json)、[当前短训](../evidence/current/dense-training-20.json)及[当前压力](../evidence/current/dense-pressure.json)。这些是单卡有限样本观测，系统显存背景和频率未固定；dense 的步时变化不归因于注意力；GPU 频率、主机调度和预热影响未隔离。

层原字节占 16,840,133,760 字节，实际锁页分配 16.00 GiB。上述当前运行中，进程工作集观测峰值为 21.76 GiB，私有提交峰值为 37.71 GiB，系统可用内存最低为 29.37 GiB。CPU 词嵌入表另占 2.37 GiB，不包含在层锁页中。不同主存口径不能相加，1 GiB = 2³⁰ 字节。

## 哪些效率问题只做了测量

第 20 步恢复后的一组八条短样本共 276 个 token，其中 34 个受监督；它不更新优化器，并与无计时控制核对全部梯度逐位相同。计时 wall 为 63.51 s，控制为 44.69 s；这不是额外的训练步，部件占比不能外推到 4096 长度。

| 区间 | 测得时间 |
|---|---|
| 层字节 H2D（主存至显存） | 5.330 s |
| NVFP4 解码 | 26.762 s |
| FP8 转换 | 1.641 s |
| 显式矩阵乘（aten 算子） | 3.321 s |
| [GDN](../glossary.md) 前向与反向包含区间 | 19.592 s |
| GDN 检查点重算（嵌套区间） | 6.939 s |
| 常规注意力 SDPA 前向与反向 | 0.040 s |
| CPU 词嵌入输入同步 | 0.317 ms |

GDN 与 NVFP4 解码分别占该组 CUDA 总包含区间的 30.9% 和 42.2%；CPU 词嵌入同步占观测 wall 的 0.00050%。解码按显式数据种类区分，aten 是 PyTorch 的算子层，融合注意力内部的矩阵乘不单独展开。包含区间有嵌套、重算与 CPU 入队空隙，H2D 可以重叠，不能相加或当成纯 kernel 时间。

本次没有替换 GDN、解码或词嵌入路径。优先级与验证要求见[路线](../../ROADMAP.md)，原始计时口径和高效后端实际调用数见[短训摘要](../evidence/current/dense-training-20.json)。

## 限制

- 金丝雀证明学习机制，没有领域效果评测；本轮 Qwen3.8 实际训练 20 步，未重跑真实 100 步。
- Qwen3.6-27B 的 10 步归档结果不代表当前实现的真实验证。
- 训练不模拟部署 FP8 激活量化；输入范围检查不能证明所有推理输入无截断。
- 一秒整卡采样可能漏掉瞬态；单 GPU、非思考纯文本之外未测。

## 制品

当前交付包含第 20 步发布、第 10/20 步完整恢复状态、固定语料、配置及源码归档，全部逐文件 SHA256 读回一致。发布格式保持不变；FreeToken 两个 fork 分支的实际 CPU 读取器通过[格式检查](../evidence/current/format-readers.json)，本轮未重新运行原生 GPU 推理。

- 当前 `adapter_model.safetensors` SHA256：`493c053a3c10409ea7a62ac6ad30288343f20e73081df5ecc9798f3c7ef40184`。
- 交付根 manifest SHA256：`daa66400361d27e6b1adfe0db934a44dc7ecb72f28f159e2486f32b70aef7750`。
- 制品文件清单和范围见[当前交付摘要](../evidence/current/release-artifacts.json)；归档制品身份见[归档摘要](../evidence/dense/release-artifacts.json)。

本机当前交付根为 `C:\workspace\artifacts\nvfp4lora\0.0.1\20261030T135610Z-multi-family`，使用 `dense/publish` 或 `dense/checkpoints`。完整旧材料保存在 [归档 tag](https://github.com/siltok-ai/nvfp4lora/tree/archive/dense27b-v1/docs/evidence)，保留摘要未改内容。
