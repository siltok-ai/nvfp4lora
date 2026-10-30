# 密集模型实测结果

本文汇总 Qwen3.8-27B 与 Qwen3.6-27B 的训练正确性、功能效果、资源消耗和制品身份。

归档版本的 Qwen3.8-27B 完成 100 步功能训练、精确续训和发布重建，Qwen3.6-27B 完成 10 步功能训练和发布重建。当前版本的验证单独列出，未测项目不沿用归档结论。

## 测试条件

| 条件 | 值 |
|---|---|
| 显卡 | RTX 5070 Ti，16 GiB 显存 |
| 主机 | 64 GB 内存，原生 Windows |
| 依赖 | [Windows CUDA 锁文件](../../requirements/windows-cu130.lock) |
| 当前训练代码 | `9f7ff26` |
| Qwen3.8-27B 来源 | `RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30` |
| Qwen3.6-27B 来源 | `nvidia/Qwen3.6-27B-NVFP4@0893e1606ff3d5f97a441f405d5fc541a6bdf404` |

bundle 是保存原始量化字节、配置和哈希的训练权重包，NVFP4 的含义见[权重布局](../architecture.md#内存中分别放什么)。所有真实训练均使用该模型自己的分词器和聊天模板。

## 当前代码通过了哪些检查

逐位一致表示张量原始字节相同。同表达式参考采用相同运算顺序和精度，由普通 autograd 求梯度；独立 HF+PEFT 参考另按误差判据检查。结构整理阶段的比对绑定表中提交，其余通过项使用上面的当前训练代码。

| 检查 | 状态 | 关键数字 | 证据 |
|---|---|---|---|
| 单元测试与共享层边界 | 通过 | 72 项单测；三种实际源码违规均被拒绝 | [训练闭环摘要](../evidence/current/tiny-training.json)、[结构检查摘要](../evidence/current/structure.json) |
| 结构整理阶段的数值一致性 | 通过 | `c7c9c83`：dense 50 个、MoE 74 个参数，中间与最终检查点逐位相同 | [结构检查摘要](../evidence/current/structure.json) |
| 合成模型对参考实现训练 | 通过 | 100 步，同表达式 loss、全部梯度及最终参数逐位相同；保留独立参考 | [训练闭环摘要](../evidence/current/tiny-training.json) |
| 合成模型续训与发布重建 | 通过 | 新进程从第 50 步继续到第 100 步，状态及发布探针逐位一致；冻结权重不变 | 同上 |
| 真实形状注意力对照 | 通过 | dense 四组输出与输入梯度对 math 的最大误差 0.371%，重复运行逐位一致；完整模型各长度及掩码均使用高效后端 | [注意力摘要](../evidence/current/attention.json) |
| 真实模型前向 | 未测 | CPU fp32 前 4 层无条件误差不超过 1%，另查三条贪心回答 | — |
| 真实激活梯度 | 未测 | 两层投影的同表达式逐位对照与 fp32 参考 | — |
| 功能短训、续训与重建 | 未测 | 20 步；第 10 步保存，新进程继续到第 20 步 | — |
| 长序列训练 | 未测 | 2048／4096 token，各 2 步，每步累积 8 条样本 | — |
| Qwen3.6-27B 当前版本真实检查 | 未测 | 归档结果单列 | — |

## 归档版本的功能训练结果

本节证据保存在 tag `archive/dense27b-v1`（`61d8831`）。Qwen3.8 的 100 步运行使用提交 `3fa6205`，Qwen3.6 的 10 步运行使用 `ede44f1`；两者均采用显式展开 KV 的高效注意力。

金丝雀是刻意加入训练集的固定关联，用于检查 adapter 是否学到新信息；留出集使用不同问法。回放集由该基座生成，用于检查这次训练是否破坏已知输出；回放损失是按监督 token 平均的负对数似然，单位 nat 使用自然对数。

| 模型与训练 | 留出集，训练前 → 后 | 回放损失，训练前 → 后 | 续训 | 发布重建 |
|---|---|---|---|---|
| Qwen3.8-27B，100 步 | 0/50 → 50/50 | 2.34 × 10⁻⁴ → 3.35 × 10⁻⁷ nat/token | 第 50 → 100 步逐位一致 | 参数、探针与 400 个权重增量指纹逐位一致 |
| Qwen3.6-27B，10 步 | 0/50 → 50/50 | 1.85 × 10⁻⁴ → 6.26 × 10⁻⁶ nat/token | 未测 | 通过 |

数据来自 [Qwen3.8 训练摘要](../evidence/dense/qwen38-functional-100.json)、[续训与重建摘要](../evidence/dense/qwen38-training-summary.json)和 [Qwen3.6 检查摘要](../evidence/dense/qwen36-short-training.json)。功能门槛为留出至少 45/50、回放损失增加不超过 0.01 nat/token，见[门槛确认](../evidence/dense/threshold-decision.json)。

训练范围为全部层内量化投影，配置名 `D-ALL`，LoRA rank 为 16；词嵌入、输出头和其他小投影冻结。序列上限为 2048 token，每步累积 8 条样本。短功能样本没有填满序列上限，不能用其步时代表完整长序列吞吐。

## 显存、内存和速度

归档结果的资源按实际场景列出。`allocated` 是 PyTorch 活跃张量占用，`reserved` 是分配器保留显存；整卡剩余由独立的一秒采样给出。

| 场景 | 平均步时 | 峰值 allocated | 峰值 reserved | 整卡最少剩余 |
|---|---|---|---|---|
| Qwen3.8，100 步功能样本 | 50.8 s | 5.2 GiB | 7.5 GiB | 7.1 GiB |
| Qwen3.6，10 步功能样本 | 48.8 s | 5.2 GiB | 7.5 GiB | 7.1 GiB |
| Qwen3.8，2048 token 压力测试 | 128.7 s | 8.4 GiB | 8.8 GiB | 5.8 GiB |
| Qwen3.8，4096 token 压力测试 | 284.4 s | 12.1 GiB | 12.5 GiB | 2.2 GiB |

来源为上述训练摘要和[归档压力测试摘要](https://github.com/siltok-ai/nvfp4lora/blob/archive/dense27b-v1/docs/evidence/dense/d6-preflight-summary.json)。1 GB 为 10⁹ 字节，1 GiB 为 2³⁰ 字节。

原始层字节合计 16.84 GB，实际锁页分配为 16.0 GiB；CPU 词嵌入表另占 2.37 GiB。Qwen3.8 的工作集（驻留物理内存）观测峰值为 21.2 GiB，进程私有提交内存为 30.1 GiB，两种内存口径不能相加。来源见[归档层存储摘要](https://github.com/siltok-ai/nvfp4lora/blob/archive/dense27b-v1/docs/evidence/dense/d0-summary.json)与 Qwen3.8 训练摘要。

注意力优化前的同配置基线使用 `c7c9c83`：20 步平均 58.2 s，2048／4096 token 压力测试分别为 128.0/280.3 s。来源见[优化前基线](../evidence/current/training-before.json)。当前注意力版本对应的真实训练和资源数据未测。

## 限制

- 功能语料只验证单个固定关联与算术回放，没有领域效果评测。
- Qwen3.6-27B 的归档结果为 10 步，不代表完成 100 步验证。
- 训练不模拟部署推理中的 FP8 激活量化；FP8 输入范围观测不能证明所有输入都无截断。
- 核心训练和推理集成分别验证；单 GPU、非思考纯文本之外的用法未测。
- 一秒整卡采样可能漏掉更短的瞬态；分配器峰值单独记录。

## 制品

当前版本的发布产物状态为未测。归档制品通过独立逐文件 SHA256 读回，见[制品摘要](../evidence/dense/release-artifacts.json)。

| 归档制品 | `adapter_model.safetensors` 的 SHA256 |
|---|---|
| Qwen3.8-27B，第 100 步 | `206ea21afe34cecb4cfd9d87e62fd1ff003b9a1c8a035de8a87f9db7943258d2` |
| Qwen3.6-27B，第 10 步 | `552dc8992fa1ffe924e3160c2d123f69fab2b7ee232f6058457b8ac36cdbdf85` |

本机归档根目录为 `C:\workspace\artifacts\nvfp4lora-dense\0.0.1\20261030T054636Z`。完整原始证据见 [tag `archive/dense27b-v1` 的证据目录](https://github.com/siltok-ai/nvfp4lora/tree/archive/dense27b-v1/docs/evidence)，摘要内容来自该提交，保留原字节。
