# 混合专家模型实测结果

本文汇总 Qwen3.6-35B-A3B 的训练验证、功能效果、资源和制品身份。

归档版本完成非专家与专家 rank 2 两种 100 步功能训练。当前高效注意力版本已通过合成模型闭环及后端验证；真实前向、首步、短训和压力结果单独列出。

## 测试条件

| 条件 | 值 |
|---|---|
| 显卡 | RTX 5070 Ti，16 GiB |
| 主机 | 64 GB 内存，原生 Windows |
| 依赖 | [哈希锁](../../requirements/windows-cu130.lock) |
| 已验证训练实现 | `9f7ff26` |
| 来源 | `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5` |

C3 表示非专家的注意力大投影与共享专家 LoRA；专家 rank 2 在其上增加路由专家参数。结构与参数布局见[专家设计](../design/moe.md)。

## 当前代码通过了哪些检查

| 检查 | 状态 | 关键数字 | 证据 |
|---|---|---|---|
| 单元测试及合成训练 | 通过 | 训练实现 72 项单测；合成模型 100 步同表达式梯度及最终参数逐位一致 | [训练闭环](../evidence/current/tiny-training.json) |
| 合成续训与发布重建 | 通过 | 新进程 50→100，参数、Adam、随机状态、数据游标和发布探针逐位一致 | 同上 |
| 结构整理阶段数值 | 通过 | `c7c9c83`，各步与工程基线相同，三类实际源码违规被拒绝 | [结构摘要](../evidence/current/structure.json) |
| 注意力误差与确定性 | 通过 | 四组真实头形状对 math 最大误差 0.371%，重复逐位一致 | [注意力摘要](../evidence/current/attention.json) |
| 完整模型实际后端 | 通过 | 128／2048／4096 token，有无 mask，全部前向、重算、反向均为 efficient | 同上 |
| CPU 前四层与三条贪心 | 未测 | 路由拆分判据及生成分别检查 | — |
| 首步及两层真实梯度 | 未测 | B=0 开关逐位相同，同表达式和独立参考分别检查 | — |
| 20 步功能、续训与重建 | 未测 | 第 10 步保存，新进程 10→20，第 20 步独立重建 | — |
| 2048／4096 训练压力 | 未测 | C3 与专家 rank 2，各两步，每步八条样本 | — |

当前代码不继承归档版本的训练结论；实测、推断和未测分别使用。

## 归档版本的功能训练

本节对应 tag `archive/moe-qwen36-35b-a3b-v1`（`efdfc55`），数值训练提交为 `24888c9`。它使用默认 SDPA，无 mask 路径与当前高效实现不同。

| 配置 | 留出，训练前→后 | [回放 NLL](../glossary.md)，训练前→后 | 精确续训 | 发布重建 |
|---|---|---|---|---|
| C3 rank 16，100 步 | 0/50→50/50 | 2.12 × 10⁻⁴→1.17 × 10⁻⁷ nat/token | 50→100 逐位一致 | 通过 |
| 增加专家 rank 2，100 步 | 0/50→50/50 | 2.12 × 10⁻⁴→9.93 × 10⁻⁴ nat/token | 50→100 逐位一致 | 通过 |

功能数字来自 [C3 评测](../evidence/moe/c3-functional-100.json)与[专家评测](../evidence/moe/expert-r2-functional-100.json)，续训及发布检查来自[最终训练审查](../evidence/moe/archive-training-checks.json)。通过的含义限定为这些训练项目；无条件 CPU 参考的差异见[参考判据](../guides/validation.md#专家模型的-cpu-参考怎样判定)。

语料组成及身份见[数据摘要](../evidence/moe/canary-data.json)。金丝雀门槛确认见[确认记录](../evidence/moe/threshold-decision.json)，含义见[训练评测指南](../guides/training.md#怎样评测适配器)。

## 优化前的同配置资源基线

下表来自结构整理提交 `c7c9c83`，用于和高效注意力版本对照。allocated 为活跃张量，reserved 为分配器保留量；整卡剩余是一秒采样最小值。

| 场景 | 平均步时 | 峰值 allocated | 峰值 reserved | 整卡最少剩余 |
|---|---|---|---|---|
| C3，20 步短功能样本 | 85.4 s | 6.3 GiB | 6.6 GiB | 7,998 MiB |
| C3，2048 token | 145.4 s | 7.9 GiB | 8.2 GiB | 6,674 MiB |
| C3，4096 token | 199.9 s | 10.8 GiB | 11.8 GiB | 3,042 MiB |
| 专家 rank 2，2048 token | 241.7 s | 10.3 GiB | 10.5 GiB | 4,162 MiB |
| 专家 rank 2，4096 token | 305.9 s | 13.2 GiB | 14.8 GiB | 119 MiB |

来源为[优化前基线](../evidence/current/training-before.json)，含锁页、主存与逐步数据。归档的完整 4096 压力检查分别见 [C3](../evidence/moe/c3-pressure-4096.json)与[专家 rank 2](../evidence/moe/expert-r2-pressure-4096.json)。高效注意力版本的对应训练资源与部件占比未测。

## 限制

- 功能语料证明固定关联的学习机制，没有领域效果评测。
- MoE 跨精度路由是离散选择，CPU 参考采用固定路由及独立路由检查。
- 训练不模拟部署 W8A8 激活量化，输入范围观测不等于实际推理误差。
- 一秒整卡采样可能漏掉瞬态，分配器峰值单独记录；CUDA 包含区间不能直接相加。
- 单 GPU、非思考纯文本之外的用法未测。

## 制品

当前高效注意力版本的真实发布制品未测。归档交付根 manifest 的 SHA256 为 `e6bf104787f588c6c2605e332287124a8755324242d0835b8a073a76ed29dcea`，读回结果见[制品摘要](../evidence/moe/release-artifacts.json)。

本机归档根为 `C:\workspace\artifacts\nvfp4lora\0.0.1\20261028T024826Z`。完整原始证据见 [tag 的证据目录](https://github.com/siltok-ai/nvfp4lora/tree/archive/moe-qwen36-35b-a3b-v1/docs/evidence)，保留的原始摘要未改内容。
