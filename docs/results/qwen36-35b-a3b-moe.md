# 混合专家模型实测结果

本文汇总 Qwen3.6-35B-A3B 的训练验证、功能效果、资源和制品身份。

当前高效注意力版本完成真实参考、20 步功能训练、精确续训和发布重建。归档版本的两种 100 步结果单独列出。

## 测试条件

| 条件 | 值 |
|---|---|
| 显卡与主机 | RTX 5070 Ti，16 GiB 显存；64 GB 内存；原生 Windows |
| 依赖 | [哈希锁](../../requirements/windows-cu130.lock) |
| 当前训练代码 | `b383ce8`；之后的结果整理只修改文档与证据 |
| 常规注意力 | 无 mask：math→efficient；有 padding mask：保持 efficient |
| 来源 | `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5` |

C3 表示非专家的注意力大投影与共享专家 LoRA；专家 rank 2 在其上增加路由专家参数，见[专家设计](../design/moe.md)。常规注意力与两个构建器在完整后端观测提交 `9f7ff26` 和当前训练提交间字节相同，当前提交另重跑真实几何及训练调用观测。

## 当前代码通过了哪些检查

| 检查 | 状态 | 关键数字 | 证据 |
|---|---|---|---|
| 单元测试与合成训练 | 通过 | 76 项单测；两族合成模型各 100 步，同表达式 loss、梯度及最终参数逐位相同 | [训练闭环](../evidence/current/tiny-training.json) |
| 合成续训与重建 | 通过 | 新进程 50→100，完整状态、发布探针和增量指纹逐位相同 | 同上 |
| 结构整理阶段数值 | 通过 | `c7c9c83`：MoE 74 个、dense 50 个参数在中间及最终检查点与工程基线逐位相同 | [结构摘要](../evidence/current/structure.json) |
| 注意力误差与确定性 | 通过 | 真实头形状对 math 最大误差 0.371%，重复逐位一致 | [注意力摘要](../evidence/current/attention.json) |
| 完整模型实际后端 | 通过 | 128／2048／4096 token，有无 mask，前向、重算和反向均为 efficient | 同上 |
| CPU 前四层与贪心 | 通过 | 三个样本，固定路由最大误差 0.574%；三条回答正确并正常终止 | [前向摘要](../evidence/current/moe-forward.json) |
| 首步与真实梯度 | 通过 | C3、专家 rank 2 的 B=0 loss 逐位相同；26 项普通投影 A/B 同表达式累积梯度逐位相同；另查两层专家独立公式及输入梯度 | [梯度摘要](../evidence/current/moe-gradients.json) |
| 20 步、续训与发布 | 通过 | 留出 50/50，10→20 与独立重建逐位相同 | [短训摘要](../evidence/current/moe-training-20.json) |
| 长序列压力 | 通过 | C3 与专家 rank 2，各 2048／4096 两步×八条样本 | [压力摘要](../evidence/current/moe-pressure.json) |

CPU 参考使用[路由拆分判据](../guides/validation.md#专家模型的-cpu-参考怎样判定)，无条件自由路由的 1% 检查仍未通过，两者分别报告。专家两层独立公式的输入梯度相对误差最大 0.407%，输出、路由权重与六组 A/B 的相对误差为零；普通算子的同表达式逐位检查另行通过。

## 当前版本的功能训练

20 步训练达到留出与回放的既定门槛，发布后的独立评测复现功能结果。[回放 NLL](../glossary.md)按监督 token 平均，使用自然对数。回放损失增加，增量低于 0.01 nat/token 上限。

| 项目 | 训练前→第 20 步 | 门槛 |
|---|---|---|
| 留出正确数 | 0/50→50/50 | 至少 45/50 |
| 回放 NLL | 0.000211944→0.000754353 nat/token | 增量不超过 0.01 |
| 回放 NLL 增量 | +0.000542409 nat/token | 通过 |

第 10 步检查点由新进程继续到第 20 步，参数、Adam、调度器、随机数、数据游标和逐步数值均逐位一致。独立重建不读取训练检查点，发布参数、探针与 250 个权重增量指纹相同；开关适配器的 logits 确有变化。详见[短训、恢复及发布摘要](../evidence/current/moe-training-20.json)。

## 归档版本的功能训练

下表对应 tag `archive/moe-qwen36-35b-a3b-v1`（`efdfc55`），数值训练提交为 `24888c9`，使用默认 SDPA。它不替代当前版本的检查。

| 配置 | 留出，训练前→后 | 回放 NLL，训练前→后 | 续训与发布 |
|---|---|---|---|
| C3 rank 16，100 步 | 0/50→50/50 | 2.12 × 10⁻⁴→1.17 × 10⁻⁷ nat/token | 50→100 及重建通过 |
| 增加专家 rank 2，100 步 | 0/50→50/50 | 2.12 × 10⁻⁴→9.93 × 10⁻⁴ nat/token | 50→100 及重建通过 |

功能数值来自 [C3 评测](../evidence/moe/c3-functional-100.json)及[专家评测](../evidence/moe/expert-r2-functional-100.json)，恢复与发布见[训练审查](../evidence/moe/archive-training-checks.json)。语料和门槛分别见[数据身份](../evidence/moe/canary-data.json)与[确认记录](../evidence/moe/threshold-decision.json)。归档完整 4096 压力检查分别见 [C3](../evidence/moe/c3-pressure-4096.json)及[专家 rank 2](../evidence/moe/expert-r2-pressure-4096.json)。

## 显存、内存和速度

相同配置、语料、seed 与执行顺序的对照显示，20 步平均耗时为 85.4→83.4 s（-2.4%）。前者使用结构整理提交 `c7c9c83`，后者使用 `b383ce8`；完整步时含梯度累积、裁剪、更新和审核，不含评测、保存和发布。

下表为优化前→后。allocated 是活跃张量，reserved 是分配器保留量，两列单位 GiB；整卡剩余为一秒采样最小值。功能语料使用短样本，压力测试每个长度真实更新两步、每步八条满长样本。

| 场景 | 平均步时 | allocated | reserved | 整卡最少剩余 |
|---|---|---|---|---|
| 20 步功能样本 | 85.4→83.4 s | 6.32→6.32 | 6.61→6.61 | 7,998→8,336 MiB |
| C3，2048 | 145.4→143.3 s | 7.92→7.92 | 8.19→8.11 | 6,674→6,889 MiB |
| C3，4096 | 199.9→197.7 s | 10.81→9.55 | 11.79→9.77 | 3,042→5,103 MiB |
| 专家 rank 2，2048 | 241.7→221.3 s | 10.29→10.29 | 10.53→10.57 | 4,162→4,360 MiB |
| 专家 rank 2，4096 | 305.9→278.4 s | 13.16→12.00 | 14.75→12.23 | 119→2,662 MiB |

来源见[优化前基线](../evidence/current/training-before.json)、[当前短训](../evidence/current/moe-training-20.json)及[当前压力](../evidence/current/moe-pressure.json)。这些是单卡有限样本观测，系统显存背景和频率未固定；短训步时变化不能全部归因于注意力。

层原字节占 18,119,639,040 字节，实际锁页分配 20.00 GiB。上述当前运行中，进程工作集观测峰值为 24.77 GiB，私有提交峰值为 38.85 GiB，系统可用内存最低为 27.34 GiB。不同主存口径不能相加，1 GiB = 2³⁰ 字节。

## 哪些效率问题只做了测量

第 20 步恢复后的一组八条短样本共 276 个 token，其中 34 个受监督；它不更新优化器，并与无计时控制核对全部梯度逐位相同。计时 wall 为 116.73 s，控制为 82.36 s；这不是额外的训练步，部件占比不能外推到 4096 长度。

| 区间 | 测得时间 |
|---|---|
| 层字节 H2D（主存至显存） | 5.606 s |
| NVFP4 解码 | 52.501 s |
| FP8 转换 | 0.000 s |
| 显式矩阵乘（aten 算子） | 9.310 s |
| [GDN](../glossary.md) 前向与反向包含区间 | 11.212 s |
| GDN 检查点重算（嵌套区间） | 4.179 s |
| 常规注意力 SDPA 前向与反向 | 0.025 s |
| CPU 词嵌入输入同步 | 不适用，词嵌入常驻 GPU |

GDN 与 NVFP4 解码分别占该组 CUDA 总包含区间的 9.6% 和 45.1%。MoE 词嵌入常驻 GPU，没有 CPU 词嵌入输入同步路径。解码按显式数据种类区分，aten 是 PyTorch 的算子层，融合注意力内部的矩阵乘不单独展开。包含区间有嵌套、重算与 CPU 入队空隙，H2D 可以重叠，不能相加或当成纯 kernel 时间。

本次没有替换 GDN、解码或词嵌入路径。优先级与验证要求见[路线](../../ROADMAP.md)，原始计时口径和高效后端实际调用数见[短训摘要](../evidence/current/moe-training-20.json)。

## 限制

- 金丝雀只检查固定关联的学习机制，没有领域质量评测；当前实现实际训练 20 步，未重跑真实 100 步。
- MoE 的 CPU 路由拆分与无条件参考分别报告；训练不模拟部署 W8A8 激活量化。
- 一秒整卡采样可能漏掉瞬态；显存余量只适用于实测机器和工作负载。
- 单 GPU、非思考纯文本之外的用法未测。

## 制品

当前交付包含第 20 步发布、第 10/20 步完整恢复状态、固定语料、配置及源码归档，全部逐文件 SHA256 读回一致。发布格式保持不变；FreeToken 两个 fork 分支的实际 CPU 读取器通过[格式检查](../evidence/current/format-readers.json)，本轮未重新运行原生 GPU 推理。

- 当前 `adapter_model.safetensors` SHA256：`d12172ef6bee1d562377602f1f5a87dec1af201764d4975a60795d500aa57ca4`。
- 交付根 manifest SHA256：`daa66400361d27e6b1adfe0db934a44dc7ecb72f28f159e2486f32b70aef7750`。
- 制品文件清单和范围见[当前交付摘要](../evidence/current/release-artifacts.json)；归档制品身份见[归档摘要](../evidence/moe/release-artifacts.json)。

本机当前交付根为 `C:\workspace\artifacts\nvfp4lora\0.0.1\20261030T135610Z-multi-family`，使用 `moe/publish` 或 `moe/checkpoints`。完整旧材料保存在 [归档 tag](https://github.com/siltok-ai/nvfp4lora/tree/archive/moe-qwen36-35b-a3b-v1/docs/evidence)，保留摘要未改内容。
