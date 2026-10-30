# 怎样验证改动

本文按改动类型说明需要运行的检查、判据和失败排查方向。

先跑单元测试，再用合成小模型完成训练闭环；涉及模型数值或资源时补上真实模型检查。命令使用当前工具路径，完整索引见[工具目录](../../tools/README.md)。

## 改了什么就检查什么

| 改动 | 最少检查 | 关注点 |
|---|---|---|
| 任意代码 | `python -m unittest discover -s tests -v` | 算子、身份拒绝、分层和状态测试全部通过 |
| 解码、算子或 LoRA | 两族 tiny 完整训练 | 普通 autograd 同表达式参考、精确续训、发布重建 |
| 构建、注意力或 GDN | 真实前向、注意力几何对照 | 跨精度误差、确定性、实际后端 |
| 训练循环、数据或检查点 | 首步、短训、续训及发布 | 梯度、数据顺序、Adam 状态、留出与回放 |
| 显存或搬运 | 2048／4096 压力及完整累积组计时 | allocated、reserved、整卡剩余、冻结权重生命周期 |
| 文档 | `tools/check_docs.py` | 链接、锚点、证据引用、写作规则和规模 |

本机单测约半分钟，完整真实训练与压力检查按小时安排；实际时间取决于长度和配置，见结果文档。所有真实模型命令外层使用[记录器](environment.md#哪些运行需要记录)。

## 怎样区分三类一致性

E1 比较冻结张量的原始字节，E2 比较同表达式计算和恢复状态的原始字节，E3 比较独立或跨精度参考的明确误差。三者定义见[术语表](../glossary.md)。

E2 包括形状、dtype、正负零和随机状态，不用近似相等替代。E3 为每类参考设定判据；不通过时检查来源、精度、缩放和路由，不能把误差预算直接调大。

## 怎样验证合成模型训练闭环

先生成 tiny bundle，再运行 100 步、50→100 新进程续训和独立发布重建。以下两条命令以 dense 为例。

```powershell
& '<venv>\Scripts\python.exe' tools/prepare_tiny.py --family dense --bf16 '<fixture>\bf16' --nvfp4 '<fixture>\nvfp4' --processed '<fixture>\bundle' --output '<fixture>\prepare'
& '<venv>\Scripts\python.exe' tools/validation/dense/check_tiny_dense_training.py --bundle '<fixture>\bundle' --steps 100 --mode all --streaming async --commit '<commit>' --output '<check>'
```

MoE 改用 `--family moe` 与 [`check_tiny_training.py`](../../tools/validation/moe/check_tiny_training.py)。三类子进程均须退出零；同表达式的每步 loss、梯度和最终参数逐位相同，B=0 时开关损失相同，冻结指纹不变。

dense 另有标准 HF+PEFT 独立参考；MoE 另有独立专家公式参考。它们保留跨精度误差检查，不能与逐位参考混为一项。

## 怎样检查注意力

对两个真实 bundle 的头几何运行有无 mask、2048／4096 的输出、q/k/v 梯度和确定性对照。

```powershell
& '<venv>\Scripts\python.exe' tools/validation/check_attention.py --bundle '<moe-bundle>' --bundle '<dense-bundle>' --output '<check>'
```

每组对 math 的相对 L2 必须小于 1%，重复结果逐位一致，原生算子计数确认 efficient 前向和反向。配置字符串本身不证明后端；完整模型的观测结果见[注意力摘要](../evidence/current/attention.json)。

## 密集模型的 CPU 参考怎样判定

dense 比较前四层独立 CPU FP32 参考，每层相对 L2 不超过 1%，不固定中间激活来掩盖误差。

```powershell
& '<venv>\Scripts\python.exe' tools/validation/dense/check_dense_forward.py --bundle '<bundle>' --mode reference --output '<check>'
```

另以 `--mode greedy` 检查三条短回答。前向参考和贪心回答分别报告，不能用生成通顺替代数值检查。

## 专家模型的 CPU 参考怎样判定

MoE 的 top-k 路由是离散选择，BF16 与 FP32 在边界附近可能选出不同专家。对不同函数的输出直接要求小误差，无法单独判断算子是否正确。

采用拆分检查：用 GPU 实际路由固定专家组合后，前四层相对 L2 不超过 1%；另以相同输入独立检查路由投影及所选专家的权重。实现与具体字段见[路由参考工具](../../tools/validation/moe/check_routed_reference.py)。

```powershell
& '<venv>\Scripts\python.exe' tools/validation/moe/check_routed_reference.py --bundle '<bundle>' --output '<reference>'
& '<venv>\Scripts\python.exe' tools/validation/moe/check_real_forward.py --bundle '<bundle>' --mode greedy --output '<greedy>'
```

路由拆分只用于独立 CPU 前向参考，不放宽同一版本的续训或发布逐位检查。归档版本的无条件 CPU 参考未通过，不能将固定路由结果描述成无条件通过。

## 怎样检查真实梯度与功能训练

[`check_real_training.py`](../../tools/validation/moe/check_real_training.py) 和 [`check_dense_training.py`](../../tools/validation/dense/check_dense_training.py) 分别执行首步、两层真实激活梯度和冻结指纹检查。同表达式普通 autograd 用逐位判据，独立 CPU 解析梯度用原定误差及余弦判据。

真实短训使用[训练指南](training.md)的流程，在第 10 步保存，由新进程继续到第 20 步；用 [`compare_real_resume.py`](../../tools/validation/compare_real_resume.py) 比较全部命名状态和逐步数值，再执行独立重建及功能门槛。

功能门槛是留出至少 45/50、回放 NLL 增量不超过 0.01 nat/token；它只验证给定金丝雀任务。确认记录见 [MoE](../evidence/moe/threshold-decision.json)和 [dense](../evidence/dense/threshold-decision.json)。

## 怎样排查资源与耗时

压力测试对每个长度真实更新两步，每步累积八条满长样本。独立采样整卡剩余，分配器峰值从训练指标读取。

[`profile_training.py`](../../tools/validation/profile_training.py) 恢复检查点后测量一组累积并与无计时控制比较，包含搬运、解码、矩阵乘、GDN、常规注意力和 CPU embedding 同步。它不更新优化器，不能代替完整步时。

CUDA 区间包含 CPU 入队间隙，搬运可与计算重叠，反向区间包含检查点重算；这些数字不能直接相加。模板检查使用 [`check_template.py`](../../tools/validation/check_template.py) 对照实际 Speed 源码，归档示例见[模板摘要](../evidence/moe/template-golden.json)。
