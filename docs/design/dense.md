# 密集模型

本文说明密集模型怎样使用流式冻结投影、CPU 词嵌入和普通线性 LoRA。

dense 构建器逐个替换层内量化投影，使用共用层存储与调度器；模型配置决定投影形状，已知几何表拒绝未支持结构。实现见[构建器](../../src/nvfp4lora/model/build_dense.py)。

## 为什么冻结投影不持有 Parameter

`PackedLinear` 只持有层号、投影名、格式种类和几何，权重原字节由外部 store 管理。它不注册冻结权重参数或 buffer，防止 `model.to()` 将完整量化基座或解码权重搬到 GPU。

[`PackedLinearFn`](../../src/nvfp4lora/ops/packed_linear_fn.py) 的五个顶层参数为 `x`、层号、投影名、`kind`、配置；上下文只保存非 Tensor 元信息，不为反向保存输入或解码权重。反向重新获取原字节、解码并计算输入梯度。

## LoRA 怎样包在基座旁边

`LoRAPackedLinear` 用 FP32 的 `lora_A.default.weight` 和 `lora_B.default.weight` 计算增量，加到冻结基座结果后转回基座输出 dtype。参数规范名去掉 `default`，发布时添加标准 PEFT 前缀。

这与标准 PEFT 命名兼容，但构建器直接使用本项目包装，不调用 PEFT 注入冻结量化投影。B 初始化严格为零，alpha 默认两倍 rank，见[模块实现](../../src/nvfp4lora/model/packed_linear.py)。

## 两种训练范围包含什么

| 范围 | 目标 | rank 16 的参数张量数 | 参数量 |
|---|---|---|---|
| `D-ALL` | 全部层内量化投影 | 800 | 108,789,760 |
| `D-ATTN` | 去掉 MLP gate/up/down 的注意力投影 | 416 | 39,583,744 |

两者均不训练词嵌入、输出头和其他小投影。几何计算由 `expected_dense_trainable` 生成，测试见[范围与构建检查](../../tests/test_dense_build.py)。

## 为什么词嵌入留在 CPU

词嵌入表不需要梯度，保存在 `CPUEmbedding.table` 普通 CPU 属性中，占用约 2.37 GiB 主存。`BuiltModel.frozen_extra` 引用实际表，冻结指纹显式覆盖它；模型参数和 buffer 迭代均不会遗漏地把它搬上 GPU。

每次前向将 token ID 拷到 CPU 查表，再将得到的隐藏状态搬到 GPU；`ids.cpu()` 会产生同步。这项代价由部件计时独立测量，本版本不改变输入传递方式。

## 输出头和层槽怎样占用显存

输出头以 BF16 常驻 GPU，分块交叉熵只为有效监督位置生成词表分数。GPU 层槽存放原始字节，当前投影计算时才生成浮点权重。

64 层流式字节约 16.84 GB，不能全放入该 16 GiB GPU。两层原字节槽、当前解码工作区、LoRA/梯度/Adam 状态和激活的实际资源见[结果文档](../results/qwen3x-27b-dense.md#显存内存和速度)。

## 边界

真实模型不允许构建全量物化参考；物化路径只为 tiny 测试开放。FP8 输入缩放保留来源原值，但训练不执行 W8A8 激活量化，具体差异见[数值策略](numerics.md#与部署推理有什么边界)。
