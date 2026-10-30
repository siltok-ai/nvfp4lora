# 训练的数值策略

本文说明各部分的计算精度、缩放位置和验证承诺。

缩放位置是模型函数的一部分。训练直接读取部署权重，分别执行下表的显式策略，不将不同舍入顺序视为逐位等价。

| 部分 | MoE | dense |
|---|---|---|
| 流式冻结线性层 | 专家 BF16 码值矩阵乘，在 FP32 输出端乘全局缩放 | NVFP4／FP8 码值转 BF16，矩阵乘后在 FP32 输出端乘缩放，再转 BF16 |
| 常驻冻结张量 | 非专家量化张量在 FP32 解码和乘缩放后转 BF16 | 输出头和小张量按常驻策略加载；词嵌入在 CPU |
| 普通 LoRA | FP32 旁路加到基座输出后，再舍入回输出 dtype | 同左，使用自带包装 |
| 专家 LoRA | FP32 增量参与 gate/up 激活及 down 路由加权 | 不适用 |
| 常规注意力 | 显式展开 KV，再走 SDPA 高效后端 | 同左 |
| 线性注意力 | Transformers 纯 Torch 分块 GDN | 同左 |
| 交叉熵 | 仅监督位置分块生成词表分数，再按监督 token 总数归一 | 同左 |
| 优化器 | AdamW，权重衰减零，非 fused、非 foreach | 同左 |

## 冻结线性层怎样计算

NVFP4 的 E2M1 码值乘 E4M3 块缩放后可由 BF16 精确表示；矩阵全局缩放仍保留 FP32。dense 的 FP8 码值本身可由 BF16 精确表示，矩阵级缩放同样在输出端施加。

设码值矩阵为 `Wc`、全局缩放为 `s`，dense 的前向为 `BF16(FP32(BF16_GEMM(x,Wc))*s)`。反向先求 `BF16(FP32(dy)*s)`，再乘 `Wc`，不是把最终输入梯度统一乘缩放。实现见 [`PackedLinearFn`](../../src/nvfp4lora/ops/packed_linear_fn.py)。

MoE 非专家和 dense 常驻输出头采用不同策略：先在 FP32 解码并乘缩放，再转 BF16，见[共用解码](../../src/nvfp4lora/format/bundle.py)和 [FP8 解码](../../src/nvfp4lora/format/fp8.py)。不能把这两条路径互换来节省工作区而沿用数值结论。

## 适配器和路由怎样累积

普通 LoRA 计算 `scale * B(A(x.float()))`，初始 B 为零；只有适配器参数参与优化。B=0 开关 loss 必须逐位相同，见[普通包装](../../src/nvfp4lora/model/packed_linear.py)。

专家的 gate、up、SwiGLU、down 增量和路由加权使用 FP32 累积。每条路由先写唯一位置，再按原 top-k 顺序求和；输入梯度也按该顺序以 FP32 汇总，见[专家算子](../../src/nvfp4lora/ops/experts_fn.py)。

FP64 仅用于小形状梯度检查，不能据此推断真实 BF16 路径与高精度参考逐位相同。

## 注意力为什么先展开头

锁定 Windows 构建的确定性 GQA 无 mask 路径会使用 math；先把 key/value 头展开为 query 头数，可使用 efficient。项目注册自己的入口，不替换 Transformers 默认 callable，见[共用注意力](../../src/nvfp4lora/model/attention.py)。

高效实现对 math 的输出与 q/k/v 梯度相对 L2 必须小于 1%，同一配置重复运行逐位一致。实际原生前向、重算、反向的计数见[注意力实测](../evidence/current/attention.json)。

GDN 使用 Transformers 自带纯 Torch 分块实现；其独立 CPU 参考按 token 递推。更换 GDN 实现需要单独验证并遵守依赖决定，本版本没有引入 FLA。

## 损失和优化器怎样归一

损失采用 causal shift，只计算最后答案及终止符的监督位置。一次更新累积所有样本的交叉熵总和，除以本组监督 token 总数，不再除以累积样本数。实现见[分块交叉熵](../../src/nvfp4lora/ops/chunked_ce.py)和[训练器](../../src/nvfp4lora/train/loop.py)。

裁剪使用全体可训练参数，学习率先线性预热到基础学习率，之后保持常数。AdamW 的 beta 和 epsilon 使用锁定 PyTorch 默认值，权重衰减为零；某个专家本步未命中，已有动量仍可能更新其参数。

## 确定性承诺到哪里

`configure_training` 在首次 CUDA 矩阵乘前设置 CUBLAS 工作区，启用确定性算法，关闭 TF32、BF16 降精度归约和 cuDNN benchmark。相同实现、硬件、配置和数据流中的恢复与重建使用逐位判据。

E1 证明来源原字节，E2 证明同表达式或同版本状态，E3 证明独立参考的误差满足约定。更换注意力可改变训练轨迹；跨实现不要求逐位相同，但必须重新完成[验证指南](../guides/validation.md)中的检查。

## 与部署推理有什么边界

训练不模拟 FreeToken 的 FP8 W8A8 激活量化。输入范围观察器能报告激活相对于原缩放的范围，不能证明部署误差或领域质量，相关限制集中在[结果文档](../results/qwen3x-27b-dense.md#限制)。
