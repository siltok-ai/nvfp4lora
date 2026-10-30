# 训练权重包的格式

本文说明 bundle 保存什么、怎样绑定来源，以及量化字段怎样还原为计算值。

流式层文件保存部署 checkpoint 的原始量化字节；非流式文件按已声明策略保存常驻张量。完整 manifest 是加载前提，来源和输出文件均有 SHA256，见[共用读取逻辑](../../src/nvfp4lora/format/bundle.py)。

## 两族目录怎样组织

| 项目 | MoE | dense |
|---|---|---|
| bundle 标识 | `nvfp4-stream-v1` | `nvfp4-stream-dense-v1` |
| 层目录 | `experts/layer-NN.safetensors` | `layers/layer-NN.safetensors` |
| 层内布局标识 | `packed-nvfp4-experts-v1` | `packed-dense-layers-v1` |
| 非流式张量 | 按层拆开的非专家文件及词嵌入、输出头 | 词嵌入、输出头及按层小张量文件 |
| 公共材料 | 配置、分词器、模板、生成配置、来源布局报告 | 同左 |
| 读取入口 | `family_of_bundle` 按布局解析 | 同左 |

实际文件名和哈希来自 manifest，不由使用者拼接猜测。两族目录写入逻辑分别见 [MoE 预处理](../../src/nvfp4lora/format/preprocess_moe.py)与 [dense 预处理](../../src/nvfp4lora/format/preprocess_dense.py)。

## manifest 绑定哪些内容

| 字段类别 | 内容 |
|---|---|
| 完整状态与版本 | `status`、`layout_version`；dense 另有显式 `family` |
| 来源 | 仓库、revision、三个原始分片及元数据的 SHA256 |
| 结构 | 层数、维度、投影种类及每层字节规模 |
| 数值策略 | 常驻张量与流式线性层的策略标识 |
| 输出身份 | 每个输出文件的 SHA256 和必要的张量布局 |
| 观测 | 预处理时间、内存峰值、原字节比对结果 |

模型函数的身份排除工作路径、时间和内存观测，但保留来源、几何、数值策略和文件内容。训练和发布通过这个身份绑定基座，见[发布格式](adapters.md)。

## 层文件的张量形状

令输入宽度为 `K`、输出宽度为 `N`，MoE 专家数为 `E`。dense 的每个字段以前面的投影名为前缀，例如 `mlp.gate_proj.weight`。

| 字段 | 原字节类型 | dense NVFP4 | MoE NVFP4 |
|---|---|---|---|
| `weight` | uint8 | `[N,K/2]` | `[E,N,K/2]` |
| `weight_scale` | E4M3FN 原字节 | `[N,K/16]` | `[E,N,K/16]` |
| `weight_scale_2` | FP32 | 每矩阵一个标量 | 每专家一个标量 |
| `input_scale` | FP32 | 保留来源缩放 | 保留来源缩放 |

dense 的注意力大投影使用 FP8：`weight` 为 `[N,K]` 的 E4M3FN 原字节，`weight_scale` 和 `input_scale` 为矩阵级缩放。运行时由显式 `kind` 选择格式，不通过字段缺失猜测。

## 一个字节怎样还原

每字节先取低 4 位、再取高 4 位，沿输入宽度方向展开。E2M1 的非负码值依次是 `0, 0.5, 1, 1.5, 2, 3, 4, 6`，高符号位表示负号，包括负零。

例如字节 `0x21, 0xB4` 对应码值 `0.5, 1, 2, -1.5`。若所在 16 元素块的缩放为 `2`、矩阵全局缩放为 `0.25`，四个最终值为 `0.25, 0.5, 1, -0.75`。

该例只展开一个块中的四个位置；真实块仍有 16 个元素。解码顺序见 [`dequantize`](../../src/nvfp4lora/format/nvfp4.py)，全局缩放放在矩阵乘哪一侧见[数值策略](numerics.md)。

## 层字节和锁页分配怎样预算

dense 的 48 个线性注意力层与 16 个常规注意力层分别布局；每层小于 256 MiB，整组流式字节约 16.84 GB。原字节可按层校验，不能按解码后 BF16 总规模估算层槽。

逻辑字节与锁页分配不同：分配器可能按更大的块取整。两族本机实测由[优化前基线](../evidence/current/training-before.json)记录逻辑字节、物理锁页字节和槽大小；GB 与 GiB 的定义见[dense 资源结果](../results/qwen3x-27b-dense.md#显存内存和速度)。

## 边界

只接受已知几何和来源字段。权重重排不引入新的量化误差；tiny 生成器的量化能力仅用于测试夹具，不用于真实 checkpoint。
