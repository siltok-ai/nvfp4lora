# 预处理格式与数值边界

`nvfp4-stream-v1` 由原 checkpoint 派生，原文件保留。`manifest.json` 绑定完整 revision、分片/配置/tokenizer/template 的 SHA256、数值策略、输出文件摘要、几何、E1 抽样和预处理峰值。写入新目录，不覆盖已有产物。

## 专家文件

每层一个 `experts/layer-NN.safetensors`，按原 expert ID 排列。

| 字段（gate_proj/up_proj/down_proj 各一组） | dtype | 形状 |
| --- | --- | --- |
| weight | uint8 原字节 | [E, OUT, IN/2] |
| weight_scale | uint8，重解释为 E4M3FN | [E, OUT, IN/16] |
| weight_scale_2 | float32 原值 | [E] |
| input_scale | float32 原值 | [E] |

gate/up 的 OUT=I、IN=H；down 的 OUT=H、IN=I。低 nibble 对应较小 K 下标。`experts/index.json` 保存各层 hash、文件头字节数、payload 字节数和每个张量的 payload 相对偏移。源中的 input_scale 保留为校准信息，当前 W4A16 训练路径不据此量化激活。

真实 H=2048、I=512、E=256 时，每层预计 452990976 字节、40 层 18119639040 字节，最终须由实际 header 验证。冻结专家不重新量化、不预先导出全量 BF16。

## 非专家文件

`nonexpert-*.safetensors` 按层/组件分文件，`model.safetensors.index.json` 提供参数名映射。删除来源文本前缀中的 `language_model.`；仅保留模型参数。配置取 `text_config`，去掉量化配置并关闭训练缓存。

策略 `nonexpert-fp32-decode-bf16-v1`：FP8 先转 FP32 再乘原始标量；共享专家和 lm_head 按 NVFP4 原式以 FP32 解码；最后舍入 BF16。lm_head 按行处理，限制解码临时空间。该策略只承诺跨精度 E3，不代表 FreeToken 的 W8A8 激活量化或 FP16 global 与训练完全 E2 等价。

## 验证与边界

真实 CLI 要求完整 G0.8 收据，并再次验证全部来源文件摘要。格式夹具测试覆盖 E4M3 全部有限位型、所有 nibble、低位次序、E2M1 中点舍入、零矩阵、分专家 global、文件头/索引错配、改动后的源文件及单分片收据拒绝。

`tools/check_preprocessed.py` 从磁盘读回每层 8 个专家的四类字段，对照原分片的原始字节；可在现有 Speed 环境以 FreeToken 的原 FP32 global 输入做 ≤1 ULP 的真实权重对照。它也比较非专家参数目录与 HF meta 模型；这部分不能替代物化模型和 RoPE buffer 的 A1.4 验收。

NVFP4 数值格式的原始参考：[NVIDIA Model Optimizer 0.44.0](https://github.com/NVIDIA/Model-Optimizer/blob/0.44.0/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)。本实现的标量位型参考与流式代码独立编写。`quantize` 仅用于 tiny 格式夹具及指定的格式自检，不用于转换真实专家基座。
