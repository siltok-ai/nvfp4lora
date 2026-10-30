# 适配器和恢复状态

本文说明训练检查点、两种发布格式和权重增量指纹的用途与兼容边界。

检查点用于继续优化，发布目录用于从同一 bundle 独立重建适配器。两者都绑定模型函数身份，但发布目录没有优化器、随机状态或数据游标。

## 训练检查点包含什么

| 文件 | 内容 |
|---|---|
| `lora_params.safetensors` | 按规范名保存的 FP32 可训练参数 |
| `optimizer.pt` | 按参数名保存的 Adam 状态和参数组 |
| `scheduler.pt` | 学习率调度器状态 |
| `rng.pt` | Python、NumPy、Torch CPU/CUDA 随机状态 |
| `data_state.json` | 数据身份、epoch、位置及消耗样本数 |
| `trainer_state.json` | 全局步、监督量及族审核状态 |
| `train_config.json` | 完整逻辑配置 |
| `checkpoint_manifest.json` | 完整状态、布局、配置哈希与逐文件哈希 |

只有优化器更新和清梯度完成后才能保存。加载前校验完整状态、文件哈希、逻辑配置和命名参数集合，再恢复 Adam、调度器与随机状态，见[检查点实现](../../src/nvfp4lora/train/checkpoint.py)。

恢复绑定同一版本的计算语义；格式可读不等于跨数值实现仍能精确续训。

## 两种发布格式怎样区分

| 内容 | MoE | dense |
|---|---|---|
| `layout_version` | `nvfp4-lora-v1` | `nvfp4-lora-dense-v1` |
| 普通线性适配器 | `adapter_model.safetensors` 与 `adapter_config.json` | 同左，标准 PEFT LoRA 配置 |
| 路由专家适配器 | `expert_lora.safetensors` | 不存在 |
| 完整 manifest | 来源、基座指纹、构建参数、缩放、参数规格、文件 SHA256 | 同左，额外显式 dense 族身份 |
| 加载器 | MoE 专用格式读取器 | dense 专用格式读取器 |

每个读取器先核对 layout，再核对基座身份和全部文件内容；两族互相拒绝。统一入口只是按[注册表](../../src/nvfp4lora/families/__init__.py)选择读取器，不转换发布格式。

普通 LoRA 的发布键以 `base_model.model.` 为前缀，使用规范层名和 `lora_A.weight`／`lora_B.weight`。专家 A/B 保留专家维度，布局见[专家设计](moe.md#适配器参数怎样组织)。

## 标准 PEFT 能加载到什么对象

dense 的适配器文件可由标准 PEFT 加载到匹配的普通浮点 HF 模型；本项目 tiny 独立参考实际验证这一路径。它不意味着 PEFT 能直接替代本项目的 NVFP4 流式构建和执行。

从发布目录重建时只需 bundle 和发布产物，不能依赖训练检查点中的参数补全；验证流程见[训练指南](../guides/training.md#怎样验证发布产物)。

## 权重增量指纹证明什么

ΔW 为 `scale * B @ A`，哈希覆盖实际张量原字节。专家按有限块计算，避免物化整个模型的权重增量，见[指纹实现](../../src/nvfp4lora/export/delta_w.py)。

参数逐位一致证明 A/B 已读回；ΔW 指纹进一步核对缩放和参数方向。探针 loss 还检查实际启用路径；三者共同使用，不能只比一个文件哈希就认定训练效果生效。

## 与推理仓库怎样对接

| FreeToken fork 分支 | 读取格式 |
|---|---|
| [`feat/qwen36-nvfp4-lora`](https://github.com/wenlive/FreeToken/tree/feat/qwen36-nvfp4-lora) | `nvfp4-lora-v1` |
| [`feat/dense27b-nvfp4-lora`](https://github.com/wenlive/FreeToken/tree/feat/dense27b-nvfp4-lora) | `nvfp4-lora-dense-v1` |

发布与 bundle 格式是外部接口，变更必须同步消费者并单独批准。训练端重建、格式读取和真实推理执行是不同检查，结果范围见[模型结果](../results/qwen3x-27b-dense.md#限制)。
