# 工具目录

本文列出流程、验证和维护工具的用途与适用模型族。

用户日常操作从流程 CLI 开始；验证工具按[验证指南](../docs/guides/validation.md)选取。真实模型运行使用记录器，每次使用新目录，完整记录放在仓外。

## 流程入口

| 脚本 | 用途 | 模型族 |
|---|---|---|
| `fetch_checkpoint.py` | 下载固定 revision 并核对可信来源哈希 | 两族 |
| `preprocess_checkpoint.py` | 识别来源结构并生成 bundle | 两族 |
| `train_lora.py` | 训练、优化器步恢复、评测和发布 | 两族 |
| `verify_training_release.py` | 新进程重建发布产物并对照训练探针 | 两族 |
| `evaluate_training_adapter.py` | 独立评测发布适配器及功能门槛 | 两族 |
| `prepare_canary_replay.py` | 用本模型基座生成金丝雀和回放语料 | 两族 |
| `audit_fp8_input_ranges.py` | 观测冻结 FP8 投影的实际输入范围 | 两族 |
| `prepare_tiny.py` | 生成合成来源、量化夹具和 tiny bundle | 两族 |

## 验证工具

| 脚本，相对 `tools/validation/` | 用途 | 模型族 |
|---|---|---|
| `check_attention.py` | 真实头几何的 math 对照、确定性与原生后端检查 | 两族 |
| `check_template.py` | 与 Speed 模板逐 token 对照 | 两族 |
| `check_freetoken_dequant.py` | 用 FreeToken 解码器对照冻结码值 | 两族 |
| `check_grouped_mm.py` | 分组矩阵乘与独立梯度对照 | 通用 |
| `bench_h2d.py` | 测量锁页池分配和层大小搬运 | 通用 |
| `compare_tiny_baseline.py` | 比较结构改动的 tiny 数值及参数原字节 | 两族 |
| `compare_training_runs.py` | 比较 tiny 检查点和发布状态 | 两族 |
| `compare_real_resume.py` | 比较真实逐步数值和完整恢复状态 | 两族 |
| `profile_training.py` | 恢复后测一组累积，并以无计时梯度控制核对 | 两族 |
| `summarize_training_session.py` | 汇总训练状态、参数、资源与族审核结果 | 两族 |
| `moe/check_environment.py` | CUDA、meta 结构和普通 PEFT 环境资格 | MoE |
| `moe/check_quantizer_layout.py` | 核对来源量化字段和布局 | MoE |
| `moe/check_preprocessed.py` | 比较原始来源与 bundle 张量 | MoE |
| `moe/check_real_store.py` | 验证真实层存储的随机读取 | MoE |
| `moe/check_async.py` | 验证异步槽竞争、错误预取和全字节读取 | MoE |
| `moe/check_experts.py` | 单层真实专家的输出、梯度、重算及工作区 | MoE |
| `moe/check_gdn.py` | 纯函数 GDN 对独立递推参考 | MoE |
| `moe/check_gdn_module.py` | GDN 模块及检查点重算对照 | MoE |
| `moe/check_model_build.py` | 完整构建与冻结参数读回 | MoE |
| `moe/check_routed_reference.py` | 固定路由 CPU 参考及独立路由判据 | MoE |
| `moe/check_real_forward.py` | 完整前向和三条贪心回答 | MoE |
| `moe/check_real_training.py` | 首步与真实两层普通 LoRA 梯度 | MoE |
| `moe/check_real_expert_gradients.py` | 两层真实激活的专家向量雅可比积 | MoE |
| `moe/check_tiny_training.py` | 合成模型完整训练、续训和发布闭环 | MoE |
| `dense/check_dense_preprocessed.py` | 全部流式字节与来源对照 | dense |
| `dense/check_dense_linear.py` | 冻结线性输出、输入梯度及 LoRA 参考 | dense |
| `dense/check_dense_streaming.py` | 真实层随机读取和异步竞争 | dense |
| `dense/check_dense_forward.py` | CPU 前四层参考、贪心和激活观测 | dense |
| `dense/check_dense_training.py` | 首步及真实激活冻结层／LoRA 双参考 | dense |
| `dense/check_tiny_dense_training.py` | 双参考训练、续训和标准 PEFT 重建 | dense |

## 通用与维护

| 脚本 | 用途 | 范围 |
|---|---|---|
| `run_recorded.py` | 保存实际命令、源码、环境、日志及退出码 | 全部运行 |
| `check_docs.py` | 检查链接、锚点、有效证据和写作规模 | 文档 |
| `audit_snapshot.py` | 分析已保存的 CUDA 分配快照 | 两族 |
| `prepare_sequence_stress.py` | 生成可复现满长训练样本 | 两族按各自分词器处理 |
| `prepare_environment.py` | 从 Speed 实装版本导出闭包与锁文件 | 环境维护，非通用安装器 |

每个工具的准确参数以 `--help` 为准。计时、格式读取和资格探针的结果分别使用，不替代完整真实训练结论。
