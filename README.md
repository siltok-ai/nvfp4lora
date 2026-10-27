# nvfp4lora

探索在原生 Windows、RTX 5070 Ti 16 GB 与 64 GB 主存上，使用部署用的 Qwen3.6-35B-A3B NVFP4 checkpoint 训练 LoRA。

## 设计与交接原文

- [技术方案 v2](docs/reference/design-v2.md)
- [v2 实施交接说明与关键风险](docs/reference/handoff-v2.txt)
- [原文来源与 SHA256](docs/reference/sources.json)

两份文件按桌面原文的字节保存，不改写原文。交接材料中“只做分析交接”描述的是此前会话；本次用户已授权在独立仓库继续探索和实施。

## 当前阶段

M1 的32个必需关卡已通过。真实40层前向正常，P5的FP32参考按已记录的离散路由拆分判据验证；真实C3和全部专家r2均完成首个累积8的优化器步，两层C3梯度与独立解析参考一致，发布后新进程重载E2。长序列、多步训练/效果和真实断点续训仍待验证。

- [当前逐项状态与证据](docs/progress.md)
- [环境安装与探针命令](docs/environment.md)
- [预处理格式与数值边界](docs/format.md)
- [专家反向算子与模型构建](docs/expert-op.md)
- [训练、检查点与发布](docs/training.md)
- [异步搬运与 M1 验收](docs/streaming.md)
- [实施决策](docs/decisions.md)

计划采用独立核心库和精简训练入口：CPU 保存原始打包专家，GPU 按层搬运、按块解码，反向重新解码；非专家 LoRA 使用 PEFT，专家 LoRA 由核心库管理。

模型权重、venv、运行日志和大型产物保存在仓库外。
