# nvfp4lora

探索在原生 Windows、RTX 5070 Ti 16 GB 与 64 GB 主存上，使用部署用的 Qwen3.6-35B-A3B NVFP4 checkpoint 训练 LoRA。

## 设计与交接原文

- [技术方案 v2](docs/reference/design-v2.md)
- [v2 实施交接说明与关键风险](docs/reference/handoff-v2.txt)
- [原文来源与 SHA256](docs/reference/sources.json)

两份文件按桌面原文的字节保存，不改写原文。交接材料中“只做分析交接”描述的是此前会话；本次用户已授权在独立仓库继续探索和实施。

## 当前阶段

M1 的 32 个必需关卡已通过：真实权重格式/加载、目标形状专家前后向、异步事件竞争、HF tiny 的 100 步训练/发布/精确续训。同步与异步的 100 步训练结果逐位一致。下一阶段 P5～P8 验证真实完整模型；真实训练能力尚未验收。

- [当前逐项状态与证据](docs/progress.md)
- [环境安装与探针命令](docs/environment.md)
- [预处理格式与数值边界](docs/format.md)
- [专家反向算子与模型构建](docs/expert-op.md)
- [训练、检查点与发布](docs/training.md)
- [异步搬运与 M1 验收](docs/streaming.md)
- [实施决策](docs/decisions.md)

计划采用独立核心库和精简训练入口：CPU 保存原始打包专家，GPU 按层搬运、按块解码，反向重新解码；非专家 LoRA 使用 PEFT，专家 LoRA 由核心库管理。

模型权重、venv、运行日志和大型产物保存在仓库外。
