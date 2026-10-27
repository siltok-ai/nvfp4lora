# nvfp4lora

探索在原生 Windows、RTX 5070 Ti 16 GB 与 64 GB 主存上，使用部署用的 Qwen3.6-35B-A3B NVFP4 checkpoint 训练 LoRA。

## 设计与交接原文

- [技术方案 v2](docs/reference/design-v2.md)
- [v2 实施交接说明与关键风险](docs/reference/handoff-v2.txt)
- [原文来源与 SHA256](docs/reference/sources.json)

两份文件按桌面原文的字节保存，不改写原文。交接材料中“只做分析交接”描述的是此前会话；本次用户已授权在独立仓库继续探索和实施。

## 当前阶段

先归档方案与交接，再深入核对源码、环境和风险，收敛执行目标并按关卡推进。本仓库尚无已验收的完整模型训练能力。

计划采用独立核心库和精简训练入口：CPU 保存原始打包专家，GPU 按层搬运、按块解码，反向重新解码；非专家 LoRA 使用 PEFT，专家 LoRA 由核心库管理。

模型权重、venv、运行日志和大型产物保存在仓库外。
