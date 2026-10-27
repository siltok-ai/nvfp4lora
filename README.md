# nvfp4lora

探索在原生 Windows、RTX 5070 Ti 16 GB 与 64 GB 主存上，使用部署用的 Qwen3.6-35B-A3B NVFP4 checkpoint 训练 LoRA。

## 设计与交接原文

- [技术方案 v2](docs/reference/design-v2.md)
- [v2 实施交接说明与关键风险](docs/reference/handoff-v2.txt)
- [原文来源与 SHA256](docs/reference/sources.json)

两份文件按桌面原文的字节保存，不改写原文。交接材料中“只做分析交接”描述的是此前会话；本次用户已授权在独立仓库继续探索和实施。

## 当前阶段

已归档方案和交接，正在推进独立环境资格及格式层。现有 GDN、256 组 GEMM、18.5GB 锁页和 PEFT 预检已通过；完整模型训练能力尚未验收。

- [当前逐项状态与证据](docs/progress.md)
- [环境安装与探针命令](docs/environment.md)
- [预处理格式与数值边界](docs/format.md)
- [实施决策](docs/decisions.md)

计划采用独立核心库和精简训练入口：CPU 保存原始打包专家，GPU 按层搬运、按块解码，反向重新解码；非专家 LoRA 使用 PEFT，专家 LoRA 由核心库管理。

模型权重、venv、运行日志和大型产物保存在仓库外。
