# nvfp4lora

探索在原生 Windows、RTX 5070 Ti 16 GB 与 64 GB 主存上，使用部署用的 Qwen3.6-35B-A3B NVFP4 checkpoint 训练 LoRA。

> **下一阶段改进：[性能与演进路线图](ROADMAP.md)**。优先补齐真实训练端到端 profiler、专家计算与重算优化、实际推理后端质量闭环；同时整理训练范围配置、分级存储与成熟训练前端集成。这是后续计划，尚未实施。

## 设计与交接原文

- [技术方案 v2](docs/reference/design-v2.md)
- [v2 实施交接说明与关键风险](docs/reference/handoff-v2.txt)
- [原文来源与 SHA256](docs/reference/sources.json)

两份文件按桌面原文的字节保存，不改写原文。交接材料中“只做分析交接”描述的是此前会话；本次用户已授权在独立仓库继续探索和实施。

## 当前阶段

本机G1实施和交付已完成：M1的32个必需关卡通过，最终严格M2为31 PASS、1 PASS_WITH_DOCUMENTED_CHANGE、0待办。P5原无条件CPU FP32 1%失败保留，按已记录的离散路由拆分判据验证；不能称作所有原始判据无条件通过。完整结果与限制见[实施结果](docs/results.md)。

C3与全部专家r2均完成100个优化器步、新进程50→100十项E2、GPU全部参数/固定probe/250或4090份ΔW发布重建及20/50/100检查点后验诊断。两套2048/4096两步压力和逐层重算检查通过。10/20/50/100步人工关联留出均50/50（基座0/50），最终回放NLL分别约1.17×10⁻⁷和0.0009931；专家终点高于基座，完整变化保留。[用户确认](docs/evidence/effect-threshold-owner-decision.json)的留出≥90%、回放NLL增量≤0.01门槛均满足，批准时序如实记录。这些是固定语料的功能验收，领域收益G3另行。

本地制品：`C:/workspace/artifacts/nvfp4lora/0.0.1/20261028T024826Z`，含两套publish、完整20/50/100检查点、固定数据及复现信息。全部清单文件hash及严格审核通过，操作入口见[真实训练CLI](docs/real-training-cli.md)。Speed源码和原环境未改，Speed内嵌A′与FreeToken带LoRA推理G2属于后续范围。

按原FP8 input_scale观测，禁用适配器的训练后端有5个模块超出原尺度，两套启用后均有4个，未新增越界模块；最大比值2.514。该结果记录W8A8接入风险，不能代替Speed/FreeToken部署一致性验证。专家4096一次整卡观测仅余528 MiB，不保证所有语料或后台负载均有足够余量。

- [当前逐项状态与证据](docs/progress.md)
- [环境安装与探针命令](docs/environment.md)
- [预处理格式与数值边界](docs/format.md)
- [专家反向算子与模型构建](docs/expert-op.md)
- [训练、检查点与发布](docs/training.md)
- [真实训练 CLI 与续训命令](docs/real-training-cli.md)
- [异步搬运与 M1 验收](docs/streaming.md)
- [实施决策](docs/decisions.md)

实现采用独立核心库和精简训练入口：CPU 保存原始打包专家，GPU 按层搬运、按块解码，反向重新解码；非专家 LoRA 使用 PEFT，专家 LoRA 由核心库管理。

模型权重、venv、运行日志和大型产物保存在仓库外。


<!-- dense27b-entry -->
## Dense 27B 特性分支

Qwen3.8-27B完成NVFP4基座100步、精确续训和发布重建；Qwen3.6-27B完成设计的10步缩短验收。两个adapter均通过独立FreeToken原生Windows完整模型推理；MoE位级回归通过。详细范围、原始失败及有记录调整见[实测结果](docs/dense/results.md)、[训练脚本](docs/dense/training-cli.md)和[进展/证据](docs/dense/progress.md)。原MoE结论保持不变。
