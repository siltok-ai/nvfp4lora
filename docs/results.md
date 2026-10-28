# 本机 G1 实施结果

2026-10-28，原生 Windows、RTX 5070 Ti（16303 MiB）及64 GB主存上的真实 Qwen3.6-35B-A3B NVFP4 LoRA 训练、发布、精确续训和本地交付已完成。M1全部32项必需检查通过；[最终严格M2](evidence/m2-final-strict-audit.json)为 **31 PASS、1 PASS_WITH_DOCUMENTED_CHANGE、0待办**。调整项是下文保留原失败的A5.2参考判据，不能称作原方案所有原始判据无条件通过。

交付独立核心库和精简训练CLI，遵循v2的A交付形态。Speed内嵌A′、FreeToken带LoRA推理G2和领域收益G3是后续范围。

## 实测与验收

| 项目 | C3 | C3＋全部路由专家r2 |
| --- | --- | --- |
| 配置 | 非专家r16、alpha32 | 同左；40层×256专家r2、alpha4 |
| FP32可训练参数 | 500张量／19,169,280元素 | 740张量／176,455,680元素 |
| 完整训练 | 100个optimizer step，累积8 | 同左、实际数据顺序相同 |
| 新进程50→100 | 十项完整状态及50步记录均E2 | 十项完整状态及50步记录均E2 |
| 新进程发布重建 | 全部参数、固定启用/禁用probe及250份ΔW E2 | 全部参数、固定启用/禁用probe及4090份ΔW E2 |
| 最终改写留出 | 50/50，基座0/50 | 50/50，基座0/50 |
| 最终回放NLL | 0.0000001170189461 | 0.0009931222000905 |
| 相比基座NLL变化 | −0.0002118272157452 | ＋0.0007811779653992 |
| 4096压力训练峰值allocated／reserved | 11,606,167,040／12,664,700,928字节 | 14,127,787,520／15,841,886,208字节 |

回放基座NLL为0.0002119442346913，300条、981个监督token。固定功能语料是300条人工关联训练、50条改写留出及300条实际基座生成的算术回放。[用户确认门槛](evidence/effect-threshold-owner-decision.json)为留出≥90%（45/50）、回放NLL增量≤0.01 nat/token，两套均满足；独立发布评测的完整生成token及回放记录与训练step100一致。用户在结果披露后确认了该工程门槛，批准时序如实保留，不作为预注册盲测或领域泛化证据。专家第20步回放回升、第50步回落、第100步再回升全部保留，没有选用中点替代终点。

V1证据包含真实目标形状专家前后向、路由权重及六个A/B梯度、FP64全坐标gradcheck、tiny100步独立稠密参考，以及两层真实激活VJP。V2比较完整参数、命名Adam、scheduler、四RNG、数据/trainer/config状态、每步九项数值记录、发布probe和ΔW。V3使用新进程仅从基座＋publish重建，比较启用/禁用行为及全部参数/增量。详细关卡与历史失败见[逐项进度](progress.md)。

两套2048/4096均实跑至少两个optimizer step×累积8，覆盖Adam首次分配与稳态；独立重算峰值控制检查及非零B解码生命周期通过。专家4096一次整卡观测剩余528 MiB，余量很小，不能推广为所有语料或后台负载下的保证。r4未试。快照事件环已满，不能声称完整事件轨迹；峰值来自独立CUDA计数及经过E2控制的逐层采集。

20/50/100检查点的A/B分组范数、50条留出NLL和固定8条样本的40层路由已补测：[C3](evidence/real-c3-checkpoint-diagnostics.json)、[专家](evidence/real-expert-checkpoint-diagnostics.json)。这些是保存参数的离线重建，排除padding并覆盖rank0；不是原训练时刻的probe采集或另一份续训试验。

## 制品与复现

本机目录：`C:/workspace/artifacts/nvfp4lora/0.0.1/20261028T024826Z`。101个受校验文件（另有根manifest.json），受校验文件共7,838,582,554字节，实际复制及最终逐文件hash核验通过。[清单](evidence/delivery-manifest.json)、[交付收据](evidence/delivery-verification-summary.json)。

- `c3/publish`、`expert-r2/publish`：标准非专家PEFT文件、自有专家safetensors及绑定基座的manifest。
- 两套`checkpoints/step-000020`、`step-000050`、`step-000100`：全部FP32参数、命名Adam、scheduler、四RNG、数据游标、trainer/config及完整收据。
- 固定语料、训练/独立评测、ΔW、路由诊断、FP8观测、环境与锁文件。

基座继续引用`C:/workspace/models/bundles/qwen3.6-nvfp4-lora/1355db6a052410cf-train-v2`，未复制全量权重或合并LoRA。解释器为`C:/workspace/envs/T-20261027-nvfp4-lora/train/Scripts/python.exe`。39个必要依赖精确对齐Speed，环境已登记为`verified_g1_with_documented_change`；[最终库存](evidence/environment-final-inventory.json)、[登记收据](evidence/environment-final-registration.json)。

数值训练提交为`24888c9983d51e1ea476f363d867d60765a94ce8`；制品源码提交为`80c31b9c091c0245f453c9faa265acdb1b27221a`，之后本轮收尾提交只更新文档及证据。复现的安装、生成、训练与从制品step50恢复命令见[真实训练CLI](real-training-cli.md)和[环境说明](environment.md)。数据按内容绑定，[实际重定位核查](evidence/delivery-data-portability.json)通过；不能静默修改逻辑max_steps100再宣称同一精确续训。

制品内README保持构建时的复验步骤；其“正式M2仍需执行”的后续步骤已由`20261028T024918Z-m2-final-strict-audit`完成。最终资格以上述独立严格审核为准，不修改已经核验的制品字节。

## 保留的调整与后续风险

原A5.2无条件CPU FP32前四层1%门槛失败，初条提示后三层约2.05%、1.77%、2.46%。top-k在cutoff处跨精度分叉；条件化到实际GPU离散路由后，另三条测量前已提交的新提示连续FP32误差最高0.580%，独立router投影/top-k/权重检查通过。原自由路由失败全部保留，依据和时序见[决策](decisions.md)。

两套启用adapter后均有4个FP8来源模块的输入最大值超出原input_scale范围，未新增越界模块，最大比值2.514；[C3](evidence/real-c3-functional-fp8.json)、[专家](evidence/real-expert-functional-fp8.json)。这是训练后端范围观测，没有运行实际W8A8推理，不能推断截断比例或部署质量。FreeToken teacher-forcing对照未执行。GDN采用已验证的现成Torch chunk回退，FLA未试。

[最终边界检查](evidence/final-workspace-boundary-audit.json)确认两份桌面原文、Speed/材料仓版本与干净状态、97个Speed锁定版本、39个必要依赖及原33个数值源码保持一致。原始NVFP4专家码值/缩放未重量化；训练用标准AdamW接受历史动量更新，未改为masked optimizer；Speed源码及原环境未改。
