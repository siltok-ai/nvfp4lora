# 实施状态

总体目标：依据 v2 在本机实现并验证真实 Qwen3.6-35B-A3B NVFP4 LoRA 训练，交付训练代码、可重载 adapter 和续训证据。当前推进 M1，不缩减最终目标。

## P0 资格

| 关卡 | 状态 | 证据与边界 |
| --- | --- | --- |
| G0.1 | PASS | 独立 venv 的 39 个必要依赖与 Speed 实装版本一致；SM120、BF16 GEMM、输入梯度 |
| G0.2 | PASS | 18,500,000,000 字节锁页分配；三轮层大小 H2D 的 wall 带宽 47.08、50.28、50.27 GB/s；进程结束释放 |
| G0.3 | FALLBACK PASS | 现成纯 Torch chunk GDN：独立递推参考、q/k/v/g/beta 梯度、真实 T4096、字节级检查点对照通过；完整 GDN+PEFT 模块的两种检查点对照也通过 |
| G0.4 | PASS | 完整 meta 模型、真实专家接口和融合维度；固定来源 text_config 几何吻合 |
| G0.5 | PASS | 实际 HF 小模型的 PEFT 零初始化、冻结状态、梯度及一次 AdamW 更新 |
| G0.6 | PASS | 256 组，K2048/N512 和 K512/N2048；前向、dX、dW 逐专家 autograd 对照 |
| G0.7（可选） | PASS | 现有 FreeToken wrapper 的合成 512×2048 反量化；原 FP32 global 与 Torch 参考最大 0 ULP；不替代真实权重 A1.3 |
| G0.8 | PASS | 固定完整 revision 的三个分片全部下载完成，完整 SHA256 与上游一致；真实预处理前又重做校验 |

GPU：RTX 5070 Ti 16GB，主存约 61.6GiB。GDN 原始内核 allocated 峰值约 2.23GiB；完整单模块加 LoRA 和检查点约 2.5GiB。均为单算子/单模块测试，不是完整训练峰值。未试 FLA；选择现成 Torch 路径是已验证回退，不声称 FLA 失败。

## 格式层开发

已有 NVFP4/FP8 解码、tiny 格式量化、真实文件头与索引核对、流式预处理，以及相应格式夹具测试。真实来源的 CLI 要求三个分片完整校验收据；预处理时再次验证分片和配置/模板/分词器摘要。

专家输出保留 `weight`、`weight_scale` 的原始 uint8 字节、FP32 `weight_scale_2` 和 `input_scale`，不重数量化。非专家采用 FP32 解码再舍入 BF16 的显式策略；lm_head 按行限制解码临时空间。格式夹具测试不替代真实 checkpoint 的 A1 验收，更不替代 P4 的 HF tiny 训练。

最小同步 `ExpertStore`/`SyncLayerStreamer` 已实现：直接读取 CPU 原字节 payload、校验实际文件与索引、在两个层槽间做同步拷贝，保留 pin/generation 与计数。CPU 和 CUDA 的实际数据、占用失败、释放后句柄、校验和篡改检测已纳入 13 项测试并通过；尚未与专家 autograd 组合，也没有异步预取资格。`20261027T114214Z-sync-store-unit` 的首次失败来自“后续访问仍只有一次缓存命中”的错误测试预期；修正后证据为 `20261027T114256Z-sync-store-regression`，原始失败未覆盖。

## 真实 P1 格式验收

真实产物：`C:/workspace/models/bundles/qwen3.6-nvfp4-lora/1355db6a052410cf-train-v2`。可审计摘要见 [p1-format-summary.json](evidence/p1-format-summary.json)。

| 项目 | 状态 | 实际证据 |
| --- | --- | --- |
| A1.1 | PASS | 124468 个源张量逐头/索引核对；122880 个主干专家字段、130 个 FP8 非专家模块、121 个 NVFP4 非专家模块的类型/几何吻合 |
| A1.2 | PASS | 从磁盘重读 40 层×8 专家×3 投影×4 字段，共 3840 组原字节对照通过 |
| A1.3 | PASS | 960 个真实专家投影与未修改 FreeToken wrapper（原 FP32 global）对照，最大 1 BF16 ULP |
| A1.4 | PASS | 613 个非专家实际 GPU 参数逐一读回，与文件原字节一致；RoPE 两个 buffer 重建为 FP32 并与正常 CPU 构造相同；无 meta 残留；740 个可训练 Tensor 集合/形状/FP32/零 B 正确 |
| A1.5 | PASS | 预处理进程 peak working set 3246047232 字节，小于 16GB；专家 payload 18119639040 字节，非专家 BF16 payload 4896711936 字节 |

首次真实预处理因把 MTP 两个融合专家误计入文本主干集合而在布局阶段停止。已用小夹具复现并修复前缀过滤；失败 run `20261027T114512Z-real-preprocess` 和输出 `*-train-v1` 保留。修复后的预处理 `20261027T114706Z-real-preprocess-v2`、磁盘/FreeToken/meta 目录验收 `20261027T114846Z-real-format-validation` 均 PASS。

追加 A1.4 证据为 `20261027T123808Z-real-model-build`，摘要见 [a1-materialization-summary.json](evidence/a1-materialization-summary.json)。只执行构建和加载，没有执行完整真实模型前向或训练。P1 全部通过；M1 尚未通过，没有真实模型训练 adapter。

## P2 真实专家算子

[算子与构建说明](expert-op.md)、[P2 数值与显存摘要](evidence/p2-experts-summary.json)。真实 bank 的 40 层前后顺序全 payload 读回在 `20261027T120045Z-real-sync-bank` 通过；80 次 acquire、78 次真实 copy，冻结 CPU payload 哈希不变。

| 关卡 | 状态 | 证据 |
| --- | --- | --- |
| A2.1 | PASS | 真实一层、T4096/H2048/I512/E256/top8：FP32/BF16 前向与独立稠密参考一致 |
| A2.2 | PASS | 同形状 x/路由权重/六个 A/B 梯度；FP32 dX 相对 L2 7.3e-8，BF16 0.005084，其他比较为零 |
| A2.3 | PASS | H64/I32/E8/top2/r2 真 FP64 全坐标 gradcheck；非零 A/B；独立逐 token 数学参考 |
| A2.4 | PASS | 目标形状刻意不命中 expert255，六组原始梯度精确零；小形状空专家/空块/单 token/重复 route/非连续输入覆盖 |
| A2.5 | PASS | 真实权重 B=0 与无专家 adapter 的输出原字节相等；小形状首步 dB 非零 |
| A2.6 | PASS | 目标形状重复输出和所有梯度原字节相等；外部 autocast 不改变结果 |
| A2.7 | PASS | 两 micro-batch 的标准 `.grad` 累积与分别求导之和满足 E3 |
| A2.8 | PASS | 目标形状重入/非重入检查点与直接路径的输出及全部梯度原字节相等 |
| A2.9 | PASS | 块大小 8/32/256 峰值 1.513/1.966/6.193 GB；反向后 allocated 恢复原值；快照确实采集到 Python 解码分配且无存活解码块 |

主要 target run：`20261027T121353Z-expert-target`；数学单元 run：`20261027T120623Z-expert-math-unit`。当前完整回归 `20261027T124338Z-model-op-regression` 共 24 项通过，包括真正四层 HF tiny 的首步梯度对照、实际加载、RoPE、统一开关和一次 AdamW 更新的冻结不变量。它不替代 P4 的 100 步与新进程续训。

HF tiny 的 BF16 来源、NVIDIA 同布局 NVFP4 来源和预处理结果已由 `20261027T123808Z-tiny-assets` 持久生成。位置见主任务 `tiny-bundle.txt`。初次 tiny 集成测试有一处标量直接 `view(uint8)` 的测试错误，保留在 `20261027T123509Z-hf-tiny-build-unit`；reshape 后 `20261027T123631Z-hf-tiny-build-regression` 和完整回归通过。

## 待完成

- 异步搬运事件竞争测试、回归和逐项 M1 审核。
- M1 通过后的真实模型 P5～P8；当前没有真实训练 adapter。

## P4 同步端到端验收

[训练/检查点/发布说明](training.md)、[P4 汇总](evidence/p4-training-summary.json)。正式 run `20261027T132210Z-tiny-training-100-v2`，完整代码 `dba15ba0fd6ae301a51d6964a589dff4ac0ffaa0`，启动时工作区 clean。生产与独立 dense reference 初值相同；100 optimizer 步、每步累积2；实际四层 HF tiny，启用非重入层检查点和 8-token 分块 CE。

| 项目 | 状态 | 实测 |
| --- | --- | --- |
| A4.1 | PASS | 首步各 Tensor 梯度最差相对 L2 0.013178、最低余弦 0.999914；100 步后最低权重余弦 0.992094 |
| A4.2 | PASS | 100 个 step 的最大 loss 相对差 0.001905（约0.19%）；未放宽容差或改变独立参考 |
| A4.3 | PASS E2 | 新进程恢复第50步；第51～100步 loss、梯度范数、LR、样本顺序和四类 RNG 探针逐位一致；最终全部参数、Adam 矩/step、scheduler、数据游标、trainer 和 RNG 状态相同 |
| A4.4 | PASS E2 | 另一新进程只从预处理基座与发布目录构建；固定 batch loss、全部参数、所有 tiny 专家 ΔW 原字节一致；禁用 adapter 的基线也相同 |
| A4.5 | PASS | CPU 专家 payload、冻结参数及 buffer 的训练前后哈希不变 |
| A4.6 | PASS | 74 个 FP32 Tensor、115712 元素；optimizer/预期/发布 canonical 集合及形状相同 |

真实 tokenizer 的 200 黄金样本通过，run `20261027T132406Z-template-goldens`，[摘要](evidence/template-summary.json)：逐 token 等于固定 Speed `_prompt_tokens` 的输出；200 次强制左截断均保留完整答案，另200次答案超长被丢弃；末尾换行不监督。使用与 Speed 同样的 tokenizer 默认设置，保留上游 Mistral-regex 警告，不单方面改变训练分词。

真实 layer0 的诊断重数量化在 `20261027T132636Z-quantizer-layout-zero-diagnosis` 完成，[摘要](evidence/quantizer-layout-summary.json)。三投影块缩放与非零缩放块的码值一致率均100%，解码数值一致率100%。gate/up 的总码一致率只有58.605%，原因已逐块定位：6944896 个零缩放块中，原码不为零但乘缩放后为零；全部码差异都在这些零缩放块。down 为4096个零缩放块，总码一致率99.976%。诊断没有写出替换权重，源文件哈希不变。

### Tiny 的内存实测

所有数值均为 tiny，不推算为真实模型可训练峰值。生产单独的新进程续训 allocated 峰值 78526464 字节、reserved 79691776 字节；其工作集约2.41GB。带参考模型的第一步峰值89744384字节，评测/发布峰值88538112字节。加载、首步、稳态、保存、评测和恢复的观测保存在 P4 汇总的 `memory`；CPU 字段是观测点工作集，不冒充区间峰值。

| 生产对象 | 字节 | 生命周期 |
| --- | --- | --- |
| 冻结 GPU 非专家 | 4236272 | 常驻 |
| RoPE buffers | 64 | 常驻 |
| 两个 GPU U8 层槽 | 885120 | 常驻 |
| FP32 LoRA | 462848 | 常驻 |
| FP32 梯度 | 462848 | backward 到 zero_grad |
| FP32 Adam 矩 | 925696 | 首步后常驻 |
| CPU Adam step | 296 | 首步后常驻 |
| CPU pinned 专家 payload | 1770240 | 常驻，区别于 allocator 实际占用 |

第一步完整 CUDA 快照的派生审核 `20261027T132748Z-tiny-snapshot-audit`，[摘要](evidence/p4-snapshot-summary.json)：1944 次带解码栈的分配，反向后0个活跃解码分配。快照还定位到两个32MiB长期工作区，分别首次出现在 RoPE 的矩阵乘和 CE 的 Linear 重算；没有把它们误算为常驻解码权重。

首次带默认 memory-history 栈采集的 run `20261027T132104Z-tiny-training-100` 在第一个 CE 重算报 SystemError。改为 `context='alloc', stacks='python'` 后，短程及正式100步通过；保留原始失败。量化诊断首版的文件路径漏了 `experts/`，在读文件前失败，`20261027T132523Z-quantizer-layout` 保留。最新完整单元回归 `20261027T132008Z-training-regression` 共31项 PASS，包括损失归一化、参数重排恢复、四类 RNG、发布重载和数据游标。

P0/P1/P2/P4 同步路径通过，P3 尚未验收；M1 未通过，没有真实完整模型训练 adapter。

## 本机证据根

`C:/workspace/runs/T-20261027-nvfp4-lora/`。

- `20261027T103336Z-bootstrap`：初始环境、meta/PEFT、原 GDN/256 组 GEMM、文档远端归档。
- `20261027T105133Z-pinned-h2d`：18.5GB 锁页和 H2D。
- `20261027T105826Z-gdn-module`：完整单 GDN 模块和两种检查点。
- `20261027T111042Z-freetoken-dequant`：只读 FreeToken 合成数据对照。
- `20261027T111254Z-gdn-byte-check`：严格按原始字节比较的 GDN 回归。
- 后续格式夹具与真实来源 run 的命令/源代码快照保存在各自目录，主任务在 `C:/workspace/tasks/T-20261027-nvfp4-lora/README.md` 交接。
