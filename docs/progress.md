# 实施状态

总体目标：依据 v2 在本机实现并验证真实 Qwen3.6-35B-A3B NVFP4 LoRA 训练，交付训练代码、可重载 adapter 和续训证据。M1 已在 `20261027T134521Z-m1-audit-final` 通过全部32个必需关卡，当前进入 P5～P8；最终目标仍未完成。

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

- 2048/4096长序列、真实专家梯度/未命中及历史动量检查、多步C3/专家LoRA训练和canary/replay验证。
- 真实训练中间步的新进程精确续训，以及训练进程内ΔW与发布重建的逐位对照。当前已有真实首步适配器和新进程重载证据，尚未完成最终训练验收。

## P5 真实整模前向与参考

真实40层完整前向已执行。`20261027T135549Z-real-forward-cpu-reference` 的完整前向有限值、加载和内存测量完成，但原A5.2无条件CPU FP32的1%阈值失败，**没有把该run改成PASS**。[初次原始摘要](evidence/p5-initial-forward-summary.json)。

| 项目 | 状态与证据 |
| --- | --- |
| A5.1 | 已实测。非专家4896711936字节、C3参数76677120字节、两GPU层槽905981952字节、CPU原字节专家18119639040字节；128-token前向峰值allocated6497629696字节、reserved6692012032字节。不是训练峰值 |
| 原A5.2 | 无条件CPU FP32各层误差0.896%、2.052%、1.766%、2.464%，后三层未满足原1%阈值；首层也有18/128个token的top-k集合发生跨精度变化 |
| 修订A5.2 | 按v2第6.2节调整为“实际路由条件下的FP32连续图＋独立路由核对”；在尚未测量的三条新提示上全通过。详见下文，不能称作原无条件1%通过 |
| A5.3 可选 | NOT RUN，尚未开发FreeToken teacher-forcing接口；不修改其环境 |
| A5.4 | 三条非思考短贪心回答分别为“2”“北京”“Hello”，均紧接im_end停止；使用固定真实模板、完整无cache前向，[回答证据](evidence/p5-greedy-answers.json) |
| A5.5 | 无捕获D2H的128-token前向wall3.531秒；CUDA区间：专家解码2.350秒、专家计算0.773秒、GDN0.189秒、全注意力0.010秒；40次H2D共18119639040字节，copy区间0.499秒，acquire等待0.062秒。区间有重叠/入队间隔，不应直接相加 |

CPU参考按层加载非专家、按实际命中专家逐个解码；从磁盘embedding开始，FP32 GDN使用独立逐token递推，全注意力用eager FP32。四层HF tiny上与正常chunk GDN/SDPA/稠密专家输出相对误差均<1e-5，`20261027T135239Z-cpu-reference-unit` 与后续回归通过。真实CPU四层参考约2.635秒，没有构造40层完整浮点专家模型。

`20261027T140008Z-forward-precision-diagnosis` [控制实验](evidence/p5-precision-diagnosis.json)分离了误差来源：固定GPU实际route后，CPU FP32端到端误差为0.331%、0.403%、0.419%、0.535%；逐层同时固定输入时约0.294%～0.331%。CPU使用BF16策略但自由选路时仍有最高2.738%差异。不同top-k token的误差明显更高，支持离散cutoff/舍入差异而非格式错位。生产模型的路由、参数和计算策略均未修改。

修订判据与三条新提示在测量前提交于 `69a8cbd`，依据和完整限制见 [决策](decisions.md)。`20261027T140549Z-routed-reference-new-fixtures` [摘要](evidence/p5-routed-reference-summary.json)三条新的128-token提示均执行完整40层GPU前向。前四层条件FP32相对L2最高0.580%，router真实输入与磁盘权重的独立CPU FP32投影误差约0.154%～0.158%，所选集合均满足实际logits的top-k、无重复，softmax/归一化权重一致。选路变化均位于两倍logit扰动可解释的cutoff范围。三条新提示的原无条件1%判据仍均未通过，该字段明确保留为false。

`20261027T141440Z-p5-training-core-regression` 完整34项通过。P5按拆分后的判据进入真实训练验证，首步结果见下文。

## 真实 C3 与全专家 r2 首步机制验证

代码 `431e938` 下两个独立真实训练run均PASS：C3为 `20261027T141616Z-real-c3-step-probe`，[摘要](evidence/real-c3-step-summary.json)；C3＋全部40层×256专家r2为 `20261027T142254Z-real-expert-r2-step-probe`，[摘要](evidence/real-expert-r2-step-summary.json)。配置为C3 r16/alpha32，专家alpha4，标准AdamW/weight_decay0/foreach=False，warmup10的首步LR2e-5，累积8，非重入层检查点和128-token分块CE。数据仅8条合成关联问答，每条监督6个token，实际输入28～33个token；绝非2048/4096训练验收。

| 项目 | C3 | C3＋专家r2 |
| --- | --- | --- |
| 可训练Tensor/元素 | 500 / 19169280 | 740 / 176455680 |
| 首步平均loss / 监督数 | 5.9424897 / 48 | 5.9424897 / 48 |
| 裁剪前梯度范数 | 14.577094 | 15.160194（非专家14.577094、专家4.164112） |
| 首步wall，含首micro快照 | 约86.6秒 | 约109.6秒 |
| 首步allocated峰值 | 6631879680字节 | 8705713664字节 |
| 已有Adam状态下的梯度诊断/评测峰值 | 6786391040字节 | 9319226368字节 |
| CPU工作集观测 | 约24.2～24.4GB | 约24.2GB |
| 固定单样本loss，更新前→后 | 5.057181→5.092501 | 5.057181→4.871934 |

两者step0开启/禁用adapter的loss均E2；所有投影B Tensor均有非零首步梯度（这不等于每个专家都命中）。更新后关闭两套adapter恢复原基线loss E2，冻结GPU参数/buffer及CPU原始专家bank哈希不变。优化器边界检查点与发布文件均成功写出。

B已经更新为非零后，又运行8个micro的真实损失反向、没有第二次optimizer更新：第0层GDN/共享专家和第3层全注意力/共享专家共13个PEFT Linear、26个A/B梯度，与真实输入/上游dY的独立CPU解析矩阵微分对照；最差相对L2分别约2.789e-7、2.769e-7，最低余弦约1。这里检查的是这两层C3梯度，**不代替A7.1的专家A/B、dX和路由VJP对照**。

第一micro反向的真实CUDA快照均通过无存活解码权重审核：[C3快照](evidence/real-c3-snapshot-summary.json)、[专家r2快照](evidence/real-expert-snapshot-summary.json)。分别观测4347/2322次带解码栈的分配，结束时活跃解码分配均0；完整pickle保存在各run。

`20261027T143034Z-real-c3-publish-reload` 与 `20261027T143211Z-real-expert-r2-publish-reload` 各用新Python进程，只从预处理基座＋发布产物建模；全部GPU LoRA参数原字节等于发布文件、canonical集合完整，固定batch loss及禁用基线loss均E2：[C3重载](evidence/real-c3-reload-summary.json)、[专家r2重载](evidence/real-expert-reload-summary.json)。两份真实适配器位于对应训练run的 `results/publish`。

首步探针只证明真实梯度、更新、冻结不变量、保存和重载机制。单样本loss的一次变化不能代替canary/replay效果评估，尚无真实多步续训证据；P6/P7/P8整体验收与最终goal继续推进。

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

该同步 run 结束时 P3 尚未验收；后续 P3 与 M1 的结论见下一节。尚无真实完整模型训练 adapter。

## P3 与 M1 完成

[异步协议与命令](streaming.md)、[P3 摘要](evidence/p3-async-summary.json)、[逐关 M1 审核及原始证据 SHA256](evidence/m1-audit.json)。异步实现只支持一个 CUDA 计算流和一个拷贝流；CPU pinned bank 持久保存原始字节，槽由 `copy_done / last_reader / pins / generation` 共同保护。

真实 bank 的 run `20261027T133606Z-real-async-qualification` 全部 PASS：

| 关卡 | 实测 |
| --- | --- |
| A3.1 | 1000次随机 acquire，每次完整 payload D2H 逐字节等于 CPU 源；949次实际H2D，共429888436224字节 |
| A3.2 | 0→39→39→0，共80次完整读回；每次 GPU 人为延迟50000000 cycles，release 后仍未完成的读不被覆盖 |
| A3.3 | 同序列故意预测 L+17，80次完整读回仍正确；158次实际拷贝，71572574208字节，错误预测产生额外流量 |
| A3.4 | 真实T4096/H2048/I512/E256/top8，输出、x/路由权重和六个A/B梯度相对同步全部 E2 |
| A3.5 | 记录拷贝事件耗时/等待和字节；Profiler 32个H2D、26个kernel、15对跨流重叠。计算为人为sleep，只证明调度重叠，不冒充训练提速 |

随机全量读回测试的 H2D GPU-event 带宽21.83GB/s、整个验证器有效H2D约4.96GB/s；后者包含D2H与CPU全字节比较，不能当训练吞吐。完整CPU bank训练用payload前后hash不变。

`20261027T133917Z-async-full-regression` 共33项 PASS，含1000次小bank延迟读、pin/generation/跨流/close错误检查。源码 `1442f684ecb83a5fca5de752778118767c280e7f` 下，`20261027T134007Z-async-training-100` 的独立稠密对照、50→100精确续训、发布重载均 PASS，[异步训练摘要](evidence/p4-async-training-summary.json)。

`20261027T134332Z-sync-async-100-comparison` [对照](evidence/sync-async-training-comparison.json)还检查了同步/异步之间的全部100步loss、全局/分组梯度范数、LR、样本顺序、四类RNG探针，以及第50/100步全部参数、Adam、scheduler、数据/RNG状态和最终探针：全部E2。两份逻辑配置除 streaming 选择外相同。

`20261027T134521Z-m1-audit-final` 对32个必需关卡逐项审核通过，且原P2数学/格式解码/独立参考源码与真实目标形状验收快照SHA完全一致。200真实模板黄金样本、真实层量化布局诊断、tiny存活对象清单与无活跃解码权重快照作为辅助检查通过。GDN沿用已验证Torch回退；没有声称测试FLA。

**M1 PASS 只授予进入 P5～P8 的资格；真实整模前向与训练仍待验证。**

## 本机证据根

`C:/workspace/runs/T-20261027-nvfp4-lora/`。

- `20261027T103336Z-bootstrap`：初始环境、meta/PEFT、原 GDN/256 组 GEMM、文档远端归档。
- `20261027T105133Z-pinned-h2d`：18.5GB 锁页和 H2D。
- `20261027T105826Z-gdn-module`：完整单 GDN 模块和两种检查点。
- `20261027T111042Z-freetoken-dequant`：只读 FreeToken 合成数据对照。
- `20261027T111254Z-gdn-byte-check`：严格按原始字节比较的 GDN 回归。
- 后续格式夹具与真实来源 run 的命令/源代码快照保存在各自目录，主任务在 `C:/workspace/tasks/T-20261027-nvfp4-lora/README.md` 交接。
