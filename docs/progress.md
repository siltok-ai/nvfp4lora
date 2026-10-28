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

- C3与全部专家r2的2048/4096两步长序列压力、独立40层重算峰值及解码生命周期补测均已通过；后续仍需完整功能训练/效果和严格续训。
- 真实C3和专家r2均已完成4步及2→4步新进程E2续训；v2 A6.3/A7.3要求的完整50→100步续训仍待执行，不能用短程证据替代。
- 300训练/50改写留出及300基座回放语料已准备；第4步两者留出均0/50，尚无金丝雀效果通过结论。阈值仍待用户确定，训练和独立机械验收继续推进。
- C3与专家r2的训练内ΔW与发布重建已全部E2。还需FP8来源层输入范围观测及最终产物/操作说明。

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

## A7.1 实际专家梯度与功能语料

`20261027T154022Z-real-expert-activation-vjp` [真实激活VJP](evidence/real-expert-activation-vjp.json) PASS：从已经有非零B的真实r2发布产物运行完整训练反向，捕获第0/3层的实际x、idx、路由权重、输出和上游dY。局部生产算子重算的A/B梯度与全模型梯度E2；独立稠密autograd参考的输出、路由和六个A/B梯度误差均0，dX相对L2为0.004053/0.003649。两层分别有117/150个未命中专家，原始梯度严格为0。参考只物化当前层BF16专家，不物化完整35B浮点专家。

`validation/route_audit.py` 已接入真实训练CLI：更新前检查本step未命中行的原始梯度；更新后检查从未命中行的参数原字节和Adam矩为零，历史未命中行的变化仅计数而不误判失败。ever历史进入trainer state；恢复后用已验证checkpoint的未命中行继续归纳。Tensor级Adam step会增长，与每专家矩为零不冲突。实际小算子+标准AdamW的命中切换、历史动量、恢复审计和篡改梯度检测在 `20261027T154432Z-route-audit-unit` 通过。

`20261027T155244Z-prepare-canary-replay` [语料摘要](evidence/canary-replay-data-summary.json) PASS：本机 `data/datasets/nvfp4lora/canary-replay-v1` 含300训练金丝雀、50个新问题模板的改写留出、300条真实基座生成的算术回放，训练文件600条按两类交错。基座留出命中0/50。[三条批量/单条生成控制](evidence/base-generation-batch-controls.json)全部一致。它是一个人工关联的功能测试，不代表领域/通用能力基准。

`20261027T161652Z-prepare-sequence-stress` [压力材料](evidence/long-sequence-data-summary.json)含8条每条1035监督token的完整答案，同一数据按左截断提示得到精确2048/4096输入；用于后续显存/长序列验证，尚未作为该项训练通过证据。

`20261027T160531Z-session-core-regression` 共36项 PASS，新增真实HF tiny的批量左padding NLL/总监督分母、评测模式恢复，以及发布前后ΔW指纹一致；没有用假模型代替以上计算。

## 真实 C3 多步、续训与全部 ΔW

代码 `4086426`，`20261027T161035Z-real-c3-multistep-baseline` [基线](evidence/real-c3-multistep-summary.json)在600条功能语料上运行4 optimizer step，累积8；逻辑上限100与执行stop-at4分开，保存第1/2/4步。`20261027T162157Z-real-c3-resume-step2` [新进程](evidence/real-c3-resumed-summary.json)从第2步恢复到第4步。

`20261027T162936Z-real-c3-resume-comparison` [严格比较](evidence/real-c3-resume-comparison.json)全部PASS：第3/4步loss、监督数、序列token数、样本ID、LR、全局/分组范数、路由状态和四类真实消耗的RNG探针E2；最终全部参数、命名Adam矩/step、scheduler、data/trainer/RNG、逻辑配置和发布probe相同。训练内的全部ΔW字节指纹也相同。两执行冻结权重哈希不变。

`20261027T162939Z-real-c3-release-delta-verify` [独立发布进程](evidence/real-c3-release-delta-summary.json)再从bundle+publish建模：固定batch/禁用基线loss、全部参数原字节和全部250个C3增量矩阵指纹E2。ΔW按FP32 `scaling * B@A` 在训练进程内计算，逐块读取回CPU哈希，不保存/合并全量权重。

第4步的金丝雀仍0/50，回放NLL从0.000211944变为0.000267063 nat/token；这证明恢复/发布协议，不是效果验收完成。效果阈值已按v2要求向用户异步询问，训练与机制验证继续。

专家r2的 `20261027T163147Z-real-expert-r2-multistep-baseline` 已完成4步，[摘要](evidence/real-expert-multistep-summary.json) PASS：每步全40层未命中原始梯度为0，从未命中参数和Adam矩不变；到第4步仍有3196个从未命中的专家-层对。第2/3/4步分别观察到1221/12703/3031个历史未命中参数行因动量变化。第4步留出0/50，回放NLL0.000295614；新进程2→4恢复正在验证，不能提前声称该项通过。

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

## 专家 r2 短程续训及发布增量完整核对

`20261027T164847Z-real-expert-r2-resume-step2` 从真实基线第2步新进程恢复到第4步，执行PASS。`20261027T165726Z-real-expert-r2-resume-comparison` 对照全部740个参数、命名Adam矩/step、scheduler、四类RNG、data/trainer state、逻辑配置、逐步3/4的loss/范数/LR/样本/路由审计/RNG探针、固定发布probe及训练内ΔW，全部E2，[对照摘要](evidence/real-expert-resume-comparison.json)。原第4步仍是金丝雀0/50，不代表效果通过。

`20261027T165738Z-real-expert-r2-release-delta-verify` 在新进程仅凭bundle+publish建模，固定loss、禁用基线loss、全部GPU参数和4090份ΔW指纹全部E2，[发布摘要](evidence/real-expert-release-delta-summary.json)。与C3的250份对应，4090份含250个非专家矩阵及40层×3投影×32个专家分块；分块内按FP32 scaling×B@A逐字节哈希，未合并或改写基座。

两套真实2→4步证据都只是短程验证。v2 A6.3/A7.3的完整第50→100步对照仍待执行；不将tiny的100步或真实短程结果代替该项。

## C3真实长序列压力训练

C3在2048和4096均完成新鲜初始化的2 optimizer step×累积8，完整答案1035 token/条、每步8280个监督token。各长度均有首micro快照、首步及已有Adam的第二步、checkpoint1/2和独立压力发布目录。冻结参数/hash不变；Adam矩按全部可训练元素实际分配；所有8条完整答案保留。没有使用这些压力适配器作金丝雀或P8效果证据。

| 长度 | 训练run | 训练两步最高allocated / reserved字节 | 每步wall秒 |
| --- | --- | --- | --- |
| 2048 | `20261027T170053Z-real-c3-sequence2048` | 8499081728 / 8793358336 | 177.37 / 187.78 |
| 4096 | `20261027T170840Z-real-c3-sequence4096` | 11606167040 / 12664700928 | 213.81 / 211.20 |

[2048审核](evidence/real-c3-sequence2048-audit.json)、[4096审核](evidence/real-c3-sequence4096-audit.json)包含加载、首micro、首步、稳态、保存、发布/探针各阶段及CPU对象清单、每步H2D/等待、分组A/B范数。CPU工作集字段是观测点，不能当作区间峰值。吞吐按实际输入token数除以step wall，wall包含审计开销。

两份原始首micro快照的200000事件环已满，保留了尾部轨迹；allocated/reserved的框架峰值计数独立于该环，仍有效。快照全局存活块检查均无活跃解码权重，但不能用尾部轨迹宣称覆盖整个micro。为分离逐层重算峰值，新增 `profile_recompute_memory.py`：从各压力step2恢复LoRA与真实Adam，只做一个micro的反向，用always_call层hook覆盖非重入检查点的提前停止，各次reset前合并原峰值，另采集上限2000000的轨迹。它不更新或发布参数。

`20261027T171813Z-real-c3-recompute2048` PASS：40层重算全部记录，逐层重算最高allocated8410685440字节，整micro最高8421765120字节。去掉采集hook再跑同一micro，loss与全部500个A/B梯度原始字节E2；冻结hash不变，[证据](evidence/real-c3-recompute2048.json)。这次profile带已有Adam，其吞吐不代替原压力训练的吞吐。

补测的2000000事件环同样已满，`history_complete=false`如实记录；不宣称完整轨迹。逐层重算峰值由每层入口reset和always_call退出时的CUDA独立峰值计数给出，不依赖事件环是否完整。表中2048第二步耗时已按原始metrics更正为187.78秒。

`20261027T172124Z-real-c3-recompute4096` 同样PASS：[证据](evidence/real-c3-recompute4096.json)。40层重算最高allocated10149885440字节，整micro最高11530374656字节；独立快照审核 `20261027T172415Z-real-c3-recompute4096-snapshot` 无活跃解码块。两次补测均从已有Adam、梯度起始None的optimizer边界开始，只测一个micro；后续累积micro的总峰值仍以完整两步压力训练为准。

## 专家r2真实2048压力训练

`20261027T172405Z-real-expert-r2-sequence2048` 两步×累积8 PASS，[审核](evidence/real-expert-sequence2048-audit.json)。与C3使用同一8条压力记录、2048完整输入/1035完整答案，首步266.68秒、第二步251.57秒；已有Adam时最高allocated11045118464、reserved11303649280字节。全部40层未命中原始梯度0、从未命中参数/Adam矩不变，冻结hash不变。首micro快照审核无活跃解码权重；其200000事件轨迹截断状态不改变框架峰值读数。

`20261027T173447Z-real-expert-r2-recompute2048` 独立补测PASS，[证据](evidence/real-expert-recompute2048.json)。从压力step2恢复真实Adam的首micro，40层重算最高allocated10672066560、整micro10699755520字节。去掉hook后同一micro的loss及全部740个A/B梯度字节E2，冻结hash不变；2000000事件环仍截断，峰值由逐层独立计数给出。4096专家压力仍待完成，不据此宣称全部长度或P7已通过。

## 专家r2真实4096压力与容量结论

`20261027T174224Z-real-expert-r2-sequence4096` 两步×累积8 PASS，[审核](evidence/real-expert-sequence4096-audit.json)。每条实际输入4096、完整监督1035，全部8条保留；首步339.36秒、第二步322.91秒。首步allocated12716862976/reserved14403239936字节，已有Adam的第二步最高allocated14127787520/reserved15841886208字节。冻结hash、原始未命中梯度、从未命中参数/矩、检查点及压力发布均通过。使用全部40层×256专家r2、标准AdamW、chunk32、CE128，没有减少专家或更换优化器。

[整卡观测](evidence/real-expert-sequence4096-gpu-observation.csv)是第二步运行中的单次样本：总16303MiB、用15775MiB，约余528MiB；不是区间峰值或未来工作负载保证。当前4096专家配置余量较小，应保持单GPU作业。r4是v2可选项，本次未试。

`20261027T175535Z-real-expert-r2-recompute4096` PASS，[补测](evidence/real-expert-recompute4096.json)：已有Adam且梯度起始None的首micro，40层重算最高allocated12101009408、整micro13470559744字节。完整两步的后续累积micro峰值仍以上述14127787520为准，不用首micro低值代替。补测无optimizer更新，冻结hash不变，事件环仍截断。C3与专家r2、两种长度的四份非零B补测快照，最终存活解码块均0，[生命周期汇总](evidence/real-recompute-lifecycle.json)；最终快照全局存活块的检查不依赖有上限的事件尾部是否完整。

到这里，两套配置的2048/4096各2步压力和重算补测全部通过；它们证明本机容量和训练机制，不代表金丝雀、领域效果或完整100步续训通过。下一步使用已准备的串行脚本完成真实600条功能语料的100步及50→100对照，保留独立压力适配器，不混入功能训练。

## 完整真实验收已启动；C3第10步首次功能效果

父run `20261027T180151Z-real-full-acceptance-campaign` 已实际启动，C3子run `20261027T180152Z-real-c3-functional100` 从同一600条功能语料的step4继续5～100。两套模型的独立50→100、比较、发布和风险观测仍未完成，不能把父run的RUNNING或已启动写成通过。

C3第10步：[原始评测](evidence/real-c3-evaluation-step10.json)实际50条换模板留出全部正确（基座0/50），300条回放NLL 0.00005730681090969801（基座0.00021194423469130964）。这是单个人工关联的功能结果，不证明领域收益；训练继续到100，完整续训仍待验证。第10步仅有固定评测记录，不单独保存/发布适配器；预设保存点为20/50/100。

审查v2 6.5发现：训练入口已有replay NLL和canary成功率，但C3的训练路由审计因rank0不持有专家A/B而不记录路由分布，也尚无canary留出NLL。新增 `inspect_training_checkpoints.py` 从保存的20/50/100参数离线重建这些诊断及A/B分组范数，固定训练数据前8条作探测批次，排除左padding；记录的是保存参数的离线重建，不伪称原训练时刻捕获。

新增 `MaskedRouteProbe` 的3项CPU测试在 `20261027T181113Z-route-probe-cpu-regression` PASS：rank0仍记录、padding排除、错误mask/漏层拒绝及hook卸载。真实C3/专家step4的500/740个Tensor已在CPU逐字节载入对照，错误配置和形状均拒绝，`20261027T181325Z-checkpoint-diagnostics-cpu-loader` [摘要](evidence/checkpoint-diagnostics-cpu-loader.json)PASS。尚未在GPU执行新诊断工具，需在当前串行campaign之后分别对两套实际20/50/100执行；不并行占用训练GPU，不更改训练数学或现有逻辑配置。

## M2逐项审核与C3第20步

C3第20步仍50/50且50条均EOS停止，回放NLL0.00006776975508623592，低于基座0.00021194423469130964；[原始记录](evidence/real-c3-evaluation-step20.json)。完整step20检查点已保存，训练继续，尚未得到最终100步适配器。

新增 `tools/audit_m2.py` 将P5～P8、两套完整50→100、长序列与重算/非零B生命周期、周期诊断、FP8观察和交付列为独立证据项。`20261027T185909Z-m2-audit-provenance` [部分审核](evidence/m2-audit-partial.json)当前为 **INCOMPLETE**：15项PASS、1项PASS_WITH_DOCUMENTED_CHANGE、16项PENDING。这里是M2证据目录的32项，不等同于M1的32项资格，也不表示已通过完整M2。

调整项明确保留原无条件CPU FP32 1%失败；另用Git祖先关系和源文件内容核对，证明三个新夹具及CPU参考在运行前已提交于69a8cbd9d1123ab3c15b4443499e156b97dd2568，不能事后换夹具掩盖旧失败。`20261027T185508Z-m2-strict-incomplete-guard` 使用require-complete故意得到exit1，[门槛收据](evidence/m2-incomplete-guard.json)确认不完整状态会被拒绝；该run的FAIL是预期负向检查，未改为PASS，不是训练失败。

最终审核仍须在实际100步/恢复/发布/风险/后验诊断和制品准备完成后重跑严格模式。效果阈值未由用户确定时不填造provenance；原始观测与正式门槛分别记录。当前16个待完成项完整保留，没有缩小goal。

## C3中点检查点已具备；基线继续运行

第50步的[固定评测](evidence/real-c3-evaluation-step50.json)仍为50/50且全部EOS停止，300回放NLL0.00000021567823518715448。完整step50已保存，原基线进程继续51～100，未在中点停止或用恢复进程替代后半程参考。

`20261027T191524Z-c3-midpoint-checkpoint-audit` [CPU边界审核](evidence/real-c3-midpoint-checkpoint-audit.json)PASS：所有文件SHA一致，500个可训练参数/唯一命名optimizer集合一致，所有参数及FP32 Adam矩有限、形状正确，逐参数step均50；scheduler.last_epoch50，累积位置0，数据消耗400（epoch0/position400/seed113），四类RNG存在。该检查仅证明边界文件与状态完整，不能代替后续实际新进程50→100的E2对照，也不算一次模型发布重载。

## 制品归档工具准备

`tools/archive_training_delivery.py` 为最终交付准备，只写不存在的新目录。输入是新一次M2预审：所有技术项必须已通过，仅允许交付目录本身和真实用户门槛未提供两类PENDING；后者不等同效果批准。它重新核对预审source SHA、已完成run、原运行的数值源码及实际后验诊断源码，并核对39个锁定依赖版本。

计划归档两套results结构（publish、完整20/50/100、summary/config/metrics/ΔW/评测），固定语料、关键比较/风险/诊断证据、锁文件、环境库存和使用说明。完整基座仍引用models/bundles；逐文件复制前后SHA一致后最后写manifest，`verified`只标记复制和机械来源资格，正式M2仍由严格审核决定。当前长训练尚未结束，没有实际交付制品。

`20261027T193127Z-delivery-copy-cpu-regression` 两项CPU测试PASS：1MiB全字节范围二进制复制一致，既有目录拒绝且不变；注入损坏会拒绝认证、保留失败副本且不写manifest。`20261027T193245Z-delivery-incomplete-guard` 用当前INCOMPLETE预审正确exit1且未创建制品目录，[预期门槛收据](evidence/delivery-incomplete-guard.json)PASS；原run的FAIL保留为负向测试，不是训练失败。当前依赖再次读取39个版本，0差异；未重跑或并行占用GPU的旧回归。

## 本机环境登记与实际首轮数据覆盖

`20261027T194457Z-environment-registry-in-progress` [登记收据](evidence/environment-registration-in-progress.json)PASS，39个锁定版本无差异，pip check无缺失依赖；[当前库存](evidence/environment-current-inventory.json)。在C:/workspace/envs/registry.json新增nvfp4-lora-win-cp312-cu130，状态preparing_real_g1_acceptance，注明M1已过但真实G1完整验收仍在运行。原有三个Speed环境的条目逐对象比较保持不变；登记前/后原文均在该run保留。更新了对应环境README的旧预检描述，没有改变依赖或迁移venv。

`20261027T194954Z-c3-data-epoch-coverage` [实际数据覆盖](evidence/real-c3-data-epoch-coverage.json)PASS：拼接原1～4步与当前5～76步的完整JSONL记录，逐步连续、每步8条；首轮600个实际sample_id与accepted_rows逐项计数相同，每条恰好一次。第76步已观测第二轮8条、与首轮前缀顺序不同。该检查不重用采样器排序实现作参考，只检查实际覆盖/重复遗漏；完整输入快照与SHA保留，仍不替代未来新进程50→100样本顺序/RNG/E2对照。

当前仍只有原C3基线GPU作业，未重启、未并行推理，完整100步和后续专家阶段/发布/恢复仍由同一父脚本继续。

## 工作区边界复核

`20261027T200028Z-workspace-boundary-audit` [只读审核](evidence/workspace-boundary-audit.json)PASS：桌面两份原文、归档文件和Git提交字节的SHA256一致；Speed及材料仓HEAD与工作树保持初始基线。Speed任务锁定的97个依赖版本一致，本训练所需39个依赖在两套环境中与固定闭包一致；Speed原Qwen模块SHA与交接快照一致。当前33个原训练src文件仍与完整campaign启动时的源码快照一致。

该审核检查文档字节、Git状态、依赖版本和指定源文件，不声称已枚举或逐字节验证整个venv。Speed环境只运行标准库元数据查询，无GPU导入/安装/升级；训练保持在同一父进程中继续，完整M2尚未结束。

## C3完整100步完成，新进程50→100已开始

`20261027T180152Z-real-c3-functional100` 已正常退出，manifest PASS/exit0。它使用同一功能语料和逻辑100步配置，从原step4恢复后连续完成5～100，在step50保存后同一进程不中断继续至100；完整保存20/50/100和publish，冻结权重hash不变。C3共500个FP32可训练Tensor/19169280元素。

第100步[原始评测](evidence/real-c3-evaluation-step100.json)：改写留出50/50且全部EOS结束（基座0/50），300条回放的981个监督token平均NLL为0.00000011701894606100652（基座0.00021194423469130964）。第10/20/50/100步均50/50。这是人工关联/算术回放功能观测，不是领域收益或已获用户批准的效果门槛。

`20261027T202515Z-real-c3-functional100-observations` 汇总MEASURED，完整原始观测保留在run中。5～100共96步平均85.517秒、中位85.468秒，optimizer step wall合计8209.60秒（不含单独评测/保存）；输入token合计27695、监督3609，属于实际短功能样本。该会话各记录阶段最高allocated6790487552/reserved7096762368字节，不能替代4096压力测试的更高峰值。

`20261027T202634Z-c3-final-checkpoint-audit` [CPU终点审核](evidence/real-c3-final-checkpoint-audit.json)PASS：全部检查点文件SHA一致，500个唯一命名参数及FP32 Adam矩有限且形状正确，逐参数step均100；scheduler.last_epoch100，累积位置0，四类RNG存在。数据状态consumed800/epoch1/position200，拼接实际1～100步共800个sample_id后确认首轮600条恰好一次、第二轮200条均有效且不重复。CPU读回发布参数与step100检查点逐字节一致。这不代替新进程GPU发布重建，也不代替实际恢复终点对照。

同一父脚本已自动启动 `20261027T202444Z-real-c3-resume50`，从完整step50恢复、实际开始51～100，未重启或替换已完成的参考段。`20261027T202805Z-c3-resume50-prefix-check` [早期前缀](evidence/real-c3-resume50-prefix-check.json)只覆盖已完成第51步：loss、样本/监督/输入token、LR、总/分组梯度范数、路由审计、四类实际RNG探针九字段与基线完全相同。完整参数、Adam、scheduler、数据/RNG终态、发布probe和全部ΔW仍须等100结束后执行正式比较。

`20261027T202710Z-m2-c3-full100-audit` [更新审核](evidence/m2-c3-full100-audit.json)仍为INCOMPLETE：16 PASS、1 PASS_WITH_DOCUMENTED_CHANGE、15 PENDING。C3完整训练关闭了冻结/参数集合项；完整续训、GPU发布重建、独立效果/FP8观测、两套后验诊断、专家功能训练和最终交付仍未完成。保留原无条件A5.2失败及调整依据，真实用户效果阈值尚待决定，goal保持active。

恢复期间的提前检查 `20261027T204044Z-c3-resume50-first10-check` 覆盖实际第51～60步的九项逐步记录，全部与不中断基线相同；[前10步收据](evidence/real-c3-resume50-first10-check.json)。原始metrics字节快照和SHA保留，观察时训练进程已继续第61步。它只核对该前缀的loss、样本/监督/输入token、LR、总/分组梯度范数、路由审计与四RNG探针，不能替代第100步的完整参数/Adam/scheduler/data/RNG/publish/ΔW比较；正式M2状态仍为INCOMPLETE。

`20261027T210310Z-c3-resume50-epoch-boundary-check` [跨轮次检查](evidence/real-c3-resume50-epoch-boundary-check.json)PASS：恢复后51～76共26步的九项记录仍与不中断基线完全相同。拼接实际基线前400个样本和恢复后的208个样本，首轮600条accepted record各出现一次，第二轮前8条有效且顺序与不中断参考相同；检查没有调用采样器排序实现来生成参考。输入metrics原字节快照和固定输入SHA保留。这只证明已执行前缀及实际跨轮次覆盖，完整50→100终态比较仍未完成。

工具更新使旧unified session36600失效后，已用实际父/子PID与持续新增日志重新确认原任务活跃，训练没有重启。后续通过PID、manifest和日志跟踪同一作业，旧等待句柄失效不作为训练退出的证据。

## C3完整50→100一致，发布重建与独立观测完成

恢复run `20261027T202444Z-real-c3-resume50` 已PASS/exit0，实际完成51～100共50步，冻结hash不变。`20261027T213801Z-real-c3-resume50-comparison` [完整比较](evidence/real-c3-resume50-comparison.json)十项全E2：全部参数、命名Adam、scheduler、四RNG、数据游标、trainer state、逻辑配置、全部50步九项记录、发布probe和训练内全部ΔW。差异列表为空，端点100、起点50，已超过此前仅前缀的证据范围。

`20261027T213804Z-real-c3-functional-release` [新进程发布重建](evidence/real-c3-functional-release.json)PASS。只读真实bundle/publish，固定启用/禁用loss、全部GPU参数字节及250份FP32 ΔW指纹与原训练记录全部E2；启用loss7.947285496356926e-8，禁用基线4.523958683013916。没有使用训练checkpoint或optimizer建模，也没有合并/重新量化原始专家。

`20261027T213837Z-real-c3-functional-evaluation` [独立效果观测](evidence/real-c3-functional-evaluation.json)MEASURED，进程PASS/exit0。50条改写留出仍50/50（基座0/50），全部生成token及回放记录与训练step100一致，回放平均NLL1.1701894606100652e-7，变化-0.00021182721574524864。固定金丝雀提示的启用/禁用logits差异L2=1463.29443359375、max_abs42.09375、248001个词表项变化。真实用户阈值仍未提供，不把观测MEASURED改为效果正式批准。

`20261027T214023Z-real-c3-functional-fp8` [原FP8尺度观测](evidence/real-c3-functional-fp8.json)OBSERVED，进程PASS/exit0。再次校验原三分片；两阶段各600条、每个130 FP8来源模块19次调用/21454个有效token向量，排除左padding。禁用适配器的训练后端有5个模块max|x|/(448×原input_scale)>1；启用C3后为4个，无新增越界模块。最坏是第2层GDN out_proj，2.3836734693877553→2.5142857142857142；其它启用后越界为第5/8/10层GDN out_proj，原第35层attention o_proj不再越界。它测量训练后端输入范围，没有运行W8A8量化推理；只记录最大值，不能据此推断截断比例或实际部署质量。该风险留给G2验证，没有据此静默更换训练数值策略。

`20261027T214601Z-m2-c3-stages-audit` [更新逐项审核](evidence/m2-c3-stages-audit.json)INCOMPLETE：20 PASS、1 PASS_WITH_DOCUMENTED_CHANGE、11 PENDING。C3完整续训、容量阶段和发布重建已关闭；C3效果观测/重载一致已检查，但owner thresholds/provenance未提供仍PENDING。两套20/50/100后验诊断、专家完整功能训练/续训/发布/风险及最终交付也尚未完成。

原父脚本现已串行进入 `20261027T214341Z-real-expert-r2-functional100`：从原step4继续5～100，C3 r16/alpha32＋全部40×256专家r2/alpha4，标准AdamW、累积8、相同600条数据和逻辑100配置。该进程的git head88d2f86且启动工作树clean，数值提交仍24888c9；没有重启已通过C3阶段或额外占用GPU。训练、后验诊断和最终交付尚需继续，goal保持active。

## 专家r2第10步功能观测与实际输入核对

第10步[原始评测](evidence/real-expert-evaluation-step10.json)留出50/50且全部EOS结束（基座0/50）；300条回放的981个监督token平均NLL2.7792592353503638e-5，基座0.00021194423469130964。同样步数的C3为50/50和5.730681090969801e-5。它们是人工关联/算术回放的阶段观测，不证明专家LoRA的领域收益；专家完整100步和50→100对照仍在后续。第10步只评测，独立保存点仍为20/50/100。

`20261027T220041Z-expert-step10-progress-audit` [CPU前缀审核](evidence/real-expert-step10-progress-audit.json)PASS：拼接各自原1～4步与实际5～10步，C3和专家的80个实际sample_id、每步输入/监督token数相同，数据和评测身份也一致。专家1～10步loss/梯度有限；40层各256个路由计数精确求和为8×实际输入token，排除了把检查点重算重复算成真实路由的情形。全部步骤未命中原始梯度0、从未命中参数不变/Adam矩0；第10步仍有2507个从未命中专家层对，同时7510个历史命中但本步未命中的参数行因动量变化，属于预期标准AdamW语义。源metrics字节快照与固定输入SHA保留；该审核不替代完整100步/发布/续训或真实owner效果门槛。

## 专家第20步保存与独立持久状态核对

第20步[原始评测](evidence/real-expert-evaluation-step20.json)留出仍50/50且全部EOS结束；回放平均NLL为0.0006732941454677073，高于基座0.00021194423469130964及本配置第10步。该回升如实保留，不能沿用早期“回放改善”的结论；训练继续既定100步，最终效果及用户门槛仍未验收。完整step20检查点已保存，原进程继续后续步骤。

`20261027T222027Z-expert-step20-checkpoint-audit` [CPU检查点核查](evidence/real-expert-step20-checkpoint-audit.json)PASS：step20及接受的step4参考检查点全部文件SHA一致，逻辑配置hash相同。740个唯一命名FP32参数/176455680元素及Adam矩形状正确且有限，二阶矩非负，逐参数step均20；两组标准AdamW元数据、scheduler.last_epoch20/LR0.0002、累积位置0、四RNG及data.consumed160/epoch0/position160/seed113完整。

该CPU检查没有调用运行时ExpertRouteAudit重算结论：直接从1～20实际路由计数构造每层ever集合，与保存history逐层相同；第4步ever集合是其子集。对当前2200个从未路由专家层组合的6组A/B，共13200个参数行，直接读取SF比对step20与已验证step4原字节相同，保存的两类Adam矩严格零。原1～20步的原始未命中梯度检查记录也全通过。它证明持久状态和相对已接受step4的归纳链，不代替实际GPU恢复、发布重建或完整100步验收；最终M2仍INCOMPLETE。

## 专家中点已保存，原进程继续后半程参考

第50步[原始评测](evidence/real-expert-evaluation-step50.json)留出50/50且全部EOS结束；回放平均NLL1.1498948365688314e-5，低于基座0.00021194423469130964。本配置第20步的回升仍保留，没有覆盖或删去；阶段回落不代表最终100步、发布重载或部署质量已经通过。

`20261027T231833Z-expert-midpoint-checkpoint-audit` [CPU中点核查](evidence/real-expert-midpoint-checkpoint-audit.json)PASS：检查点全部文件SHA一致，740个唯一参数及FP32 Adam矩有限，二阶矩非负，逐参数CPU step50；标准AdamW两组元数据、scheduler.last_epoch50/LR0.0002、累积0、四RNG和data.consumed400/epoch0/position400/seed113完整。保存的40层ever与实际1～50步路由计数逐层相同，step4 ever为其子集；1957个从未路由专家层组合的11742个A/B参数行与已验证step4原字节相同，两类Adam矩严格零。该CPU核查仍不代替实际新进程50→100或GPU发布重建。

原专家full100进程保存50后已经继续51～100，没有中途停止或用恢复进程替代参考。完整50→100比较由父脚本在该参考正常结束后执行；当前M2仍INCOMPLETE，goal保持active。

## 专家完整100步结束，新进程50→100进行中

`20261027T214341Z-real-expert-r2-functional100` 已正常PASS/exit0：从已接受step4连续完成5～100，在50保存后同一进程继续至100，冻结hash不变，740个FP32参数/176455680元素，完整20/50/100与publish保留。训练进程已计算4090份FP32 ΔW指纹；没有写出全量增量矩阵或合并原始专家。

第100步[原始评测](evidence/real-expert-evaluation-step100.json)留出50/50且全部EOS，回放NLL0.0009931222000904999，较基座0.00021194423469130964增加约0.0007811779654。第20步回升、第50步回落及第100步再次回升全部保留；不能只引用中点较好的值，或由该功能语料宣称领域收益。C3终点回放NLL为1.1701894606100652e-7，两者最终效果门槛仍待真实用户决定。

`20261028T010022Z-expert-final-checkpoint-audit` [CPU终点审核](evidence/real-expert-final-checkpoint-audit.json)PASS：全部文件SHA与唯一命名状态集合一致，740个参数/FP32矩有限、二阶矩非负，逐参数CPU step100、scheduler.last_epoch100/LR0.0002、累积0、四RNG及data800/epoch1/position200完整。实际1～100步的40层路由计数构造ever与保存值一致；1812个从未路由组合的10872个A/B参数行与已接受step4原字节相同，两矩严格零。CPU发布参数与最终检查点逐字节相同，800个实际样本首轮600条各一次、第二轮200条有效不重复，全部100步实际输入/监督数量与C3一致。该核查不是实际恢复或GPU发布重建。

`20261028T010022Z-real-expert-functional100-observations` 汇总MEASURED，完整观测留在run。5～100共96步平均113.770秒、中位113.016秒，step wall合计10921.90秒（另计评测/保存）；短功能样本各记录阶段最高allocated9323847168/reserved9508487168字节，不替代4096压力验证的更高峰值。

父已自动启动 `20261028T005502Z-real-expert-r2-resume50`，从完整50恢复并实际执行51～100。`20261028T010210Z-expert-resume50-prefix-check` [早期前缀](evidence/real-expert-resume50-prefix-check.json)只覆盖已完成51～53步，九项记录与不中断参考全部相同，包括样本、loss、梯度范数、路由审计和四RNG探针；最终全参数/Adam/scheduler/data/RNG/trainer/config/逐步/publish/ΔW比较仍未完成。

`20261028T010211Z-m2-expert-full100-audit` [更新M2](evidence/m2-expert-full100-audit.json)仍INCOMPLETE：22 PASS、1 PASS_WITH_DOCUMENTED_CHANGE、9 PENDING。专家完整冻结/集合与路由历史项已关闭；完整恢复、GPU发布、独立效果/FP8、两套周期诊断和最终交付尚未完成。原A5.2失败/修订依据及真实owner门槛未提供的状态继续保留，goal保持active。

## 交付数据重定位核查与后续诊断排队

`20261028T014301Z-delivery-data-portability` [CPU核查](evidence/delivery-data-portability.json)PASS/exit0。固定600条train.jsonl复制到新的run内路径后，文件SHA、数据身份及全部input_ids/labels原字节一致；C3和专家的绑定逻辑配置保持不变。分别读取hash校验的step50数据状态，用重定位的数据实际生成51～100共400个样本，逐步sample_id、输入/监督token数量与不中断基线全部匹配，终点游标与step100相同。仅追加空行的负例虽然编码样本不变，仍因原文件SHA改变被绑定配置拒绝。过程未初始化CUDA，证明数据/配置/采样器的路径可移植性，不代替GPU恢复或最终制品实际复制核验。

`20261028T014038Z-post-campaign-diagnostics` 已记录串行后续任务，先等待原父runner6480的Windows进程句柄；原整轮及12个子阶段都PASS/exit0后，才依次为C3、专家调用现有`inspect_training_checkpoints.py --steps 20 50 100`。当前等待过程不占GPU，尚不能写成诊断完成；实际阶段由`real-post-campaign-diagnostics-run.txt`及`continuation.json`追踪。该任务不改训练数值代码，最终M2预审、制品归档及用户效果门槛仍单独处理。
