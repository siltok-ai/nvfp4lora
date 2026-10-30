# 实施决策

## 2026-10-28：用户确认本轮功能效果门槛

v2 A6.4要求金丝雀阈值由用户确定，并检查回放loss变化。用户先回复“同意你的建议”接受此前80%/NLL增量0.1建议，随后询问由实施者决定应取多少；实施者推荐收紧为留出成功率≥90%（45/50）、回放平均NLL增量≤0.01 nat/token，用户明确回复“那还是采用你的阈值吧”。本轮正式使用后一组，替代前一组，适用于C3和expert-r2的固定功能语料；[决定记录](evidence/effect-threshold-owner-decision.json)。

理由是按PPL=exp(NLL)，0.01对应相同回放协议下PPL最多增加约1.005%，而0.1允许约10.52%。这是本轮工程验收选择，不作为通用领域质量标准。批准是在两套100步观测披露之后作出的，不能记成训练前批准或新的盲测；保留全部阶段观测及原门槛历史。原始evaluation.json的MEASURED状态不改写，正式判定由新M2审核带这组数值及真实对话provenance完成。无数值实现或数据改动，也不因此提前宣称M2整体通过。

## 2026-10-28：真实阶段的机制预探针与完整验收

补记执行顺序：在P6的金丝雀效果和完整50→100步尚未关闭时，已经做了P7专家r2的首步、真实激活VJP、短程续训及长序列机制探针。这样先确认专家autograd、标准Adam动量和16GB硬件容量，避免在数小时的完整功能训练之后才发现基础实现或资源问题。依据v2 0.4第9项记录该阶段顺序调整；没有宣称P6/P7或M2整体通过。

这类探针只授予所测机制的证据，不能代替A6.3/A7.3完整50→100步或A6.4效果。保留原已预设max_steps100的600条逻辑训练配置，长序列后顺序完成C3、专家r2的100步及新进程50→100对照；失败时停下保存原run，不自动改基座、优化器或数值规则。金丝雀阈值仍待用户确定，独立于机械检查继续推进。

## 2026-10-27：启动与既有决定

- 本次用户授权实施；交接材料的旧会话“只分析”状态不再适用。
- 保留 v2 的 A→A′ 交付路线、原生 Windows、独立环境、原专家格式与完整梯度协议。
- 继承同版本传递依赖授权：accelerate==1.15.0 是 PEFT 必需依赖，只安装，不用其训练循环。
- GDN 先实测 Transformers 5.16.1 中现有纯 Torch chunk 实现。FLA 未安装，不因方案误称“只有逐 token”而先重写或新增依赖。
- M1 可下载/校验/预处理真实分片和执行真实单层验证；M1 通过前不执行完整 35B 模型训练。该解释消解 v2 G0.8/P1/P2 与“M1 后碰真实模型”的矛盾，遵循交接说明 6.2。
- 同步 store→P2→同步 tiny→异步 streamer→受影响项回归，M1 仍须包含异步验收。
- 优先标准 AdamW(weight_decay=0, foreach=False)，接受历史动量更新本步未命中专家。
- BF16 GEMM 输出及舍入、非专家解码、FP32/TF32 和 BF16 reduction 策略显式记录。单靠同 checkpoint 不声称与 FreeToken 整体 E2 等价。

## 2026-10-27：P0 实测与格式层准备

- GDN 的现成纯 Torch chunk 实现已通过独立短序列递推对照、真实 T4096 全输入梯度，以及完整单 GDN+PEFT 模块的两种检查点。先选这一已验收候选，不新增未安装 FLA 依赖；未测试 FLA 不等于 FLA 不可用。
- E2 比较采用 contiguous 原始 uint8 字节，避免浮点 torch.equal 把正负零视为相同。
- 专家库同时保留原始 `input_scale`：index 中共有 30720 个；按 F32 共 122880 字节。四个字段全部保留，预计专家 payload 与 v2 的 18119639040 字节吻合，最终以真实 header 为准。
- 冻结非专家采用 FP32 解码再舍入 BF16；lm_head 按行分块。这是明确的训练基座数值策略，不声称与运行时 W8A8/FP16 global 完整 E2 等价。
- G0.8 下载期间准备格式解码/预处理代码及小格式夹具，不运行未校验的真实权重，也不提前宣布进入 P1 验收。真实预处理 CLI 要求完整三分片 PASS 收据，并重做来源哈希检查。

## 2026-10-27：真实 P1 首轮过滤修复

三个原始分片的 SHA256 已全部通过。首次真实预处理在布局检查阶段失败：范围过宽的 `.mlp.experts.` 选择同时计入 `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`，使源专家集合比文本主干多两个名字。已用含 MTP 的小夹具复现；将专家集合限定为 `model.language_model.layers.*`。MTP 张量仍由原始分片保留，不加载为文本训练专家。失败 run 和空输出 `*-train-v1` 保留，修复后使用新的 `*-train-v2` 输出。

## 2026-10-27：P2 与物化构建

- 先采用逐专家 GEMM 的纯 Torch 数值基线；grouped_mm 资格已通过，但暂不叠加实现复杂度。真实目标形状前向、路由及六个 A/B 梯度与独立参考通过原阈值。BF16 dX 相对误差约 0.005084；FP32 为 7.3e-8，未调整容差。
- 生产按 route 唯一位置写回，再按 top-k 的原槽顺序用 FP32 累加。局部 A/B 叶子只覆盖本块，不用 allow_unused 隐藏 None，不直接写 `.grad`。只支持一阶训练梯度。
- CPU bank 的逻辑字节不是锁页 allocator 的物理分配量：真实构建时约 18.12GB 逻辑数据对应约 21.47GB pinned allocated，进程 peak working set 约 27.04GB。仍在本机主存预算内；后续按实测账本计算。
- meta 模型先转非专家 BF16，再替换专家并分配。`to(BF16)` 也会影响 meta RoPE buffer 的 dtype；实际加载后用正常 CPU 构造重新注册 FP32 inv_freq/original_inv_freq，不能只 copy 到 BF16 buffer。
- 专家 A 使用每个专家自己的 1/sqrt(fan_in) 范围，禁止对整个 [E,r,in] 套三维 Kaiming。PEFT 包装后显式恢复自有 A/B 的可训练性，并与预期 canonical 集合核对。
- 仅 tiny 参考模型允许常驻解码 W。真实 A1.4 只做构建和参数读回，没有运行完整模型前向或训练；M1 仍等待 P4 与 P3。

## 2026-10-27：A5.2 的离散路由与参考判据

真实首条128-token提示的无条件CPU FP32前四层对照误差为0.90%、2.05%、1.77%、2.46%，原1%判据未通过，失败run保留。五组控制实验发现：固定GPU实际路由但仍以CPU FP32端到端计算时误差仅0.33%、0.40%、0.42%、0.54%；相同逐层输入、固定路由时约0.29%～0.33%。CPU使用相同BF16计算策略也会因路由选择分叉而产生最高2.74%端到端差异。同路由token与不同路由token误差分布已分开记录，不将无条件失败改写成通过。

v2第6.2节明确容差是起点，允许依据实测和原理调整。这里不更改生产模型的BF16路由、不放宽专家数学/梯度阈值；将A5.2拆分为以下检查，并在尚未测量的三条新128-token提示上验证：

- 连续数值路径：CPU FP32逐层流式计算，采用GPU的实际离散route及其权重，仍要求每层端到端相对L2≤1%。
- 独立路由核对：用GPU实际归一化输入和磁盘BF16 router权重，在CPU做FP32投影，要求logits相对L2≤1%；从实际GPU logits检查所选集合确为top-k（相等分数允许任意tie次序），路由权重与独立CPU softmax/归一化相对L2≤1%。
- 保留未固定路由的完整CPU FP32误差、选路变化个数及cutoff margin，不能把调整后的条件对照写成原无条件1%通过。

依据是top-k在cutoff处不连续：微小投影舍入或相等分数的CPU/GPU tie次序可替换第8个专家，产生有限输出跳变。只比较两条不同路由图的最终残差，无法单独验收反量化/矩阵乘/GDN。拆分后分别检验数值图和路由选择，原始自由路由结果继续作为跨精度诊断。若新的独立提示不能满足拆分判据，继续诊断，不自动放宽。

## dense 27B

### 2026-10-29：D0 / DA0.1 原文封存和分支来源

- 设计：0.4、4、D0 / DA0.1；用户要求首个提交保存设计原文、交接提示及 SHA256。
- 实测：目标 `siltok-ai/nvfp4lora` 和旧 checkout 的 `wenlive/nvfp4lora` main 均为 `efdfc55a26d9531c904470a55ab35793004ed104`。旧 checkout clean，原 editable 环境仍绑定旧路径。
- 决定：独立 clone 为 `C:/workspace/wt/T-20261029-nvfp4-dense/nvfp4lora`；其 origin 指向 `git@github.com:siltok-ai/nvfp4lora.git`，从已 fetch 的 origin/main 建 `feat/dense27b-lora`。不更改旧 checkout 的分支或 remotes。
- 原文 SHA256：`docs/reference/dense27b-design-v1.md` = `81fc79306ae0458b4a28f3aa7a5ee5148d202a62a426fdfd6c07b5182e9f95a8`（桌面原字节）；`docs/reference/dense27b-handoff-v1.md` = `8937c60abc0da98a67faaa2cd03aa3dacd6c335c70e3e3df03d0017227f34778`（本次用户交接提示，UTF-8/LF，末尾换行）。
- 影响范围：仅文档封存与 Git 属性；第4节清单外新增 `.gitattributes` 两条 `-text`，用于保证原文字节不会被行尾转换。不修改数值代码；B0 在未改动的 main 上记录。
- 硬约束：遵守，无变更。用户确认：原交接明确授权创建分支、独立环境和首个提交；没有请求额外确认。

### 2026-10-29：用户新增 FreeToken 推理交付范围

- 设计原文：0.1 / G2-D、9 将 FreeToken/Speed dense LoRA 推理排除在训练分支范围之外。
- 实测：训练方案估算的30.4GB推理占用尚未在新 fork 验证；不能当作本机成功推理的证据。
- 决定：按用户最新指令，另从 `wenlive/FreeToken` 最新 main 建独立特性分支，参考已有 `feat/qwen36-nvfp4-lora`，适配并独立记录验收；训练分支硬约束与 DA 编号不变。完整模型推理和局部算子验收分别判定。
- 影响范围：扩展 G2 交付，单独仓库、分支、环境和证据，不改设计原文。
- 硬约束：不变更训练硬约束。用户已确认，原话：“最终实际带lora推理可以参考一下 https://github.com/wenlive/FreeToken 这个我自己的fork仓库 此前有feat/qwen36-nvfp4-lora 这么一个分支来做moe模型带lora推理 你可以基于这个仓库main搞个新分支来做适配和验收 （也要推送到我的freetoken fork上以独立分支）”。

### 2026-10-29：D0 / DA0.2、DA0.3 独立环境和MoE基线

- 设计：0.4.2、0.4.7、DA0.2、DA0.3要求独立锁定环境及改代码前main基线。
- 实测：新venv从原39个离线wheel用原锁、`--require-hashes`安装；逐包版本及每个wheel SHA256一致，pip check通过，pip freeze仅editable注释/URL不同。旧main及editable绑定未动。基线main单测41/41、MoE tiny100步/新进程50→100/发布重载全部PASS；真实C3、expert-r2初始probe均为FP32 bits `1083229253`，loss `4.523958683013916`。
- 实测配置哈希：real-c3=`e58cf9acbedd898b0d584601775997ebcfa3afba11321f16f12490c1e2453e5d`；real-expert-r2=`2f8da3e3b5e4fd0751d14a5aadeb8503261a3e01b8acce4476b4c82db0c82d4a`。实际配置JSON、完整metrics、终点参数及运行快照在本任务B0 runs保留；不是D7回归结论。
- 决定：新venv登记为`preparing_dense_d0`，不提前标为真实训练合格。D7逐项对照B0。
- 影响范围：环境登记、新的D0证据；旧代码和旧环境无数值改动。
- 硬约束：遵守，无变更。用户确认：原交接及最新原话“缺环境 权重就自己配置就好 无需我同意”；未请求重复确认。

### 2026-10-29：D0工具清单与payload标量计数

- 设计：第4节没有列出D0专用资格入口；1.3节矩阵payload合计16,840,130,560字节。
- 实测：加上MLP/FP8的FP32标量，64层申请总量为16,840,133,760字节，差3,200字节；每层仍小于256MiB。实际64块分配为17,179,869,184字节（16GiB），全遍H2D约51.341GB/s。
- 决定：新增清单外`tools/check_dense_preflight.py`，独立运行meta、GDN、SDPA、GEMM/解码和pinned关卡；不修改原工具或MoE数值代码。按真实标量字节预算和记录。独立记录小型辅助脚本放仓外task目录，其源码副本随run保存。
- 初次pinned测量保留FAIL：分配64块已完成，但进程尚未初始化CUDA，host_memory_stats返回空字典，读取统计键报KeyError；修正测量入口显式cuda.init后新run通过。这不是分配失败或32GiB取整，不将首轮FAIL改写为PASS。
- 影响范围：D0资格工具和资源账本；不增加bf16小张量到层文件。
- 硬约束：遵守，无变更。用户确认：一般工程实施已授权，无需新增确认。

### 2026-10-29：DA0.7 原始HF GQA分派失败与dense专用回退

- 设计：DA0.7要求B1、24/4头、T4096、head_dim256在有/无padding mask时不退回math；第5.1节据此估算注意力预算。
- 实测：默认HF无mask使用enable_gqa=True，实际`aten::_scaled_dot_product_attention_math`，峰值allocated6,979,321,856字节，原始关卡FAIL。诊断12种组合：该Windows锁定Torch未编译Flash；确定性模式拒绝cuDNN SDPA；已有efficient实现要求q/k/v头数相同。显式repeat_kv后有/无mask均实际使用efficient前后向，峰值allocated分别659,292,160/525,074,432字节；重复前后向E2，math参考输出相对L2约0.214%～0.215%，q梯度约0.327%～0.328%，全部输出/梯度小于1%。
- 决定：dense构建采用单独注册的attention和mask分派，先显式repeat_kv再调用现有HF SDPA入口，限定dense配置；保持原MoE配置、全局sdpa实现及确定性设置不变。该候选资格通过不等于默认HF关卡通过，也不代替D4/D5的整模验证。
- 影响范围：D0回退资格及后续`build_dense.py`；不新增依赖，不新增kernel，不更改PackedLinearFn、lm_head或LoRA数值策略。
- 硬约束：不涉及硬约束变更；仍为原生Windows、原锁、确定性SDPA、按family隔离。用户确认：非硬约束工程选择，按“遇到问题尽量解决”授权执行；未声称用户单独批准某个新数值策略。

### 2026-10-29：DA0.4 来源校验、传输重试和完整收据

- 设计：1.2、0.4.1、DA0.4要求固定revision及三份完整分片SHA256匹配HF LFS。
- 实测：选择部署候选RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30。三分片完整SHA256依次为`fbcdb5ba1cdda462b5f38592d071e772c4d398afea61a0aa9188b32d1a239a79`、`db6146a5464fb0a891181b93c81593f0ca65c602eb14120a1c2b1b09bca11f85`、`d3cfb92742e30c8b46564665791dbe0a86ed64cfc02b1275081530793c0c9581`，均与固定revision的可信LFS元数据一致。总文件字节21,921,697,280；设计中21,921,428,072是tensor payload，不含文件头。
- 决定：原HF Xet和整片HTTP传输较慢，改用仓外32MiB分段工具，校验Content-Range、逐段字节数并持久化分段摘要。先8线程，再从已有226段恢复32线程；剩余653/654时仅重试末段。达到请求字节数即结束读取，再做完整文件SHA，不依赖连接EOF。全部完整SHA通过后，再以已参数化的仓库fetch_checkpoint入口取得全量PASS收据。原中断run均保留FAIL/真实退出码，不视为来源验收通过，也没有更换基座或哈希。
- 影响范围：仅下载运输、仓外可恢复工具和D0证据；预处理/数值策略无变更，无新增依赖。未在未验SHA的真实dense权重上计算。
- 硬约束：遵守，无变更。用户确认：已明确授权自行配置权重、使用本机资源；无需额外确认。

### 2026-10-29：DA0.6完整GDN补测与速度预算

- 设计：DA0.6及5.1/5.3估算GDN显存和训练速度。
- 实测：完整HF GDN含因果卷积、bf16 beta、norm和冻结合成bf16投影，在T2048/T4096与两种checkpoint下输出/dX均E2；T4096最高allocated4,042,380,800字节、reserved4,441,767,936字节，前后向/重算约0.396～0.440s。独立bf16 beta短递推最坏相对L2约6.922e-5。它不是带LoRA/packed投影的整模峰值。
- 决定：补齐清单外D0工具的完整模块关卡；将各范围分别报告。根据大GEMM、解码与GDN测量暂以2048约12～18s/micro、4096约25～35s/micro规划（均为推算），不把GPU事件或独立峰值相加当实测整模性能；见docs/dense/d0-budget.md。D6用真实值替换，原超过两倍且原因不明的停止规则不变。
- 影响范围：资格覆盖及规划预算；不改模型数学、LoRA范围/rank或lm_head策略。
- 硬约束：无变更。用户确认：原方案允许按实测修正规划估计；用户明确“无时间限制可持续推进”，无需新增确认。

### 2026-10-29：D1最小store接口、独立读回与lm_head策略

- 设计：第4节列出ExpertStore的layout_version参数；DA1.4已经要求store加载后哈希可复现，而完整streamer关卡在D3。
- 实测：tiny dense生成、预处理及四项格式/拒绝测试通过；按默认构造的MoE store拒绝dense布局，显式传`packed-dense-layers-v1`后全字段原字节和重载哈希一致。全部8个MTP/visual假键被排除，剩余非流式键集合完整。
- 决定：在D1提前加入最小layout_version构造参数（默认MoE原值不变），供DA1.4读回；D3竞争/异步等完整关卡仍按顺序执行。清单外新增`tools/check_dense_preprocessed.py`和`tests/test_dense_preprocess.py`，分别提供独立源→store全量E1/两次加载验证及来源、过滤、fail-closed回归。预处理和大型store检查在不同进程，分别报告内存，不能将store分配算作预处理峰值。
- lm_head：按设计3.1、第4节显式例外，使用现有`_decode_nvfp4_rows`和`nonexpert-fp32-decode-bf16-v1`离线生成bf16；MLP NVFP4及注意力/GDN FP8仅原字节分层保存。交接摘要的通用原字节表述以设计中更具体的lm_head安排为准；没有启用O3或更改其数值策略。
- 影响范围：D1格式、夹具、store最小接口与验收；不改MoE解码、构建、streamer或训练数学。原文不改。
- 硬约束：无变更。用户确认：既有实施授权与设计的明确安排，不请求额外确认。

### 2026-10-29：DA1.3可选FreeToken对照的依赖隔离

- 设计：DA1.3可选只读调用FreeToken；0.4.2禁止新增依赖。
- 实测：首轮参考工具在计算前因infer环境无psutil退出，未产生数值对照。psutil只是完整store检查所需的监测依赖，解码采样不需要；将其及训练格式模块导入移至完整检查路径后，原推理环境无需安装任何包即可执行36个真实样本。FP32原global对照最大0 ULP；运行时FP16 global另列最大1 ULP。
- 决定：保留首轮FAIL及新run，不给infer环境添加psutil，不修改锁文件。两种global精度结果分列，不能把FP16 runtime误差混入训练原FP32策略。
- 影响范围：清单外D1检查入口的可选参考路径；无数值实现或依赖变更。
- 硬约束：遵守。用户确认：无需新增确认，实际未新增依赖。

### 2026-10-29：D2 / DA2.3 的FP32参考控制（保留原1e-6失败）

- 设计：3.3的BF16生产数值策略不变；DA2.3要求测试FP32相对FP64的L2≤1e-6。
- 实测：真实down_proj、K=17408、17行输入的前向误差为1.0246188661070007e-6，原判据FAIL；dx为4.442192395720577e-7。用整张独立解码矩阵和普通Torch autograd重算，FP32前向/dx均与PackedLinear逐字节一致。相同17行输入仅零补齐为32/128/4096行，前向误差分别1.0246e-6/8.6455e-7/1.3309e-6；4096行dx为1.2220e-6，说明是现有GEMM按M/K选择算法后的累加舍入，不是自定义反向或码值解码误差。原始失败和诊断run保留，不能宣称真实大形状原1e-6全部通过。
- 对照实验：在调整验收前固定三个新seed（20261030、20261031、20261101），对down/gate/o三种长归约形状和17/128/257/4096行共36组测试。每组FP32、FP64都和普通autograd E2；相对FP64最坏L2分别1.3195891e-6、1.3194406e-6、1.2607241e-6。小形状NVFP4/FP8仍满足原1e-6，并通过FP64 gradcheck。
- 决定：拆分DA2.3：小形状原1e-6不变；真实大形状要求FP32与独立物化普通autograd E2，且相对FP64≤2e-6。2e-6只为实际大归约的FP32库舍入留有限余量，不替代E2实现核对；每个结果同时保存原1e-6是否通过。该项汇总标为QUALIFIED_WITH_DOCUMENTED_REFERENCE_CONTROL，原真实形状1e-6失败单列，不写成原判据PASS。若E2或2e-6任一失败，继续诊断。
- 影响范围：DA2.3测试判据和独立控制；不更改PackedLinearFn表达式、解码精度、B=0、BF16生产路径、确定性、依赖或任何MoE代码。没有启用TF32。清单外增加tests/dense_fixture.py与tests/test_packed_linear.py作为独立格式/梯度夹具。
- 硬约束：无变更。用户确认：未声称用户单独确认该判据；根据原交接“判据调整要有原理依据和对照实验”及“遇到问题尽量解决”的一般授权执行，保留并汇报偏离。

### 2026-10-29：D2 / DA2.4 工作区测量口径

- 设计：DA2.4要求解码矩阵加分块中间量上界，5.1把激活与解码空间分项预算。
- 实测：T4096的gate/up完整前反向峰值增量包含FP32输出缩放/激活梯度临时量，不能全部算作解码工作区。工具同时报告完整调用峰值、物化同表达式autograd控制峰值、直接解码峰值、返回后live bytes和ctx保存张量数，另保留真实gate内存快照。参考的常驻W在其测量基线中，生产侧额外增量据此与矩阵+块上界比较；直接解码测量防止激活峰值掩盖解码峰值。
- 决定：完整峰值始终保留，不把差分当作整模显存；DA2.4同时检查直接解码及对照额外工作区≤矩阵+每块20字节/元素+4096字节，上界来自当前Torch解码的U8码、int64索引和FP32临时量。生产BF16前向返回后只允许输出存活，反向后只允许输出/dx存活（allocator对齐容差256KiB），ctx不得保存张量。内存快照留仓外，仅hash和摘要入仓。
- 测量修复：首轮独立FP32/FP64控制复用了BF16 live-byte断言，首次cuBLAS FP64调用增加约32MiB库工作区导致退出，尚未完成控制。将live-byte关卡限定在生产BF16；FP32/FP64仅用于独立数值控制。不放宽生产BF16存活量断言，原失败run保留。
- 影响范围：D2测量和证据；没有改变模型、解码块大小或MoE实现。
- 硬约束：无变更。用户确认：一般工程和证据口径修复已授权，不请求额外确认。

### 2026-10-29：D3验收入口与逐层统计

- 设计：D3要求把原store/streamer测试参数化用于dense；第4节只列store的layout参数，未列dense专用D3入口。
- 实测：tiny先通过后，真实64层完成1000次随机完整GPU→CPU字节对照，累计263,169,033,008字节；所有层均覆盖。人为延迟的正反顺序128次读取及128次完全错误预取都与源相等，两个过程完整摘要一致。两种层的13个投影（覆盖10种形状）在T4096下同步、异步无预取、异步错误预取的输出/dx/dA/dB均E2；冻结store哈希不变。
- 决定：清单外新增tools/check_dense_streaming.py，复用原同步/异步实现，只在验证器子类记录逐层acquire/命中/拷贝字节/等待事件，不修改生产streamer。错预取偏移真实64层为17，tiny四层为2，保证始终不是当前/下一层。D1已加入的layout参数保持MoE原默认。
- 修复记录：首轮tiny在统计断言失败，原streamer会计入发现全部槽被pin之前的acquire请求，验证器只计成功请求；改为同步记录实际计数增量及failed_acquires后tiny通过。句柄/槽保护本身未失败，原失败run保留。没有修改调度或放宽数值判据。
- 资源与口径：真实锁页物理17,179,869,184字节（16GiB），working-set峰值19,325,108,224字节；GPU峰值allocated2,934,964,224/reserved3,370,123,264字节。1000次完整回读、CPU逐字节比对及SHA共264.84s，含双向传输和验证器开销，不代表训练速度或纯H2D带宽。同run延迟读场景H2D事件合计约43.93GB/s，错误预取约47.61GB/s；全程和逐层原始计时分列。真实训练耗时仍待D5/D6，不用验证器吞吐替代。
- 影响范围：D3验证器、逐层统计和证据；ExpertStore实际加载相同原字节，streamer与MoE数值文件不改，无依赖变更。
- 硬约束：遵守，无变更。用户确认：一般工程和清单外验收入口已获实施授权，未请求额外确认。

### 2026-10-29：D4 CPU embedding冻结指纹与构建边界

- 设计：3.1、DA4.1要求CPU embedding为普通属性，但现有train.loop.frozen_fingerprint只扫描Parameter/buffer和store.hashes；第4节要求不改MoE loop/build。
- 实测：tiny整模的B=0开关loss E2，同表达式非零LoRA前向/全部梯度E2；分别修改CPU表、raw层字节、GPU冻结norm，指纹都能检测变化。真实构建实际得到800个FP32 CUDA LoRA张量、108,789,760元素，400个无参数PackedLinear及400个LoRA旁路，全部B严格为零，无meta残留，RoPE为FP32。没有进行真实前向、反向或optimizer step。
- 决定：在新build_dense.py内用DenseLayerStore子类复用ExpertStore，hashes加入当前CPUEmbedding表并把layer键统一为字符串，直接接入原冻结检查；每个模型独立绑定表，拒绝复用已绑定store，tiny参考用自己的store。新增dense attention/mask registry键，不覆盖原sdpa。核心生产LoRA不通过PEFT注入。
- 影响范围：新dense构建文件，以及共享adapters.py中增加LoRAPackedLinear开关类型；原MoE build/loop/experts_fn、checkpoint、data/template/generate数学未改。清单外新增tests/test_dense_build.py覆盖构建、冻结、梯度、发布拒绝和逻辑配置；仓外真实构建验证器源码随run保存。
- 硬约束：遵守。用户确认：实现方式属于已授权工程范围；embedding仍在CPU，bf16小张量仍独立文件，数值策略不变，无需额外确认。

### 2026-10-29：D4 / 3.5的D-ATTN张量数勘误

- 设计原文：3.5列D-ATTN为352个张量、39,583,744元素；D-ALL为800个、108,789,760元素。
- 实测：16个全注意力层各4投影，48个GDN层各3投影，共208个投影、每个A/B两个张量，D-ATTN应为416个。其39,583,744元素正确。默认D-ALL的800个及108,789,760元素同时经独立形状计算和真实构建确认。
- 决定：保留原设计，修正新增验收中D-ATTN预期为416。初次构建单测照抄352时失败，原run保留，修正后56项全部单测通过。首期实际配置仍为D-ALL r16，没有缩小LoRA范围或rank。
- 影响范围：文档勘误和备选范围的形状核对；不改变默认训练、发布语义或参数预算。
- 硬约束：不涉及变更。用户确认：未声称用户另行批准D-ATTN训练；本项仅事实勘误，不启用备选方案。

### 2026-10-29：D4真实锁页统计的1字节辅助分配

- 设计：DA0.9及停止点关注64层被取整成512MiB或分配失败；DA4.1真实构建记录资源。
- 实测：首轮模型所有构建检查均通过，但验证器在GPU布尔值读回后要求整个host allocator恰为16GiB，实际为17,179,869,185字节，因多1字节FAIL。新进程分开记录：构建后64次host分配、17,179,869,184字节；执行CUDA布尔标量验证后65次分配、17,179,869,185字节。由验证阶段新增的1字节缓存造成，不是层bank取整改变或分配失败。
- 决定：在构建前后隔离测量bank增量，保留验证后的完整统计和首轮FAIL。层bank仍恰好16GiB。真实构建峰值allocated约3.576GB、reserved约3.586GB；GPU frozen参数2,595,273,728字节、LoRA435,159,040字节、双槽531,497,056字节、CPU embedding2,542,796,800字节。它不是完整训练显存。
- 影响范围：仓外测量器及资源证据；生产实现、allocator、依赖未改。
- 硬约束：遵守，无变更。用户确认：测量口径修复已授权，无需额外确认。

### 2026-10-29：D4发布、两个参考及MoE中途回归

- 设计：DA4.2–DA4.6及第4节要求两个100步参考、50→100十项状态、发布重建、PEFT兼容和MoE位级中途回归。
- 实测：tiny100步与同表达式参考loss全程E2；独立HF bf16预乘global+标准PEFT参考首步最小梯度余弦0.9995077116（最大相对L2为3.16625%，只作观测，设计首步判据是余弦≥0.999），每步loss最大相对误差0.0251278%，最终最小权重余弦0.9998401944。新进程50→100十项完整状态E2；发布重建所有参数、冻结指纹、固定loss及25个DeltaW指纹E2；标准PEFT加载后的probe相对误差0.0110244%。adapter开/关probe loss分别7.4265113/7.6956329，均为tiny结果，不是27B功能结论。
- 决定：artifacts.py按新layout分派，dense只发布标准PEFT A/B文件及配置，不写MoE专家文件；校验family、目标集合、rank/alpha/scaling、基座和逐文件hash后再复制参数。原main的v1读取器在独立旧环境中明确拒绝新layout；新load_adapter也拒绝把dense装进MoE/未知family。session.py仅对显式dense family新增独立logical_config/build/无路由audit/profile，MoE默认分支保持原值。
- MoE回归：56项单测通过；新分支MoE两步metrics与原B0100步的前两步E2。B0原run未保存step2参数，因此在未改动的main上补做两步，终点参数/Adam/scheduler/RNG/data/trainer/config/probe/DeltaW与新分支两步全E2。补充run不替换原B0、不冒称代码改动前已记录step2权重。两套真实logical_config及e58cf9.../2f8da3...哈希仍完全一致。D7完整100步、真实首步和已交付checkpoint身份回归仍待完成。
- 影响范围：设计清单内dense tiny工具、configs和共享family/layout分派；只修正D-ATTN计数和测量器，不调整DA4阈值。不新增依赖。
- 硬约束：遵守。用户确认：常规工程已授权；DA6.3功能门槛尚未确认，未开始真实27B训练。

### 2026-10-29：D5独立CPU参考、tiny诊断与原真实1%判据

- 设计：DA5.2要求128-token真实提示、无条件CPU FP32前四层各≤1%，GDN独立逐token、attention eager；第4节允许新cpu_layers_dense.py，不修改MoE参考。
- 实测：真实层0/1/2/3误差依次0.291637%、0.294074%、0.406958%、0.339739%，原判据全部通过。GPU完整64层有限；重复前向隐藏状态按U8逐字节E2；实际16个全注意力调用均为既有efficient SDPA，冻结指纹不变。
- tiny诊断：最初额外把真实1%门槛施加到65-token随机tiny，误差依次0.532428%、0.769469%、0.960096%、1.191991%，最后一层FAIL，原run保留。控制实验中，同一独立CPU递推与普通HF CPU FP32分块GDN四层端到端最大差6.379013e-7；BF16对普通HF FP32同样为1.191991%，说明该随机tiny的跨精度累积不来自递推算法错误。未通过修改生产精度来迎合tiny。
- 决定：新增参考算法单测改为检验独立FP32递推对普通FP32分块的≤1e-6数学控制，继续打印tiny BF16误差，明确原tiny1%仍FAIL。真实DA5.2仍使用原始无条件1%门槛、没有放宽或teacher输入。CPU逐层只保留当前层FP32矩阵，按行限定解码中间量；复用原CPURecurrentGDN的无路由数学，不改MoE代码。57项完整单测通过。
- 影响范围：新dense CPU参考/前向工具和已有dense构建测试；不改PackedLinear、HF前向、量化格式、依赖或真实验收判据。
- 硬约束：遵守。用户确认：一般诊断/工程修复已授权，真实原门槛通过，无需额外确认。

### 2026-10-29：D5真实前向、FP8范围与计时口径

- 设计：DA5.1、DA5.3–DA5.6要求资源、短回答、单次前向拆分、原FP8 input_scale观测和200条Speed模板黄金样本。
- 实测：加载allocated3,573,386,240/reserved3,586,129,920字节；128-token带捕获/范围观测前向峰值allocated3,906,082,304/reserved3,961,520,128字节，整卡剩余11,214MiB。前向观察时working set21.773GB，进程private峰值约30.61GB，均非训练峰值。三条非思考greedy分别输出2、北京、Hello，每条下一token就是im_end（248046），不再生成。200条使用原dense tokenizer/template的Speed提示、标签、截断/丢弃黄金全部通过；直接编译engine_bridge.py当前_prompt_tokens AST，函数SHA725a75...，保持与Speed相同加载选项。
- FP8：基座单条128-token提示覆盖全部208个来源投影，3个比值>1，最大1.022831（51层o_proj、62层in_proj_qkv/in_proj_z）。这是完成观测，不是“无截断”通过；D6.8还需训练后语料观测，不模拟W8A8或修改训练数学。
- 计时：真实128-token仪器化CUDA区间，投影GEMM99.78ms、解码1218.13ms、H2D332.21ms、acquire等待8.44ms；GDN含投影/解码总区间543.65ms，扣除投影/解码后467.66ms；attention分别51.54/26.71ms。事件区间包括CPU入队空隙，H2D可与计算重叠，inclusive与exclusive口径分列，不能直接相加。整体仪器化wall包含CPU profiler启动/结束成本，不作纯计算或训练速度。原始CPU/GPU Tensor轨迹只留仓外，仓库保存摘要及sha。
- 决定：audit_fp8_input_ranges.py只增加显式dense family路径，复用PackedLinear输入hook和原bank FP32 scale；dense可无adapter观察基座，有adapter时双相位。MoE现有参数命令和数学分支不改。训练/推理regex警告按原样保留，没有擅自启用fix_mistral_regex改变Speed token协议。
- 影响范围：设计清单内前向工具及FP8观测分派、证据；真实训练、FP8模拟或性能kernel均未加入。
- 硬约束：遵守。用户确认：一般工程已授权，DA6.3尚未确认，仍未进行真实训练。

### 2026-10-29：D6资格工具、真实梯度和压力测量

- 设计：DA6.0–DA6.2、第4节及5.1/5.3；第4节未单列真实梯度捕获和dense逐层重算的独立入口。
- 实测：旧canary300/holdout50文本SHA逐字节相同，replay300由当前dense基座生成；基座留出0/50、回放981监督tokens的NLL为0.00023400843633333113。真实1步×acc8的B=0开关loss均为4.820327281951904、FP32bits1083850783；全部B首步梯度非零。更新后probe4.715414047241211，禁用恢复原loss位级相同，冻结不变。两种层共13投影的真实激活/dY对照，同表达式output/dx/dA/dB E2，FP32参考forward<=1%、梯度<=2%且cos>=.999；八次累积A/B与实际训练梯度E2。首层冻结输入的qkv/z没有实际dx边，明确由隔离replay检查，不能声称捕获了不存在的dx。
- 决定：清单外新增tools/check_dense_training.py、tools/profile_dense_recompute_memory.py；后者直接复用原RecomputePeaks，不改MoE profiler。audit_snapshot.py仅新增显式dense检测，把decode_codes内FP8 cast和NVFP4预分配矩阵纳入，原MoE默认路径不变。仓外baseline、整卡监控和证据收集器保存自身源码；只把摘要和manifest入仓。prepare_canary_replay.py新增family分派及dense自身EOS/pad，旧MoE默认seed/EOS/pad不变。verify_training_release.py增加dense布局读取分派，完整真实DA6.6仍待正式训练。
- 资源：2048两步×acc8为129.03/128.47s，allocated9,005,052,416B、reserved9,483,321,344B、整卡最低剩余5938MiB；4096为284.10/284.72s，allocated12,996,858,368B、reserved13,398,704,128B、整卡最低剩余2204MiB。整卡每秒采样可能漏过亚秒瞬态，Torch峰值另存。2048/4096 steady-state各恢复真实step2 Adam（870,318,080B）、64层重算、完整history、loss/全部800梯度对无仪器U8 E2、冻结不变；首micro及重算快照均证明反向后无decoded分配存活。
- 计时决定：用实测替换估算；2048约16.1s/micro、4096约35.6s/micro，接近D0修订预算12–18/25–35。相对原5.3上沿没有超过两倍；保留原估算与实测差异，不把资格step当正式训练性能。profile的单micro峰值较完整acc8小约435MB，是初始尚无累积grad，不能替代压力峰值。4096剩余>1GiB，无需O1；未启用O3/O4，也未改lm_head、rank、范围或数值策略。
- 影响范围：D6资格和family加法，未新增依赖、未改变生产PackedLinearFn/训练循环/MoE数学。57项单测通过。首步max1检查点是资格产物，不延长配置为正式100步；压力logical max100但stop-at2，明确不是功能100步。
- 硬约束：遵守，无变更。用户确认：一般工程已授权；DA6.3尚待明确确认，未开始正式100步。证据见docs/evidence/dense/d6-preflight-summary.json。

### 2026-10-29：DA6.3功能门槛确认请求（待答）

- 设计：DA6.3要求“确认后再跑正式100步”，并记录确认原话和时间；不能继承旧MoE门槛。
- 实测：基座留出0/50，回放NLL0.00023400843633333113 nat/token；2048/4096压力均通过，4096整卡最低剩余2204MiB。
- 决定：2026-10-29约13:35 UTC向用户提出新门槛确认：推荐留出>=45/50且回放NLL增量<=0.01（最终<=0.010234008436333332）；另提供>=48/50且增量<=0.005。此时尚无用户答复，推荐选项不是批准，未运行DA6.4正式100步。等待期间继续用户独立授权的FreeToken推理适配。
- 影响范围：正式功能验收准入，其他已通过资格不变。
- 硬约束：遵守必停。用户确认：**尚未收到，不得写成已确认**；收到后另记原话、时间和最终数值。

### 2026-10-29：用户新增FreeToken范围的真实整模资格

- 设计原文：0.1/G2-D及9将FreeToken/Speed推理排除；用户随后明确新增自己的FreeToken fork推理适配和验收范围，已在本日志前文保留确认原话。训练方案和DA门槛仍不变。
- 实测：独立FreeToken分支在原生Windows、原独立推理环境上已加载全部64层和首步资格adapter。来源为同一RadixArk固定revision及三SHA，新的dense读取器启动时校验全部原分片/metadata/tokenizer及两个PEFT文件；旧read_adapter仍拒绝dense。400源投影/800张量映射256融合op，FP32源435,159,040B、融合驻留657,457,152B。64层finalized packed bytes共16,903,096,320B，physical pinned16GiB+29B辅助缓存；单GPU槽267,000,512B。
- 决定：FreeToken只增加显式eager流式选项，沿用其CPU checkpoint reader和每层quant finalizer，按compute stream顺序复用单槽；不引入训练库依赖或新kernel。共享生成RoPE缓存排除在层bank外，保留引擎既有embedding/head策略；**训练CPU embedding、FP32 global、PackedLinear及lm_head策略完全不改**。推理仍为FreeToken既有NVFP4 W4A16、FP8 W8A8和FP16 runtime global，不宣称与训练BF16整模E2。
- 验收：三条greedy2/北京/Hello；zeroB对disabled完整首步词表logits E2；重复开关E2；非零adapter相对logit效应0.06536/0.07307/0.05590。新进程全部3提示×5模式的答案和logits E2，64 bank hash不变。54项源码LoRA测试通过，包含40项原测试。首轮harness把disabled_repeat仍按enabled执行而FAIL，捕获Tensor证明与enabled E2；只修正测试mode选择后新run通过，失败保留。真实初次通过peak allocated5,199,308,288B/reserved5,414,846,464B，重载监控minimum free9838MiB。设计30.4GB是既有驻留Speed路径的估算，不能用于否定另一个实际流式引擎，也不能将本次数字当Speed结果。
- 影响范围：仅用户fork `feat/dense27b-nvfp4-lora`，资格提交`b697fd9`，证据`docs/evidence/dense/streamed-real-qualification-summary.json`位于该fork。参考安装wheel不改，以显式PYTHONPATH测试新source。不是领域评测、完整吞吐或最终100步adapter推理验收；最终产物及Qwen3.6仍待做。两个分支均未推送。
- 硬约束：训练硬约束无变化；原有v1读取器fail-closed保留，无新增依赖，无WSL。用户确认：沿用前文逐字保存的新增FreeToken授权；DA6.3正式100步门槛**仍待明确答复**。

### 2026-10-30：DA6.3用户确认功能门槛并继续执行

- 设计原文：DA6.3要求用户确认本轮功能门槛后才能启动正式100步，记录确认原话和时间。
- 实测：此前资格基座留出0/50，回放NLL0.00023400843633333113 nat/token；2048/4096压力通过，4096整卡最低剩余2204MiB。原正式100步尚未启动。
- 决定：采用留出>=45/50（>=90%）、回放NLL相对同基座同语料增量<=0.01；按此前基线对应最终NLL<=0.010234008436333332。正式运行另记零适配器基线以核对一致性。以D-ALL r16、max_length2048、累积8、新100步会话继续，评测10/20/50/100，保存20/50/100；保留原精确续训和发布重建要求。
- 影响范围：DA6.3准入和本轮DA6.7功能判定；SDPA回退、DA2.3历史判据、生产数值策略和其他模型的门槛无变更。
- 硬约束：遵守第0.4.9，不改变任何硬约束。用户已明确确认，原话：“同意采用推荐选项 这个可以记录到文档中；可以开始继续工作了”。完整回复及门槛保存在docs/evidence/dense/effect-threshold-owner-decision.json。记录时间2026-10-30 02:06:34 UTC（北京时间10:06:34），为读取实际回复的记录时间，不冒称消息服务器时间戳。

### 2026-10-30：训练耗时拆分的独立测量入口

- 设计原文：DA6.9和交接最终汇报要求真实步耗时及H2D、解码、GEMM、GDN、注意力拆分；第4节未单列累积组的计时工具。
- 实测：正式第1步49.92s，功能语料与长序列压力范围不同；原D5只有128-token前向拆分，不能替代完整训练累积组。新工具尚待训练结束后先tiny、后真实控制验收，当前没有计时通过结论。
- 决定：新增清单外tools/profile_dense_training.py，只在独立进程恢复检查点，量一个原配置累积组的forward/backward，不更新optimizer；用无仪器同组loss/全部梯度E2控制验证计时器，分别报告CUDA区间、inclusive/recompute子集及H2D重叠限制。复用原CudaTimings和LoRATrainer，不改生产循环或算子数学。实际GPU验证安排在正式训练GPU空闲后。
- 影响范围：独立测量工具和资源证据，不作为新增训练步，不改变正式100步历史、采样或学习率。
- 硬约束：无变更；原生Windows、原依赖、原数值策略。用户确认：属于既有最终耗时汇报授权，不新增依赖或另行性能kernel；当前仅实现，未经实测的值不填写。

### 2026-10-30：按DA编号机械审核的独立dense入口

- 设计原文：沿用现有验收体系，最终逐DA编号汇总；第4节未单列dense审核工具。旧audit_m2包含两套MoE专用语义，不能据此宣称dense完整通过。
- 实测：新增入口的部分审核得到41 PASS、2 PASS_WITH_DOCUMENTED_CHANGE、11 PENDING；正式100、续训/发布、Qwen3.6、D7和最终FreeToken仍按实际缺失保留PENDING。严格--require-complete在该状态真实退出1；外部负向控制记录该退出码并确认拒绝原因为验收未完成，而不是把部分审核写成完整通过。
- 决定：清单外新增tools/audit_dense.py，直接复用audit_m2的Audit、Pending和十状态集合，独立检查dense语料、100步、20/50/100检查点、400个DeltaW、用户门槛来源及最终分支。原DA0.7默认失败和DA2.3真实大形状原1e-6失败继续保留，备选资格明确列为带记录调整。
- 影响范围：CPU证据审核和最终表格，未修改原MoE审核器或生产训练代码，不新增依赖、不启动任何GPU测试。后续实际运行证据到齐后再运行完整审核，当前INCOMPLETE不作全任务结论。
- 硬约束：无变更。用户确认：属于原有验收及如实报告授权，无额外门槛或确认要求。

### 2026-10-30：DA6.9核对整组SDPA调用次数

- 设计：DA0.7/DA6.9及当前efficient后端解释，需要核对真实整组前向、检查点重算、反向；只观察到至少一次efficient不足以证明全部注意力都走此路径。
- 实测：新增检查后，原CPU仪器化梯度对照1项/4个子组合仍通过（冻结输入是否requires_grad、是否非重入检查点）；它验证hook可移除且输出/梯度E2，不是GPU后端计数的实测。真实续训仍在运行，此处没有另跑GPU。
- 决定：仅在独立profile_dense_training.py中汇总实际efficient前向与反向算子数，并要求等于attention层数×累积数×重算次数。当前真实配置的预期为256次前向（含重算）/128次反向；这些是待核对预期，尚不是实测值。保留实际每个operator的计数，tiny GPU先通过才进入真实计时。
- 影响范围：验收工具和证据，不改变SDPA分派、模型forward、PackedLinearFn、依赖、训练循环或MoE。已完成正式100步的源码和数值不变。
- 硬约束：无变更。用户确认：属于已授权的运行证据核验，无额外确认要求；GPU计数仍按未验证记录。

### 2026-10-30：DA6.9计时器适配HF attention的tuple输出与keyword输入

- 设计：DA6.9完整反向拆分；不得把未完整测量的区间宣称完整通过。
- 实测：首次tiny GPU计时中loss/全部梯度E2、冻结不变、step未更新、efficient4次前向/2次反向全部通过，但模块反向区间漏记attention，整体FAIL保留。HF attention返回(hidden, weights) tuple，原hook仅处理Tensor；HF还以keyword传hidden_states，原hook未纳入该输入梯度的结束边界。
- 决定：只修计时器hook，提取tuple首个hidden并接收keyword参数，把实际输入梯度和可训练参数梯度都纳入反向结束条件。CPU测试改为混合GDN/attention两层并包含tuple/keyword：修复前四种输入/检查点组合全复现3!=4，修复后全通过；新tiny GPU全部检查通过，含4/2次efficient及全部模块反向区间。
- 影响范围：tools/profile_dense_training.py、已有tests/test_dense_profiling.py和证据。原FAIL不重写，真实计时尚未启动；正式100、精确续训及发布重建的生产源码完全不变，不需重训。
- 硬约束：无变更。用户确认：验收器的常规修复属于已有授权，没有更改PackedLinearFn/SDPA策略、依赖、lm_head或LoRA范围。

### 2026-10-30：MoE受保护源码的checkout行尾差异

- 设计：0.4/DA7.1要求MoE数值、配置和续训协议不变；源码保护用于证明未重构MoE，不能用源码哈希代替数值E2。
- 实测：检查15个受保护文件，14个Python文件的原始文件SHA与B0快照不同。独立读取原B0快照并验证其记录SHA后，确认差异全部是CRLF/LF；归一化换行后的字节完全相同，Git相对efdfc55的受保护文件diff为空。锁文件原始SHA也相同。
- 决定：源码保护同时保存B0/current原始SHA、CRLF数量、归一化文本SHA和Git blob不变证明。没有改写文件行尾或原B0快照。数值/梯度/状态/配置/权重/原文SHA继续使用原要求的精确比较，D7真实数值回归尚待执行，不据此预称通过。
- 影响范围：仓外源码诊断及最终D7证据收集器的文本身份检查，生产代码无变更。这不是数值判据调整。
- 硬约束：无变更。用户确认：只读诊断属于现有授权，无新增确认要求。

### 2026-10-30：Qwen3.6固定来源与原设计短10执行

- 设计：第6节要求Qwen3.8 DA6.6后再验收Qwen3.6的来源、D1/D5、own replay、机制、短10和发布重建；额外100需询问用户。
- 实测：官方0893e1606ff3d5f97a441f405d5fc541a6bdf404的3个完整分片全部匹配LFS SHA，再独立重hash复核。预处理/全E1/36个独立FreeToken样本、200模板黄金、无条件CPU FP32四层原1%均通过。原始source、helper、manifest及完整SHA见qwen36-summary.json。
- 决定：复用D0的bounded HTTP分段获取，仓外verify_local_dense_checkpoint.py对固定官方revision重新查询并复核完整文件，给原预处理入口提供完整receipt；无新依赖。原始短10按既有logical max100配置执行stop-at10，并保存1/5/10，允许将来在同一逻辑身份下继续；不把资格max1检查点延长或把10写成100。
- 实测训练：B=0 loss FP32bits1084881418相同、全部B有梯度、13投影参考通过；10步结束50/50、replay NLL6.259719674569478e-6，400DeltaW发布重建E2。Qwen3.6自己的原生推理仍在执行。
- 影响范围：独立来源/数据/证据和执行长度；核心生产库未新增模型特例，没有改变数值策略、lm_head、rank、范围、embedding位置或bf16小张量布局。
- 硬约束：无变更。用户确认：原设计已授权短10；Qwen3.8实际DA6.6后已实际询问可选额外100，截至本记录未收到新选择，未把沉默当作额外100批准。


### 2026-10-30：完整D7、最终原生推理与独立环境登记

- 设计：DA6.9、DA7.1/DA7.2及交接环境登记规则；第4节之外仅增加dense结果/CLI文档和既有environment登记说明，未重构MoE。
- 实测：真实整组计时完成，loss/全部800梯度对无仪器控制E2，256次efficient前向（含重算）/128次反向、零math。58项单测通过；MoE tiny100完整历史和状态对B0 E2、两真实初始probe FP32bits均1083229253、两个logical_config/hash不变。原交付两个step50检查点按原协议真实加载，参数/Adam/scheduler/全部RNG/sampler/trainer/route audit E2，未新增优化器步。原MoE读取器拒绝两份最终dense发布。
- 实测：FreeToken分别使用Q38实际100步和Q36实际10步adapter完成完整64层、各700条质量记录、zeroB/开关、两次缓存重建、64层bank不变和独立新进程全部答案/首logits E2；两模型留出均0/50→50/50，回放均300/300。54项源码LoRA测试通过。不是资格step1的替代报告。
- 决定：只把摘要/manifest入仓，完整制品按逐文件verified-copy归档，原部署权重以revision/三SHA引用。最终HEAD/remote-inclusive审核留在仓外交付回执，避免提交自身SHA的循环引用。Q36仍是原设计短10，未执行额外100。
- 环境：最终39个训练锁版本和两个pip check通过，原MoE editable与推理参考wheel绑定/依赖不变；只将本任务训练、推理两条登记更新为已完成技术资格。源码、依赖及训练数学无变更。
- 影响范围：D7证据、dense最终文档、README尾部新增入口及本任务环境登记。训练README原MoE字节前缀保持不变；实际制品和最终远端核对另记，未提前宣称已推送。
- 硬约束：无变更，原默认SDPA和DA2.3的失败证据继续保留。用户确认：沿用原训练/验收及新增FreeToken分支最终推送授权，无新门槛或依赖请求。

- 最终只读源码复核补充：README特性checkout为CRLF、原main checkout为LF；首次错误地跨checkout比较原始文件前缀而FAIL，失败manifest保留。按既有行尾诊断核对Git基线及特性checkout原字节前缀后PASS，原MoE内容无改动。生产src及锁文件与正式100步源码SHA全部一致；没有放宽任何数值判据。
