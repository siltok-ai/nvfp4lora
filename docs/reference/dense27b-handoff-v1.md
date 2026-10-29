在仓库 siltok-ai/nvfp4lora（https://github.com/siltok-ai/nvfp4lora）上新建特性分支 feat/dense27b-lora，让 dense 的 Qwen3.8-27B 和 Qwen3.6-27B，能在部署用的 NVFP4 checkpoint 上训练 LoRA。

运行环境：原生 Windows，RTX 5070 Ti 16 GB，64 GB 内存。
验收：沿用仓库现有的验收体系，证明训练流程正确：
梯度正确；
可以精确续训；
发布产物可以重建；
adapter 确实生效。
完成后：把分支推送到远端，并向用户汇报。
完整的技术方案、阶段划分和验收编号都在设计文档里。本提示词只讲工作方式、边界和必须停下来的情况。两者冲突时以设计文档为准，并把冲突报告给用户。

设计文档是方案起点，不是不可改的规格：

实测和文档不一致时，可以修正实现，前提是记录清楚并说明理由；
但第 0.4 节的硬约束不能自行改变。
背景（先读，不要重新争论已定事项）
已有成果：仓库 main 已经完成 Qwen3.6-35B-A3B（MoE）NVFP4 LoRA 训练，并交付了真实 adapter。
架构要点：
冻结的专家以原始 NVFP4 字节放在锁页内存，按层异步搬到 GPU，用时解码；
反向时重新取层、重新解码；
自定义 autograd 按顶层参数协议；
分块交叉熵；
精确续训和发布格式。
这套代码已经验收，是本任务的基础。
dense 和 MoE 的区别（设计文档第 2 节）：
dense 没有路由，所有层都是普通线性层，算子简单得多；
但每个 token 要算全部约 25.6B 参数，瓶颈是算力而不是搬运；
64 层打包后约 16.84 GB，放不下 16 GB 显存，所以必须按层流式。
用户已确定的范围：
只用原生 Windows，不用也不探索 WSL；
依赖版本和 Speed 对齐（就是 requirements/windows-cu130.lock），不新增依赖；
交付形态是「核心库 + 自带训练脚本」，不引入 LLaMA-Factory、DeepSpeed 或 bitsandbytes；
AdamW、weight_decay=0，接受 Adam 的默认动量语义；
首期是非思考、纯文本。
用户说过「暂不把 MoE 和 dense 锁定在一个框架里」。所以：
dense 代码放新文件；
共享文件只做按 family 分派的加法；
不重构 MoE 代码。
本任务不包括：
FreeToken 或 Speed 的 dense LoRA 推理：Speed 推理 27B 需要约 30.4 GB 显存，16 GB 卡上加载不了；
领域效果评测；
Triton 等性能 kernel；
序列打包；
FLA。
先读这些（按顺序）
设计文档全文。重点是：
第 0.4 节，硬约束；
第 3.3 节，PackedLinearFn 协议；
第 4 节，改动清单；
第 6 节，分阶段验收；
第 7 节，停止点。
仓库内：
README.md、ROADMAP.md；
docs/reference/design-v2.md：MoE 的主方案，E1/E2/E3 和 V1～V4 的定义出自这里；
docs/reference/handoff-v2.txt；
docs/decisions.md：特别是 2026-10-28 的门槛决定，和 A5.2 的 1% 判据记录；
docs/environment.md：venv 登记和不共用 editable 的规定；
docs/streaming.md、docs/training.md、docs/real-training-cli.md。
代码：
model/build.py、ops/experts_fn.py；
store/expert_store.py、store/streamer.py、store/async_streamer.py；
train/session.py、train/loop.py；
export/artifacts.py、export/delta_w.py；
format/nvfp4.py、format/fp8.py、format/preprocess.py；
tools/check_tiny_training.py、tools/run_recorded.py。
transformers 5.16.1 的 models/qwen3_5/modeling_qwen3_5.py：
Qwen3_5ForCausalLM、Qwen3_5TextModel；
Qwen3_5Attention、Qwen3_5GatedDeltaNet、Qwen3_5MLP。
读完后，先用不超过一页给用户回一份理解摘要，然后开始 D0。内容包括：

你准备新增和修改的文件；
你对 PackedLinearFn 数值策略的理解；
你认为设计文档里最可能和实测不符的三处。
不必等用户确认。

分支和提交规则
从 origin/main 的最新提交建分支 feat/dense27b-lora。写设计文档时 main 是 efdfc55；如果 main 已经更新，就用最新的提交，并在汇报里写明。
用单独的 checkout（或 git worktree），不要切换已登记 venv 所对应的主 checkout。
分支的第一个提交：把设计文档原样放到 docs/reference/dense27b-design-v1.md，把本提示词放到 docs/reference/dense27b-handoff-v1.md。
在 docs/decisions.md 记录两者的 sha256。
之后对设计的修改，写进决策日志或 docs/dense/ 下的新文件，不改原文。
提交要小：每个阶段至少一个提交，说明写清做了什么、对应哪些验收编号。
证据：运行证据按现有约定放进 docs/evidence/dense/，只放摘要和 manifest。大文件、权重、checkpoint 不进仓库。
推送：
只推 feat/dense27b-lora；
不推 main，不合并，不 force push，不改其他分支；
全部阶段完成（或用户叫停）后推送最终版本；
中途想推送备份，先问用户。
硬约束（摘要，全文见设计文档 0.4 节）
基座：训练基座就是部署的 NVFP4 checkpoint，按 revision 和三个分片的 sha256 绑定。
NVFP4 和 FP8 以原字节存放，只在用时解码；
不重新量化，不预先展开成全量 bf16。
环境：只用原生 Windows；依赖和锁文件一致；特性分支用单独的 venv（同一锁文件、--require-hashes）。
流式：64 层按层流式，反向时重新取层、重新解码，不为反向保存解码后的权重。
LoRA 初始化：B=0。第 0 步开 LoRA 和关 LoRA 的 loss 必须 fp32 位级相同。
模板：训练提示和 Speed 的推理提示逐 token 一致，用 dense checkpoint 自己的分词器和模板重跑黄金测试。
MoE 零回归：
MoE 配置的 logical_config 和哈希不变；
MoE tiny 训练和真实 MoE 首步 probe 与 main 位级一致；
已交付的 MoE 检查点仍能按原命令续训。
发布格式：dense 用新的 layout_version: "nvfp4-lora-dense-v1"，现有 MoE 加载器遇到它必须报错。
功能门槛：不继承 MoE 那一轮的门槛，正式 100 步之前请用户确认。
阶段顺序
阶段	内容	汇报
D0	分支、独立 venv、MoE 基线 B0、checkpoint 来源校验、GDN 和 SDPA 关卡、GEMM 和解码测速、锁页内存测试	D0 结束汇报一次：各关卡实测值；和设计文档估算的差异；是否需要调整预算
D1	dense 预处理（先 tiny，后真实）	合并到 D2 后汇报
D2	PackedLinearFn 和 LoRA 包装	
D3	层 store 和 streamer 复用	
D4	模型构建和 tiny 端到端（两个参考、续训、发布重载）	M1-D 汇报：D1～D4 的全部验收结果
D5	真实 27B 加载和前向（CPU fp32 参考、贪心回答、模板黄金测试）	合并到 D6 停止点汇报
D6	首步探针、压力测试 → 停止点（门槛确认） → 正式 100 步、续训、发布重建	DA6.3 必须停下等用户确认门槛
D7	MoE 回归、文档、推送	最终汇报
除了表里的汇报点：

遇到下面「必须停下」的情况，随时停下汇报；
其他情况连续推进，不需要每步请示。
必须停下来问用户的情况
任何硬约束无法满足，或者你认为某条硬约束本身有问题；
需要新增或升级任何依赖（包括 FLA）；
checkpoint 的 sha256 和 HF 元数据对不上，或者找不到可信来源；
锁页内存实际按 512 MiB 取整（64 层约 32 GiB），或者分配失败；
4096 压力测试在启用检查点卸载（设计文档 5.1 节 O1）后仍然超预算，需要改 lm_head 数值策略（O3），或者缩小 LoRA 范围或 rank（O4）；
真实模型 CPU fp32 参考的 1% 判据（DA5.2）经排查仍不通过；
DA6.3：功能门槛确认（必停）；
MoE 回归出现任何非位级一致的差异；
实际耗时比设计文档 5.3 节的估算慢 2 倍以上，而且原因不明；
你想改变设计文档里的关键设计，例如：
改用 PEFT 注入；
把 embedding 放到 GPU；
把 bf16 小张量合进层文件；
改变 PackedLinearFn 的数值策略。
停下来汇报时，包括：

现象和实测数据；
已排查的原因；
两到三个可选方案及各自影响；
你的推荐。
决策日志规则
偏离设计文档的地方都写进 docs/decisions.md。在现有日志末尾加一个 ## dense 27B 段落，每条包含：

日期；
设计文档原文或编号；
实测；
决定；
影响范围；
是否涉及硬约束；
用户是否确认（附原话）。
**不要把失败的验收改写成通过。**判据调整要有原理依据和对照实验，参考 docs/decisions.md 里 MoE A5.2 拆分判据的做法。

工作方式
先 tiny 后真实：tiny dense 端到端（D4）全部通过之前，不在真实 27B 上训练。
用记录器运行：每次运行用 tools/run_recorded.py，新建运行目录。实际命令、代码快照和退出码以 manifest 为准。
证据如实：汇报区分「已实测」「推断」「未验证」。估算值不能写成实测值，资格探针的结果也不能写成完整训练的结论。
MoE 基线先记：B0 在 main 上、在动任何代码之前记录。
复用优先：解码、streamer、分块 CE、训练循环、检查点、模板和数据代码都尽量直接复用。只有设计文档第 4 节列出的文件需要新增或修改；超出范围的改动写进决策日志。
长任务：真实训练这类长时间运行，在后台执行并定期查看。中断后从最近的优化器步检查点续训，不从头重来。
最终交付和汇报内容
远端分支 feat/dense27b-lora，以及最终提交的哈希。
验收汇总表，按设计文档的 DA 编号逐项列出：通过、未通过或跳过，加实测值，以及证据位置。
实测资源和速度：
显存（allocated、reserved、整卡剩余）；
内存和锁页字节；
每步耗时及拆分：H2D、解码、GEMM、GDN、注意力。
真实训练结果：
留出集和回放集的数值；
发布产物的位置和 sha256；
续训和重建的比对结果。
MoE 回归结果（对照 B0）。
决策日志里新增的条目摘要。
遗留风险和建议的下一步，包括 ROADMAP 里哪些优化对 dense 最有价值。
