# nvfp4lora 下一阶段改进路线图

面向原生 Windows、16 GB 显存与 64 GB 主存，继续保留压缩专家在 CPU 存储、GPU 承担主要计算的后端，优先通过端到端测量减少解码、专家调度和重复计算，再逐步接入成熟训练前端。是否改用 CPU 专家计算，由同机、同质量目标的测试决定。

**状态：后续研究与优化计划，尚未实施。** 整理日期为 2026-10-29，源码阅读基线为 `c15cfc70488e505e826d11e1d2711f0b00fec5b6`。本页延续现有正确性基线；已完成能力与验收范围见[实施结果](docs/results.md)，历史方案与判据见[实施决策](docs/decisions.md)。

## 优先投入的三项工作

| 优先项 | 下一步工作 | 应交付的判断依据 |
| --- | --- | --- |
| 真实训练端到端 profiler | 在 C3 和全部专家 r2 上区分搬运、等待、解码、路由、GEMM、重算与训练外围开销 | 同一输入和运行条件下的时间线、吞吐、内存与热点占比，判断实际临界路径 |
| 专家计算与重算优化 | 按热点推进融合解码、grouped GEMM、路由与聚合融合、选择性保留中间量 | 独立梯度参考、真实训练吞吐、工作区上限及长序列压力对照 |
| 实际推理后端质量闭环 | 衔接已有 G2/G3 范围，在实际部署精度和 adapter 加载路径上评测 | 发布重载、teacher forcing、生成结果、领域留出与通用回放对照 |

现有异步搬运具备事件保护与统计，但[跨流重叠探针](docs/streaming.md)使用了人为延迟，不能直接证明真实训练提速。[逐层重算工具](tools/profile_recompute_memory.py)主要验证峰值与生命周期，不能代替时间占比分析。新增 profiler 应覆盖预热后的完整优化器步、全部梯度累积 microbatch，并分别报告训练核心耗时和含审计、评测、保存的总耗时。加载及首次状态分配另列，避免与稳态混算。

记录真实输入 token 和监督 token 吞吐、步耗时分布、路由负载分布、H2D 字节及等待、各阶段 kernel 与 CPU 调度时间、GPU allocated/reserved、整卡余量、主存与 pinned allocator 占用。CUDA 事件区间可能重叠，不能与 wall time 简单相加；避免为测量逐层强制同步。沿用固定硬件、数据顺序、长度、累积数、精度和检查点配置做对照。

## 六个主要改进方向

### 一 依据临界路径选择权重搬运或 CPU 专家计算

[ExpertStore](src/nvfp4lora/store/expert_store.py) 保存冻结的 packed 专家，[streamer](src/nvfp4lora/store/async_streamer.py) 把层数据搬入 GPU 槽，[专家算子](src/nvfp4lora/ops/experts_fn.py) 在 GPU 解码并计算。CPU offload 在这里首先是存储与供给职责，不要求 CPU 执行专家矩阵乘法。

KT 的 CPU 专家路线以主机算力减少权重搬运，本项目以 PCIe 权重传输换取 GPU 算力。可用下面的近似理解两条路线的单次执行成本：

$$
T_{\text{GPU 流式}} \approx T_{\text{GPU 计算}} + T_{\text{未隐藏的权重传输与调度}}
$$

$$
T_{\text{CPU 专家}} \approx T_{\text{CPU 计算}} + T_{\text{未隐藏的激活传输与同步}}
$$

两式帮助定位瓶颈；流水线吞吐仍需按实际依赖和重叠测量。比较应覆盖前向、输入与路由权重梯度、LoRA 梯度及重算，不能以单 token 解码或单次 GEMM 代替训练。

后续可在同机比较两条路线，记录 CPU 指令集、内存带宽、PCIe、专家形状、token 数、量化格式及 kernel 效率。使用相同来源权重并说明数值策略差异；若 CPU 候选尚不支持当前格式或平台，应先记录能力缺口。桌面 CPU 配合较强 GPU 时，保留 GPU 计算完全可能更合适，具体交叉点需实测。KT 值得优先借鉴的是布局、批量执行、负载调度和工作区管理。[KT 路线说明](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/papers/kt-ft/README_ZH.md)

### 二 按整个 microbatch 的路由负载组织专家计算

单 token 稀疏不代表整个 microbatch 只使用少数专家。当前目标为每层 256 个专家、每 token 选择 8 个。4096 个参与路由的 token 共形成 32768 次 token 与专家分配，平均每专家为：

$$
\frac{4096\times8}{256}=128
$$

128 是平均分配数，不代表每个专家都接收相同数量，也不能单凭平均数断言实际覆盖率。长序列通常有机会覆盖大量专家；应记录每层命中专家数、最大值和分位数，以及真实语料下的偏斜。这里的 token 数是单次专家调用的输入规模，不是多个梯度累积 microbatch 的合计。训练的权重复用和矩阵规模与单 token 解码明显不同，按层搬入多数专家未必浪费。

当前 `_chunk_forward` 逐专家执行 Python 循环及多个 `F.linear`；`_route_plan` 中 `torch.bincount(...).tolist()` 把计数取回 CPU，在 CUDA 路径上产生同步，前向和反向分别构建计划。后续重点是减少小 kernel、Python 调度和路由同步，并评估 grouped GEMM 与路由、聚合融合。[实现](src/nvfp4lora/ops/experts_fn.py)

**grouped GEMM 已有资格测试通过，尚未接入生产专家算子。** 现有[探针](tools/check_grouped_mm.py)使用 256 组、每组 128 行的规则负载；下一步应扩展到真实不等长分组、零命中、热门专家、LoRA 分支和块内反向。该探针计时含 CPU 拷贝，不能据此推断整模加速比。[已有决定](docs/decisions.md)

工作区应同时限制专家数、路由 token 数和总字节，不能只固定为 32 个专家。热门专家必要时沿 token 维分片；是否跳过未命中专家的解码由覆盖率和成本决定。还要计入块外路由缓冲：当前单个 `[T×K,H]` FP32 输出或 dX 缓冲在 T4096、K8、H2048 时就有 256 MiB，缩小专家块不会消除它。融合聚合时继续保证路由权重梯度和确定性的 top-k 汇总语义。

### 三 先优化 NVFP4 解码与执行 再评估原生低精度乘法

NVFP4 的首要收益已经来自存储。当前生产路径把 `E2M1 × block_scale` 局部解码成 BF16，执行 BF16 基座 GEMM，再在 FP32 输出端施加原始 global scale；LoRA、SwiGLU 和路由聚合用 FP32。down 基座读舍入后的 BF16 hidden，down LoRA 读 FP32 hidden。它已经支持训练，不依赖完整的原生 W4A4 前反向链路。[数值协议](docs/expert-op.md)

[现有解码](src/nvfp4lora/format/nvfp4.py)包含多个 PyTorch 位运算、索引查表、FP32 中间张量和 dtype 转换。建议按以下顺序逐项测量：

1. 融合 nibble 解包、E2M1 映射、block scale 和 BF16 写出，减少中间张量与启动开销，保持 global scale 的施加位置。
2. 接入适合不等长路由的 grouped GEMM，并减少路由、LoRA 分支和聚合调度。
3. 针对 profiler 确认的反向热点实现输入梯度、路由权重梯度和 A/B 梯度计算，减少通用 autograd 图重建。
4. 再评估原生 FP4 乘法，以及可能需要的激活量化。其舍入、缩放和训练轨迹变化应独立做质量实验。

这是默认演进顺序，具体投入以端到端收益为准。反向访问数学上的 $W^\top$ 时，必须按原始 K 方向的每 16 元素缩放组解释权重，不能简单转置 packed 字节后沿用前向解码。任何融合也都要保留原始缩放语义、BF16 舍入位置和 FP32 LoRA 策略；更换策略应显式标识为数值变更。

### 四 依据计算收益选择重算与保留边界

当前[训练循环](src/nvfp4lora/train/loop.py)启用非重入层级 gradient checkpointing，[专家反向](src/nvfp4lora/ops/experts_fn.py)又在 `_backward_block` 中重新解码，并调用 `_chunk_forward` 和 `autograd.grad`。因此同一专家路径可能经历原始前向、层级重算和算子内部重建，重复付出解码与计算成本。

非重入 checkpoint 可以提前停止，不能把源码中的嵌套直接等同于每个专家完整执行三遍。先在真实时间线中分别计数解码、专家前向与内部重建，关联层号和 microbatch，确认哪些成本重复以及能否被搬运隐藏。

可以借鉴 KT 的 activation retain/recompute 思路，按每增加一字节保留内存能够省掉多少临界路径计算时间选择边界。候选包括复用紧凑的排序索引、计数和偏移，选择性保留昂贵激活，或重新划分层级与专家算子的 checkpoint 边界。评估 GPU 保留和有界 CPU 激活缓存时同时计入传输、同步与生命周期成本；大块解码权重不宜成为默认保留对象。[KT 激活策略](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/python/sft/config.py#L56-L80)

保留[已有逐层重算与冻结检查](tools/profile_recompute_memory.py)作为回归门槛，并用没有 profiler 的运行复核吞吐和峰值。4096 长度的专家训练曾仅剩 528 MiB 整卡余量，新增保留项必须有明确预算。[已有容量结果](docs/results.md)

### 五 把 LoRA 训练范围作为一等配置

对每层每个专家的 gate、up、down 分别添加 rank 为 $r$ 的独立 LoRA，参数量为：

$$
N_{\text{expert LoRA}}=3LEr(H+I)
$$

代入 L40、E256、r2、H2048、I512，专家 LoRA 有 **157,286,400** 个参数；加上 C3 r16 的 **19,169,280** 个参数，共 **176,455,680** 个。当前参数、梯度和 Adam 两个矩均按 FP32 计，基本预算为：

$$
176{,}455{,}680\times(4+4+4+4)=2{,}823{,}290{,}880\ \text{bytes}\approx2.63\ \text{GiB}
$$

该预算不含冻结基座、激活、GPU 权重槽、算子返回的临时梯度、更新临时量或 allocator 余量。[参数实现](src/nvfp4lora/model/packed_experts.py)、[实际参数与内存清单](docs/evidence/real-expert-r2-step-summary.json)

当前已有 `expert_rank=0` 与所有层全部专家统一 rank 的开关，非专家范围由固定 C3 正则选择；尚未提供任意层、投影和专家子集的训练策略。[构建入口](src/nvfp4lora/model/build.py)

后续应支持按层、投影、专家集合和 rank 配置，并在构建前估算参数与优化器预算。只把不训练的专家行梯度置零，仍会保留整个 Parameter 及 Adam 状态；若要节省容量，需要相应的参数组织和导出映射。继续区分“永久冻结的专家”与“本步未命中但已有历史动量的专家”，保持标准 AdamW 的既有语义。

现有固定功能任务中，C3 和全部专家 r2 都达到验收目标；这不证明全部专家带来额外领域收益，也不证明其对其他任务无益。在同一领域质量目标下，比较少量层、单投影、专家子集及全部专家的时间和内存成本，可能比进一步压缩冻结基座更有收益。共享 adapter 分支会改变模型方法，应单独训练和评测，不能作为保持行为不变的性能替换。[功能结果及适用范围](docs/results.md)

### 六 用有界分级存储支持更大模型

`ExpertStore` 初始化时读取全部专家层，默认 `pin_memory=True`；CUDA 构建路径也启用 pinned bank。异步 streamer 当前要求全部 bank 持续 pinned，GPU 使用两个层槽。它是主存驻留的压缩权重供给，还没有运行时 SSD 按需缓存。[store](src/nvfp4lora/store/expert_store.py)、[异步约束](src/nvfp4lora/store/async_streamer.py)

专家逻辑 payload 为 **18,119,639,040 bytes，约 16.88 GiB**。已有构建记录中的 pinned allocator 实际分配为 **21,474,836,480 bytes，即 20 GiB**，预算还需包括进程其他工作集，不能只看逻辑 payload。[已有账本](docs/evidence/real-expert-r2-step-summary.json)

更大模型需要显式管理以下层级及其容量：

```text
SSD 原始压缩专家
  → 有容量上限的主存缓存
  → 少量可复用的 pinned 传输缓冲区
  → GPU 层槽或专家槽
  → 有界解码与路由工作区
```

主存缓存可以采用普通可分页内存或受控文件映射，热数据才进入 pinned 缓冲。需要同时设计按层顺序读取与预取、缓存淘汰、未命中背压、I/O 完成通知、冻结字节校验，以及 CPU 缓冲与 GPU 事件共同保护的生命周期。引入这一层级需要调整现有异步接口，不能仅关闭 `pin_memory` 就认为已完成 SSD offload。避免锁页全部主存，也避免依赖操作系统随机换页维持训练。

对某个测量区间，必要 SSD 未命中字节数除以实际有效读取带宽，是 I/O 服务时间的下界，也是总耗时的一项下界；总耗时还受计算和传输约束。它可以与其他工作重叠，但 CUDA stream 不会消除实际读取需求。长序列覆盖大量专家时，较小缓存可能在前向与反向间反复未命中，应记录实际读取放大。

KT 所核对路径同样在加载后保有 CPU 专家权重。其 `ephemeral` 的 `consume_layer` 在同步 C++ 加载后删除临时转换文件，是文件生命周期管理，不能据此推断支持训练期间的 SSD 权重分页。[KT 权重加载](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/python/sft/amx.py)、[ephemeral 实现](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/python/sft/ephemeral.py#L475-L496)

35B 参数的纯 BF16 权重粗略为 70 GB，约 65.2 GiB，尚未计运行时工作集，在 64 GB 主存上难以全部容纳。NVFP4 也让主存侧工作集可承受。BF16 加 SSD 能扩大可用容量，性能仍取决于实际读取量、带宽、访问局部性与调度。

## 成熟训练前端与专用后端的维护边界

优先把专用执行后端做薄，再逐步复用训练生态。普通线性层当前已经使用 PEFT，应继续保持这一边界。

| 能力 | 后续维护方向 |
| --- | --- |
| 数据处理、训练循环、调度、日志、常规评测 | 分阶段接入 Trainer、TRL 或 LLaMA-Factory，优先验证一个入口 |
| 普通线性层 LoRA | 继续复用 PEFT |
| NVFP4 专家加载、流式生命周期、专家前反向 | 暂时保留专用后端，按能力与实测收益逐步替换 |
| 格式绑定、专家 adapter 导出与加载 | 保留严格契约，逐步提供标准化转换和往返验证 |
| 精确续训、冻结不变量、梯度参考验证 | 保留现有测试，作为前端集成回归门槛 |

外部框架能力需要持续复核。截至 2026-10-29 核对的 Unsloth 源码已有保留 packed NVFP4 的 W4A16 线性层、LoRA 路径所需的前向及输入梯度计算，以及加载期间把层放入 CPU、训练时层交换的集成入口。[NVFP4 kernel](https://github.com/unslothai/unsloth/blob/fa7ad0d609a601bcbeb623ac42d2e741703bf06f/unsloth/kernels/nvfp4.py)、[NVFP4 加载与检查](https://github.com/unslothai/unsloth/blob/fa7ad0d609a601bcbeb623ac42d2e741703bf06f/unsloth/models/loader_utils.py#L2601-L2644)、[层交换集成](https://github.com/unslothai/unsloth/blob/fa7ad0d609a601bcbeb623ac42d2e741703bf06f/unsloth/models/_utils.py#L10167-L10179)

这几项能力不能合并推断为“本项目 ModelOpt 风格 NVFP4 MoE、全部专家 LoRA 和层交换的组合已完整支持”。该 NVFP4 加载路径检查 compressed-tensors scheme、二维 `nn.Linear`、分组形状和 global scale；其解码采用 `block_scale / weight_global_scale`，本项目保留 `block_scale` 并在 GEMM 输出端乘 `weight_scale_2`，格式转换与舍入位置都需要核对。层交换还有 checkpoint 和 `unsloth_zoo` 依赖条件，组合支持应独立做最小复现。[层交换条件](https://github.com/unslothai/unsloth/blob/fa7ad0d609a601bcbeb623ac42d2e741703bf06f/unsloth/models/_utils.py#L10884-L10922)

前端接入时重点核对：按监督 token 总数归一化梯度累积、实际可训练集合、优化器参数分组、数据游标、随机数状态、scheduler、专家路由历史与 streamer 前后向阶段。成熟 Trainer 能接管训练流程，不等于自动继承现有精确续训协议；先完成同配置对照，再扩大支持范围。

## 优化验收与质量边界

性能候选先保持冻结字节与格式绑定，再通过[独立专家参考](src/nvfp4lora/validation/experts_reference.py)、[专家测试](tests/test_experts.py)、[真实激活梯度检查](tools/check_real_expert_gradients.py)和长序列内存检查。必须覆盖输入、路由权重、gate/up/down 六个 A/B 的梯度、零命中、非零 B、重算及异步槽覆盖。

冻结基座和发布绑定继续严格核对原字节；相同实现和配置的保存、重载与续训继续检查完整状态。更换 GEMM、融合或精度策略可能改变浮点累加顺序，应在实验前确定对照和误差预算，保留实际离散路由差异；不能把新 kernel 的近似相符写成 E2，也不能通过放宽判据掩盖回归。

当前 G1 验收证明固定任务上的训练机制与功能效果；实际 FreeToken/Speed 推理的精度一致性和领域质量仍属于 G2/G3。后续用同一基座、adapter、模板和生成设置，先核对实际推理后端的 teacher forcing 与启用/禁用行为，再评测领域留出、通用回放、生成质量及吞吐。训练端观测到原 FP8 input scale 越界的风险也要纳入该闭环。[已知范围与风险](docs/results.md)

每个候选应同时提交正确性、端到端速度、峰值工作集及质量对照；只有在目标硬件和同一质量目标下收益明确，才决定是否成为默认路径。
