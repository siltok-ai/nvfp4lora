# Qwen3.6-35B-A3B：在部署用的 NVFP4 权重上训练 LoRA——规划与实施指导（v2）

版本 v2　2026-10-27　作者：@macmini1-opus　对应：#文本lora训练可行性与方案分析 task #3
读者：在 RTX 5070 Ti 16 GB + 64 GB 内存的**原生 Windows** 机器上做探索实现的 codex
v1 → v2 的修订记录见附录 C；对 codex 14 条意见的逐条回应见附录 D。

---

## 0. 怎么读这份文档

**这份文档要做到的事**：讲清每个组件为什么这样设计、做错了会出什么问题；给出分阶段、能直接动手的步骤，每个阶段都有验收标准和排查方向。遇到和文档不一致的实测结果，按 0.4 节处理。

**标注约定**
- **【已核实】**：读过源码（附文件和行号、commit）或 checkpoint 文件头。行号对应的是文中注明的版本，换了版本以函数名为准。
- **【估算】**：按公式算的，没有实测，误差可能有 ±30% 甚至更大。**所有估算都必须在对应阶段实测替换**。
- **【推断】**：根据代码或原理推出，还没验证。
- **必须**：硬约束。**建议**：默认做法，可以按实测调整，但调整要记录理由。

### 0.1 交付形态和范围（2026-10-27 与 @liboqi 确认）

- **交付形态 A**：一个和训练框架无关的**核心库**（格式读取、按层搬运、专家算子与专家 LoRA、模型构建、产物读写），加上一个**自带的精简训练脚本**。
- **演进方向 A′**：之后把核心库嵌进 Speed，成为 Speed 的训练功能（第 10 节是大纲）。
- **不在交付范围**：LLaMA-Factory 集成、DeepSpeed ZeRO-3。附录 B 保留了它们的接入要点，供以后参考。
- **这和最初讨论的「二开 LLaMA-Factory + ZeRO-3」有明确偏离**。原因有三：
  1. 原生 Windows 上，ZeRO-3 只能用 gloo，它在初始化、保存、梯度归约时对 CUDA 张量做的集合通信无人验证，CPU Adam 也只有 GCC 的编译参数；
  2. LLaMA-Factory 锁定的 transformers ≤5.8 和 Speed 的 5.16 冲突，无法进入 Speed 的环境；
  3. 单卡场景下，自己实现按层搬运更简单、更可控。

### 0.2 三个目标

| 目标 | 内容 | 本文覆盖 |
|---|---|---|
| **G1** | 在这台机器上训练 LoRA，有产物，并分类证明训练流程正确（见 0.3 节的四类验收） | 详细（第 5～8 节） |
| G2 | 改造 FreeToken，支持带 LoRA 推理 | 大纲（第 9 节） |
| G3 | 验证训练是否改善了指定领域的行为 | 大纲（第 9 节） |

### 0.3 验收分四类，分别判定

| 类别 | 证明什么 | 主要手段 |
|---|---|---|
| **V1 梯度正确** | 前向、反向（含路由权重的梯度和专家 LoRA 的梯度）和参考实现一致 | 逐算子、逐层和参考实现对比；小形状上做 fp64 gradcheck；小模型的逐步训练对比 |
| **V2 可精确续训** | 从训练检查点恢复后，后续训练和不中断时一致 | 在优化器 step 的边界保存和恢复，然后对比 |
| **V3 adapter 确实生效** | 训练出的 adapter 改变了模型行为，加载后能复现 | 金丝雀任务：基座和 adapter 对比；adapter 开关对比；保存后再加载对比 |
| **V4 领域泛化** | 在目标领域上真的变好了 | G3 的评测，不属于 G1 |

**单次训练的 loss 曲线、过拟合测试只是诊断手段，不作为 V1～V3 的判据。**

### 0.4 硬约束（必须）

1. **训练基座就是部署用的 NVFP4 checkpoint**（HF `nvidia/Qwen3.6-35B-A3B-NVFP4` @`1355db6a`，用前先校验 sha256）。专家只在计算时做反量化，不转成其他量化格式。
2. **只用原生 Windows**，不使用、也不探索 WSL。
3. **环境和 Speed 对齐**：用独立的 venv，但精确复制 Speed 的锁定版本（4 节），不改 Speed 本身的环境。只在遇到明显的环境问题时偏离，偏离要记录。
4. **路由专家不能整体常驻显存**，按层搬运。
5. **反向时重新反量化专家，不为反向保存反量化后的专家权重。**
6. **专家算子必须按 5.P2 节的协议，返回对路由权重和专家 LoRA 每个张量的梯度。**
7. **LoRA 的 B 初始化为 0**。同一条计算路径上，adapter 为零时的输出必须和无 adapter 时逐位一致（1.3 节的 E2 级）。
8. **训练样本按 Speed 首期的用法渲染**：关闭思考，只对答案和 `<|im_end|>` 计 loss，通过 6.1 节的模板和标签测试。
9. **每个阶段的验收通过后，再进入下一阶段**。跳过要在决策日志里写明原因和风险。
10. **记录**：环境、关卡结果、实测数据、所有偏离（11 节）。

### 0.5 遇到和文档不一致的情况

1. 先在更小的规模上复现（单层、小模型、fp32），确认不是偶发；
2. 判断是文档错了、环境不同，还是实现有 bug；
3. 文档错了就按实测修正，并在决策日志里写清「原文怎么说、实测是什么、改成了什么」；
4. 涉及硬约束的，停下来汇报。

---

## 1. 背景

### 1.1 为什么要自己做

| 现成方案 | 卡点 | 出处 |
|---|---|---|
| 标准 QLoRA（bitsandbytes 4bit） | 只替换 `nn.Linear`，3D 融合专家仍是 bf16（约 64 GB）；HF 量化器会把整个模型放上 GPU | 【已核实】transformers `integrations/bitsandbytes.py:189`、`quantizers/quantizer_bnb_4bit.py:94-97` |
| LLaMA-Factory + ZeRO-3 | ZeRO-3 下预量化模型直接报错；FP8 被反量化成 bf16；bf16 全量放内存约 69 GB | 【已核实】LLaMA-Factory `model_utils/quantization.py:94-105` |
| PEFT 给专家挂 LoRA | 每次前向为全部 256 个专家构造 W + B·A，每层约 1.6 GB，和 rank 无关 | 【已核实】PEFT `tuners/lora/layer.py:2441, 2705` |
| Axolotl 的 NVFP4 MoE LoRA | 专家必须常驻显存（18.1 GB），没有 qwen3_5_moe 的 NVFP4 适配，构建偏 Linux | 【已核实】Axolotl `integrations/kernels/README.md` |
| KTransformers | 需要 AVX-512，不读 NVFP4，Windows 原生不维护 | 见 task #1 |
| transformers 直接加载 | 不认识 `quant_method: "modelopt"`，打包数据和 bf16 形状对不上 | 【推断】transformers `quantizers/auto.py:296-301` |

### 1.2 数值一致性分三级（全文的「一致」都按这三级理解）

| 级别 | 含义 | 例子 |
|---|---|---|
| **E1 字节一致** | 码值、缩放等原始字节和部署文件完全相同 | 预处理前后的专家数据；GPU 槽里的数据和 CPU 源数据 |
| **E2 同路径逐位一致** | 同一个实现、同样的输入和计算顺序，输出逐位相同 | adapter 为零 vs 无 adapter；保存再加载后在同一 batch 上的 loss；同一输入跑两次（要求用确定性的算子） |
| **E3 跨实现在容差内** | 不同实现（参考实现、FreeToken、不同的计算顺序）在给定容差内接近 | 我们的专家算子 vs transformers 的 eager 实现；训练进程 vs FreeToken |

**改变计算顺序（例如全局缩放乘在权重上还是乘在输出上）在实数上等价，但 bf16 的舍入不同，只能保证 E3。** codex 在 CPU 上的反例中，两种顺序的最大绝对差是 0.001953125。

---

## 2. 模型和 checkpoint 的事实

### 2.1 模型结构【已核实，transformers c0d024f `modeling_qwen3_5_moe.py`；在 5.16 上需在 P0 复核】

| 项 | 值 |
|---|---|
| 训练用的类 | `Qwen3_5MoeForCausalLM`，配置取 `text_config`；它会忽略 `mtp.*` 和 `model.visual.*` |
| 规模 | 40 层，hidden 2048，词表 248,320，embedding 和 lm_head 不共享 |
| 层类型 | 每 4 层：3 层 Gated DeltaNet（GDN）+ 1 层全注意力；全注意力在第 3、7、…、39 层 |
| 全注意力 | 16 个 query 头、2 个 KV 头，head_dim 256；`q_proj` 输出按头交错 [query_h \| gate_h]；输出 `o_proj(attn · sigmoid(gate))` |
| GDN | `in_proj_qkv` 切成 q 2048 \| k 2048 \| v 4096；因果卷积（kernel 4）+ SiLU；β = sigmoid(b)，g = −exp(A_log)·softplus(a + dt_bias)；`chunk_gated_delta_rule` |
| 路由 | fp32 softmax → top-8 → 重新归一化；没有 bias |
| 专家 | `gate_up_proj` [256, 1024, 2048]，gate 在前；`down_proj` [256, 2048, 512]；silu(gate)·up → down → 乘路由权重 → 求和 |
| MoE 输出 | 路由专家输出 + sigmoid(shared_expert_gate(x)) · shared_expert(x) |
| Norm | 普通 norm 是 x̂·(1 + w)；GDN 的 norm 是 w·x̂·silu(z) |

**专家的接入方式（建议）**：直接把每层的 `mlp.experts` 替换成我们的 `PackedExperts`，调用签名和原类一致（`forward(hidden_states, top_k_index, top_k_weights)`，P0 在 5.16 上核对）。备选是 transformers 的专家实现注册接口（`integrations/moe.py:550-614`），但替换模块不依赖这个接口，对版本更不敏感。

**GDN 的 kernel 依赖**【已核实】：查找顺序是 hub kernel → 已安装的 flash-linear-attention → 纯 torch 参考实现（逐 token 递推，很慢）。参考实现遇到打包序列（`cu_seqlens`）会直接报错（`hub_kernels.py:990-1050`）。

### 2.2 部署 checkpoint【已核实】

- 3 个分片，sha256 前缀分别为 `07141c2d…`、`6dea9c75…`、`9758875f…`，完整值以 HF 的 LFS 元数据为准；共 124,468 个张量，23,407,580,856 字节。
- 前缀：文本部分 `model.language_model.`，另有 `model.visual.`、根上的 `lm_head.*` 和 `mtp.*`。

| 模块（相对 `model.language_model.layers.N.`） | weight | weight_scale | weight_scale_2 | input_scale |
|---|---|---|---|---|
| `mlp.experts.{0..255}.gate_proj` / `up_proj` | U8 [512, 1024] | E4M3 [512, 128] | F32 标量 | F32 标量 |
| `mlp.experts.{e}.down_proj` | U8 [2048, 256] | E4M3 [2048, 32] | F32 标量 | F32 标量 |
| `mlp.shared_expert.{gate,up,down}_proj` | 同上 | 同上 | 同上 | 同上 |
| `lm_head`（在根上） | U8 [248320, 1024] | E4M3 [248320, 128] | F32 标量 | F32 标量 |
| `linear_attn.in_proj_qkv` / `in_proj_z` / `out_proj` | F8_E4M3 [8192, 2048] / [4096, 2048] / [2048, 4096] | F32 标量 | — | F32 标量 |
| `self_attn.q_proj` / `k_proj`、`v_proj` / `o_proj` | F8 [8192, 2048] / [512, 2048] / [2048, 4096] | F32 标量 | — | F32 标量 |

其余张量都是 bf16：`in_proj_a/b`、`conv1d`、`A_log`、`dt_bias`、各种 norm、路由器 `mlp.gate`、`shared_expert_gate`、`embed_tokens`。

路由专家合计 18,119,639,040 字节，每层约 453 MB。

元数据互相矛盾：`config.json` 写的是 ModelOpt 0.37.0，NVFP4 组声称 4bit 激活；`hf_quant_config.json` 写的是 0.44.0，标的是 W4A16。**一律按 W4A16 处理，按张量的 dtype 和键名判断格式，不依赖元数据。**

### 2.3 位级格式【已核实：ModelOpt `qtensor/nvfp4_tensor.py`、FreeToken `nvfp4_dequant.py`、Axolotl `dequant_grouped.py`】

- **打包**：`byte = (code[2j+1] << 4) | code[2j]`，低 4 位是 K 方向上下标较小的元素。
- **E2M1 码表**（第 3 位是符号）：`[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]`。
- **块缩放**：沿 K 每 16 个元素一块，`weight_scale` 是 E4M3FN，行主序 [out, in/16]，不做 swizzle。
- **全局缩放**：`weight_scale_2 = amax / (6 × 448)`，不是 2 的幂。
- **反量化**：`W = LUT[code] × float(weight_scale) × weight_scale_2`。`LUT × 块缩放` 能在 bf16 中精确表示，再乘全局缩放就不能了。
- **FP8**：`W ≈ fp8 × weight_scale`。推理时 `x_q = e4m3(clamp(x / input_scale, ±448))`。
- 【推断】同一专家的 gate 和 up 共用一个全局缩放。**代码一律按每个张量自己的标量处理，不依赖这个推断。**

```python
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
def dequant_nvfp4_ref(packed_u8, scale_e4m3, global_scale):      # 参考实现，CPU，fp32
    lo, hi = packed_u8 & 0x0F, packed_u8 >> 4
    codes = torch.stack([lo, hi], -1).reshape(packed_u8.shape[0], -1)
    s = scale_e4m3.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return LUT[codes.long()] * s * float(global_scale)
```

---

## 3. 总体设计

### 3.1 组件

```
离线预处理（P1）：NVFP4 checkpoint ──逐张量──► ① 非专家 bf16 模型（纯文本 CausalLM 键名，无 quantization_config）
                                                ② 按层打包的专家存储（字节和原文件一致，E1）
训练进程：
  模型 = transformers Qwen3_5MoeForCausalLM（先在 meta 设备上构建，再分配显存）
    ├─ 非专家权重：bf16，常驻 GPU；PEFT 在线性层上挂 LoRA
    └─ 每层 mlp.experts → PackedExperts
         ├─ 不持有冻结权重：打包专家在 ExpertStore（CPU 锁页内存），通过 LayerStreamer 按层取到 GPU 层槽
         ├─ 持有专家 LoRA 参数（nn.Parameter，在 GPU 上，可训练；C3 阶段没有）
         └─ forward → NVFP4ExpertsFn.apply(...)（协议见 5.P2）
  训练循环：数据和模板 → 前向 → 从隐藏状态分块算交叉熵 → 反向 → 优化器；设置 streamer 的阶段提示；保存训练检查点和发布产物
产物（P8）：发布产物（adapter + 专家 LoRA + manifest）；训练检查点（续训用）
```

### 3.2 一层 MoE 在一个训练步里的经过

1. **前向**（`streamer.phase = FORWARD`）：检查点区域里不保存中间结果。专家计算时 `acquire(L)`，同时预取 L+1，算完 `release`。
2. **反向到第 L 层**（`phase = BACKWARD`）：检查点触发重算，再次 `acquire(L)`，同时预取 L−1。
3. **第 L 层专家的反向**：`NVFP4ExpertsFn.backward` 再取一次，按块重新反量化，重算前向并求梯度。

每步每层的专家至少经 PCIe 搬运 2 次，槽位不够时 3 次，**实际次数用计数器统计**。

### 3.3 关键设计决策

| 编号 | 决策 | 理由 | 备选 |
|---|---|---|---|
| D1 | 用自己写的 streamer 做 offload，不用 DeepSpeed | 原生 Windows 上 ZeRO-3 风险高（见 0.1 节）；打包数据可以直接以 uint8 放锁页内存，不用伪装成 bf16 | 不做 |
| D2 | 核心库和训练框架无关，自带精简训练循环 | 形态 A 和 A′ 的需要；数值问题好定位 | 附录 B |
| D3 | 非专家部分 bf16 常驻 GPU | 约 4.9 GB【估算】，可以直接用 PEFT 挂 LoRA | 显存紧时把 embedding 放 CPU |
| D4 | 专家算子 v1：分块反量化 + 分组矩阵乘（或逐专家循环），纯 torch | 原生 Windows 上依赖最少，作为后续优化的数值基准 | v2 换成 Triton 反量化，或 Axolotl 的 Marlin kernel（后者反向约有 2% 的梯度误差，只作优化项） |
| D5 | 反向时按块重算前向，用 `torch.autograd.grad` 求这一块的梯度，累加后作为 Function 的返回值 | 不手推公式，不直接修改 `.grad`，梯度累积、检查点都走标准 autograd 路径 | 速度不够时再手写反向 |
| D6 | 取层的动作放在 Function 内部；预测方向靠训练循环显式设置的阶段提示 | Function.forward 总在关闭梯度的状态下执行，`is_grad_enabled()` 恒为 False，没法用它区分普通前向和重算；检查点的 early_stop 也会让模块钩子不可靠 | — |
| D7 | 精度策略显式化，Function 内部关闭 autocast（5.P2） | backward 里不会自动恢复前向的 autocast 状态；显式规则让前向和重算完全一致 | — |
| D8 | 全局缩放的施加顺序尽量和 FreeToken 的 kernel 一致；做不到时只要求 E3 | 见 1.2 节 | — |
| D9 | 专家 LoRA 的 gate、up、down 各自独立，A 为 [E, r, in]，B 为 [E, out, r]，注册为 PackedExperts 上的 nn.Parameter，归我们自己管理（不交给 PEFT） | 和 checkpoint 的分开存储一致，便于映射到 FreeToken；避开 PEFT 对名字里没有 `lora_` 的参数一律冻结的规则，以及它的设备放置逻辑 | 以后需要 PEFT 格式时写转换器 |
| D10 | v1 不做序列打包 | 打包要求 GDN 的 `cu_seqlens` 路径和正确的分段掩码 | 吞吐成为瓶颈后再加，并且要先通过「打包 vs 不打包」的逐样本 loss 对比 |
| D11 | 交叉熵从隐藏状态和 lm_head 权重出发分块计算，纯 torch 实现 | 不能先生成完整 logits；不依赖 Triton | 以后可以换 Liger 的融合线性交叉熵 |
| D12 | 先训 C3（注意力 + GDN + 共享专家），再训专家 LoRA | C3 不需要专家 LoRA 的基础设施，可以先把全流程验证完 | — |
| D13 | 优化器用标准 AdamW，权重衰减 0；接受 Adam 的默认语义：之前被路由过的专家，本步即使梯度为 0，也会因为动量继续更新 | 和 @liboqi 确认过 | 有需要时再做带掩码的优化器 |

---

## 4. 环境（与 Speed 对齐，原生 Windows）

**原则**：
- 用独立的 venv，但精确复制 Speed Windows 运行环境的版本，不修改 Speed 本身的环境；
- 生成带哈希的 lock 文件；
- 只有遇到明显的环境问题时才偏离 Speed 的版本，偏离要在决策日志中写明原因，并评估对 A′ 的影响。

| 依赖 | 版本 | 依据 / 说明 |
|---|---|---|
| Python | 3.12 | Speed 是 cp312 |
| torch | `2.11.0+cu130` | Speed `pyproject.toml:125`；FreeToken `release.json` |
| triton-windows | **以机器上 Speed 环境实际安装的版本为准** | FreeToken 的 `release.json` 锁 `3.6.0.post26`，Speed 的 pyproject 写 `>=3.5`，注释提到「torch 2.11 配 3.8」，两处不一致 |
| transformers | 5.16.x | Speed `pyproject.toml:128`（`>=5.16,<5.17`） |
| PEFT | 先试 0.18.1，确认和 transformers 5.16 兼容；不兼容就用兼容的最低版本 | 只用标准的线性层 LoRA；专家 LoRA 自己管理 |
| flash-linear-attention | **由 P0 测试决定**，0.4.2 和 0.5.x 都是候选 | 必须在这套 torch 和 triton-windows 上，用真实 GDN 形状测前向、反向和重算 |
| safetensors、tokenizers | 和 transformers 5.16 匹配的版本 | Speed 锁 safetensors `>=0.6,<1` |
| 不装 | DeepSpeed、LLaMA-Factory、Liger、Cut Cross Entropy、bitsandbytes、accelerate | 用不到，或者有纯 torch 的替代 |

**磁盘**：预留 100 GB 以上。

**可选的对照工具**：机器上 Speed 环境里的 FreeToken，提供推理端实际使用的 NVFP4 反量化 kernel。**只导入调用，不修改**。

---

## 5. 分阶段实施（G1）

**里程碑**
- **M1 = P0～P4**：独立环境、目标形状的专家算子、先同步后异步的搬运、小模型的训练对比和保存恢复。**M1 通过后才碰真实的 35B 模型。**
- **M2 = P5～P8**：真实模型的加载和前向校验、C3 训练、专家 LoRA、产物和续训。

codex 已经在本机跑过的小尺寸 bf16 反向、`grouped_mm` 前向和输入反向测试，算作预研；**不能代替**真实 GDN、256 专家目标形状、完整流式训练和最终产物的验收。

### P0　环境和关卡（0.5～1 天）

| 关卡 | 做法 | 通过标准 | 不通过时（不用 WSL） |
|---|---|---|---|
| G0.1 环境 | 按第 4 节建 venv；记录操作系统、CPU、内存、GPU、驱动和各包版本；跑 bf16 矩阵乘 | sm_120 可用；版本和 Speed 一致 | 记录差异，评估是否需要偏离 |
| G0.2 锁页内存 | 分配 18.5 GB 锁页内存；按 0.45 GB 一块测 H2D 带宽 | 分配成功；记录带宽 | 改用「可分页内存 + 2～3 个锁页中转缓冲」，记录带宽 |
| G0.3 GDN 训练 kernel | flash-linear-attention 的候选版本，用真实形状（T=4096，16 个 qk 头、32 个 v 头，dk=dv=128）测 `chunk_gated_delta_rule` 的前向、反向和梯度检查点重算，对照纯 torch 参考实现（fp32，可用较短的 T） | 输出和梯度在 E3 容差内（见 6.2 节）；记录耗时 | **回退：自己用纯 PyTorch 写分块的 gated delta rule**（前向和反向都走 autograd，按 64 token 一块），先在小形状上对照参考实现，再测真实形状的速度。transformers 自带的逐 token 参考实现只用于小模型 |
| G0.4 transformers 5.16 | 在 meta 设备上构建 `Qwen3_5MoeForCausalLM`；核对专家模块的 forward 签名，以及 `q_proj` 交错、gate_up 顺序等 2.1 节的结构事实 | 和 2.1 节一致 | 按实测更新本文 |
| G0.5 PEFT | 在 transformers 5.16 上给 bf16 线性层注入 LoRA，前向和反向都正常 | 正常 | 换版本 |
| G0.6 分组矩阵乘 | `torch._grouped_mm` 在 sm_120 上的前向、对输入的反向和对权重的反向（用 256 组、目标形状） | 和逐专家循环在 E3 内一致 | 用逐专家循环 |
| G0.7 FreeToken 反量化（可选） | 导入 `_dequant_nvfp4`（`freetoken/models/qwen3_5_moe/weight.py:83-93`） | 能调用 | 只用参考实现 |
| G0.8 checkpoint | 校验三个分片的 sha256 | 和 HF 一致 | 重新获取，**不要继续** |

### P1　格式层和离线预处理（1～2 天）

**步骤**
1. `format/nvfp4.py`、`format/fp8.py`：按 2.3 节写参考反量化。
2. `format/inspect.py`：只读文件头，按 dtype 和键名判断每个模块的格式。可以参考 Axolotl `nvfp4_moe_loading.py:152-236`（Apache-2.0，借用时保留版权声明）。
3. `format/preprocess.py`：逐分片、逐张量处理。
   - **非专家部分**：FP8 → bf16；共享专家和 lm_head（NVFP4）→ bf16；去掉 `language_model.`，输出 CausalLM 键名的 safetensors；`config.json` 取 `text_config` 并去掉 `quantization_config`；复制分词器文件。
     - 这一步是有损的：`fp8 × weight_scale` 和乘全局缩放都只有 E3。在决策日志中记下选择。
     - 也可以保存「码值的精确 bf16 + fp32 标量」，在 forward 里乘到输出上（D8）。
   - **专家部分**：每层一个文件，原样保留 E1 字节：gate、up、down 的打包数据和 E4M3 缩放（uint8），三组 [256] 的 fp32 全局缩放。附带 `index.json`（形状、dtype、偏移、每层 sha256）。
4. `model/build.py`，**不用 `from_pretrained`**：
   - 在 meta 设备上构建模型；
   - 把每层的 `mlp.experts` 替换成 `PackedExperts`；
   - `to_empty(device="cuda")`；
   - 逐张量拷贝非专家权重；
   - **重新计算 RoPE 的 `inv_freq` 等非持久 buffer**；
   - 检查没有残留的 meta 参数，键名一一对应。

**验收**
- A1.1：布局报告和 2.2 节完全一致。
- A1.2（E1）：每层抽 8 个专家，预处理文件的字节和原始分片完全相同。
- A1.3（E3）：参考反量化和 FreeToken 的 `_dequant_nvfp4` 在 ≤1 ulp 内（如果 G0.7 通过）。
- A1.4：构建出的模型非专家参数全部加载、键名无多无少，RoPE buffer 和正常构建的 CPU 小模型一致，各层权重统计正常。
- A1.5：预处理的内存峰值实测值 ≤16 GB。

### P2　专家算子（2～3 天）

**输入和梯度协议（必须严格遵守）**

```python
class NVFP4ExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, top_k_index, top_k_weights,
                Ag, Bg, Au, Bu, Ad, Bd,          # 专家 LoRA：Ag/Au [E, r, H]，Bg/Bu [E, I, r]，Ad [E, r, I]，Bd [E, H, r]，fp32
                layer_id: int, cfg):             # cfg：lora_scaling、分块大小、streamer 引用等非张量参数
        ...
        return out                               # [T, H]，bf16
    @staticmethod
    def backward(ctx, grad_out):
        ...
        return (dx, None, d_top_k_weights,
                dAg, dBg, dAu, dBu, dAd, dBd,
                None, None)                      # 和 forward 的输入逐个位置对应
```

- **每个需要梯度的张量都作为顶层参数传入**。不要放在 tuple 或 list 里，否则 autograd 不会追踪它们（codex 在 CPU 上复现过：tuple 里的 A/B 梯度为 None）。
- C3 阶段没有专家 LoRA，就用一个不带 A/B 的版本 `NVFP4ExpertsBaseFn(x, top_k_index, top_k_weights, layer_id, cfg)`，返回 `(dx, None, d_top_k_weights, None, None)`。
- `top_k_index` 是整数，没有梯度；`top_k_weights` 必须有梯度（硬约束 6）。

**精度策略（前向和重算完全相同）**
- Function 内部用 `torch.autocast(device_type="cuda", enabled=False)` 包住，所有 dtype 显式管理，不依赖外部的 autocast 状态。
- 基座矩阵乘：输入 bf16，权重是 bf16 的 `LUT × 块缩放`（精确），fp32 累加。全局缩放乘在输出上，**乘的时候用 fp32**。
  - 如果分组矩阵乘只能输出 bf16，就先转 fp32 再乘全局缩放，并在决策日志中记录它的舍入顺序和 FreeToken 的区别（D8，只要求 E3）。
- LoRA 分支：x 转成 fp32，计算 `(x @ Aᵀ) @ Bᵀ × scaling`，和 fp32 的基座输出相加（和 PEFT 的语义一致：PEFT 把输入转到 LoRA 的 dtype，结果再转回基座的 dtype）。
- SwiGLU 用 fp32 计算，结果转 bf16 后进入 down 的基座矩阵乘；down 的 LoRA 分支用 fp32 的 h。
- 对 k 的加权求和用 fp32 累加，最后转成 bf16 输出。
- 梯度：dx 的 dtype 和 x 相同（bf16），d_top_k_weights 和 top_k_weights 相同，dA/dB 是 fp32。
- **参考实现必须用同样的精度策略**，才能做 E2/E3 级对比；跨精度的对比只要求 E3。

**反向（D5）**
```
对每个专家块 [e0, e1)：
    取层（streamer.acquire），反量化这个块的 w_block
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        x_c = x_sorted[块内行].detach().requires_grad_()
        w_c = top_k_weights_sorted[块内行].detach().requires_grad_()
        A/B 切片 = A[e0:e1].detach().requires_grad_()  ...
        y_c = chunk_forward(x_c, w_c, w_block, A/B 切片)          # 和 forward 完全相同的函数
        grads = torch.autograd.grad(y_c, [x_c, w_c, A/B 切片…], grad_outputs=grad_out_sorted[块内行], allow_unused=True)
    把 dx_c、dw_c 按行放回，dA/dB 切片写进预分配的 fp32 缓冲（每个专家只属于一个块，所以只写一次，不会重复）
    streamer.release
最后按 perm 把 dx、d_top_k_weights 还原成原始顺序，返回
```

- 不在 backward 里调用 `.backward()` 或直接改 `.grad`。返回的梯度由 autograd 累加到参数的 `.grad` 上，所以梯度累积、检查点（重入和非重入）都走标准路径。
- 没有被路由到的专家，返回的 dA/dB 对应切片为 0。

**验收**（全部在**目标形状**上做：H=2048、I=512、E=256、top-8、T=4096，权重取真实的一层；另做小形状的 fp64 测试）

| 编号 | 检查 | 标准 |
|---|---|---|
| A2.1 V1 | 前向 vs 参考实现（同精度策略，用 fp32 反量化权重和相同的 top_k） | E3：bf16 相对 L2 ≤1e-2；把我们的实现切到 fp32 模式时 ≤1e-5 |
| A2.2 V1 | dx、d_top_k_weights、六个 dA/dB vs 参考实现的 autograd | E3：bf16 ≤2e-2，fp32 ≤1e-5 |
| A2.3 V1 | 小形状（H=64，I=32，E=8，top-2）fp64 `gradcheck`，覆盖所有输入 | 通过 |
| A2.4 V1 | 未被路由到的专家，返回的 dA/dB 切片 | 严格为 0 |
| A2.5 E2 | B=0 时，带 LoRA 和不带 LoRA 的输出 | 逐位一致 |
| A2.6 E2 | 同一输入跑两次 | 逐位一致（需要确定性的聚合，用排序后分段求和代替 `index_add`） |
| A2.7 | 梯度累积：两个 micro-batch 连续 backward 后的 `.grad`，等于两次单独 backward 的梯度之和 | E3 |
| A2.8 | 放进开梯度检查点的层里（重入和非重入两种模式），梯度和不开检查点时一致 | E3 |
| A2.9 | 显存：分块大小从 32 改成 256 时峰值明显上升，从 32 改成 8 时下降；用内存快照确认反向结束后没有残留的反量化权重 | 实测记录 |

### P3　ExpertStore 和 LayerStreamer（1～2 天）

**先实现同步版**（拷贝在当前流上做，不预取），通过 P2、P4 的验收后再加异步。

**异步版的槽位协议（正确性不依赖预测）**
- 每个槽的状态：`layer_id`、`generation`（每次装入新层加 1）、`copy_done`（拷贝完成事件）、`last_reader`（最后一个读者的完成事件）、`pins`（当前占用数）。
- `acquire(L)`：
  - 如果槽里已经是第 L 层：`pins += 1`，计算流等待 `copy_done`，返回带 `(slot, generation)` 的句柄；
  - 否则选一个 `pins == 0` 的牺牲槽（不选「下一个预测会用到」的槽），拷贝流先等待它的 `last_reader`，再从锁页内存发起拷贝，记录 `copy_done`，`generation += 1`，然后和前一种情况一样处理。
- `release(handle)`：在计算流上记录一个事件，作为 `last_reader`；`pins -= 1`。
- `prefetch(L)`：只有存在 `pins == 0` 的空闲或牺牲槽时才发起拷贝，不等待。预测错了，最多浪费一次拷贝。
- 使用句柄时检查 `generation`，不一致就报错，防止拿着过期的数据去算。
- **Python 的引用计数归零不代表 GPU 已经读完**，所以覆盖槽位只能以 `last_reader` 事件为准。
- **阶段提示**：训练循环在调用前向前设置 `streamer.phase = FORWARD`（预取 L+1），在 `loss.backward()` 前设置 `BACKWARD`（预取 L−1）；评测时设为 `FORWARD`。层号的访问顺序作为辅助信号。
- 打包张量**不注册成 Parameter 或 buffer**，只作为普通属性或放在外部注册表里，避免 `model.to()` 之类的操作把它们搬上 GPU。

**验收**
- A3.1 E1：随机顺序 acquire 1000 次，GPU 数据和 CPU 源逐字节一致。
- A3.2：两个槽时，模拟「前向 0→39 + 反向 39→0」的访问序列，并在计算流上插入人为延迟（`torch.cuda._sleep`）来逼出竞争：不出现「被读的槽在读完之前被覆盖」。
- A3.3：故意让预取完全猜错，结果仍然正确（E2：和不预取时逐位一致），只是更慢。
- A3.4 E2：和 P2 组合后，单层 forward/backward 在「同步版」和「异步版」之间逐位一致。
- A3.5：记录实测的有效带宽、等待时间和每步拷贝的字节数。

### P4　小模型端到端（2～3 天）★ M1 的核心

**步骤**
1. `tools/make_tiny_model.py`：用 `Qwen3_5MoeTextConfig` 建小模型：hidden 256，4 层（3 层 GDN + 1 层全注意力），8 个专家选 top-2，专家宽 128，共享专家 128，头数和维度相应缩小，vocab 2048。随机初始化，存成 bf16。
2. `tools/quantize_tiny.py`：用自己的量化函数，按 nvidia checkpoint 的布局写出小模型的 NVFP4 checkpoint：专家按专家拆开存，FP8 层带 `weight_scale` 和 `input_scale`，键名带 `model.language_model.` 前缀，`quant_algo` 为 `MIXED_PRECISION`。
   - 另做一个布局一致性检查：对真实 checkpoint 的某一层，用「我们的量化函数 + 真实的全局缩放」重新量化它反量化后的值，统计块缩放和码值的一致率。不要求 100%（块里没有 ±6 时，重算的块缩放可能不同），但一致率明显偏低说明我们对格式的理解有误。
3. **参考训练**：用反量化后的权重建普通的 transformers 模型，用 PEFT 在相同的线性层上挂 LoRA，专家 LoRA 用稠密写法实现，精度策略和我们的一致。两边用相同的 LoRA 初始值、数据顺序和超参。
4. 两边各训练 100 个优化器 step（梯度累积 2）。

**验收**
- A4.1 V1：第 1 步各模块的梯度和参考实现在 E3 内（余弦相似度 ≥0.999，相对 L2 ≤2e-2）；100 步后各模块 LoRA 权重的余弦相似度 ≥0.99。
- A4.2 V1：每一步的 loss 和参考实现在 E3 内（相对误差 ≤2%）。
- A4.3 V2：在第 50 步（优化器 step 边界）保存训练检查点，新进程恢复后，第 51～100 步的 loss 和不中断的训练 E2 一致（要求算子是确定性的；做不到时退为 E3 ≤1e-3，并注明是哪个算子不确定）。
- A4.4 V3：保存发布产物后，新进程只靠「预处理产物 + 发布产物」，在固定 batch 上的 loss 和保存前 E2 一致。
- A4.5：冻结权重（专家存储、非专家权重）训练前后的哈希不变。
- A4.6：实际放进优化器的参数集合 == 预期的可训练集合 == 发布产物里的键（按名字和元素数比对）。
- 诊断（不作判据）：单 batch 过拟合的曲线、各组的梯度范数。

**M1 通过条件**：G0.1～G0.6、G0.8 全部通过，或者已经选定回退方案并且回退方案通过验收（例如 G0.3 改用纯 PyTorch 的 GDN）；A1.1～A1.5、A2.1～A2.9、A3.1～A3.5、A4.1～A4.6 全部通过；小模型的显存和内存存活对象清单（6.4 节）有实测值。

### P5　真实模型的加载和前向（1～2 天）

- A5.1：加载后实测显存和内存，并填写 6.4 节的存活对象清单。
- A5.2 V1：取一条 128 token 的提示，用 CPU fp32 参考实现逐层流式计算前 4 层（每次只反量化一层），和训练进程的隐藏状态对比，E3，每层相对 L2 ≤1e-2。
- A5.3（可选）：和 FreeToken 对比。**FreeToken 的公共接口不一定能拿到 teacher-forcing 的 logits**，需要先写一个内部工具：直接调用模型的前向，拿到每个位置的 logits。有了这个工具，再在固定文本上比较 token 级的 KL 和 top-1 一致率，只记录不设阈值。FreeToken 的 FP8 层是 W8A8，有差异是预期的。
- A5.4：贪心生成几条短回答（关闭思考，用 Speed 的模板），检查语义是否正常。
- A5.5：记录每次前向的耗时拆分：H2D、反量化、计算、等待。

### P6　C3 训练（1～2 天）

**配置**
- LoRA 挂在全注意力的 q/k/v/o（10 层）、GDN 的 `in_proj_qkv`/`in_proj_z`/`out_proj`（30 层）、共享专家的 gate/up/down（40 层）。r=16，alpha=32，dropout=0，共 1917 万参数。
- AdamW，学习率 1e-4～2e-4，warmup 10 步，梯度裁剪 1.0，权重衰减 0。
- 每个 micro-batch 一条样本，长度先 ≤2048、再到 4096；梯度累积 8。
- 非重入梯度检查点；`output_router_logits=False`；`use_cache=False`；注意力用 `sdpa`。

**数据**
- **金丝雀**：约 300 条训练、50 条留出。留出部分换一种说法提问，用来区分「记住了」和「学会了」。
- **回放**：约 300 条，答案由基座生成。
- 全部按 6.1 节渲染。

**验收**
- A6.1 E2：第 0 步，adapter 为零和无 adapter 时的 loss 逐位一致。
- A6.2 V1：选 2 层，截取真实激活，和参考实现比较各 LoRA 模块的梯度（E3）。
- A6.3 V2：同 A4.3。
- A6.4 V3：
  - 留出的金丝雀测试集上，基座和 adapter 的成功率对比（贪心解码）；**阈值由 @liboqi 定**，至少要显著高于基座；
  - adapter 开和关时，在金丝雀提示上的 logits 差异非零；
  - 回放集上的 loss 变化在事先定好的范围内。
- A6.5：冻结权重的哈希不变；优化器参数集合检查通过。
- A6.6：在 6.4 节的各阶段实测峰值；记录吞吐和每步的耗时拆分；记录搬运计数。

### P7　专家 LoRA（2～4 天）

- 配置：C3 + 全部路由专家 r_e=2（1.57 亿参数，fp32 + Adam 约 2.5 GB【估算】）。显存允许再试 r_e=4。不够时考虑三种办法：8bit 优化器（要另行评估 bitsandbytes 在原生 Windows 上的可用性）、专家 LoRA 的优化器放到 CPU、只在一半的层上挂专家 LoRA。
- A7.1 V1：A2.2 在真实权重和真实激活上重做（2 层）。
- **A7.2（按 D13 修订）**：
  - 本步没有被路由到的专家，**本步的原始梯度**严格为 0；
  - 从初始化起**从未被路由过**的专家，参数保持不变。这时 Adam 的状态全为 0，更新量是 0；
  - **曾经被路由过、本步没有路由到的专家，因为动量而继续变化，这是预期行为，不算失败**。
- A7.3：A6.1～A6.6 重做；和 C3 在同样数据上比较。

### P8　产物与续训检查点（1 天）

**发布产物**（给推理用）
```
adapter/
  adapter_config.json, adapter_model.safetensors   # PEFT 格式：线性层部分（CausalLM 键名）
  expert_lora.safetensors                          # model.layers.{L}.mlp.experts.{gate,up,down}_proj.lora_A [E,r,in] / .lora_B [E,out,r]
  manifest.json
```

`manifest.json` 包含：
- `layout_version: "nvfp4-lora-v1"`；
- 基座的仓库名、revision 和分片 sha256；
- 目标模块，以及每组的 r、alpha、scaling、是否 rsLoRA（否）、dtype；
- 专家 LoRA 的布局说明；
- 分词器和模板的哈希，以及 `enable_thinking=False`；
- 训练配置的哈希和代码 commit。

**训练检查点**（只用于续训，**只在优化器 step 的边界保存**）
```
ckpt-step{N}/
  lora_params.safetensors     # 所有可训练参数（fp32），包括专家 LoRA
  optimizer.pt                # 优化器状态（按参数名对应，不按位置）
  scheduler.pt
  rng.pt                      # python、numpy、torch CPU、CUDA 的随机状态
  data_state.json             # epoch、采样器的打乱种子和当前位置、已消耗的样本数
  trainer_state.json          # global_step、累积的 token 数、配置哈希
```
**限制**：不支持在梯度累积的中途保存。

**验收**
- A8.1 E2：从发布产物重建的 ΔW = scaling × B·A，和训练进程内存里算出的逐位一致。
- A8.2 V3：同 A4.4。
- A8.3 V2：同 A4.3（真实模型上，用少量步数验证）。

**不做**：把 LoRA 合并进 NVFP4 专家。FP8 层的合并另行评估，见第二轮报告 3.4 节。

---

## 6. 横向的检查体系

### 6.1 模板、标签和损失

- **渲染**：用 checkpoint 自带的 `chat_template.jinja`，`apply_chat_template(..., enable_thinking=False)`。Speed 默认关闭思考（Speed `freetoken.py:1390`），也支持开关和采样参数覆盖，以 codex 核对的 Speed 版本为准；工具参数目前不接受。**首期选择非思考**，这是我们的选择，不是产品的限制。
- **标签**：labels 与 input_ids 对齐后再移一位，即 `labels[t] = input_ids[t+1]`，或者沿用 transformers 内部做移位的约定，二者择一并写进测试。只对最后一轮答案的 token 和 `<|im_end|>` 计 loss，`<|im_end|>` 之后的换行不计，其余位置为 −100。
- **截断**：截断后答案必须完整。超长样本要么从左截断提示，要么丢弃，不能截掉答案。
- **padding**：padding 的位置 label 为 −100，attention mask 为 0。v1 每个 micro-batch 只有一条样本，不需要 padding。
- **损失归一化**：一个优化器 step 的 loss = 这个 step 内所有被监督 token 的交叉熵之和 ÷ 这个 step 内被监督 token 的总数。每个 micro-batch 按这个全局分母缩放后再 backward，不要「每个 micro-batch 先取平均再平均」。
- **分块交叉熵（D11）**：从最后的隐藏状态 h [T, H] 和 lm_head 权重出发，按 512 token（可调）一块计算 logits 和交叉熵，每块用检查点包起来，**整个过程不生成完整的 logits**。
  - 测试：小词表、短序列上和普通交叉熵对比，loss 和对 h、对 lm_head LoRA（如果有）的梯度在 E3 内。
- **测试（至少 200 条样本）**：
  - 移位后，被监督位置的 label token ID 逐个等于答案的 token ID 加上 `<|im_end|>` 的 ID；
  - 不被监督的位置全部是 −100；
  - 提示部分和 Speed 实际发给模型的提示逐 token 一致（Speed 用同一份模板，`engine_bridge.py:408-421`）；
  - 截断后的样本答案完整。

### 6.2 容差和参考实现

| 对比 | 参考实现 | 容差 |
|---|---|---|
| 反量化 | 2.3 节的参考实现；FreeToken 的 kernel | E1 字节一致；数值 ≤1 ulp |
| 专家算子 | 同精度策略的稠密参考实现 | bf16：前向相对 L2 ≤1e-2，梯度 ≤2e-2；fp32：≤1e-5；fp64 gradcheck |
| GDN kernel | 纯 torch 参考实现（fp32） | 相对 L2 ≤1e-2（bf16 输入时） |
| 小模型训练 | PEFT + 反量化模型 | 见 A4.1、A4.2 |
| 真实模型的前几层 | CPU fp32 逐层流式参考实现 | 每层相对 L2 ≤1e-2 |

这些容差是起点。**如果实测稳定地好得多或差一些，按实测和原理调整，并记录理由**，例如 bf16 的 K=2048 累加误差本身的量级。

### 6.3 每次训练自动检查的不变量

- 第 0 步，adapter 为零和无 adapter 时 E2 一致；
- 冻结权重的哈希不变；
- 出现 NaN 或 Inf 立刻停下，并保存当前 batch；
- 实际优化器的参数集合 == 预期的可训练集合 == 发布产物里的键；
- 本步没有被路由到的专家，原始梯度为 0。

### 6.4 显存和内存：存活对象清单（必须实测填写）

| 对象 | dtype | 设备 | 大小公式【估算值】 | 生命周期 |
|---|---|---|---|---|
| 非专家权重 | bf16 | GPU | 约 2.45B 参数 × 2 字节【4.9 GB】 | 常驻 |
| 专家层槽 | uint8 | GPU | 槽数 × 453 MB【2 槽 0.9 GB】 | 常驻 |
| 反量化临时空间 | bf16 | GPU | 块大小 C × 3 × 512 × 2048 × 2 字节【C=32 时 0.2 GB】 | 每个块 |
| 专家激活（排序后的 x、gate/up、h） | bf16 / fp32 | GPU | 约 8T × (2048×2 + 1024×4 + 512×4) 字节【T=4096 时约 0.4 GB】 | 每层前向或重算期间 |
| 检查点保存的层输入 | bf16 | GPU | 40 × T × 2048 × 2 字节【T=4096 时 0.67 GB】 | 前向到反向 |
| 单层重算和反向的峰值 | 混合 | GPU | 实测【约 1.5 GB】 | 每层 |
| 交叉熵的块 | fp32 | GPU | 块大小 × 248320 × 4 字节 × 约 2【512 时约 1 GB】 | 每块 |
| LoRA 参数、梯度、Adam 状态 | fp32 | GPU | 参数量 × 16 字节【C3 0.31 GB；专家 r_e=2 再加 2.5 GB】 | 常驻；Adam 状态在第一步才分配 |
| CUDA 上下文和碎片 | — | GPU | 实测【1.5～2 GB】 | 常驻 |
| 专家存储 | uint8 | CPU 锁页 | 18.1 GB | 常驻 |
| 加载时的非专家中转 | bf16 | CPU | 【约 4.9 GB】 | 只在加载时 |
| 保存时的拷贝 | fp32 | CPU | 参数量 × 12 字节左右 | 只在保存时 |

**要分别实测峰值的阶段**：加载完成、第一个优化器 step（Adam 状态在这时分配）、稳态、重算峰值、评测、保存、恢复。

**工具**：每个阶段前后调用 `torch.cuda.reset_peak_memory_stats()` 和 `max_memory_allocated/reserved`；第一步用 `torch.cuda.memory._record_memory_history` 导出内存快照；CPU 侧记录 RSS 和锁页字节数。

**搬运和重叠的确认**：用 streamer 的计数器统计每步拷贝的字节数、未命中次数、等待时间；用 `torch.profiler` 看拷贝流和计算流是否重叠。

**注意**：分块只限制了反量化的临时空间；层槽、专家 LoRA 的参数和状态仍然随专家数增长。

### 6.5 每次运行都记录

- **每步**：loss、被监督的 token 数、学习率、总梯度范数和各组梯度范数、耗时拆分。
- **每 N 步**：各组 LoRA 权重范数、留出集 loss（分块计算）、金丝雀成功率、探测 batch 的路由分布。
- **一次性**：环境、配置、数据哈希、代码 commit、各阶段峰值。

---

## 7. 风险和回退（原生 Windows）

| 风险 | 应对 |
|---|---|
| flash-linear-attention 在 triton-windows 上跑不了训练 | 自己用纯 PyTorch 写分块的 gated delta rule，见 G0.3 |
| 18 GB 锁页内存分配失败或很慢 | 可分页内存 + 锁页中转缓冲 |
| `torch._grouped_mm` 不可用 | 逐专家循环 |
| 吞吐太低 | 按 6.4 节的计数器和 profiler 定位；Triton 反量化（可借用 Axolotl 的 `dequant_grouped.py`，需先确认它能在 triton-windows 上运行）；「按层优先调度」（见 #后训练:14ee91f5） |
| FP8 层的激活截断 | 在任务数据上统计 max\|x\| / (448 × input_scale)；大于 1 时记录并上报；必要时在训练中加 FP8 伪量化（梯度直通） |
| 专家 LoRA 显存不够 | 见 P7 |
| 环境必须偏离 Speed | 记录原因，评估对 A′ 的影响，上报 |

## 8. 时间粗估

| 阶段 | 时间 |
|---|---|
| P0 | 0.5～1 天 |
| P1 | 1～2 天 |
| P2 | 2～3 天 |
| P3 | 1～2 天 |
| P4 | 2～3 天 |
| **M1 合计** | **约 1.5～2 周** |
| P5～P8（M2） | 约 1～1.5 周 |

如果 G0.3 需要自己写纯 PyTorch 的 GDN，再加 3～5 天。

---

## 9. G2 / G3 大纲

**G2：FreeToken 推理**（详见第二轮报告第 6 节）
1. **先做注意力、GDN、共享专家的不合并旁路**：
   - 按段映射：`qkv_proj` = [q_proj（query 和 gate 按头交错），k，v]；`in_proj_qkvz` = [qkv, z]；`in_proj_ba` = [b, a]，**b 在前**；共享专家的 `gate_up` = [gate, up]；
   - adapter 在 CUDA Graph 录制之前加载。
2. **再做路由专家的旁路**：
   - decode 时会把 `topk_ids` 原地改写成缓存槽位号，所以要先复制原始的专家编号；
   - 在 kernel 改写 hidden_states 之前读出 x；
   - down 的增量要乘 `topk_weight`；
   - 开 LoRA 时强制使用 Triton 后端。
3. **验收**：
   - 从 adapter 文件重建的 ΔW 和 FreeToken 加载后读回的一致；
   - 用「探针 adapter」抓段错位；
   - 逐模块比较「带 LoRA − 不带 LoRA」的增量（E3）；
   - 金丝雀在 FreeToken 上的结果和训练进程一致；
   - 前缀缓存、CUDA Graph、专家换出再换入这三种状态下，结果都一致。

**G3：领域效果**（详见第二轮报告第 4、5 节）：先做路由统计，再在对齐的参数预算下比较 C3 和专家 LoRA；在 NVFP4 的部署路径上评测，同时测通用能力有没有退化。

## 10. A′：嵌入 Speed（大纲）

- 核心库作为 Speed 的一个后端，环境和 Speed 完全一致（这也是第 4 节要求对齐的原因）。
- 需要另行设计：
  - 训练任务的调度；
  - 和推理争用 GPU（16 GB 卡上不能同时训练和推理）；
  - 产物的管理和版本绑定（manifest 里的基座哈希）；
  - 训练中断和恢复（P8 的训练检查点）。

## 11. 工作方式

### 11.1 代码结构（建议）

```
nvfp4lora/
  format/   nvfp4.py  fp8.py  inspect.py  preprocess.py
  store/    expert_store.py  streamer.py
  ops/      experts_fn.py  chunked_ce.py  gdn_torch.py（如果需要回退）
  model/    build.py  packed_experts.py
  train/    data.py  template.py  loop.py  ckpt.py  metrics.py
  export/   artifacts.py  to_peft_param_wrapper.py（可选）
  tools/    bench_h2d.py  check_gdn.py  make_tiny_model.py  quantize_tiny.py
  tests/    ...
  docs/     env.md  decisions.md  results.md
```

### 11.2 决策日志（`docs/decisions.md`）

每条写清：日期、阶段或关卡、文档原文、实测结果和数据、做出的决定、对后续的影响、是否涉及硬约束。

### 11.3 每个阶段结束时汇报

汇报内容：通过和未通过的验收，关键数据，新增的决策，下一阶段的风险。

---

## 附录 A：可以借用的开源代码

| 来源 | 内容 | 注意 |
|---|---|---|
| Axolotl（Apache-2.0）`scattermoe_lora/nvfp4_moe_loading.py:152-236` | `inspect_nvfp4_layout` | 保留版权声明 |
| Axolotl `scattermoe_lora/dequant_grouped.py` | Triton 版 NVFP4 反量化 | 需要确认能在 triton-windows 上运行；数值先对照参考实现 |
| Axolotl `sonicmoe/nvfp4_quant.py:140-251` | NVFP4 量化函数 | 做小模型的 NVFP4 checkpoint 时参考 |
| FreeToken `models/qwen3_5_moe/weight.py:83-93` | `_dequant_nvfp4` | 推理端的对照 |
| transformers `integrations/moe.py:550-614` | 专家实现的注册接口 | 备选的接入方式 |

## 附录 B：可选扩展的接入要点（不在交付范围，仅供以后参考）

- **LLaMA-Factory**（c683e59）：
  - 在 `load_config` 里去掉 `quantization_config`（`loader.py:128`）；
  - 在 `apply_liger_kernel` 之后替换专家类（`loader.py:148`；Liger 会重新绑定 `Qwen3_5MoeExperts`）；
  - 在 `patch_model` 和 `init_adapter` 之间挂上 streamer（`:184`～`:187`）；
  - 训练时 LF 会强制 `device_map` 到 cuda（`parser.py:673`）；
  - `_init_weights` 会按类名初始化专家（`modeling_qwen3_5_moe.py:1022-1024`）；
  - 模板 `qwen3_6` 要设 `enable_thinking: false`（`data_args.py:124`）；
  - 梯度检查点默认是重入模式（`model_args.py:149-151`）；
  - **LF 锁定 transformers ≤5.8、PEFT ≤0.18.1，和 Speed 冲突，需要独立环境。**
- **DeepSpeed ZeRO-3**：
  - 打包数据必须装进**创建时就是 bf16** 的参数，`zero.Init` 会把创建时的浮点张量转成 bf16；
  - 不用 `offload_optimizer`；
  - 保存 PEFT 时会把全部专家 all-gather 一遍；
  - Windows 上用 gloo，集合通信未经验证。

## 附录 C：v1 → v2 修订记录

1. 明确交付形态 A → A′，不交付 LF 和 ZeRO-3，并写明这偏离了最初的设想（0.1 节）。
2. 只用原生 Windows，删除所有 WSL 回退；新增 GDN 的纯 PyTorch 回退（G0.3）。
3. 环境与 Speed 对齐，重写依赖表（第 4 节）；flash-linear-attention 不再禁止 0.5，由测试决定。
4. 新增数值一致性三级（1.2 节），全文的「逐位一致」都按这三级重写。
5. 重写专家算子的输入和梯度协议、精度策略，反向改为块内 `autograd.grad`（P2）。
6. 重写 streamer 的槽位协议和阶段提示，先同步后异步（P3）。
7. 专家 LoRA 改为 PackedExperts 上的 nn.Parameter，并加入参数集合一致性检查（D9、A4.6）。
8. 新增训练检查点规范，区分发布产物和续训检查点（P8）。
9. 预算改为存活对象清单，要求分阶段实测（6.4 节）。
10. 分块交叉熵改为从隐藏状态出发，补上移位、mask、截断和归一化的要求（6.1 节）。
11. 验收拆成四类（0.3 节）；去掉「200 步降到 5%」「金丝雀 95%」「PPL 几到十几」等经验阈值；FreeToken 的 logits 对照降为可选，并说明需要先写内部工具。
12. A7.2 按 Adam 的默认语义修订（D13）。
13. 里程碑：M1 = P0～P4，用目标形状验收；codex 之前的预研不算完成。

## 附录 D：对 codex 14 条意见的逐条回应

| # | codex 的意见 | 回应 | 修改位置 |
|---|---|---|---|
| 1 | 交付架构不清，LF 和 ZeRO-3 是否必交付 | 接受。经 @liboqi 确认：交付形态 A → A′，不交付 LF 和 ZeRO-3，并明确写出偏离 | 0.1 节、附录 B |
| 2 | autograd 接口：tuple 里的张量不被追踪，返回值和输入不对应 | 完全接受，v1 写错了。改为所有 A/B 作为顶层参数，并给出完整的返回协议 | P2 |
| 3 | 反向重算时 autocast 状态不会恢复；内层 backward 会重复写梯度 | 完全接受。Function 内部关闭 autocast，显式规定精度；改用块内 `autograd.grad` 并作为返回值交回，每个专家只写一次 | D5、D7、P2 |
| 4 | 未命中专家「不更新」不符合 Adam | 完全接受，v1 写错了。验收区分「本步原始梯度为 0」和「从未被路由过的参数不变」；动量带来的更新按 Adam 的默认语义接受 | D13、A7.2 |
| 5 | 「逐位一致」承诺不成立 | 完全接受。定义 E1、E2、E3 三级，全文重写 | 1.2 节、D8 |
| 6 | 用 `is_grad_enabled` 判断方向不可靠；引用计数不代表 GPU 读完 | 完全接受。改为阶段提示 + generation + 拷贝完成和读者完成事件；预测错了只影响性能；加入竞争测试 | D6、P3 |
| 7 | 「不能有任何 Parameter」过宽 | 接受。只有冻结的打包专家不能是 Parameter；专家 LoRA 是 nn.Parameter，归属、优化器和保存路径都明确，并自动比对三个集合 | D9、A4.6、6.3 节 |
| 8 | adapter 不足以精确续训 | 接受。新增训练检查点规范，只在优化器 step 的边界保存 | P8 |
| 9 | 依赖版本需要重新论证 | 接受。和 Speed 对齐（torch 2.11.0+cu130、transformers 5.16.x、triton-windows 以实装版本为准），PEFT 先试 0.18.1；不再照搬 LF 的版本组合；不改 Speed 的环境 | 第 4 节 |
| 10 | 对 FLA 0.5 的禁止缺乏依据 | 接受。#792 不足以作为禁止的依据；改为由 SM120 原生 Windows 上的真实形状测试决定。WSL 已经由 @liboqi 明确排除 | G0.3 |
| 11 | 预算需要可测量的边界 | 接受。改为存活对象清单，分阶段实测，并用计数器和 profiler 确认搬运和重叠 | 6.4 节 |
| 12 | 分块交叉熵必须在生成完整 logits 之前 | 接受。从隐藏状态分块计算，补上移位、mask、截断、EOS、padding 和归一化的要求，测试改为按 label ID 比对 | D11、6.1 节 |
| 13 | 产品能力描述和阈值需要收紧 | 接受。改为「首期选择非思考」；去掉经验阈值；验收拆成四类；FreeToken 的 logits 对照需要先写内部工具 | 0.3 节、6.1 节、P5、P6 |
| 14 | 第一里程碑的通过条件 | 接受。M1 = 修订后的 P0～P4，用目标形状验收；之前的预研不算完成 | 第 5 节开头、P4 末尾 |
