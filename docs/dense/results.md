# Dense 27B 实测结果

Qwen3.8完成完整D0-D6及真实100步、50→100十项状态E2、发布重建；Qwen3.6完成设计规定的来源/预处理/前向/机制/短10和发布重建。两个实际adapter均通过FreeToken完整64层原生Windows推理、全语料、缓存重建和新进程E2。DA7.1 MoE零回归通过。最终远端发布的HEAD在交付时只读核对。

整体技术资格保留两项有记录的调整：DA0.7默认HF GQA路径原FAIL，使用经过独立控制的dense-only efficient SDPA；DA2.3真实FP32相对FP64原1e-6有FAIL，普通autograd E2及36组控制支持记录的2e-6补充判据。原FAIL没有改成PASS。生产PackedLinear数值策略未改；两模型真实CPU FP32原无条件1%均通过。详见[历史与依据](numerical-history.md)、[决策](../decisions.md#dense-27b)。

## 逐项DA验收

下面保留原失败及替代检查的独立行。数值、命令、源代码快照、退出码以链接的summary/manifest为准；原始大文件在仓外。静态发布行不伪造尚未写入Git的自身SHA，最终remote-inclusive审核另行给出。

| DA | 状态 | 实测值与范围 | 证据 |
| --- | --- | --- | --- |
| DA0.1 | PASS | Independent checkout/branch from target origin/main `efdfc55a26d9531c904470a55ab35793004ed104`; original files archived in first commit `cb5d5d8`. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.2 | PASS | Independent venv; all 39 locked versions and wheel SHA256 match, hash-required offline install, pip check passes; freeze differs only in editable binding. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.3 / B0 | PASS | Unmodified main: 41 unit tests; MoE tiny100/50-to-100/reload all pass. Both real initial probes FP32 bits `1083229253`; both logical configs and hashes retained. This is a baseline, not D7 regression. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.4 | PASS | RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30: all three complete files match upstream LFS SHA256. Bounded ranges and the parameterized repository fetcher independently verify the files; full source receipt retained. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.5 | PASS | Real text architecture on meta: 64 layers, 26,895,998,464 text parameters, all 400 quantized linear shapes recorded and required projection dimensions checked. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.6 | PASS | Independent short recurrence: worst relative L2 7.983e-5 across output/all five input gradients; production BF16 beta case 6.922e-5. Core and full HF GDN module at T2048/T4096 pass both checkpoint modes E2. Full-module T4096 max allocated 4,042,380,800 B, reserved 4,441,767,936 B; frozen synthetic BF16 projections, no LoRA, not whole-model memory. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.7 default HF | FAIL | No-mask GQA selects math; allocated peak 6,979,321,856 B. Failure retained. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.7 dense fallback | FALLBACK PASS | Explicit KV repeat uses existing efficient backend with/without padding: allocated 525,074,432/659,292,160 B; repeated forward/backward E2; all output/input-gradient errors vs math <1%. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.8 | MEASURED | Six forward/dx shape cases at T2048/4096: 93.51-100.66 TFLOPS. Gate NVFP4 decode 5.42ms/89,128,960 elements (16.46G elements/s); FP8 0.32ms/52,428,800 (163.58G elements/s). Synthetic valid packed bytes, not source equivalence evidence. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA0.9 | PASS | Logical payload including scalar fields 16,840,133,760 B; physical pinned 17,179,869,184 B (16GiB), allocation 3.05s. Full-bank H2D 328.00ms/51.34GB/s, not an overlap/whole-model throughput proof. | [d0-summary.json](../evidence/dense/d0-summary.json) |
| DA1.1 | PASS | Real 64-layer geometry matches; 400 quantized linears, 1,392 streamed fields. Tiny contains 8 deliberately misleading MTP/visual keys; all excluded. Real excludes 348 non-text tensors. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA1.2 | PASS | Every streamed byte compared after serialization and independently from original shards to loaded CPU store. Real payload 16,840,133,760 B. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA1.3 | PASS | All 7,214,202,880 FP8 weight code elements convert exactly to BF16. Optional 36 real MLP samples over gate/up/down in layers 0/3/31/63 match FreeToken raw-FP32-global decoding at 0 ULP; FP16 runtime global differs by at most 1 ULP, reported separately. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA1.4 | PASS | Every layer below 256MiB, max 265,748,528 B; index/header/file SHA checked. Independent store reload reproduces all payload hashes. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA1.5 | PASS | Real preprocessing peak working set 6,071,853,056 B. Independent pageable-store validation is a separate process and not included in this preprocessing peak. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA1.6 | PASS | Nonstream payload 5,138,070,528 B; BF16 embed/head/small-parameter keys complete. No streamed MLP/attention weight decoded into these files. Head uses existing nonexpert policy; source config/tokenizer and output hashes recorded. | [d1-summary.json](../evidence/dense/d1-summary.json) |
| DA2.1 | PASS | All ten real projection shapes at T4096: forward and dx E2 vs independently materialized same-expression ordinary autograd. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.2 | PASS | NVFP4 and FP8 small FP64 gradcheck; no saved base tensors or Tensor attributes in ctx. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.3 original | PARTIAL / FAIL | Small shapes pass original 1e-6; real down_proj forward 1.0246188661e-6 fails. Ordinary FP32 autograd matches that result E2. Original failed run retained. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.3 amended | QUALIFIED | All ten real shapes satisfy ordinary-autograd E2 plus FP64 relative L2 <=2e-6. Thirty-six predeclared held-out controls over down/gate/o, three seeds and four lengths: both precision modes E2 vs ordinary autograd, max L2 1.31958903495e-6. See the dated decision; this is not original-threshold PASS. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.4 | PASS | Max direct decoder peak delta 287,310,336 B, returned live delta 0; max extra weight workspace over materialized control 178,257,920 B, within matrix+chunk bound. Full forward/backward peak delta max 891,289,600 B includes required activation/cast workspace and is separately reported. No saved base tensor; gate allocator snapshot retained outside Git with SHA256. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.5 | PASS | All ten real shape repeated forward/dx E2. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.6 | PASS | All ten real shapes: two-backward accumulation, reentrant checkpoint and non-reentrant checkpoint E2. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA2.7 | PASS | All ten real shapes: B=0 enabled/disabled output E2; nonzero LoRA output/dx/dA/dB E2 vs handwritten ordinary autograd expression. | [d2-summary.json](../evidence/dense/d2-summary.json) |
| DA3.1 | PASS | 1,000 random acquires, all 64 layers covered; 263,169,033,008 full payload bytes read back and compared directly with CPU source, not just hashes. | [d3-summary.json](../evidence/dense/d3-summary.json) |
| DA3.2 | PASS | 128 full-payload reads with 50,000,000 GPU sleep cycles per request; released readers survive subsequent slot reuse. Pins, stale generation, double release, close and alternate compute stream guards pass. | [d3-summary.json](../evidence/dense/d3-summary.json) |
| DA3.3 | PASS | 128 deliberately wrong prefetches; full bytes equal source and combined readback SHA matches no-prefetch run. All 13 operator projections also match no-prefetch output/dx/dA/dB E2. | [d3-summary.json](../evidence/dense/d3-summary.json) |
| DA3.4 | PASS | GDN and full-attention layers, 13 projections covering ten shapes, T4096: synchronous vs asynchronous forward/dx/LoRA A/B gradients E2. | [d3-summary.json](../evidence/dense/d3-summary.json) |
| DA3.5 | PASS | Per-layer acquires/hits/copies/bytes/GPU copy and wait times retained. Random read run: 999 hits, 973 copies, 256,056,737,264 H2D bytes; event wait total 13.17ms. | [d3-summary.json](../evidence/dense/d3-summary.json) |
| DA4.1 | PASS | Tiny and actual real construction: no meta parameters/buffers, exact nonstream keys, FP32 RoPE, ordinary CPU embedding, expected canonical trainables. Actual real D-ALL r16: 800 FP32 CUDA tensors / 108,789,760 elements; all B=0; 400 packed projections. Fingerprint mutation controls detect CPU table, raw bank and frozen GPU norm changes. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.2 same expression | PASS | 100 optimizer steps, two microbatches each. Every loss E2; first gradients relative L2=0; final weights relative L2=0. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.2 HF+PEFT | PASS | Independent HF BF16 pre-scaled frozen Linear weights + ordinary PEFT. First gradient min cosine 0.9995077116; max relative L2 0.0316625 is reported, while the design criterion is cosine>=0.999. Max per-step relative loss 0.000251277849 (<0.02); final weight min cosine 0.9998401944. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.3 | PASS | New process step50-to100: all ten checkpoint/history/probe/DeltaW checks E2; all 50 subsequent losses, sample IDs, LR, gradient norms and consumed RNG values agree. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.4 | PASS | New process reconstructs from preprocessed bundle+publish: all parameter bytes, frozen fingerprint, enabled/disabled FP32 probe loss and 25 DeltaW fingerprints E2. Standard PEFT reads published files on independent HF BF16 base: relative probe loss 0.000110244147 (<0.02). | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.5 | PASS | Optimizer IDs == expected canonical set == 50 published tensors. Production and both references' frozen fingerprints unchanged. Tiny probe enabled7.4265113 vs disabled7.6956329 demonstrates effect in this synthetic test. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA4.6 | PASS | 56 unit tests. Feature MoE tiny2 matches original B0 first two metric rows E2; final states/probe match an unchanged-main two-step supplement E2. Both MoE logical configs/hashes unchanged; original MoE reader rejects dense release layout. | [d4-summary.json](../evidence/dense/d4-summary.json) |
| DA5.1 | PASS | Load allocated/reserved3,573,386,240/3,586,129,920 B. Observed128-token forward peak3,906,082,304/3,961,520,128 B; whole-card remaining11,214MiB. Raw layer banks16GiB pinned plus ordinary CPU table2,542,796,800 B; full allocator/process snapshots retained. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA5.2 | PASS | Original unconditional CPU FP32 first-four-layer relative L2:0.0029163732,0.0029407353,0.0040695826,0.0033973938, all<=0.01. Token-recursive GDN and eager attention, one FP32 layer at a time. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA5.3 | PASS | Greedy outputs2,北京,Hello; each followed immediately by im_end248046. Short uncached sanity only. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA5.4 | PASS | One128-token instrumented forward: H2D332.21ms, decode1218.13ms, projection GEMM99.78ms; GDN inclusive543.65ms and non-projection467.66ms; attention inclusive51.54ms and non-projection26.71ms. Copy/compute overlap and enqueue gaps mean the intervals cannot be summed. Actual SDPA operator count:16 efficient, zero math; repeated hidden states byte-E2. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA5.5 | PASS: observation complete | All208 FP8 source projections observed. Three ratios above original448*input_scale range, max1.022831; layer51 o_proj and layer62 in_proj_qkv/in_proj_z. This does not certify zero clipping or W8A8 equivalence. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA5.6 | PASS | Dense tokenizer/template200 Speed AST goldens; every prompt/label, forced left truncation and answer-too-long rejection passes. Tokenizer options match Speed. | [d5-summary.json](../evidence/dense/d5-summary.json) |
| DA6.0 | PASS | Canary300/holdout50 text bytes equal original corpus. Replay300 generated by this dense base; combined training600. Base holdout0/50, replay981 supervised tokens, NLL0.00023400843633333113 nat/token. | [d6-preflight-summary.json](../evidence/dense/d6-preflight-summary.json) |
| DA6.1 | PASS | Real1 step x accumulation8. Initial enabled/disabled FP32bits1083850783, loss4.820327281951904. All initial B gradients nonzero. Thirteen projections across GDN/full-attention layers: same-expression output/dx/A/B E2, FP32 E3, eight-micro actual A/B sum E2. Updated probe4.715414047241211; disabling restores original bits. Frozen fingerprint unchanged. | [d6-preflight-summary.json](../evidence/dense/d6-preflight-summary.json) |
| DA6.2 2048 | PASS | Two actual optimizer steps x8:129.03/128.47s. Peak allocated9,005,052,416B/reserved9,483,321,344B; monitored whole-card minimum5938MiB. | [d6-preflight-summary.json](../evidence/dense/d6-preflight-summary.json) |
| DA6.2 4096 | PASS | Two actual optimizer steps x8:284.10/284.72s. Peak allocated12,996,858,368B/reserved13,398,704,128B; monitored whole-card minimum2204MiB (>1024MiB). O1/O3/O4 not needed. | [d6-preflight-summary.json](../evidence/dense/d6-preflight-summary.json) |
| DA6.2 recomputation | PASS | Both lengths restore step2 Adam870,318,080B; all64 layers, complete allocation history, loss and all800 gradients U8 E2 against uninstrumented control. First-micro and steady-state snapshots: decoded allocations occurred, none remain live after backward. | [d6-preflight-summary.json](../evidence/dense/d6-preflight-summary.json) |
| DA6.3 | CONFIRMED 2026-10-30 | Actual owner reply approves holdout>=45/50 and replay NLL increase<=0.01. Full quote, recorded time and baseline are in effect-threshold-owner-decision.json. | [effect-threshold-owner-decision.json](../evidence/dense/effect-threshold-owner-decision.json) |
| DA6.4 | PASS | D-ALL r16, max_length2048, accumulation8,100 steps; evaluation10/20/50/100; complete20/50/100 checkpoints. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA6.5 | PASS | All50 resumed step histories and all ten named state/probe/DeltaW checks E2; zero differences. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA6.6 | PASS | New process using bundle+publish only: all800 parameter tensors, enabled/disabled FP32 probe bits and all400 DeltaW fingerprints E2. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA6.7 | PASS | Reconstructed published adapter:50/50 (base0/50); replay NLL3.3538927875694635e-7, increase-0.00023367304705457418; both approved thresholds satisfied. Prompt247,983 vocabulary entries change, max absolute logit change43.0625. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA6.8 | PASS: observation complete | All208 original FP8 scales,600 samples, disabled/enabled phases. Above-range modules74/73, max ratio1.3784860/1.3944223. No zero-clipping or W8A8 equivalence claim. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA6.9 | PASS | Restored100, one8-micro accumulation,303 input tokens; no optimizer update. Loss/all800 gradients E2 against uninstrumented control; all module intervals complete.256 efficient forward calls including recompute,128 efficient backward, zero math. | [d6-formal-summary.json](../evidence/dense/d6-formal-summary.json) |
| DA7.1 | PASS | 58 tests; original B0 41 retained. MoE tiny100 histories/parameters/states E2; both real probes FP32bits1083229253, both logical configs unchanged; original delivered C3/expert-r2 step50 full restore/route-audit identity E2, zero optimizer updates; old reader rejects both final dense layouts. | [D7](../evidence/dense/d7-summary.json) |
| DA7.2 | PASS | 本页、progress、training-cli、decisions和README新增dense入口；原MoE结论文字保持原字节前缀。 | [CLI](training-cli.md) |
| DA7.3 | 交付时核对 | 仅推两条独立feature，禁止main/merge/force。最终HEAD以交付回复和仓外remote-inclusive严格审核为准；避免将包含自身HEAD的回执再次提交。 | [发布核对说明](#发布与证据) |
| Qwen3.6 shortened | PASS | 官方三分片全SHA；全E1；CPU四层0.2890%/0.3245%/0.5442%/0.4228%；own replay300；机制；实际10步；800 LoRA参数/400DeltaW重建E2。可选额外100未执行。 | [Q36](../evidence/dense/qwen36-summary.json) |
| FreeToken final | PASS | 两完整模型：off0/50→on50/50、replay均300/300；各700记录和完整首logits新进程E2、zeroB/开关/缓存重建及64bank不变；54源码测试。 | [fork报告](https://github.com/wenlive/FreeToken/blob/feat/dense27b-nvfp4-lora/docs/dense27b_nvfp4_lora.md) |

## 真实训练与adapter效果

| 项目 | Qwen3.8 | Qwen3.6 |
| --- | --- | --- |
| 固定源 | RadixArk@319f741cce68d7914884900c138a1fbb70a42f30 | nvidia@0893e1606ff3d5f97a441f405d5fc541a6bdf404 |
| 实际optimizer steps |100 |10（logical max100/stop10，非完整100） |
| LoRA | D-ALL r16/alpha32，800张量/108,789,760 FP32元素 | 相同 |
| 优化器 | AdamW，weight_decay0，默认betas/eps，foreach/fused均False；lr2e-4、warmup10、clip1 | 相同 |
| 数据/模板 |300原canary+300自己的base replay，独立留出50；max_length2048，acc8，CE chunk128，非思考 | 相同规则，各自基座生成和绑定 |
| 留出 |0/50→50/50 |0/50→50/50 |
| 回放NLL nat/监督token |0.00023400843633333113→3.3538927875694635e-7 |0.00018482663403820553→6.259719674569478e-6 |
| 精确续训实测 |50→100，十项状态/全50步记录/400DeltaW E2 | 按缩短设计未另跑50→100 |
| 发布重建 | 全800 LoRA张量、开关probe FP32位、400DeltaW E2 | 同样通过 |
| 原生FreeToken | on/off/zeroB、700条新进程E2、缓存重建，留出50/50、replay300/300 | 同样通过 |

功能门槛45/50、回放NLL增量<=0.01来自本轮[实际用户确认](../evidence/dense/effect-threshold-owner-decision.json)，用于Q38正式100。Q36的10步结果是原缩短范围的实测，不能写成已执行可选100。这里是合成关联记忆与算术回放的流程验收，未做领域效果或通用能力评测。

## 资源与速度

| 运行 | peak allocated B | peak reserved B | 整卡最小观测free MiB | 平均step s |
| --- | ---: | ---: | ---: | ---: |
| Q38 functional100 | 5,587,110,912 | 8,009,023,488 | 7222 | 50.818 |
| Q36 short10 | 5,587,110,912 | 8,002,732,032 | 7228 | 48.764 |
| Q38 pressure2048 x2 | 9,005,052,416 | 9,483,321,344 | 5938 | 128.748 |
| Q38 pressure4096 x2 | 12,996,858,368 | 13,398,704,128 | 2204 | 284.409 |

锁页层payload16,840,133,760B、实际64bank分配16GiB；CPU embedding普通冻结属性2,542,796,800B。GPU原字节双槽531,497,056B；LoRA435,159,040B，Adam矩870,318,080B，梯度另计435,159,040B。Q38正式run观察working set峰值22,713,417,728B/private32,271,273,984B。private是Windows进程提交口径，不能当物理RAM相加。4096整卡余量2204MiB超过1024MiB，无需O1/O3/O4；没有改head精度、scope或rank。

Q38实际100步总wall5504.912s，active-step平均50.818s；输入28,746 tokens、监督3,741 tokens。短功能样本不是固定2048长度吞吐。压力两步使用完整2048/4096长度，单列。整卡1秒采样可能漏亚秒瞬态；allocator峰值独立保留。

实际100恢复后，303输入tokens/8 micro，无optimizer更新的仪器化组71.918s，对照45.788s且loss/全部梯度E2。H2D5.312s、解码28.727s、全部GEMM3.739s、SDPA0.05357s；GDN forward+backward inclusive24.573s（recompute子集8.767s）、attention2.046s（0.900s）、MLP43.251s（10.867s）。CUDA区间含CPU入队间隙，H2D可重叠，子集/包含区间不可相加。全部256次attention前向含重算及128反向为efficient、零math。

## 发布与证据

发布布局为nvfp4-lora-dense-v1，只含标准PEFT A/B文件、配置及manifest，绑定部署NVFP4原revision/三分片SHA；没有重新量化或全量BF16基座。旧MoE读取器在原环境中拒绝两份最终dense发布。检查点保存Adam/scheduler/全部RNG/sampler/trainer与逻辑配置；发布目录用于重建，不携带续训状态。

| 模型 | adapter_model.safetensors SHA256 |
| --- | --- |
| Q38 actual100 | `206ea21afe34cecb4cfd9d87e62fd1ff003b9a1c8a035de8a87f9db7943258d2` |
| Q36 actual10 | `552dc8992fa1ffe924e3160c2d123f69fab2b7ee232f6058457b8ac36cdbdf85` |

原始发布位置：`C:\workspace\runs\T-20261029-nvfp4-dense\20261030T020753Z-d6-functional100\results\publish`、`C:\workspace\runs\T-20261029-nvfp4-dense\20261030T050306Z-qwen36-short10\results\publish`。最终verified-copy制品目标：`C:\workspace\artifacts\nvfp4lora-dense\0.0.1\20261030T054636Z`，包含Q38 step20/50/100、Q36 step10、两发布目录、固定数据及重建源码/锁文件；原始部署权重仅按revision/SHA引用。复制前后逐文件校验，根manifest最后生成；其实际回执随后归档，未执行前不把目标目录当已生成。

最终只推siltok-ai/nvfp4lora的feat/dense27b-lora和wenlive/FreeToken的feat/dense27b-nvfp4-lora。最终HEAD与remote-inclusive严格审核留在交付回执/仓外run，避免在提交里引用它自身的最终SHA。Git中只有摘要和manifest；权重、检查点、原始logits、快照和大日志不入仓。

MoE两套logical_config及哈希E2：real-c3: `e58cf9acbedd898b0d584601775997ebcfa3afba11321f16f12490c1e2453e5d`; real-expert-r2: `2f8da3e3b5e4fd0751d14a5aadeb8503261a3e01b8acce4476b4c82db0c82d4a`。原始41测试保留，现58全部通过；MoE tiny100记录/最终参数/完整状态E2，两真实初始probe均1083229253；原交付step50加载后参数/Adam/scheduler/RNG/sampler/trainer/route audit全E2，无新增优化器步。15受保护Git源码不变；14个Python checkout行尾差异单独记录，数值比较不忽略任何位。

## 遗留范围与下一步

训练仍执行原Torch解码、BF16码值GEMM→FP32 global→BF16，LoRA旁路FP32；反向重新取层/解码，不保存解码权重。FP8原input_scale观测不是无截断证书：全600语料base74/adapter73个模块超过原范围，最大1.378486/1.394422。训练未模拟部署W8A8；FreeToken自身完整语料实测通过，不能推断所有输入的跨引擎数值E2。

FreeToken需显式layer streaming、关闭CUDA Graph，仅验收单GPU、非思考纯文本及本次缓存配置；Speed本身的dense推理没有接入。原生推理wall包含集成检查且可与CPU来源准备重叠，不作为速度benchmark。

[ROADMAP](../../ROADMAP.md)中对dense优先的是完整代表性长度时间线、NVFP4解码/执行融合与GDN的调度/重算优化，再用同源权重做实际推理质量闭环。当前短样本解码区间显著大于SDPA，H2D已经重叠，MoE路由/grouped-expert调度和CPU专家计算不适合作为dense的首项优化。选择性保留仅评估合法激活，继续禁止为反向保留解码W。本分支未新增性能kernel、FLA或序列打包；更换数值策略/依赖需按硬边界另行确认。
