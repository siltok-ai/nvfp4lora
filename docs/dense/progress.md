# Dense 27B progress

Authority: [original design](../reference/dense27b-design-v1.md), [original handoff](../reference/dense27b-handoff-v1.md), and dated additions under [dense decisions](../decisions.md#dense-27b). Originals are immutable. Qwen3.8-27B receives full acceptance; Qwen3.6-27B receives the design's shorter campaign. Native Windows and the original dependency lock remain mandatory.

Current state: **All technical acceptance completed: Q38 full100, Q36 design-short10, final native inference for both and full DA7.1 MoE bit regression.** Documentation and verified-copy delivery are being finalized; remote publication is checked at delivery. Earlier pending statements below are historical stage records; final measurements are consolidated in results.md.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA0.1 | PASS | Independent checkout/branch from target origin/main `efdfc55a26d9531c904470a55ab35793004ed104`; original files archived in first commit `cb5d5d8`. |
| DA0.2 | PASS | Independent venv; all 39 locked versions and wheel SHA256 match, hash-required offline install, pip check passes; freeze differs only in editable binding. Registry state is preparing, not a real-training qualification. |
| DA0.3 / B0 | PASS | Unmodified main: 41 unit tests; MoE tiny100/50-to-100/reload all pass. Both real initial probes FP32 bits `1083229253`; both logical configs and hashes retained. This is a baseline, not D7 regression. |
| DA0.4 | PASS | RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30: all three complete files match upstream LFS SHA256. Bounded ranges and the parameterized repository fetcher independently verify the files; full source receipt retained. |
| DA0.5 | PASS | Real text architecture on meta: 64 layers, 26,895,998,464 text parameters, all 400 quantized linear shapes recorded and required projection dimensions checked. Tiny replacement-interface forward finite; production PackedLinear remains untested until D2/D4. |
| DA0.6 | PASS | Independent short recurrence: worst relative L2 7.983e-5 across output/all five input gradients; production BF16 beta case 6.922e-5. Core and full HF GDN module at T2048/T4096 pass both checkpoint modes E2. Full-module T4096 max allocated 4,042,380,800 B, reserved 4,441,767,936 B; frozen synthetic BF16 projections, no LoRA, not whole-model memory. |
| DA0.7 default HF | FAIL | No-mask GQA selects math; allocated peak 6,979,321,856 B. Failure retained. |
| DA0.7 dense fallback | FALLBACK PASS | Explicit KV repeat uses existing efficient backend with/without padding: allocated 525,074,432/659,292,160 B; repeated forward/backward E2; all output/input-gradient errors vs math <1%. Dense integration and whole-model checks remain D4/D5 work. |
| DA0.8 | MEASURED | Six forward/dx shape cases at T2048/4096: 93.51-100.66 TFLOPS. Gate NVFP4 decode 5.42ms/89,128,960 elements (16.46G elements/s); FP8 0.32ms/52,428,800 (163.58G elements/s). Synthetic valid packed bytes, not source equivalence evidence. |
| DA0.9 | PASS | Logical payload including scalar fields 16,840,133,760 B; physical pinned 17,179,869,184 B (16GiB), allocation 3.05s. Full-bank H2D 328.00ms/51.34GB/s, not an overlap/whole-model throughput proof. |

DA0.9's first measurement failed because CUDA was not initialized before querying host allocator statistics; allocation itself succeeded. The corrected run and original failed run are separate and preserved. See decisions for the 3,200-byte scalar accounting correction and the SDPA fallback experiment.

Budget implications: pinned allocation matches the 16GiB estimate. Full GDN is at the upper edge of the planned range, while LoRA/packed-layer and whole-model memory remain unmeasured. Large-shape GEMM is faster than the estimated 40-60TFLOPS, but the measured GDN cost increases the provisional long-sequence time budget. See [D0 budget](d0-budget.md) for scope and estimates. No functional threshold is approved by this measurement.

User additionally requested a separate FreeToken inference branch. It exists locally as `wenlive/FreeToken:feat/dense27b-nvfp4-lora` from latest main `f417397f339050e6f400e75db0f1e6acbc40f13a`; reference MoE branch is `b354e6df87108143d1277f62520ce070806e03a2`. Reused Python adapter infrastructure and LoRA tests match the reference Git blobs; 40 existing tests pass in the new isolated inference environment. That is a reference-foundation result, not dense model inference. Current FreeToken head is `2f0165a`; dense format/mapping/capacity remain pending and unpushed.

## D1

[D1 summary](../evidence/dense/d1-summary.json) includes independent readback and optional FreeToken sampling. Tiny was generated and passed before real preprocessing. The minimal store layout argument was added for DA1.4; the MoE default is unchanged. The 45-test suite passes (41 existing plus 4 dense format tests).

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA1.1 | PASS | Real 64-layer geometry matches; 400 quantized linears, 1,392 streamed fields. Tiny contains 8 deliberately misleading MTP/visual keys; all excluded. Real excludes 348 non-text tensors. |
| DA1.2 | PASS | Every streamed byte compared after serialization and independently from original shards to loaded CPU store. Real payload 16,840,133,760 B. |
| DA1.3 | PASS | All 7,214,202,880 FP8 weight code elements convert exactly to BF16. Optional 36 real MLP samples over gate/up/down in layers 0/3/31/63 match FreeToken raw-FP32-global decoding at 0 ULP; FP16 runtime global differs by at most 1 ULP, reported separately. |
| DA1.4 | PASS | Every layer below 256MiB, max 265,748,528 B; index/header/file SHA checked. Independent store reload reproduces all payload hashes. |
| DA1.5 | PASS | Real preprocessing peak working set 6,071,853,056 B. Independent pageable-store validation is a separate process and not included in this preprocessing peak. |
| DA1.6 | PASS | Nonstream payload 5,138,070,528 B; BF16 embed/head/small-parameter keys complete. No streamed MLP/attention weight decoded into these files. Head uses existing nonexpert policy; source config/tokenizer and output hashes recorded. |

Real bundle: `C:/workspace/models/bundles/qwen3.8-27b-nvfp4-lora/319f741cce68d791-train-v1`.
Tiny bundle: `C:/workspace/models/bundles/nvfp4lora-tiny-dense/seed-20261029-train-v1`.
They are preprocessing artifacts, not trained adapters. The first optional FreeToken invocation failed before computation because its environment lacks psutil; the reference-only import path was reduced to its actual dependencies and rerun without installing any package. The failed run remains separate.

## D2

[D2 summary](../evidence/dense/d2-summary.json) preserves the original FP32 failure and independent controls. Production arithmetic follows design 3.3 exactly: BF16 code-value GEMM, FP32 output scale, BF16 result; backward reacquires/redecodes, and ctx stores only layer/name/config. FP32 LoRA is an ordinary autograd side path with zero B. No production PEFT injection or MoE numerical changes.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA2.1 | PASS | All ten real projection shapes at T4096: forward and dx E2 vs independently materialized same-expression ordinary autograd. |
| DA2.2 | PASS | NVFP4 and FP8 small FP64 gradcheck; no saved base tensors or Tensor attributes in ctx. |
| DA2.3 original | PARTIAL / FAIL | Small shapes pass original 1e-6; real down_proj forward 1.0246188661e-6 fails. Ordinary FP32 autograd matches that result E2. Original failed run retained. |
| DA2.3 amended | QUALIFIED | All ten real shapes satisfy ordinary-autograd E2 plus FP64 relative L2 <=2e-6. Thirty-six predeclared held-out controls over down/gate/o, three seeds and four lengths: both precision modes E2 vs ordinary autograd, max L2 1.31958903495e-6. See the dated decision; this is not original-threshold PASS. |
| DA2.4 | PASS | Max direct decoder peak delta 287,310,336 B, returned live delta 0; max extra weight workspace over materialized control 178,257,920 B, within matrix+chunk bound. Full forward/backward peak delta max 891,289,600 B includes required activation/cast workspace and is separately reported. No saved base tensor; gate allocator snapshot retained outside Git with SHA256. |
| DA2.5 | PASS | All ten real shape repeated forward/dx E2. |
| DA2.6 | PASS | All ten real shapes: two-backward accumulation, reentrant checkpoint and non-reentrant checkpoint E2. |
| DA2.7 | PASS | All ten real shapes: B=0 enabled/disabled output E2; nonzero LoRA output/dx/dA/dB E2 vs handwritten ordinary autograd expression. Whole-model zero-step loss remains D4. |

Fifty unit tests pass (41 original, four D1 and five D2). CPUEmbedding holds its frozen CPU BF16 table as an ordinary attribute, with no parameters/buffers/state_dict entries; D4 must still prove whole-model fingerprint coverage. The tiny-only reference keeps decoded matrices and clones global scales to survive slot reuse; real-checkpoint materialization is explicitly rejected. These operator qualifications do not establish whole-model training, exact resume, publication or inference; those remain subsequent gates.

## D3

[D3 summary](../evidence/dense/d3-summary.json) contains every layer's counters and timings. Production SyncLayerStreamer and AsyncLayerStreamer are unchanged; the explicit dense store layout argument is the D1 addition. Tiny passes before the real 64-layer run.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA3.1 | PASS | 1,000 random acquires, all 64 layers covered; 263,169,033,008 full payload bytes read back and compared directly with CPU source, not just hashes. |
| DA3.2 | PASS | 128 full-payload reads with 50,000,000 GPU sleep cycles per request; released readers survive subsequent slot reuse. Pins, stale generation, double release, close and alternate compute stream guards pass. |
| DA3.3 | PASS | 128 deliberately wrong prefetches; full bytes equal source and combined readback SHA matches no-prefetch run. All 13 operator projections also match no-prefetch output/dx/dA/dB E2. |
| DA3.4 | PASS | GDN and full-attention layers, 13 projections covering ten shapes, T4096: synchronous vs asynchronous forward/dx/LoRA A/B gradients E2. |
| DA3.5 | PASS | Per-layer acquires/hits/copies/bytes/GPU copy and wait times retained. Random read run: 999 hits, 973 copies, 256,056,737,264 H2D bytes; event wait total 13.17ms. |

Physical pinned allocation is 16GiB. Validation process peak working set 19,325,108,224 B; peak GPU allocated/reserved 2,934,964,224/3,370,123,264 B. Frozen CPU store hashes remain unchanged. The 264.84s random-read wall time includes full D2H, CPU comparisons and hashing, and is not training throughput. Delayed-reader copy event totals correspond to 43.93GB/s (no prefetch) and 47.61GB/s (wrong prefetch); copy/compute overlap from CUDA event intervals is retained separately. No full training-memory or speed claim follows from these qualification peaks.

## D4 / M1-D

[D4 summary](../evidence/dense/d4-summary.json) and [M1 audit](../evidence/dense/m1-audit.json) bind the actual commands and source snapshots. M1-D uses the recorded qualified D0/D2 paths; the original failed gates remain failed, with no change to the production PackedLinear arithmetic or any DA4 threshold.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA4.1 | PASS | Tiny and actual real construction: no meta parameters/buffers, exact nonstream keys, FP32 RoPE, ordinary CPU embedding, expected canonical trainables. Actual real D-ALL r16: 800 FP32 CUDA tensors / 108,789,760 elements; all B=0; 400 packed projections. Fingerprint mutation controls detect CPU table, raw bank and frozen GPU norm changes. |
| DA4.2 same expression | PASS | 100 optimizer steps, two microbatches each. Every loss E2; first gradients relative L2=0; final weights relative L2=0. |
| DA4.2 HF+PEFT | PASS | Independent HF BF16 pre-scaled frozen Linear weights + ordinary PEFT. First gradient min cosine 0.9995077116; max relative L2 0.0316625 is reported, while the design criterion is cosine>=0.999. Max per-step relative loss 0.000251277849 (<0.02); final weight min cosine 0.9998401944. |
| DA4.3 | PASS | New process step50-to100: all ten checkpoint/history/probe/DeltaW checks E2; all 50 subsequent losses, sample IDs, LR, gradient norms and consumed RNG values agree. |
| DA4.4 | PASS | New process reconstructs from preprocessed bundle+publish: all parameter bytes, frozen fingerprint, enabled/disabled FP32 probe loss and 25 DeltaW fingerprints E2. Standard PEFT reads published files on independent HF BF16 base: relative probe loss 0.000110244147 (<0.02). |
| DA4.5 | PASS | Optimizer IDs == expected canonical set == 50 published tensors. Production and both references' frozen fingerprints unchanged. Tiny probe enabled7.4265113 vs disabled7.6956329 demonstrates effect in this synthetic test. |
| DA4.6 | PASS | 56 unit tests. Feature MoE tiny2 matches original B0 first two metric rows E2; final states/probe match an unchanged-main two-step supplement E2. Both MoE logical configs/hashes unchanged; original MoE reader rejects dense release layout. Full D7 campaign remains pending. |

Real construction only: frozen GPU parameters2,595,273,728 B; LoRA435,159,040 B; two raw GPU slots531,497,056 B; CPU embedding2,542,796,800 B. Construction GPU peak allocated/reserved is about3.576/3.586GB; this excludes activations, gradients and initialized Adam moments. The 64 layer banks consume exactly16GiB pinned; validation Boolean readback adds a separate cached1-byte allocation. Both measurements and the initial overly broad assertion failure are retained. The alternative D-ATTN count in design3.5 is corrected to416 tensors in decisions; its element count and default D-ALL counts are unchanged, and D-ATTN is not enabled.

Next: D5 full real forward, independent CPU FP32 first-four-layer reference at the original unconditional1% threshold, generation and checkpoint-specific tokenizer/template goldens. No real training before those gates and no formal100-step run before DA6.3 user confirmation. FreeToken dense inference remains an independent unfinished deliverable.

## D5

[D5 evidence](../evidence/dense/d5-summary.json) retains complete commands, resources and per-projection ranges; raw tensor captures stay outside Git with SHA256. The source remains RadixArk@319f741c with the original three-shard bindings. These are full training-model base forwards, not FreeToken inference-engine acceptance.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA5.1 | PASS | Load allocated/reserved3,573,386,240/3,586,129,920 B. Observed128-token forward peak3,906,082,304/3,961,520,128 B; whole-card remaining11,214MiB. Raw layer banks16GiB pinned plus ordinary CPU table2,542,796,800 B; full allocator/process snapshots retained. |
| DA5.2 | PASS | Original unconditional CPU FP32 first-four-layer relative L2:0.0029163732,0.0029407353,0.0040695826,0.0033973938, all<=0.01. Token-recursive GDN and eager attention, one FP32 layer at a time. |
| DA5.3 | PASS | Greedy outputs2,北京,Hello; each followed immediately by im_end248046. Short uncached sanity only. |
| DA5.4 | PASS | One128-token instrumented forward: H2D332.21ms, decode1218.13ms, projection GEMM99.78ms; GDN inclusive543.65ms and non-projection467.66ms; attention inclusive51.54ms and non-projection26.71ms. Copy/compute overlap and enqueue gaps mean the intervals cannot be summed. Actual SDPA operator count:16 efficient, zero math; repeated hidden states byte-E2. |
| DA5.5 | PASS: observation complete | All208 FP8 source projections observed. Three ratios above original448*input_scale range, max1.022831; layer51 o_proj and layer62 in_proj_qkv/in_proj_z. This does not certify zero clipping or W8A8 equivalence. |
| DA5.6 | PASS | Dense tokenizer/template200 Speed AST goldens; every prompt/label, forced left truncation and answer-too-long rejection passes. Tokenizer options match Speed. |

The first additional random-tiny BF16/FP32 test exceeded1% at layer3 (1.191991%); its failed run is preserved. Independent token-recursive FP32 vs ordinary HF chunk FP32 agrees within6.379013e-7, and both yield the same larger tiny cross-precision error. The reference-algorithm unit checks that mathematical control. The real DA5.2 criterion was never relaxed and passes independently. Full unit suite:57 tests. Frozen model parameters, buffers, raw banks and CPU table remain unchanged through reference and generation.

Next: D6 own-base replay300, canary300/holdout50, actual one-step mechanisms,2048/4096 two-step pressure, then **DA6.3 user confirmation before formal100 steps**. FreeToken dense inference and the later Qwen3.6 campaign remain unfinished.

## D6 preflight / DA6.3 confirmed

[Preflight evidence](../evidence/dense/d6-preflight-summary.json) records actual DA6.0–DA6.2 commands and manifests. This is qualification only; formal DA6.4–DA6.9 are not run. The original PackedLinear arithmetic, D-ALL r16, CPU embedding, lm_head strategy and dependencies are unchanged.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA6.0 | PASS | Canary300/holdout50 text bytes equal original corpus. Replay300 generated by this dense base; combined training600. Base holdout0/50, replay981 supervised tokens, NLL0.00023400843633333113 nat/token. |
| DA6.1 | PASS | Real1 step x accumulation8. Initial enabled/disabled FP32bits1083850783, loss4.820327281951904. All initial B gradients nonzero. Thirteen projections across GDN/full-attention layers: same-expression output/dx/A/B E2, FP32 E3, eight-micro actual A/B sum E2. Updated probe4.715414047241211; disabling restores original bits. Frozen fingerprint unchanged. |
| DA6.2 2048 | PASS | Two actual optimizer steps x8:129.03/128.47s. Peak allocated9,005,052,416B/reserved9,483,321,344B; monitored whole-card minimum5938MiB. |
| DA6.2 4096 | PASS | Two actual optimizer steps x8:284.10/284.72s. Peak allocated12,996,858,368B/reserved13,398,704,128B; monitored whole-card minimum2204MiB (>1024MiB). O1/O3/O4 not needed. |
| DA6.2 recomputation | PASS | Both lengths restore step2 Adam870,318,080B; all64 layers, complete allocation history, loss and all800 gradients U8 E2 against uninstrumented control. First-micro and steady-state snapshots: decoded allocations occurred, none remain live after backward. |
| DA6.3 | CONFIRMED 2026-10-30 | Actual owner reply approves holdout>=45/50 and replay NLL increase<=0.01. Full quote, recorded time and baseline are in effect-threshold-owner-decision.json. |
| DA6.4–DA6.9 | NOT RUN | Formal100, real50-to100 E2, formal release reconstruction, functional result and post-training ranges remain pending. |

The first frozen layer's qkv/z inputs have no actual autograd dx edge; those are checked by isolated replay, rather than mislabeled as captured real dx. One-step mechanism checkpoints use logical max1 and cannot silently become the100-step campaign. Pressure checkpoints use logical max100 but execution stops at2 on synthetic stress data. Their publications are qualification artifacts, not the final functional adapter.

The steady-state single-micro allocated peaks are8,543,853,056B (2048) and12,543,980,032B (4096). They start without existing accumulated gradients and must not replace the larger actual x8 pressure peaks. Whole-card sampling is every second and may miss shorter transients; CUDA allocator peaks are retained independently. Fifty-seven unit tests pass. The owner approved the recommended functional thresholds on2026-10-30; the earlier preflight manifest remains an unchanged historical record from before approval. See [the actual decision](../evidence/dense/effect-threshold-owner-decision.json).

The production release CLI's dense dispatch is additionally tested against the
unchanged saved D4 tiny100 publication. A schema fixture serializes the saved
expected probe loss and25 DeltaW fingerprints without recomputing expectations;
a fresh `verify_training_release.py` process passes all parameter bytes,
enabled/disabled loss bits and all25 DeltaW fingerprints. See the
[CLI qualification](../evidence/dense/d6-tiny-release-cli-summary.json).
No new training or production changes are involved. This establishes CLI wiring,
not the still-pending real DA6.6. The existing resume comparator already compares
all ten named state/history/probe/DeltaW groups without a MoE layout assumption.

[CPU data relocation](../evidence/dense/data-portability20-summary.json) restores
the actual step20 sampler cursor on an identical corpus at a new path. Logical
configuration/checkpoint hashes are unchanged; all remaining80 groups/640 sample
IDs and token tensors agree E2. Altering the copied file bytes is rejected. CUDA
is not initialized. This qualifies data portability only; real50-to100 GPU
resume remains a separate pending requirement.

## DA6.4 formal100 completed

[Formal100 evidence](../evidence/dense/d6-functional100-summary.json) records the
actual100-step run from the approved source commit3fa6205, with complete20/50/100
checkpoints,800 FP32 trainables and400 streamed DeltaW fingerprints. Frozen
weights are unchanged. Final holdout is50/50 and replay NLL3.3538927875694635e-7;
the base was0/50 and2.3400843633333113e-4. Step10/20/50 evaluations also give50/50.
These actual functional observations satisfy the approved thresholds; DA6.7's
separate published-adapter reconstruction/evaluation still remains pending.

Mean active optimizer-step wall time50.818s (median50.110s, range44.306–60.169s).
Full recorded run5504.912s includes initialization, audits, evaluation, saving
and publication. Observed peak allocated5,587,110,912B/reserved8,009,023,488B;
whole-card monitor5240 samples, minimum free7222MiB. Process observed peak
working set22,713,417,728B/private32,271,273,984B. These are the short functional
samples; they do not replace the larger2048/4096 pressure peaks above.

A new process is now restoring the actual step50 checkpoint and continuing to100
with the identical logical configuration and data. DA6.5/DA6.6 cannot be marked
passed until the full ten-state comparison and independent publication rebuild
finish. Final FreeToken acceptance must use this100-step adapter rather than the
earlier qualification step1 publication. The two feature branches remain local.

## Qwen3.8 DA6.4-DA6.9 completed

[Complete D6 evidence](../evidence/dense/d6-formal-summary.json) binds every run
manifest, formal checkpoint, publication SHA and measurement. All100 steps and
the independent50-to100 continuation complete successfully; no production
training arithmetic changed after the formal source commit3fa6205.

| Gate | State | Actual result |
| --- | --- | --- |
| DA6.4 | PASS | D-ALL r16, max_length2048, accumulation8,100 steps; evaluation10/20/50/100; complete20/50/100 checkpoints. |
| DA6.5 | PASS | All50 resumed step histories and all ten named state/probe/DeltaW checks E2; zero differences. |
| DA6.6 | PASS | New process using bundle+publish only: all800 parameter tensors, enabled/disabled FP32 probe bits and all400 DeltaW fingerprints E2. |
| DA6.7 | PASS | Reconstructed published adapter:50/50 (base0/50); replay NLL3.3538927875694635e-7, increase-0.00023367304705457418; both approved thresholds satisfied. Prompt247,983 vocabulary entries change, max absolute logit change43.0625. |
| DA6.8 | PASS: observation complete | All208 original FP8 scales,600 samples, disabled/enabled phases. Above-range modules74/73, max ratio1.3784860/1.3944223. No zero-clipping or W8A8 equivalence claim. |
| DA6.9 | PASS | Restored100, one8-micro accumulation,303 input tokens; no optimizer update. Loss/all800 gradients E2 against uninstrumented control; all module intervals complete.256 efficient forward calls including recompute,128 efficient backward, zero math. |

The timing control takes45.788s; the instrumented group takes71.918s. CUDA event
intervals include enqueue gaps: H2D5.312s, decoder28.727s, all GEMM3.739s,
SDPA0.05357s. GDN inclusive forward+backward24.573s (recompute subset8.767s),
attention2.046s (subset0.900s), MLP43.251s (subset10.867s). These are overlapping
and nested intervals, not additive pure-kernel timings; the instrumented cost
must not replace the actual100-step mean50.818s. This is short functional data,
not a2048/4096 throughput benchmark.

An initial tiny profiler run failed because its hook expected a Tensor whereas
HF attention returns a tuple and receives keyword inputs. All gradients already
matched E2; only interval coverage failed. The original failed GPU run and CPU
negative controls remain preserved. The corrected CPU and tiny GPU controls pass
before real timing. See [the interface qualification](../evidence/dense/profiler-tuple-interface-summary.json).

Qwen3.6 metadata/weight acquisition and final FreeToken acceptance continue.
The owner has been asked the design's optional full100-vs-short10 question;
the original10-step scope remains authorized and no extra100 is inferred from
silence. Neither feature branch is pushed before final completion.

## Qwen3.6 shortened acceptance completed

[Qwen3.6 evidence](../evidence/dense/qwen36-summary.json) binds the official
nvidia/Qwen3.6-27B-NVFP4 revision0893e1606ff3d5f97a441f405d5fc541a6bdf404,
all three independently verified full-shard SHA256 values and each actual run.
Acquisition began only after Qwen3.8 passed DA6.6.

| Shortened gate | State | Actual result |
| --- | --- | --- |
| DA0.4 | PASS |21,921,697,184 original shard bytes; all3 LFS SHA256 matches, then independent full-file recheck and metadata binding. |
| D1 | PASS |64-layer full E1/FP8 code identity and reload E2; preprocessing peak6,072,393,728B (<16GB).36 FreeToken NVFP4 row-sample comparisons pass. |
| D5 | PASS |Unconditional CPU FP32 first-four-layer L2:0.0028898339/0.0032451096/0.0054424449/0.0042275474, all<=0.01;16 efficient SDPA and zero math, repeated hidden E2. Greedy2/北京/Hello then im_end;200 own-tokenizer Speed goldens pass. |
| DA6.0 | PASS |Own-base regenerated replay300; original canary300/holdout50 bytes preserved; train600. Base holdout0/50, replay NLL0.00018482663403820553. |
| DA6.1 | PASS |Actual1 step x8; B=0 enabled/disabled FP32bits1084881418, all B gradients nonzero,13 real-activation projection references and8-micro accumulation pass; frozen unchanged. Snapshot58,224 decoder allocations,0 live after backward. |
| Short10 | PASS |Actual10 steps x8, D-ALL r16/max_length2048; logical max100 with execution stop10, not a100-step campaign. Final holdout50/50, replay NLL0.000006259719674569478; mean step48.764s. |
| Release rebuild | PASS |Bundle+publication only: all800 trainable tensor bytes, enabled/disabled FP32 probe and400 DeltaW fingerprints E2. |

The10-step adapter SHA256 is
552dc8992fa1ffe924e3160c2d123f69fab2b7ee232f6058457b8ac36cdbdf85.
It is undergoing separate native FreeToken controls/full-corpus/cache/fresh
process checks. Its short-run functional numbers are observations, not a claim
of optional full100-step acceptance. The design's additional100-step question
was asked; this evidence claims only the original authorized10-step scope.


## D7 technical acceptance completed

[D7 evidence](../evidence/dense/d7-summary.json):58 units,54 FreeToken source tests,MoE tiny100/parameters/states vs original B0 E2,both real probes1083229253,both logical configs/hashes unchanged,both delivered step50 full restores including route-audit E2 with zero updates. Original main loader rejects both actual final dense publications. Native full-model/fresh/corpus/cache qualification for both is complete. Final consolidated measurements and remaining publication receipt are in [results](results.md).
