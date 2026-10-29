# Dense 27B progress

Authority: [original design](../reference/dense27b-design-v1.md), [original handoff](../reference/dense27b-handoff-v1.md), and dated additions under [dense decisions](../decisions.md#dense-27b). Originals are immutable. Qwen3.8-27B receives full acceptance; Qwen3.6-27B receives the design's shorter campaign. Native Windows and the original dependency lock remain mandatory.

Current state: **D0 qualified with a documented SDPA fallback; D1 tiny and real preprocessing passed; D2 qualified with documented FP32 reference controls. D3-D7 pending. No real dense training or release yet.** Complete commands, code snapshots, stdout/stderr and checkpoints remain in the referenced external runs. No intermediate push has been made.

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
