# Dense 27B progress

Authority: [original design](../reference/dense27b-design-v1.md), [original handoff](../reference/dense27b-handoff-v1.md), and dated additions under [dense decisions](../decisions.md#dense-27b). Originals are immutable. Qwen3.8-27B receives full acceptance; Qwen3.6-27B receives the design's shorter campaign. Native Windows and the original dependency lock remain mandatory.

Current state: **D0 qualified with a documented SDPA fallback. D1-D7 not started. No real dense training or release yet.** Latest compact receipts: [D0 summary](../evidence/dense/d0-summary.json). Complete commands, code snapshots, stdout/stderr and checkpoints remain in the referenced external runs. No intermediate push has been made.

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
