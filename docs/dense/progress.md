# Dense 27B progress

Authority: [original design](../reference/dense27b-design-v1.md), [original handoff](../reference/dense27b-handoff-v1.md), and dated additions under [dense decisions](../decisions.md#dense-27b). Originals are immutable. Qwen3.8-27B receives full acceptance; Qwen3.6-27B receives the design's shorter campaign. Native Windows and the original dependency lock remain mandatory.

Current state: **D0 in progress. D1-D7 not started. No real dense training or release yet.** Latest compact receipts: [D0 summary](../evidence/dense/d0-summary.json). Complete commands, code snapshots, stdout/stderr and checkpoints remain in the referenced external runs. No intermediate push has been made.

| Gate | State | Measured evidence |
| --- | --- | --- |
| DA0.1 | PASS | Independent checkout/branch from target origin/main `efdfc55a26d9531c904470a55ab35793004ed104`; original files archived in first commit `cb5d5d8`. |
| DA0.2 | PASS | Independent venv; all 39 locked versions and wheel SHA256 match, hash-required offline install, pip check passes; freeze differs only in editable binding. Registry state is preparing, not a real-training qualification. |
| DA0.3 / B0 | PASS | Unmodified main: 41 unit tests; MoE tiny100/50-to-100/reload all pass. Both real initial probes FP32 bits `1083229253`; both logical configs and hashes retained. This is a baseline, not D7 regression. |
| DA0.4 | RUNNING | Trusted fixed-revision HF metadata received for RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30; three weight shards downloading. No claim of source SHA qualification before all three checks finish. |
| DA0.5 | PASS | Real text architecture on meta: 64 layers, 26,895,998,464 text parameters, all 400 quantized linear shapes checked. Tiny replacement-interface forward finite; production PackedLinear remains untested until D2/D4. |
| DA0.6 | PASS | Independent short recurrence: output/all five input gradients within 1%; T2048/T4096 direct and both checkpoint modes E2. T4096 maximum allocated 3,658,202,112 B, reserved 3,986,685,952 B; isolated kernel qualification, not full-layer/full-model memory. |
| DA0.7 default HF | FAIL | No-mask GQA selects math; allocated peak 6,979,321,856 B. Failure retained. |
| DA0.7 dense fallback | QUALIFIED CANDIDATE | Explicit KV repeat uses existing efficient backend with/without padding: allocated 525,074,432/659,292,160 B; repeated forward/backward E2; all output/input-gradient errors vs math <1%. Dense integration and whole-model checks remain D4/D5 work. |
| DA0.8 | MEASURED | Six forward/dx shape cases at T2048/4096: 93.51-100.66 TFLOPS. Gate NVFP4 decode 5.42ms/89,128,960 elements (16.46G elements/s); FP8 0.32ms/52,428,800 (163.58G elements/s). Synthetic valid packed bytes, not source equivalence evidence. |
| DA0.9 | PASS | Logical payload including scalar fields 16,840,133,760 B; physical pinned 17,179,869,184 B (16GiB), allocation 3.05s. Full-bank H2D 328.00ms/51.34GB/s, not an overlap/whole-model throughput proof. |

DA0.9's first measurement failed because CUDA was not initialized before querying host allocator statistics; allocation itself succeeded. The corrected run and original failed run are separate and preserved. See decisions for the 3,200-byte scalar accounting correction and the SDPA fallback experiment.

Budget implications so far: pinned allocation matches the 16GiB estimate. GDN kernel memory is consistent with the planned 3-4GB range, but full-layer/LoRA/recompute memory remains unmeasured. Large-shape GEMM is faster than the estimated 40-60TFLOPS; whole-model step time cannot yet be inferred from isolated GEMM. No budget or functional threshold is silently approved.

User additionally requested a separate FreeToken inference branch. It exists locally as `wenlive/FreeToken:feat/dense27b-nvfp4-lora` from latest main `f417397f339050e6f400e75db0f1e6acbc40f13a`; reference MoE branch is `b354e6df87108143d1277f62520ce070806e03a2`. No FreeToken implementation changes or inference qualification yet. Full-model VRAM capacity must be measured separately.
