# D0 resource and time budget

These are native Windows RTX 5070 Ti measurements and planning estimates. They
do not establish whole-model training performance. Raw commands and snapshots
are in the external runs linked by [the D0 summary](../evidence/dense/d0-summary.json).

| Measurement | Result | Scope |
| --- | --- | --- |
| Pinned bank | 17,179,869,184 B / 16GiB, 3.05s allocation | 64 real-sized payload allocations, including scalar fields |
| Full-bank H2D | 328.00ms, 51.34GB/s | Sequential asynchronous copies; not evidence of training overlap |
| BF16 GEMM | 93.51-100.66TFLOPS | Required large forward/dx shapes at T2048/T4096 |
| NVFP4 decode | 5.42ms / 89,128,960 elements | Valid synthetic codes, 8Mi-element chunks, no global scale folded into weights |
| FP8 code conversion | 0.32ms / 52,428,800 elements | Valid synthetic E4M3 bytes |
| GDN core T4096 | max allocated 3.658GB, reserved 3.987GB | All five input gradients; direct and both checkpoint modes |
| Full HF GDN T2048 | 0.159-0.290s; max allocated 2.176GB | Frozen synthetic BF16 projections; direct and both checkpoint modes, no LoRA |
| Full HF GDN T4096 | 0.396-0.440s; max allocated 4.042GB, reserved 4.442GB | Same scope; includes convolution, norm and production BF16 beta |
| Efficient SDPA T4096 | allocated 0.525GB without mask / 0.659GB with mask | Explicit KV repeat; repeated output/gradients E2; math reference E3 <1% |

GB is decimal; GiB is binary. Each peak belongs to its own isolated process.
They are not simultaneous whole-model allocations. The full GDN test retains
BF16 projection weights, whereas production will stream packed weights and
decode per projection. Actual LoRA, checkpoint inputs, CE, allocator history and
desktop occupancy still require D5/D6 measurements.

The original pinned-memory estimate is confirmed. Matrix-only payload omits
3,200 bytes of FP32 scalar fields; the corrected requested total is
16,840,133,760 B and every layer remains below 256MiB. No allocation redesign,
dependency or numerical-policy change is needed for this finding.

The default no-mask HF GQA path selected math and failed DA0.7. The qualified
alternative explicitly repeats KV and uses the existing efficient backend.
The original failure remains recorded; production integration is dense-only.
It must still pass the complete tiny and real model checks.

Time extrapolation must include GDN. A rough sum of the measured non-reentrant
full-GDN costs gives about 8.2s for 48 layers at T2048 and 21.1s at T4096 before
separating their projection GEMMs. These sums are estimates, not measured
model times, and already include the GDN projections. Counting those GEMMs a
second time would overestimate total work. Separately, three passes over the
NVFP4 MLP matrices plus FP8 conversion extrapolate to roughly 3.3s of code
decode from this synthetic probe; launches and the real memory schedule may
change that cost.

For planning only, use roughly 12-18s/micro at T2048 and 25-35s/micro at T4096,
or 96-144s and 200-280s per accumulation-8 optimizer step. Short functional
samples are not covered by the large-GEMM throughput measurement. Their actual
step time and the 100-step wall budget remain unmeasured. These revised ranges
are intentionally provisional and do not change any acceptance threshold.
If real steps are unexpectedly more than twice the design estimate, the
original stop/diagnose rule still applies.
