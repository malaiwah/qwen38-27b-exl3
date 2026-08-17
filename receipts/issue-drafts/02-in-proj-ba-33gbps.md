# Issue draft 02 — in_proj_ba (unquantized 5120×96 bf16) runs at 33 GB/s and costs 5.2 % of decode GPU time

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F6, plan P2.1

## Summary
On Qwen3.5-class GDN models the per-layer `in_proj_ba` linear (5120×96, kept unquantized/bf16 by the
serving overlay for fidelity) is dispatched by cuBLAS to
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` at **28 µs/call** — 33 GB/s
effective for a 983 KB weight, 48 calls per decode forward = **1.34 ms/step = 5.2 % of decode GPU
time** (torch-profiler receipt: `receipts/kernel-gap-profiled-decode.json`, category table + top-30
kernels). At 1.4 TB/s this op is worth ~3 µs, not 28.

## Probe result (2026-08-17): dispatch fix only — NO custom kernel needed
Measured on the same card (`receipts/kernel-gap-ba-probe.json`, graph-replayed):
`F.linear` (served TN path) 20.08 µs; **pre-transposed K×N `mm` 5.61 µs (3.6×)**;
`torch.compile` max-autotune 4.23 µs. cuBLAS simply picks a bad TN kernel at this shape.

## Proposed fix — measured end-to-end
~15-line patch (`receipts/kernel-gap-ba-transpose.patch`): in `UnquantizedLinearMethod`,
`process_weights_after_loading` registers a `weight_kn = weight.t().contiguous()` buffer for
`N≤128, K≥4096` CUDA weights (behind `VLLM_TINY_N_MM_TRANSPOSE`, ~1 MB/layer extra), and `apply`
uses `torch.mm(x, weight_kn)`. Graph-capturable (pure mm, no allocation).
**Measured: +8.0 % single-stream decode alone (96.19 → 103.91 tok/s median,
`receipts/kernel-gap-ba-ab.json`).** Caveat: stacked with the draft-01 gate patch the local gain
saturates at the gate-alone level because the post-gate step is CPU-dispatch-bound under proot —
the two fixes need a bare-metal stacked measurement before summing their gains.

## What would falsify this
The probe already settled the kernel question (5.6 µs achievable). Remaining falsifier: a
bare-metal A/B showing the end-to-end gain <1 % (would mean the GPU stretch is never the critical
path at C1 on production hosts — unlikely given the F6 trace, but that is the check).
