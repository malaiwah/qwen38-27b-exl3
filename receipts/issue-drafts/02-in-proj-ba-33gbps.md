# Issue draft 02 — in_proj_ba (unquantized 5120×96 bf16) runs at 33 GB/s and costs 5.2 % of decode GPU time

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F6, plan P2.1

## Summary
On Qwen3.5-class GDN models the per-layer `in_proj_ba` linear (5120×96, kept unquantized/bf16 by the
serving overlay for fidelity) is dispatched by cuBLAS to
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` at **28 µs/call** — 33 GB/s
effective for a 983 KB weight, 48 calls per decode forward = **1.34 ms/step = 5.2 % of decode GPU
time** (torch-profiler receipt: `receipts/kernel-gap-profiled-decode.json`, category table + top-30
kernels). At 1.4 TB/s this op is worth ~3 µs, not 28.

## Proposed fix (probe first)
1. 30-minute probe: does cublasLt heuristic selection / torch.compile already beat the observed
   pick at (m≤16, K=5120, N=96)? If yes: dispatch fix only.
2. Else: a split-K GEMV (Triton) for `m≤16, N≤128, K≥4096` bf16 in `UnquantizedLinearMethod.apply`,
   graph-capturable (preallocated split-K workspace), behind `VLLM_TINY_N_GEMV`.
Expected recovery ~1.25 ms/step ≈ +5–7 % single-stream decode.

## What would falsify this
A probe showing the 28 µs is not kernel-bound (e.g. unavoidable launch+sync floor at this grid on
SM120): if a bare split-K kernel cannot beat ~15 µs at this shape, the finding downgrades from
"wrong kernel" to "launch floor" and belongs with the structural list (docs/47 F10).
