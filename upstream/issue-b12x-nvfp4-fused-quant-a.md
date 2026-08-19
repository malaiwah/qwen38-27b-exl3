# Feature: extend fused_quant_a to the NVFP4 (W4A4) dense GEMM

## Motivation
The `gemm.dense_fused_quant_a` producer path (BF16 A quantized inside the
GEMM mainloop, no HBM round-trip for quantized activations) exists for MXFP8
only — the UE8M0 sites at dense_gemm.py:2874-2884 / 3411-3416 / 3630-3635 are
all in the MXFP8 producer, and `can_implement`/dispatch never select
fused-quant for `Float4E2M1FN`.

For NVFP4 W4A4 serving (EXL3-converted Qwen3.8-27B, RTX 5090), the separate
`Bf16ToFp4TmaKernel` quantizer is a measurable cost:

- Decode (M=1 padded to the quantizer's 128-row tile): 19 ms of quantizer vs
  11 ms of GEMM per forward across 256 FP4 calls (64 layers x 4 linears).
- Prefill: the quantized-A tensor makes a full HBM round trip
  (write in quantizer, read in GEMM) per linear.

A fused NVFP4 A-quant producer (per-16 E4M3 block scale + global scale, same
recipe as `quantize_grouped_nvfp4`; global scale supplied as an input since it
is a per-tensor reduction) would remove the extra kernel launch and the HBM
round trip. On our step-time breakdown that is worth several percent of
prefill and more of decode.

## Notes from reading the current code
- The MXFP8 fused producer already demonstrates the pipeline-stage structure;
  NVFP4 differs in sf_vec (16 vs 32) and scale type (E4M3 vs UE8M0).
- The standalone NVFP4 quantizer hoists the global-scale reciprocal to CTA
  smem but re-reads `global_scale[0]` and the smem reciprocal per 16-element
  block (nvfp4/_kernel.py:413-414) — if a fused producer is written, keeping
  those in registers is free.

## Ask
`fused_quant_a` support for `ab_dtype=float4_e2m1fn` (NVFP4, sf_vec 16), with
the per-tensor global scale passed as a kernel argument. Can validate
bit-exactness against the standalone quantizer and benchmark on SM120.
