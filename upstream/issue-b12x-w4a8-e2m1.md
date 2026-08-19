# Feature: W4A8 dense GEMM (A=float8_e4m3fn, B=float4_e2m1fn) for SM120

## Motivation
Serving EXL3-trellis checkpoints on RTX 5090 (SM120) with load-time weight
conversion to NVFP4 (Qwen3.8-27B, 64 layers). Measured on this workload:

- W4A4 NVFP4 (b12x dense GEMM, per-call activation amax): PP=6413 tok/s,
  mean KLD vs BF16 = 0.0567 over a 512-context suite.
- W6A8 MXFP6 (same harness, e4m3 activations): KLD = 0.0107 (5.3x better),
  but PP drops to 4671 because B streams at 6 bits.

Activation precision dominates the quality gap; weight bits dominate prefill
throughput. A W4A8 dense GEMM (e4m3 activations x e2m1 weights) would combine
the FP4 weight-streaming rate with the FP8 activation fidelity — on our
numbers that is the difference between choosing KLD 0.0567 (throughput
profile) and 0.0107 (quality profile) instead of getting both.

## Current behavior
`dense_gemm` with `a_fmt=e4m3, b_fmt=e2m1` fails:

```
packed_k_bytes must be divisible by 3
```

The e2m1 B format routes into the MXFP6 3:4 packing path; there is no
NVFP4-B x FP8-A branch. (Probed on image b12xcd3ce19, CUDA 13.2, SM120.)

## Why it should be cheap on SM120
The `mma.sync.aligned.kind::mxf8f6f4.block_scale` family already allows mixed
operand formats (A and B independently e4m3/e5m2/e2m3/e3m2/e2m1). The NVFP4
GEMM (`MmaMXF4NVF4Op`, m16n8k64, sf_vec 16) and the MXFP8 GEMM (sf_vec 32)
both exist; W4A8 is "A-side of the MXFP8 path + B-side of the NVFP4 path".
The main design decision is the scale recipe for the mixed case (e.g. E4M3
block scales sf_vec 16 on B, UE8M0 sf_vec 32 on A, as the mxf8f6f4 kind
permits per-operand scale metadata).

## Ask
A `dense_gemm` branch accepting `a_dtype=float8_e4m3fn, b_dtype=float4_e2m1fn`
on SM120, with whatever scale-vector combination is natural for the kind.
Happy to benchmark/KLD-validate on the RTX 5090 + Qwen3.8-27B setup and
report numbers back.
