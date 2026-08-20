
# Multi-Precision FP4/FP6 Strategy (User Directive 2026-08-18)

## Core Insight
The model already uses multi-bitrate K4/K5/K6 to balance speed vs fidelity.
Apply the SAME principle to tensor core precision:

- **FP4 (mxf4nvf4 m16n8k64, 4x MMA)** for the LARGEST weight matrices
  - MLP down_proj (5120×17408 = 89M params/layer × 64 layers = 5.7B params)
  - MLP gate+up_proj (17408×5120×2 = 178M params/layer × 64 = 11.4B params)
  - These dominate prefill FLOPs → maximum speedup from 4x MMA

- **FP6 (mxf8f6f4 m16n8k32, 2x MMA)** for precision-sensitive layers
  - Attention QKV/O projections (smaller, more precision-sensitive)
  - GDN in_proj/out_proj (hybrid architecture, critical for correctness)
  - These are smaller GEMMs → less speedup benefit from FP4, more fidelity risk

## Expected Throughput (weighted upper bound)
- If MLP is 70 % of serial FP16 work and speeds up 4× while the remaining
  30 % speeds up 2×, Amdahl gives
  `1 / (0.7/4 + 0.3/2) = 3.077×`, not the arithmetic mean 3.4×.
- Applied to the 4,527 tok/s MMA-only estimate: ~13,930 tok/s before
  quantization, launch and non-GEMM overhead.
- 55 % of that bound is ~7,660 tok/s; 65 % is ~9,055. Reaching 10k requires
  ~71.8 % of the bound and remains an experiment, not a prediction.

## Fidelity Preservation
- FP6 for attention preserves the precision-sensitive projections
- FP4 for MLP: the Hadamard fold spreads values → better FP4 quantization
- K6→FP6 error: ~3% (negligible)
- K6→FP4 error with Hadamard fold: ~6-8% (acceptable for MLP)
- The original K5/K6 mix already has variable precision → multi-precision FP4/FP6 is natural

## Implementation
1. Load time: convert each layer to FP4 or FP6 based on layer type
   - MLP gate/up/down → FP4 (MXFP4, UE8M0, sf_vec_size=32)
   - Attention QKV/O, GDN → FP6 (MXFP6, UE8M0, sf_vec_size=32)
2. Runtime: dispatch to dense_gemm with appropriate ab_dtype
   - FP4 layers: ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu"
   - FP6 layers: ab_dtype="float6_e2m3fn", a_fmt="e4m3", b_fmt="e2m3"
3. Memory: FP4 weights 0.5 bytes + FP6 weights 0.75 bytes
   - Total: ~12-14 GB (less than current K6's 15.9 GB)

## KLD Strategy
- Run KLD test with multi-precision weights
- If KLD > 2.9e-5: move more layers from FP4 to FP6
- If KLD passes and PP < 10k: move more layers from FP6 to FP4
- Binary search for the optimal FP4/FP6 split
