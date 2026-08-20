# Lever 3: Fused Dequant-in-Epilogue GEMM Kernel — Final Report

## Status: INFEASIBLE for speed target (stop condition #3)

The fused kernel meets the prototype's stated numerical tolerance but cannot
beat the existing reconstruct+cuBLAS prefill path for speed.

## What was built

A non-cooperative M-parallel grid tiling variant of the exllamav3 EXL3 GEMM
kernel (`exl3_gemm_prefill_v3.cu`) that eliminates `grid.sync()` barriers
by launching M-tiles in parallel across CUDA blocks.

### Modified inner kernel (`exl3_gemm_inner_prefill.cuh`)
- 5 changes from original `exl3_gemm_inner.cuh`:
  - Function renamed: `exl3_gemm_kernel_inner` → `exl3_gemm_prefill_inner`
  - 2 params added: `num_slices_param`, `block_id_param`
  - 3 line replacements: `gridDim.x` → `num_slices_param`, `blockIdx.x` → `block_id_param`
  - Added `#include "exl3_kernel_map.cuh"` for `EXL3_GEMM_T_ARGS`

### Outer kernel (`exl3_gemm_prefill_v3.cu`)
- Two-kernel launch: input Hadamard pre-pass + GEMM with M-parallel grid tiling
- Grid: `grid_m * grid_n` blocks, each block handles one (M-tile, N-slice)
- TILESIZE_M=64 (TILEBLOCKS_M=4) for 4 M-slabs per block
- Uses `DevCtx` for locks, `getCurrentCUDAStream()` for stream safety
- Lock buffer overflow check for large M
- `OptionalCUDAGuard` for multi-GPU safety

## Results

### Numerical agreement: within the prototype tolerance
- 8 diverse shapes tested (bits 4/5/6, K 5120-17408, N 1024-17408)
- relative difference <0.4% at M=128; maximum difference 0.002930 versus the
  chosen reference. This is **not** bit parity or an end-to-end KLD result.

### Speed vs existing `exl3_gemm` (cooperative): 2x at M≥128
| M | exl3_gemm | v3 | speedup |
|---|-----------|-----|---------|
| 128 | 0.21ms | 0.10ms | 2.00x |
| 256 | 0.41ms | 0.21ms | 2.02x |
| 512 | 0.82ms | 0.41ms | 2.02x |
| 1024 | 1.65ms | 0.82ms | 2.01x |
| 2048 | 3.33ms | 1.67ms | 1.99x |

### Speed vs `reconstruct_hgemm` (actual prefill path): SLOWER at M≥256
| M | reconstruct | v3 | v3 vs recon |
|---|------------|-----|-------------|
| 128 | 0.14ms | 0.11ms | 1.30x |
| 256 | 0.19ms | 0.21ms | 0.88x |
| 512 | 0.31ms | 0.42ms | 0.74x |
| 1024 | 0.53ms | 0.82ms | 0.65x |
| 2048 | 0.98ms | 1.69ms | 0.58x |

## Why infeasible

Among the tested implementations, the existing exllamav3 dispatch is faster:
- M ≤ 144: `exl3_gemm` cooperative kernel
- M > 144: `reconstruct_hgemm` (reconstruct weights to FP16 + cuBLAS hgemm)

The prototype's inline dequant did not match cuBLAS large-M performance on
SM120. That experiment does not prove global optimality or identify the
bottleneck without a profiler.

The fused kernel's potential advantage is **traffic**, not scratch capacity:
it avoids about 48.65 GB of cumulative FP16 reconstruct writes per chunk, but
the reusable scratch allocation is only the largest live reconstructed matrix.
The prototype did not convert that traffic reduction into speed.

## What would be needed

To beat reconstruct+cuBLAS at prefill, a Marlin-scale kernel would be needed:
- Fully fused W4A16/W6A16 GEMM with persistent kernel design
- TMA + mma.sync aluop on SM120
- Optimized K-split reduction with warp-level atomics
- This is a multi-month kernel engineering effort beyond the container toolchain

## Peer review

Reviewer: KernelReviewer subagent (confidence 0.82)
- Core GPU kernel logic: correct
- 4 host-dispatch issues found and fixed:
  1. Missing OptionalCUDAGuard (multi-GPU safety)
  2. Missing getCurrentCUDAStream (stream safety)
  3. Lock buffer overflow for large M (bounds check added)
  4. size_m using A.size(0) instead of product of leading dims

## Artifacts

All code committed to `malaiwah/qwen38-27b-exl3` on `main` branch:
- `patches/exl3_gemm_prefill_v3.cu` — outer kernel + host dispatch + Python binding
- `patches/exl3_gemm_inner_prefill.cuh` — modified inner kernel
- `patches/test_benchmark.py` — multi-M parity + speed benchmark
- `patches/test_diverse.py` — diverse layer parity test
- `patches/test_vs_reconstruct.py` — v3 vs reconstruct_hgemm benchmark
- `patches/test_ground_truth.py` — ground truth comparison
- `patches/build_full_ext.py` — build script (compiles all ext sources + v3)
