# Lever 4: Marlin-Scale Persistent W6A16 Kernel — Report

## Status: IMPLEMENTED, peer-reviewed, pushed. Does not beat reconstruct+cuBLAS.

## What was built

A persistent W6A16 kernel for EXL3 trellis quantization on SM120 (RTX 5090),
designed as a Marlin-scale kernel with TMA bulk load support.

### Key design decisions

1. **SM120 capabilities discovered**:
   - `tcgen05` (Blackwell datacenter tensor cores) is NOT available on SM120
   - `cp.async.bulk` (TMA bulk copy) IS available and compiles on SM120
   - `mma.sync m16n8k16` (same as Ampere/Ada) is the only tensor core path
   - 170 SMs, 101KB optin shared memory, 1024 max threads/block

2. **Persistent grid**: Launch `MIN(num_sms, total_tiles)` blocks. Each block
   iterates over (M-tile, N-slice) pairs via a for loop. Falls back to
   non-persistent launch when `grid_n > num_sms` (deadlock avoidance).

3. **K-split**: Same as v3 — `num_slices=grid_n, block_id=n_slice` so each
   block processes a fraction of K×N tiles.

4. **TMA helpers defined**: `cp.async.bulk`, `mbarrier_init`, `mbarrier_wait`
   PTX wrappers are defined but NOT yet integrated into the inner kernel loop.
   Integration would require creating `CUtensorMap` descriptors on the host
   and rewriting the `async_load_gl` lambda to use `cp.async.bulk.tensor`.

5. **TILE_M=64**: TILE_M=128 exceeds shared memory (344KB > 101KB limit).

## Results

### Numerical agreement: within the prototype tolerance
- M=1–2048 cases report relative difference <0.4% against `exl3_gemm`.
- This is not bit parity, full-model output parity or a KLD qualification.

### Speed vs existing exl3_gemm (cooperative): 1.92x at M≥128
| M | exl3_gemm | Marlin | speedup |
|---|-----------|--------|---------|
| 128 | 0.21ms | 0.11ms | 1.92x |
| 2048 | 3.33ms | 1.74ms | 1.92x |

### Speed vs reconstruct+cuBLAS: 0.56x at M=2048
The inline-dequant prototype is slower. `SH_STAGES=6` also failed to improve
it, but that single null intervention does not prove integer-ALU saturation.
A profiler counter breakdown is required before assigning the bottleneck.

## Peer review

Reviewer: MarlinReviewer subagent (confidence 0.9)
3 issues found and fixed:
1. (P1) Self-deadlock when grid_n > num_sms → conditional persistent/non-persistent launch
2. (P2) Missing lock buffer overflow guard → re-added TORCH_CHECK
3. (P2) Persistent iteration splits K-split partners → fixed by non-persistent fallback

## What would close the gap vs cuBLAS

1. **TMA integration**: Replace per-thread `cp.async` with `cp.async.bulk.tensor`
   for B-tile loads. This requires host-side `CUtensorMap` creation and kernel-side
   TMA PTX. Would reduce load instruction count, freeing ALU for dequant.

2. **Custom dequant for W6**: The current `dq4` function uses bit manipulation
   (shift, mask, `lop3`) to extract 6-bit codes. A custom PTX sequence optimized
   for SM120's ALU pipeline could reduce dequant instruction count.

3. **Register-level dequant pipelining**: Pre-dequant the next K-tile's B
   fragments in registers while the current tile's MMA is executing. This
   requires careful register allocation and may need reduced TILE_M.

4. **Warp-specialized dequant**: Dedicate some warps to dequant, others to MMA,
   communicating via shared memory. This is the Marlin approach but requires
   a complete kernel rewrite.

All four approaches are multi-day to multi-week efforts requiring deep PTX
optimization and kernel architecture work beyond the current toolchain's
inline-asm capabilities.

## Artifacts

- `patches/exl3_gemm_marlin.cu` — persistent kernel + TMA helpers + Python binding
- `patches/build_marlin.py` — build script
- Pushed to `malaiwah/qwen38-27b-exl3@main`
