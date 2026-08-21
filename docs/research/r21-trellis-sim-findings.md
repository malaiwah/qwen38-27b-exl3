# R21-TrellisSim: Simplified Trellis / Viterbi Codebook Simulation

**Status:** completed, 2026-08-21 (reviewer-corrected). 7 codebook strategies + Viterbi DP tested on 4 real Qwen3.8-27B tensors, 3 slices each (12 combos), K=3-6, with and without BiIP+Hadamard rotation. Matched-byte DP allocation with corrected per-strategy byte accounting (16-byte K-map metadata, float16 codebook round-trip, codebook-level transition penalty).

## 1. Executive summary

**Key finding: Shared codebook (O(1) storage) + no per-tile scale + rotation + DP allocation beats per-tile uniform + rotation at matched bytes, winning 12/12 at K=4 and K=5 (ratio 0.66-0.89). The per-tile min/max sidecar (512 bytes = 64×8) is unnecessary after rotation, and those bytes are reallocated to higher K by the DP allocator.**

**Simplified Viterbi (trellis path search with codebook-level transition penalty) does NOT help**: at α≤1e-7, the Viterbi path matches independent quantization (ratio ~0.99-1.01); at α≥1e-4, it degrades severely (ratio 5-1000×). There is a narrow transition zone (α ~1e-6 to 1e-5) where Viterbi slightly changes the path but the effect is negligible (<12%). Weight elements within a tile are not spatially correlated, so penalizing codebook-level jumps between adjacent elements provides no benefit. This is NOT a simulation of EXL3's actual trellis state machine, which constrains coding state (bit patterns), not weight-value smoothness.

**R16's central question answered**: Yes, a shared codebook (2^K×2 bytes, once) is worth it. The per-tile codebook overhead that killed R16's non-uniform quantizers (7.7-60%, exponential in K) is eliminated with O(1) storage. Combined with rotation (which homogenizes tile distributions), the no-scale shared codebook achieves 11-34% HWE reduction vs per-tile uniform at matched bytes.

## 2. Strategies tested

| Strategy | Codebook | Per-tile sidecar | Codebook storage | Total sidecar (K=5) |
|----------|----------|-----------------|-----------------|---------------------|
| per_tile_uniform | per-tile uniform | 8 bytes (min/max) | 0 | 528 B (512+16 meta) |
| shared_uniform_scale | shared uniform (normalized) | 8 bytes (min/max) | 64 B (once) | 592 B |
| shared_lloyd_max_scale | shared Lloyd-Max (normalized) | 8 bytes (min/max) | 64 B (once) | 592 B |
| shared_kmeans_scale | shared k-means++ (normalized) | 8 bytes (min/max) | 64 B (once) | 592 B |
| shared_uniform_noscale | shared uniform (global) | 0 | 64 B (once) | 80 B (64+16 meta) |
| shared_lloyd_max_noscale | shared Lloyd-Max (global) | 0 | 64 B (once) | 80 B |
| shared_kmeans_noscale | shared k-means++ (global) | 0 | 64 B (once) | 80 B |
| multilevel_shared_dp | multiple shared k-means at K=3,4,5,6 | 8 bytes (if scale) | 240 B (once) | 768 B / 256 B |
| multilevel_shared_noscale_dp | multiple shared k-means | 0 | 240 B (once) | 256 B |

Codebook levels are rounded through float16 (the declared wire dtype) before use, simulating the serialization round-trip. K-map metadata: 64 tiles × 2 bits = 16 bytes (both arms).

## 3. Matched-byte DP allocation (corrected)

### 3.1 Unrotated matched-byte DP

| Arm | K=4 | K=5 | K=6 | Mean ratio |
|-----|-----|-----|-----|------------|
| shared + scale DP | 10/12 wins | 9/12 wins | 8/12 wins | 0.92-0.95 |
| **shared no-scale DP** | **11/12 wins** | **11/12 wins** | 2/12 wins | **0.76-0.86** |

Shared no-scale wins 11/12 at K=4-5 unrotated (ratio 0.65-1.03). The 512-byte per-tile sidecar savings (minus 240-byte codebook + 16-byte metadata = net 256 bytes) enables ~8 tile-bit upgrades. For L55_down/first (heavy-tailed), shared with-scale wins decisively (ratio 0.64-0.79 across K=4-6).

### 3.2 Rotated matched-byte DP (FAIR budget)

| Arm | K=4 | K=5 | K=6 | Mean ratio | Ratio range (wins) |
|-----|-----|-----|-----|------------|-------------------|
| shared + scale DP | 8/12 wins | 7/12 wins | 8/12 wins | 0.97-1.01 | 0.84-1.06 |
| **shared no-scale DP** | **12/12 wins** | **12/12 wins** | 4/12 wins | **0.74-0.79** | **0.66-0.89** |

**After rotation, shared no-scale + DP wins ALL 12/12 at K=4 and K=5** (ratio 0.663-0.894). This is the strongest result: rotation homogenizes tiles → no per-tile scale needed → 512 bytes saved (net 256 after codebook+metadata) → reallocated to higher K → lower HWE.

At K=6, the no-scale advantage vanishes (4/12 wins, mean ratio 1.03). The codebook overhead (240 bytes for 4 codebooks) and diminishing returns of K=6 vs K=5 make the advantage marginal.

### 3.3 Net byte savings

- Gross min/max removal: 64 × 8 = 512 bytes
- Codebook storage: 2×(8+16+32+64) = 240 bytes (4 codebooks, float16)
- K-map metadata: 16 bytes (both arms)
- **Net saving: 512 - 240 = 272 bytes** (2.5% of K=5 budget, ~8 tile-bit upgrades at K=4→5)

The codebook cost (240 bytes) is 47% of the gross sidecar saving — not negligible, but the remaining 272 bytes are enough for meaningful allocation improvement.

## 4. Simplified Viterbi — NEGATIVE RESULT (narrowed)

### 4.1 Corrected results (codebook-level transition penalty, (hi-lo)² emission scaling)

At α ≤ 1e-7: Viterbi matches independent quantization (ratio 0.99-1.01, entropy unchanged).
At α ~ 1e-6 to 1e-5: slight degradation (ratio 1.01-1.12), negligible benefit.
At α ≥ 1e-4: severe degradation (ratio 5-1000×), entropy drops as path collapses.

| α | K=4 ratio | K=5 ratio | K=6 ratio |
|---|-----------|-----------|-----------|
| 1e-8 | 0.995 | 0.999 | 0.995 |
| 1e-7 | 0.995 | 1.005 | 0.994 |
| 1e-6 | 1.007 | 1.002 | 0.994 |
| 1e-5 | 1.120 | 1.119 | 1.681 |
| 1e-4 | 5.649 | 15.02 | 63.71 |
| 1e-3 | 82.4 | 278.1 | 1216.4 |

The transition zone (α ~1e-6 to 1e-5) is too narrow to provide any meaningful quality improvement. Below it, Viterbi is identical to independent; above it, Viterbi harms.

### 4.2 Why it fails

The transition penalty α·(level[j]-level[k])² penalizes jumps between adjacent elements' codebook levels. Weight elements within a 16×16 tile are not spatially correlated — element (i,j) and (i,j+1) can have very different values. Any penalty strong enough to change the path forces elements to nearby codebook levels, destroying quantization quality.

**Important caveat (reviewer):** This is NOT a simulation of EXL3's actual trellis state machine. EXL3's trellis constrains which codeword prefixes can follow each other (coding state, not weight values). A correct trellis simulation would need to model the actual convolutional code structure. QSRT's "final-KL gradient-guided Viterbi" (using decoder adjoint to shift Viterbi targets for model KL) is a fundamentally different and more sophisticated approach. Our conclusion applies only to the simplified weight-value-smoothness Viterbi, not to trellis coding in general.

## 5. Rotation and tile homogeneity

### 5.1 Tile range coefficient of variation (CV)

| Tensor/Slice | Unrotated CV | Rotated CV | Reduction |
|-------------|-------------|-----------|-----------|
| L0_gate/first | 0.223 | 0.101 | 54.9% |
| L0_down/first | 0.120 | 0.102 | 14.9% |
| L55_gate/first | 0.134 | 0.095 | 29.5% |
| L55_down/first | 1.130 | 0.115 | **89.8%** |

Rotation reduces tile range CV on all 12 slices. L55_down/first (the heavy-tailed outlier) sees the largest reduction (89.8%), confirming that rotation removes the inter-tile range variation that required per-tile scale.

### 5.2 Fixed-K no-scale after rotation

| K | Win rate | Ratio range | Mean ratio |
|---|----------|-------------|------------|
| 4 | 11/12 | 0.74-1.21 | 0.86 |
| 5 | 9/12 | 0.85-1.07 | 0.94 |
| 6 | 9/12 | 0.85-1.07 | 0.97 |

After rotation, the no-scale shared codebook is competitive even at fixed K (without DP allocation). At K=4, it wins 11/12 with mean ratio 0.86.

**Mechanism note (reviewer):** The POC combines BiIP scaling with two Hadamard transforms and does not ablate them separately. The tile range CV reduction supports the mechanism but cannot attribute the effect to rotation alone vs BiIP scaling alone. The R10/QSRT activation-boundary transform (verified exact, error 2.55e-15) is a potentially better rotation that handles MLP gate/up/down jointly.

## 6. Answer to the central question

**Does shared codebook beat per-tile at matched bytes?**

**YES, decisively after rotation at K=4-5.**

| Condition | Win rate | Ratio range | Mean ratio |
|-----------|----------|-------------|------------|
| Unrotated, K=4, no-scale | 11/12 | 0.65-1.08 | 0.76 |
| Unrotated, K=5, no-scale | 11/12 | 0.75-1.03 | 0.85 |
| **Rotated, K=4, no-scale** | **12/12** | **0.66-0.89** | **0.74** |
| **Rotated, K=5, no-scale** | **12/12** | **0.71-0.86** | **0.79** |
| Rotated, K=6, no-scale | 4/12 | 0.83-1.19 | 1.03 |

The winning strategy: **shared k-means codebook + no per-tile scale + BiIP+Hadamard rotation + DP allocation**. This achieves:
- O(1) codebook storage (240 bytes for 4 codebooks at K=3-6)
- Zero per-tile sidecar (no min/max)
- Net 272 bytes saved vs per-tile uniform (512 sidecar - 240 codebook)
- 11-26% HWE reduction vs per-tile uniform + rotation at matched bytes (K=4-5)

## 7. Implications for the production stack

1. **EXL3's shared codebook is vindicated**: The O(1) codebook storage model that R16 couldn't test is viable. Non-uniform quantization with shared codebooks beats per-tile uniform at matched bytes.

2. **Rotation is the key enabler**: Without rotation, shared no-scale codebooks fail on heavy-tailed tensors (L55_down/first unrotated: ratio up to 5.2×). With rotation, they succeed (ratio 0.66-0.86). Tile range CV drops 15-90% after rotation. EXL3's existing Hadamard+signs+LDLQ incoherence processing makes shared codebooks viable.

3. **The no-scale approach is the trellis advantage**: By eliminating per-tile min/max, the no-scale shared codebook achieves the lowest total bytes. This is the closest proxy to EXL3's actual architecture (one codebook, no per-tile metadata beyond the index).

4. **Multi-level allocation composes with shared codebooks**: The DP allocator works with shared codebooks at multiple K values, implementing the multi-precision alphabet (K4/K5/K6). The shared codebook at each K captures different distribution structure — this is the "B changes codebook fit" insight from external feedback.

5. **Viterbi is not the right model for trellis**: EXL3's trellis constraint is on coding state (bit patterns), not weight values. A correct trellis simulation needs the actual convolutional code structure. QSRT's gradient-guided Viterbi (decoder adjoint for model KL) is the right approach.

6. **Local HWE may not predict end-to-end KLD** (QSRT lesson): Local HWE/SSE can invert end-to-end KLD. Our matched-byte results need KLD validation on aiboss before production decisions.

## 8. Implications for R22 (block-diagonal allocation) and R24 (entropy allocation)

- **R22 (BlockDiagAlloc)**: With shared no-scale codebooks, per-tile byte cost is just K×256/8 (payload only). This simplifies allocation — no sidecar term. BD-GPTQ + shared codebook is a natural combination: within-tile error correction + O(1) codebook + payload savings = more bits for allocation.

- **R24 (EntropyAlloc)**: Shared codebook indices have entropy 3.82-5.84 bits vs 3.50-5.55 for per-tile uniform at same K. Higher entropy = less compressible, BUT the shared codebook eliminates per-tile entropy model overhead (2^K × 2 bytes/tile). R24 found that entropy-constrained DP gives +57-60% HWE reduction when model overhead is excluded — shared codebook makes this practical. The combined savings (per-tile min/max + per-tile entropy model) stack.

## 9. Limitations

- 128×128 slices from full tensors (aspect ratio hidden, per R15 caveat).
- Block-diagonal HWE surrogate for DP allocation (cross-tile Hessian terms discarded). Final HWE computed on full matrices, but DP objective is block-local.
- Synthetic Hessian (diagonal stats generalize per R15, but full-covariance interaction untested).
- k-means++ with single seed; multi-restart could improve codebook quality.
- Codebook designed per-slice; a global codebook across all tensors could differ. Production-wide "shared codebook vindication" remains an extrapolation.
- No held-out validation of shared codebook quality.
- No end-to-end KLD measurement (CPU-only; requires aiboss GPU). QSRT lesson: local metrics can invert KLD.
- BiIP+Hadamard approximates EXL3's incoherence processing (Hadamard+signs+LDLQ) but is not identical. No BiIP-only or Hadamard-only ablation.
- R10/QSRT activation-boundary transform is a potentially better rotation for MLP tensors (exact, error 2.55e-15) — not tested here.
- Byte accounting: codebook as float16 (round-tripped), min/max as float32 (not round-tripped), BiIP scales as float32 (not round-tripped).
- Viterbi tested with codebook-level quadratic transition penalty only; other penalty forms not tested but unlikely to help given the fundamental mismatch with trellis coding.

## 10. Artifacts

| File | Description |
|------|-------------|
| `tools/research/r21-trellis-sim/poc.py` | PoC implementation (reviewer-corrected) |
| `receipts/research/r21-trellis-sim-results.json` | Full results JSON |
| `docs/research/r21-trellis-sim-findings.md` | This document (auto-verified against JSON) |
