# R24 — Entropy-Constrained Allocation: Findings (Reviewer-Corrected)

## Summary

Entropy-constrained DP allocation — using empirical entropy H(quantized_indices) as the rate instead of fixed K — reduces Hessian-weighted error by **+56-59% over fixed-K DP** at matched payload bytes (without per-tile model overhead), across 4 real tensors × 3 slices × K=3-6.

**Important caveats (from reviewer):**
- This is an **empirical-entropy lower bound**, not a demonstrated coded-byte result. A real arithmetic/range coder would add model overhead.
- The DP uses a **block-diagonal surrogate** for the full HWE (cross-tile terms omitted).
- All results use **proxy HWE on 128×128 slices**, not end-to-end KLD. Local metrics can invert end-to-end KLD (QSRT lesson).
- At K=6, the improvement is **exactly 0%** — both arms select uniform K6.

## Key Results

### 1. Entropy Savings (Exp1)
Quantization index entropy is consistently below K:
| K | Mean entropy H | Savings vs K |
|---|---|---|
| 3 | 2.33 bits | 22.1% |
| 4 | 3.38 bits | 15.6% |
| 5 | 4.36 bits | 12.7% |
| 6 | 5.29 bits | 11.8% |

### 2. Entropy DP vs Fixed-K DP (Exp2) — Core Result
At matched payload bytes (no per-tile entropy model overhead):
| avg_K | Entropy DP improvement over fixed-K DP |
|---|---|
| 3 | +57.7% ± 4.0% |
| 4 | +56.1% ± 2.6% |
| 5 | +58.9% ± 2.7% |
| 6 | +0.0% ± 0.0% |

**Mechanism**: The entropy DP can afford higher K for low-entropy tiles (concentrated distributions) because their actual coded rate is below K. Allocated K is negatively correlated with tile entropy (~-0.5), confirming the mechanism.

**At K=6, improvement is 0%**: All tiles are already at K_MAX=6, so the entropy budget doesn't unlock additional K levels.

### 3. Entropy Model Overhead (Critical Practical Finding)
When including per-tile entropy model cost (2^K × 2 bytes/tile for arithmetic coding probability table):
- At K=3: entropy DP still helps (+14.0% aggregate, not +20.4% as initially reported)
- At K≥4: model overhead exceeds the entropy savings, making entropy DP **worse** than fixed-K

**Implication**: Entropy-constrained allocation is only practical if the entropy model is shared (R21's shared codebook) or uses a compact parametric representation.

### 4. Entropy + Rotation (Exp3, Reviewer-Corrected)
Uses transformed Hessians H_G' = U S_G⁻¹ H_G S_G⁻¹ Uᵀ, H_X' = Vᵀ S_X⁻¹ H_X S_X⁻¹ V for rotated distortion tables.

BiIP+Hadamard rotation increases entropy by +1.3% ± 1.0% (K=4 average).

Entropy allocation helps **more before rotation** than after (combined ent+rot: +57.6% vs fixed-K only, but entropy alone gives +58.5%). Rotation and entropy allocation are partially redundant.

### 5. R-D Frontier (Exp4)
The entropy Pareto frontier dominates fixed-K at matched byte budgets. Typical gain: +40-62%.

### 6. Empirical Entropy Lower Bound (Exp5, Renamed)
NOT a real arithmetic coder — computes H(indices) as theoretical lower bound. Compression ratios: 1.13x-1.30x. A real coder would add model/framing overhead.

### 7. Inter-layer Entropy Allocation (Exp6, Brute-Force)
| avg_K | Fixed-K D | Entropy D | Gain |
|---|---|---|---|
| 4 | 1.560e-3 | 1.285e-3 | +17.7% |
| 5 | 1.202e-3 | 1.154e-3 | +4.0% |

Note: These use block-local proxy ΣD_t, not full HWE. With full HWE, gains are +17.0% and +3.7%.

### 8. B-as-Action (Exp7, Per-Tensor)
BiP is a full-matrix transform — cannot be applied per-tile. Tested as per-tensor decision.

**Entropy DP+B at 6.0 bpw beats uniform K6**: +8.5-11.9% across 4 tensors (mean +7.5%). At 6.0 bpw all tiles are K6, so the gain is from rotation, not allocation.

**Uniform K+B rarely beats uniform K at same bpw**: the 0.5 bpw sidecar cost typically exceeds the distortion reduction.

### 9. Sidecar Comparison (Exp8, Corrected)
Correct sidecar cost: 128+128 float16 = 512 bytes = **0.25 bpw** (not 0.5).

| Arm | Sidecar | K4 Benefit vs no-rotation |
|---|---|---|
| Hadamard-only | 0 bpw | **Harmful** (-62000% to -80000%) |
| BiIP-8bit | 0.125 bpw | +7.5-14.7% |
| BiIP-full | 0.25 bpw | +7.1-14.9% |

**Key finding**: Hadamard rotation WITHOUT BiIP scaling is catastrophically harmful — it destroys weight magnitude balance. BiIP scaling is essential. 8-bit quantized scales (0.125 bpw) retain 86-109% of full BiIP benefit.

## Limitations (Reviewer-Identified)
1. **Block-diagonal surrogate**: D_t values are not additive for full HWE (H_G, H_X are dense)
2. **Empirical-entropy lower bound**: Not a real arithmetic coder; no model/framing bytes counted
3. **Proxy HWE**: Not end-to-end KLD; local metrics can invert model-level results
4. **128×128 slices**: Not full-tensor; Hadamard only works on power-of-2 dimensions
5. **Single transform seed**: Entropy change varies from -0.5% to +2.9% across seeds
6. **Synthetic calibration**: Not real held-out activations

## Implications for Other Researchers
- **R21 (shared codebook)**: Eliminates per-tile model overhead, making entropy allocation practical. Combined savings (no sidecar + no model tables) stack.
- **R19 (interlayer alloc)**: Drop-in replacement: swap cost(K) = K×elements with cost(K) = H_t(K)×elements. At inter-layer level, model overhead is amortized (one model per tensor).
- **R3 (rotations)**: Rotation and entropy allocation are partially redundant. Optimal: entropy allocation on unrotated weights.
- **R28 (multi-precision)**: B as a per-tensor action is viable with entropy DP, giving +7.5% at 6.0 bpw. Sidecar at 0.125 bpw retains most benefit.
