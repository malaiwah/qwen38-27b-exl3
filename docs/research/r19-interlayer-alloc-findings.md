# R19-InterlayerAlloc: Inter-layer Optimal K Allocation

**Status:** PoC complete, reviewer-revised. 2026-08-21.

## Reviewer-revised claim

Inter-layer DP allocation (multiple-choice knapsack across all layers × roles)
reduces total Hessian-weighted quantization error by **+39.1-47.3%** over uniform-K
at matched byte budget (K4-K6), and by **+40.7%** over the K5K6 sample recipe at
the same budget — on a 24-tensor (6 layers × 4 roles) synthetic+real proxy. The
KronQ joint-trace proxy `tr(H_G)×tr(H_X)` correlates with measured layer
sensitivity (Spearman ρ = 0.920, log-Pearson r = 0.928), but **simpler baselines
outperform it**: `tr(H_G)` alone achieves r = 0.966, and quantization MSE × joint-trace
achieves r = 0.990.

### Reviewer corrections applied
1. **BiIP rotation fixed**: Now uses correct R3 convention (W_t = S_G @ W @ S_X) with
   properly transformed Hessians (H_X,t = S_X^{-1} H_X S_X^{-1}). Rotation invariance
   verified to 7.37e-16 relative error.
2. **hash() removed**: Replaced with stable `role_seed()` function. Results are now
   reproducible across Python process restarts.
3. **Synthetic weights rescaled**: After adding outliers and low-rank structure, weights
   are rescaled to match the target std. For gate/down, targets are interpolated from
   real L0/L55 statistics. For qkv/out, L55 endpoints are missing from the npz, so
   hard-coded 0.01→0.015 defaults are used (real L0 qkv/out stats are NOT used for
   these roles due to the two-endpoint requirement).
4. **Simpler baselines added**: tr(H_G) alone, tr(H_X) alone, weight MSE, MSE×KronQ.
5. **Full-model caveats**: Extrapolation uses 56 layers (not the actual 64-layer
   architecture with mixed linear/full attention) and equal-size slices (not real
   per-module element counts). Results are illustrative, not production-grade.
6. **"4 tensors" typo fixed**: Every role has 6 tensors (6 sample layers).

## Experiment setup

- Synthetic weights for L10-L40 with layer-interpolated statistics. For gate/down:
  log-linear interpolation between real L0/L55 weight stds. For qkv/out: hard-coded
  0.01→0.015 defaults (L55 endpoints missing from npz). All synthetic weights include
  5% outliers and low-rank structure, rescaled to match target std.
- Per-tile (16×16) uniform quantizer, K=3,4,5,6,7
- Output-covariance-proxy HWE: `tr(H_G · E · H_X · E^T)` where E = W - Q(W),
  H_G = Y Y^T / P (NOT true gradient/Fisher Hessian)
- Synthetic calibration: Gaussian + per-channel log-uniform scales + 5% outlier channels.
  Each tensor gets independently generated calibration (not shared activations).
- Stable seeds via `role_seed(role, layer)` — no hash() dependency
- Full 56-layer model extrapolated via log-linear interpolation of distortion curves
  between sample layers. **WARNING**: Uses simplified 56-layer/4-role model, not the
  actual 64-layer architecture with mixed linear/full attention topology.

## Results

### 1. Per-layer distortion curves D_l(K)

All 24 tensors show the expected exponential decay of HWE with increasing K
(`D(K) ≈ c · 2^{-2K}`). The constant `c` varies by ~2 orders of magnitude across
tensors, indicating variable sensitivity.

### 2. Inter-layer DP allocation

| Budget level | Uniform D | DP D | Improvement |
|---|---|---|---|
| K3-all | 0.928 | 0.928 | 0.0% (all same K) |
| K4-all | 0.215 | 0.114 | **47.3%** |
| K5-all | 0.043 | 0.026 | **40.2%** |
| K6-all | 0.011 | 0.007 | **39.1%** |
| K7-all | 0.003 | 0.003 | 0.0% (all same K) |

At K5 budget, DP gives **+40.2%** improvement over uniform. The gain is substantial
at intermediate budgets (K4-K6) and zero at extremes (K3, K7) where all tensors
get the same K. Note: these are OC-proxy HWE results on one synthetic realization,
not held-out validated.

### 3. Layer sensitivity ranking

**Most sensitive** (highest HWE at K5):
1. L55_qkv (1.29e-2) — late-layer attention QKV (SYNTHETIC weight)
2. L55_out (4.30e-3) — late-layer attention output (SYNTHETIC weight)
3. L0_out (4.12e-3) — early-layer attention output (REAL weight)
4. L40_out (3.69e-3)
5. L30_gate (3.03e-3)

**Least sensitive**:
23. L10_gate (2.96e-4)
24. L0_gate (1.12e-4)

**Sensitivity ratio**: max/min ≈ 115×. This is a property of one synthetic
realization; the ratio varies with calibration seed (37× to 350× in reviewer's
robustness tests).

**Role sensitivity ranking** (sum across 6 layers at K5, each role has 6 tensors):
- **qkv**: 1.82e-2 (most sensitive)
- **out**: 1.41e-2
- **gate**: 5.82e-3
- **down**: 4.99e-3 (least sensitive)

Attention tensors (qkv, out) are 2.5-3.6× more sensitive than MLP tensors (gate, down).
This ranking is consistent with the K5K6 recipe's intuition of giving attention more bits.

**Layer sensitivity** (sum across 4 roles at K5):
- L55: 1.89e-2 (most sensitive, 2.4× over mean — but partly synthetic)
- L0: 7.83e-3
- L30: 5.74e-3
- L40: 5.36e-3
- L10: 2.87e-3
- L20: 2.40e-3 (least sensitive)

### 4. Role-dependent allocation

| Strategy | Total D at K5 budget | vs Global DP |
|---|---|---|
| Global DP | 0.0258 | baseline |
| Role-partitioned (proportional) | 0.0305 | -18.2% (worse) |
| Sensitivity-weighted (not budget-matched) | 0.241 | -834% (much worse) |

**Global DP dominates role-partitioned allocation by 18.2%.** The global DP can
transfer bits across roles — giving more to sensitive attention tensors regardless
of role partition. Sensitivity-weighted allocation starves low-sensitivity roles,
pushing them to K3 where distortion explodes. Note: the sensitivity-weighted
strategy is not budget-matched (spends 245,784 vs 251,928 bytes), making the
comparison illustrative rather than fair.

### 5. Interaction with rotation (CORRECTED)

BiIP + Hadamard rotation reduces HWE by 29.2-85.8% across all tensors (mean ~71%).
Rotation invariance verified to 7.37e-16 relative error.

| Metric | Unrotated | Rotated |
|---|---|---|
| CV across tensors | 1.457 | 1.156 |
| CV reduction | — | **20.7%** |
| DP improvement over uniform | +40.2% | +26.3% |
| Assignment differences | — | 4/24 differ |

**Key finding: Rotation partially homogenizes inter-layer sensitivity (20.7% CV
reduction), but the inter-layer DP gain remains substantial post-rotation (+26.3%).**
This contrasts with R13's finding that tile-level allocation nearly vanishes after
rotation (+6.6% marginal gain), because R13's homogenization is within-layer
(tile-level CV → 0.257), while inter-layer CV (1.156) remains much higher. Only
4/24 tensors change K assignment post-rotation.

### 6. Budget frontier

The full R-D frontier (40 budget points) shows smooth monotonic improvement.

**K5K6 recipe comparison:** At the K5K6 budget (288,792 bytes), DP achieves
0.0091 vs K5K6's 0.0154 — **+40.7% improvement**. DP gives more bits to the
most sensitive tensors (L55 attention → K7) and fewer to the least sensitive
(L0 gate → K4).

**K4K5 recipe comparison:** At K4K5 budget (239,640 bytes), DP achieves 0.0385
vs K4K5's 0.0685 — **+43.9% improvement**.

### 7. KronQ joint-trace comparison

The KronQ joint-trace sensitivity `tr(H_G) × tr(H_X) / (M×N)` correlates with
measured HWE sensitivity:

| Metric | Value |
|---|---|
| Spearman ρ | 0.920 |
| Pearson r (log) | 0.928 |

**Simpler baselines outperform joint-trace:**

| Proxy | log-Pearson r with HWE@K5 |
|---|---|
| tr(H_G) alone | **0.966** |
| tr(H_X) alone | 0.800 |
| Weight MSE@K5 | 0.601 |
| KronQ tr(H_G)×tr(H_X) | 0.928 |
| MSE@K5 × KronQ | **0.990** |

The joint-trace is not the best single proxy — `tr(H_G)` alone is better (0.966 vs 0.928).
The best proxy is MSE × KronQ (0.990), which combines quantization error with Hessian
trace. All correlations are in-sample (same synthetic H_X/H_G used for both score and
target), so they overestimate held-out predictive value. **KronQ-guided tiered
allocation performs worse than uniform** (-77.8% vs uniform K5) because tiered
allocation cannot exploit marginal distortion trade-offs the way DP does.

### 8. Full 56-layer model extrapolation (ILLUSTRATIVE ONLY)

**WARNING**: This uses a simplified 56-layer/4-role model, not the actual 64-layer
Qwen3.8-27B architecture (which has 48 linear-attention + 16 full-attention layers,
gate/up/down on all, z-gate on linear-attention, and different attention topology).
All tensors are 128×128 slices with equal byte costs. Results are illustrative.

Extrapolating to 56 layers × 4 roles (224 tensors):

| Strategy | Total D | vs K5K6 |
|---|---|---|
| Uniform K5 | 0.316 | — |
| Uniform K6 | 0.076 | — |
| K5K6 recipe | 0.113 | baseline |
| DP optimal | 0.080 | **+29.5%** |

**DP K distribution by role:**
- down: mostly K5 (37), some K6 (19)
- gate: mostly K6 (29), some K5 (24), few K4 (2), K7 (1)
- out: mostly K6 (41), some K7 (11), few K5 (4)
- qkv: mostly K6 (37), some K7 (10), some K5 (9)

**DP average K by layer depth:**
- Early (L0-L9): avg K ≈ 5.75-6.00
- Middle (L10-L45): avg K ≈ 5.75
- Late (L46-L55): avg K ≈ 6.00-6.25 (higher — more sensitive)

## Limitations

1. **Synthetic calibration**: Uses Gaussian + outliers, not real model activations.
   Each tensor gets independently generated calibration, not shared activations from
   a real forward pass.
2. **OC-proxy H_G**: H_G = Y Y^T / P is output-activation covariance, NOT true
   gradient/Fisher Hessian. Since Y = WX and quantization is homogeneous, HWE scales
   approximately as W_std^4, so independently generated calibration dominates
   sensitivity differences.
3. **Synthetic weights**: Only L0 (gate/down/qkv/out/z) and L55 (gate/down) are real.
   L55_qkv/out are synthetic. L10-L40 are interpolated. For gate/down, targets use real
   L0/L55 statistics. For qkv/out, hard-coded 0.01→0.015 defaults are used (L55 endpoints
   missing). Synthetic weight std is rescaled to match target after adding structure.
4. **Wrong architecture**: Uses 56 layers × 4 roles, not the actual 64-layer model
   with mixed linear/full attention topology, gate/up multiplicity, z-gate, etc.
5. **Equal-size slices**: All 128×128, hiding aspect ratio effects and per-module
   element count differences. Byte costs are equal across tensors.
6. **Per-tile uniform quantizer**: Not EXL3 trellis/Viterbi. R16 showed DP allocation
   composes orthogonally with quantizer type.
7. **In-sample evaluation**: No held-out validation of inter-layer allocation.
8. **Additive distortion model**: Minimizes sum of per-tensor OC-proxy HWE, not true
   end-to-end KLD. Cross-layer error accumulation is not modeled (R18's block
   propagation factors could improve this).
9. **hash() nondeterminism fixed**: Now uses stable `role_seed()` function.
10. **Sidecar precision mismatch**: Distortion uses FP64-exact tile min/max, while
    byte accounting charges two float16 sidecar values.

## Composition with existing stack

- **R1 (tile-level DP)**: R19 allocates K per layer/role; R1 allocates K per tile
  within each tensor. Hierarchical composition.
- **R3 (rotation)**: R19 inter-layer DP gain persists post-rotation (+26.3%), unlike
  R13's tile-level allocation (+6.6% marginal). Rotation homogenizes within-layer
  but not across-layer sensitivity.
- **R16 (non-uniform codebook)**: DP allocation composes orthogonally with quantizer
  type. R19's inter-layer DP should similarly compose.
- **R18 (block propagation)**: R18's block amplification factors could weight the
  DP objective: amp_l × D_l(K) instead of D_l(K). This would integrate block-level
  error propagation into the inter-layer knapsack.
- **R20 (info-theoretic)**: Channel capacity could serve as a prior for allocation
  when Hessian data is unavailable.

**Recommended composition order:**
1. BiIP + Hadamard rotation (R3)
2. Inter-layer K allocation (R19) — determines K per (layer, role)
3. Tile-level K allocation (R1) — determines K per tile within each tensor
4. GPTQ correction (R7/R14) — gated behind accept-if-improve (R9/R15)
