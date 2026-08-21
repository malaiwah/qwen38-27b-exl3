# R7-NoiseShaping: Hessian Noise Shaping — Findings (v4, corrected Cholesky)

**Researcher:** R7-NoiseShaping  
**Date:** 2026-08-21  
**Status:** v4 POC with corrected Cholesky, reviewer-verified GPTQ recurrence

## Scope

- **Quantizer:** per-16-column-group uniform (NOT EXL3 trellis)
- **H_G:** output activation covariance Y^T Y/N (NOT true gradient/Fisher Hessian)
- **Tensors:** 4 fixed top-left 128×128 slices (L0/L55 gate/down), 3 calibration seeds
- **Metric:** tr(H_G · E · H_X · E^T) (covariance-weighted reconstruction proxy)
- **All budget-matched arms:** same quantizer, same fixed group scales from original W, clipped codes
- **GPTQ:** uses U[i, i+1:]/U[i, i] where U = chol((H_perm + λI)^{-1}).T (upper triangular, U^T U = (H_perm + λI)^{-1})
- **Reviewer-verified:** U is exactly upper triangular, U^T U = (H_perm + λI)^{-1} (worst rel residual 5.8e-16), row-ratio recurrence matches explicit Schur to 2.7e-15

## Sanity Check

- GPTQ alpha=0 equals RTN exactly (max diff 0.0)
- GPTQ alpha=1 differs from RTN (max diff 8.74e-3)
- U norm = 18.35 (non-zero, correction active)

## Results

### HWE Improvement vs RTN (macro mean, paired wins vs RTN)

| Strategy | K=3 | K=4 | K=5 | K=6 |
|---|---:|---:|---:|---:|
| GPTQ_LR | +12.3% (92%) | +14.6% (100%) | +13.0% (92%) | +15.1% (100%) |
| GPTQ_RL | +15.2% (100%) | +16.2% (100%) | +14.8% (100%) | +19.2% (100%) |
| GPTQ_random | +11.5% (92%) | +16.4% (100%) | +13.2% (92%) | +18.2% (100%) |
| **GPTQ_actorder** | **+21.0% (100%)** | **+22.1% (100%)** | **+20.6% (100%)** | **+24.4% (100%)** |
| GPTQ_rev_actorder | +5.6% (75%) | +5.5% (92%) | +3.1% (67%) | +9.9% (92%) |
| Error_projection* | +87.1% (100%) | +87.2% (100%) | +86.8% (100%) | +87.4% (100%) |

Win rates are vs RTN. * = non-budget-matched diagnostic.

### Head-to-Head: Act-Order vs Other Orderings (paired wins)

| Comparison | K=3 | K=4 | K=5 | K=6 | Total |
|---|---:|---:|---:|---:|---:|
| vs GPTQ_LR | 12/12 | 10/12 | 11/12 | 12/12 | 45/48 |
| vs GPTQ_RL | 11/12 | 11/12 | 11/12 | 12/12 | 45/48 |
| vs GPTQ_random | 12/12 | 12/12 | 12/12 | 12/12 | 48/48 |
| vs GPTQ_rev_actorder | 12/12 | 12/12 | 12/12 | 12/12 | 48/48 |
| **Overall** | 47/48 | 45/48 | 46/48 | 48/48 | **186/192 (97%)** |

Note: These 192 observations reuse only 3 distinct calibration matrices/orders across 4 tensors and 4 K values. They are correlated repeated comparisons, not 192 independent trials.

### Anti-Correlation (H_G-weighted, K=4)

| Strategy | Anti-corr |
|---|---:|
| RTN | -0.01 |
| GPTQ_LR | -0.18 |
| GPTQ_actorder | -0.23 |
| Error_projection* | -0.26 |

### Noise Shaping Trade-off (K=4)

| Strategy | HWE/RTN | MSE/RTN |
|---|---:|---:|
| RTN | 1.000 | 1.000 |
| GPTQ_actorder | 0.779 | 1.576 |
| GPTQ_rev_actorder | 0.944 | 1.156 |
| Error_projection* | 0.127 | 0.501 |

### Norm-Matched Filter Ablation (K=4, act-order, seed 42 only)

All filters normalized to unit upper-triangular Frobenius norm. At fs=0.01, the added filter is ~0.1% of the GPTQ correction norm (~8.7).

| Filter | L0_gate HWE | L0_down HWE | L55_gate HWE | L55_down HWE |
|---|---:|---:|---:|---:|
| No filter | 6.110e-4 | 5.308e-4 | 7.835e-4 | 6.336e-2 |
| Flat (fs=0.01) | 6.124e-4 | 5.303e-4 | 7.809e-4 | 6.334e-2 |
| Inverse-sqrt (fs=0.01) | 6.113e-4 | 5.309e-4 | 7.834e-4 | 6.328e-2 |
| Inverse-eig (fs=0.01) | 6.115e-4 | 5.309e-4 | 7.822e-4 | 6.331e-2 |

Across all 36 K4 tensor/seed/filter comparisons, HWE changes range from -0.41% to +0.23% (max absolute change <0.5%). At this filter strength (~0.1% of base correction norm), the perturbation is tiny. **No conclusion about filter redundancy or optimality can be drawn at this strength.** A proper test would sweep filter strength relative to the base correction norm.

## Key Findings

1. **Act-order (descending diag(H_X)) is the best of five tested orderings in this planted-outlier synthetic fixture**: +21-24% over RTN, winning 100% of paired comparisons vs RTN. Wins 186/192 (97%) head-to-head vs other orderings (correlated repeated comparisons, not independent trials).

2. **Noise shaping trades raw MSE for HWE**: Act-order HWE/RTN=0.78, MSE/RTN=1.58. Error is REDISTRIBUTED from high-curvature to low-curvature directions.

3. **Error spectrum is anti-correlated with Hessian spectrum**: Act-order achieves -0.23 (vs RTN's -0.01), approaching the diagnostic projection's -0.26.

4. **Reverse act-order shapes less aggressively**: +3-10% HWE improvement, +16% MSE increase. Confirms direction matters (descending is better than ascending).

5. **Spectral feedback filters at unit-norm fs=0.01 produce <0.5% max HWE change**: This is a tiny perturbation relative to the GPTQ correction (~0.1% of base norm). No conclusion about filter redundancy or optimality can be drawn. A strength-matched-to-base sweep remains unrun.

## Dead Ends

1. **H_G^{1/2} transform**: Output covariance has extreme eigenvalue spread. 10-100× worse.
2. **Hessian-basis rotation**: Non-budget-matched (n×n rotation). At low K, hurts HWE.
3. **Adaptive alpha from condition number**: Too conservative. Alpha=1.0 used throughout but not formally tested against alternatives in this version.

## Limitations

- Uniform quantizer, not trellis (may interact differently with Viterbi)
- H_G is covariance proxy, not true Fisher Hessian
- Synthetic calibration with planted 5× outliers on 6 channels, in-sample evaluation (same X used for ordering, Hessian, and scoring)
- 4 fixed top-left 128×128 MLP slices (aspect ratio hidden — real tensors are 17408×5120)
- 3 seeds resample calibration, not independent weights; only 3 unique calibration matrices
- No attention/GDN tensors, no random tiles, no full matrices
- No end-to-end KLD
- Scale/offset metadata not budgeted
- Act-order permutation is offline-only (Wq written to original positions, no stored permutation needed at inference)
- Cholesky fallback path returns pinv (not a valid upper Cholesky factor); should fail/retry instead

## Compatibility with Other Axes

Per R9's corrected v3 finding (Cholesky bug fixed): rotation and GPTQ are complementary on 3/4 tensors (+13-45% over best single method), with the optimizer correctly rejecting correction on L55_down where rotation alone is optimal. R9 also reports GPTQ (alpha=0, error propagation only) beats GPTAQ (alpha=0.25, P-matrix) after rotation on 3/4 tensors. The composition is acceptance-selected per-tensor, not a universal requirement. Act-order GPTQ is a sub-component: it determines the order of error propagation within the GPTQ correction step.
