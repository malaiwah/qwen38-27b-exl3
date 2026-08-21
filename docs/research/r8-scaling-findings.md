# R8-Scaling: AWQ/SmoothQuant Alternatives and Novel Scaling — Findings (v3, corrected)

## Executive summary

14 scaling strategies × 3 correction levels × 3 bit widths × 7 tensors = **882 runs, 0 errors, 4.6s**.

**v3 fixes (from 2 rounds of openai-reviewer):**
- v1 bugs: per-column quantizer tautology, wrong Cholesky orientation (lower not upper), kurtosis/Hessian shape errors (84 swallowed)
- v2 bugs: wrong Cholesky convention (U@U^T instead of U^T@U), per-column grid refit (16 grids/tile not 1), outer P used original W not cached w_pre, absolute damping instead of relative
- v3: U^T@U=H^{-1} Cholesky, cached tile grids (1 grid/tile), cached w_pre for outer P, relative damping (λ=0.01·mean(diag(H)))

**Key finding: GPTAQ does NOT subsume scaling.** With matched per-tile quantizer, scaling strategy choice matters at all correction levels. GPTQ error propagation provides +6-11% improvement; GPTAQ P-matrix adds ~1-2% on top.

**Best scaling with GPTAQ (K5, averaged):** Lp-norm p=∞ (+3.3% vs none+gptaq), AWQ/Hessian/Lp-norm p=2 (+2.6%), SmoothQuant (+2.5%).

**Novel weight-activation product hybrid: slightly NEGATIVE** (-0.4% at K5). Not a viable contribution.

## Corrected results (v3)

### GPTQ now helps (was broken in v1/v2)

| K | GPTQ improvement (unscaled) | GPTAQ improvement |
|---|---------------------------|-------------------|
| K4 | +11.0% | +11.3% |
| K5 | +10.1% | +11.4% |
| K6 | +6.0% | +8.5% |

GPTAQ adds only ~1-2% over GPTQ error propagation. The P-matrix correction is a modest refinement.

### Scaling impact with GPTAQ (K5, averaged across 7 tensors)

| Strategy | vs none+gptaq | Assessment |
|----------|-------------|------------|
| **lp_pinf** | **+3.3%** | Best |
| awq | +2.6% | Good |
| hessian | +2.6% | Good |
| lp_p2 | +2.6% | Good |
| lp_p1 | +2.6% | Good (= AWQ) |
| awq_no_norm | +2.6% | Good (= AWQ) |
| smoothquant | +2.5% | Good |
| per_tile | +0.1% | Neutral |
| none | 0.0% | Baseline |
| **product** | **-0.4%** | Slightly negative |
| kurtosis | -4.4% | Harmful |
| adaptive_α | -113.5% | Very harmful |
| variance | -161.1% | Very harmful |
| outlier | -533.9% | Worst |

### Novel strategies

**Weight-activation product** s_j = (mean|X_j| · max|W_{:,j}|)^α: **NOT viable**. -0.4% at K5. The product form doesn't capture the activation-to-weight ratio that makes SmoothQuant effective.

**Per-channel adaptive α** α_j = σ(log(|X_j|/|W_j|)): **HARMFUL**. -113.5% at K5. Non-uniform α creates extreme scale ratios that worsen per-tile dynamic range.

## Implications

### R3-Rotations / R9-GroupOrbit: scaling + rotation + GPTAQ may compose
Scaling is a diagonal preprocessing transform (not a correction). It composes with rotations: W' = H_out · W · diag(s) · H_in. R9's corrected results (post-Cholesky fix) show rotation+GPTAQ is SYNERGISTIC (+40.2% mean over best single method), not antagonistic. The full stack — scaling + rotation + GPTAQ + allocation — may be the winner. This has not been measured for scaling specifically; test scaling+rotation+GPTAQ vs rotation+GPTAQ to see if scaling adds incremental benefit.

### R6-GDN: gate-aware Hessian
Hessian-based scaling (+2.6%) is tied with AWQ. For GDN, gate-aware Hessian weighting (row-weight on output sensitivity) should compose with scaling (column-weight on input difficulty) since they operate on different dimensions.

### R1-RateDistortion: allocation + scaling + rotation + GPTAQ
R1's DP allocation (+25.5% unrotated) and R8's scaling (+2.6%) are both preprocessing steps. R9's corrected results suggest all four (allocation, scaling, rotation, GPTAQ) may compose synergistically. Test order: scaling → allocation → rotation → GPTAQ.

## Caveats and limitations

1. **Synthetic dominance:** The averaged percentages are dominated by synthetic tensors (98.6% of K5 aggregate). Real-weight proxy effects are near zero and mixed-sign (real tensors have KLD ≈ 1.01× noise floor). The +3.3% headline is effectively a 3-seed synthetic result.
2. **Per-tensor macro-relative:** If macro-averaged per-tensor relative effects instead of raw KLD means: lp_pinf +1.4%, SmoothQuant +1.3%, AWQ +1.1% — smaller but same ranking.
3. **Product hybrid qualified:** The -0.4% product result is for the α=0.5 toy formulation. Per-seed synthetic K5 effects are +1.88%, -6.18%, +1.03% — mixed, not uniformly dead. A different α or formulation might work.
4. **Per-tile scaling is near no-op:** The implemented per-tile scale copies one value per input-tile-column (8 unique values for 128×128), which nearly commutes with the per-tile quantizer. Its +0.1% effect is from GPTQ conditioning changes, not dynamic range improvement.
5. **In-sample calibration:** Calibration and scoring use the same samples. Real model activations would provide cleaner signal.
6. **128×128 subsamples:** Full-size tensors would show stronger channel variation.
7. **block_size == tile_size dependency:** One-grid-per-tile correctness requires block=tile=16. This is asserted in config but not enforced in code.
## Sidecar bytes
| Strategy | Sidecar (128×128, K5) | Total | Overhead |
|----------|----------------------|-------|----------|
| none | 0 | 10240 | 0% |
| per-channel | 256 | 10496 | 2.5% |
| per-tile | 128 | 10368 | 1.25% |
