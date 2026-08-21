# R7-NoiseShaping: Dead Ends

## 1. H_G^{1/2} Transform (Hessian-Weighted GPTQ)

**Approach:** Transform W̃ = H_G^{1/2} W, run GPTQ on W̃, inverse-transform.

**Why it fails:** H_G = Y^T Y/N has extreme eigenvalue spread (condition number ~1000×). Multiplying by H_G^{1/2} amplifies weight dynamic range by 10×, making quantization catastrophically worse (HWE 10-100× worse than RTN). Even with normalization (mean diagonal = 1), the result is still dominated by outlier channels.

**Root cause:** Output covariance inherits weight magnitude distribution. L55_down has range [-0.18, 0.48] — square root amplifies outliers.

## 2. Hessian-Basis Quantization (Rotation to H_X Eigenbasis)

**Approach:** Rotate W to H_X eigenbasis, quantize, rotate back.

**Why it fails:**
- Requires dense n×n rotation matrix (non-budget-matched unless folded into architecture)
- At low K, spreads error across ALL eigen-directions (including high ones) — anti-correlation near zero
- GPTQ in the rotated space is a no-op (diagonal Hessian → independent per-column updates)
- At K=6, slight HWE improvement (+2%) but not from shaping — from decorrelation

## 3. Schur Ordering via Eigendecomposition (v1 bug)

**Approach (original, buggy):** Eigendecompose H_X, sort columns by descending eigenvalue.

**Why it fails:** `np.linalg.eigh` returns eigenvalues sorted ascending; `argsort[::-1]` gives [127,...,0] for every Hessian. Eigenvalues index eigenMODES, not coordinate columns. The eigenvectors mapping modes to coordinates were discarded.

**Fix:** Use diag(H_X) (OBS column saliency) for coordinate-based ordering. This is the act-order approach used in v3/v4.

## 4. Adaptive Alpha from Condition Number

**Approach:** Derive alpha from Hessian condition number: α = 1/(1 + log10(cond)·0.3).

**Why it fails:** Produces values in [0.25, 1.0] that are too conservative. Alpha=1.0 was used throughout v4 but not formally tested against alternatives in this version. The condition number doesn't capture the relevant information about optimal correction strength.

## 5. Sigma-Delta Feedback Without GPTQ (v1/v2)

**Approach:** Pure spectral feedback without GPTQ error propagation.

**Why it fails:** Shapes error direction but doesn't reduce error magnitude. The feedback ADDS noise to future columns without compensating. Consistently 3-4% worse than RTN.

## 6. Independent Spectral Feedback at Tiny Strength (v4)

**Approach:** Add spectral feedback filter (inverse-sqrt, inverse-eig, flat) on top of GPTQ, normalized to unit upper-triangular Frobenius norm, at strength 0.01.

**Result:** At this strength (~0.1% of GPTQ correction norm), HWE changes by <0.5% (range -0.41% to +0.23% across 36 comparisons). This is a tiny perturbation.

**Status:** NOT a dead end per se — we simply cannot conclude anything about filter redundancy or optimality at this strength. A strength-matched-to-base sweep remains unrun and is needed for any conclusion about whether spectral filters add value on top of GPTQ.

## 7. Flat (Uniform) Feedback at Unmatched Norm (v1/v2)

**Approach:** Add uniform feedback (filter_type='flat') on top of GPTQ without norm matching.

**Why it fails:** The flat filter has much larger Frobenius norm than spectral filters (127.5 vs 3.5-7.2), so the same strength coefficient applies a 18-36× larger update. The "flat hurts" result from v1/v2 was confounded by unmatched norms. With norm matching (v4), flat is indistinguishable from other filters at the tested strength.
