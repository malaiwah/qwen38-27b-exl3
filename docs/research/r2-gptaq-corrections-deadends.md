# R2-GPTAQ: Dead Ends (v2, reviewer-corrected)

## 1. Repeated Requantization (NOT True Iterative Refinement)

**What was tested:** After one complete GPTAQ pass, treat the quantized+compensated
matrix as the new original and run another GPTAQ pass. 2 or 3 iterations.

**Result:** Hurts. 2 passes: +0.8% to +18% worse than single-pass. 3 passes: +2.5% to +67% worse.

**Why:** This is repeated requantization, not iterative refinement. The second pass
quantizes already-compensated weights (Ww = Q column-by-column after first pass), using
the same P-matrix computed from the original Hessian/drift. It over-corrects because the
compensated weights have different statistics.

**Correct conclusion:** This specific ad-hoc requantization loop hurts. A proper iterative
refinement would retain the original W/X̃ target and recompute residuals — this was NOT tested.

## 2. Adaptive α via Asymmetry Ratio

**Formula:** α = ||ΔX·X^T||_F / ||X·X^T||_F

**Result:** Too conservative. Produces α ≈ 0.005–0.04 on synthetic data (NOT 0.3–0.6 as
initially claimed — that was a reporting error). Gives only +0.1% over GPTQ at K5–K6,
vs +6–10% for α=1.0.

**Why:** Frobenius norm averages over all directions. The correction should be strong in
high-curvature directions, but the global ratio underestimates. The P-matrix already encodes
direction-specific information; scaling by a too-small global α wastes this.

## 3. Per-Column Adaptive α

**Formula (corrected):** α_j = |sum_k(ΔX[j,k]·X[j,k])| / |sum_k(X[j,k]²)| — diagonal of
cross-covariance ratio per input feature.

**Result:** Also too conservative. No benefit over scalar adaptive α. Per-column scaling
of P rows breaks the matrix coupling structure.

**Note:** The original implementation used sum(abs(dX*X)) (L1 statistic) instead of
abs(sum(dX*X)) (diagonal of cross-covariance). This was corrected per reviewer feedback.

## 4. Error-Vector Correction

**What was tested:** Use (w_pre - Q[:,c]) · P instead of w_pre · P.

**Result:** Collapses toward GPTQ. It's approximately neutral vs GPTQ (within ±0.3%) but
0.1% to 13.8% WORSE than α=1.0 GPTAQ.

**Why:** The quantization error (w_pre - Q) is small relative to w_pre, especially at high K.
Multiplying P by this small error largely disables the correction. The P-matrix encodes
activation cross-covariance (D = ΔX·X^T), not quantization-error structure.

**Correct conclusion:** Error-vector correction suppresses GPTAQ, not preserves it.

## 5. ResComp Alone (Without GPTAQ)

**Result:** Hurts asymmetric error by 1–2% vs GPTQ at all K values.

**Why:** ResComp optimizes alignment with original FP-flow output (w^(0)·X̃), which conflicts
with the asymmetric error metric (||Wq·X - W·X̃||²). The CAE correction pulls weights toward
the original, fighting the GPTQ compensation.

**Conclusion:** Only useful combined with GPTAQ, where GPTAQ dominates.

## 6. Eigendecomposition P-Matrix on Well-Conditioned Hessians

**Result:** Identical to Cholesky on our 128×128 matrices with 512 calibration samples.

**Why:** The Hessian is well-conditioned (condition number ~10–100). Eigenvalue clipping
at 1e-10 never activates (damping=0.01 ensures eigenvalues ≥ 0.01). The stability advantage
is not demonstrated.

**Note:** Would need to test with larger matrices, fewer calibration samples, or
near-singular Hessians to show a difference. Not a dead end per se, just unvalidated.

## 7. α=0.25 Scaling Explanation

**Initial claim:** The 0.25 multiplier in reference code is caused by H=(2/N)XX^T scaling.

**Reviewer finding:** Mathematically unsupported. If H and D are both multiplied by s and
damping scales with H, then L scales as 1/√s and P is invariant. The 0.25 is an empirical
reference-code constant, not a scaling artifact.

**Conclusion:** The 0.25 is just an empirical choice. Our unscaled Hessian doesn't justify
any particular α — the data shows α=1.0 is better but not because of scaling cancellation.
