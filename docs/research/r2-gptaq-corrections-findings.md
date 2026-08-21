# R2-GPTAQ: GPTAQ/ResComp Corrections + Adaptive Strength — Findings (v2, reviewer-corrected)

## Executive Summary

We fixed three critical bugs in the existing GPTAQ/ResComp implementation and explored
seven correction strategies. Key results (reviewer-verified):

1. **ResComp lazy-block propagation FIXED and verified** — block-size invariant to 3e-16.
2. **P-matrix formula and Cholesky convention verified** — U^T U = H^{-1}, P = triu(D U^T, 1) U.
3. **α=1.0 (paper-faithful) outperforms α=0.25** in 34/36 aggregate settings (wins 103/108 paired comparisons).
4. **Grid-searched α is the in-sample best** — often picks α > 1.0 when search extends beyond 1.0.
5. **Repeated requantization (not true iterative refinement) hurts** — 2 passes: +0.8% to +18% worse; 3 passes: +2.5% to +67% worse.
6. **Error-vector correction collapses toward GPTQ** — it largely suppresses the GPTAQ correction, not preserving it.
7. **Eigendecomposition P-matrix matches Cholesky** on well-conditioned Hessians.

## Important Caveats (from reviewer)

1. **Real-weight results use synthetic activations.** We use real Qwen3.8-27B weight slices
   (128×128) but generate synthetic Gaussian calibration (Xt + Gaussian drift for X). These
   cannot support claims about L0 vs L55 behavior, Qwen-specific patterns, or deployment
   recommendations. Relabel as "real weight slice + synthetic activation toy."

2. **Grid search is an in-sample oracle.** It selects α using the same asymmetric error
   subsequently reported, per seed. It is guaranteed to be no worse than candidate arms on
   the reported data. It does not identify a deployable or generalizing α.

3. **α=1.0 does NOT "consistently" beat α=0.25.** It wins 34/36 aggregate and 103/108 paired
   comparisons. It loses at L55_down K3 and L55_gate K3. Improvement vs α=0.25 ranges from
   -0.23% to +7.12%, and vs GPTQ from +0.18% to +12.12% — not a uniform "2-6%."

4. **The α=0.25 scaling explanation is unsupported.** If H and D are both scaled by s, P is
   invariant. The 0.25 is an empirical reference-code constant, not caused by Hessian scaling.

5. **"Unmodified-weight drift" is NOT an irreducible floor.** Asymmetric calibration can
   deliberately change weights to partially compensate. GPTAQ α=1.0 on large-drift synthetic
   reaches 6.23e-3 vs the unmodified-weight drift of 7.05e-3.

## Bug Fixes (Reviewer-Verified)

### Fix 1: ResComp Lazy-Block Propagation ✓ VERIFIED

**Bug:** Outer-block CAE term used `(W0[:, block] - W[:, block])` where `W0 = W.copy()`, so
the term was identically zero. No compensated state was cached.

**Fix:** Cache `w_pre = Ww[:, c].copy()` (before quantization) and `CAE = W0[:, c] - w_pre`
at each column. Use `CAE_cache @ P2[i:i+B, i+B:]` in the outer-block update.

**Verification:**
- `rescomp_broken`: max deviation 1.94e-2 across block sizes (NOT invariant)
- `rescomp_fixed`: max deviation 3.05e-16 (invariant to machine precision)
- `gptaq_rescomp`: max deviation 4.58e-16 (invariant)
- Block-size comparison at K4: broken varies 8.005e-4 to 8.062e-4; fixed constant at 8.062e-4.

### Fix 2: P-Matrix Factor Order ✓ VERIFIED (with convention correction)

**Convention:** H^{-1} = L L^T where L is lower Cholesky. Code returns U = L^T (upper),
so H^{-1} = U^T U. P = α · triu(D · U^T, 1) · U = α · triu(D · L, 1) · L^T.

**Verification:** Reviewer confirmed P matches row-wise D[q,q+1:] · inv((H+λI)[q+1:,q+1:])
to 8.33e-17. U^T U matched H^{-1} to 2.78e-17.

### Fix 3: GPTAQ Outer-Block Uses Cached w_pre ✓ VERIFIED

**Fix:** Cache `W_pre_cache` before quantization at each column; use it in outer-block:
`W_pre_cache @ P[i:i+B, i+B:]`.

## Mathematical Derivation

### GPTAQ Correction (GPTQv2, Eq. 9)

The asymmetric calibration objective: min ||(w + Δw)X - wX̃||²

Introduces residual r = wX̃ - wX, yielding (Eq. 9):
Δw = Standard GPTQ update + r X^T H^{-1}_{-q}

Via Cholesky (H^{-1} = U^T U) and neuron decomposition, this becomes a P-matrix multiplication:
correction = w_pre · P[q, q:], where P = α · triu(D · U^T, 1) · U, D = ΔX · X^T.

### ResComp CAE Correction (arXiv:2604.07955)

ResComp identifies that GPTAQ's column-level objective uses compensated weights w^(q)
instead of original w^(0). The CAE term: (w^(0) - w^(q)) · P2[q, q:],
where P2 = α · triu(X̃ · X^T · U^T, 1) · U.

### Adaptive α Formula

α = ||ΔX · X^T||_F / ||X · X^T||_F — ratio of asymmetric cross-covariance to Hessian magnitude.

**Per-column variant:** α_j = |sum_k(ΔX[j,k] · X[j,k])| / |sum_k(X[j,k]²)| — diagonal
of the cross-covariance ratio per input feature.

## Experimental Results

### Block-Size Invariance (Reviewer-Verified)

| Arm | Max Deviation | Invariant? |
|-----|-------------|-----------|
| GPTQ | 2.50e-16 | ✓ |
| GPTAQ α=1.0 | 2.50e-16 | ✓ |
| ResComp broken | 1.94e-02 | ✗ |
| ResComp fixed | 3.05e-16 | ✓ |
| GPTAQ+ResComp | 4.58e-16 | ✓ |

### Synthetic Results (Normal Drift, Best Per K)

| K | Best Arm | Asym Err | vs RTN | vs GPTQ | α |
|---|---------|---------|--------|---------|---|
| K3 | gptaq_grid | 3.207e-3 | +12.7% | +0.6% | 1.10 |
| K4 | gptaq_grid | 7.746e-4 | +13.0% | +2.0% | 0.93 |
| K5 | gptaq_grid | 2.637e-4 | +12.8% | +6.0% | 1.00 |
| K6 | gptaq_grid | 1.490e-4 | +12.5% | +9.6% | 1.03 |

### Large Drift Results (Best Per K)

| K | Best Arm | Asym Err | vs RTN | vs GPTQ | α |
|---|---------|---------|--------|---------|---|
| K3 | blended | 9.290e-3 | +11.8% | +8.6% | — |
| K4 | blended | 6.841e-3 | +12.2% | +11.1% | — |
| K5 | gptaq_grid | 6.339e-3 | +12.3% | +12.0% | 0.97 |
| K6 | gptaq_a1.0 | 6.230e-3 | +12.2% | +12.1% | 1.00 |

### Real Weight Results (Qwen3.8-27B slices + synthetic activations)

Consistent pattern across 7 real tensors. Best per K typically gptaq_grid or gptaq_a1.0.
vs GPTQ ranges from +0.2% (K3) to +10.2% (K6). **These are toy results with synthetic
activations — not deployment-grade.**

## Key Findings

### Finding 1: α=1.0 Outperforms α=0.25 (34/36 aggregate, 103/108 paired)

The paper-faithful α=1.0 outperforms the reference-code α=0.25 in 34 of 36 aggregate
tensor/K settings and 103 of 108 paired seed comparisons. It loses at L55_down K3 and
L55_gate K3. Direct improvement vs α=0.25 ranges from -0.23% to +7.12%.

The 0.25 multiplier is an empirical reference-code constant. The scaling explanation
(H=(2/N)XX^T) is mathematically unsupported: if H and D are both scaled, P is invariant.

### Finding 2: Grid Search Picks α ≈ 0.7–1.1 (In-Sample Oracle)

Grid search (now extended to α ∈ {0, 0.1, ..., 1.0, 1.1, 1.2, 1.5, 2.0}) selects
optimal α ranging from 0.70 (L55_down K3) to 1.13 (L55_gate K4). Several optima are
at or above 1.0, suggesting the true optimum may be slightly above 1.0 in some cases.

**Caveat:** This is an in-sample oracle — it selects α using the same error metric
subsequently reported. It does not identify a deployable α.

### Finding 3: Adaptive α (Asymmetry Ratio) is Too Conservative

The data-driven α = ||ΔX·X^T||_F / ||X·X^T||_F produces α ≈ 0.005–0.04 on synthetic
data (not 0.3–0.6 as previously claimed). It gives only +0.1% over GPTQ at K5–K6,
vs +6–10% for α=1.0. The Frobenius norm ratio averages over all directions, underestimating.

### Finding 4: Error-Vector Correction Collapses Toward GPTQ

Using (w_pre - Q[:,c]) · P instead of w_pre · P is approximately neutral relative to
GPTQ (within ±0.3%) but ranges from 0.1% to 13.8% WORSE than α=1.0 GPTAQ. The
correction is largely suppressed because the quantization error (w_pre - Q) is small
relative to w_pre, especially at high K. The correct conclusion: error-vector correction
disables the GPTAQ benefit, not preserves it.

### Finding 5: Repeated Requantization Hurts

Multi-pass requantization (2 or 3 iterations) consistently hurts:
- 2 passes: +0.8% to +18% worse than single-pass GPTAQ (aggregate degradation)
- 3 passes: +2.5% to +67% worse

This is repeated requantization (treating prior quantized matrix as new target), NOT
a derived iterative refinement. A proper iterative refinement would retain the original
W/X̃ target and recompute residuals. The current arm only shows that this ad-hoc
requantization loop hurts.

### Finding 6: Eigendecomposition P-Matrix Matches Cholesky

The eigendecomposition-based P-matrix (eigen → inverse → Cholesky → triangular) produces
identical results to standard Cholesky on well-conditioned Hessians. The eigenvalue
clipping path (min 1e-10) is never exercised because damping=0.01 already ensures
eigenvalues ≥ 0.01. The stability advantage is not demonstrated on this data.

### Finding 7: ResComp Alone Hurts Asymmetric Error

ResComp (both broken and fixed) produces worse asymmetric error than plain GPTQ at all K
values. The CAE correction optimizes a different objective (aligning with original FP
output) that conflicts with the asymmetric error metric. GPTAQ+ResComp combined is better
than ResComp alone — GPTAQ dominates and ResComp adds a small secondary correction.

### Finding 8: GPTAQ Benefit Grows with K

GPTAQ improvement over GPTQ increases monotonically with K:
- K3: +0.2–0.8% (quantization error dominates)
- K4: +1.9–2.6%
- K5: +5.3–6.2%
- K6: +9.5–10.2% (drift correction dominates when quant error is small)

## Dead Ends

1. **Repeated requantization** — diverges; not a valid iterative refinement experiment.
2. **Adaptive α via asymmetry ratio** — too conservative (α ≈ 0.005–0.04).
3. **Per-column adaptive α** — also too conservative, no benefit over scalar.
4. **Error-vector correction** — collapses toward GPTQ, disables GPTAQ benefit.
5. **ResComp alone** — hurts asymmetric error; only useful combined with GPTAQ.

## Architectural Compatibilities (from team findings)

- **GPTAQ + BAQ allocation (R1):** Compatible — allocation chooses WHERE to spend bits,
  GPTAQ corrects WHAT drift remains. Orthogonal.
- **GPTAQ + Hadamard rotation (R3/R9):** Now confirmed SYNERGISTIC (not antagonistic —
  the antagonism was a Cholesky convention bug). Full stack: rotation + GPTAQ + allocation.
- **GPTAQ + permutations (R4):** Compatible — permutations don't make error i.i.d.
- **GPTAQ + scaling (R8):** Scaling not subsumed by GPTAQ; they compose.
