# R14-NoiseShapeStack: Noise Shaping Within the Full Stack

**Status:** completed, 2026-08-21. PoC verified with real Qwen3.8-27B weights.
**In-sample only — GPTQ results may not generalize (see R15 held-out caveat below).**

## Executive Summary

**Act-order GPTQ becomes a near-no-op after rotation, but GPTQ itself (Schur-complement
error propagation) still provides +29.2% improvement post-rotation.** The key distinction:
act-order is a *diagonal-based* heuristic that becomes irrelevant when rotation uniformizes
diag(H_X) (CV drops 10.7× from 2.40 to 0.23), while GPTQ's error propagation uses the *full*
Hessian inverse, which retains exploitable off-diagonal structure even after rotation.

Rotation and noise shaping are **complementary in mechanism but partially redundant in effect**:
rotation already creates stronger error-Hessian anti-correlation (-0.39) than act-order does
(-0.22), so adding GPTQ post-rotation slightly *weakens* the anti-correlation (-0.35).

## Key Findings

### F1: Act-order is a near-no-op after rotation (confirmed)

Aggregation: macro mean = mean over 4 tensors of per-tensor mean over 3 slices × 3 seeds.
Paired win rate = fraction of 36 (tensor×slice×seed) comparisons where act-order HWE < LR HWE.
Per-K macro-mean act-order improvement over LR:

| Transform | K=3 | K=4 | K=5 | K=6 | Paired win rate (all K) |
|-----------|------|------|------|------|--------------------------|
| None      | +7.5% | +10.4% | +11.6% | +12.1% | 67-81% |
| Scale     | +13.4% | +15.0% | +13.8% | +15.3% | 86-97% |
| Rotate    | +0.2% | -1.1% | +1.0% | -0.6% | 58-64% |

After rotation, act-order's macro-mean improvement over LR drops from ~10% to ~0%
(descriptively near zero), and the paired win rate drops from ~80% to ~61%. Note:
the same 3 seeds are reused across all 4 tensors and 3 slices, so the 36 paired
comparisons share only 3 distinct calibration matrices (12× reuse each). Treat
win rates as descriptive, not independent — cluster-bootstrap CIs would be needed
for inference. For L55_down (the tensor with largest weight range), act-order
becomes **harmful** post-rotation (-4.8% to -10.9% per-tensor).

**Root cause:** Act-order sorts columns by descending diag(H_X). Rotation uniformizes
diag(H_X) — CV drops from 2.40 to 0.23 — making the sort nearly arbitrary. The diagonal
no longer contains useful ordering information.

**Caveat (tile regrouping):** Global act-order reorders which columns share each 16×16
frozen codebook, so the AO-vs-LR comparison conflates propagation order with tile
regrouping. A diagonal-H control (zero cross-column propagation) shows AO and LR still
differ solely from regrouping (max code change 3.05, MSE 0.574→1.150). The mechanistic
"diagonal no longer contains ordering information" claim would need a static-group
control to fully isolate propagation order from regrouping effects. The headline
result (AO near-no-op post-rotation) holds, but the root cause may be partly tile
regrouping rather than purely propagation-order irrelevance.

### F2: GPTQ (Schur complement) still helps post-rotation (+29.2%)

| Arm | HWE vs Rot+RTN |
|-----|----------------|
| Rot+GPTQ_LR (α=0) | +29.2% |
| Rot+GPTQ_actorder (α=0) | +27.7% |
| Rot+GPTAQ_LR (α=0.25) | +28.1% |
| Rot+GPTAQ_actorder (α=0.25) | +27.3% |
| Rot+GPTAQ_LR (α=1.0) | +14.0% |
| Rot+GPTAQ_actorder (α=1.0) | +14.9% |

GPTQ's error propagation `W[:,remaining] -= e * U[c, remaining] / U[c,c]` uses the full
Hessian inverse (U = chol(H^{-1}).T), not just the diagonal. The off-diagonal structure
of H^{-1} retains exploitable correlations even after rotation uniformizes the diagonal.

**GPTAQ α=1.0 is harmful post-rotation**. Matched comparisons (same order, varying α):
LR α=0 → +29.2% vs LR α=1.0 → +14.0% (α=1.0 HWE is 21.5% higher than α=0);
AO α=0 → +27.7% vs AO α=1.0 → +14.9% (α=1.0 HWE is 17.8% higher than α=0).
The P-matrix correction overcorrects when rotation has already made quantization error
more uniform. α=0 (pure GPTQ) or α=0.25 are both effective post-rotation.

**Caveat (α result):** Xt = X + iid noise in this experiment, so D = (Xt−X)Xᵀ has
population mean zero. The α>0 P-matrix is fitting random cross-covariance, not
realistic FP-flow drift. The α=1.0 harm result needs validation with actual paired
upstream activations or correlated drift; it should be read as a synthetic-null
qualification, not a general claim about α.

### F3: Rotation creates stronger anti-correlation than act-order

| Arm | Anti-correlation (K=5) |
|-----|----------------------|
| RTN | -0.005 |
| GPTQ_LR | -0.178 |
| GPTQ_actorder | -0.219 |
| GPTAQ_actorder (α=1) | -0.220 |
| **Rot+RTN** | **-0.385** |
| Rot+GPTQ_LR | -0.344 |
Rotation alone creates -0.385 anti-correlation vs act-order's -0.219 (Δr = -0.166;
note: Pearson r is not ratio-scale, so "2× stronger" is not a valid comparison).
Adding GPTQ post-rotation makes anti-correlation less negative (Δr = +0.041, from
-0.385 to -0.344). This descriptive Δr is consistent with partial redundancy but
does not establish causation — HWE still falls 29% from GPTQ post-rotation, so
the net effect is positive despite the weaker spectral anti-correlation.

**Interpretation:** Rotation redistributes error energy away from high-Hessian directions
by making the weight incoherent (uniformizing magnitude). GPTQ redistributes error by
propagating it through the Hessian inverse. After rotation, the Hessian inverse is more
uniform, so GPTQ's propagation may be less spectrally targeted — but it still reduces
HWE, suggesting it operates through a different mechanism (error cancellation, not
spectral redistribution).

### F4: Direction still matters post-rotation (but the optimal direction is unclear)

Reverse act-order (ascending diag(H_X)) is consistently worse than LR post-rotation
(-1.1% to -42.6%), confirming that GPTQ propagation direction matters. But the correct
direction (descending diag) is no longer clearly better than LR after rotation.

This means: **GPTQ should still be used post-rotation (for the Schur-complement benefit),
but act-order column reordering is unnecessary.** LR order is sufficient and avoids the
risk of harmful reordering for outlier-heavy tensors like L55_down.

### F5: Act-order composes with scaling (not just rotation)

Act-order provides +5.9-31.4% per-tensor improvement over LR WITH scaling
(macro-per-K mean ~14%, range 13.4-15.3%), comparable to or better than without
scaling (macro-per-K mean ~10%, range 7.5-12.1%). Scaling (BiIP) partially
uniformizes the diagonal (CV: 2.40 → 0.73) but not as aggressively as rotation
(CV → 0.23), so the diagonal still contains useful ordering information.

**Implication:** If using scaling without rotation, act-order is valuable. If using
rotation (which includes scaling), act-order is unnecessary.

### F6: Allocation composes with all components

| Arm | HWE vs RTN (K=5) |
|-----|------------------|
| RTN | baseline |
| GPTQ_LR | +12.4% |
| GPTQ_actorder | +30.7% |
| Alloc+RTN | +66.4% |
| Alloc+GPTQ_LR | +74.0% |
| Alloc+GPTQ_actorder | +79.5% |
| Rot+RTN | +90.6% |
| Rot+GPTQ_LR | +93.3% |
| Rot+GPTQ_actorder | +93.2% |
| Rot+Alloc+RTN | +91.8% |
| Rot+Alloc+GPTQ_LR | +93.5% |
| **Rot+Alloc+GPTQ_actorder** | **+93.7%** |
| Rot+Alloc+GPTAQ_actorder (α=1) | +92.6% |

Adding GPTQ to Alloc+RTN reduces HWE by 39.1% unrotated (5.11e-04 → 3.11e-04),
but only 23.5% post-rotation (1.25e-04 → 9.53e-05). So GPTQ's marginal benefit
shrinks after rotation (from 39.1% to 23.5%) but does not vanish — it remains
a meaningful improvement.

Act-order vs LR in the full stack: Rot+Alloc+GPTQ_actorder (9.53e-05) vs
Rot+Alloc+GPTQ_LR (9.84e-05) = 3.1% residual-HWE reduction (or +0.2 percentage
points of RTN reduction: 93.7% vs 93.5%). Unrotated, the gap is larger:
Alloc+GPTQ_actorder vs Alloc+GPTQ_LR = 21.3% residual-HWE reduction.

**Caveat (DP allocation):** The DP is exact for the supplied additive per-tile
distortion table, but compute_tile_distortions uses only principal H_G/H_X blocks
(diagonal tile submatrices) and omits cross-tile terms of the full Hessian-weighted
objective tr(H_G E H_X E^T). It is a separable proxy, not globally HWE-optimal.

### F7: MSE vs HWE trade-off confirms complementary mechanisms

| Arm | HWE/RTN | MSE/RTN |
|-----|---------|---------|
| RTN | 1.000 | 1.000 |
| GPTQ_actorder | 0.693 | 1.457 |
| Rot+RTN | 0.094 | 0.592 |
| Rot+GPTQ_LR | 0.067 | 0.729 |
GPTQ alone: HWE ↓30% but MSE ↑46% (error redistribution, not reduction).
Rotation alone: HWE ↓91% AND MSE ↓41% (error magnitude reduction).
Combined (Rot+GPTQ_LR): HWE ↓93% with MSE ↓27% vs RTN (rotation reduces magnitude,
GPTQ partially offsets by redistributing — MSE/RTN goes from 0.592 to 0.729).

This confirms the architectural insight: **rotation reduces error magnitude, GPTQ shapes
error direction.** They address different error structures and are complementary.

## Diag(H_X) Uniformization

| Tensor | CV original | CV after BiIP | CV after rotate |
|--------|-------------|---------------|-----------------|
| L0_gate | 2.40 | 0.73 | 0.23 |
| L0_down | 2.40 | 0.73 | 0.23 |
| L55_gate | 2.40 | 0.72 | 0.23 |
| L55_down | 2.40 | 0.74 | 0.22 |

BiIP scaling reduces CV by 3.3×, Hadamard rotation by 10.7× total.
The CV is identical across tensors because the synthetic calibration is the same
(Gaussian + outliers, seed-dependent). Real activations would show tensor-specific CV.

## Full Stack Marginal Contributions (K=5, macro mean)

| Step | HWE | vs RTN | Marginal |
|------|-----|--------|----------|
| RTN (baseline) | 1.52e-03 | 0% | — |
| + GPTQ act-order | 1.05e-03 | +30.7% | +30.7% |
| + GPTAQ α=1.0 | 1.07e-03 | +29.6% | -1.7% |
| + BiIP scaling | 3.97e-04 | +73.9% | +62.9% |
| + Hadamard rotation | 1.22e-04 | +92.0% | +69.2% |
| Full: Rot+GPTAQ α=0.25 | 1.04e-04 | +93.1% | +14.6% |

Biggest marginal gains: rotation (+69.2%), scaling (+62.9%), GPTQ (+30.7%).
GPTAQ α=1.0 is slightly harmful (-1.7%) unrotated and significantly harmful post-rotation.
Switching to α=0.25 post-rotation recovers +14.6%.

## Answer to Key Questions

1. **Does act-order become a no-op after rotation?**
   YES — nearly. Act-order's benefit drops from ~11% to ~0% post-rotation (58-64% win rate,
   barely above chance). For L55_down, it becomes harmful (-5% to -11%).

2. **Does act-order still shape error after rotation?**
   BARELY. The anti-correlation improvement from act-order (RTN: -0.01 → act-order: -0.22)
   is swamped by rotation's effect (Rot+RTN: -0.39). Act-order post-rotation actually
   *weakens* the anti-correlation (-0.39 → -0.32).

3. **Interaction between noise shaping (direction) and rotation (magnitude)?**
   COMPLEMENTARY BUT PARTIALLY REDUNDANT. Rotation reduces magnitude (MSE ↓41%) and creates
   spectral anti-correlation (-0.39). GPTQ shapes direction (HWE ↓30%) but partially
   disrupts rotation's spectral benefit. The net effect is still positive (+29% HWE
   reduction from GPTQ on top of rotation), but less than the sum of individual effects.

4. **Can noise shaping and rotation be complementary?**
   YES, but the composition is sub-additive. Rotation + GPTQ gives +93.3% over RTN,
   while rotation alone gives +90.6% and GPTQ alone gives +12.4%. The synergy (+2.7%
   over rotation alone) is real but modest. Act-order adds only 3.1% residual-HWE
   reduction over LR post-rotation (93.7% vs 93.5% with allocation) — a marginal
   gain that does not justify the complexity.

## Recommended Stack (from R14 in-sample findings)

**WARNING: All GPTQ results below are in-sample only.** R15 held-out validation
shows GPTQ overfits with synthetic calibration (in-sample +26.5%, held-out
-30.0%, gap +56.5pp; act-order gap +56.5pp vs +33pp for LR). However, R15's
synthetic calibration uses independent Gaussian channels, so the population H_X
is diagonal and every off-diagonal Cholesky term GPTQ exploits is pure
finite-sample noise by construction. This proves small-sample covariance risk
in a synthetic null, NOT that GPTQ generally fails — real correlated activations
remain untested. Components using diagonal/marginal statistics (BiIP, Hadamard,
DP allocation) generalize perfectly (gap ~0). **Unconditional GPTQ needs
held-out validation or accept-if-improve gating (R9 pattern: held-out +81.9%
vs in-sample +86.0%, gap +4.0pp).**

In-sample optimal stack:

1. BiIP diagonal balancing (scaling)
2. Signed Hadamard rotation (both sides)
3. DP tile allocation
4. GPTQ with LR order (α=0) — act-order is a near-no-op post-rotation and risks
   harmful reordering for outlier tensors like L55_down
5. If using GPTAQ P-matrix: α=0.25 (NOT α=1.0) post-rotation

This achieves +93.5% HWE reduction over RTN at K=5 in-sample (Rot+Alloc+GPTQ_LR).
Using act-order instead of LR gives +93.7% — a 3.1% residual-HWE reduction
(+0.2 percentage points of RTN reduction) that does not justify the added
complexity and risk.
## Limitations

- 128×128 slices from full tensors (aspect ratio hidden)
- Synthetic calibration (Gaussian + outliers), not real model activations
- Per-tile (16×16) uniform quantizer, not EXL3 trellis/Viterbi
- Output-covariance proxy for H_G, not true gradient covariance (Fisher)
- **In-sample evaluation only — GPTQ overfits on synthetic calibration**
  (R15: in +26.5%, held-out -30.0%), but real correlated activations untested.
  Use accept-if-improve gating (R9 pattern) or held-out validation for GPTQ.
