# R16-NonUniformCodebook: Non-uniform and Learned Codebooks for Tile Quantization

**Status:** completed, 2026-08-21. 5 quantizer strategies tested on 4 real Qwen3.8-27B tensors, 3 slices each, K=3-6, with and without BiIP+Hadamard rotation.

## 1. Executive summary

**Key finding: Non-uniform codebooks significantly improve quantization quality at matched K levels (+31-75% HWE for Hessian-weighted Lloyd-Max unrotated, +18-36% even after rotation at K≥5). However, the per-tile codebook storage cost (exponential in K) makes them NOT worth it at matched byte budget — uniform K+1 wins in 11/12 cases (one exception: L55_down/first, highest kurtosis). At low K (K=3-4) after rotation, non-uniform quantizers HURT (uniform is better); at high K (K=5-6) they help but the codebook overhead negates the gain.**

The central hypothesis (rotation removes heavy-tail structure → uniform quantization near-optimal) is partially confirmed: rotation dramatically reduces kurtosis and shrinks the non-uniform advantage, but does NOT eliminate it at high K (K≥5). The recommended stack (BiIP+Hadamard+DP) does not benefit from non-uniform codebooks at matched byte budget. Note: "Gaussian → uniform-optimal" is false — the optimal quantizer for Gaussian is non-uniform (Bennett's theorem). The correct mechanism is outlier removal, not Gaussianization.

## 2. Quantizer strategies implemented

| Strategy | Description | Init | Sidecar |
|----------|-------------|------|---------|
| Uniform | Min-max uniform per tile | N/A | 2 float32 (min, max) = 8 bytes |
| Lloyd-Max | Iterative optimal scalar quantizer | Uniform levels | 2^K float16 centroids |
| K-means | k-means++ init + Lloyd iterations | k-means++ | 2^K float16 centroids |
| Hessian-weighted Lloyd-Max | Centroids weighted by H_G[i,i]·H_X[j,j] | Uniform levels | 2^K float16 centroids |
| Distribution-optimal | Laplacian-fit + quantile init + Lloyd refine | Laplacian quantiles | 2^K float16 centroids |

## 3. Results at matched K levels (same number of quantization levels)

### 3.1 Unrotated weights — HWE improvement over uniform

| Quantizer | K=3 | K=4 | K=5 | K=6 | Win rate (K≥4) |
|-----------|-----|-----|-----|-----|----------------|
| **hw_lloyd** | **+31.1%** | **+52.2%** | **+59.0%** | **+75.2%** | **100%** |
| kmeans | -2.9% | +30.2% | +37.7% | +68.6% | 92% |
| lloyd_max | -13.5% | +9.7% | +10.1% | +35.0% | 67% |
| dist_optimal | — | — | — | — | 0% (harmful) |

**Hessian-weighted Lloyd-Max is the clear winner**: +52-75% improvement at K=4-6, 100% win rate. It directly optimizes the HWE objective by shifting centroids toward high-Hessian-weight elements. The improvement grows with K because more levels give the weighted centroids more flexibility.

**K-means (k-means++ init) outperforms Lloyd-Max (uniform init)**: 30-69% vs 10-35%. k-means++ initialization finds better local optima, especially at low K where uniform init gets stuck. At K=3, Lloyd-Max is actually harmful (-13.5%) due to poor local optima from uniform initialization.

**Distribution-optimal (Laplacian-fit) is HARMFUL**: mean -93.9% improvement (i.e., 94% worse). The Laplacian model is a poor fit for tile-level weight distributions. The quantile-based initialization produces worse codebooks than even uniform init. **Dead end.**

### 3.2 Rotated weights (BiIP + Hadamard) — HWE improvement over uniform, BY K

**CORRECTED (reviewer):** The pooled-K mean in v1 was misleading. Non-uniform quantizers
HURT at K=3-4 but HELP at K=5-6, even after rotation. The K=3 negative values dominated
the pooled mean.

| Quantizer | K=3 | K=4 | K=5 | K=6 |
|-----------|-----|-----|-----|-----|
| lloyd_max | -66.2% (0/12) | -21.4% (0/12) | **+15.5% (12/12)** | **+31.2% (12/12)** |
| kmeans | -62.4% (0/12) | +7.7% (10/12) | **+36.6% (12/12)** | **+61.2% (12/12)** |
| hw_lloyd | -49.3% (0/12) | -18.0% (1/12) | **+17.9% (12/12)** | **+35.5% (12/12)** |
| dist_optimal | -44.0% (0/12) | -29.1% (1/12) | -27.8% (0/12) | -23.7% (0/12) |

(Win rate = n/12 tensors×slices with positive improvement)

**After rotation, non-uniform quantizers HURT at K=3-4 but HELP at K=5-6** (100% win rate
for lloyd_max, kmeans, hw_lloyd at K≥5). The advantage shrinks from unrotated (+52-75% at
K=4-6) to rotated (+16-61% at K=5-6), but is NOT eliminated.

Root cause: **rotation reduces kurtosis to ~0** (sub-Gaussian), shrinking but not
eliminating the non-uniform advantage. At low K, the residual non-Gaussianity is
insufficient to overcome the non-uniform quantizer's sensitivity to initialization
(Lloyd-Max with uniform init gets stuck in bad local optima). At high K, more levels
give the quantizer enough resolution to exploit even small distribution deviations.

K-means (k-means++ init) is the most robust post-rotation: positive at K=4 (+7.7%),
strong at K=5-6 (+37-61%), because k-means++ initialization avoids the bad local optima
that plague Lloyd-Max at low K.

### 3.3 Kurtosis analysis

| Tensor | Kurtosis (unrotated) | Kurtosis (rotated) | Change |
|--------|---------------------|--------------------|----|
| L0_gate | 0.28 | -0.05 | -0.33 |
| L0_down | 0.15 | -0.03 | -0.18 |
| L55_gate | 0.18 | -0.05 | -0.23 |
| **L55_down** | **2.79** | **-0.02** | **-2.81** |

L55_down has the highest kurtosis (mean 2.79, max tile kurtosis 100!) and benefits most
from non-uniform quantizers unrotated. After rotation, kurtosis drops to near-zero
(slightly negative, sub-Gaussian), confirming that rotation removes heavy-tail structure
(outliers).

**Important correction (reviewer):** "Gaussian → uniform-optimal" is mathematically false.
The optimal scalar quantizer for a Gaussian distribution is non-uniform (Bennett's point
density theorem: optimal point density ∝ f(x)^{1/3}, which is denser near the mean).
The correct mechanism is: rotation removes heavy-tail structure (outliers) that non-uniform
quantizers exploit. For light-tailed distributions (low kurtosis), the optimal quantizer's
point density is close to uniform, making uniform quantization near-optimal. The non-uniform
advantage comes primarily from handling outliers, which rotation eliminates. Zero kurtosis
alone does not prove Gaussian — it means the distribution has no excess tail weight relative
to Gaussian, which is sufficient to remove the outlier structure that non-uniform quantizers
exploit.

## 4. Matched byte budget: the codebook overhead problem

### 4.1 Codebook storage cost

| K | Uniform bytes | Non-uniform bytes | Overhead |
|---|--------------|-------------------|----------|
| 3 | 6,657 | 7,169 | 7.7% |
| 4 | 8,705 | 10,241 | 17.6% |
| 5 | 10,753 | 14,337 | 33.3% |
| 6 | 12,801 | 20,481 | 60.0% |

The codebook overhead grows exponentially with K: 2^K float16 centroids per tile. At K=6, the codebook alone costs 60% more than the entire uniform payload.

### 4.2 Matched-byte DP: non-uniform vs uniform at IDENTICAL total bytes

**Proper byte-budget DP (reviewer-requested):** Both uniform and non-uniform use DP
allocation with exact per-tile byte costs (uniform: 32K+8, non-uniform: 32K+2^(K+1)),
allowing mixed K, at identical total byte budgets. K=3 is skipped for non-uniform because
the minimum codebook cost exceeds the uniform K=3 budget.

#### Unrotated

| Quantizer | K=4 | K=5 | K=6 | Win rate |
|-----------|-----|-----|-----|----------|
| lloyd_max | 1.83× | 2.80× | 2.97× | 1/12 (L55_down/first only) |
| kmeans | 1.41× | 2.35× | 2.25× | 1/12 (L55_down/first only) |
| hw_lloyd | 1.28× | 1.99× | 1.88× | 1/12 (L55_down/first only) |

(Ratio > 1 means uniform DP wins; lower is better for non-uniform)

**At matched byte budget with DP allocation, non-uniform wins ONLY on L55_down/first**
(the extreme outlier tensor, kurtosis up to 100). On L55_down/first, non-uniform dominates:
kmeans K=5 ratio=0.288 (3.5× better!), kmeans K=6 ratio=0.174 (5.7× better!),
hw_lloyd K=5 ratio=0.457, hw_lloyd K=6 ratio=0.240. For all other 11 tensor/slice
combinations, uniform DP wins by 1.2-3.6×.

This is a stronger result than the fixed-K comparison: even with optimal bit allocation,
non-uniform codebooks only beat uniform on the heaviest-tailed tensor. For typical tensors
(kurtosis < 3), uniform DP allocation at the same byte budget is always better.

#### Rotated

| Quantizer | K=4 | K=5 | K=6 | Win rate |
|-----------|-----|-----|-----|----------|
| lloyd_max | 3.23× | 3.98× | 7.19× | 0/12 |
| kmeans | 2.74× | 3.08× | 5.48× | 0/12 |
| hw_lloyd | 3.00× | 3.87× | 6.83× | 0/12 |

**After rotation, uniform DP wins ALL 12/12 cases at every K.** The ratios are much worse
for non-uniform (3-7× vs 1.3-3× unrotated), confirming that rotation + uniform is the
optimal strategy at matched byte budget.

### 4.3 Fixed-K comparison (superseded by §4.2 but retained for reference)

The original fixed-K comparison (non-uniform K vs uniform K+1 without DP allocation) showed
similar results: hw_lloyd K=4 vs uniform K=5 unrotated had ratio 2.20× mean (11/12 uniform
wins, one exception L55_down/first ratio 0.790). The byte-budget DP in §4.2 is more
rigorous and supersedes this analysis.

## 5. DP allocation with non-uniform quantizers

### 5.1 Unrotated

| Quantizer | K=4 mean improvement | K=5 mean improvement |
|-----------|---------------------|---------------------|
| uniform | 50.4% | 42.9% |
| lloyd_max | 58.6% | 52.4% |
| dist_optimal | 60.9% | 55.8% |

DP allocation composes with all quantizer types, with comparable or slightly better improvement for non-uniform quantizers. The allocation and quantizer choice are **orthogonal** — the allocator redistributes bits across tiles regardless of how each tile is quantized.

### 5.2 Rotated

After rotation, DP allocation improvement drops to ~0% for all quantizers, confirming R13's finding that rotation homogenizes tile sensitivity.

## 6. Lloyd-Max convergence

| Quantizer | Mean iters | Std | Max iters |
|-----------|-----------|-----|-----------|
| Lloyd-Max | 7.4-8.8 | 1.8-3.4 | 11-25 |
| K-means | 7.8-8.3 | 1.9-3.1 | 14-20 |
| Dist-optimal | 7.8-8.8 | 2.7-3.0 | 14-22 |

All quantizers converge in ~8 iterations on average, max ~25. Convergence is fast and reliable. Higher-kurtosis tiles (L55_down) take slightly more iterations (8.8 vs 7.4 for Lloyd-Max).

## 7. Key insights

1. **Diagonal-Hessian-weighted Lloyd-Max is the best non-uniform quantizer** (+52-75% HWE at K=4-6, 100% win rate unrotated). It minimizes the diagonal separable surrogate Σ_ij diag(H_G)_i·diag(H_X)_j·e_ij² (not full HWE tr(H_G E H_X E^T), which has off-diagonal cross-terms). The weighted centroid and midpoint boundary formulas are correct for this surrogate. In-sample evaluation only; held-out validation not performed.

2. **Initialization matters more than algorithm**: k-means++ init (33% mean improvement) vastly outperforms uniform init (10% mean). The Lloyd algorithm is the same; only initialization differs. At K=3, uniform init is actively harmful.

3. **Rotation shrinks but does NOT eliminate the non-uniform advantage** (kurtosis → ~0, but residual structure exploitable at K≥5). Post-rotation: non-uniform HURTS at K=3-4 (bad local optima) but HELPS at K=5-6 (+16-61%, 100% win rate). K-means++ init is the most robust post-rotation (positive at K=4 already). The mechanism is outlier removal (not "Gaussian → uniform-optimal" — the optimal Gaussian quantizer is non-uniform per Bennett's theorem). Rotation removes heavy-tail structure that non-uniform quantizers exploit; at high K, residual non-uniformity in the distribution is still exploitable.

4. **Per-tile codebook storage is prohibitive** (7.7-60% overhead, exponential in K). At matched bytes with DP allocation, uniform beats non-uniform in 11/12 cases (exception: L55_down/first, highest kurtosis, where kmeans wins by 2-6×). After rotation, uniform wins all 12/12. This is a fundamental limitation of per-tile non-uniform quantization.

5. **EXL3 trellis coding uses a SHARED codebook** (O(1) storage, not per-tile). This avoids the exponential overhead. Our proxy's per-tile model overestimates the real cost. A shared-codebook experiment would be needed to assess whether non-uniform quantization is worthwhile in the trellis framework.

6. **Laplace-quantile-init Lloyd is harmful for HWE** (-94% macro mean) but improves MSE by 11% on average. The method is misnamed "distribution-optimal" — it only uses Laplacian quantiles for initialization, then runs ordinary empirical Lloyd. The HWE harm comes from objective mismatch (minimizing MSE, not HWE). **Renamed: Laplace-quantile-init Lloyd.** The Laplacian model fit itself is not evaluated; the dead-end claim is specifically about this initialization strategy, not about Laplacian modeling in general.

7. **Allocation composes with quantizer choice** (payload-budget DP): ~50-60% improvement for uniform, Lloyd-Max, and Laplace-init at K=4-5. K-means and hw_lloyd were not tested in the payload-budget DP. The byte-budget DP (§4.2) shows that at matched bytes, the codebook cost differential means non-uniform allocation is at a disadvantage. "Orthogonal" is approximate — the interaction has not been quantified.

## 8. Implications for the recommended stack

- **After rotation at K≥5**: non-uniform quantizers still help at matched K (+18-36% for hw_lloyd, +37-61% for kmeans), but at matched bytes with DP allocation, uniform wins ALL 12/12 (ratio 3-7×).
- **After rotation at K=3-4**: non-uniform quantizers HURT at matched K (uniform is better). Use uniform.
- **Before rotation**: non-uniform quantizers help at matched K (+52-75% for hw_lloyd), but at matched bytes with DP, uniform wins 11/12 (exception: L55_down/first, highest kurtosis).
- **At matched bytes with DP**: uniform beats non-uniform in 23/24 unrotated cases and 36/36 rotated cases. The one exception is L55_down/first (extreme outliers).

**Conclusion (scoped to per-tile codebooks): per-tile non-uniform codebooks are NOT recommended for the production stack at matched byte budget.** The codebook overhead (7.7-60%, exponential in K) exceeds the quality gain for all tensors except the most heavy-tailed. Rotation removes the heavy-tail structure that non-uniform exploits, making uniform near-optimal post-rotation. The production/shared-codebook question (EXL3 trellis) remains OPEN — a shared codebook with O(1) storage could change this conclusion.

## 9. Implications for R17 (diagonal GPTQ) and R19 (interlayer alloc)

- **R17**: Non-uniform codebooks change the quantization distortion model. For diagonal GPTQ, the per-tile distortion c_t depends on the quantizer, not just Δ²/12. However, since rotation makes uniform optimal, the GPTQ error propagation should use uniform quantization. Non-uniform codebooks are not needed for GPTQ.
- **R19**: The allocation formula c_t = f(quantizer, tile) changes with non-uniform quantizers. The DP allocation results show that allocation and quantizer are orthogonal, so interlayer allocation should also be orthogonal to quantizer choice. The key insight for R19: use the actual quantizer distortion (measured, not assumed) for allocation, regardless of whether it's uniform or non-uniform.

## 10. Limitations

- Per-tile codebook storage model (worst case). EXL3 trellis uses shared codebooks — a shared/groupwise codebook experiment is the most important missing experiment.
- Centroids computed in float64 but counted as float16 — no serialize/decode round-trip. At K=5-6, float16 rounding of centroids could change HWE.
- 128×128 slices from full tensors (aspect ratio hidden, per R15 caveat).
- Synthetic calibration (diagonal Hessian stats generalize per R15, but full-covariance GPTQ may not). hw_lloyd evaluated in-sample only.
- No held-out validation of hw_lloyd specifically (though it uses diagonal Hessian stats which generalize per R15).
- No end-to-end KLD measurement.
- K=3 Lloyd-Max results are unstable (uniform init gets stuck in bad local optima).
- Empty clusters not reseeded; single k-means++ seed per tile; no multi-restart comparison.
- No normality test beyond kurtosis; no true Gaussian/Laplacian Lloyd-Max baseline.
- Tile-size and codebook-sharing-granularity sweeps not performed — the prohibitive overhead is specific to 16×16 per-tile codebooks.

## 11. Artifacts

| File | Description |
|------|-------------|
| `tools/research/r16-nonuniform-codebook/poc.py` | PoC implementation |
| `receipts/research/r16-nonuniform-codebook-results.json` | Full results JSON |
| `docs/research/r16-nonuniform-codebook-findings.md` | This document |
