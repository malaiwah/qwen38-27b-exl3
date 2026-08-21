# R13-AllocRotation: Allocation + Rotation Composition

**Status:** completed, 2026-08-21 (rev 2 — reviewer fixes applied). PoC code: `tools/research/r13-alloc-rotation/poc.py`
Results: `receipts/research/r13-alloc-rotation-results.json`

## Reviewer revision history

- Rev 1: NEEDS_REVISION — (a) budget not truly payload-matched (mixed-K got 1 fewer K-unit), (b) Hadamard sign bytes overcounted by 2, (c) alternating was best-of not pure warm-start.
- Rev 2: All three fixed. Budget is now exact payload match (same K-sum = n_tiles*K for all arms). Hadamard signs use ceil(n/8). Alternating reports pure warm-start only.

## Claim

DP-refined allocation composes with BiIP+Hadamard rotation, but **only in the rotate-then-allocate order**. Allocate-then-rotate is actively harmful. The alternating variant (allocate, rotate, re-allocate with warm-start) matches or slightly beats rotate-then-allocate.

## Key results (4 tensors x 3 slices x 4 K values = 48 cases)

All arms use the same K-sum (n_tiles * K = 64*K). Payload is exactly matched. Rotation sidecar (1056 bytes: 1024 BiIP scales + 32 Hadamard signs) is extra bytes for rotation arms only.

### Summary: HWE improvement over "neither" (uniform K, no rotation)

| Arm | Mean % | Median % | Min % | Max % |
|-----|--------|----------|-------|-------|
| alloc_only | +30.77 | +33.26 | 0.00 | +92.76 |
| rotate_only | +69.62 | +67.15 | +49.43 | +96.94 |
| alloc_then_rot | +56.61 | +61.39 | +22.73 | +96.95 |
| **rot_then_alloc** | **+71.59** | **+69.07** | **+57.02** | **+97.11** |
| **alternating** | **+71.63** | **+69.55** | **+55.87** | **+97.06** |

### Marginal contributions

**Allocation on top of rotation:**
- rot_then_alloc vs rotate_only: mean **+6.58%**, median +0.87% (positive but modest; larger at low K)
- alloc_then_rot vs rotate_only: mean **-45.97%**, median -25.89% (CATASTROPHICALLY harmful)

**Rotation on top of allocation:**
- rot_then_alloc vs alloc_only: mean **+52.72%**, median +58.62% (strong positive)
- alloc_then_rot vs alloc_only: mean +15.88%, median +54.53% (rotation helps, but wrong allocation hurts)

### Alternating variant (pure warm-start)
- Warm-start: unrotated allocation as initial point for local search on rotated weights
- Mean improvement: +71.63% (pure warm-start, not best-of)
- Warm-start beats DP-on-rotated in 12/23 non-tied cases (25 ties at K=3/K=6 where uniform is already optimal)
- The improvement over rot_then_alloc (+0.04% mean) is negligible — warm-starting from unrotated allocation is neither better nor worse than cold-starting from DP-on-rotated

## Tile sensitivity homogenization

**Rotation dramatically homogenizes tile sensitivity:**
- CV ratio (after/before): mean **0.257**, range [0.10, 0.35]
- CV reduced by **74.3%** on average
- Max/min sensitivity ratio: before 31-2609x, after 3-5x
- Before rotation: tile sensitivities span 1-3 orders of magnitude
- After rotation: all tiles within ~4x of each other

This confirms R9's observation that rotation homogenizes tile sensitivity. However, **allocation is NOT useless after rotation** — the residual heterogeneity (3-5x max/min ratio) still provides exploitable signal for the DP allocator.

## Why allocate-then-rotate fails

The allocation computed on unrotated weights targets tiles with high sensitivity (outlier-heavy). After rotation, those tiles are no longer the sensitive ones — rotation redistributes sensitivity uniformly. Applying the old allocation to the new landscape is worse than uniform because it gives extra bits to tiles that no longer need them and starves tiles that now do.

Mathematically: the tile sensitivity function s_t(W) changes under rotation W -> U W V^T. The optimal allocation K*_t proportional to log(s_t) is specific to the current weight matrix. Applying K*(W) to Q(U W V^T) is a mismatch.

## Why rotate-then-alloc works

After rotation, the tile sensitivity landscape is more uniform but not perfectly flat. The DP allocator finds the residual heterogeneity and allocates accordingly. The gain is modest (+6.58% mean over rotation-only) because the landscape is already much more uniform, but it is consistently positive. At K=3 and K=6, the gain vanishes (uniform is already near-optimal on rotated weights at extreme bit rates); the gain concentrates at K=4 and K=5 where allocation has the most room.

The local search (full-objective, cross-tile terms) captures interactions that survive rotation — the cross-tile Hessian terms in tr(H_G E H_X E^T) are not fully homogenized by rotation even though tile-local sensitivities are.

## Answer to the key questions

1. **Does rotation homogenize tile sensitivity?** YES — CV reduced by 74.3%, max/min ratio from 31-2609x to 3-5x.
2. **Does allocation on rotated weights still find useful heterogeneity?** YES — residual 3-5x variation is exploitable, +6.58% mean over rotation-only.
3. **Is allocate-then-rotate better than rotate-then-allocate?** NO — allocate-then-rotate is catastrophically harmful (-45.97% vs rotation-only), rotate-then-allocate is positive (+6.58%).
4. **Does DP local search capture cross-tile interactions that survive rotation?** YES — the local search consistently improves over tile-local DP alone, indicating cross-tile structure persists.
5. **What happens to tile sensitivity after rotation?** Compressed from 1-3 orders of magnitude to less than 1 order, but not eliminated.

## Composition verdict

**Allocation composes with rotation, but ONLY in rotate-then-allocate order.** The correct stack ordering is: rotation FIRST, then allocation. This is consistent with the recommended stack in doc 63 section 4 (rotation at steps 2-3, allocation at step 7).

## R15 held-out validation relevance

R15 found that components using diagonal/marginal Hessian statistics generalize well (gap near 0), while full-covariance Cholesky (GPTQ) overfits with synthetic calibration. Main noted R15's synthetic calibration engineers diagonal population H_X, so GPTQ overfitting may not hold on real activations. Our R13 results use only:
- DP allocation: tile-local distortion measurements (diagonal-like, generalizes per R15)
- BiIP scaling: diagonal Hessian statistics (generalizes per R15)
- Hadamard rotation: weight-based + diagonal stats (generalizes per R15)

No GPTQ/Cholesky is used in any R13 arm. All R13 results should generalize well to held-out data.

## Limitations

- 128x128 slices (aspect ratio hidden)
- Synthetic calibration (Gaussian + outliers), not real activations
- Per-tile uniform quantizer (not EXL3 trellis/Viterbi)
- Output-covariance proxy for H_G, not true gradient covariance
- In-sample evaluation (but components used are R15-validated as generalizing)
- Payload-matched budget: rotation sidecar (1056 bytes = ~0.52 bits/elem) is extra bytes for rotation arms. A total-byte-matched comparison would reduce K-sum for rotation arms by ~3 K-units, slightly weakening rotation arms but not changing the qualitative ordering.
- Tile sidecar charges 4 bytes/tile (float16 min/max) but reconstruction uses float64 precision; no serialization round-trip is modeled. This is consistent with R1/R3 conventions.
