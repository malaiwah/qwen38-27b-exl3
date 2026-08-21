# R17 — Diagonal-Covariance GPTQ That Generalizes

**Status:** completed, 2026-08-21. Reviewer v2: code confirmed, findings revised for accuracy.

## Executive summary

R15 found that full-covariance GPTQ overfits held-out (+57.5pp gap) while diagonal-stat transforms generalize perfectly. R17 tested whether diagonal-only, block-diagonal, or threshold GPTQ variants generalize better.

**Key finding: Block-diagonal GPTQ (16×16 blocks, global λ) is the safest GPTQ variant — it provides most of the error-correction benefit with minimal overfitting. Post-rotation, Full GPTQ is net positive (+1.8pp over rotation-only held-out) but overfits (gen gap +6.36pp); Block-diagonal GPTQ achieves +1.4pp with only +0.65pp gap. Forward-H GPTQ is catastrophically bad. True diagonal H^{-1} GPTQ overfits identically to Full GPTQ. GPTQ interferes with allocation and should not be combined ungated.**

## Setup

- 4 real Qwen3.8-27B tensors (L0_gate, L0_down, L55_gate, L55_down), 128×128 slices (first slice only)
- K = 3, 4, 5, 6; per-tile 16×16 uniform quantizer (matched for all arms)
- 5 random 80/20 train/test splits of 512 calibration samples
- Synthetic independent-channel calibration (R15 recipe: Gaussian + outliers, NO cross-channel correlation)
- Correct Cholesky: U = chol(inv(H+λI)).T
- Correct Hadamard convention: W' = U @ W @ V^T, H_X' = V @ H_X @ V^T (R11 convention)
- Block-diagonal uses GLOBAL λ (from full H diagonal mean), not per-block λ

## GPTQ variants tested

1. **Full GPTQ**: U = chol(inv(H+λI)).T, full error propagation with Schur complement
2. **Forward-H GPTQ**: W[:,rem] -= e_q * H[q,rem] / (H[q,q] + λ) — uses forward Hessian row + diagonal preconditioner. NOT a true diagonal H^{-1} approximation; it's a forward-H preconditioned heuristic.
3. **Diagonal H^{-1} GPTQ**: W[:,rem] -= e_q * Hinv[q,rem] / Hinv[q,q] — uses raw H^{-1} rows (no Cholesky, no Schur complement). Tests whether the Cholesky/Schur structure is the source of overfitting.
4. **Block-diagonal GPTQ**: Full Cholesky within 16×16 blocks (global λ), zero between blocks
5. **Threshold GPTQ**: Full Cholesky but zero U[i,j] if |U[i,j]| < 0.1 * |U[i,i]| (sparse Cholesky)

### Arm matrix (actual tested combinations)

| GPTQ variant | Standalone | + Rotation | + Rotation + Alloc |
|-------------|-----------|------------|-------------------|
| Full | ✓ | ✓ | ✓ |
| ForwardH | ✓ | — | — |
| DiagHinv | ✓ | ✓ | ✓ |
| BlockDiag | ✓ | ✓ | ✓ |
| Threshold | ✓ | ✓ | — |

ForwardH was not tested post-rotation (catastrophic standalone results made it pointless). Threshold was not tested with allocation (to limit runtime). All other combinations are tested.

## Results (corrected, reviewer-verified)

### Aggregate (macro mean across 4 tensors, 5 splits)

| Arm | K | In-sample | Held-out | Gen Gap | Overfit? |
|-----|---|-----------|----------|---------|----------|
| Full_GPTQ | 5 | +16.5% | −21.2% | +37.7pp | YES |
| ForwardH_GPTQ | 5 | −1203.5% | −1146.5% | −57.0pp | Catastrophic |
| DiagHinv_GPTQ | 5 | +14.6% | −21.5% | +36.1pp | YES |
| BlockDiag_GPTQ | 5 | +0.5% | −3.5% | +4.0pp | Marginal |
| Threshold_GPTQ | 5 | +13.5% | −17.4% | +30.9pp | YES |
| Rotation_Only | 5 | +73.7% | +74.3% | −0.6pp | No |
| Rotation_Full_GPTQ | 5 | +82.4% | +76.1% | +6.4pp | YES |
| Rotation_DiagHinv_GPTQ | 5 | +80.2% | +74.0% | +6.2pp | YES |
| Rotation_BlockDiag_GPTQ | 5 | +76.3% | +75.7% | +0.7pp | Marginal |
| Rotation_Threshold_GPTQ | 5 | +78.5% | +76.0% | +2.5pp | YES |
| Rotation_Alloc_Only | 5 | +76.3% | +76.3% | +0.0pp | No |
| Rotation_Alloc_Full_GPTQ | 5 | +80.8% | +73.7% | +7.1pp | YES |
| Rotation_Alloc_DiagHinv_GPTQ | 5 | +78.4% | +71.5% | +6.9pp | YES |
| Rotation_Alloc_BlockDiag_GPTQ | 5 | +75.5% | +74.6% | +1.0pp | Marginal |

### Paired comparisons (held-out HWE, 80 comparisons each)

| Comparison | A wins | B wins |
|-----------|--------|--------|
| ForwardH vs Full (standalone) | 0/80 (0%) | 80/80 (100%) |
| DiagHinv vs Full (standalone) | 42/80 (52%) | 38/80 (48%) |
| BlockDiag vs Full (standalone) | 76/80 (95%) | 4/80 (5%) |
| Threshold vs Full (standalone) | 48/80 (60%) | 32/80 (40%) |
| Rot+DiagHinv vs Rot+Full | 14/80 (18%) | 66/80 (82%) |
| Rot+BlockDiag vs Rot+Full | 42/80 (52%) | 38/80 (48%) |
| Rot+Threshold vs Rot+Full | 36/80 (45%) | 44/80 (55%) |
| Rot_Only vs Rot+Full | 17/80 (21%) | 63/80 (79%) |
| Rot_Only vs Rot+BlockDiag | 17/80 (21%) | 63/80 (79%) |
| Rot+Alloc_Only vs Rot+Alloc+Full | 52/80 (65%) | 28/80 (35%) |
| Rot+Alloc_Only vs Rot+Alloc+BlockDiag | 45/80 (56%) | 35/80 (44%) |

## Key findings

### 1. Forward-H GPTQ is catastrophically bad (confirmed)

The forward-H update W[:,rem] -= e_q * H[q,rem] / (H[q,q] + λ) produces −1147% HWE vs RTN at K=5. The forward Hessian row H[q,rem] points in the wrong direction for error correction — it represents correlation between channels, not the decorrelated residual direction that H^{-1} provides. This is NOT a true diagonal H^{-1} approximation and should not be confused with one.

### 2. True diagonal H^{-1} GPTQ overfits identically to Full GPTQ

Using raw H^{-1} rows (Hinv[q,rem] / Hinv[q,q]) without Cholesky/Schur complement produces the same overfitting pattern as Full GPTQ: in-sample +14.6%, held-out −21.5%, gen gap +36.1pp. The Schur complement (sequential column elimination) is NOT the source of overfitting — the off-diagonal H^{-1} entries themselves are the source. With synthetic independent-channel calibration, off-diagonal H^{-1} terms are dominated by sampling noise, and using them for propagation overfits.

### 3. Block-diagonal GPTQ is the best generalizing variant

Standalone: near-neutral (−3.5% vs RTN), gen gap +4.0pp (marginal).
Post-rotation: +75.7% held-out, gen gap +0.7pp (marginal). Wins 76/80 (95%) vs Full GPTQ standalone, 42/80 (52%) vs Full post-rotation.

**Why it works:** The block-diagonal approximation retains 8 blocks × 256 entries = 2048 entries (1920 directed off-diagonals) out of 16384 total (16256 directed off-diagonals) — roughly 12% of the full Hessian's off-diagonal entries. This limits the number of noise dimensions available for overfitting. With synthetic independent-channel calibration, the unrotated H_X off-diagonals are sampling noise; the block-diagonal structure bounds how much of this noise GPTQ can exploit.

Post-rotation, the situation changes: Hadamard rotation of heterogeneous diagonal variances (S_X depends on W column norms and H_X diagonal) creates DETERMINISTIC off-diagonal structure in H_X_t = V @ S_X^{-1} @ H_X @ S_X^{-1} @ V^T. These structured off-diagonals are real (not sampling noise) and GPTQ can exploit them. The block-diagonal approximation captures the within-block portion of this structure while avoiding the cross-block noise.

### 4. Post-rotation, Full GPTQ is net positive but overfits

Rotation_Only: +74.3% held-out (K=5)
Rotation_Full_GPTQ: +76.1% held-out → GPTQ helps by +1.8pp (wins 79% of paired comparisons)
Rotation_BlockDiag_GPTQ: +75.7% held-out → GPTQ helps by +1.4pp (wins 79% of paired comparisons)

But Full GPTQ gen gap is +6.36pp (in-sample 82.4% vs held-out 76.1%). The in-sample improvement is inflated by overfitting to train calibration noise. The held-out improvement is real but modest.

Block-diagonal GPTQ has gen gap +0.65pp — most of the held-out benefit with minimal overfitting.

### 5. GPTQ interferes with allocation (ungated)

Rot+Alloc_Only: +76.3% held-out
Rot+Alloc+Full_GPTQ: +73.7% held-out → GPTQ hurts by −2.6pp (Alloc_Only wins 65%)
Rot+Alloc+BlockDiag_GPTQ: +74.6% held-out → GPTQ hurts by −1.7pp (Alloc_Only wins 56%)

Allocation redistributes bits to minimize HWE. GPTQ's error correction changes the weight landscape that allocation was optimized for, degrading the allocation's effectiveness. **Ungated GPTQ should not follow allocation.**

### 6. Ranking is not 100% stable

The top-5 ranking varies across splits at K=5. The consistent patterns are:
- **Rotation_Alloc_Only** is top-5 in every split, and #1 in splits 1 and 5
- **Rotation_BlockDiag_GPTQ** is top-5 in every split (ranks 3-4 consistently), never #1
- **Rotation_Full_GPTQ** is top-5 in every split, #1 in split 4
- **Rotation_Threshold_GPTQ** is top-5 in 4/5 splits, #1 in splits 2 and 3
- **Rotation_Only** is top-5 in only 1/5 splits (rank varies 2-8)
- Only Alloc_Only, Threshold, and Full ever attain #1

## Implications for the recommended stack

**Current stack (R15):** BiIP → Hadamard → DP allocation → GPTQ(α=0, accept-if-improve gated)

**R17 findings:**
1. **Block-diagonal GPTQ is the safest GPTQ variant.** It provides ~80% of Full GPTQ's held-out benefit with ~10% of the overfitting risk (gen gap +0.65pp vs +6.36pp).
2. **Full GPTQ is net positive post-rotation** (+1.8pp held-out), confirming R14's finding. The overfitting is handled by accept-if-improve gating.
3. **Ungated GPTQ should NOT be combined with allocation.** GPTQ interferes with allocation's bit distribution (Alloc_Only beats Alloc+GPTQ 65% of the time). If both are used, GPTQ must be independently gated with separate validation data.
4. **Forward-H and DiagHinv GPTQ are dead ends.** Forward-H is catastrophic. DiagHinv overfits identically to Full GPTQ but provides no Cholesky/Schur benefit.

**Revised recommendation:** The recommended stack should use EITHER:
- BiIP → Hadamard → DP allocation (no GPTQ), OR
- BiIP → Hadamard → GPTQ (no allocation, with accept-if-improve gating)

If both allocation and GPTQ are desired, GPTQ must be gated with independent validation data. Block-diagonal GPTQ is the preferred GPTQ variant when gating is unavailable.

## Dead ends

- **Forward-H GPTQ:** Catastrophically bad (−1147% vs RTN). Forward H direction is wrong for error correction.
- **Diagonal H^{-1} GPTQ:** Overfits identically to Full GPTQ (gap +36pp). Off-diagonal H^{-1} entries are the source of overfitting, not the Cholesky/Schur structure.
- **Any GPTQ variant standalone (unrotated):** All overfit or are harmful with synthetic independent-channel calibration.
- **GPTQ + allocation (ungated):** GPTQ interferes with allocation's bit distribution. Net harmful.

## Open questions

1. **Block-diagonal GPTQ with real activations:** Does it sacrifice significant benefit vs full GPTQ? (Requires GPU forward pass)
2. **Optimal block size:** Is 16×16 optimal? Smaller blocks = safer but less correction. Larger blocks = more correction but more overfitting risk.
3. **Block-diagonal GPTQ with correlated calibration:** Does the R11 recipe change the standalone results?
4. **Threshold GPTQ with allocation:** Not tested in this experiment due to runtime constraints.
