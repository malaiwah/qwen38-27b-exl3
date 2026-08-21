# R22 — Block-Diagonal GPTQ + Allocation Composition (final, reviewer-corrected)

**Status:** completed, 2026-08-21. Two reviewer rejections (v1/v2: false budget-bug claim; v2/v3: additive filter changed allocator semantics). Final code uses R17-exact allocator. Reviewer confirmed exact grid match on L0_gate seed 42 K=5.

## Executive summary

**Block-diagonal GPTQ does NOT compose synergistically with per-tile allocation.** With R17's exact allocator, BD-GPTQ+alloc is worse than BD-GPTQ alone (−0.94pp, 21 wins / 39 losses / 20 ties out of 80) and worse than allocation-only (−0.39pp, 35/45/0). Re-allocation after GPTQ is neutral (−0.05pp, 13/15/52 ties). **R17's bifurcation recommendation is confirmed: use EITHER allocation OR GPTQ, not both.** BD-GPTQ remains the safer GPTQ variant (gen gap +0.65pp vs Full's +6.36pp).

## Setup

- 4 real Qwen3.8-27B tensors (L0_gate, L0_down, L55_gate, L55_down), 128×128 first-slice
- K = 3, 4, 5, 6; per-tile 16×16 uniform quantizer (matched for all arms)
- 5 random 80/20 train/test splits of 512 calibration samples
- Synthetic independent-channel calibration (R15 recipe, worst case for GPTQ)
- R17-exact allocator: bit-budget DP (`tile_k_dp_allocate` with `k*tile_elements`) + full-HWE `local_search_refine` (K range [3,7], max_iters=10 matching R17's `rotation_gptq`)
- No additive filter in local search (reviewer found it pruned 59/80 grids)
- Re-allocation: DP-only on corrected distortions (no local_search_refine, which evaluates plain RTN HWE and erases correction)
- Paired comparisons report wins/losses/ties separately (ties not counted as B wins)

## Results (aggregate, 5 splits × 4 tensors)

### Held-out improvement vs RTN (K=5)

| Arm | In-sample | Held-out | Gen Gap | Overfit? |
|-----|-----------|----------|---------|----------|
| Rot_Only | +73.7% | +74.3% | −0.59pp | no |
| Rot_Alloc | +76.3% | +76.3% | +0.00pp | no |
| Rot_FullGPTQ | +82.4% | +76.1% | +6.36pp | YES |
| Rot_BDGPTQ | +76.3% | +75.7% | +0.65pp | marginal |
| Rot_Alloc_FullGPTQ | +80.8% | +73.7% | +7.12pp | YES |
| Rot_Alloc_BDGPTQ | +75.5% | +74.6% | +0.95pp | marginal |
| Rot_BDGPTQ_Realloc | +76.4% | +75.6% | +0.81pp | marginal |
| Rot_Alloc_BDGPTQ_Realloc | +76.4% | +75.6% | +0.78pp | marginal |
| Rot_Alloc_FullGPTQ_Realloc | +82.1% | +75.6% | +6.55pp | YES |

### Paired comparisons (wins/losses/ties, n=80)

| Comparison | A wins | B wins | Ties | Δ (pp) |
|-----------|--------|--------|------|--------|
| **Alloc+BDGPTQ vs BDGPTQ-only** | **21** | **39** | **20** | **−0.94** |
| **Alloc+BDGPTQ vs Alloc-only** | **35** | **45** | **0** | **−0.39** |
| Alloc+FullGPTQ vs FullGPTQ-only | 12 | 48 | 20 | −1.85 |
| Alloc+FullGPTQ vs Alloc-only | 28 | 52 | 0 | −1.30 |
| BDGPTQ+Realloc vs BDGPTQ-only | 13 | 15 | 52 | −0.05 |
| Alloc+BDGPTQ+Realloc vs Alloc+BDGPTQ | 44 | 16 | 20 | +0.90 |
| Full+Realloc vs Alloc+Full | 49 | 11 | 20 | +1.53 |
| Triple(realloc) vs Alloc-only | 51 | 29 | 0 | +0.51 |
| Triple(realloc) vs BDGPTQ-only | 18 | 13 | 49 | −0.04 |

### Per-K interaction (Alloc+BDGPTQ vs BDGPTQ-only)

| K | A wins | B wins | Ties | Δ (pp) | Note |
|---|--------|--------|------|--------|------|
| K=3 | 0 | 0 | 20 | +0.00 | All ties: allocation is uniform (min K) |
| K=4 | 5 | 15 | 0 | −1.48 | BD-GPTQ alone better |
| K=5 | 9 | 11 | 0 | −1.09 | BD-GPTQ alone slightly better |
| K=6 | 7 | 13 | 0 | −1.18 | BD-GPTQ alone better |

## Interference mechanism

### Cross-block coupling (measured directly from U matrix)

| Tensor | Full off-diag | BD off-diag | Full coupling mass | BD coupling mass |
|--------|--------------|-------------|-------------------|-----------------|
| L0_gate | 8128/8128 | 960/8128 | 316.06 | **0.0000** |
| L55_gate | 8128/8128 | 960/8128 | 316.34 | **0.0000** |
| L0_down | 8128/8128 | 960/8128 | 316.14 | **0.0000** |
| L55_down | 8128/8128 | 960/8128 | 335.82 | **0.0000** |

BD-GPTQ has exactly zero cross-block coupling (U is block-diagonal by construction; 960 within-block off-diagonals out of 8128 total). Full GPTQ couples all columns. Coupling mass = Σ|U[i,j]/U[i,i]| for cross-block entries (not actual update magnitude, which also depends on e_q).

### Allocation stability (max_iters=10, matching R17)

| Tensor | Full alloc change | BD alloc change |
|--------|------------------|-----------------|
| L0_gate | 2/64 | 2/64 |
| L55_gate | 6/64 | 6/64 |
| L0_down | 0/64 | 9/64 |
| L55_down | 12/64 | 12/64 |

GPTQ changes 0-12 tiles out of 64. More than initially claimed (0-2 was from the buggy additive filter). The allocation is moderately stable but not as static as v2 suggested.

### Correlation(GPTQ benefit, tile sensitivity)

| Tensor | Full | BD |
|--------|------|-----|
| L0_gate | +0.222 | +0.628 |
| L55_gate | +0.110 | +0.404 |
| L0_down | +0.280 | +0.494 |
| L55_down | +0.459 | +0.723 |

Both positive — GPTQ helps sensitive tiles more. BD has stronger correlation, targeting sensitive tiles more effectively.

## Key findings

1. **BD-GPTQ interferes with allocation** (−0.94pp vs BDGPTQ-only, 21/39/20). R17's finding is confirmed with the exact allocator. The combination is net harmful.
2. **Full GPTQ interferes more** (−1.85pp vs FullGPTQ-only, 12/48/20). Both GPTQ variants hurt when combined with allocation.
3. **Re-allocation is neutral** (−0.05pp, 13/15/52 ties). Most cells are ties because the DP-only reallocation (no local search) produces similar grids.
4. **R17's bifurcation stands**: EITHER allocation OR GPTQ. BD-GPTQ is safer (gen gap +0.65pp vs +6.36pp).
5. **Post-rotation allocation is mostly uniform**: Hadamard rotation equalizes tile sensitivities, so DP allocation often produces uniform K. Only L55_down (outlier-heavy) has significant non-uniformity (10-16 tiles differ). The allocation headroom post-rotation is small.
6. **Inter-layer DP + GPTQ shows stale-order mismatch** (not intrinsic harm): pre-rotation DP + post-rotation GPTQ is mismatched. R27's rotate-then-recompute-allocate approach should fix this.

## Statistical caveats

- 80 paired cells from 5 splits × 4 tensors × 4 K. K=3 is all ties (uniform at min K). Active allocation rates are K=4, K=5, K=6.
- No equivalence CI computed.
- Re-allocation uses post-hoc correction factors (DP-only, no local search refinement). Correction-inside-each-K R-D refit remains untested (Wave 5).
- 128×128 slices only; full-tensor scale may differ.

## Remaining limitations (for Wave 5)

1. **Correction-inside-RD**: proper re-allocation would recompute per-tile D(k) for each k after GPTQ, not use post-hoc scaling. Untested.
2. **Full-tensor scale**: 128×128 slices may not capture allocation headroom in full 5120×17408 tensors.
3. **Real activations**: synthetic independent-channel calibration is worst-case. Correlated activations may change interaction.
4. **KLD authorization**: local HWE is a proxy. QSRT lesson warns local metrics can invert end-to-end KLD.
5. **Inter-layer DP on post-rotation distortions**: R27 showed rotate-then-recompute-allocate works. Need to test with BD-GPTQ.
6. **Heterogeneous action menu** (Wave 5): per R28/Main, the future stack should select per-tensor actions from a menu (K, Hadamard, GPTQ, allocation), not a one-size-fits-all stack.
