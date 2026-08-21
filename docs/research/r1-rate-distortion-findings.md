# R1-RateDistortion: Exact Trellis Rate-Distortion Allocation — Findings (v3)

## Summary

We generalize BAQ's closed-form bit allocation (arXiv:2506.05664) to the trellis-tile
level, where each 16×16 tile receives its own K value based on Hessian-weighted
sensitivity. We implement and compare 9 allocation strategies across 4 real
Qwen3.8-27B weight tensors (L0/L55 gate+down), 3 slices each, and 3 average K
values (K=4,5,6), totaling 36 configurations.

**Key finding:** DP-refined (tile-local DP + full-objective local search) is the
best strategy, winning **100% of 36 cases** with **+25.5% mean improvement**
(median +16.9%) over uniform K in Hessian-weighted error tr(H(H_G · E · H_X · E^T).
The local search refinement adds +13.6% on top of the tile-local DP by closing
the cross-tile coupling gap. Tile-BAQ (closed-form) achieves +10.5% as a cheap
approximation. Column-BAQ's mean-rounding aggregation collapses to near-uniform.

## Mathematical Derivation

### BAQ Original (per-element, from paper Eq 1-6)

$$L_{ij}(R_{ij}) \approx \frac{(\text{range}_{ij})^2}{12 \cdot [\mathbf{H}_F^{-1}]_{jj} \cdot 2^{2R_{ij}}} = c_{ij} \cdot 2^{-2R_{ij}}$$

$$R_{ij}^* = \frac{1}{2}\log_2\frac{c_{ij}}{\lambda} + \frac{R_{\text{sum}}}{MN}, \quad \lambda = \left(\prod_{ij} c_{ij}\right)^{1/MN}$$

Equal-loss principle: $c_{ij} \cdot 2^{-2R_{ij}^*} = c_{kl} \cdot 2^{-2R_{kl}^*}$.

### Tile-Level Generalization (novel)

**One-sided (OBS inverse-Hessian):**
$$c_t^{\text{1-sided}} = \frac{(\text{range}_t)^2 \cdot s}{12} \cdot \sum_{j \in \text{tile cols}} \frac{1}{[\mathbf{H}_X^{-1}]_{jj}}$$

**Two-sided (OBS inverse-Hessian):**
$$c_t^{\text{2-sided}} = \frac{(\text{range}_t)^2}{12} \cdot \sum_{(i,j) \in t} \frac{H_G[i,i]}{[\mathbf{H}_X^{-1}]_{jj}}$$

**Direct Hessian (no OBS assumption):**
$$c_t^{\text{direct}} = \frac{(\text{range}_t)^2}{12} \cdot \sum_{(i,j) \in t} H_G[i,i] \cdot H_X[j,j]$$

$$K_t^* = \frac{1}{2}\log_2\frac{c_t}{\lambda} + K_{\text{avg}}$$

### Tile-Local DP (additive surrogate)

$$\min \sum_{t,k} D_{t,k}^{\text{local}} z_{t,k} \quad \text{s.t.} \quad \sum_k z_{t,k} = 1, \quad \sum C_{t,k} z_{t,k} \leq B$$

$D_{t,k}^{\text{local}} = \text{tr}(H_G^{\text{sub}} E_t H_X^{\text{sub}} E_t^T)$. Additive surrogate — omits cross-tile terms.

### DP-Refined (full-objective local search)

After DP, try all single-bit transfers. Accept if tr(H_G E' H_X E'^T) < tr(H_G E H_X E^T).
All 36 cases converged (asserted, flag persisted in JSON, no improving one-bit transfers remain).

## Strategies

| Strategy | Description | Mean improvement | Win rate |
|----------|-------------|-----------------|----------|
| **DP-refined** | Tile-local DP + full-objective local search | **+25.5%** | **100%** |
| Tile-local DP | Additive surrogate knapsack | +14.2% | 0% |
| Tile-BAQ (1-sided) | Closed-form, OBS inverse | +10.6% | 0% |
| Tile-BAQ (2-sided) | Closed-form, OBS inverse | +10.5% | 0% |
| Tile-BAQ (direct) | Closed-form, direct Hessian | +10.4% | 0% |
| Iterative BAQ | 3 rounds refinement | +10.4% | 0% |
| BAQ + weight mag | Augmented with |w|^2 | +8.8% | 0% |
| Column-BAQ | Per-column → tile aggregation | -5.7% | 0% |

All arms use the same per-tile (16×16) uniform quantizer. Mixed-K arms use ≤ uniform
byte budget (9 bytes under from K-metadata packing: 3 bits/tile vs 1 byte for uniform).

## Key Observations

1. **DP-refined wins 100% of cases** with +25.5% mean. The local search is essential.

2. **Cross-tile coupling is significant.** +13.6% gap between tile-local DP and DP-refined.

3. **Tile-BAQ is a strong cheap approximation** (+10.5%, 41% of DP-refined's gain).

4. **OBS vs direct Hessian doesn't matter** (+10.5% vs +10.4%).

5. **Column-BAQ's mean-rounding aggregation collapses.** Averaging 16 column K values
   per tile produces near-uniform allocations (63 tiles at avg K, 1 at avg-1). The -5.7%
   mean measures the damage from this collapsing heuristic, not a fair test of column
   allocation. A global integer projection (not per-tile mean rounding) would be needed
   for a fair column-BAQ test. Column-BAQ improves over uniform in 1/36 cases (+0.9%).

6. **Late-layer weights benefit most.** L55_down: +95.9%, L55_gate: +57.1%.

7. **Composition with rotation is an open question.** Earlier rotation-allocation
   antagonism claims (R9) were based on a Cholesky bug (now corrected — rotation+GPTQ
   is synergistic). Whether rotation+allocation composes needs testing. R4's scale-
   homogeneous permutation may compose with allocation (permutations don't homogenize
   like rotations).

## Block-Size Invariance Test

**PASSED:** DP monotonicity for tile sizes 8, 16, 32 (finer ≤ coarser). Byte budgets
≤ uniform-K bytes. Uniform K improves with finer tiles (tighter ranges, expected).

## Limitations

1. Per-tile uniform quantizer, not EXL3 Viterbi.
2. Synthetic calibration.
3. 128×128 slices.
4. No GPTQ correction (isolates allocation; R2 handles correction).
5. Local search is O(T²) per iteration.
6. Column-BAQ aggregation heuristic collapses — a global knapsack over column
   distortion curves would be needed for a fair test.
7. Composition with rotation/permutation needs testing.

## Artifacts

| File | Description |
|------|-------------|
| `tools/research/r1-rate-distortion/poc.py` | PoC code (v3) |
| `receipts/research/r1-rate-distortion-results.json` | Full results |
| `docs/research/r1-rate-distortion-findings.md` | This document |
| `docs/research/r1-rate-distortion-deadends.md` | Dead ends |
