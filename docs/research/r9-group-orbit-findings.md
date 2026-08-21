# R9-GroupOrbit: Unified Alternating Group-Orbit Optimizer — Findings (v4, frozen codebook + correct GPTQ)

## Summary

The 6-step alternating optimizer (equilibrate, partition, rotate, allocate, quantize, correct) beats the best individual method by **+15.2% to +55.3%** across all 4 real tensors (mean +40.8%). Rotation + GPTQ correction are **synergistic** (+30.5% to +53.8% on all 4 tensors). The alternating framework's accept-if-improve discovers multi-step interactions where previously-rejected steps become useful after correction changes the landscape.

## Bug Fixes Applied (v1→v4)

1. **Cholesky convention** (v1→v2): `cholesky(inv(H+damp)).T` giving U^T U = H^{-1} (upper triangular). Reversed antagonism to synergy.
2. **Frozen codebooks** (v2→v3): One codebook per 16×16 tile, matching RTN/Rotate. 
3. **Column-sequential with frozen codebook** (v3→v4): Codebook (lo, step) frozen before column-tile processing, then applied to CURRENT w_pre each column. This enables true GPTQ feedback within tiles.
4. **Alpha parameter**: Functional. Both alpha=0 (GPTQ) and alpha=0.25 (GPTAQ) tested.
5. **Receipt path**: Fixed to repo root `receipts/research/`.

## Results

### 6-step vs Best Individual

| Tensor | Best Individual | 6-step Herr | Improvement |
|--------|----------------|------------|-------------|
| L0_gate | Rotate 8.28e-08 | 4.23e-08 | **+49.0%** |
| L0_down | Rotate 9.62e-08 | 5.44e-08 | **+43.5%** |
| L55_gate | Rotate 1.73e-07 | 7.72e-08 | **+55.3%** |
| L55_down | Rotate 4.00e-07 | 3.39e-07 | **+15.2%** |

**Mean: +40.8%**

### Rotation + GPTQ Synergy

| Tensor | Rotate | GPTQ(a=0) | Rot+GPTQ | Synergy |
|--------|--------|-----------|----------|---------|
| L0_gate | 8.28e-08 | 2.04e-07 | 4.35e-08 | +47.5% |
| L0_down | 9.62e-08 | 1.80e-07 | 6.25e-08 | +35.0% |
| L55_gate | 1.73e-07 | 3.74e-07 | 7.98e-08 | +53.8% |
| L55_down | 4.00e-07 | 6.62e-06 | 2.78e-07 | +30.5% |

**GPTQ (alpha=0) is generally better than GPTAQ (alpha=0.25) after rotation.** The P-matrix asymmetric correction provides marginal benefit (L55_gate: +55.3% vs +53.8% for GPTQ), but GPTQ error propagation alone is the main contributor to synergy.

## Multi-Step Interactions

The alternating optimizer discovers emergent pipelines:
- **L0_gate**: rotate → correct → rotate (re-rotation after correction improves)
- **L0_down**: rotate → correct → rotate (same pattern)
- **L55_down**: rotate → correct → partition (partition becomes useful after correction)

This shows the optimal pipeline isn't fixed but emerges from alternation.

## Monotonicity

All 20 configurations: **PASS** (objective non-increasing, enforced by accept-if-improve gate).
Note: Monotonicity is guaranteed by the outer acceptance gate, not by intrinsic sub-step properties. Sub-step proposals are heuristics (Osborne, partition, Givens) or surrogates (DP); only the acceptance gate ensures monotone descent.

## Known Limitations

1. **Byte budget**: Dense U/V rotation (131KB) vs K5 payload (10KB). Hadamard transforms are foldable in production (zero storage) but this POC charges full dense cost. Comparison is error-only, not matched-rate.
2. **DP surrogacy**: K-allocation DP uses additive per-tile distortions, omitting cross-tile terms.
3. **In-sample**: Same X/Xt for selection and evaluation. No held-out activations.
4. **Unequal search budget**: Alternating optimizer gets 5 outer iterations of rotation search; individual Rotate baseline gets 1. Part of the 4-step improvement over Rotate is extra search.
5. **Synthetic calibration**: Real activations needed for production conclusions.

## Code

- `tools/research/r9-group-orbit/poc.py` — Full implementation
- `receipts/research/r9-group-orbit-results.json` — Numerical results
