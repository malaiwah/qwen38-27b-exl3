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
**GPTAQ (alpha=0.25) outperforms GPTQ (alpha=0) after rotation on 3/4 tensors.** The P-matrix asymmetric correction IS contributing positively when composed with rotation. Only L55_down favors GPTQ (alpha=0). Both alpha values show strong synergy with rotation.

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

1. **Byte budget**: Dense U/V rotation (131KB) vs K5 payload (10KB). However, per the second review brief, **BiIP rotation may cost ZERO additional bytes in production**: EXL3 already stores FP16 `suh`/`svh` scale vectors per tensor. If BiIP merely changes their values (from RMS-derived to Hessian-balanced), incremental storage is ~0. For real Qwen dimensions (5120×17408), the sidecar is 0.004 bpw — negligible and already paid. This means the rotation step is potentially FREE in storage terms, and the allocation step chooses between K5_stock and K5_BiIP at the same rate. Our POC's byte accounting (charging dense U/V) is overly conservative.
2. **Correction inside R-D arm**: Per R17/second review, correction should be INSIDE the rate-distortion arm, not appended after allocation. The alternating optimizer handles this correctly via accept-if-improve — it tries correction and rejects if it doesn't help. Future work should build D_i(K, stock_LDLQ), D_i(K, BiIP+LDLQ), D_i(K, BiIP+truncated_LDLQ) curves and let the allocator choose among corrected states.
3. **EXL3 already implements**: Block-Hadamard(128) both dimensions, input/output channel scaling, random signs, Hessian-aware LDLQ. Our gains are vs NAIVE unrotated baseline, NOT vs stock EXL3. Novel contributions: BiIP Hessian-balanced scales, output sensitivity (H_G), K allocation, validation-gated LDLQ truncation.
4. **DP surrogacy**: K-allocation DP uses additive per-tile distortions, omitting cross-tile terms.
5. **In-sample**: Same X/Xt for selection and evaluation. No held-out activations.
6. **Unequal search budget**: Alternating optimizer gets 5 outer iterations; individual baselines get 1.
7. **Synthetic calibration**: Real activations needed for production conclusions.
8. **Per-tile mixed K deployment cost**: Per-tensor allocation should come first (compatible with EXL3). Per-tile mixed K requires variable offsets, kernel branching, reduced tensor-core regularity.
9. **QSRT MLP activation-boundary rotation**: R10 verified a legal MLP rotation pattern for SwiGLU: interleave gate/up, apply U_A^T @ Q @ U_R, and crucially INVERT U_A before SiLU (SiLU always sees original basis), then apply U_B after activation. This is exact (error 2.55e-15) and REVIVES MLP rotation — the previous "MLP rotations illegal" finding only applies to naive rotations. The alternating optimizer can include this as a new rotation arm for MLP tensors, optimizing gate+up+down jointly. Runtime cost: GEMM epilogue fusion needed.
## Code

- `tools/research/r9-group-orbit/poc.py` — Full implementation
- `receipts/research/r9-group-orbit-results.json` — Numerical results
