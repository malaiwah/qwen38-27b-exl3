# R5-Subspace: ResQ + Subspace Quantization Findings (v3 Final, All Bugs Fixed)

## Executive Summary

Correct ResQ (PCA subspace quantization, arXiv:2412.14363 Eq 3) with all reviewer bugs fixed. 7 subspace strategies tested across K=3,4,5,6 on synthetic and real Qwen3.8-27B weights.

**Key findings (honest, with matched quantizers):**
1. GPTQ within PCA subspaces provides 8-25% improvement over plain PCA quantization (verified with matched shared-tile-scale quantizer and propagation ON/OFF control). Not the 56% claimed in v2 (that was confounded by quantizer mismatch).
2. At matched byte budgets, global ResQ is dominated by uniform FP16. n²×2 projection overhead is fatal for weight-only PTQ.
3. Tile PCA has practical overhead (1.22-1.56× K+1 bytes) and achieves 2.0-2.6× lower error than K+1.
4. Activation PCA r=0.25 is actually the best standalone ResQ at K=5 (8.02e-4), beating resq_gptq r=0.125 (9.29e-4). At K=6, resq_gptq wins (2.53e-4 vs 2.89e-4 for act_pca r=0.25).
5. Activation PCA > Weight PCA > Joint PCA (7× gap).

## Bug Fixes (v1 → v3 final)

| Bug | Fix |
|-----|-----|
| inv_cholesky lower-triangular | chol(inv(H)).T, U.T@U=inv(H) |
| GPTQ on PCA-diagonalized Hessian | Apply Hadamard within subspaces before GPTQ |
| Quantizer mismatch (16×1 vs 16×16) | Shared tile scale for all 16 columns in GPTQ block |
| Sanity check confounded by quantizer | Propagation ON vs OFF with identical quantizer |
| Tile PCA 8× byte overcounting | One U_tile per input tile |
| Adaptive rank no budget enforcement | Enforces bytes ≤ target |
| FP64 for bits≥16 | FP16 ceiling |
| Adaptive rank proj_bytes bug | bytes_for_projection(n) not bytes_for_projection(n, r) |

## Results (synthetic, 3 seeds, K=5, matched quantizers)

| Method | Mean Hess Err | Bytes |
|--------|-------------|-------|
| baseline_K21_eq (FP16) | 1.27e-7 | 43008 |
| resq_act_pca_r0.25 | 8.02e-4 | 44544 |
| resq_tile_pca_r0.25 | 8.32e-4 | 15872 |
| resq_gptq_r0.125 | 9.29e-4 | 43776 |
| resq_act_pca_r0.125 | 1.02e-3 | 43776 |
| resq_baq_r0.125 | 1.05e-3 | 43776 |
| baseline_K6 | 2.36e-3 | 12288 |
| baseline_K | 1.05e-2 | 10240 |

GPTQ propagation (matched quantizer, ON vs OFF): 8.6% at K=5, 25% at K=6.

## Tile PCA — viable variant

| K | Tile PCA r=0.25 | Bytes | K+1 Err | K+1 Bytes | Ratio |
|---|----------------|-------|---------|-----------|-------|
| 5 | 8.32e-4 | 15872 | 2.36e-3 | 12288 | 1.29× |
| 6 | 2.66e-4 | 17408 | 5.97e-4 | 14336 | 1.22× |

## Joint PCA Derivation

Maximize u^T H_X u s.t. u^T H_W u = 1. Lagrangian: H_X u = λ H_W u. Generalized eigenvectors are H_W-orthogonal; QR-orthonormalize for ResQ. Weakest variant (7× worse).
