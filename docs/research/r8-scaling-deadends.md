# R8-Scaling: Dead Ends (v3, final corrected)

## Documented dead ends

### 1. Variance scaling — HARMFUL at all correction levels
s_j = Var(X_j)^α. Result: -161.1% vs none+gptaq at K5. Variance dominated by outliers.

### 2. Outlier-aware scaling — WORST at all correction levels
AWQ base with top-5% boosted 4×. Result: -533.9% vs none+gptaq at K5. Amplifying outliers is backwards.

### 3. Per-channel adaptive α — HARMFUL
α_j = σ(log(|X_j|/|W_j|)). Result: -113.5% vs none+gptaq at K5. Non-uniform α creates extreme scale ratios.

### 4. Weight-activation product hybrid — DEAD (retracted from v2)
s_j = (mean|X_j| · max|W_{:,j}|)^α. v2 reported +6.5% but that was with buggy Cholesky. v3 corrected: **-0.4%** at K5. The product misses the activation-to-weight ratio that makes SmoothQuant effective. NOT a viable contribution.

### 5. Kurtosis scaling — slightly HARMFUL
s_j = kurt(X_j)^α. Result: -4.4% vs none+gptaq at K5. Kurtosis is tail-heaviness, not magnitude.

### 6. AWQ normalization — IRRELEVANT
AWQ with/without geometric normalization produce identical KLD. Normalization is no-op for well-conditioned data.

### 7. Per-tile scaling — MARGINAL
+0.1% vs none+gptaq at K5. Cheapest sidecar (128 bytes) but negligible benefit.

### 8. AWQ = Lp-norm p=1 — CONFIRMED EQUIVALENT
mean|X_j| = ||X_j||_1 / k. Constant k^α cancels in geometric normalization.

### 9. v1 "GPTAQ subsumes scaling" — INVALID (tautology)
v1 used per-column quantizer that commutes with per-channel scaling: Q(s_j·w_j)/s_j = Q(w_j). v2/v3 with per-tile quantizer: spread 270-430%, GPTAQ does NOT subsume scaling.

### 10. Hessian scaling was wrongly dismissed in v1
v1 had shape bug (X.T@X → 512×512 instead of X@X.T → 128×128). v3 corrected: Hessian scaling is one of the best (+2.6% vs none+gptaq at K5).

### 11. Cholesky convention bugs (v1, v2)
- v1: inv_cholesky returned lower-triangular (solve(R.T, I) from lower chol), future-error propagation was zero
- v2: U@U^T = H^{-1} (wrong for GPTQ row-suffix), needed U^T@U = H^{-1}
- v3: U = chol(inv(H+λI)).T, verified U^T@U = H^{-1}, matches GPTQv2Ref reference
- Lesson: verify Cholesky convention with assert U^T@U = inv(H) and check triu(U,1) has nonzeros
