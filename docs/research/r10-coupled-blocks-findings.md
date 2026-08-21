# R10-CoupledBlocks: Coupled Attention/MLP Block-Level Quantization

## Summary

This axis explores quantization optimization at the block level, exploiting architectural invariants that per-matrix methods miss. Three coupling invariants enable joint optimization:

1. **Q/K coupled rotation** (attention): Common orthogonal R preserves QK^T. **CAVEAT: Not free under RoPE** — R must commute with the relative position transform. Highly restrictive in practice.
2. **V/O inverse-pair rotation** (attention): V'=VR, O'=R^T O preserves VO. **This is the FREE invariant** — no RoPE interaction, no position dependence.
3. **MLP coupled permutation**: Same P for gate/up, P^T for down preserves SiLU(gate)⊙up product. Only permutations are legal (rotations break SiLU commutativity).

All invariants verified to machine precision. With held-out validation (20 fresh Gaussian X draws), coupled rotation achieves **29-39% block error reduction** on attention, and MLP coupled permutation achieves **3-8% reduction** with per-tile quantization on held-out data.

**NOTE on rotation + GPTAQ composition:** An earlier finding (R9-GroupOrbit) claimed rotation and GPTAQ were antagonistic. This was caused by a Cholesky convention bug (wrong orientation: U U^T = H^{-1} instead of correct U^T U = H^{-1}). With the correct Cholesky factor, rotation + GPTAQ are SYNERGISTIC (+42-76% improvement over best single method). The correct stack is: rotation (make weights incoherent) + GPTAQ correction (with correct Cholesky) + allocation. This POC does not include GPTAQ; the composition claim is from R9's corrected results.

## Mathematical Derivations

### 1. Q/K Coupled Rotation Invariant

**Theorem:** For any orthogonal R ∈ O(d_head), applying R to both Q and K weight columns preserves the attention score matrix QK^T.

**Proof:**
```
Q' = X @ (W_Q @ R) = Q @ R
K' = X @ (W_K @ R) = K @ R
Q'K'^T = (QR)(KR)^T = Q R R^T K^T = Q K^T  ∎
```

**RoPE constraint (reviewer-flagged):** In real Qwen3.8-27B attention, RoPE is applied after Q/K projections. With row-vector convention, the score becomes q_i R T_i T_j^T R^T k_j^T (where T_i are position-dependent rotations). Equality with the original q_i T_i T_j^T k_j^T for all positions requires R to commute with every relative RoPE transform — a random/Hadamard/Givens O(d) matrix does not satisfy this. The Q/K rotation would require a runtime inverse rotation before RoPE or a much smaller RoPE-commuting search space.

### 2. V/O Inverse-Pair Rotation Invariant (FREE)

**Theorem:** For any orthogonal R ∈ O(d_head), V'=VR and O'=R^T O preserves the VO product.

**Proof:**
```
V' = X @ (W_V @ R) = V @ R
O' = R^T @ W_O
V' O' = (V R)(R^T W_O) = V (R R^T) W_O = V W_O = VO  ∎
```

This invariant has no RoPE interaction (V/O are not position-rotated). In GQA, the inverse rotation must be applied to every O slice corresponding to the same KV head. This is the **practically free** invariant for attention block optimization.

### 3. MLP Coupled Permutation Invariant

**Theorem:** For any permutation matrix P, applying P to the intermediate dimension of gate and up, and P^T to down, preserves the MLP output.

**Proof:**
```
gate' = W_gate @ P  →  g' = X @ W_gate @ P = g @ P
up' = W_up @ P      →  u' = X @ W_up @ P = u @ P
h' = SiLU(g') ⊙ u' = SiLU(g @ P) ⊙ (u @ P)
```

Since P is a permutation and SiLU is elementwise:
```
SiLU(g @ P)[n,j] = SiLU(g[n, P^{-1}(j)])  (permutes elements, applies SiLU elementwise)
h' = (SiLU(g) ⊙ u) @ P = h @ P
Y' = h' @ (P^T @ W_down) = (h @ P) @ (P^T @ W_down) = h @ W_down = Y  ∎
```

**Key distinction:** Permutations commute with SiLU (elementwise function on permuted elements = permuted elementwise function). General orthogonal rotations do NOT commute with SiLU, because R mixes coordinates before the elementwise application. Confirmed empirically: rotation gives ||Y - Y_rot||^2 = 2938.31 (massive violation).

**Quantizer interaction:** Per-column quantization makes permutation a no-op (each column quantized independently, so reordering doesn't change error). Permutation only helps with per-tile quantization (grouping similar-scale channels reduces tile range). This is confirmed: 0% improvement with per-column, 3-8% held-out with per-tile.

### 4. Jacobian Trace Sensitivity Summary

**Corrected naming (reviewer-flagged):** The 4×4/3×3 matrix is NOT a full block Hessian. It is a scalar summary S[A,B] = <J_A, J_B>_F = trace(J_A J_B^T), where J_A = d(vec(Y))/d(vec(W_A)). This summary:
- Diagonal S[A,A] = ||J_A||_F² measures total output sensitivity to matrix A's parameters
- Off-diagonal S[A,B] can change under reordering of parameter vectors
- Small off-diagonal values may result from sign cancellation, not weak coupling

For actual cross-coupling with real quantization errors, we compute error-direction cross terms: C[A,B] = e_A^T J_A^T J_B e_B where e_A = vec(W_A - Q(W_A)).

**Analytic diagonal formulas (verified against numerical finite differences, ratio > 0.9999):**

For the MLP block (g = X@W_gate, u = X@W_up, h = SiLU(g)⊙u, Y = h@W_down):

- **S_gate_gate** = Σ_n ||X[n,:]||² · Σ_b α[n,b]² · ||W_down[b,:]||²
  where α = SiLU'(g) ⊙ u (gate derivative modulated by up)
- **S_up_up** = Σ_n ||X[n,:]||² · Σ_b β[n,b]² · ||W_down[b,:]||²
  where β = SiLU(g)
- **S_down_down** = Σ_n ||h[n,:]||² · d_model
- **S_gate_up** = Σ_n ||X[n,:]||² · Σ_b α[n,b] · β[n,b] · ||W_down[b,:]||²

**Key insight:** Per-ROW norms of W_down (diagonal of W_down@W_down^T) appear, NOT the full projection. The full cross-coupling J_gate^T J_up uses off-diagonal entries of W_down@W_down^T, which matter for correlated quantization errors but cancel in the trace summary.

## Experimental Results (Held-Out Validated)

### Invariant Verification

| Invariant | Residual ||·||² | Status |
|-----------|----------------|--------|
| Q/K rotation preserves QK^T | 1.57e-25 | ✅ Verified (machine precision) |
| V/O rotation preserves VO | 9.91e-28 | ✅ Verified |
| Full attention Y preserved | 4.72e-29 | ✅ Verified |
| MLP permutation preserves h | 0.00e+00 | ✅ Verified (exact) |
| MLP permutation preserves Y | 3.68e-28 | ✅ Verified |
| MLP rotation violates Y | 2938.31 | ✅ Negative result confirmed |

### Attention: Coupled Rotation (K=4, n=20 held-out Gaussian X draws)

| Method | Search X Error | Held-out Mean ± Std | Held-out Improvement |
|--------|---------------|---------------------|---------------------|
| Independent (per-column) | 0.9407 | 0.8191 ± 0.091 | — |
| Q/K coupled rotation | 0.7625 | 0.7255 ± 0.103 | 11.4% |
| V/O coupled rotation (FREE) | 0.6854 | 0.6349 ± 0.057 | 22.5% |
| Full coupled (QK+VO) | 0.5320 | 0.5345 ± 0.066 | **34.7%** |

The rotation search minimizes weight Frobenius error (||W - Q(W@R@R^T)||_F^2 for Q and K, or V and O), NOT block output error. The rotation makes weights easier to quantize by distributing outliers, and the block error improvement follows from lower per-matrix quantization error. The search does not use calibration X for candidate selection, so the search/eval gap (43.4% search X vs 34.7% held-out) is NOT overfitting — it reflects the baseline's different sensitivity on different X draws (independent baseline: 0.9407 on search X vs 0.8191 held-out). V/O rotation alone provides 22.5% held-out improvement and is the practically free invariant (no RoPE constraint).

### Attention: Bit-Width Sweep (held-out)

| Bits | Held-out Independent | Held-out Coupled | Improvement |
|------|---------------------|-----------------|-------------|
| 3 | 3.786 | 2.457 | 35.1% |
| 4 | 0.819 | 0.504 | 38.5% |
| 5 | 0.180 | 0.127 | 29.5% |
| 6 | 0.044 | 0.029 | 34.2% |

Consistent 29-39% held-out improvement across all bit widths.

### Attention: Joint Rate Allocation

The greedy allocator finds the **globally optimal** allocation (verified by exhaustive enumeration of all balanced integer allocations). However, the best allocation also happens to be the best "balanced" allocation, so the improvement vs the best balanced baseline is 0%. The improvement vs the MEAN balanced allocation is 78-89%, showing that the choice of which matrices get more bits matters enormously — but the greedy allocator and the best balanced baseline agree.

| Budget | Joint Alloc | Best Balanced | Mean Balanced | Improvement vs Mean |
|--------|-------------|---------------|---------------|---------------------|
| 3.5 avg | Q:3,K:3,V:4,O:4 | Q:3,K:3,V:4,O:4 | (84 allocs) | 78.2% |
| 4.5 avg | Q:4,K:5,V:5,O:4 | Q:4,K:5,V:5,O:4 | (206 allocs) | 89.0% |

V and O get more bits than Q and K, consistent with V/O having higher Jacobian trace sensitivity (S_VV=3424, S_OO=3264 vs S_QQ=2034, S_KK=2109).

### Attention: Error-Direction Cross Terms

Using actual quantization errors e_A = vec(W_A - Q(W_A)):

| | Q | K | V | O |
|---|---|---|---|---|
| Q | 0.166 | 0.008 | 0.012 | 0.005 |
| K | 0.008 | 0.196 | 0.007 | -0.014 |
| V | 0.012 | 0.007 | 0.280 | 0.006 |
| O | 0.005 | -0.014 | 0.006 | 0.213 |

**Diagonal fraction: 89.2%** — most of the error energy is in per-matrix terms, but 10.8% is cross-coupling. The off-diagonal max / diagonal max ratio is 5.1%, meaning cross-coupling is real but modest. This justifies block-level optimization but suggests the primary benefit comes from making individual matrices easier to quantize (via rotation), not from cross-error cancellation.

### MLP: Coupled Permutation (held-out, per-tile quantizer)

| Quantizer | Search Error | Held-out Mean ± Std | Held-out Improvement |
|-----------|-------------|---------------------|---------------------|
| Per-column (K=4) | 42.285 | 40.169 ± 2.0 | — |
| Per-column + perm | 42.285 | — | 0.0% (no-op, confirmed) |
| Per-tile 16×16 (K=4) | 60.130 | 55.543 ± 2.5 | — |
| Per-tile + perm | 41.709 | 52.951 ± 2.4 | **4.7%** |

The in-sample improvement (30.6%) drops to 4.7% on held-out data, confirming substantial selection bias in the search (the permutation overfits to the calibration X). The held-out improvement is positive on 19/20 draws (one tiny regression of -0.009), with 18/20 permutation errors below the independent mean. The signal is real but modest. Note: the permutation range [49.2, 58.3] overlaps the independent mean (55.5), so individual-draw improvement is not guaranteed. Population std is reported, not paired CI; approximate paired 95% CI on improvement is [2.0, 3.2] (reviewer-computed).

### MLP: Bit-Width Sweep (per-tile, held-out)

| Bits | Held-out Independent | Held-out Perm | Improvement |
|------|---------------------|--------------|-------------|
| 3 | 268.110 | 247.086 | 7.8% |
| 4 | 55.543 | 53.660 | 3.4% |
| 5 | 12.602 | 11.892 | 5.6% |
| 6 | 3.119 | 2.901 | 7.0% |

Consistent 3-8% held-out improvement. The benefit is largest at low bit widths (K=3: 7.8%), where tile range reduction matters most.

### Real Weight Experiments (Qwen3.8-27B, held-out)

| Experiment | Held-out Independent | Held-out Perm | Improvement |
|-----------|---------------------|--------------|-------------|
| L0 (per-tile K=4) | 0.00524 | 0.00480 | 8.3% |
| L55 (per-tile K=4) | 0.03512 | 0.03043 | 13.4% |

Late-layer (L55) benefits more (13.4% vs 8.3%), consistent with late-layer weights having more heterogeneous channel scales. Note: real weight experiments use synthetic up_proj and synthetic calibration X (only gate and down are real weights).

### MLP: Joint Rate Allocation

Budgets are now exact (verified `used == budget`). The greedy allocator finds the global optimum (verified by exhaustive enumeration):

| Budget | Joint Alloc | Best Balanced | Improvement |
|--------|-------------|---------------|-------------|
| 3.333 avg (81920 bits) | gate:3, up:3, down:4 | gate:3, up:3, down:4 | 0.0% (agrees) |
| 4.333 avg (106496 bits) | gate:4, up:4, down:5 | gate:4, up:4, down:5 | 0.0% (agrees) |

Down gets more bits, consistent with the Jacobian trace sensitivity showing down has comparable sensitivity to gate (S_down=161234 vs S_gate=167789) but errors propagate linearly (no SiLU attenuation).

### MLP: Error-Direction Cross Terms

| | gate | up | down |
|---|---|---|---|
| gate | 13.78 | 0.10 | 0.06 |
| up | 0.10 | 11.70 | 0.06 |
| down | 0.06 | 0.06 | 15.48 |

**Diagonal fraction: 98.95%** — MLP cross-coupling is very weak (off-diag max / diag max = 0.66%). The SiLU gating creates near-separable error propagation: gate errors are attenuated by SiLU'(g), up errors are attenuated by SiLU(g), and down errors propagate linearly. The three matrices barely interact in the output error.

### Analytic vs Numerical Jacobian Trace (corrected)

| Term | Analytic | Numerical | Ratio |
|------|---------|-----------|-------|
| S_gate_gate | 167789 | 167789 | 0.999999 |
| S_up_up | 156900 | 156900 | 1.000000 |
| S_down_down | 161234 | 161234 | 1.000000 |
| S_gate_up | -30.897 | -30.898 | 0.999964 |

All four analytic formulas match numerical finite differences to 5+ decimal places.

## Key Insights for Other Researchers

### For R3-Rotations
- **V/O rotation is the free attention invariant** (no RoPE constraint). Q/K rotation requires R commuting with RoPE — highly restrictive.
- **MLP rotations are illegal** (SiLU non-commutativity, confirmed: 2938× error).
- Our search uses random orthogonal + signed Hadamard + Givens refinement. Results are "best found," not optimal.

### For R4-Permutations
- **MLP legal permutation:** Same P for gate AND up, P^T for down.
- **Per-column quantization makes permutation a no-op.** Only per-tile quantizers benefit.
- **Held-out improvement is modest (3-8%)** — the permutation overfits to calibration data but retains a real signal.

### For R9-GroupOrbit
- Rotation and GPTAQ are SYNERGISTIC (corrected R9 finding, after Cholesky fix). The correct stack is rotation + GPTAQ + allocation together.
- The alternating optimizer's accept-if-improve pattern should accept GPTAQ after rotation when the Cholesky convention is correct (U^T U = H^{-1}).
- For the coupled block case: V/O rotation (free invariant) can be combined with GPTAQ on the rotated V/O matrices for compound gains.

### For R1-RateDistortion
- Block-level joint allocation should weight by the Jacobian trace diagonal, not individual matrix Hessians.
- For attention: V/O have higher sensitivity (S_VV=3424, S_OO=3264). For MLP: gate ≈ down > up.
- Our greedy allocator finds the global optimum for small problems (verified by enumeration). R1's DP would scale better.
- **R9's corrected finding:** Allocation DP and rotation ARE complementary when Cholesky is correct (earlier "antagonism" was a Cholesky bug). The full stack (rotation + GPTAQ + allocation) achieves +40.2% mean improvement.

## Limitations

1. **Synthetic attention block:** Single-head, no GQA, no RoPE. Real GQA has K/V sharing across Q heads.
2. **Q/K rotation not free under RoPE.** Only V/O rotation is practically applicable to real Qwen3.8-27B.
3. **Synthetic up_proj and calibration X.** Real weight experiments use only real gate+down, synthetic up, synthetic X.
4. **Uniform quantizer, not EXL3 trellis.** Per-tile uniform is a proxy.
5. **Small dimensions:** d_model=64, d_head=32. Real model has d_model=5120, d_head=128.
6. **MLP permutation overfits calibration.** Held-out improvement (3-8%) is much smaller than in-sample (30.6%). Multiple calibration sets and real activations needed for production-grade evidence.
7. **Joint rate allocation equals best balanced baseline.** The greedy allocator finds the optimal allocation, but the best balanced allocation happens to be the same. The improvement is vs the MEAN/WORST balanced allocation, not the best.
8. **"Block Hessian" is a trace summary.** The scalar matrix captures Jacobian Frobenius inner products, not the full cross-coupling structure. Error-direction cross terms provide a better measure.
10. **Rotation search minimizes weight Frobenius, not block error.** The search finds R that minimizes ||W - Q(W@R@R^T)||_F^2 (per-matrix quantization error), not the block output error. This is a proxy: lower weight error generally leads to lower block error, but the relationship is not exact due to Jacobian sensitivity differences. A block-error-based search might find different (potentially better) rotations.
11. **Search is heuristic, not optimal.** Random orthogonal + signed Hadamard + Givens refinement is a reasonable heuristic but provides no optimality guarantee. Results are "best found," not optimal. Multiple search seeds are not reported.
12. **MLP held-out improvement is modest and variable.** 19/20 draws positive (one tiny regression), paired CI ~[2.0, 3.2]. The signal is real but small with synthetic Gaussian activations.

## Noise Floor

| Block | Noise floor (Ŵ=W) |
|-------|-------------------|
| Attention | 0.00e+00 |
| MLP | 0.00e+00 |

Zero noise floor (float64 arithmetic), so all reported improvements are pure quantization effects.
