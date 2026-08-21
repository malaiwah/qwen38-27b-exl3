# R10-CoupledBlocks: Dead Ends

## 1. MLP Coupled Permutation with Per-Column Quantization

**What was tried:** Applied the coupled permutation with per-column uniform quantization.

**Result:** 0.0% improvement (identical error, 0 channels displaced).

**Why it fails:** Per-column quantization quantizes each column independently. Permuting columns just reorders which columns get which scales — the set of quantized values is identical. The permutation is mathematically invisible to the quantizer.

**Implication:** Coupled permutation for MLP is ONLY useful with quantizers that share information across columns (per-tile, shared-codebook, trellis). Confirmed by R4-Permutations finding that tile structure is essential.

## 2. In-Sample Permutation Overfitting

**What was tried:** Searched permutation on calibration X, reported block error on the same X.

**Result:** 30.6% in-sample improvement, but only 4.7% on held-out (n=20 fresh Gaussian X draws). First reviewer caught this: fresh-X mean was 4.253%, range [-8.270%, 11.877%].

**Why it happens:** The permutation search evaluates ~24,384 pair-swap candidates on the calibration X. With 128 channels, there's enough freedom to find a permutation that happens to reduce error on this specific X but doesn't generalize.

**Fix applied:** Search on calibration X, evaluate ONLY on held-out validation X (20 fresh draws). Report mean ± std. The held-out improvement is consistently positive (range [49.2, 58.3] vs independent mean 55.5), so the signal is real but modest.

**Implication:** For production use, need real calibration activations (not synthetic Gaussian) and multiple calibration sets. The permutation overfits to the activation distribution.

## 3. Q/K Rotation Under RoPE

**What was tried:** Claimed "any orthogonal R on Q/K head dim is legal."

**Why it fails:** Real Qwen3.8-27B attention applies RoPE after Q/K projections. The score becomes q_i R T_i T_j^T R^T k_j^T (where T_i are position-dependent rotations). Equality for all positions requires R to commute with every relative RoPE transform. A random/Hadamard/Givens O(d) matrix does not satisfy this.

**Fix:** Acknowledged in findings. V/O rotation is the free invariant (no RoPE interaction). Q/K rotation would require either (a) runtime inverse rotation before RoPE, (b) conjugated/dense RoPE implementation, or (c) a much smaller RoPE-commuting search space (block-diagonal rotations within RoPE rotation pairs).

**Implication:** The practical attention coupled optimization is V/O rotation only, not Q/K. The 22.5% held-out improvement from V/O alone is the deployable result.

## 4. Block Hessian Misnaming

**What was tried:** Called the 4×4/3×3 scalar matrix a "block Hessian."

**Why it's wrong:** The actual block Hessian is J_A^T J_B, a p_A × p_B matrix. The code computes S[A,B] = sum(J_A * J_B) = trace(J_A J_B^T), which is a scalar summary. This summary can change under reordering of parameter vectors and may exhibit sign cancellation.

**Fix:** Renamed to "Jacobian trace sensitivity summary." Added error-direction cross terms C[A,B] = e_A^T J_A^T J_B e_B using actual quantization errors, which measures real cross-coupling. The diagonal fraction is 89% (attention) and 99% (MLP), confirming cross-coupling is real but modest.

## 5. Budget Underfill

**What was tried:** Used non-representable average bit budgets (3.5, 4.5) for MLP with equal-size matrices.

**Why it fails:** With 3 equal-size matrices (each 8192 elements), a budget of 3.5×24576 = 86016 is not divisible by 8192. The greedy allocator used only 81920 bits (3.333 avg), not the advertised 86016.

**Fix:** Use representable averages (3.333 = (3+3+4)/3, 4.333 = (4+4+5)/3). Assert `used == budget` for every arm.

## 6. Cherry-Picked Equal Baseline

**What was tried:** Compared joint allocation against the FIRST balanced allocation in dictionary order.

**Why it's wrong:** The first allocation happens to be the WORST balanced allocation, inflating the improvement. The best balanced allocation matches the joint allocation (both are globally optimal).

**Fix:** Enumerate ALL balanced allocations (84 for attention at 3.5 avg, 206 at 4.5 avg). Report improvement vs best, mean, and worst. The improvement vs best is 0% (they agree), vs mean is 78-89%.

## 7. Analytic Formula with Full W_down Projection

**What was tried:** Computed S_gate_gate as sum_n ||X[n,:]||^2 * ||alpha @ W_down[n,:]||^2, projecting through W_down.

**Why it's wrong:** The Frobenius inner product <J_gate, J_gate>_F sums [X[n,a] * alpha[n,b] * W_down[b,m]]^2 over all (n,m,a,b). This factors using per-ROW norms of W_down (diagonal of W_down@W_down^T), not the full projection which includes off-diagonal cross-terms between different intermediate dimensions.

**Fix:** Use W_down_row_sq = sum(W_down^2, axis=1) (per-row norms). Verified: all four analytic terms match numerical to ratio > 0.9999.

## 8. Rotation-GPTAQ "Antagonism" — RETRACTED (Cholesky Bug)

**Initial claim:** R9-GroupOrbit found rotation and GPTAQ were antagonistic (476-5239% degradation). This was included in R10's findings.

**Why it was wrong:** The antagonism was caused by a Cholesky convention bug in the shared `inv_cholesky` function. The code returned C^{-T} (lower triangular, U U^T = H^{-1}) instead of the correct upper-triangular U with U^T U = H^{-1}. With the correct Cholesky factor, rotation+GPTAQ FLIPS to synergistic: +42-76% improvement over best single method.

**Corrected implication:** Coupled attention rotation CAN be combined with GPTAQ correction. The correct stack is: V/O rotation (free invariant) + GPTAQ (with correct Cholesky) + allocation. R9's corrected alternating optimizer achieves +40.2% mean improvement across 4 real tensors.

**Lesson:** Always verify the Cholesky convention: GPTQ needs U^T U = H^{-1} with U upper-triangular. The correct construction is `U = np.linalg.cholesky(np.linalg.inv(H + lam*I)).T`.
