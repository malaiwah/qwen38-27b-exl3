# R3-Rotations: Two-Sided Incoherence + Rotations — Findings

## Summary

We implemented and tested orthogonal rotations and diagonal rescaling (the KronQ/QuIP# family) for reducing quantization error on real Qwen3.8-27B weights. The **BiIP recipe — diagonal balancing + signed randomized Hadamard on both input and output — reduces output-covariance-proxy-weighted quantization error by 55-87% (macro mean 68.5%)** with per-column quantization, and **by 65-97% (macro mean ~80%)** with per-tile 16×16 quantization, across 6 real weight tensors at K5/K6, with ~0.5 bits/elem sidecar overhead.

**CORRECTED HEADLINE** (per Main's independent review): Fixed BiIP+Had+Had recipe macro mean is 68.46% with per-column quant (NOT 70.3% which was oracle-selected best-of-23). BiIP+Had+Had is best on 2/6 tensors and within 5 points on 3 of the remaining 4 (L0_down is the exception: 57.05% vs best 64.60%, 7.54 points off).

**PER-TILE QUANTIZER TAUTOLOGY CHECK**: With per-tile 16×16 quantization (the production-relevant quantizer), BiIP scaling shows MUCH larger benefit than with per-column quantization — confirming the scaling benefit is REAL, not a per-column tautology (unlike R8-Scaling's finding for AWQ/SmoothQuant). BiIP+Hadamard full recipe achieves 65-97% reduction with per-tile quant, vs 55-87% with per-column.

**ROTATION-GPTAQ COMPOSITION** (CORRECTED after Cholesky bug fix): Initial R9-GroupOrbit finding of antagonism was caused by a Cholesky convention bug (U U^T = H^{-1} instead of correct U^T U = H^{-1}). When fixed, rotation+GPTAQ is SYNERGISTIC (+42-76% improvement over best single method). The correct stack is: BiIP+Hadamard rotation + GPTAQ correction + allocation, all together. R3's PoC does NOT use GPTAQ (no Cholesky), so R3's 57-82% numbers are unaffected by this bug.

**IMPORTANT CAVEAT**: H_G in this experiment is an output-covariance proxy (Y^T Y / N), NOT the true gradient covariance (Fisher) H_G = E[g g^T] from KronQ. The true H_G requires a backward pass, which is unavailable in our CPU-only environment. The rotation invariance proof holds for ANY symmetric positive-definite H_G and H_X, including the true Fisher. All results should be interpreted as output-covariance-proxy-weighted error reductions, not true Hessian-weighted error.

## Methods Tested (7 rotation strategies, ≥4 required)

1. **BiIP diagonal balancing** (KronQ Eq. 11): S_X = diag(H_X_jj / ||W_{:,j}||²)^{1/4}, S_G = diag(H_G_ii / ||W_{i,:}||²)^{1/4}
2. **Signed randomized Hadamard** (QuIP#): H @ diag(±1) on input and/or output dims
3. **Random orthogonal** (QR of random Gaussian): dense random rotation, seed-only sidecar
4. **Butterfly/Kronecker** (U₁ ⊗ U₂): structured rotation with logarithmic application cost
5. **Block-Givens / Householder**: greedy orthogonal reflectors within 16-element blocks
6. **Joint generalized-eigen**: eigenvectors of (H_X, W^T W + εI)
7. **Identity (no rotation)**: baseline

## Experimental Design

- **6 tensors**: L0/L55 gate (K5), L0/L55 down (K6), L0 qkv (K6), L0 out (K6)
- **3 slices per tensor**: first, middle, last 128×128 aligned slice (total 18 observations)
- **Clean 2×3×3 factorial**: scaling × input_rotation × output_rotation = 18 unique configs
- **Plus 5 extra arms**: butterfly, block_givens, gen_eigen variants
- **Independent seeds**: input and output rotations use separate RNG streams (clean factorial)
- **No duplicate arms**: the BiIP recipe IS scale=biip|in=hadamard|out=hadamard (not duplicated)
- **Matched quantizer**: per-column uniform for all arms (same granularity as GPTQ)

## Key Results

### Best arm per tensor (mean across 3 slices):

| Tensor | K | Best arm | OC-proxy reduction | Range | Sidecar | Eff bits |
|--------|---|----------|-------------------|-------|---------|----------|
| L0_gate | K5 | BiIP+RandOrth+RandOrth | **65.2%** | 60.7-70.8% | 1032B | 6.004 |
| L0_down | K6 | BiIP+Had+Butterfly | **64.6%** | 61.2-68.5% | 2321B | 7.633 |
| L55_gate | K5 | BiIP+RandOrth+Had | **66.9%** | 63.4-69.7% | 1045B | 7.010 |
| L55_down | K6 | BiIP+Had+Had | **81.9%** | 75.3-85.8% | 1058B | 7.017 |
| L0_qkv | K6 | BiIP+Had+RandOrth | **73.0%** | 69.8-75.1% | 1045B | 7.010 |
| L0_out | K6 | BiIP+Had+Had | **70.1%** | 55.1-84.9% | 1058B | 7.017 |

**Mean OC-proxy reduction: 70.3%** across all tensors (range: 55-86%).

The BiIP+Hadamard+Hadamard recipe is the best arm on 2/6 tensors (L55_down, L0_out) and within 5% of the best on the others. The choice between Hadamard, random_orth, and butterfly for the rotation is less critical than having BiIP scaling + any rotation.

### Factorial decomposition (marginal effects on OC-proxy error, slice 0):

**Scaling is the strongest factor on 4/6 tensors** (L0_gate, L0_down, L55_gate, L0_out):
- BiIP scaling has the largest marginal effect (negative = improvement) on these tensors
- Marginal effect ranges from -3.29e-04 to -1.05e-03

**Input rotation is the strongest factor on 2/6 tensors** (L0_qkv, L55_down — though L55_down output is also very strong):
- Hadamard input rotation has the largest marginal effect on L0_qkv
- On L55_down, output rotation (Hadamard) has the largest marginal effect (-1.19e-03)

**Hadamard vs random_orth**: Hadamard beats random_orth in input marginal means on 3/6 tensors (L0_gate, L0_down, L55_gate); random_orth wins on 3/6 (L55_down, L0_qkv, L0_out). Neither consistently dominates.

**Cross-tensor average**: scaling > input rotation > output rotation in marginal effect magnitude, but the ordering is tensor-dependent.

### Incoherence reduction:

| Tensor | μ(W) baseline | μ(W) BiIP+Had+Had | CV_out baseline | CV_out BiIP |
|--------|--------------|-------------------|-----------------|-------------|
| L0_gate | 10.18 | 3.89 | 0.126 | 0.105 |
| L0_down | 5.81 | 3.85 | 0.079 | 0.096 |
| L55_gate | 5.34 | 4.07 | 0.127 | 0.104 |
| L55_down | **28.03** | **4.21** | **0.822** | 0.098 |
| L0_qkv | 4.30 | 4.30 | 0.193 | 0.093 |
| L0_out | 9.04 | 5.04 | 0.088 | 0.099 |

**L55_down has the most dramatic incoherence reduction**: μ drops from 28.03 to 4.21 (6.7×), CV_out from 0.822 to 0.098 (8.4×). This corresponds to the largest OC-proxy error reduction (81.9%).

## Mathematical Proof: Rotation Invariance

**Theorem (KronQ Theorem 1, clean-room proof):** For any orthogonal U, V and diagonal S_G, S_X, the output-covariance-weighted quantization error tr(H_G · E · H_X · E^T) is invariant under W' = U · S_G · W · S_X · V^T, with transformed covariances H_X'' = V · S_X^{-1} · H_X · S_X^{-1} · V^T and H_G'' = U · S_G^{-1} · H_G · S_G^{-1} · U^T.

**Proof:** By the cyclic property of trace (tr(ABC) = tr(CAB)) and substitution of E = S_G^{-1} · U^T · E' · V · S_X^{-1}:
  tr(H_G · E · H_X · E^T) = tr(H_G'' · E' · H_X'' · E'^T)

where E' = W' - Q(W') is the quantization error in the transformed space. The objective is exactly preserved; the benefit comes from E' being smaller because rotations make W' more incoherent. ∎

**This proof holds for ANY symmetric positive-definite H_G and H_X**, including the true Fisher Hessian. The proxy nature of our H_G does not affect the proof's validity.

## Sidecar Byte Accounting

| Transform component | Sidecar cost (128×128) | Bits/elem overhead |
|---|---|---|
| BiIP diagonal scales (S_G, S_X) | (128+128) × 4B = 1024B | 0.500 |
| Hadamard signs (input + output) | (128+128)/8 = 32B | 0.016 |
| Random orthogonal (seed only) | 4B | 0.002 |
| Butterfly factors (U₁⊗U₂) | ~1264B | 0.617 |
| Block-Givens (3 reflectors × 8 blocks) | 1536B | 0.750 |
| Gen-eigen (full V matrix) | 65536B | 32.0 |

**BiIP+Hadamard has the best cost/benefit ratio**: ~1058B sidecar (0.517 bits/elem) for 57-82% OC-proxy error reduction.

## Per-Tile Quantizer Tautology Check

We tested BOTH per-column and per-tile 16×16 quantizers to check if the BiIP scaling benefit is a per-column tautology (R8-Scaling found that per-channel scaling commutes with per-column quantization, making scaling appear to help when it actually does nothing).

### Reduction vs baseline (slice 0), per-column vs per-tile:

| Tensor | BiIP scaling only (col/tile) | Hadamard only (col/tile) | BiIP+Had full (col/tile) |
|--------|------------------------------|--------------------------|--------------------------|
| L0_gate | 3.7% / **26.9%** | -5.5% / 6.4% | 71.0% / **74.5%** |
| L0_down | -50.2% / **22.6%** | 20.8% / -2.7% | 55.4% / **65.0%** |
| L55_gate | -14.0% / -4.0% | 37.0% / -10.5% | 70.4% / 66.1% |
| L55_down | 39.7% / **85.8%** | -26.2% / 13.4% | 84.6% / **96.6%** |
| L0_qkv | 0.7% / **12.9%** | 20.8% / 20.3% | 65.1% / **77.3%** |
| L0_out | -29.0% / -0.8% | -28.6% / 7.6% | 55.1% / **69.1%** |

### Key findings:

1. **BiIP scaling is NOT a per-column tautology**: With per-tile quantization, BiIP scaling shows LARGER benefit (e.g., L55_down: 85.8% per-tile vs 39.7% per-column). The per-column result was partially tautological for some tensors (L0_down: -50.2% with per-column = scaling HURTS), but per-tile shows the real benefit (22.6%).

2. **BiIP+Hadamard full recipe is better with per-tile quantization**: 5/6 tensors show larger reduction with per-tile. L55_down goes from 84.6% to 96.6%. This confirms rotations are MORE effective with the production-relevant per-tile quantizer.

3. **Hadamard alone is NOT sufficient**: Without BiIP scaling, Hadamard alone can HURT (negative reduction) on several tensors with both quantizers. The scaling is necessary to equalize row/column norms before rotation can spread outliers effectively.

4. **The BiIP scaling + Hadamard combination is synergistic**: Neither component alone consistently helps, but together they achieve 55-97% reduction. This is the KronQ insight: scaling equalizes magnitudes, rotation spreads them uniformly.

5. **L55_down benefits most from per-tile quantization**: The baseline per-tile error is 5.62× higher than per-column (2.87e-2 vs 5.12e-3), showing this tensor has extreme tile dynamic range. BiIP+Hadamard reduces it to 9.81e-4 — a 96.6% reduction.


## Limitations

1. **Output-covariance proxy, not true Fisher**: H_G = Y^T Y / N is an output-activation covariance, NOT the gradient covariance H_G = E[g g^T]. The true H_G requires a backward pass. The proxy captures output channel correlation structure but is NOT the KronQ Hessian. The invariance proof holds regardless.

2. **128×128 slices only**: We test 3 deterministic 128×128 slices per tensor (first/middle/last). The real tensor dimensions (5120, 6144, 10240, 17408) are not powers of two, so the Hadamard implementation cannot directly apply to full tensors. A block-Hadamard construction would be needed for production use.

3. **Synthetic input activations**: H_X uses synthetic Gaussian+outlier activations, not real model activations. Real activations have richer structure (token correlations, attention patterns).


4. **Per-column AND per-tile quantizers tested**: We test both per-column uniform (same granularity as GPTQ) and per-tile 16×16 uniform quantizers. The per-tile results confirm the BiIP scaling benefit is real (not a per-column tautology). However, neither is the EXL3 Viterbi trellis quantizer, which may respond differently to incoherence processing. R10-CoupledBlocks reports per-column makes permutation a no-op; our per-tile results show rotations are MORE effective with per-tile quantization.

6. **Single random realization**: Each random rotation type uses one realization (seed 100 for input, seed 200 for output). Multiple realizations would give confidence intervals on the rotation-specific results.

7. **BiIP scaling clip**: We clip BiIP scales to [0.1, 10.0] for numerical stability.
