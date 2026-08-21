# R23-FullTensor: Block-Hadamard on Full Tensors

## Summary

**Direction of 128×128 slice conclusions replicates on full tensors for four selected MLP tensors.** Block-Hadamard (block size 16) works for non-power-of-2 dimensions (all model dims divisible by 16). BiIP + block-Hadamard gives 64-81% HWE improvement on full tensors. When the same H16 transform, restricted signs, restricted BiIP scales, and restricted Hessians are used for both arms, the pooled-slice HWE improvement matches the full-tensor result within 0.5-2.5pp for 3 of 4 tensors. L0_down remains an outlier (4-11pp gap).

**Important caveat (per external feedback and QSRT review):** The 64-81% improvement is vs an unrotated baseline. EXL3 already uses Hadamard+signs+LDLQ. The relevant question is whether optimized BiIP beats EXL3's existing incoherence processing — this experiment does NOT answer that. Additionally, local HWE can INVERT end-to-end KLD (QSRT lesson); KLD harness must authorize promotion.

## Block-Hadamard for non-power-of-2 dimensions

Tensor dimensions (5120, 17408, 10240, 6144) are not powers of 2. Solution: **block-Hadamard with block size 16**. All model dimensions are divisible by 16, so no padding needed. Apply H_16 to each 16-element block along rows and columns, equivalent to multiplying by kron(I_{n_blocks}, H_16).

Verified:
- Orthogonal: norm preserved to 1e-15
- Invertible: forward+inverse recovers original to 1e-15
- kron-equivalent: matches kron(I, H_16) to 0.00e+00

## Tensors tested
- L0_gate [17408, 5120], L55_down [5120, 17408], L0_down [5120, 17408], L55_gate [17408, 5120]
- K=4, 5, 6

## Results

### Incoherence (μ) reduction

| Tensor | μ_before | μ_after | Reduction |
|--------|---------|---------|-----------|
| L0_gate | 41.49 | 7.98 | 80.8% |
| L55_down | 68.22 | 12.99 | 81.0% |
| L0_down | 92.19 | 18.44 | 80.0% |
| L55_gate | 28.55 | 8.82 | 69.1% |

### HWE improvement (BiIP+BH vs no transform, matched H16 transform)

Slice comparison uses restricted signs, restricted BiIP scales, restricted Hessians from the full tensor, 10 random 128×128 slices, pooled HWE (aggregate before ratio).

| Tensor | K=4 full | K=4 pooled | K=5 full | K=5 pooled | K=6 full | K=6 pooled | Max gap |
|--------|---------|-----------|---------|-----------|---------|-----------|---------|
| L0_gate | 64.8% | 63.6% | 64.8% | 64.3% | 64.8% | 64.3% | 1.2pp |
| L55_down | 64.5% | 64.4% | 64.4% | 65.0% | 64.5% | 65.4% | 0.9pp |
| L0_down | 80.6% | 89.6% | 81.1% | 84.8% | 80.8% | 91.7% | 10.9pp |
| L55_gate | 65.4% | 63.0% | 65.4% | 63.3% | 65.4% | 63.9% | 2.4pp |

**3 of 4 tensors match within 2.5pp.** L0_down has a persistent 4-11pp gap. Individual slices for L0_down span 34-95%, with one slice dominating the pool (87% of pooled baseline HWE at K=4). This tensor has high row-norm CV (0.1442), suggesting heterogeneous outlier structure that 128×128 slices sample inconsistently.

### Mean tile range reduction: 26-40% across tensors.

### Lagrangian allocation

Proper Lagrangian relaxation: for each λ, assign tile to argmin_K [D_t(K) + λ·K], binary search λ to hit avg=5.0 bits, marginal repair for exact budget. Width map: 2 bits/tile = 0.0078 bpw overhead.

| Tensor | Time | K4 | K5 | K6 | Avg bits | Improvement vs K5 | Width map |
|--------|------|-----|-----|-----|----------|-------------------|-----------|
| L0_gate | 1.15s | 6,873 | 334,414 | 6,873 | 5.0000 | 0.7% | 0.0078 bpw |
| L55_down | 1.17s | 7,676 | 332,808 | 7,676 | 5.0000 | 0.9% | 0.0078 bpw |
| L0_down | 1.14s | 8,065 | 332,030 | 8,065 | 5.0000 | 2.0% | 0.0078 bpw |
| L55_gate | 1.16s | 7,156 | 333,848 | 7,156 | 5.0000 | 0.8% | 0.0078 bpw |

Lagrangian allocation is fully tractable (~1.2s per tensor). Improvement over uniform K5 is modest (0.7-2.0%) because block-Hadamard already homogenizes tile sensitivity. The width map adds 0.0078 bpw (1.25% of K5 payload), which should be included in any equal-rate comparison.

### Memory and time: 713 MB per tensor, ~6s total per tensor. Total ~23s for all 4.

### Sidecar: 92,928 bytes = 0.00104 bpw = 0.167% of K5 payload.

## Conclusions

1. **Block-Hadamard works for non-power-of-2**: Verified orthogonal, invertible, kron-equivalent.

2. **Slice direction replicates**: For 3 of 4 MLP tensors, pooled-slice HWE improvement matches full-tensor within 2.5pp. L0_down is an outlier (4-11pp gap) due to heterogeneous outlier structure. This supports using 128×128 slices as directional probes but not as precise magnitude predictors.

3. **Full-tensor incoherence reduction is strong** (69-81%) and consistent.

4. **K-independence** is approximately consistent with the high-resolution affine-quantizer scaling model (HWE ∝ (2^K-1)^-2 for min/max quantization, ratio cancels). Not evidence of precision-independent BiIP for nonlinear quantizers.

5. **Lagrangian allocation is tractable** (348K tiles, ~1.2s) with modest gains (0.7-2.0% over uniform K5 at equal payload rate; 0.0078 bpw width map overhead not included).

## Caveats

1. **Synthetic diagonal Hessians**: Not real model curvature. Local HWE can INVERT end-to-end KLD (QSRT lesson). KLD harness must authorize promotion.

2. **vs unrotated baseline**: EXL3 already uses Hadamard+signs+LDLQ. Need to compare against EXL3's existing incoherence, not naive unrotated.

3. **Block-Hadamard ≠ full Hadamard**: H16 only mixes within 16-element blocks.

4. **L0_down outlier**: 4-11pp full-vs-slice gap. High CV_out (0.1442) suggests row-dominant outliers that slices miss.

5. **Four MLP tensors only**: No attention, qkv, out, z tensors tested. No formal tolerance or CI.

6. **Rate accounting**: FP32 metadata assumed; experiment uses float64. Width map for mixed-K not included in equal-rate comparison.

7. **QSRT legal MLP rotation**: R10 verified that MLP gate/up/down can be rotated JOINTLY via activation-boundary transforms (inverse-transform before SiLU, forward after). This is a new rotation arm to test alongside BiIP, not covered here.
