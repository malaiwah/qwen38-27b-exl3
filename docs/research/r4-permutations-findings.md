# R4-Permutations: Tile Packing + Channel Permutations — Findings (v3, Corrected)

## Summary

Permutations of weight matrix columns that group similar-scale channels into 16×16 tiles reduce within-tile dynamic range and improve codebook quantization. The strongest composition is **Hadamard incoherencing followed by p99-scale permutation**: +31.6% aggregate-mean MSE / +13.7% paired-median MSE, and +60.9% aggregate-mean HW / +15.3% paired-median HW at K3, beating Hadamard alone with **28/28 paired MSE wins** (HW wins: 21/28 at K3). The permutation is complementary to Hadamard — consistently beneficial after incoherencing, though not super-additive (p99 alone: +6.9% mean MSE; incremental after Hadamard: +5.6%).

**IMPORTANT CORRECTION from v2:** v2 reported that "permutation adds nothing on top of Hadamard" — this was caused by a composition-order bug (permute-then-Hadamard instead of Hadamard-then-permute). The corrected order (Hadamard first, then compute P on transformed W, then permute) shows the permutation DOES improve on Hadamard alone, with 28/28 paired MSE wins.

## Strategies Implemented (10 column strategies + 5 compositions)

| # | Strategy | Type | Description |
|---|---------|------|-------------|
| 1 | identity | baseline | No permutation |
| 2 | act_order_desc | act-order | Descending H_X diagonal (OBS saliency) |
| 3 | act_order_asc | act-order | Ascending H_X diagonal |
| 4 | scale_homogeneous | scale packing | Sort by column RMS |
| 5 | p99_scale | scale packing | Sort by p99 of |W| column values |
| 6 | correlation_weight | spectral seriation | Fiedler vector of |corr(W^T W)| |
| 7 | spectral_HX | spectral seriation | Fiedler vector of |corr(H_X)| |
| 8 | variance_based | variance packing | Sort by column variance |
| 9 | balanced_scale | balanced assignment | Round-robin deal across tiles |
| 10 | random | control | Random permutation |
| 11 | hadamard | rotation (no perm) | Block Hadamard incoherencing |
| 12 | hadamard+scale | composition | Hadamard → scale-homogeneous sort |
| 13 | hadamard+p99 | composition | Hadamard → p99-scale sort |
| 14 | two_sided_scale | two-sided | Row + column scale-homogeneous |
| 15 | two_sided_corr | two-sided | Row + column correlation-based |

## Key Results (28 runs per cell: 7 tensors × 4 slices, matched calibration)

### MSE improvement over identity (mean / median)

| Strategy | K3 mean | K3 median | K4 mean | K4 median | K5 mean | K5 median |
|----------|---------|-----------|---------|-----------|---------|-----------|
| scale_homogeneous | +17.3% | +3.3% | +12.2% | +3.3% | +11.8% | +2.9% |
| act_order_desc | +5.8% | -0.1% | +2.7% | -0.3% | +2.3% | -0.0% |
| balanced_scale | +1.2% | -0.7% | -2.8% | +0.2% | -2.0% | -0.3% |
| hadamard | +27.3% | +10.8% | +24.1% | +10.3% | +23.9% | +10.1% |
| hadamard+scale | +29.7% | +11.4% | +26.6% | +12.3% | +26.7% | +11.8% |
| **hadamard+p99** | **+31.6%** | **+13.7%** | **+28.7%** | **+14.7%** | **+28.7%** | **+14.7%** |

### Hessian-weighted error improvement (mean / median)

| Strategy | K3 mean | K3 median | K4 mean | K4 median |
|----------|---------|-----------|---------|-----------|
| hadamard | +56.3% | +6.1% | +40.8% | +7.7% |
| hadamard+scale | +59.9% | +8.5% | +50.7% | +13.6% |
| **hadamard+p99** | **+60.9%** | **+15.3%** | **+52.2%** | **+14.6%** |

### Hadamard+p99 vs Hadamard alone (paired)

| K | MSE mean | MSE median | MSE wins | HW mean | HW median |
|---|---------|-----------|----------|---------|-----------|
| 3 | +5.6% | +5.2% | 28/28 | +7.3% | +8.9% |
| 4 | +5.8% | +5.5% | 28/28 | +7.2% | +4.3% |
| 5 | +5.8% | +5.3% | 28/28 | +8.9% | +5.7% |
| 6 | +5.5% | +5.3% | 28/28 | +6.0% | +7.0% |

The permutation reliably adds +5-6% paired-median MSE on top of Hadamard, with 28/28 MSE wins across all K values. HW wins are 21/28 (K3), 17/28 (K4), 23/28 (K5), 20/28 (K6). This is robust for MSE, less consistent for HW.

Note: "aggregate-mean" is the ratio of per-strategy mean MSE to identity mean MSE. "paired-median" is the median of per-block (1 - MSE_arm/MSE_identity) × 100. These are different estimands.

### Within-tile range reduction

| Strategy | Range reduction |
|----------|----------------|
| scale_homogeneous | 3.5% |
| hadamard | 7.6% |
| hadamard+scale | 9.1% |
| hadamard+p99 | 10.4% |

## Key Findings

### 1. Hadamard + p99 permutation is complementary (corrected)
With the correct composition order (Hadamard first, then compute P on transformed W, then permute), the p99-scale permutation adds +5.6% paired-median MSE on top of Hadamard alone, with 28/28 paired MSE wins. However, the interaction is not super-additive: p99 alone gives +6.9% mean MSE, while its incremental benefit after Hadamard is +5.6% — the additive interaction is negative in 22/28 runs (median -2.1 pp). The permutation is consistently beneficial after Hadamard but not synergistic in the formal sense.

### 2. Scale-homogeneous packing is the strongest pure column permutation
Sorting columns by RMS groups similar-scale channels into tiles. Aggregate-mean: +17.3% MSE, paired-median: +3.3% — the mean is outlier-driven. The benefit is real but modest for most blocks. Note: two_sided_scale (row+column) achieves +24.8% aggregate-mean / +8.4% paired-median, but requires both row and column permutation.

### 3. Act-order saliency grouping is outlier-driven
Act-order (descending Hessian diagonal) gives +5.8% MSE mean but -0.1% median at K3, with only 13/28 wins. The mean improvement is driven by a few high-error slices, not a general effect. Direction (desc vs asc) is identical for tile quantization, but saliency-based grouping itself provides only outlier-driven benefit.

### 4. Balanced scale assignment is neutral-to-harmful
Round-robin dealing of sorted channels across tiles gives +1.2% mean / -0.7% median at K3. Scale homogeneity within tiles is the goal, not scale balance.

### 5. Composition order matters
Permute-then-Hadamard (the buggy v2 order) gives no improvement over Hadamard alone. Hadamard-then-permute (correct order) gives +5.6% improvement. The permutation must be computed on the Hadamard-transformed weights to exploit the residual scale variation after incoherencing.

## Mathematical Constraints (Verified Numerically, Machine Epsilon)

### MLP-Safe Permutation

A permutation P on the intermediate dimension (d_inter) is safe for MLP if:
- W_gate and W_up rows permuted by P: W'_gate = W_gate[P, :], W'_up = W_up[P, :]
- W_down columns permuted by P (SAME direction): W'_down = W_down[:, P]

Proof: SiLU is elementwise (SiLU(P·g) = P·SiLU(g)), elementwise product commutes with permutation (P·(a*b) = (P·a)*(P·b)), and W_down[:,P]·h[P,:] = W_down·h.

Verified: output error < 3.25e-19. Rotations do NOT commute with SiLU (error 0.015).

### Attention-Invariant Permutation (GQA)

Per-KV-head permutation P on head_dim coordinates preserves attention:
- All Q heads sharing KV head kv_h use P_kv[kv_h]
- K and V for KV head kv_h use P_kv[kv_h]
- O columns for each Q head h use P_kv[h // q_per_kv]

Verified on 8-head, 4-KV-head GQA: scores error < 2.6e-18, output error < 3.5e-18.

**RoPE caveat:** Per-head head_dim permutations are safe IFF the RoPE operator is conjugated as R'_t = P @ R_t @ P^T for all positions t, or equivalently the frequency table and coordinate pairs are permuted consistently. Merely preserving 2-D pairs is necessary but not sufficient if a pair moves to a different frequency. This test does NOT verify RoPE — only basic QK^T contraction and V/O inverse-pair invariance.

**GDN:** The RoPE obstruction is absent for GDN (no rotary embeddings), but GDN recurrence and all channelwise parameters must still be proved permutation-equivariant before claiming unconditional safety. This test does NOT verify GDN recurrence.

**Cross-head:** Whole-head swaps can be safe IF KV groups and O blocks are coordinated. Arbitrary cross-head mixing that breaks Q→KV head mapping is NOT safe.

## Limitations

1. **Per-tile uniform quantizer**, not EXL3 trellis/Viterbi. Results may differ with actual trellis coding.
2. **Independent per-slice permutations**, not architecturally coupled. A real deployment needs one coordinated P per legal coupled group (gate+up+down; Q/K/V/O).
3. **Synthetic calibration**, not real model activations. H_G is output covariance Y Y^T/N, not observed loss Hessian/Fisher.
4. **128×128 slices** from full tensors. Full-size tensors may have different channel correlation structure.
5. **Permutation storage cost** (n integers per matrix) not included in byte budget.
6. **RoPE and GDN recurrence** not tested in constraint verification.

## Implications for Other Researchers

- **R3-Rotations:** Hadamard + permutation is complementary (corrected from v2). The full pipeline is: Hadamard incoherencing → p99-scale packing → tile quantization. The permutation exploits residual scale variation after incoherencing (28/28 MSE wins, but not super-additive).
- **R1-RateDistortion:** Scale-sort permutation before DP allocation should compose — permutation reduces within-tile variance, making allocator more effective.
- **R9-GroupOrbit:** Permutations compose with GPTAQ corrections (unlike rotations which had the Cholesky bug — now fixed). Permutations don't make error i.i.d., so GPTAQ's structured error shaping still applies. The alternating optimizer can include permutation as a discrete step.
- **R10-CoupledBlocks:** Per-tile permutation gives modest held-out improvement (+4.7%), consistent with our median +3.3% for pure scale-packing.
