# R4-Permutations: Dead Ends (v3, Corrected)

## 1. Balanced Scale Assignment (interleaving high/low in tiles)

**Hypothesis:** Deliberately mixing high-scale and low-scale channels in the same tile would "balance" the dynamic range.

**Result:** With the fixed valid permutation (round-robin deal), balanced_scale gives +1.2% MSE at K3 (mean), -0.7% (median), and is harmful at K4-K6 (-2.8% mean at K4). The earlier reported +1100% MSE was from an INVALID permutation (only 64 unique values out of 128).

**Lesson:** Scale homogeneity within tiles is the goal, NOT scale balance.

## 2. Spectral Seriation on H_X Correlation

**Hypothesis:** Permuting channels by the Fiedler vector of |corr(H_X)| groups Hessian-correlated channels.

**Result:** Negative or marginal. MSE improvement only 2.4% (worse than random at some K values). Hessian correlation structure does not translate to useful tile packing for weight quantization.

**Lesson:** The Hessian tells you about sensitivity, not about scale homogeneity.

## 3. Act-Order for Tile Quantization

**Hypothesis:** GPTQ's act-order (descending Hessian diagonal) would improve tile quantization.

**Result:** Direction (desc vs asc) is identical for tile quantization. Saliency-based grouping gives +5.8% MSE mean but -0.1% median at K3, with only 13/28 wins. The mean is outlier-driven — saliency grouping does NOT generally help.

**Lesson:** Act-order direction is a no-op, and saliency grouping itself is only outlier-driven (not a general effect).

## 4. Permute-then-Hadamard (WRONG composition order)

**Hypothesis:** Applying permutation before Hadamard would be equivalent to Hadamard-then-permute.

**Result:** Permute-then-Hadamard gives NO improvement over Hadamard alone (-1.5% MSE, -9.4% HW). Hadamard-then-permute gives +5.6% MSE, +7.3% HW improvement (28/28 wins). These operations do NOT commute.

**Lesson:** Composition order matters. Hadamard must be applied FIRST to incoherence the weights, THEN the permutation must be computed on the transformed weights to exploit residual scale variation.

## 5. Two-Sided Correlation-Based Packing

**Hypothesis:** Permuting both rows and columns by correlation-based spectral seriation.

**Result:** Worst two-sided strategy. dynrange increased by 23.4%. Correlation-based packing on rows doesn't help for per-tile codebook quantization.

**Lesson:** For two-sided permutations, use scale-based packing on both dimensions.

## 6. Median Scale Packing

**Hypothesis:** Using median of |W| instead of RMS would be more robust to outliers.

**Result:** Worse than RMS (5.2% vs 17.3% MSE improvement at K3). The median is too insensitive to the scale variation that matters for quantization range.

**Lesson:** RMS or p99 captures the full distribution of scale; median only captures central tendency.

## 7. RETRACTED: "Hadamard and scale-packing are substitutes"

**v2 claim:** Hadamard incoherencing and scale-homogeneous packing address the same problem and are substitutes.

**v3 correction:** This was caused by a composition-order bug. With the correct order (Hadamard first, then p99-scale on transformed W), the permutation adds +5.6% paired-median MSE on top of Hadamard (28/28 MSE wins, 21/28 HW wins at K3). They are COMPLEMENTARY, not substitutes. Hadamard equalizes overall scale; the permutation exploits residual variation for tighter tile packing. The interaction is not super-additive (p99 alone: +6.9% mean; incremental after Hadamard: +5.6%), but the permutation is consistently beneficial.
