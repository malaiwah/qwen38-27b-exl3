# R20 — Information-Theoretic Bounds and Distributional Quantization (v3)

**Status:** completed, 2026-08-21. v3 reviewer-corrected (two review rounds).
PoC verified, results reproducible (seeded).

## Executive summary

We computed the theoretical limits of quantization for real Qwen3.8-27B weight
distributions and measured the gap between achieved and optimal performance.

**Key findings:**

1. **Uniform per-tile quantization is 11–16% above the Panter-Dite high-rate
   asymptotic reference** (π√3/2 · σ²/2^{2K}) at K=5 for non-heavy-tailed
   tensors. This is an ASYMPTOTIC reference, not a finite-K bound. Lloyd-Max
   goes below it (in-sample advantage at finite K).

2. **The gap to the Gaussian R-D reference** (σ²/2^{2K}) is 200–430%. This is
   structural: per-tile uniform quantization uses tile range (≈6σ), giving
   D ≈ 3σ²/2^{2K}, while Gaussian R-D uses σ²/2^{2K} (requires vector
   quantization + entropy coding to achieve).

3. **Lloyd-Max improves over uniform by 46–48%** (all 192 tiles, seeded
   multistart with k-means++ initialization). R16 showed codebook storage
   overhead makes this non-deployable at matched byte budget.

4. **Hadamard rotation drives kurtosis to ~0** (marginal Gaussianization).
   BiIP adds variance scaling (15–45×) but not further Gaussianization. Both
   effects confirmed using same U,V pair (paired comparison).

5. **Entropy coding of indices saves 13–24% of rate** (ideal H(Q|tile),
   not measured coded length — actual savings depend on coder overhead).

6. **Δ²/12 high-resolution approximation is accurate per-tile at K≥3**
   (pooled 192 tiles, mean 0.98–1.01, p5–p95 within 0.88–1.10).

7. **GPTQ (α=1.0) improves HWE by 9–11%** over RTN. Full stack achieves
   75–95% HWE improvement.

8. **L55_down is the outlier**: 430% gap (vs 200%), kurtosis 60.3, largest
   rotation response (93.5% HWE improvement vs 65–69% for others).

**Caveats (from reviewer):**
- KDE-based differential entropy has finite-sample bias (entropy power N(X)
  exceeds Var(X) in some slices). SLB values are ESTIMATES, not rigorous bounds.
- Lloyd-Max at K=7,8 memorizes 256 points with 256 levels — in-sample
  advantage, not source-level R-D performance. Codebook side information is
  not charged in the rate.
- The Panter-Dite scalar formula is a high-rate asymptotic reference, not a
  finite-K bound.
- Negentropy estimates are biased (negative values from KDE bias); kurtosis
  is the reliable Gaussianization metric.

## 1. Shannon lower bound (Axis 1)

h(W) ≈ −4.2 to −4.6 bits (LOO KDE, in bits). Entropy power N(X) = 2^{2h}/(2πe)
exceeds Var(X) in some slices due to KDE positive bias at n=16384. SLB values
are estimates, not rigorous lower bounds. D_achieved ≥ D_SLB for all non-
Lloyd-Max methods (420/420 comparisons pass).

## 2. Lloyd-Max vs uniform (Axis 2, all 192 tiles, seeded)

| Tensor | K=3 mean | K=5 mean | K=3 max | K=5 max |
|--------|----------|----------|---------|---------|
| L0_gate | 46.1% | 46.8% | 88.2% | 90.7% |
| L0_down | 45.7% | 45.7% | 79.1% | 83.9% |
| L55_gate | 46.1% | 48.6% | 67.4% | 74.9% |
| L55_down | 47.9% | 47.9% | 97.3% | 98.2% |

Improvement is 46–48% (consistent across tensors). Seeded multistart (uniform +
quantile + k-means++). R16: codebook overhead makes non-uniform lose to K+1.

## 3. Empirical mutual information H(Q|tile) (Axis 3+8)

| Tensor | K=3 H(Q) | K=3 util | K=5 H(Q) | K=5 util |
|--------|----------|----------|----------|----------|
| L0_gate | 2.32 | 77% | 4.35 | 87% |
| L0_down | 2.34 | 78% | 4.36 | 87% |
| L55_gate | 2.32 | 77% | 4.34 | 87% |
| L55_down | 2.28 | 76% | 4.28 | 86% |

13–24% rate redundancy. This is IDEAL entropy (H(Q|tile)), not measured coded
length. Actual Huffman/arithmetic coding has finite-block overhead.

## 4. Distribution modeling (Axis 4, MLE)

| Tensor | Gaussian | GGD | GGD β |
|--------|----------|-----|-------|
| L0_gate | 151 (79%) | 41 (21%) | 1.989 |
| L0_down | 151 (79%) | 41 (21%) | 1.991 |
| L55_gate | 149 (78%) | 43 (22%) | 1.927 |
| L55_down | 142 (74%) | 50 (26%) | 1.863 |

Weights are 74–79% Gaussian by AIC. GGD β ≈ 1.86–1.99. Laplacian never wins.

## 5. Transform coding (Axis 5, paired U,V)

| Tensor | Kurt before→Had→BiIP+Had | Var before→Had→BiIP+Had |
|--------|--------------------------|------------------------|
| L0_gate | 0.487→0.006→0.006 | 1.04e-4→1.04e-4→4.73e-3 |
| L55_down | 60.29→-0.016→0.004 | 2.93e-4→2.94e-4→4.49e-3 |

Hadamard drives kurtosis to ~0 (Gaussianization). BiIP adds 15–45× variance
(range normalization). Same U,V used for paired comparison. Negentropy estimates
are biased (KDE); kurtosis is the reliable metric.

## 6. Gap analysis (Axis 6)

Gap to Panter-Dite asymptotic reference at K=5:
| Tensor | Uniform | LM | GPTQ | Full stack |
|--------|---------|-----|------|------------|
| L0_gate | 16.2% | -42.4% | 100.2% | 92.6% |
| L0_down | 11.4% | -42.3% | 94.7% | 93.7% |
| L55_gate | 15.8% | -42.7% | 98.9% | 97.5% |
| L55_down | 94.8% | -49.5% | 230.9% | 76.3% |

Gap to Gaussian R-D reference at K=5:
| Tensor | Uniform | LM | Full stack |
|--------|---------|-----|------------|
| L0_gate | 216% | 57% | 424% |
| L0_down | 203% | 57% | 427% |
| L55_gate | 215% | 56% | 437% |
| L55_down | 430% | 37% | 380% |

Uniform is 11–16% above the asymptotic scalar reference (near-optimal for
fixed-rate scalar). GPTQ increases MSE (trades MSE for HWE). The 200–430% gap
to Gaussian R-D is structural (scalar vs vector quantization).

HWE improvement at K=5 (% vs RTN):
| Tensor | GPTQ | Hadamard | BiIP+Had | Full Stack |
|--------|------|----------|----------|------------|
| L0_gate | 9.3% | 5.6% | 65.9% | 76.9% |
| L0_down | 11.4% | 19.2% | 68.5% | 78.0% |
| L55_gate | 11.0% | -15.9% | 65.7% | 75.9% |
| L55_down | 10.2% | 76.9% | 93.5% | 95.1% |

## 7. High-resolution approximation (Axis 7, pooled 192 tiles)

| Tensor | K=3 mean | K=3 p5–p95 | K=5 mean | K=5 p5–p95 |
|--------|----------|------------|----------|------------|
| L0_gate | 0.987 | 0.88–1.08 | 0.990 | 0.89–1.09 |
| L55_down | 1.010 | 0.92–1.10 | 0.990 | 0.91–1.07 |

Δ²/12 accurate on average at K≥3 (pooled n=192). Per-tile variation ±10–12%.

## 8. How much room is left?

- **MSE**: Uniform is 11–16% above the asymptotic scalar reference at K=5 —
  near-optimal for fixed-rate scalar quantization. The 200–430% gap to Gaussian
  R-D is structural (requires vector quantization). Lloyd-Max improves 46–48%
  but isn't deployable (codebook overhead, R16).
- **HWE**: Full stack achieves 75–95% improvement. No HWE-specific theoretical
  bound exists.
- **L55_down**: Largest gap, largest rotation response. Deserves more bits
  (supports K5K6 recipe's K6 for down_proj).
- **Entropy coding**: 13–24% ideal rate savings (compression layer benefit).

## 9. Implications

- **R16**: Confirmed 46–48% Lloyd-Max gain; rotation makes uniform near-optimal
  (kurtosis→0). Non-deployable due to codebook overhead.
- **R19**: L55_down has lowest H(Q) (most compressible, intrinsically hardest).
  Supports K6 for down_proj. Channel capacity (H(Q)) is complementary to HWE
  and block propagation (R18) for allocation.
- **R1/R13**: Δ²/12 valid at K≥3 on average (BAQ formula sound). Per-tile
  variation ±10%. Allocation after rotation (R13 confirmed).
- **R3**: Hadamard drives kurtosis→0 (Gaussianization). BiIP adds variance
  scaling (range normalization), not Gaussianization. Paired comparison
  isolates the two effects.
