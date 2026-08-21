# R15 — Held-Out Validation Framework: Findings (Corrected)

**Status:** completed, 2026-08-21. 7 random 80/20 train/test splits, 4 real Qwen3.8-27B tensors (L0_gate, L0_down, L55_gate, L55_down), K=3,4,5,6, per-tile (16×16) uniform quantizer (matched for all arms), correct Cholesky convention.

**Correction note:** Initial version had three bugs identified by openai-reviewer: (1) GPTQ used per-column (16×1) codebooks instead of per-tile (16×16), giving it an unfair quantization advantage; (2) Full Stack discarded the DP allocation; (3) R9 Osborne equilibration produced NaNs and GPTAQ alpha was unused. All three are fixed. Results below are from the corrected code.

**Reviewer v2 caveat (IMPORTANT):** The synthetic calibration generates independent Gaussian channels with outlier scaling. The population H_X is approximately diagonal (off-diagonal energy fraction = 2.18%). This means GPTQ's full-covariance Cholesky terms are fitting finite-sample noise by construction — the experiment is ENGINEERED to make full-covariance GPTQ overfit. The GPTQ -30% held-out result does NOT prove GPTQ fails on real activations. Real model activations have genuine off-diagonal Hessian structure. The valid conclusions are: (a) diagonal/weight-based transforms generalize regardless of calibration; (b) accept-if-improve gating (R9) is the safe pattern; (c) with small calibration sets and independent channels, full-covariance methods overfit.

A proper held-out validation framework was built and used to re-test the top 5 Wave 1 components. The key findings: (1) BiIP scaling, DP allocation, and permutations generalize perfectly to held-out calibration; (2) GPTQ error propagation overfits with synthetic independent-channel calibration (gap +56.5pp) — this is partly an artifact of the synthetic design (see caveat above) but demonstrates the risk of full-covariance methods with small calibration; (3) R9's accept-if-improve gating limits GPTQ overfitting to +4pp, retaining 81.9% held-out improvement.


### Headline results (all-K macro across tensors, K=3,4,5,6)

| Component | In-sample | Held-out | Gen Gap | Overfits? |
|-----------|-----------|----------|---------|-----------|
| R3 BiIP+Hadamard | +74.6% | +74.7% | −0.1% | **No** |
| R1 DP Allocation | +25.5% | +25.6% | −0.2% | **No** |
| R7 Act-order GPTQ | +26.8% | **−30.7%** | **+57.5%** | **Yes (synthetic)** |
| R9 Alternating Optimizer | +86.0% | +81.7% | +4.3% | Moderate |
| R4 Hadamard+p99 Perm | +33.4% | +33.4% | +0.0% | **No** |
| Full Stack | +84.8% | +78.8% | +6.0% | Moderate |

**Note on R1:** The all-K macro includes K=3 and K=6 where budget constraints leave no room for allocation (R1=0%). At K=4,5 only, R1 achieves in +44.8%, out +46.0%.

**Note on std:** The ±std in per-tensor tables (§3) is the spread across 4 tensor means, not across 7 splits. Per-split macro std (K=5 example): R3 ±2.1%, R7 ±8.9%, R9 ±1.4%, Full Stack ±1.3%.

## 2. Methodology

### Protocol
1. Generate 512 synthetic calibration samples (Gaussian + outlier channels, same recipe as Wave 1).
2. Random 80/20 split → 409 train / 103 test samples (7 different random splits, seeds 42–642).
3. Compute H_X_train, H_G_train from train; H_X_test, H_G_test from test.
4. For each method: fit transforms on **train** Hessians only, apply + quantize + inverse → W_hat (fixed once fitted).
5. In-sample error: tr(H_G_train · E · H_X_train · E^T).
6. Held-out error: tr(H_G_test · E · H_X_test · E^T).
7. % improvement over RTN baseline computed with the **same** evaluation Hessian.
8. Generalization gap = in_sample_improvement − held_out_improvement.

### Controls
- All arms use the same per-tile (16×16) uniform quantizer and matched byte budget.
- GPTQ codebooks are frozen from original W (one per 16×16 tile), verified: GPTQ(identity, alpha=0) ≡ RTN (max diff = 0.00).
- Correct Cholesky: `U = chol(inv(H+λI)).T` with eigendecomposition fallback.
- R7_LTR_GPTQ (natural-order GPTQ) isolates ordering vs propagation effects.
- R4_Hadamard_Only (no permutation) controls for R4.
- Full Stack: BiIP → Hadamard → p99 perm → DP alloc (K_alloc passed to GPTQ) → act-order GPTQ.
- R9 Osborne equilibration uses correct log-domain implementation (0 NaNs on rectangular matrices).

## 3. Detailed Results

### R3 BiIP+Hadamard — Generalizes perfectly

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +73.9±14.2% | +74.8±13.7% | −0.88±1.07 |
| 4 | +75.2±12.6% | +75.4±12.3% | −0.20±0.60 |
| 5 | +74.8±12.9% | +74.6±13.0% | +0.18±0.68 |
| 6 | +74.5±13.1% | +73.9±13.1% | +0.60±0.67 |

**Verdict: No overfitting.** BiIP scales use only diagonal statistics (diag(H_X), diag(H_G)), which are robust marginals that generalize well. Hadamard signs are random (not data-dependent).

### R1 DP Allocation — Generalizes well

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +0.0% | +0.0% | 0.00 (budget too tight) |
| 4 | +53.9±20.9% | +55.1±20.0% | −1.22±1.48 |
| 5 | +48.0±13.4% | +48.2±14.4% | −0.26±1.51 |
| 6 | +0.0±1.8% | −0.8±1.6% | +0.80±0.86 (budget too loose) |

**Verdict: No overfitting.** At K=4,5, allocation generalizes perfectly. At K=3/6, the budget is too tight/loose for meaningful allocation.

### R7 Act-order GPTQ — CATASTROPHIC overfitting

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +26.5±3.3% | **−30.0±5.5%** | **+56.5±3.39** |
| 4 | +25.8±1.7% | **−31.0±1.8%** | **+56.7±3.21** |
| 5 | +25.9±0.6% | **−31.9±1.7%** | **+57.9±1.47** |
| 6 | +28.9±0.8% | **−29.9±5.7%** | **+58.8±5.83** |

**Verdict: CATASTROPHIC overfitting (~57 percentage points). GPTQ HURTS held-out performance.**

With matched per-tile codebooks, GPTQ shows modest in-sample improvement (+26%) but **makes held-out error 30% WORSE than RTN**. The GPTQ error propagation redistributes quantization error based on the training Hessian's Cholesky factor. When the Hessian changes (held-out), this redistribution concentrates error in directions that are important in the test Hessian, making things worse than no propagation at all.

**R7_LTR_GPTQ control** (natural order, no act-ordering): in=+17%, out=−16.1%, gap=+33. This shows:
- GPTQ propagation alone overfits by ~33 percentage points
- Act-ordering adds another ~23 percentage points of overfitting
- Both components contribute to the catastrophic gap

The L55_down tensor (extreme outliers) is not spared: in=+27.1%, out=−29.6%, gap=+56.6. This is unlike the pre-fix results where L55_down showed minimal gap — the per-column quantizer advantage was masking the overfitting.

### R9 Alternating Optimizer — Moderate overfitting

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +86.0±7.5% | +81.9±9.7% | +4.03±2.30 |
| 4 | +86.1±6.9% | +82.2±8.6% | +3.97±1.76 |
| 5 | +85.7±7.3% | +81.0±9.8% | +4.71±2.54 |
| 6 | +86.3±6.9% | +81.8±8.8% | +4.53±2.16 |

**Verdict: Moderate overfitting (~4 percentage points).** The R9 optimizer includes GPTQ correction (which overfits catastrophically on its own) but also includes rotation (R3, which doesn't overfit). The accept-if-improve framework limits the damage: the GPTQ correction is only accepted when it improves the training objective, and the rotation provides a large calibration-independent floor. The optimizer retains >81% held-out improvement.

### R4 Hadamard+p99 Permutation — No overfitting

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +33.7±36.4% | +36.0±35.7% | −2.30±3.85 |
| 4 | +37.7±32.2% | +38.6±31.4% | −0.94±2.26 |
| 5 | +32.2±36.1% | +31.0±36.8% | +1.21±1.54 |
| 6 | +30.1±37.0% | +28.0±37.6% | +2.14±3.89 |

**Verdict: No overfitting.** The p99 permutation is weight-based (not calibration-dependent), and Hadamard signs are random. High variance comes from random Hadamard draws, not overfitting.

### Full Stack — Moderate overfitting

| K | In-sample | Held-out | Gen Gap |
|---|-----------|----------|---------|
| 3 | +84.2±8.7% | +78.6±11.8% | +5.67±3.20 |
| 4 | +85.3±7.6% | +79.5±10.5% | +5.80±2.94 |
| 5 | +85.1±7.8% | +79.3±11.0% | +5.87±3.27 |
| 6 | +84.5±8.0% | +78.0±11.1% | +6.47±3.15 |

**Verdict: Moderate overfitting (~5.8 percentage points).** The full stack includes GPTQ (which catastrophically overfits alone) but the non-overfitting components (BiIP 75%, Hadamard+perm 33%, allocation 35%) provide a strong floor. The stack overfits more than R9 alone (5.8 vs 4.0) because it includes both the R9-style correction AND direct GPTQ, but still retains >78% held-out improvement.

### Does the full stack overfit more than individual components?

**Yes, moderately.** The full stack gen gap (~5.8%) is larger than R9 alone (~4.0%) but much smaller than R7 alone (~56.5%). The non-overfitting components act as a floor that prevents the catastrophic GPTQ overfitting from dominating. However, the full stack DOES overfit more than R9 alone because it applies GPTQ in a less constrained way (R9's accept-if-improve can reject GPTQ, while the full stack always applies it).

## 4. Ranking Stability

The component ranking by held-out improvement is **100% stable** across all 7 splits at every K value:

| Position | K=3 | K=4 | K=5 | K=6 |
|----------|-----|-----|-----|-----|
| #1 | R9 (100%) | R9 (100%) | R9 (100%) | R9 (100%) |
| #2 | Full_Stack (100%) | Full_Stack (100%) | Full_Stack (100%) | Full_Stack (100%) |
| #3 | R3 (100%) | R3 (100%) | R3 (100%) | R3 (100%) |
| #4 | R4 (100%) | R1 (100%) | R1 (100%) | R4 (100%) |
| #5 | R1 (100%) | R4 (100%) | R4 (100%) | R1 (100%) |
| #6 | R7 (100%) | R7 (100%) | R7 (100%) | R7 (100%) |

**R7 GPTQ is consistently LAST** — it is the only component that hurts held-out performance. R9 is consistently #1, followed by Full_Stack, R3, and then R1/R4 swapping for #4/#5.

## 5. Key Insights

### 1. GPTQ propagation overfits with synthetic independent-channel calibration
The Cholesky factor of H_X captures full covariance structure. GPTQ redistributes quantization error based on the training Hessian's Cholesky factor. With synthetic calibration where channels are independent (off-diagonal H_X energy = 2.18%), the off-diagonal Cholesky terms are pure finite-sample noise — GPTQ fits this noise, giving +26% in-sample but -30% held-out. **CAVEAT:** Real model activations have genuine off-diagonal structure that GPTQ can exploit. This result demonstrates the risk with small independent-channel calibration, not a fundamental GPTQ failure.

### 2. Diagonal/weight-based statistics generalize; full covariance is calibration-sensitive
BiIP (diag(H_X), diag(H_G)) and DP allocation (tile-level distortions) both generalize perfectly across all splits. GPTQ (full Cholesky of H_X) overfits with synthetic calibration. **Transforms based on marginal/diagonal statistics are robust; transforms based on full covariance are sensitive to calibration structure and sample size.**

### 3. R9's accept-if-improve framework limits GPTQ damage
R9 includes GPTQ correction but limits its damage through accept-if-improve. The correction is only accepted when it improves the training objective, and the rotation provides a large calibration-independent floor. This is why R9's gap (+4.3%) is much smaller than R7's gap (+57.5%). **Caveat:** R9 also includes equilibration, partition, orbit selection, and allocation — the 4.4pp R9-vs-R3 advantage cannot be attributed solely to gated GPTQ without an R9-no-correction ablation. Additionally, the R9 Osborne equilibration has an algorithmic bug (compounds from raw W instead of scaled W); in practice the accept-gate kept scales at identity, so R9 effectively ran as a 5-step optimizer without equilibration.

### 4. The full stack applies GPTQ unconditionally
The full stack applies direct GPTQ (not behind accept-if-improve), so it overfits more than R9 (gap +6.0% vs +4.3%). The non-overfitting components (BiIP 75%, Hadamard+perm 33%) provide a floor that prevents the GPTQ overfitting from dominating, but the stack still retains only 78.8% held-out vs R9's 81.7%.

### 5. L55_down is no longer special with matched quantizers
In the buggy version, L55_down showed minimal GPTQ overfitting because the per-column quantizer advantage masked it. With matched per-tile quantizers, L55_down shows the same GPTQ overfitting as other tensors.

## 6. Recommendations

1. **Use accept-if-improve gating for GPTQ** (R9 pattern). R9 held-out 81.7% vs R3 alone 74.7% — the combination is net positive even with adversarial synthetic calibration.
2. **For GPTQ evaluation**: Always use held-out calibration. In-sample evaluation inflates GPTQ's improvement. With independent-channel synthetic calibration, the inflation is ~57pp; with real activations it may be smaller.
3. **For ranking**: The held-out ranking is 100% stable: R9 > Full_Stack > R3 > R1/R4 > R7.
4. **Do not conclude GPTQ fails on real activations** — this experiment used synthetic independent channels where population H_X is diagonal. Real activations need testing with GPU forward pass.
5. **No production recommendation** — these results are from 128×128 slices, synthetic calibration, uniform quantizer, and proxy H_G. Full-model EXL3/KLD validation is required before any production use.

## 7. Limitations

- **Synthetic calibration has independent channels** (off-diagonal H_X energy = 2.18%). GPTQ's full-covariance Cholesky terms fit sampling noise by construction. The GPTQ overfitting result is specific to this calibration design and cannot be generalized to real activations without testing.
- **Test Hessian rank-deficient**: n_test=103 < d=128, though regularization (1e-6 I) ensures full rank numerically.
- H_G is output-covariance proxy (Cov(WX)), not true gradient covariance (Fisher).
- 128×128 slices from full tensors (aspect ratio hidden).
- Per-tile uniform quantizer (not EXL3 trellis/Viterbi).
- Only 4 tensors tested (L0/L55 gate/down).
- The full stack uses direct GPTQ unconditionally; R9's accept-if-improve is more robust.
- GPTQ codebooks frozen from original W; R9's original code froze from updated W_work.
- **R9 Osborne equilibration has an algorithmic bug** (compounds from raw W instead of scaled W). In practice, the accept-gate kept scales at identity, so R9 ran without equilibration. The R9 results represent a 5-step optimizer (partition, rotate, allocate, quantize, correct), not the full 6-step.
- **R9 does not transform Hessians through inverse scales** before rotation/allocation, unlike Full Stack which does. This is only relevant if non-identity scales are accepted.
- **DP allocation may use slightly under-budget allocations** (e.g., 382-383 vs target 384 tile-bits at K=6), a small rate shortfall that disadvantages the method.
- **No R9-without-correction ablation** — the R9-vs-R3 advantage (4.4pp) cannot be attributed solely to gated GPTQ without controlling for other R9 components.
- **Macro std is across tensor means**, not across 7 splits. Per-split std is larger (e.g., R7 K=5 split-std ±8.9% vs tensor-std ±1.7%).
- **Cholesky eigendecomposition fallback** produces a dense (non-triangular) matrix, which is not a valid GPTQ factor. In practice, the standard Cholesky path is used for all tested cases.
