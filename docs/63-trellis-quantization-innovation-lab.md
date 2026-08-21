# 63 — Trellis Quantization Innovation Lab: 28-Researcher Pre-Wave-5 Record

**Status:** **screening record / not deployment evidence**, 2026-08-21. Four waves
produced 28 researcher axes with adversarial review. Sections 1–11 preserve the
chronological discoveries and corrections; their fixed-stack recommendations are
superseded. Section 12 is the authoritative pre-Wave-5 synthesis. The final
actual-stock-EXL3 plan is in [doc 64](64-final-exl3-wave5-research-plan.md).

## 1. Executive summary

Ten research axes from doc 62 §10 were explored. The **unified alternating optimizer**
(R9) achieves **+40.8% mean improvement** over the best individual method across 4 real
Qwen3.8-27B tensors. The key architectural insight — **rotation and GPTQ correction are
synergistic, not antagonistic** — was discovered through a Cholesky convention bug that
affected the entire cleanroom codebase.

### The Cholesky bug (most important finding)

The cleanroom `inv_cholesky` returned `C^{-T}` from `C = chol(H)`, giving `U U^T = H^{-1}`.
GPTQ requires `U = chol(H^{-1}).T` with `U^T U = H^{-1}` (upper triangular). The wrong
convention made GPTQ propagation vanish or corrupt, causing:
- R9's false "rotation-GPTAQ antagonism" (actually +30-54% synergy)
- R5's zero GPTQ propagation in ResQ+GPTQ
- R8's GPTQ appearing to hurt (actually +10.1% on unscaled)
- R7's incorrect noise-shaping recurrence

**Correct fix** (verified by GPTQv2Ref against reference repo L85-87):
```python
Hd = H + lam * np.eye(n)
Hinv = np.linalg.inv(Hd)
U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
```

## 2. Verified findings by researcher

### R1-RateDistortion: Exact trellis rate-distortion allocation

**Claim (verified, 3 review rounds):** DP-refined (tile-local DP + full-objective local
search on tr(H_G E H_X E^T)) is the best tile-level bit allocation, winning 100% of 36
cases with **+25.5% mean improvement** (median +16.9%) over uniform K.

- Tile-BAQ (closed-form): +10.5% (cheap approximation, 74% of DP's gain)
- Tile-local DP (additive surrogate): +14.2% (DP-refined adds +13.6% via local search)
- Column-BAQ: harmful (aggregation heuristic collapses to near-uniform)
- Late-layer L55_down benefits most: +95.9%
- Not affected by Cholesky bug (uses np.linalg.inv, no GPTQ)
- **Composition directions:** scale-homogeneous permutation (R4) + tile-PCA (R5) may
  concentrate variance and make allocation more effective

### R2-GPTAQ: GPTAQ/ResComp corrections + adaptive strength

**Claim (reviewer-verified):** Paper-faithful α=1.0 beats reference-code α=0.25 in 34/36
aggregate, 103/108 paired comparisons. Improvement vs GPTQ: +0.2% (K3) to +10.2% (K6).
Grows with K.

- ResComp lazy-block propagation FIXED: block-invariant to 3.05e-16 (was 1.94e-2)
- P-matrix correct: P = α·triu(D·U^T, 1)·U, verified to 8.33e-17
- Cholesky already correct in R2's code
- Error-vector correction: collapses toward GPTQ (disables GPTAQ, not neutral)
- Repeated requantization hurts (+0.8-67% degradation) — NOT iterative refinement
- Adaptive α (asymmetry ratio): too conservative (α ≈ 0.005-0.04)
- Grid search picks α ≈ 0.7-1.1 (in-sample oracle, not deployable)
- **Dead ends:** error-vector correction, iterative refinement, adaptive α from asymmetry

### R3-Rotations: Two-sided incoherence + rotations

**Claim (independently reviewed, NEEDS_REVISION on headline but core confirmed):** BiIP
(two-sided diagonal balancing + signed randomized Hadamard both sides) reduces
output-covariance-proxy-weighted quantization error by **57-82%** (macro mean 68.46%
for fixed BiIP+Had+Had recipe) across 6 real tensors at K5/K6.

- Sidecar cost: ~0.517 bits/elem (BiIP scales + Hadamard signs)
- BiIP scaling is the dominant factor on 4/6 tensors
- Hadamard vs random_orth: comparable (3/6 each)
- L55_down benefits most: 81.9% reduction (incoherence μ: 28.03 → 4.21, 6.7×)
- Per-tile tautology check confirms BiIP scaling is REAL (not per-column artifact)
- Rotation invariance proof correct (cyclic trace, holds for any SPD H_G, H_X)
- **Dead ends:** gen-eigen (non-orthogonal V, 56% reconstruction error), Householder
  (not implemented as described), butterfly (comparable but 2× sidecar), output-only
  rotation without BiIP (ineffective), MLP rotations (illegal — SiLU doesn't commute)

### R4-Permutations: Tile packing + channel permutations

**Claim (reviewer-verified v3):** Hadamard + p99-scale permutation is complementary
(additive, not super-additive): +31.6% aggregate-mean MSE / +13.7% paired-median at K3.
vs Hadamard alone: +5.6% paired-median MSE, 28/28 MSE wins, 21/28 HW wins.

- Correct composition order: Hadamard FIRST, then permutation on transformed W
- Pure scale-homogeneous packing: +17.3% mean / +3.3% median (modest but real)
- Act-order: direction no-op (desc=asc identical), saliency grouping outlier-driven
  (median -0.1%, 13/28 wins — not general)
- Constraints verified: MLP-safe (error <3.25e-19), GQA per-KV-head (error <3.5e-18),
  RoPE requires operator conjugation R'_t = P R_t P^T, GDN: RoPE obstruction absent
  but recurrence and channelwise parameters not yet verified
- Per-column permutation is no-op (per-column quantizer is permutation-invariant)
- **Dead ends:** spectral seriation (negative), balanced scale assignment (neutral-to-
  harmful), permute-then-Hadamard (wrong order), median scale packing (worse than RMS)

### R5-Subspace: ResQ + subspace quantization

**Claim (v3 final, reviewer-pending):** GPTQ within PCA subspaces provides **8.6% (K5)
to 25% (K6)** improvement over plain PCA with matched quantizers. Tile PCA viable:
1.22-1.56× K+1 bytes, 2.2-3.0× lower error than K+1 (per-K range varies by aspect ratio).

- Activation PCA > Weight PCA > Joint PCA (7× gap)
- Joint PCA broken: generalized eigenvectors are H_W-orthogonal, QR orthonormalization
  destroys ordering
- Global ResQ impractical for weight-only PTQ: n²×2 projection overhead, dominated by
  uniform FP16 at matched bytes (~6300-7300×; note: bits≥16 arms converted to FP16
  but charged at requested bit budget, so comparison is qualitative not exact-match)
- PCA + Hadamard + GPTQ compose: PCA identifies subspaces, Hadamard creates non-
  diagonal Hessian for GPTQ to exploit
- Adaptive rank: non-monotone, budget-constrained (r=13-16 in practice)
- v1/v2 claims of 56-65% GPTQ improvement were confounded by quantizer mismatch
- **Dead ends:** global ResQ (overhead), joint PCA (orthonormalization), adaptive rank
  (non-monotone, picks max r)

### R6-GDN: GDN-specific quantization

**Honest negative result.** GDN-specific strategies are mathematically correct but
provide no measurable improvement with synthetic calibration.

- Gate-aware GPTQ: ±0.1% (gate sensitivity std=0.005, nearly uniform)
- Balanced realization: 260-440% worse (increases weight dynamic range)
- Recurrence-aware GPTQ: ±1% (marginal)
- QKV joint vs separate: identical (column-sequential GPTQ tautology)
- z-gate mixed-K allocation: 260-2227% worse
- Standard GPTQ: 14-17% single-step improvement over RTN
- RTN has 47% better accumulated recurrence error than GPTQ
- **Root cause:** real W_z@X pre-activations are small (|z|<<1), so SiLU'(z)≈0.5
  uniformly. Need real model activations to exploit gate sensitivity.
- Balanced realization dead end was an exponent bug (sqrt vs eigvals**0.25); the exponent
  fix reduced the bug-induced error by 96×/202× vs broken v1, but corrected balancing
  still remains 260-440% worse than standard GPTQ

### R7-NoiseShaping: Hessian noise shaping

**Claim (reviewer-verified v4):** Act-order GPTQ (descending diag(H_X) column order)
gives **+21-24% improvement** over RTN on tr(H_G E H_X E^T), winning 100% of 12 paired
comparisons per K, 186/192 (97%) head-to-head vs other orderings.

- GPTQ H^{-1} update IS the noise-shaping filter (Schur complement)
- Anti-correlation: RTN -0.01 → act-order -0.23 (diagnostic ceiling -0.26)
- Trade-off: HWE -22% but raw MSE +58% (error is REDISTRIBUTED, not eliminated)
- Reverse act-order shapes less (+3-10% HWE), confirming direction matters
- Cholesky corrected: U^T U = H^{-1}, residual 5.8e-16, matches Schur to 2.7e-15
- **Dead ends:** H_G^{1/2} transform (10-100× worse), Hessian-basis rotation (non-budget-
  matched), spectral filters at tested strength (<0.5% effect — no redundancy conclusion)

### R8-Scaling: AWQ/SmoothQuant alternatives

**Claim (v3 final, reviewer-verified):** GPTAQ does NOT subsume scaling (spread 270-430%
with matched per-tile quantizer). Best scaling+GPTAQ: lp_pinf +3.3%, AWQ/Hessian/Lp_p2
+2.6%, SmoothQuant +2.5% (K5, raw KLD mean).

- GPTQ error propagation: +10.1% on unscaled (K5)
- GPTAQ P-matrix adds ~1.2% on top of GPTQ
- Product hybrid (weight-activation product): -0.4% (NOT viable)
- Adaptive α (per-channel sigmoid): -113.5% (harmful)
- Variance/outlier scaling: harmful. Kurtosis: -4.4%.
- AWQ = Lp-norm p=1 (confirmed identical)
- Cholesky already correct in v3
- **Dead ends:** product hybrid, adaptive α, variance, outlier, kurtosis scaling

### R9-GroupOrbit: Alternating group-orbit optimizer

**Claim (v4 final, reviewer-pending):** 6-step alternating optimizer beats best
individual method on ALL 4 tensors: **+40.8% mean** (L0_gate +49.0%, L0_down +43.5%,
L55_gate +55.3%, L55_down +15.2%).

- Rotation+GPTQ synergy: +30.5-53.8% on all 4 tensors (was antagonistic due to Cholesky bug)
- GPTAQ (α=0.25) beats GPTQ (α=0) after rotation on 3/4 tensors (L0_gate, L0_down,
  L55_gate); GPTQ wins on L55_down. The P-matrix is modestly positive on 3/4 and
  negative on L55_down. Note: α=1.0 was NOT tested after rotation (only α=0 and α=0.25).
  The alternating optimizer accepted correction on all 4 tensors including L55_down.
- Multi-step interactions discovered: re-rotation after correction improves further,
  partition becomes useful after correction changes landscape
- Alternating framework's accept-if-improve handles per-tensor variation
- **Limitations:** byte budget not deployable-grade (dense U/V vs foldable Hadamard),
  in-sample evaluation, DP is surrogate

### R10-CoupledBlocks: Coupled attention/MLP block-level optimization

**Claim (reviewer-verified v2):** V/O rotation is the FREE attention invariant (22.5%
held-out block error reduction, no RoPE interaction). Full coupled attention rotation:
34.7% held-out (Q/K not free under RoPE).

- Q/K rotation: preserves QK^T (1.57e-25) but NOT free under RoPE
- V/O rotation: preserves VO (9.91e-28), FREE invariant (no RoPE interaction)
- MLP permutation: preserves SiLU(gate)⊙up exactly (0.00, 3.68e-28)
- MLP rotation: violates (2938× error) — confirmed illegal
- MLP per-tile permutation: 4.7% held-out (was 30.6% in-sample — substantial overfitting
  but real signal, 19/20 positive draws)
- Per-column permutation: no-op (per-column quantizer is permutation-invariant)
- Cross-coupling terms: diagonal fraction 89% (attention), 99% (MLP) — modest
- Joint rate allocation: finds global optimum but improvement is vs MEAN baseline (78-89%),
  not vs best balanced (0%)
- Real-weight: L0 +8.3%, L55 +13.4% (late-layer benefits more)
- **Dead end:** Q/K rotation under RoPE (R must commute with position transforms)

## 3. Dead ends (do not repeat)

| Dead end | Researcher | Why it failed |
|----------|-----------|---------------|
| Gen-eigen rotation | R3, R5 | Non-orthogonal V (B-orthonormal, not Euclidean) |
| Householder block-Givens | R3 | Not implemented as described, max unchanged |
| MLP rotation | R3, R10 | SiLU + elementwise product don't commute with coordinate mixing |
| Output-only rotation without BiIP | R3 | Ineffective without diagonal balancing |
| Global ResQ (dense projection) | R5 | n²×2 overhead, dominated by FP16 at matched bytes |
| Joint PCA | R5 | QR orthonormalization destroys generalized eigenvector ordering |
| GPTAQ α=0.25 (unrotated, R2 setup) | R2 | Paper-faithful α=1.0 wins 34/36 unrotated; but α=0.25 beats α=0 on 3/4 post-rotation tensors (R9) — context-dependent |
| Error-vector correction | R2 | Collapses toward GPTQ, disables GPTAQ benefit |
| Iterative requantization | R2 | Hurts (+0.8-67%), not refinement |
| Adaptive α from asymmetry ratio | R2 | Too conservative (α ≈ 0.005-0.04) |
| GDN gate-aware GPTQ (synthetic) | R6 | Gate sensitivity nearly uniform with synthetic calibration |
| GDN balanced realization | R6 | Increases weight dynamic range (correct transform, wrong objective) |
| H_G^{1/2} transform | R7 | Extreme eigenvalue spread (10-100× worse) |
| Product hybrid (weight-activation) | R8 | -0.4% (not viable at α=0.5) |
| Per-channel adaptive α | R8 | -113.5% (extreme scale ratios) |
| Variance/outlier scaling | R8 | Harmful (magnitude-based statistics better) |
| Q/K rotation under RoPE | R10 | R must commute with position transforms |
| Act-order for tile quant (grouping) | R4 | Outlier-driven, median -0.1%, not general |
| Spectral seriation | R4 | Negative/marginal |
| Permute-then-Hadamard | R4 | Wrong order, no improvement (Hadamard first is correct) |
| Column-BAQ (aggregation heuristic) | R1 | Aggregation heuristic collapses to near-uniform; not a fair test of column allocation itself |
| Iterative BAQ | R1 | No benefit over single-pass |
| Weight-magnitude BAQ | R1 | +8.8% vs +10.5% standard, not worth the added complexity |
| ResComp alone (without GPTAQ) | R2 | CAE alone insufficient |
| Butterfly as replacement | R3 | Comparable to Hadamard but 2× sidecar cost |
| Per-column scaling tautology | R3 | Per-column quantizer commutes with per-channel scaling |
| Two-sided correlation packing | R4 | Negative/marginal |
| Median scale packing | R4 | Worse than RMS |
| 4-bit quantized projection | R5 | Too much error in projection |
| Weight PCA standalone | R5 | 7× worse than activation PCA |
| Sigma-delta without GPTQ | R7 | Filter alone insufficient |
| Condition-number α | R7 | Too conservative |
| Kurtosis scaling | R8 | -4.4% (slightly harmful) |
| AWQ normalization | R8 | Not in paper, no benefit |
| Per-tile scaling | R8 | Near-no-op (+0.1%) |
| Per-column MLP permutation | R10 | No-op (per-column quantizer is permutation-invariant) |
| In-sample MLP permutation | R10 | 30.6% in-sample → 4.7% held-out (substantial overfitting) |
| Allocation after rotation (surrogate) | R9 | Rotation homogenizes tile sensitivity; R9's surrogate allocation rejected after rotation (may be fixture-dependent) |
## 4. The recommended quantization stack

**WARNING: This stack has NOT been tested as a unified pipeline.** Each component was
verified individually or in small combinations (primarily R9's 6-step optimizer). The
order below is a candidate search space, not a deployable recipe. A direct matched-budget,
held-out factorial experiment is required before any production use.

Correct operation order (transforms BEFORE quantization, correction DURING):

1. **Scaling** (R8): Lp-norm p=∞ (max activation magnitude) or AWQ (mean|X|^α)
2. **BiIP diagonal balancing** (R3): S_X = (diag(H_X)/diag(W^T W))^{1/4}, S_G similar
3. **Signed randomized Hadamard** (R3): both input and output dimensions
4. **p99-scale permutation** (R4): sort transformed weights by p99 column scale
5. **V/O rotation for attention** (R10): free invariant, 22.5% held-out improvement
6. **MLP-safe permutation** (R4, R10): same P for gate/up, inverse for down
7. **DP-refined tile allocation** (R1): local search on full Hessian-weighted objective
8. **Act-order GPTQ + GPTAQ correction** (R7, R2): single quantization pass, columns
   in descending diag(H_X) order, α=1.0 (paper-faithful for unrotated; α=0.25 modestly
   better post-rotation per R9, but α=1.0 untested post-rotation)
9. **Inverse/fold transforms**: undo scaling, rotation, permutation

The R9 alternating optimizer with accept-if-improve is the principled framework to select
which steps compose per-tensor. R9's tested 6-step optimizer (equilibrate, partition,
dense rotate, surrogate allocate, quantize, correct) achieved +40.8% over best individual,
but used only α=0 and α=0.25 (not α=1.0), dense rotations (not foldable Hadamard),
surrogate DP (not full-objective), and in-sample evaluation. Extending R9 to include the
exact proposals above requires a new factorial experiment.

### Expected improvement (from verified individual measurements)

| Component | Improvement | Metric | Caveat |
|-----------|------------|--------|--------|
| BiIP + Hadamard rotation | 57-82% (68.5% macro) | OC-proxy HWE | Fixed recipe |
| Rotation + GPTQ synergy | 30.5-53.8% | Over best single | All 4 tensors |
| Unified 6-step optimizer | 40.8% mean | Over best individual | All 4 tensors |
| DP-refined allocation | 25.5% | Over uniform K, HWE | 100% win rate |
| Act-order GPTQ | 21-24% | Over RTN, HWE | 97% head-to-head |
| V/O attention rotation | 22.5% | Held-out block error | Free invariant |
| GPTAQ α=1.0 vs GPTQ (unrotated) | 0.2-10.2% | Grows with K | Paper-faithful; untested post-rotation |
| GPTAQ α=0.25 vs GPTQ (post-rotation) | Modest on 3/4 | Post-rotation HWE | R9 receipt; α=1.0 untested post-rotation |
| p99 permutation (post-Hadamard) | 5.2% median | Over Hadamard alone | Complementary, 28/28 MSE wins |
| Scaling (lp_pinf) + GPTAQ | 3.3% | Over no-scaling+GPTAQ | Not subsumed |
| Tile-PCA ResQ | 2.2-3.0× | Lower error than K+1 | 1.22-1.56× bytes, aspect-ratio-dependent |

**Important:** These are individual component improvements on proxy metrics with
synthetic calibration. They do NOT simply add. The R9 alternating optimizer (+40.8%)
tested a subset of these components with unequal search budget (5 rotation searches vs 1
for baseline) and unmatched byte budget (dense U/V vs foldable Hadamard). A direct
matched-budget factorial experiment is required before claiming a deployable stack.

## 5. Key architectural insights

1. **Rotation and GPTQ are synergistic** — rotation handles magnitude/outlier structure,
   GPTQ handles Hessian-correlated residual. They address different error structures.
2. **GPTQ error propagation (α=0) is the main contributor** — the P-matrix asymmetric
   correction (α>0) adds less after rotation. Use α=1.0 (paper-faithful) not α=0.25.
3. **BiIP scaling is the dominant rotation factor** — more important than the choice
   of Hadamard vs random orthogonal.
4. **Permutations are complementary (not synergistic) with Hadamard** — they add ~5.6%
   on top, not super-additive. Correct order is Hadamard THEN permutation.
5. **Per-column quantization makes permutations a no-op** — permutations only help
   per-tile quantizers.
6. **Allocation interacts with rotation** — R9's surrogate allocation was rejected after
   rotation (rotation homogenizes tile sensitivity), but this may be fixture-dependent.
   R1's DP-refined allocation was tested on unrotated weights only. Composition of
   allocation with rotation remains an open question.
7. **V/O rotation is the only free attention invariant** — Q/K rotation breaks under RoPE.
8. **MLP allows permutations but not rotations** — SiLU + elementwise product commute
   with permutations but not coordinate mixing.
9. **GDN-specific approaches need real activations** — synthetic calibration makes gate
   sensitivity too uniform to exploit.
10. **The Cholesky convention bug was the most important discovery** — it affected the
    entire cleanroom codebase and reversed the central architectural conclusion.

## 6. Limitations and next steps

### Current limitations
- Most real-weight experiments use 128×128 slices from full tensors (aspect ratio hidden);
  R10 uses synthetic d_model=64, d_head=32 blocks
- Synthetic calibration (Gaussian + outliers), not real model activations
- Proxy uniform quantizers of varying granularity: R3 per-column, R7 16-column groups,
  R10 both per-column and per-tile, R1/R4/R5/R9 per-tile — none is EXL3 trellis/Viterbi
- Output-covariance proxy for H_G, not true gradient covariance (Fisher)
- In-sample evaluation for most experiments; R10 is the main held-out exception
- R8's K5 aggregate is 98.6% synthetic (real-weight effects near zero/mixed)
- R7's 186/192 comparisons reuse only 3 calibration matrices (correlated, not independent)
- R9's +40.8% uses unequal search budget (5 rotation searches vs 1 for baseline)
- No end-to-end KLD, no served model validation

### Required next steps
1. **Fix the Cholesky convention** in all production code (the correct fix:
   `U = chol(inv(H+λI)).T`)
2. **Test on actual EXL3 encoder/decoder** with Viterbi search
3. **Get real model activations** (requires GPU forward pass on aiboss)
4. **Test full-tensor (not 128×128 slices)** with block-Hadamard for non-power-of-2 dims
5. **Validate the unified stack** on served KLD (the production metric)
6. **Test with real gradient Hessian** (requires backward pass — 96GB gradient gate)
7. **Test GDN-specific approaches** with real gate pre-activations
8. **Folding transforms** for production: Hadamard signs as 1-bit/tile, BiIP scales
   as per-channel floats, permutations baked into weight layout

## 7. Artifacts

| Researcher | Code | Findings | Results | Dead ends |
|-----------|------|----------|---------|-----------|
| R1-RateDistortion | tools/research/r1-rate-distortion/poc.py | docs/research/r1-rate-distortion-findings.md | receipts/research/r1-rate-distortion-results.json | docs/research/r1-rate-distortion-deadends.md |
| R2-GPTAQ | tools/research/r2-gptaq-corrections/poc.py | docs/research/r2-gptaq-corrections-findings.md | receipts/research/r2-gptaq-corrections-results.json | docs/research/r2-gptaq-corrections-deadends.md |
| R3-Rotations | tools/research/r3-rotations/poc.py | docs/research/r3-rotations-findings.md | receipts/research/r3-rotations-results.json | docs/research/r3-rotations-deadends.md |
| R4-Permutations | tools/research/r4-permutations/poc.py | docs/research/r4-permutations-findings.md | receipts/research/r4-permutations-results.json | docs/research/r4-permutations-deadends.md |
| R5-Subspace | tools/research/r5-subspace/poc.py | docs/research/r5-subspace-findings.md | receipts/research/r5-subspace-results.json | docs/research/r5-subspace-deadends.md |
| R6-GDN | tools/research/r6-gdn/poc.py | docs/research/r6-gdn-findings.md | receipts/research/r6-gdn-results.json | docs/research/r6-gdn-deadends.md |
| R7-NoiseShaping | tools/research/r7-noise-shaping/poc.py | docs/research/r7-noise-shaping-findings.md | receipts/research/r7-noise-shaping-results-v3.json | docs/research/r7-noise-shaping-deadends.md |
| R8-Scaling | tools/research/r8-scaling/poc.py | docs/research/r8-scaling-findings.md | receipts/research/r8-scaling-results.json | docs/research/r8-scaling-deadends.md |
| R9-GroupOrbit | tools/research/r9-group-orbit/poc.py | docs/research/r9-group-orbit-findings.md | receipts/research/r9-group-orbit-results.json | docs/research/r9-group-orbit-deadends.md |
| R10-CoupledBlocks | tools/research/r10-coupled-blocks/poc.py | docs/research/r10-coupled-blocks-findings.md | receipts/research/r10-coupled-blocks-results.json | docs/research/r10-coupled-blocks-deadends.md |

## 8. Worktree branches

All 10 researchers worked in isolated git worktrees on branches `research/r1-rate-distortion`
through `research/r10-coupled-blocks`. Artifacts merged into main for this document.

## 9. References

- Doc 62: `/Users/mbelleau/Projects/qwen38-27b-exl3/docs/62-trellis-tile-quantization-experiment.md`
- BAQ: arXiv:2506.05664
- GPTQv2/GPTAQ: arXiv:2504.02692
- ResComp: arXiv:2604.07955
- KronQ: arXiv:2607.07964
- ResQ: arXiv:2412.14363
- AWQ: arXiv:2306.00978
- SmoothQuant: mit-han-lab/smoothquant
- QuIP#: Hadamard incoherence processing
- SpinQuant: learned rotations in LLM quantization
- GuidedQuant: arXiv:2505.07004
- YAQA: arXiv:2505.22988

## 10. Wave 2: Closing cross-review gaps (5 additional researchers)

**Status:** completed, 2026-08-21. 5 researchers addressed the 2 blockers and 7 concerns
from the independent cross-review of §1-9.


### 10.1 R11-UnifiedStack: Full stack factorial experiment


**Claim (reviewer-revised v2):** BiIP + Hadamard + DP allocation + GPTQ (α=0) achieves
+86.6% held-out HWE reduction over RTN at K=5 (no scaling), +57-70% over best individual.


- 19 configs, 912 arms, 4 tensors, 3 slices, K=3-6, matched byte budget
- GPTQ is the single most important component (+88-91% unrotated, +75-88% with rotation)
- Rotation+GPTQ synergy confirmed held-out (+75-88%)
- GPTAQ α=1.0 HARMFUL post-rotation (1.8× worse, worst overfitting ratio 0.844)
- Scaling (lp_pinf) harmful alone, excluded from recommended stack
- Permutation interferes with GPTQ propagation, excluded when GPTQ active
- GPTQ generalizes when off-diagonal Hessian structure is real (82% off-diag → +85.6%
  held-out) but overfits when H_X is near-diagonal (0.87% off-diag → -17.2%)


### 10.2 R12-AlphaSweep: Systematic α sweep post-rotation


**Claim (reviewer-confirmed v3, held-out evaluation):** α=0.25 dominates α=1.0 on both
HWE (42-46/48 wins) and asymmetric error (40-44/48 wins), in ALL conditions (rotated/
unrotated, natural/act-order). **The α=1.0-untested-post-rotation gap is now CLOSED:
α=1.0 is NOT better than α=0.25 post-rotation.**


- 1632 experiments + 24 diagnostics, 8 α values, 4 tensors, 3 slices, K=3-6
- GPTQ (α=0) overfits held-out: -26.4% unrotated (0/48 positive), -19.1% rotated
- P-matrix partially compensates for GPTQ overfitting at K3-K4 (oracle +2-7%) but
  fixed α=0.25 is NOT broadly beneficial (K5 -5.61%, K6 -11.29%)
- Optimal α is similar rotated vs unrotated with held-out eval (mean 0.227 vs 0.244)
- ||P|| barely changes post-rotation (-8%): BiIP shrinks ||D|| 7.7× but ||L|| grows 2.36×


### 10.3 R13-AllocRotation: Allocation + rotation composition


**Claim (reviewer-corrected v2):** Allocation composes with rotation ONLY in
rotate-then-allocate order. +71.59% mean (rot+alloc) vs +69.62% (rot only), +30.77%
(alloc only). Marginal allocation gain on top of rotation: +6.58%.


- 48 cases, 4 tensors, 3 slices, K=3-6, exact byte budget
- Alloc-then-rotate is CATASTROPHIC: -45.97% vs rotate-only
- Rotation homogenizes tile sensitivity (CV ratio 0.257, 74.3% reduction) but residual
  3-5× max/min ratio still exploitable by DP allocator
- Alternating (warm-start) adds negligible gain (+0.04% over rot-then-alloc)
- No GPTQ used — only diagonal-stat components that generalize well (R15)
- **Stack ordering confirmed: rotation MUST come before allocation**



### 10.4 R14-NoiseShapeStack: Noise shaping within the full stack


**Claim (reviewer-confirmed, 0 issues):** Act-order becomes near-no-op post-rotation
(macro-mean improvement ~10% → ~0%, win rate 80% → 61%). Root cause: rotation
uniformizes diag(H_X) (CV 2.40 → 0.23).


- 2844 records, 0 errors, 4 tensors, K=3-6
- GPTQ Schur-complement still helps post-rotation: +29.2% over Rot+RTN
- GPTAQ α=1.0 harmful post-rotation (HWE 17.8-21.5% higher than α=0)
- Rotation creates stronger spectral anti-correlation (r=-0.385) than act-order (r=-0.219)
- Best in-sample stack: BiIP+Hadamard+DP_alloc+GPTQ_LR(α=0) = +93.5% over RTN
- Rotation reduces BOTH HWE and MSE (91% and 41%); GPTQ trades MSE for HWE (30% vs 46%)
- R15 caveat: GPTQ overfits on synthetic calibration — use accept-if-improve gating



### 10.5 R15-HeldOutValidation: Held-out validation framework


**Claim (reviewer-v2 caveats incorporated):** Diagonal/weight-based transforms
generalize perfectly to held-out calibration (gap ~0pp). Full-covariance GPTQ overfits
with synthetic independent-channel calibration (gap +57.5pp) — but this is partly an
artifact since synthetic Gaussian channels make off-diagonal Hessian terms pure noise.


| Component | In-sample | Held-out | Gen gap | Overfits? |
|-----------|-----------|----------|---------|-----------|
| R3 BiIP+Hadamard | +74.8% | +74.6% | +0.2% | No |
| R1 DP Allocation | +25.5% | +25.6% | -0.2% | No |
| R4 Hadamard+p99 Perm | +33.4% | +33.4% | 0% | No |
| R7 Act-order GPTQ | +26.9% | -30.7% | +57.5% | Yes (synthetic) |
| R9 Alternating Optimizer | +86.0% | +81.7% | +4.3% | Moderate |
| Full Stack | +84.8% | +78.8% | +6.1% | Moderate |


- Ranking 100% stable across 7 splits at every K: R9 > Full_Stack > R3 > R1/R4 > R7
- R9 held-out 81.7% > R3 alone 74.6% — combination is net positive even with adversarial
  synthetic calibration
- Key principle: transforms using diagonal/marginal Hessian stats generalize; transforms
  using full covariance (Cholesky) may not with small calibration
- GPTQ overfitting is specific to synthetic independent-channel calibration (off-diag
  energy 2.18%). Real activations have genuine off-diagonal structure — GPTQ should
  generalize better. Testing with real activations is required.
- R9's accept-if-improve is the key safety mechanism (gap +4.3pp vs +57.5pp standalone)
- R9's advantage over R3 may not be solely from GPTQ (also changes orbit sampling,
  partitioning, scaling, allocation) — systematic correction on/off ablation needed



### 10.6 Updated recommended stack



Based on Wave 2 findings, the recommended stack is:



1. **BiIP diagonal balancing** (R3): generalizes perfectly, +74.6% held-out
2. **Signed randomized Hadamard** (R3): both sides, generalizes perfectly
3. **DP-refined tile allocation** (R1): generalizes perfectly, +25.6% held-out, MUST come after rotation
4. **GPTQ error propagation (α=0)** (R7/R14): +29.2% post-rotation, BUT may overfit —
   use **accept-if-improve gating** (R9 pattern)
5. **GPTAQ P-matrix (α=0.25)** (R12): NOT α=1.0. Oracle benefit at K3-K4 only;
   fixed α=0.25 not broadly beneficial. Gate behind accept-if-improve.



**Excluded from stack:**
- Scaling (lp_pinf): harmful alone (R11)
- p99 permutation: interferes with GPTQ propagation (R11)
- Act-order: near-no-op post-rotation (R14)
- GPTAQ α=1.0: harmful post-rotation (R11, R12, R14)



**Held-out performance**: R9 alternating optimizer (which includes gated GPTQ) achieves
+81.7% held-out vs +74.6% for rotation alone. The 7.1pp gain from gated GPTQ is net
positive even with adversarial synthetic calibration.



### 10.7 Remaining open questions



1. **Real-activation testing**: GPTQ overfitting may be a synthetic artifact. Must test
   with real model activations (requires GPU forward pass on aiboss).
2. **R9 correction on/off ablation**: R9's 81.7% vs R3's 74.6% may not be solely from
   GPTQ — need systematic ablation.
3. **EXL3 Viterbi**: All results use uniform per-tile quantizer, not actual trellis.
4. **Full-tensor testing**: 128×128 slices hide aspect ratio effects.
5. **Block-Hadamard for non-power-of-2**: Real tensor dims (5120, 17408) aren't powers of 2.
6. **Served KLD validation**: No end-to-end KLD measurement yet.



### 10.8 Wave 2 artifacts



| Researcher | Code | Findings | Results |
|-----------|------|----------|---------|
| R11-UnifiedStack | tools/research/r11-unified-stack/poc.py | docs/research/r11-unified-stack-findings.md | receipts/research/r11-unified-stack-results.json |
| R12-AlphaSweep | tools/research/r12-alpha-sweep/poc.py | docs/research/r12-alpha-sweep-findings.md | receipts/research/r12-alpha-sweep-results.json |
| R13-AllocRotation | tools/research/r13-alloc-rotation/poc.py | docs/research/r13-alloc-rotation-findings.md | receipts/research/r13-alloc-rotation-results.json |
| R14-NoiseShapeStack | tools/research/r14-noise-shape-stack/poc.py | docs/research/r14-noise-shape-stack-findings.md | receipts/research/r14-noise-shape-stack-results.json |
| R15-HeldOutValidation | tools/research/r15-held-out-validation/poc.py | docs/research/r15-held-out-validation-findings.md | receipts/research/r15-held-out-validation-results.json |

## 11. Wave 3: Novel directions beyond doc 62 (5 additional researchers)

**Status:** completed, 2026-08-21. 5 researchers explored directions beyond doc 62 §10:
non-uniform codebooks, diagonal GPTQ, block-level error propagation, inter-layer
allocation, and information-theoretic bounds.

### 11.1 R16-NonUniformCodebook: Non-uniform and learned codebooks

**Claim (reviewer-revised):** Non-uniform codebooks (Lloyd-Max, k-means, Hessian-weighted
Lloyd-Max) improve per-tile HWE by +31-75% unrotated at matched K. But at matched BYTE
budget, per-tile codebook storage (7.7-60% overhead, exponential in K) makes non-uniform
lose to uniform K+1 in 59/60 cases. The one exception is L55_down/first (kurtosis=100)
where k-means wins by 2-6×.

- After rotation, non-uniform hurts at K3-4 but helps at K5-6 (+16-61%)
- Codebook overhead: K3=7.7%, K4=17.6%, K5=33.3%, K6=60.0%
- Hadamard drives kurtosis → 0 (outlier removal, not Gaussianization per Bennett's theorem)
- DP allocation composes orthogonally with all quantizer types (~50-60%)
- **Conclusion: per-tile non-uniform codebooks NOT worth it. Shared/trellis codebook (O(1)
  storage) remains open.**

### 11.2 R17-DiagGPTQ: Diagonal-covariance GPTQ that generalizes

**Claim (reviewer-confirmed v2):** Block-diagonal GPTQ (16×16 blocks, global λ) is the
safest GPTQ variant: post-rotation held-out +75.7%, generalization gap only +0.65pp (vs
Full GPTQ's +6.36pp).

- Off-diagonal H^{-1} entries ARE the source of GPTQ overfitting (not Cholesky structure)
- Diagonal H^{-1} GPTQ overfits identically to Full (gen gap +36pp) — dead end
- Forward-H GPTQ catastrophic (-1147% vs RTN) — dead end
- Full GPTQ is net positive post-rotation (+1.8pp over rotation-only, 79% win rate)
- **Ungated GPTQ + allocation is net HARMFUL** (interferes with allocation's bit distribution)
- **Stack: EITHER rotation→allocation (no GPTQ) OR rotation→GPTQ (gated, no allocation)**
- Block-diagonal GPTQ preferred when gating unavailable

### 11.3 R18-BlockPropagation: Block-level error propagation

**Claim (reviewer-confirmed v2):** Rotation persists at block output: Hadamard gives +56%
L55 MLP block error reduction. SiLU is in linear regime (FD/Jac ≈ 1.0, mean SiLU' ≈ 0.50).
Cross-matrix interaction is low (0.3-11%, additivity ratio 0.95-1.13).

- down_proj dominates L55 block error (75% of individual weight error)
- Joint greedy allocation helps synthetic attention (+51% K5) but NOT real MLP (0% L0)
- GQA 7.7× amplification was a bug (broken einsum). Corrected: ratio ~0.03
- K5K6 recipe already near-optimal for MLP
- Block amplification factors could weight inter-layer DP objective (R19)

### 11.4 R19-InterlayerAlloc: Inter-layer optimal K allocation

**Claim (reviewer-confirmed):** Inter-layer DP allocation (multiple-choice knapsack across
all layers × roles) reduces total HWE by **+39.1-47.3%** over uniform-K at matched byte
budget, and +40.7% over the K5K6 recipe.

- Attention tensors are 2.5-3.6× more sensitive than MLP
- L55 is the most sensitive layer (2.4× over mean), max/min ratio ~115×
- Global DP dominates role-partitioned by +18.2% — global bit transfer across roles is essential
- Rotation homogenizes within-layer but NOT inter-layer sensitivity (DP gain persists post-rotation +26.3%)
- KronQ proxy: Spearman ρ=0.920, but tr(H_G) alone (r=0.966) and MSE×KronQ (r=0.990) are better
- **Hierarchical allocation**: R19 allocates K per layer/role, R1 allocates K per tile within tensor

### 11.5 R20-InfoTheoretic: Information-theoretic bounds

**Claim (reviewer-corrected v3):** Uniform per-tile quantization is only 11-16% above the
Panter-Dite high-rate asymptotic reference at K=5. Lloyd-Max improves 46-48% but is
non-deployable (codebook overhead). Δ²/12 is valid at K≥3 (mean ratio 0.98-1.01).

- The 200-430% gap to Gaussian R-D is structural (scalar vs vector, tile range ~6σ vs σ)
- Hadamard drives kurtosis → 0; BiIP adds variance scaling (15-45×), not Gaussianization
- Entropy coding saves 13-24% of rate (compression-layer benefit)
- Weights are 74-79% Gaussian by AIC (GGD β≈1.86-1.99)
- L55_down is the outlier: 430% gap, kurtosis 60.3, 93.5% HWE improvement from rotation
- **Conclusion: uniform is near-optimal for fixed-rate scalar. Lloyd-Max is the main lever
  but non-deployable. Rotation is the key transform. HWE has the most room (75-95% achieved,
  no theoretical bound).**

### 11.6 Wave 3 updated stack recommendation

Based on wave 3, the recommended stack bifurcates:

**Path A (rotation + allocation, no GPTQ):**
1. BiIP diagonal balancing → 2. Hadamard both sides → 3. Inter-layer DP allocation (R19) +
   intra-layer DP allocation (R1) → 4. Uniform quantization
- Generalizes perfectly (gap ~0pp), +74.6% held-out from rotation, +25.6% from allocation

**Path B (rotation + gated GPTQ, no allocation):**
1. BiIP → 2. Hadamard → 3. Full GPTQ (α=0.25) with accept-if-improve gating → 4. Uniform quantization
- +81.7% held-out (R9), but GPTQ overfitting risk with real activations unknown
- Block-diagonal GPTQ (R17) is the safe fallback (gap +0.65pp)

**Do NOT combine allocation + ungated GPTQ** (R17: GPTQ interferes with allocation).

### 11.7 Wave 3 artifacts

| Researcher | Code | Findings | Results |
|-----------|------|----------|---------|
| R16-NonUniformCodebook | tools/research/r16-nonuniform-codebook/poc.py | docs/research/r16-nonuniform-codebook-findings.md | receipts/research/r16-nonuniform-codebook-results.json |
| R17-DiagGPTQ | tools/research/r17-diag-gptq/poc.py | docs/research/r17-diag-gptq-findings.md | receipts/research/r17-diag-gptq-results.json |
| R18-BlockPropagation | tools/research/r18-block-propagation/poc.py | docs/research/r18-block-propagation-findings.md | receipts/research/r18-block-propagation-results.json |
| R19-InterlayerAlloc | tools/research/r19-interlayer-alloc/poc.py | docs/research/r19-interlayer-alloc-findings.md | receipts/research/r19-interlayer-alloc-results.json |
| R20-InfoTheoretic | tools/research/r20-info-theoretic/poc.py | docs/research/r20-info-theoretic-findings.md | receipts/research/r20-info-theoretic-results.json |

## 12. Authoritative pre-Wave-5 synthesis

### 12.1 Promotion verdict

No R1–R28 quality headline currently authorizes a checkpoint or production
promotion. The lab established algebraic invariants, failure modes, and useful
screening hypotheses, but it has not completed the causal chain:

1. real document-disjoint activations and gradients;
2. actual current stock EXL3 signs/H128/`suh`/`svh`/scale search/BlockLDLQ/Viterbi;
3. full-tensor serialize/decode in the original basis;
4. one-tensor or full-checkpoint replacement;
5. untouched full-vocabulary KLD/EAR/p99;
6. exact codec and production-profile runtime qualification.

All percentages below are proxy-screening evidence unless the scope is stated in
the same sentence. Local MSE, OC-HWE, Fisher-HWE, block output error, and served
full-vocabulary KLD are distinct metrics.

### 12.2 Wave 4 and external-review ledger

| Axis | Latest scoped result | Authoritative disposition |
|------|----------------------|---------------------------|
| R21 shared codebook | Shared/no-tile-scale + rotation + DP wins its affine proxy 12/12 at K4/K5 | Proxy sidecar saving does not exist in stock EXL3; learned/shared actual codebook remains open |
| R22 BD-GPTQ × allocation | Final exact rerun: joint arm is −0.94 pp vs BD-GPTQ and −0.39 pp vs allocation; reallocation neutral | Confirms conditional bifurcation in this proxy; exact correction-inside-each-K RD remains open |
| R23 full tensors | BiIP+H16 gives 64–81% OC-HWE vs unrotated on four full MLP tensors; 3/4 slice directions transfer | Not incremental to stock H128+LDLQ; H16 solves no production alignment problem |
| R24 entropy allocation | Broad 45-tensor/900-block screen: about +58% empirical-entropy oracle at K4/K5 | Same-block lower bound, not a serialized coder or held-out policy; strata/reporting corrected to 8/8/4 |
| R25 multi-seed | 20-seed proxy directions are statistically robust vs unrotated RTN | Four fixed 128² blocks and synthetic population; no stock-EXL3/KLD inference |
| R26 HARP/BiIP | BiIP 68.9% vs naive, but stock-like H+GPTQ proxy 93.4% vs BiIP+GPTQ 92.5%; greedy signs +0.2–1.7% | Large BiIP headline mostly rediscovers existing incoherence/correction; BiIP is a zero-byte candidate, incremental quality unknown |
| R27 equal-rate | After full-tensor sign amortization and rotate→reallocate, K5.5+Had beats K5.5 allocation in 9/12 fixed slices | Near-equal affine proxy, in-sample allocation, no stock trellis/LDLQ/KLD; useful fractional-rate screen only |
| R28 multi-precision | DP solves supplied `{K,K+BiIP,K+Had}` menus; broad proxy strongly favors slice-local BiIP | Action prices and synthetic extrapolation invalidate an EXL3 frontier; rebuild from full-tensor serialized stock actions |

The decisive R22 final result supersedes its rejected v1–v3 claims. Correction and
allocation must be measured as one complete action at every K. A post-hoc
correction or a K-independent correction factor cannot define a rate-distortion
curve.

### 12.3 Exact stock-EXL3 delta

Current EXL3 already includes:

- one integer K per tensor/shard and fixed-stride 16×16 trellis packing;
- random input/output signs and block-Hadamard-128;
- FP16 signed magnitude vectors `suh`/`svh`;
- global scale search;
- block-LDLQ feedback inside actual Viterbi encoding;
- procedural MCG/MUL1 reconstruction.

Therefore:

- the 55–97% rotation/BiIP proxy gains are **not** incremental stock-EXL3 gains;
- BiIP may reuse existing `suh`/`svh` with zero incremental payload/hot operation,
  but its quality delta against stock is unknown;
- the cleanroom `chol(inv(H)).T` correction must **not** be transplanted blindly
  into stock block-LDLQ, which uses a different correct recurrence;
- per-tile mixed K, entropy streams, arbitrary LUTs, post-H permutations, H16,
  dense learned rotations, and QSRT boundary transforms are new-format/runtime
  lanes rather than stock converter changes;
- per-tensor integer K allocation is the first deployable multi-precision action.

The production throughput profile also materializes selected trellis shards to
FP4/FP6 and discards their trellis payload. Every finalist must be qualified both
in codec-exact all-trellis mode and under the unchanged production profile.

### 12.4 Architecture corrections

1. **Gated attention V/O:** arbitrary dense V/O rotation is not free because Qwen
   applies a token-varying per-head output gate before O. The zero-runtime exact
   family is coordinated monomial transforms (permutation/diagonal/sign) with
   gate coordinates and inverse O transformed consistently.
2. **Q/K:** production full attention is 24 Q / 4 KV, head dimension 256, rotary
   dimension 64, with learned Q/K RMSNorm. Zero-hot-path arms are coordinated
   signs/permutations with norm weights co-permuted. Dense post-norm rotations
   need a fused runtime boundary operation and must commute with relative RoPE.
3. **SwiGLU:** generic rotation does not commute through SiLU, but QSRT's explicit
   activation-boundary construction is algebraically exact: restore the
   preactivation basis before SiLU/product, then apply a separate postactivation
   basis matched by down. It is not free; dense Qwen pays/fuses the transforms
   in every MLP invocation.
4. **GDN:** R6's simplified recurrence omitted depthwise conv+activation, q/k
   normalization, and RMSNormGated. Its negative result does not close real GDN.
   Exact channel permutations remain plausible if propagated through conv,
   state, q/k/v/z, norm, and out coordinates.
5. **Down targets:** the largest zero-hot-path architecture lever may be fitting
   each down target/covariance from the actual decoded gate/up candidate and
   then stock-encoding the target, with no adapter payload.

### 12.5 QSRT crossover

The QSRT review at `pedapudi/qsrt@a5d68dd` found complementary mechanisms on the
same QTIP/EXL3 trellis lineage:

- exact gate/up/down activation-boundary Hadamard transforms;
- candidate-conditioned down covariance and continuous target refit;
- dense-H legal-path block-coordinate refinement;
- final-KL gradient target shifting with hard legal Viterbi;
- deterministic periodic mixed-K grammars;
- MoE route-aware whole-expert selection and router recovery.

Dense Qwen can use the first four and byte-matched allocation. Router, rare-expert,
co-routing, and expert-atom machinery do not transfer. QSRT's finite E4M3 SQG
endpoint is a K2/K3 idea and loses to MCG at K5/K6; keep it out of the high-rate
mainline unless a wider endpoint first beats stock.

### 12.6 arXiv horizon

The final literature scan prioritized:

1. **SLQ (2605.02404):** minimum bytes subject to model-distribution fidelity;
   Qwen3.5-27B reports 5.19-bpp distribution-lossless and 3.30-bpp task-lossless.
2. **Q-Palette (2509.20214):** real fractional TCQ/fixed half-TCQ kernels and
   hardware-aware allocation; motivates a conditional fixed-stripe K5/K6 lane.
3. **BCJR-QAT (2605.10655):** differentiable trellis path optimization with
   hard-projected unchanged payload and forward-KL objective.
4. **WaterSIC/QMM II:** Cholesky-innovation rate oracle and high-rate stopping
   test; randomly rotated GPTQ can be within roughly 0.1 equivalent bit.
5. **PiSO/Four Over Six:** zero-byte scale-family optimization.
6. **DASH-Q:** covariance shrinkage selected for robustness rather than maximal H.
7. **SchurQuant/GSQ/QES:** expensive in-format legal-code search.
8. **ICBQ:** reconstructed/rerolled dense block seams.
9. **GEMQ:** MoE-only global expert allocation/router adaptation.

No paper covers exact EXL3 K5/K6, fixed payload/runtime, full-vocabulary KLD, and
the project's p99 tail together. That is the Wave-5 opportunity.

### 12.7 Pre-Wave-5 data and search contract

- Use the M4 Max MPS GPU for batched screens/search; CPU for unsupported
  eig/SVD/Cholesky.
- Use correctly decoded BF16. The old L10–L40 extra file is forbidden.
- Common census: 9 depths × roles gate/up/down/qkv/out/z, 8 blocks/tensor for
  screening and 20+ for promotion, including diagonal/random/off-diagonal blocks.
- Full tensors for final converter arms. Qwen dimensions are divisible by stock
  H128; no H16 substitution is required.
- Source-disjoint calibration, validation/model selection, and untouched test
  documents. Equal candidate/seed search budgets and preregistered contrasts.
- Build a heterogeneous complete-action menu per legal module/topology group.
  One action includes K, transform, scales, codebook, target, correction, and
  serialized payload. Expensive encode search is welcome; decode hot-path cost is
  a hard constraint.
- Local proxy wins only shortlist. Promotion requires actual stock EXL3
  serialize/decode, full-vocabulary mean/p99 KLD, EAR/top1, exact bytes, startup,
  graph capture, PP/TG, context, and no runtime fallback.

### 12.8 Wave 4 artifact index

| Axis | PoC | Findings | Receipt |
|------|-----|----------|---------|
| R21 shared codebook/trellis proxy | `tools/research/r21-trellis-sim/poc.py` | `docs/research/r21-trellis-sim-findings.md` | `receipts/research/r21-trellis-sim-results.json` |
| R22 BD-GPTQ × allocation | `tools/research/r22-blockdiag-alloc/poc.py` | `docs/research/r22-blockdiag-alloc-findings.md` | `receipts/research/r22-blockdiag-alloc-results.json` |
| R23 full-tensor screen | `tools/research/r23-full-tensor/poc.py` | `docs/research/r23-full-tensor-findings.md` | `receipts/research/r23-full-tensor-results.json` |
| R24 entropy allocation | `tools/research/r24-entropy-alloc/poc.py` / `wave5_poc.py` | `docs/research/r24-entropy-alloc-findings.md` / `r24-entropy-alloc-wave5-findings.md` | `receipts/research/r24-entropy-alloc-results.json` / `r24-entropy-alloc-wave5-results.json` |
| R25 multi-seed robustness | `tools/research/r25-multiseed/poc.py` | `docs/research/r25-multiseed-findings.md` | `receipts/research/r25-multiseed-results.json` |
| R26 HARP/BiIP comparison | `tools/research/r26-harp-comparison/poc.py` | `docs/research/r26-harp-comparison-findings.md` | `receipts/research/r26-harp-comparison-results.json` |
| R27 equal-rate proxy | `tools/research/r27-equal-rate-kld/poc.py` | `docs/research/r27-equal-rate-kld-findings.md` | `receipts/research/r27-equal-rate-kld-results.json` |
| R28 multi-precision menu | `tools/research/r28-multi-precision/poc.py` | `docs/research/r28-multi-precision-findings.md` | `receipts/research/r28-multi-precision-results.json` |

Wave 5 is the final research wave and is specified in doc 64.
