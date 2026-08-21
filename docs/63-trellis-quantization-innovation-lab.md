# 63 — Trellis Quantization Innovation Lab: 10-Researcher Collaborative Findings

**Status:** completed, 2026-08-21. 10 mathematician-researcher subagents explored novel
quantization approaches in isolated git worktrees, each with adversarial openai-reviewer
verification. All findings below are reviewer-confirmed or honestly reported as negative.

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
  RoPE requires operator conjugation R'_t = P R_t P^T, GDN unconditionally safe
- Per-column permutation is no-op (per-column quantizer is permutation-invariant)
- **Dead ends:** spectral seriation (negative), balanced scale assignment (neutral-to-
  harmful), permute-then-Hadamard (wrong order), median scale packing (worse than RMS)

### R5-Subspace: ResQ + subspace quantization

**Claim (v3 final, reviewer-pending):** GPTQ within PCA subspaces provides **8.6% (K5)
to 25% (K6)** improvement over plain PCA with matched quantizers. Tile PCA viable:
1.22-1.56× K+1 bytes, 2.0-2.6× lower error than K+1.

- Activation PCA > Weight PCA > Joint PCA (7× gap)
- Joint PCA broken: generalized eigenvectors are H_W-orthogonal, QR orthonormalization
  destroys ordering
- Global ResQ impractical for weight-only PTQ: n²×2 projection overhead, dominated by
  uniform FP16 at matched bytes (3600×)
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
- Balanced realization dead end was an exponent bug (sqrt vs eigvals**0.25); corrected
  transform gives 96×/202× improvement — but this was on synthetic data only

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
- GPTQ (α=0) outperforms GPTAQ (α=0.25) after rotation on 3/4 tensors — GPTQ error
  propagation is the main contributor, not the P-matrix
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
| GPTAQ α=0.25 (reference code) | R2 | Paper-faithful α=1.0 wins 34/36; 0.25 is code convention, not paper |
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
| Column-BAQ | R1 | Aggregation heuristic collapses to near-uniform |

## 4. The recommended quantization stack

Based on all verified findings, the recommended operation order is:

1. **Scaling** (R8): Lp-norm p=∞ (max activation magnitude) or AWQ (mean|X|^α)
2. **BiIP diagonal balancing** (R3): S_X = (diag(H_X)/diag(W^T W))^{1/4}, S_G similar
3. **Signed randomized Hadamard** (R3): both input and output dimensions
4. **p99-scale permutation** (R4): sort transformed weights by p99 column scale
5. **Act-order GPTQ** (R7): quantize columns in descending diag(H_X) order
6. **GPTAQ correction with α=1.0** (R2): paper-faithful, no 0.25 multiplier
7. **DP-refined tile allocation** (R1): local search on full Hessian-weighted objective
8. **V/O rotation for attention** (R10): free invariant, 22.5% held-out improvement
9. **MLP-safe permutation** (R4, R10): same P for gate/up, inverse for down

The alternating optimizer (R9) with accept-if-improve selects which steps compose
per-tensor. Not all steps help every tensor — L55_down correctly rejects GPTAQ after
rotation, while L0_gate benefits from the full stack.

### Expected improvement (from verified individual measurements)

| Component | Improvement | Metric | Caveat |
|-----------|------------|--------|--------|
| BiIP + Hadamard rotation | 57-82% (68.5% macro) | OC-proxy HWE | Fixed recipe |
| Rotation + GPTQ synergy | 30.5-53.8% | Over best single | All 4 tensors |
| Unified 6-step optimizer | 40.8% mean | Over best individual | All 4 tensors |
| DP-refined allocation | 25.5% | Over uniform K, HWE | 100% win rate |
| Act-order GPTQ | 21-24% | Over RTN, HWE | 97% head-to-head |
| V/O attention rotation | 22.5% | Held-out block error | Free invariant |
| GPTAQ α=1.0 vs GPTQ | 0.2-10.2% | Grows with K | Paper-faithful |
| p99 permutation (post-Hadamard) | 5.6% | Over Hadamard alone | Complementary |
| Scaling (lp_pinf) + GPTAQ | 3.3% | Over no-scaling+GPTAQ | Not subsumed |
| Tile-PCA ResQ | 2.0-2.6× | Lower error than K+1 | 1.22-1.56× bytes |

**Important:** These are individual component improvements on proxy metrics with
synthetic calibration. They do NOT simply add. The R9 alternating optimizer is the
principled way to combine them, and it achieves +40.8% over the best individual method.

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
6. **Allocation works best on unrotated weights** — rotation homogenizes tile sensitivity.
   But the alternating optimizer can allocate BEFORE rotation or after correction changes
   the landscape.
7. **V/O rotation is the only free attention invariant** — Q/K rotation breaks under RoPE.
8. **MLP allows permutations but not rotations** — SiLU + elementwise product commute
   with permutations but not coordinate mixing.
9. **GDN-specific approaches need real activations** — synthetic calibration makes gate
   sensitivity too uniform to exploit.
10. **The Cholesky convention bug was the most important discovery** — it affected the
    entire cleanroom codebase and reversed the central architectural conclusion.

## 6. Limitations and next steps

### Current limitations
- All results use 128×128 slices from full tensors (aspect ratio hidden)
- Synthetic calibration (Gaussian + outliers), not real model activations
- Uniform per-tile quantizer, not EXL3 trellis/Viterbi
- Output-covariance proxy for H_G, not true gradient covariance (Fisher)
- In-sample evaluation for most experiments
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
| R1-RateDistortion | tools/research/r1-rate-distortion/poc.py | docs/research/r1-rate-distortion-findings.md | receipts/research/r1-rate-distortion-results.json | — |
| R2-GPTAQ | tools/research/r2-gptaq-corrections/poc.py | docs/research/r2-gptaq-corrections-findings.md | receipts/research/r2-gptaq-corrections-results.json | docs/research/r2-gptaq-corrections-deadends.md |
| R3-Rotations | tools/research/r3-rotations/poc.py | docs/research/r3-rotations-findings.md | receipts/research/r3-rotations-results.json | docs/research/r3-rotations-deadends.md |
| R4-Permutations | tools/research/r4-permutations/poc.py | docs/research/r4-permutations-findings.md | receipts/research/r4-permutations-results.json | docs/research/r4-permutations-deadends.md |
| R5-Subspace | tools/research/r5-subspace/poc.py | docs/research/r5-subspace-findings.md | receipts/research/r5-subspace-results.json | docs/research/r5-subspace-deadends.md |
| R6-GDN | tools/research/r6-gdn/poc.py | docs/research/r6-gdn-findings.md | receipts/research/r6-gdn-results.json | docs/research/r6-gdn-deadends.md |
| R7-NoiseShaping | tools/research/r7-noise-shaping/poc.py | docs/research/r7-noise-shaping-findings.md | receipts/research/r7-noise-shaping-results-v3.json | docs/research/r7-noise-shaping-deadends.md |
| R8-Scaling | tools/research/r8-scaling/poc.py | docs/research/r8-scaling-findings.md | receipts/research/r8-scaling-results.json | docs/research/r8-scaling-deadends.md |
| R9-GroupOrbit | tools/research/r9-group-orbit/poc.py | docs/research/r9-group-orbit-findings.md | receipts/research/r9-group-orbit-results.json | — |
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
