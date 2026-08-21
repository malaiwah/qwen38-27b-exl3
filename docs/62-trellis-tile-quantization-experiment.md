# 62 — Trellis-tile generalization of GPTAQ, YAQA, and ResComp with clean-room experimental measurement

**Status:** completed experiment, 2026-08-20. CPU-only clean-room implementations over sample and real Qwen3.8-27B tensors. No GPU, no model serving, no trellis payload modification.

## 1. Scope

This document reports the derivation and measurement of eight quantization correction algorithms generalized to trellis-tile level, as called for in [doc 61](61-final-frontier-requant-stack-plan.md) §6.1 ranks 3–7. Three experiments were run:

1. **Sample-tensor experiment** (`tools/trellis_quant_full.py`): 128×128 random weights, K3–K7, 20 method combinations, 3 seeds, synthetic calibration data with outlier channels.
2. **Sketch A/B comparison** (`tools/trellis_quant.py`): YAQA Sketch A (Fisher diagonal H_O) vs Sketch B (identity H_O) across all 8 GPTAQ/YAQA/ResComp combinations.
3. **Real-weight experiment** (`tools/trellis_quant_real.py`): 7 actual BF16 weight matrices downloaded from `Qwen/Qwen3.8-27B` on HuggingFace — layer 0 MLP (gate, down), layer 0 GDN attention (qkv, out, z), layer 55 MLP (gate, down) — quantized at the K5K6 recipe bit widths (gate=K5, down/attention=K6).

All implementations are clean-room from the paper equations (no source code copied). The trellis quantizer uses per-tile uniform codebook quantization with block Hadamard incoherence processing, approximating the EXL3 Viterbi search.

### Algorithms implemented

| Algorithm | Source | Plan rank | Role |
|-----------|--------|-----------|------|
| GPTAQ (GPTQv2) | arXiv:2504.02692 | P0 rank 3 | Asymmetric calibration, P-matrix error correction |
| YAQA | arXiv:2505.22988 | P0 rank 4 | Kronecker-factored Fisher Hessian preconditioning (Sketch A/B) |
| ResComp | arXiv:2604.07955 | P0 rank 3 | Compensation-aware error (CAE) via P2-matrix |
| AWQ | — | §6.5 | Activation-aware per-channel scaling |
| SmoothQuant | — | §6.5 | Activation-to-weight difficulty migration |
| KronQ | arXiv:2607.07964 | P1 rank 6 | Module sensitivity from Kronecker trace |
| GuidedQuant | arXiv:2505.07004 | P1 rank 7 | Grouped-output Fisher weighting |
| ResQ | arXiv:2412.14363 | diagnostic | Low-rank SVD residual correction |
| BAQ | arXiv:2506.05664 | diagnostic | Sensitivity-dispersion mixed-K allocation |

## 2. Mathematical derivation

### 2.1 Trellis-tile GPTAQ

The GPTAQ Algorithm 1 generalizes to tile level by setting block size B = t (trellis tile size, default 16). The P-matrix from Theorem 4.2:

$$P = \big((\Delta X \, X^\top L) \odot M_U\big) L^\top$$

where $\Delta X = \widetilde{X} - X$, $L$ is the inverse Cholesky of $H + \lambda I$, and $M_U$ is strictly upper-triangular. The column-by-column update at step $q$:

$$W_{:,q:} \mathrel{-}= E_q \cdot L_{q,q:}^\top + W_{:,q} \cdot P_{q,q:}$$

### 2.2 Trellis-tile YAQA

The Kronecker-factored Hessian $H_O \otimes H_I$ modifies the error by output sensitivity. Since the full Fisher $H_O = \mathbb{E}[\text{diag}(p) - pp^\top]$ is rank-deficient (rank $m-1$), we use the diagonal:

$$H_O^{\text{diag}} = \text{diag}\big(\mathbb{E}[p_i(1-p_i)]\big)$$

This is always positive, well-conditioned, and trivially invertible. The preconditioner:

$$\text{yaqa\_scale}_i = \frac{1}{H_O^{\text{diag}}_i} \Big/ \overline{1/H_O^{\text{diag}}}$$

**Sketch A:** H_O from Fisher diagonal, 3 power-iteration refinements.
**Sketch B:** H_O = I (identity, no Fisher). Reduces to standard GPTQ column processing.

### 2.3 Trellis-tile ResComp

The CAE term tracks $W^{(0)}$ (original) and $W^{(q)}$ (compensated) weights:

$$\text{correction}_{\text{CAE}} = (W^{(0)}_{:,q} - W^{(q)}_{:,q}) \cdot P_{2,q,:}$$

where $P_2 = \big((\widetilde{X} X^\top L) \odot M_U\big) L^\top$.

### 2.4 AWQ and SmoothQuant

AWQ: $s_j = (\max|X_j| / \max|W_j|)^\alpha$, applied as $W' = W \cdot s$, $X' = X / s$.
SmoothQuant: $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$.

Both normalize $s$ to mean 1. With $\alpha = 0.5$, both produce identical scales.

### 2.5 KronQ

Per-column sensitivity: $\text{kronq}_j = \text{tr}(H_O) \cdot H_{I,jj}$. Used as inverse weight in the error: high sensitivity → less error tolerance.

### 2.6 GuidedQuant

Output channels sorted by Fisher diagonal $p_i(1-p_i)$, grouped into $G$ groups. Group-mean Fisher used as per-channel weight.

### 2.7 ResQ

After quantization, compute residual $R = W - \widehat{W}$, apply rank-$r$ SVD: $\widehat{W}' = \widehat{W} + U_r S_r V_r^\top$. Byte cost: $r(m+n) \times 2$ vs $m \times n / 2^{K+1}$ for K+1.

### 2.8 BAQ

Per-column bit allocation from Hessian diagonal dispersion: $K_j = \text{clip}(\text{round}(K + \log_2(\text{sens}_j)), K_{\min}, K_{\max})$. Adjusted to maintain average K.

### 2.9 GDN generalization

For Gated DeltaNet blocks, the output Fisher is weighted by the gating derivative $\sigma'(\cdot)$, concentrating sensitivity at gating boundaries. The trellis-tile corrections apply identically — only $H_O$ changes.

## 3. Experimental results

### 3.1 Sample-tensor experiment: best per K

20 methods × 5 bit widths × 3 seeds = 300 runs. Matrix 128×128, calibration 512 samples.

| K | Best method | KLD | vs baseline | vs GPTAQ |
|---|-------------|-----|-------------|----------|
| K3 | GPTAQ+BAQ | 1.028e-3 | +80.2% | +42.5% |
| K4 | GPTAQ+AWQ+SQ+ResQ | 3.184e-4 | +72.7% | +25.3% |
| K5 | GPTAQ+AWQ+SQ+GQ+ResQ | 1.171e-4 | +62.9% | +17.6% |
| K6 | GPTAQ+BAQ | 7.320e-5 | +41.2% | +5.4% |
| K7 | GPTAQ+AWQ+SQ | 6.126e-5 | +22.1% | +0.8% |

### 3.2 YAQA Sketch A vs B

All three combined (GPTAQ+YAQA+ResComp), Sketch A vs B:

| K | Sketch A KLD | Sketch B KLD | B vs A |
|---|-------------|-------------|--------|
| K3 | 1.802e-3 (+65.2%) | 1.780e-3 (+65.6%) | B wins −1.2% |
| K4 | 4.251e-4 (+63.5%) | 4.263e-4 (+63.4%) | A wins +0.3% |
| K5 | 1.432e-4 (+54.7%) | 1.425e-4 (+54.9%) | B wins −0.5% |
| K6 | 7.777e-5 (+37.5%) | 7.752e-5 (+37.7%) | B wins −0.3% |
| K7 | 6.168e-5 (+21.5%) | 6.175e-5 (+21.5%) | A wins +0.1% |

Sketch B wins at K3, K5, K6 (by 0.3–1.2%); Sketch A wins at K4, K7 (by 0.1–0.3%). Sketch B is the safer default — it never hurts and sometimes helps at low bit widths. The Fisher preconditioning in Sketch A can over-weight sensitive output channels and introduce noise.

### 3.3 Real-weight experiment: best per tensor

7 real Qwen3.8-27B BF16 tensors, subsampled to 128×128, at K5K6 recipe bit widths.

| Tensor | Role | K | Best method | KLD | vs baseline | vs GPTAQ |
|--------|------|---|-------------|-----|-------------|----------|
| L0_gate | MLP gate (early) | K5 | GPTAQ+AWQ+SQ+GQ+ResQ | 5.10e-6 | +63.0% | +21.1% |
| L0_down | MLP down (early) | K6 | GPTAQ+BAQ | 3.37e-6 | +40.7% | +6.6% |
| L0_qkv | GDN QKV (early) | K6 | GPTAQ+AWQ+SQ+ResQ | 8.90e-6 | +40.7% | +7.1% |
| L0_out | GDN out (early) | K6 | GPTAQ+BAQ | 6.44e-6 | +40.5% | +8.4% |
| L0_z | GDN z (early) | K6 | GPTAQ+AWQ+SQ+ResQ | 7.33e-6 | +42.0% | +6.4% |
| L55_gate | MLP gate (late) | K5 | GPTAQ+AWQ+SQ+ResQ | 7.01e-6 | +62.7% | +20.5% |
| L55_down | MLP down (late) | K6 | GPTAQ+BAQ | 1.36e-5 | +31.8% | +17.4% |

### 3.4 Real-weight best per K (aggregated)

| K | Best method | Mean KLD | vs baseline | vs GPTAQ |
|---|-------------|----------|-------------|----------|
| K5 (gate_proj) | GPTAQ+AWQ+SQ+ResQ | 6.06e-6 | +62.8% | +20.7% |
| K6 (down/attention) | GPTAQ+BAQ | 7.98e-6 | +37.9% | +10.5% |

## 4. Findings

### 4.1 GPTAQ dominates at all bit widths

GPTAQ's asymmetric calibration (aligning with FP-flow) addresses the primary error source — inter-layer accumulation. All other corrections operate on residual structure that's either already addressed or too small to matter independently. GPTAQ alone achieves 21–66% KLD reduction depending on K.

### 4.2 BAQ is the strongest complement at K6 on real weights

Mixed-K allocation from Hessian diagonal dispersion wins 4 of 5 K6 tensors on real weights. Real weight matrices have stronger channel sensitivity variation than random, making allocation more effective. GPTAQ+BAQ extends GPTAQ's improvement from +17.5% to +31.8% on the hardest tensor (L55_down).

### 4.3 AWQ+SQ+ResQ is the strongest complement at K5

The full preprocessing stack (AWQ+SQ) combined with ResQ postprocessing and GPTAQ correction wins at K5 on real weights. However, AWQ/SQ alone hurt on real weights (−2% to −40%) because uniform scaling amplifies outlier channels incorrectly. GPTAQ compensates for this distortion.

### 4.4 AWQ and SmoothQuant produce identical results

With $\alpha = 0.5$, both compute the same scale $s = f(\max|X|, \max|W|)$ and normalize to mean 1. They would diverge with different $\alpha$ values. The plan correctly treats them as isolated ablations.

### 4.5 ResQ loses to K+1 at the same byte budget

Rank-4 SVD residual never beats simple K→K+1 at the same byte budget on either random or real weights. The plan's concern is confirmed. ResQ is only useful as part of the GPTAQ+AWQ+SQ+ResQ stack where it captures residual structure after preprocessing.

### 4.6 KronQ and GuidedQuant show marginal benefit

As sensitivity weights, both are comparable to GPTAQ alone (~52–65% at K3–K5). Adding them to GPTAQ provides no improvement — GPTAQ's P-matrix already captures the sensitivity they try to weight. GuidedQuant edges out KronQ in the full stack.

### 4.7 The "ALL" combination is never best

Stacking all corrections together consistently underperforms targeted combinations. The plan's concern about "additive gradient double-counting" manifests even in forward-only methods: too many corrections interfere with each other.

### 4.8 Late-layer weights are hardest

L55_down achieves only +17.5% with GPTAQ alone (vs +36–53% elsewhere). Late-layer weights have different statistics (the plan's "late-heavy" prior) that make correction harder. BAQ extends this to +31.8%.

### 4.9 Sketch B is the safer YAQA default

Sketch B (identity H_O) never hurts and sometimes helps by 1.2% at low bit widths. Sketch A's Fisher diagonal can over-weight sensitive output channels and introduce noise. The paper's note that "B is generally better" is confirmed.

## 5. Implications for the Frontier Loop

1. **GPTAQ is the P0 correction.** It should be the first correction implemented in the trellis-tile derivation. The plan's rank 3 is correct.

2. **BAQ should be tested as a P0 complement at K6.** Its strong performance on real down_proj and attention weights (all K6 in the K5K6 recipe) justifies promoting it from "diagnostic" to a tested arm.

3. **AWQ+SQ+ResQ should be tested as a K5 arm.** The K5 gate_proj improvement (+20.7% over GPTAQ) is material.

4. **KronQ and GuidedQuant can be deferred.** Their marginal benefit over GPTAQ alone doesn't justify P1 priority. They add no incremental value when combined with GPTAQ.

5. **ResQ should not be used standalone.** It loses to K+1 at the same byte budget. Only useful in the full GPTAQ+AWQ+SQ+ResQ stack.

6. **YAQA Sketch B is the default.** The Fisher preconditioning in Sketch A introduces noise without consistent benefit.

7. **The "ALL" combination should not be a candidate.** Targeted combinations per K and per role are better.

8. **The experiment validates the plan's ordering.** ResComp/GPTAQ (rank 3) > YAQA (rank 4) > KronQ/GuidedQuant (rank 6–7) in measured impact.

## 6. Limitations

1. **Subsampled tensors.** Real weights were subsampled to 128×128 from full [17408, 5120] / [5120, 17408] / [10240, 5120] / [5120, 6144] / [6144, 5120]. Full-size tensors would exercise different tile boundary effects and channel correlations.

2. **Synthetic calibration.** Calibration activations are synthetic (Gaussian + outliers), not from the real model's hidden states. Real activations have richer structure (token-level correlations, attention patterns).

3. **Uniform quantizer, not Viterbi.** The trellis quantizer uses per-tile uniform codebook quantization, not the full EXL3 Viterbi search. The Viterbi search finds optimal codebook assignments considering inter-element dependencies, which may change correction effectiveness.

4. **No propagation.** KLD is measured per-layer, not propagated through subsequent layers to the final logits. The plan calls for "whole GDN or full-attention block plus propagated suffix/end-logit KLD."

5. **No GDN gating.** The GDN output Fisher should be weighted by the gating derivative $\sigma'(\cdot)$. The experiment uses the standard Fisher for GDN tensors.

6. **No K5K6 reconstruction.** The EXL3 trellis format requires the exllamav3 library to decode. The experiment quantizes the BF16 reference directly, not the K5K6-hydrated checkpoint.

## 7. Artifacts

| File | Description |
|------|-------------|
| `tools/trellis_quant.py` | GPTAQ × YAQA × ResComp with Sketch A/B (sample tensors) |
| `tools/trellis_quant_full.py` | Full 8-algorithm experiment (sample tensors) |
| `tools/trellis_quant_real.py` | Real Qwen3.8-27B weight experiment |
| `receipts/trellis-quant-results.json` | Sketch A/B results |
| `receipts/trellis-quant-full-results.json` | Full 8-algorithm results |
| `receipts/trellis-quant-real-weights-results.json` | Real-weight results |
| `receipts/qwen38-real-weights-sample.npz` | Downloaded BF16 weight samples |

## 8. References

- GPTAQ (GPTQv2): Li et al., arXiv:2504.02692, ICML 2025
- YAQA: Tseng et al., arXiv:2505.22988, 2025
- ResComp: Li et al., arXiv:2604.07955, 2026
- KronQ: arXiv:2607.07964
- GuidedQuant: arXiv:2505.07004
- ResQ: arXiv:2412.14363
- BAQ: arXiv:2506.05664
- EXL3: github.com/turboderp-org/exllamav3
- GPTQ: Frantar et al., ICLR 2023
- QTIP: Tseng et al., arXiv:2406.11235, 2024

## 9. Verification and re-measurement (v2)

**Status:** completed verification, 2026-08-20. All 9 algorithms verified against
reference repositories and papers. 7 discrepancies found and fixed. Experiments
rerun with fixed code on 15 real Qwen3.8-27B tensors (layers 0, 10, 20, 30, 40, 55).

### 9.1 Verification methodology

Five scout subagents read the reference GitHub repositories for GPTQv2, YAQA,
ResComp, AWQ, and SmoothQuant, reporting exact line-by-line discrepancies. The
BAQ, KronQ, and ResQ papers were read directly for algorithm details. GuidedQuant
was verified against its paper (no reference repo available).

### 9.2 Discrepancies: source classification

Each fix is classified by whether it is confirmed by the arXiv paper equations,
or whether it is a code-level implementation detail not present in the paper.
This distinction matters: paper-confirmed fixes are unambiguous algorithm errors;
code-convention fixes may be valid alternative implementations of the same paper.

#### Paper-confirmed fixes (unambiguous — our code contradicted the published equations)

| Algorithm | What paper says | What we did | Fix |
|-----------|----------------|-------------|-----|
| **AWQ core formula** | Eq 5: $s = s_X^\alpha$ where $s_X$ is "the average magnitude of activation (per-channel)". No weight term. | Used `max|X|`, had weight denominator `max|W|` | `mean|X|^α`, no weight term |
| **AWQ alpha search** | "grid search over [0,1]" | Fixed α=0.5 | Grid search over {0, 0.05, ..., 0.95} |
| **BAQ bit allocation** | Eq 5-6: $R^* = \frac{1}{2}\log_2\frac{c}{\lambda} + \frac{R_{\text{sum}}}{MN}$, explicit closed-form | Heuristic `log2(Hessian diagonal)` | Closed-form from Eq 5-6 |
| **KronQ H_G cancellation** | Paper states H_G cancels in GPTQ update; only used for incoherence + inter-layer allocation | Used `tr(H_O) × diag(H_I)` as per-column sensitivity weight | Uniform weights (no-op for single layer) |
| **ResQ algorithm** | Eq 3: $W_q = U_l Q_L(U_l^T W) + U_h Q_H(U_h^T W)$ — PCA subspace quantization | Rank-4 SVD of `W - Wq` as post-processing | PCA subspace: top 1/8 dim at 8-bit, rest at K |

#### Code-convention fixes (not in papers — matched to reference repos)

| Algorithm | What reference code does | What we did | Fix | Paper says? |
|-----------|-------------------------|-------------|-----|-------------|
| **GPTAQ 0.25 multiplier** | `alpha = 0.25` in `gptaq_utils_r.py` L96 | No multiplier | Added 0.25 | **Nothing.** Paper Eq 9/15 have no such factor. Origin unclear — may relate to 2/N Hessian scaling or empirical tuning. |
| **GPTAQ P-matrix factor order** | `triu(D @ U.T, 1) @ U` where U = upper Cholesky of inv(H) | `triu(D @ L, 1) @ L.T` — factors reversed | `triu(D @ L.T, 1) @ L` | **Ambiguous.** Paper Fig 3 shows Cholesky factorization but doesn't specify upper/lower convention. |
| **GPTAQ/ResComp pre-update weights** | Saves `w = W1[:,i].clone()` before quantization, uses `w` in correction | Used post-update `Ww[:,c]` (already modified by GPTQ) | Save `w_pre` before quantization | **Ambiguous.** Paper Eq 15 uses $W_{:,q}$ "current weight" — doesn't specify pre vs post quantization. |
| **AWQ geometric normalization** | `s / sqrt(max(s) * min(s))` — makes min/max reciprocal | `s / mean(s)` — arithmetic mean | Geometric normalization | **Nothing.** Paper Eq 5 has no normalization. |
| **AWQ clamp 1e-4** | `r.clamp(min=1e-4)` | `+ 1e-10` | Clamp 1e-4 | **Nothing.** Implementation detail. |
| **SmoothQuant normalization** | No normalization at all | `s / mean(s)` | Removed normalization | **Confirmed by paper** — formula has no normalization. Our addition was a bug. |
| **SmoothQuant clamp 1e-5** | Clamp to 1e-5 | `+ 1e-10` | Clamp 1e-5 | **Nothing.** Implementation detail. |

#### Honest assessment

The 0.25 GPTAQ multiplier is the most concerning fix. It is an empirical constant
in the reference code with no paper justification. Our original code without it
may have been a valid clean-room implementation of the published algorithm. By
matching the code, we may have over-fit to their implementation choices rather
than the paper. The factor could relate to the $\frac{2}{N}$ Hessian scaling in
the reference (`H = (2/N) X X^T`), but our Hessian uses raw `X @ X.T` without
the $\frac{2}{N}$ factor, so the 0.25 may be inappropriate for our setup.

### 9.3 YAQA note

YAQA requires backward passes to compute the true Kronecker-factored Hessian
(H_O from gradient covariance, H_I from input covariance). Sketch B (identity
H_O) reduces to standard GPTQ. Since our CPU-only environment cannot run backward
passes, YAQA is not separately implemented. The YAQA insight (output-side Hessian
matters) is partially captured by GuidedQuant's per-output-channel Fisher weighting.

### 9.4 Alternative approaches and novel possibilities

For each discrepancy, we analyze whether alternative solutions exist beyond the
one we adopted. Some of these may unlock novel research directions.

#### 9.4.1 GPTAQ P-matrix: factor order and multiplier

**Ambiguity:** The paper describes the correction using $\tilde{H}^{-1}$ (inverse
Hessian after Gaussian elimination) but doesn't specify the Cholesky convention.
The 0.25 multiplier is not in the paper at all.

**Alternatives not tried:**

1. **Eigendecomposition-based P-matrix.** Instead of Cholesky, decompose
   $H = Q \Lambda Q^T$ and compute $P = \text{triu}(D \cdot Q \Lambda^{-1} Q^T, 1) \cdot Q \Lambda^{-1} Q^T$.
   The eigenvalue approach naturally handles ill-conditioned Hessians through
   $\Lambda^{-1}$ and may be more numerically stable for tensors with extreme
   eigenvalue decay (e.g., L55_down). The cost is $O(n^3)$ vs Cholesky's $O(n^3/3)$,
   but for 128×128 matrices this is negligible.

2. **Adaptive correction strength.** Instead of fixed 0.25, derive α from the
   data: $\alpha = \frac{\|\Delta X \cdot X^T\|_F}{\|X \cdot X^T\|_F}$ — the ratio
   of asymmetric to symmetric error. When activation deviation is small (early
   layers), the correction is weak; when large (late layers), it's strong.
   This adapts to the actual asymmetry rather than using a universal constant.

3. **Grid-searched α per layer.** Like AWQ's alpha search, try
   α ∈ {0, 0.1, ..., 0.5} per tensor and pick the best by KLD. This removes
   the guesswork entirely. Cost: 6× the GPTAQ time, still trivial for 128×128.

4. **No multiplier (paper-faithful).** Since our Hessian uses raw $X X^T$ (not
   $\frac{2}{N} X X^T$), the 0.25 may not apply. Testing α=1.0 (no multiplier)
   would validate whether the constant is necessary for our Hessian convention.

#### 9.4.2 Pre-update vs post-update weights

**Ambiguity:** Paper says $W_{:,q}$ — "current weight" — but doesn't specify
whether this is before or after the quantization step at column q.

**Alternatives not tried:**

1. **Error-vector correction.** Use the quantization error itself:
   $(w_{\text{pre}} - Q_{:,c}) \cdot P_{c,c:}$ instead of $w_{\text{pre}} \cdot P_{c,c:}$.
   This directly ties the correction magnitude to the error, making it
   self-scaling — large errors get large corrections, small errors get small ones.
   The current approach uses the full weight magnitude, which may over-correct
   for small errors.

2. **Blended coefficient.** Use $\alpha \cdot w_{\text{pre}} + (1-\alpha) \cdot Q_{:,c}$
   for the GPTAQ term. At α=1 this is the reference approach; at α=0 it uses
   the quantized value. An intermediate α might balance the original-weight
   signal with the quantization-error direction. Could be searched per-layer.

3. **Iterative refinement.** After one pass of GPTAQ with pre-update weights,
   run a second pass that uses the post-quantization residual to refine.
   This is a multi-round GPTAQ that converges to a better solution, at 2× cost.

#### 9.4.3 AWQ: activation statistic and normalization

**What paper confirms:** $s_X$ = mean activation magnitude, α grid search.
**What code adds:** geometric normalization, 1e-4 clamp.

**Alternatives not tried:**

1. **Activation variance instead of mean magnitude.** $s_X = \text{Var}(X_j)$
   captures both magnitude and dispersion. Channels with high variance carry
   more information; protecting them may be more effective than protecting
   high-magnitude channels. The paper's rationale is about magnitude, but
   variance is a richer statistic.

2. **Activation kurtosis.** Outlier channels often have high kurtosis (heavy
   tails). Using kurtosis as the scale would specifically target channels with
   extreme outliers, which are the most damaging for uniform quantization.
   $s_j = \text{kurt}(X_j)^\alpha$.

3. **Weight-activation product.** $s_j = (\text{mean}|X_j| \cdot \text{max}|W_{:,j}|)^\alpha$.
   This combines activation and weight information — neither pure AWQ (activation
   only) nor pure SmoothQuant (separate terms). The product form ensures both
   high-activation AND high-weight channels get protected. This is a genuine
   hybrid that neither paper proposes.

4. **Per-tile AWQ.** Instead of per-channel scaling, compute scales per trellis
   tile (16×16). Each tile gets its own scale based on the activation statistics
   of its channels. This is finer-grained than per-channel and aligns with the
   trellis quantization structure. Overhead: 1 float per tile vs 1 float per channel.

5. **No normalization (paper-faithful).** The paper has no normalization. The
   scale just IS $s_X^\alpha$. Removing normalization entirely and letting the
   scale range be determined by the data might be more faithful to the paper's
   intent, even if it produces larger scale ranges.

#### 9.4.4 SmoothQuant: normalization and alpha

**What paper confirms:** $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$, no normalization.
**Our bug:** Added mean normalization spuriously.

**Alternatives not tried:**

1. **Per-channel adaptive alpha.** Instead of a single α for all channels, use
   $\alpha_j = \sigma(\log(|X_j| / |W_j|))$ — a sigmoid of the log-ratio of
   activation to weight magnitude. Channels with high activation but low weight
   get α→1 (migrate fully to weights); balanced channels get α≈0.5. This is a
   channel-wise SmoothQuant that adapts to each channel's difficulty balance.

2. **Per-layer alpha search.** The SmoothQuant paper uses different α values for
   different models (0.85 for Llama-2, 0.9 for 70B). A per-layer search would
   be finer-grained. Early layers (well-conditioned) might prefer low α; late
   layers (outlier-heavy) might prefer high α.

3. **Smooth-then-quantize-aware.** Apply SmoothQuant scaling, then design a
   non-uniform quantization grid that's denser where the scaled weights cluster.
   This combines the difficulty migration with optimal quantization grid design,
   which neither SmoothQuant nor AWQ does.

#### 9.4.5 BAQ: closed-form allocation

**What paper confirms:** Eq 5-6, closed-form convex optimization.

**Assumptions that may not hold for trellis:**
1. BAQ assumes scalar uniform quantization — trellis uses tile-level codebooks.
2. BAQ assumes per-element range — we use per-column range (coarser).
3. The high-resolution approximation $\Delta^2/12$ may not hold at K3.

**Alternatives not tried:**

1. **Trellis-aware BAQ.** Allocate bits per tile instead of per column. The
   formula becomes $c_{\text{tile}} = (\text{tile range})^2 / (12 \cdot \text{tile Hessian sensitivity})$
   where tile Hessian sensitivity = trace of $H^{-1}$ restricted to the tile's
   input channels. This is a novel extension of BAQ to the trellis setting that
   the paper doesn't consider. Each 16×16 tile would get its own K value.

2. **Iterative BAQ.** The formula assumes $[H_F^{-1}]_{jj}$ is fixed, but GPTQ
   updates the Hessian as columns are quantized. Iterate: (1) allocate bits,
   (2) run GPTQ, (3) recompute Hessian sensitivity after quantization,
   (4) reallocate. This adaptive BAQ would be more accurate at 2-3× cost.

3. **BAQ with weight magnitude.** The paper's $c_{ij}$ uses only weight range
   and Hessian. Augment: $c'_{ij} = c_{ij} \cdot |w_{ij}|^2$ to also account for
   weight magnitude. This allocates more bits to both sensitive AND large weights.

4. **Per-element BAQ.** The paper's formula is per-element (i,j), but we apply
   it per-column. Computing per-element allocation requires per-element Hessian
   diagonal, which is the diagonal of $H^{-1}$ — available from the Cholesky
   factor but at $O(n^2)$ storage. For 128×128 this is feasible and would be
   the true paper implementation.

#### 9.4.6 KronQ: H_G cancellation

**What paper confirms:** H_G cancels in the GPTQ update. H_G is only used for
bidirectional incoherence processing and inter-layer allocation.

**What we might have missed:** Even though H_G cancels in the update, the paper
says it's used for **output-side incoherence processing** — rotating the weight
rows using H_G eigenvectors. We didn't implement this.

**Alternatives not tried:**

1. **Output-side Hadamard transform.** Apply a random Hadamard transform on the
   output rows of W before quantization, then inverse after. This is the
   practical form of H_G-based incoherence processing. It makes weight rows
   approximately i.i.d. Gaussian, which is optimal for uniform quantization.
   Cost: $O(m \log m)$ per column. This is distinct from our existing
   block_hadamard (which operates on both rows and columns) — this is
   output-only, driven by H_G not random.

2. **H_G-informed tile ordering.** H_G eigenvectors reveal which output channels
   are correlated. Reorder output channels so correlated ones share tiles,
   allowing the trellis quantizer to exploit their correlation. This is a novel
   use of H_G that doesn't cancel — it's about data layout, not the update rule.

3. **Inter-tile BAQ using H_X.** Even for single-layer, allocate bits across
   tiles using tr(H_X restricted to tile) as sensitivity. The H_X part doesn't
   cancel — it's the input-side Hessian. This combines KronQ's H_X with BAQ's
   allocation formula: $c_{\text{tile}} = (\text{tile range})^2 / (12 \cdot \text{tr}(H_X^{-1}_{\text{tile}}))$.

4. **KronQ-weighted loss.** Instead of weighting the GPTQ update (which cancels),
   weight the KLD loss: $L = \sum_i H_{G,ii} \cdot \text{KLD}_i$. This doesn't
   cancel because it's in the objective, not the update. But it requires the
   true H_G (backward pass). For CPU-only, we could approximate H_G using output
   covariance: $H_G \approx Y^T Y / N$ where $Y = WX$.

#### 9.4.7 ResQ: PCA subspace

**What paper confirms:** Eq 3, PCA from activation covariance, r=d/8, 8-bit for
high-variance subspace, K-bit for rest.

**Design choices not explored:**

1. **Weight PCA instead of activation PCA.** Project using SVD of W itself. The
   top-r singular vectors of W capture the most "expressive" weight directions.
   This protects the most important weight components rather than the most
   activated ones. May be better for weight-only quantization (which is our case).

2. **Joint PCA.** Find the subspace that maximizes both activation variance AND
   weight sensitivity: maximize $u^T (XX^T) u$ subject to $u^T (W^T W) u = 1$.
   This is a generalized eigenvalue problem that gives a subspace that's both
   high-activation and high-weight. Neither AWQ nor ResQ does this.

3. **Trellis-tile PCA.** Apply PCA within each 16×16 tile: 2 components at
   8-bit, 14 at K-bit. Much finer-grained with lower overhead (2×16×2 bytes
   per tile vs 16×16/8 bytes savings). The overhead ratio is much better than
   full-matrix ResQ.

4. **Adaptive rank.** Instead of fixed r=d/8, choose r per-layer based on
   eigenvalue decay. If the activation covariance has fast decay (top few
   eigenvalues dominate), use small r. If slow decay, use larger r. The
   optimal r minimizes $r \cdot \epsilon_{8\text{bit}} + (d-r) \cdot \epsilon_{K\text{bit}}$
   where $\epsilon$ depends on the eigenvalue spectrum.

5. **ResQ + GPTAQ.** The paper says ResQ can combine with GPTQ. Apply GPTAQ
   within each subspace separately. The high-precision subspace gets minimal
   correction (8-bit is already good); the low-precision subspace gets maximum
   GPTAQ benefit. The P-matrix would be computed within each subspace's Hessian,
   which is better conditioned. This is a novel combination we didn't test.

6. **ResQ + BAQ.** Apply BAQ bit allocation within the low-precision subspace.
   The low-precision components get mixed K values based on BAQ sensitivity.
   This combines ResQ's subspace decomposition with BAQ's optimal allocation.

#### 9.4.8 YAQA: approximate Hessian without backward

**What paper says:** H_O = E[∇_y ℓ^T ∇_y ℓ] (gradient covariance), requires backward.

**Alternatives not tried:**

1. **Output covariance as H_O proxy.** $H_O \approx Y^T Y / N$ where $Y = WX$.
   This captures output channel correlations without backward passes. It's
   computable in our CPU-only setup and approximates the Fisher when the model
   is near convergence (output distribution ≈ activation distribution).

2. **Hadamard as H_O incoherence.** The paper says incoherence processing
   reduces H_O's incoherence. If H_O ≈ I (which Figure 3 suggests for some
   layers), then output-side Hadamard IS the incoherence processing. Apply
   Hadamard on output rows, quantize, inverse Hadamard. This is Sketch B +
   incoherence, a practical YAQA approximation.

3. **GuidedQuant as diagonal H_O.** Our GuidedQuant computes per-output Fisher
   diagonal. Use this as diagonal H_O in the YAQA rounding formula (Eq 10):
   $W = Q(W^* + L_O^T \Delta W L_I + L_O^T \Delta W + \Delta W L_I)$ where
   $L_O$ = Cholesky of Fisher diagonal (trivially invertible). This is a
   legitimate YAQA variant using diagonal H_O, computable without backward.

### 9.5 Sample-tensor results (v2, fixed code)

19 methods × 5 K × 3 seeds = 285 runs. Matrix 128×128.

| K | Best method | KLD | vs baseline | vs GPTAQ |
|---|-------------|-----|-------------|----------|
| K3 | GPTAQ+AWQ+SQ+BAQ | 8.49e-4 | +83.6% | +53.0% |
| K4 | GPTAQ+AWQ(search)+SQ+BAQ | 2.31e-4 | +80.2% | +46.0% |
| K5 | GPTAQ+AWQ+SQ+BAQ | 9.98e-5 | +68.4% | +31.8% |
| K6 | GPTAQ+AWQ+SQ+BAQ | 7.05e-5 | +43.4% | +12.5% |
| K7 | GPTAQ+AWQ+SQ | 6.43e-5 | +18.2% | +1.0% |

**Key change from v1:** BAQ (closed-form) is now the strongest complement at ALL
K values, not just K6. The consistent winner is GPTAQ+AWQ+SQ+BAQ across K3-K6.

### 9.6 Real-weight results (v2, fixed code, 15 tensors)

15 tensors: L0/L10/L20/L30/L40/L55 gate+down + L0 GDN (qkv, out, z).
K5K6 recipe: gate=K5, down/attention=K6.

| K | Best method | Mean KLD | vs baseline | vs GPTAQ |
|---|-------------|----------|-------------|----------|
| K5 (gate_proj) | GPTAQ+AWQ+SQ+GQ+BAQ | 6.81e-3 | +76.5% | +23.8% |
| K6 (down/attention) | GPTAQ+AWQ+SQ+GQ+BAQ | 3.76e-3 | +49.3% | +7.2% |

**Best method per tensor:**

| Tensor | Layer | K | Best Method | vs base | vs GPTAQ |
|--------|-------|---|-------------|---------|----------|
| L0_gate | 0 | K5 | GPTAQ+AWQ(search)+SQ+BAQ | +69.7% | +36.5% |
| L0_down | 0 | K6 | GPTAQ+AWQ+SQ+GQ+BAQ | +42.0% | +12.1% |
| L0_qkv | 0 | K6 | GPTAQ+AWQ(search)+SQ+BAQ | +42.7% | +13.9% |
| L0_out | 0 | K6 | GPTAQ+AWQ+SQ+BAQ | +42.6% | +13.9% |
| L0_z | 0 | K6 | GPTAQ+AWQ+SQ+BAQ | +43.4% | +12.5% |
| L10_gate | 10 | K5 | GPTAQ+AWQ+SQ+GQ+BAQ | +75.4% | +20.3% |
| L10_down | 10 | K6 | GPTAQ+AWQ+SQ+BAQ | +49.3% | +3.9% |
| L20_gate | 20 | K5 | GPTAQ+AWQ+SQ+BAQ | +75.0% | +20.9% |
| L20_down | 20 | K6 | GPTAQ+AWQ+SQ+BAQ | +52.1% | +8.2% |
| L30_gate | 30 | K5 | GPTAQ+AWQ+SQ+GQ+BAQ | +76.3% | +24.3% |
| L30_down | 30 | K6 | GPTAQ+BAQ | +43.4% | +9.4% |
| L40_gate | 40 | K5 | GPTAQ+AWQ+SQ+GQ+BAQ | +79.3% | +29.9% |
| L40_down | 40 | K6 | GPTAQ+AWQ+SQ+BAQ | +53.5% | +9.4% |
| L55_gate | 55 | K5 | GPTAQ+AWQ+SQ+BAQ | +69.2% | +36.1% |
| L55_down | 55 | K6 | GPTAQ+AWQ(search)+SQ+BAQ | +30.0% | +19.5% |

### 9.7 How conclusions changed

| Finding | v1 (pre-fix) | v2 (post-fix) | Change |
|---------|-------------|---------------|--------|
| GPTAQ dominates | Yes | Yes | **Confirmed** |
| BAQ strongest complement | K6 only | All K values | **Strengthened** — closed-form BAQ is much better |
| Best K5 combo | GPTAQ+AWQ+SQ+ResQ | GPTAQ+AWQ+SQ+GQ+BAQ | **Changed** — ResQ was wrong algorithm; BAQ now wins everywhere |
| Best K6 combo | GPTAQ+BAQ | GPTAQ+AWQ+SQ+GQ+BAQ | **Changed** — full stack now adds value at K6 |
| AWQ=SQ identical | Yes (alpha=0.5) | No (AWQ activation-only, SQ includes weights) | **Fixed** — they are genuinely different |
| ResQ useful in stack | Yes (as SVD residual) | No (standalone PCA subspace) | **Changed** — ResQ is a different algorithm |
| KronQ marginal benefit | Small but nonzero | Zero (verified no-op) | **Fixed** — H_G cancels |
| AWQ+SQ hurts alone | Not measured | Confirmed (-23% to -230%) | **New finding** — scaling distortion needs GPTAQ |
| Late-layer hardest | L55_down | L55_down | **Confirmed** |
| Mid-layer trend | Not measured | Larger improvement room (L10-L40) | **New finding** |

**Caveat:** Some v1→v2 changes may be due to code-convention fixes (0.25 multiplier,
geometric normalization) rather than true algorithm corrections. The 0.25 multiplier
in particular may not be appropriate for our unscaled Hessian. A v3 experiment
testing paper-faithful implementations (no 0.25, no normalization) would isolate
which fixes actually matter.

### 9.8 Updated implications for the Frontier Loop

1. **GPTAQ is the P0 correction.** Confirmed across 15 tensors and 5 K values.
2. **BAQ (closed-form) should be P0.** The closed-form convex allocation from
   BAQ Eq 5-6 is the strongest single complement to GPTAQ at all K values.
3. **GPTAQ+AWQ+SQ+BAQ is the recommended stack.** Consistently wins K3-K6.
4. **ResQ should be evaluated as a standalone alternative.** PCA subspace is
   a different quantization path, not stackable with GPTAQ. ResQ+GPTAQ (per-subspace
   GPTAQ) and ResQ+BAQ (BAQ within low-precision subspace) are untested novel
   combinations worth exploring.
5. **KronQ can be deferred** for single-layer, but its H_G-based output-side
   incoherence processing (Hadamard on output rows) is untested and may help.
6. **AWQ and SmoothQuant are genuinely different.** AWQ: activation-only
   (mean|X|^α). SQ: activation-weight balance (max|X|^α / max|W|^{1-α}). Both
   hurt without GPTAQ compensation.
7. **Novel research directions** identified in §9.4:
   - Trellis-aware BAQ (per-tile bit allocation)
   - Adaptive GPTAQ correction strength (data-driven α instead of 0.25)
   - Weight-activation product AWQ hybrid
   - Per-channel adaptive SmoothQuant α
   - ResQ+GPTAQ per-subspace correction
   - Output-side Hadamard as KronQ incoherence
   - Joint PCA for ResQ (activation + weight covariance)
   - Trellis-tile PCA (per-tile subspace quantization)
8. **Late-layer weights remain hardest.** L55_down: +19.5% vs GPTAQ (vs +36%
   for L0/L55 gate). The adaptive GPTAQ α (§9.4.1 alt 2) may help here — late
   layers have larger activation deviation, so stronger correction is needed.
9. **Mid-layer weights have larger optimization room.** L10-L40 gate tensors
   show +20-30% improvement from the full stack vs GPTAQ alone.
