# R18-BlockPropagation: Block-Level Error Propagation Through MLP and Attention

**Status:** revised after two adversarial review rounds. Implementation confirmed correct (GQA einsum, V/O rotation, synthetic weights, distinct up). Interpretation revised per reviewer v2.

## 1. Executive summary

Block-level error propagation through MLP and attention blocks measures how per-tensor quantization error translates to block output error. Key findings:

1. **MLP block output error is much smaller than the sum of individual weight errors** (output/weight ratio 0.13 for L0, 0.32–0.38 for L55 at batch=64). This ratio is setup-specific (depends on input scale, batch size, matrix dimensions) and cannot be interpreted as intrinsic attenuation. For a same-space additivity measure, see Exp 6.

2. **Attention block output error (synthetic weights) is also smaller than sum of weight errors** (ratio ~0.03 for both single-head and GQA in this synthetic draw). The initially reported 7.7× GQA amplification was a bug (broken einsum). The GQA vs single-head difference (0.035 vs 0.026) is from one independent weight draw and cannot be attributed to the GQA architecture.

3. **Rotation benefits persist at block output for MLP**: Hadamard rotation reduces mean block output error by +6–8% for L0, +55–66% for L55 (5-seed mean). L55 benefits more due to down_proj outlier structure. BiIP+Hadamard is comparable to Hadamard-only. For synthetic attention, V/O rotation has mixed effects across K (+11% K3, −15% K4, −1.2% K5, +4.8% K6).

4. **Joint greedy allocation helps synthetic attention (+36–51% at K3–K5)** but not real MLP (0% at L0, −3% to −7% at L55 — greedy overfits calibration). The synthetic attention benefit comes from giving more bits to V/O.

5. **SiLU operates in linear regime** (finite-difference/Jacobian ratio ≈ 1.0, first-order Taylor is accurate). Mean SiLU derivative ≈ 0.50. The SiLU/identity block-error ratio (0.27–0.29) compares different models (SiLU(g)·u vs g·u), not an intrinsic damping factor. Softmax empirical squared gain ≈ 0.00024 in a near-uniform attention regime (entropy 4.156 ≈ ln(64)=4.159).

6. **Cross-matrix interaction is low (0.3–11%)** with distinct up_proj. Output-space additivity ratio is 0.95–1.13 — nearly additive. This ratio is same-space but input-distribution dependent.

## 2. Experimental setup

- **MLP weights**: 128×128 slices of real Qwen3.8-27B L0 and L55 (gate, down). up_proj uses a distinct 128×128 slice of gate (columns 128:256) as a statistics-matched proxy — NOT an exact copy. Real up_proj is not in the archive.
- **Attention weights**: SYNTHETIC (single Gaussian draw, std matched to L0 gate slice — not attention statistics). The archive's L0_qkv/L0_out are GDN projections, not standard softmax attention.
- **MLP block**: gate [128,128], up [128,128], down [128,128]. Forward: y = down(SiLU(gate(x)) ⊙ up(x))
- **Attention block**: single-head (Q,K,V,O all [128,128]) and GQA (Q [384,128], K/V [128,128], O [128,384], 3:1 ratio). Each token attends to ALL tokens (proper seq-to-seq attention).
- **Quantizer**: per-column uniform RTN, K ∈ {3,4,5,6}
- **Input**: synthetic Gaussian + 5% outlier channels, batch/seq=64, 5 seeds (inputs varied, weights fixed)
- **Metrics**: block error = ||y_fp − y_quant||², individual weight error = ||W − Wq||², output/weight ratio (setup-specific, not a threshold), additivity ratio = all_quantized_block_error / sum(single_matrix_block_errors) (same-space, input-dependent)
- **No confidence intervals**: N=5 seeds with point estimates only. Paired per-seed differences not reported.

## 3. Detailed findings

### 3.1 MLP block error propagation (Exp 1) — all K values

| Layer | K | Block Error | Sum Indiv. Weight Err | Output/Weight Ratio |
|-------|---|-------------|----------------------|---------------------|
| L0 | K3 | 0.03682 | 0.26271 | 0.140 |
| L0 | K4 | 0.00769 | 0.05715 | 0.135 |
| L0 | K5 | 0.00182 | 0.01334 | 0.136 |
| L0 | K6 | 0.000427 | 0.00321 | 0.133 |
| L55 | K3 | 0.42401 | 1.11901 | 0.379 |
| L55 | K4 | 0.07149 | 0.19755 | 0.362 |
| L55 | K5 | 0.01542 | 0.04767 | 0.324 |
| L55 | K6 | 0.00382 | 0.01139 | 0.335 |

The output/weight ratio is stable across K within a layer but changes with input scale (reviewer confirmed: scaling X by 0.1× moves L0 K5 ratio to 1.25e-5; scaling by 10× moves it to 2444). It is a descriptive setup-specific quantity, not an intrinsic block property.

L55 individual weight errors are dominated by down_proj (75% of total at K5), consistent with the K5K6 recipe (down=K6, gate=K5).

### 3.2 Attention block error propagation (Exp 2, synthetic weights) — all K values

| Variant | K | Block Error | Sum Indiv. Weight Err | Output/Weight Ratio |
|---------|---|-------------|----------------------|---------------------|
| Single-head | K3 | 0.00880 | 0.30963 | 0.028 |
| Single-head | K4 | 0.00162 | 0.06734 | 0.024 |
| Single-head | K5 | 0.000404 | 0.01571 | 0.026 |
| Single-head | K6 | 0.000103 | 0.00383 | 0.027 |
| GQA | K3 | 0.02192 | 0.69051 | 0.032 |
| GQA | K4 | 0.00486 | 0.15012 | 0.032 |
| GQA | K5 | 0.00122 | 0.03510 | 0.035 |
| GQA | K6 | 0.000305 | 0.00851 | 0.036 |

The GQA vs single-head comparison uses independent random weight draws, so the difference (0.035 vs 0.026) cannot be attributed to the GQA architecture. The corrected implementation proves the original 7.7× was an einsum artifact, and this synthetic draw produces the values above.

### 3.3 Error propagation with rotation (Exp 3) — all K values

| Block | K | No Rot | Hadamard | Had Imp. | BiIP+Had | BiIP Imp. |
|-------|---|--------|----------|----------|-----------|-----------|
| MLP L0 | K3 | 0.03682 | 0.03417 | +7.2% | 0.03601 | +2.2% |
| MLP L0 | K4 | 0.00769 | 0.00713 | +7.3% | 0.00753 | +2.2% |
| MLP L0 | K5 | 0.00182 | 0.00168 | +7.6% | 0.00175 | +3.8% |
| MLP L0 | K6 | 0.000427 | 0.000401 | +6.0% | 0.000416 | +2.5% |
| MLP L55 | K3 | 0.42401 | 0.14573 | +65.6% | 0.14482 | +65.8% |
| MLP L55 | K4 | 0.07149 | 0.02732 | +61.8% | 0.03115 | +56.4% |
| MLP L55 | K5 | 0.01542 | 0.00672 | +56.4% | 0.00712 | +53.8% |
| MLP L55 | K6 | 0.00382 | 0.00173 | +54.6% | 0.00176 | +54.0% |

| Block | K | No Rot | VO Rot | VO Imp. | All Had | All Had Imp. |
|-------|---|--------|--------|---------|---------|-------------|
| Attn(synth) | K3 | 0.00880 | 0.00783 | +11.0% | 0.00783 | +11.0% |
| Attn(synth) | K4 | 0.00162 | 0.00186 | −15.1% | 0.00187 | −15.5% |
| Attn(synth) | K5 | 0.000404 | 0.000409 | −1.2% | 0.000420 | −3.9% |
| Attn(synth) | K6 | 0.000103 | 0.000099 | +4.8% | 0.000097 | +6.3% |

**Finding**: Rotation reduces mean block output error for MLP (L0: +6–8%, L55: +55–66%). L55 is consistently positive across seeds; L0 is modest and less robust seed-by-seed (reviewer found L0 K5 paired gains of +20.7%, +11.1%, +6.9%, −7.6%, +3.3%). For synthetic attention, V/O rotation has mixed effects across K — no consistent benefit. "Persists at block output" means the mean block-level error decreases after jointly rotating matrices; it does not prove the benefit compounds relative to individual-matrix rotation.

### 3.4 Joint block bit allocation (Exp 4, greedy marginal) — all K values

| Block | K | Uniform | Greedy | Improvement | Allocation |
|-------|---|---------|--------|------------|------------|
| MLP L0 | K3 | 0.03682 | 0.03682 | 0.0% | (3,3,3) |
| MLP L0 | K4 | 0.00769 | 0.00769 | 0.0% | (4,4,4) |
| MLP L0 | K5 | 0.00182 | 0.00182 | 0.0% | (5,5,5) |
| MLP L0 | K6 | 0.000427 | 0.000427 | 0.0% | (6,6,6) |
| MLP L55 | K3 | 0.42401 | 0.45549 | −7.4% | (2,3,4) or (3,2,4) or (3,3,3) |
| MLP L55 | K4 | 0.07149 | 0.07357 | −2.9% | (4,4,4) or (4,3,5) |
| MLP L55 | K5 | 0.01542 | 0.01631 | −5.7% | (5,5,5) or (5,4,6) |
| MLP L55 | K6 | 0.00382 | 0.00382 | 0.0% | (6,6,6) |
| Attn(synth) | K3 | 0.00880 | 0.00493 | +43.9% | (2,2,4,4) |
| Attn(synth) | K4 | 0.00162 | 0.00104 | +35.8% | (3,3,5,5) |
| Attn(synth) | K5 | 0.000404 | 0.000197 | +51.2% | (4,4,6,6) |
| Attn(synth) | K6 | 0.000103 | 0.000103 | 0.0% | (6,6,6,6) |

**Finding**: Greedy allocation helps synthetic attention (+36–51% at K3–K5) by giving more bits to V/O. It does not help real MLP: L0 stays uniform, L55 gets worse (−3% to −7%) because the greedy search overfits calibration data. At K6, everything is already at max K.

Note: For K5 MLP, exhaustive search over 10 budget-feasible allocations (from the 125-point grid) confirmed greedy matches the calibration optimum for both layers on all 5 seeds (reviewer-verified).

### 3.5 Nonlinear error amplification (Exp 5) — all K values

| Nonlinearity | K | FD/Jac Ratio | Nonlin/Linear Ratio | Notes |
|-------------|---|-------------|--------------------|----|
| SiLU (L0) | K3 | 1.0029 | 0.271 | mean SiLU' = 0.498 |
| SiLU (L0) | K4 | 0.9995 | 0.270 | |
| SiLU (L0) | K5 | 1.0001 | 0.276 | |
| SiLU (L0) | K6 | 1.0000 | 0.274 | |
| SiLU (L55) | K3 | 1.0106 | 0.283 | mean SiLU' = 0.502 |
| SiLU (L55) | K4 | 1.0011 | 0.289 | |
| SiLU (L55) | K5 | 0.9989 | 0.291 | |
| SiLU (L55) | K6 | 0.9998 | 0.268 | |
| Softmax | K3 | — | 0.0002 | entropy=4.156, Q-only perturbation |
| Softmax | K4 | — | 0.0002 | |
| Softmax | K5 | — | 0.0002 | |
| Softmax | K6 | — | 0.0002 | |

**Finding**: FD/Jac ≈ 1.0 means the first-order Taylor approximation is accurate for these perturbation magnitudes. Mean SiLU' ≈ 0.50 is expected for symmetric zero-centered pre-activations (SiLU'(x) + SiLU'(−x) = 1). The SiLU/identity ratio (0.27–0.29) compares different models (SiLU(g)·u vs g·u) — it is not an intrinsic damping factor. The softmax gain (0.00024) is specific to Q-only perturbation in a near-uniform, unmasked attention regime with 64 tokens.

### 3.6 Cross-matrix error interaction (Exp 6) — all K values

| Block | K | All | Sum Singles | Interaction % | Additivity |
|-------|---|-----|-------------|---------------|------------|
| MLP L0 | K3 | 0.03682 | 0.03450 | +6.3% | 1.067 |
| MLP L0 | K4 | 0.00769 | 0.00767 | +0.3% | 1.003 |
| MLP L0 | K5 | 0.00182 | 0.00181 | +0.8% | 1.008 |
| MLP L0 | K6 | 0.000427 | 0.000425 | +0.4% | 1.004 |
| MLP L55 | K3 | 0.42401 | 0.37571 | +11.4% | 1.129 |
| MLP L55 | K4 | 0.07149 | 0.06728 | +5.9% | 1.063 |
| MLP L55 | K5 | 0.01542 | 0.01625 | −5.4% | 0.949 |
| MLP L55 | K6 | 0.00382 | 0.00390 | −2.1% | 0.979 |
| Attn(synth) | K3 | 0.00880 | 0.00800 | +9.1% | 1.100 |
| Attn(synth) | K4 | 0.00162 | 0.00164 | −1.5% | 0.985 |
| Attn(synth) | K5 | 0.000404 | 0.000397 | +1.9% | 1.019 |
| Attn(synth) | K6 | 0.000103 | 0.000097 | +6.2% | 1.066 |

**Finding**: With distinct up_proj, cross-matrix interaction is low (0.3–11%). The additivity ratio is 0.95–1.13 — nearly additive. This ratio is same-space (output vs output) but depends on input distribution and the nonlinear operating point. MLP L55 at K3 has the most interaction (11.4%, superadditive); at K5 it's slightly subadditive (−5.4%).

## 4. Implications for the quantization stack

1. **Rotation persists at block output for MLP** (validated for L55, modest for L0): Hadamard rotation reduces block-level output error, especially for late layers with outlier structure. This supports the recommended stack (rotation before quantization).

2. **Allocation is attention-friendly but MLP-neutral**: Joint allocation helps synthetic attention (V/O need more bits) but doesn't help real MLP with distinct gate/up. The K5K6 recipe (down=K6, gate=K5) is already near-optimal for the MLP block.

3. **Per-tensor weight error and block output error are different quantities**: The output/weight ratio (0.03–0.38 at batch=64) shows they differ in scale, but the ratio is setup-specific and cannot establish "attenuation" or "overestimation" as intrinsic properties.

4. **Late-layer MLP is more sensitive**: L55 has larger block errors, benefits more from rotation (+56% vs +8%), and has more cross-matrix interaction at low K. Inter-layer allocation (R19) should account for this.

5. **GQA bug correction**: The 7.7× GQA amplification was an einsum artifact. With proper attention, the synthetic GQA draw produces output/weight ratios similar to single-head.

## 5. Limitations

- 128×128 slices may not capture full-scale weight statistics
- up_proj is a distinct gate slice (statistics-matched proxy, not real up_proj)
- Attention uses a SINGLE synthetic Gaussian weight draw (std from L0 gate, not attention statistics). No variation across weight draws.
- N=5 input seeds only; no paired CIs or uncertainty intervals
- Synthetic inputs (Gaussian + outliers) may not match real activation distributions
- Attention has no causal mask or positional encoding
- Greedy allocation is heuristic, not dynamic programming (matches exhaustive for K5 MLP: 10 feasible allocations)
- No end-to-end KLD measurement (CPU-only, no model forward pass)
- Output/weight error ratio is setup-specific (input-scale, batch, dimension dependent) — not a threshold for attenuation/amplification
- Additivity ratio is same-space but input-distribution and operating-point dependent — not fully intrinsic
- SiLU/identity and softmax/raw-score ratios compare different models, not intrinsic damping factors
- Softmax gain is specific to Q-only perturbation in near-uniform, unmasked regime

## 6. Code and data

- PoC: `tools/research/r18-block-propagation/poc.py`
- Results: `receipts/research/r18-block-propagation-results.json` (complete K3–K6 for all experiments)
- Regression checks: GQA Q sensitivity, score shape (64,3,64), V/O invariant, Hadamard orthogonality, up≠gate
