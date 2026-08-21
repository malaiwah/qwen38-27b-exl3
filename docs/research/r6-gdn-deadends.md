# R6-GDN v2: Dead Ends (corrected)

## 1. Balanced Realization (corrected transform) — Still hurts, but NOT 30000×

**What was tried:** Compute empirical Gramians, find balanced transform T with eigvals^{0.25} (corrected from v1's sqrt), quantize in balanced basis, inverse-transform.

**v1 bug:** Used eigvals^{0.5} instead of eigvals^{0.25}. This only diagonalized the Gramians without equalizing them, and the transform compressed weights by 100-1000× (condition number 300). Reported 30000× worse.

**v2 corrected:** With eigvals^{0.25}, the transform is verified correct (T W_c T^T = T^{-T} W_o T^{-1} = Σ, off-diagonal < 1e-13, diag diff < 1e-10). Weight std changes from 0.017 to 0.023 (condition number 41). Still loses to GPTQ by 200-300%.

**Why it still hurts:** The balanced transform increases weight dynamic range (std +35%) and creates heterogeneous magnitudes across directions. Per-tile uniform quantization is sensitive to dynamic range within tiles. The balanced basis has MORE heterogeneous magnitudes, not less.

**Verdict:** Not a bug artifact — real finding. Balanced realization improves system-theoretic properties but worsens quantizability. Still a dead end for quantization.

## 2. Gate-Aware GPTQ with Real Sensitivity — No effect

**What was tried:** Weight GPTQ Hessian by σ'(W_z@X)² (gate derivative squared, computed from real weights and activations).

**v1 bug:** Used synthetic z ~ N(0,4) to manufacture gate sensitivity (std=0.187). Real sensitivity from W_z@X has std=0.0005 (essentially uniform).

**Why it doesn't help:** Real GDN gate pre-activations z = W_z @ x are very small (|z| << 1) with real-weight std=0.016 and activation std=0.1. So σ'(z) ≈ 0.25 for all channels — no differentiation. The gates operate far from decision boundaries at this scale.

**Verdict:** DEAD END with synthetic calibration. May work with real model activations where gate dynamics create actual boundary crossings. Requires GPU forward pass to measure.

## 3. QKV Joint Quantization — Tautologically identical

**What was tried:** Quantize merged [W_q; W_k] as one matrix vs separately.

**Why identical:** GPTQ processes columns sequentially. Q and K share the same input X, so the Hessian is identical. Stacking rows doesn't create cross-row coupling in column-sequential processing.

**Verdict:** DEAD END for column-sequential GPTQ. Would need a coupled loss or tile quantizer to exploit Q-K structure.

## 4. z-Gate Mixed-K Allocation — Too aggressive

**What was tried:** Allocate K+1 bits to 20% most sensitive z columns, K-1 to rest.

**v2 corrected:** Now uses real gate sensitivity. Still 4-18× worse than uniform GPTQ because K-1 bits lose too much precision.

**Verdict:** DEAD END for ±1 bit allocation. Gate sensitivity is too uniform to justify such aggressive redistribution.

## 5. Recurrence-Aware GPTQ — Marginal

**What was tried:** Weight Hessian by γ_t · x_t² where γ_t = β_t Π_{s>t} α_s.

**Why marginal:** With synthetic gates all ~0.9 and synthetic inputs with uniform energy, the recurrence weights are nearly uniform. The mathematical formulation is correct but needs real activations to create meaningful differentiation.

**Verdict:** NOT a dead end — formulation is correct, needs real data. Shows 1-2% HW improvement at K4-K5 even with synthetic data.

## v1 Retractions

- "Gate-aware GPTQ gives 11-13% improvement": RETRACTED. Was metric mismatch (standard scored unweighted, gate scored weighted). Same-metric: ±0.3%.
- "Accumulated error +15-24%": RETRACTED. Was single-seed draw. 100-seed evaluation: mean ±0.5%, not statistically significant.
- "Balanced realization 30000× worse": RETRACTED. Was exponent bug (sqrt vs fourth root). Corrected: 200-300× worse (real but less extreme).
- "Gate sensitivity std=0.187": RETRACTED. Was synthetic RNG. Real: std=0.0005.
