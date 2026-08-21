# R6-GDN v3: GDN-Specific Quantization — Final Corrected Findings

## Summary

After three rounds of adversarial review and bug fixes, the honest conclusion is:

**GDN-specific quantization strategies provide no measurable improvement over standard GPTQ on real Qwen3.8-27B GDN weights with synthetic calibration.** The gate sensitivity is too uniform, the balanced realization transform worsens quantizability, and the recurrence structure doesn't create exploitable variation with synthetic data.

## What Was Tested (6 strategies, all correct implementations)

1. **Gate-aware GPTQ (z-only)**: Weight W_z's GPTQ Hessian by SiLU'(W_z@X)². Gate sensitivity std=0.005 (nearly uniform). Result: ±0.1% vs standard GPTQ.
2. **Balanced realization**: Correct fourth-root transform (verified: off-diag < 1e-13, diag diff < 1e-10). Result: 260-440% worse — transform increases weight dynamic range.
3. **Recurrence-aware GPTQ**: Weight Hessian by γ_t·x_t². Result: ±1% (marginal).
4. **QKV joint quantization**: Stack Q+K+V and quantize jointly vs separately. Result: identical (column-sequential GPTQ can't exploit cross-projection structure).
5. **z-gate mixed-K allocation**: K+1 bits for sensitive z columns, K-1 for rest. Result: 260-2227% worse than uniform RTN.
6. **Standard GPTQ baseline**: 14-17% single-step improvement over RTN. The only positive result.

## Key Corrected Results (K4 representative)

| Method | Single-step | Accum (mean) | vs GPTQ |
|--------|------------|--------------|---------|
| RTN | 1.06e-05 | 3.56e-16 | -16.8% ss, +47% accum |
| Standard GPTQ | 9.08e-06 | 6.74e-16 | baseline |
| Gate-aware (z) | 7.28e-06* | 6.74e-16 | +0.07% z-ss, -0.03% accum |
| Balanced realiz. | 3.90e-05 | 3.04e-15 | -330% ss, -351% accum |
| Recurrence-aware | 9.08e-06 | 6.70e-16 | -0.07% ss, +0.55% accum |

*Gate-aware z-ss measured on W_z only; other matrices use standard GPTQ.

## Why GDN-Specific Strategies Don't Help (with synthetic data)

1. **Gate sensitivity is nearly uniform** (std=0.005): Real W_z@X produces small gate pre-activations (|z| << 1), so SiLU'(z) ≈ 0.5 for all channels. No differentiated sensitivity to exploit.

2. **Balanced realization increases dynamic range**: The transform (condition number 41) changes weight std from 0.017 to 0.023 (+35%), making per-tile quantization worse. System-theoretic balancing ≠ quantization-friendly balancing.

3. **Recurrence weights are nearly uniform**: Synthetic gates (~0.9 decay) and synthetic inputs (uniform energy) produce recurrence weights with little per-channel variation.

4. **QKV joint is tautologically identical**: Column-sequential GPTQ with shared input X cannot exploit cross-projection structure.

## What Would Need to Change for GDN-Specific to Help

1. **Real model activations** (not synthetic Gaussian): Real GDN gate dynamics would produce actual boundary crossings and differentiated σ' values.
2. **Full-fan-in weights** (not 128×128 slices): The reviewer showed that using full 5120-column W_z fan-in raises sensitivity std from 0.005 to 0.021 — 4× more variation.
3. **Real gate traces** (not synthetic α/β): Actual decay/write gate sequences from model forward pass would create meaningful recurrence weight variation.
4. **GPU forward pass** to capture real calibration data — CPU-only constraint limits us to synthetic data.

## Bugs Fixed Across v1→v2→v3

| Bug | Impact | Fix |
|-----|--------|-----|
| GPTQ Cholesky (full inverse vs U=chol(H^{-1}).T) | GPTQ was partially broken | Correct U^T U = H^{-1} convention |
| Gate sensitivity (synthetic RNG vs real W_z@X) | Manufactured 0.187 std vs real 0.005 | Compute from real W_z@X |
| Metric mismatch (unweighted vs weighted) | Manufactured 11-13% improvement | Same metric for all arms |
| Hessian formula (wrong trace contraction) | Manufactured 77-87% GPTQ regression | Correct: mean(D⊙(EX)²) |
| Balanced transform (sqrt vs fourth root) | 30000× worse → 300× worse | eigvals**0.25 |
| Balanced transform direction | Wrong basis quantized | K'=TK, Q'=T^{-T}Q, V'=TV, Out'=Out@T^{-1} |
| Gate activation (sigmoid vs SiLU) | Wrong derivative, lower sensitivity | SiLU: z·σ(z), g'=σ(z)(1+z(1-σ(z))) |
| Gate-aware applied to all matrices | Conflates output/input dims | Apply only to W_z |
| QKV excluded V | Incomplete test | Include Q+K+V |
| z-alloc vs GPTQ (unmatched optimizer) | Unfair comparison | Compare mixed-RTN to uniform-RTN |
| Single-seed accumulated error | Not robust | 100 random sequences |
| QKV slicing (K from rows 128:256) | K was actually Q rows | K from rows 2048:2176 |
| Recurrence timing (output before update) | Wrong state | Output after state update |

## Cross-researcher insights

- **R3/R9**: Balanced realization is a non-orthogonal transform (not a pure rotation). R9's rotation+GPTQ synergy applies to orthogonal rotations, not magnitude-changing similarity transforms.
- **R8**: Gate-aware Hessian (output-side) and scaling (input-side) operate on different axes but gate sensitivity is too uniform to matter.
- **R7**: The recurrence error propagation model is mathematically correct but needs real gate dynamics to show differentiation.
- **R1**: The gate sensitivity could inform tile-BAQ allocation, but only if real calibration data produces meaningful per-channel variation.
