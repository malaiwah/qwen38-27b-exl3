# Unsloth Dynamic 3.0: research findings and requant avenues

**Date:** 2026-08-20. Source: `receipts/unsloth-dynamic3-comparison-2026-08-20.md`.

## Summary

Unsloth Dynamic 3.0 is a PTQ method that assigns per-tensor GGUF quant *types*
(Q2_K, Q3_K_M, Q4_K_S, Q6_K, IQ4_XS, etc.) using a >1.5M-token chat-aware
imatrix calibration. Measured head-to-head at the 16 GB tier: UD-Q4_K_M (16.46
GB) vs our trellis K5K6 (16.82 GiB) → 6/6 top-1 agreement, mean |Δlogprob|
0.054. Both methods are in the high-fidelity tier at this size.

## How Dynamic 3.0 differs from our approach

| axis | our trellis K5K6 | Unsloth Dynamic 3.0 |
|---|---|---|
| encoding | single trellis code, variable K-width (K3–K8) | mixed GGUF types per tensor (Q2_K, Q3_K_M, Q4_K, Q6_K, IQ4_XS…) |
| allocation | error-driven EDA on 409-module ladder | imatrix-guided per-tensor type selection |
| calibration | our fidelity suite tokens (512 ctx × 2047) | >1.5M tokens, chat-aware, diverse sources |
| model-specificity | measured per-module sensitivity (our EDA) | model-specific scheme, different per model |
| overfitting check | none (suite = calibration) | separate held-out test set, Divergence-300 @32 |
| metric | full-vocab KL through shared BF16 LM head | KLD + top-1% + Divergence-300 @32 (32-token greedy trajectory) |
| MTP handling | kept BF16 (0.8 GiB) | dropped below 8.37 GB to save 500 MB |

## What we should change if we ever requant

### High-value, directly actionable

1. **Per-tensor-type allocation, not just bit width.** Unsloth's key insight
   is assigning different GGUF quant *types* per tensor, not just bit widths.
   Our trellis only varies K-width. We could explore mixed encoding schemes —
   e.g. K6 for sensitive tensors, K4 with a different codebook for insensitive
   ones. This is the single biggest structural difference.

2. **Chat-template-aware calibration.** Unsloth explicitly warns that
   text-only calibration is ineffective for instruct models. Our GPTQ
   `cache=None` tracing bug is the same class of problem. Any future
   calibration must flow through the model's actual inference path
   (chat template, GDN state, etc.).

3. **Multi-token divergence metric.** Unsloth's Divergence-300 @32
   (32-token greedy trajectory comparison vs BF16) is a stronger metric than
   single-token KLD. We should add this to our fidelity harness — it would
   catch trajectory drift that single-token KLD misses.

4. **Overfitting check with held-out data.** Unsloth tests KLD on data
   deliberately different from calibration. We should split our fidelity suite
   into calibration and held-out portions to detect overfitting in any future
   calibration work.

### Medium-value, research needed

5. **MTP removal KLD cost vs context gain.** Unsloth drops MTP below 8.37 GB.
   Our MTP costs 0.8 GiB. On 32 GiB, that's a meaningful fraction. Measure:
   what KLD does dropping MTP cost, and how much context does the freed VRAM
   buy?

6. **Dynamic 2.0 same-size comparison.** Unsloth says 3.0 is better for small
   quants but "the bigger ones not so much." Download a Dynamic 2.0 file at
   our size and measure the same proxy KLD to see if the v3 improvement is
   visible at the 16 GB tier.

7. **Verify our EDA allocation against Unsloth's Qwen3.8 choices.** Do they
   also find attention layers more sensitive? GDN? MLP? If their per-tensor
   type map for Qwen3.8 is published or extractable from the GGUF metadata,
   compare it to our error-driven ladder.

### Cross-references

- Our EDA allocation: `docs/57-eda-allocation-revisit.md`, `receipts/eda-resolve-2026-08-19.md`
- Our depth calibration (late-heavy): `receipts/eda-depth-calibration-2026-08-19.md`
- Our GPTQ `cache=None` diagnosis: `receipts/gptq-attempt4-2026-08-19.md`
- Prior art survey (llama.cpp, EXL2, EXL3): `docs/58-qwen36-quant-prior-art.md`
- AtomicChat GGUF comparison (same model, our size range): `receipts/atomicchat-gguf-comparison-2026-08-19.md`
- Third-party comparisons (lribeiro, turboderp): `receipts/lribeiro-fp8-comparison-2026-08-19.md`
