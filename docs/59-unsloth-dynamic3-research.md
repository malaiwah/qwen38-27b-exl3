# Unsloth Dynamic 3.0: research findings and requant avenues

**Date:** 2026-08-20. Source: `receipts/unsloth-dynamic3-comparison-2026-08-20.md`.

## Summary

Unsloth Dynamic 3.0 is a PTQ method that assigns per-tensor GGUF quant types
using a >1.5M-token chat-aware imatrix calibration. A six-prompt, top-20
cross-engine proxy found 6/6 next-token top-1 agreement and mean top-1
|Δlogprob| 0.054 between UD-Q4_K_M and our trellis build. That sample is a
sanity check, not evidence that the methods occupy the same fidelity tier.

## How Dynamic 3.0 differs from our approach

| axis | our trellis K5K6 | Unsloth Dynamic 3.0 |
|---|---|---|
| encoding | single trellis code, variable K-width (K3–K8) | mixed GGUF types per tensor (Q2_K, Q3_K_M, Q4_K, Q6_K, IQ4_XS…) |
| allocation | error-driven EDA on 409-module ladder | imatrix-guided per-tensor type selection |
| calibration | exllamav3's default converter calibration corpus | >1.5M tokens, chat-aware, diverse sources |
| evaluation separation | v5 is lexical-overlap-disjoint from converter calibration; EDA objective selection still reused a small set of measured deltas | separate held-out test set, Divergence-300 @32 |
| metric | full-vocabulary teacher-forced KL over 2,047 positions/context through a shared BF16 head | KLD + top-1% + Divergence-300 @32 (32-token greedy trajectory) |
| MTP handling | kept BF16 (0.8 GiB) | dropped below 8.37 GB to save 500 MB |

## What we should change if we ever requant

### High-value, directly actionable

1. **Per-tensor-type allocation, not just bit width.** Unsloth's key insight
   is assigning different GGUF quant *types* per tensor, not just bit widths.
   Our trellis only varies K-width. We could explore mixed encoding schemes —
   e.g. K6 for sensitive tensors, K4 with a different codebook for insensitive
   ones. This is the single biggest structural difference.

2. **Chat-template-aware calibration.** Unsloth's warning is relevant to any
   future activation calibration: data should flow through the actual chat
   template and model execution path. The GPTQ `cache=None` defect was a
   separate execution-correctness bug, not evidence about dataset style.

3. **Multi-token divergence as a complementary metric.** Divergence-300 @32
   compares 32-token greedy trajectories and can reveal compounding after a
   branch. Our KLD is not "single-token": it is teacher-forced at every scored
   continuation position. Trajectory divergence is complementary, not strictly
   stronger, because it confounds the initial branch with all downstream
   differences.

4. **Keep objective selection separate from final evaluation.** v5 is held out
   from the converter calibration corpus, but future allocation/calibration
   tuning must reserve a second untouched split rather than select and report
   on the same measured deltas.

### Medium-value, research needed

5. **MTP removal throughput cost vs context gain.** MTP does not change the
   verified target distribution when speculative decoding is implemented
   correctly, so KLD is the wrong objective. Measure acceptance, decode
   throughput and context/KV gained when the 0.8 GiB draft is removed.

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
