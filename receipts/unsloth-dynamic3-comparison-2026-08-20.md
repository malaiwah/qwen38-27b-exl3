# Unsloth Dynamic 3.0 GGUF: technique analysis and measured comparison

**Date:** 2026-08-20. Sources: Unsloth docs (unsloth.ai/docs/basics/dynamic-3.0-ggufs), Qwen3.8 model page, Aider Polyglot benchmarks page, HuggingFace file manifest, our own logprob measurements.

## How Unsloth Dynamic 3.0 works

Dynamic 3.0 is a post-training quantization (PTQ) method — no QAT, no QAD, no training on the calibration data. The key ideas:

### 1. Per-layer dynamic bit allocation

> "Dynamic 1 bit makes important layers in 8 or 16 bits and un-important layers in 1, 2, 3, 4, 5 or 6 bits."

Every layer gets its own quantization type, chosen dynamically based on sensitivity. This is the same principle as our EDA error-driven allocation, but applied to GGUF quant types (Q2_K, Q3_K_M, Q4_K_M, Q5_K, Q6_K, Q8_0, IQ1_S, IQ2_XXS, etc.) rather than trellis bit widths.

**Key difference from our approach:** Unsloth assigns GGUF quant *types* per tensor, not just bit widths. Each GGUF type (Q2_K, Q3_K_M, IQ4_XS, etc.) has a different block structure, scale format, and codebook. Our trellis uses a single encoding (trellis coding) with a variable bit width (K3–K8).

### 2. Improved imatrix calibration dataset

Dynamic 3.0 uses a "much higher-quality imatrix calibration dataset from diverse sources" refined for "agentic coding, chat, and multilingual performance." The dataset is >1.5M tokens. This is the importance matrix that guides which layers get more bits.

**Key finding:** Unsloth explicitly warns that "instruct models have unique chat templates, and using text-only calibration datasets is not effective for instruct models." This corroborates our finding that the GDN `cache=None` tracing produces unrepresentative activations — the calibration data must match the model's actual inference distribution.

### 3. Layer selection improvement over v2.0

Dynamic 2.0 "selectively quantizes layers much more intelligently and extensively" — "rather than modifying only select layers, we now dynamically adjust the quantization type of every possible layer, and the combinations will differ for each layer and model." Dynamic 3.0 further improves this with "improved layer selection."

### 4. Divergence-300 @32 metric

Unsloth created a new metric: 300 held-out examples, greedy argmax for 32 tokens, comparing quant vs BF16 trajectories. This is "KLD top-1% at 32 tokens" — extending single-token top-1 to multi-token generation. This is a stronger overfitting check than single-token KLD.

### 5. Calibration dataset overfitting awareness

Unsloth explicitly tests for overfitting: "We noticed using the calibration dataset which is also Wikipedia related causes quants to overfit." They use separate Calibration_v3 and Calibration_v5 datasets for fair testing, and do NOT use their own calibration dataset when benchmarking KLD.

### 6. Model-specific quants

"Each model now uses a custom-tailored quantization scheme. E.g. the layers quantized in Gemma 3 differ significantly from those in Llama 4." This is model-aware allocation, not a generic heuristic.

### 7. New low-bit formats

Dynamic 3.0 extends IQ1_S (1.5625 bpw) down to Q1_0 (1.1875 bpw) by reducing codebook entries from 2048 to 256. These are for extreme compression of very large models (Qwen3.8-2.4T).

### 8. MTP removal for small quants

"Removed the MTP module from smaller quants under UD-Q2_K_XL (8.37GB and lower) to converse around 500MB of disk space." This is the same trade-off we identified — MTP costs ~0.8 GiB and can be dropped to save space.

## Published KLD numbers (Unsloth's protocol)

From their Gemma 3 (27B) table (their suite, NOT ours — protocol-incomparable):

| Quant | Baseline KLD | Unsloth Dynamic KLD | GB |
|---|---:|---:|---:|
| Q2_K_XL | 0.229671 | 0.220937 | 9.95 |
| Q3_K_XL | 0.087845 | 0.080617 | 12.76 |
| Q4_K_XL | 0.024916 | 0.023701 | 15.64 |

These are on Gemma 3, not Qwen3.8. Unsloth's Qwen3.8 KLD numbers are shown in charts (not tables) on the Dynamic 3.0 page, so exact values cannot be extracted from the text.

## Our measured comparison

### Setup

- **GGUF file:** `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q4_K_M.gguf` (16.46 GB, Dynamic 3.0)
- **Our checkpoint:** `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` (16.82 GiB trellis payload, fidelity profile)
- **Served via:** llama-server (GGUF) on port 8080, vLLM (trellis) on port 8000
- **Method:** top-20 logprobs from both models on 6 diverse prompts, KL divergence computed over the union of top-20 tokens
- **Limitation:** This is a 6-prompt proxy, NOT our 512-context suite. The proxy is noisier and uses top-20 truncation over a 248,320-token vocabulary.

### Results

| metric | value |
|---|---:|
| top-1 agreement | **6/6** (both models pick the same next token on all prompts) |
| mean \|Δlogprob\| (top-1) | **0.054** |
| max \|Δlogprob\| (top-1) | 0.164 |
| proxy KL(trellis \|\| GGUF) mean | 0.222 |
| proxy KL min | 0.002 |
| proxy KL max | 0.544 |

### Interpretation

The top-1 agreement (6/6) and small mean logprob delta (0.054) show the two quantization methods produce nearly identical next-token predictions at the top-1 level. The higher proxy KL (0.222) reflects tail-distribution differences amplified by the top-20 truncation — with only 20 of 248,320 tokens, the normalization is rough and small differences in the tail compound.

**For context:**
- Our trellis K5K6 KLD vs BF16 on the 512-ctx suite: **0.003405** (mean), top-1 ~99.7%
- The Unsloth GGUF at the same size also shows strong top-1 agreement (6/6)
- The proxy KL between the two quantized models (0.222) is much larger than either's KLD to BF16, which is expected: it sums both quantization errors plus the truncation noise

**We cannot directly compare our KLD (0.003405) to Unsloth's KLD for Qwen3.8** because:
1. Our suite is 512 contexts × 2047 positions, full-vocabulary KL through a shared BF16 LM head
2. Unsloth's Qwen3.8 KLD numbers are only in charts (not extractable as text)
3. The 6-prompt proxy here is not comparable to either protocol

However, the top-1 agreement and logprob delta suggest both methods are at similar fidelity at this size — both are clearly in the "high fidelity" tier where greedy predictions are preserved.

### Size comparison

| artifact | size | method | top-1 agreement (proxy) |
|---|---:|---|---:|
| Our trellis K5K6 | 16.82 GiB (18.06 GB) | EXL3 trellis, per-module K5/K6 | — (reference) |
| Unsloth UD-Q4_K_M | 15.34 GiB (16.46 GB) | Dynamic 3.0, per-tensor GGUF types | 6/6 vs trellis |

The Unsloth file is 1.6 GB smaller while maintaining top-1 agreement. This is because GGUF's mixed-type packing (Q4_K_M + some layers at higher bits) can be more size-efficient than our uniform K5/K6 trellis split. However, our trellis has measured KLD 0.003405 on a rigorous protocol; the Unsloth KLD on the same protocol is unknown.

## What we should change if we ever requant

Based on this research, registered as todo items:

### High-value, directly actionable

1. **Per-tensor-type allocation, not just bit width.** Unsloth's key insight is assigning different GGUF quant *types* (Q2_K, Q3_K_M, Q4_K_S, Q6_K) per tensor, not just different bit widths. Our trellis only varies the bit width (K3–K8). We could explore mixed encoding schemes — e.g. K6 for sensitive tensors, K4 with a different codebook for insensitive ones.

2. **Chat-template-aware calibration.** Unsloth explicitly warns that text-only calibration is ineffective for instruct models. Our GPTQ `cache=None` tracing bug is the same class of problem. If we ever requant, we must ensure the calibration data flows through the model's actual inference path (including chat template, GDN state, etc.).

3. **Multi-token divergence metric.** Unsloth's Divergence-300 @32 (32-token greedy trajectory comparison) is a stronger metric than single-token KLD. We should add this to our fidelity harness — it would catch trajectory drift that single-token KLD misses.

4. **Overfitting check with held-out data.** Unsloth tests KLD on data deliberately different from calibration. We should split our fidelity suite into calibration and held-out portions to detect overfitting in any future calibration work.

### Medium-value, research needed

5. **MTP removal for small quants.** Unsloth drops MTP below 8.37 GB to save 500 MB. We measured MTP at 0.8 GiB. For our 32 GiB card, this is a meaningful fraction — but we need MTP for throughput. Worth measuring: what's the KLD cost of dropping MTP vs the context gain?

6. **Model-specific allocation, not heuristic.** Unsloth tailors the scheme per model. Our EDA error-driven allocation is already model-specific, but we should verify it against Unsloth's approach for Qwen3.8 specifically — do they also find attention layers more sensitive? GDN? MLP?

7. **Dynamic 2.0 vs 3.0 comparison.** Unsloth says 3.0 is better for small quants but "the bigger ones not so much, so we still use our old UD-2 for the larger quants." We should download a Dynamic 2.0 file at our size and measure the same proxy KLD to see if the improvement is visible at the 16 GB tier.

### Low-value or not applicable

8. **Extreme low-bit formats (IQ1_S, TQ1_0).** These are for 1–2 bpw, far below our 5.5 effective bpw. Not applicable.

9. **Efficiency metric.** Unsloth's `(MMLU - 25) / disk_size` metric is interesting for cross-model comparison but not useful for our single-model optimization.

10. **llama.cpp bug fixes.** Unsloth fixes chat template and tokenization bugs in llama.cpp. We use vLLM, so these are not directly applicable, but the pattern (verifying tokenization matches the original model) is a good practice.
