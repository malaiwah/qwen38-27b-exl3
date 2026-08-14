# Our KLD numbers vs published measurements

## The "FP8 is at 0.5" claim does not hold up

The best public dataset for this exact architecture family is Quesma's
[*Do Qwen3.6 27B quantizations break the pelican?*](https://quesma.com/blog/qwen-quantization-quality/)
(Piotr Migdal, 2026-07-27), which reports **mean KL divergence from BF16** for
21 quantizations of `Qwen3.6-27B` — the immediately preceding generation of the
identical architecture. Values read from the published chart data:

| quantization | size on disk | mean KLD vs BF16 |
|---|---:|---:|
| UD-IQ2_XXS | 9.6 GB | 0.28 |
| UD-IQ2_M | 11.0 GB | 0.13 |
| UD-Q2_K_XL | 12.0 GB | 0.12 |
| UD-IQ3_XXS | 12.2 GB | 0.086 |
| Q3_K_S | 12.6 GB | 0.083 |
| Q3_K_M | 13.8 GB | 0.050 |
| UD-Q3_K_XL | 14.8 GB | 0.038 |
| IQ4_XS | 15.7 GB | 0.018 |
| Q4_0 | 16.1 GB | 0.035 |
| Q4_K_S | 16.1 GB | 0.019 |
| IQ4_NL | 16.3 GB | 0.018 |
| Q4_K_M | 17.1 GB | 0.017 |
| UD-Q4_K_XL | 17.9 GB | 0.013 |
| Q5_K_S | 19.3 GB | 0.0058 |
| Q5_K_M | 19.8 GB | 0.0052 |
| UD-Q5_K_XL | 20.4 GB | 0.0046 |
| Q6_K | 22.9 GB | 0.0021 |
| UD-Q6_K_XL | 26.0 GB | 0.0014 |
| Q8_0 | 29.0 GB | 0.00049 |
| UD-Q8_K_XL | 35.8 GB | 0.00038 |
| **FP8 (vLLM)** | 30.9 GB | **0.017** |
| **NVFP4 (NVIDIA)** | 21.9 GB | **0.039** |
| **NVFP4 (Unsloth)** | 23.3 GB | **0.044** |

**FP8 measures 0.017, not 0.5.** A KLD of 0.5 is far outside anything published
for this family — Quesma's *2-bit* 9.6 GB quant only reaches 0.28. If a 0.5
figure is circulating, it is either a different metric (per-token cross-entropy
delta, or KLD scaled by 1e6 as in
[arXiv:2510.25602](https://arxiv.org/pdf/2510.25602) Table 12), a different
corpus, or a broken run. It should not be used as an FP8 baseline.

The same source independently confirms the shape of our result: **both NVFP4
builds (0.039, 0.044) are 2-3x worse than 4-bit GGUF quants of the same or
smaller size** (Q4_K_M 0.017 at 17.1 GB, UD-Q4_K_XL 0.013 at 17.9 GB), and worse
than FP8 at 8 bits. NVFP4 is not a strong 4-bit format on this architecture; it
is a fast one.

## Cross-checking our own measurements

Ours, all on one frozen 2048-token window with the same BF16 teacher logits
(`kv_cache_dtype=auto`, 3 repeats, run SD 0):

| candidate | mean KLD | resident weights |
|---|---:|---:|
| `k4-online-k6` (ours) | 0.034030 | 19.21 GB |
| `k4-bf16-attn` (ours, overlay off) | 0.036775 | 24.63 GB |
| `unsloth/Qwen3.8-27B-NVFP4` | 0.091457 | 23.42 GB |

Two things to be careful about before comparing across sources:

1. **Absolute values are corpus- and protocol-dependent.** We measure Unsloth's
   NVFP4 at 0.0915 where Quesma measures their Qwen3.6 NVFP4 at 0.044 — a factor
   2.1. Different generation (3.8 vs 3.6), different corpus (one 2048-token
   exllamav3 `wiki.utf8` window vs their sampling), different position count, and
   a different engine build. Ratios within one protocol are meaningful; absolute
   values across protocols are not.
2. **Our ratio is the claim.** Within our single protocol, this quant is
   **2.69x closer to BF16 than the same-generation NVFP4 checkpoint** while
   holding 4.2 GB less weight. That ordering matches Quesma's independent finding
   that good 4-bit non-NVFP4 formats beat NVFP4 by 2-3x.

## Harness validation

`run SD = 0.000000` across 3 repeats for every candidate shows the pipeline is
deterministic. The stronger check — scoring the **BF16 teacher against its own
captured logits**, which must return ~0 — is recorded in
[10-results-iteration-1.md](10-results-iteration-1.md) as `self-bf16`. Any
non-trivial value there would indicate harness bias (wrong window, wrong
densification, dtype drift) rather than quantization error.

Known asymmetry to keep in mind: the teacher logits are captured through the same
`prompt_logprobs=-1, flat_logprobs=True` path as the candidates, so any bias in
that extraction is common-mode and cancels in the comparison. It does not cancel
against *other people's* numbers, which is the second reason not to compare
absolutes across sources.

## What would make our numbers directly comparable to Quesma's

1. Score more than one window (they sample broadly; we use one).
2. Publish the exact token window (we do: it is frozen in `tokens.json` and its
   first 16 ids are asserted in every summary).
3. Add a GGUF candidate of the same model to our own sweep - the cheapest way to
   anchor our scale to theirs, because the GGUF numbers are the densest published
   series.
