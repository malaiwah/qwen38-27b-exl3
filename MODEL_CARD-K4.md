---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - exl3
  - exllamav3
  - trellis
  - mixed-precision
  - quantized
  - qwen3.8
  - vision-language
  - gilded-gnosis
---

# Qwen3.8-27B-K4 — EXL3 mixed-precision, the capacity edition

> ### Superseded by [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6)
>
> The successor spends the remaining memory budget on the MLP (gate/up K5, down K6) and
> measures **0.003210** mean KLD on the v5 held-out suite against this checkpoint's
> **0.010604** (0.007945 versus 0.029679 on the older overlap-corrected v3 subset), at
> 21.82 GB resident — **39 % below official FP8** on v5, 38 % below it on the v3 subset.
> It also has a quantized MTP draft head that works with speculative decoding (58.2 % acceptance,
> +101 % single-stream throughput). Prefer it unless you specifically need the smaller
> 19.21 GB footprint.
>
> Two corrections to this card, from an independent review: the first published
> comparison used evaluation prompts drawn from the quantizer's own calibration corpus
> (re-measured held-out numbers are below), and `reasoning_effort` accepts only
> `xhigh`/`medium`/`low`. This checkpoint's `lm_head` also carries the `mul1` codebook
> rather than the `mcg` implied by the documented `-cb mcg`, an artefact of that run's
> crash-and-resume; the successor is `mcg` throughout and reproducible from its command.



Dense EXL3 quant of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
built on one principle: **spend 4 bits only where the two independent NVFP4
recipes for this architecture also spend 4 bits, and protect everything they
protect — but protect it with Trellis, not with FP8.**

- **MLP** (`gate_proj`/`up_proj`/`down_proj`, all 64 layers) → **EXL3 K4**,
  serialized, calibrated LDLQ.
- **Attention** (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` on the 48 linear
  layers, `self_attn.{q,k,v,o}_proj` on the 16 full-attention layers) → **BF16 on
  disk**, encoded to **K6 at load time** by the Gilded Gnosis vLLM fork's
  `ONLINE_QUANT=exl3-b6` overlay, cached content-addressed on disk.
- **`lm_head`** → **EXL3 K6** (6 bpw, serialized).
- **`embed_tokens`, vision tower (27 blocks), MTP draft head, norms** → **BF16**,
  untouched.

**Measured resident weights: 17.89 GiB (19.21 GB)** on a single RTX PRO 6000
Blackwell, versus **21.92 GB** for `nvidia/Qwen3.6-27B-NVFP4` and **23.42 GB**
for `unsloth/Qwen3.8-27B-NVFP4` on the identical architecture — a
**2.7 GB smaller** resident footprint. Per-role bit widths are not directly comparable
across Trellis, NVFP4 and FP8, so the comparison that matters is the measured fidelity
below, not a per-role precision claim.

Headline fidelity now comes from the **v5 held-out suite — 5,120 contexts x 2,047 positions =
10,480,640 scored positions**. This build measures **0.010604** mean KLD
[0.009640, 0.011746] with **95.76 %** top-1, against **0.005294** for
`Qwen/Qwen3.8-27B-FP8`: on that suite official FP8 is the more faithful checkpoint, by
**+0.005310** [+0.004710, +0.006019] paired, with this build ahead in only **7 of 5,120**
contexts. The direction is unchanged from v3 (0.029679 here versus 0.012798 for FP8 on the
corrected subset); only the absolute values moved, because the v5 corpus mix is different.
Every earlier receipt is kept below. With CUDA graphs enabled this build still serves
**faster than the NVFP4 checkpoint** (55.39 vs 49.09 tok/s at C1). Vision works. The v3
artifacts are published as a
[dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3); the v5 run
ships as receipts (`receipts/kld5-*.json`).

> **Position in the family:** this K4 build has the highest divergence of the four published
> builds (**0.010604** on the v5 suite, 0.029679 on the corrected v3 subset) and the smallest
> resident footprint (**17.89 GiB**). It reaches **native 262,144 on a real 32 GB card with no
> overlay at all** — 289,577 KV tokens measured on an RTX 5090 with MTP-3 and vision enabled.
> The context edition is now hardware-qualified on a physical RTX 5090 as well, at **265,122 KV
> tokens** for 262,144 with MTP-3 and the full 8.4 MP image ceiling, measured at
> `--gpu-memory-utilization 0.955` with all seven gates passing
> ([`receipts/qualification-5090-context.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-context.json)),
> so between the two the trade is footprint and overlay-free simplicity against fidelity, not
> capability.

## Which of the four builds

Same architecture and tokenizer. The first KLD column is the v5 held-out suite (10,480,640
scored positions); the second is the older overlap-corrected 127-context v3 subset, kept for
continuity. Absolute KLD is suite-specific, so the two columns are not comparable with each
other — only the ordering within a column is. Capacity uses each card's documented profile:
hydrated, online and K4 are real RTX 5090 MTP-3 tests; context is MTP-3 with an 8.4 MP cap,
qualified on a physical RTX 5090 at utilisation 0.955.
These profiles are not interchangeable
([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; that engine-budget star has since been superseded by a physical RTX 5090 qualification of the context edition at 265,122 KV tokens and utilisation 0.955." src="assets/context-frontier-light.svg">
</picture>

| build | download | resident | v5 mean KLD | corrected v3 mean KLD | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | **0.007172** | ~180k | fidelity first, smallest download |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.57 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k | you want the attention width knob at launch |
| [-context](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 20.70 GB | **18.41 GiB** | 0.003509 | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window, hardware-qualified on a physical RTX 5090 |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB | 17.89 GiB | 0.010604 | 0.029679 | 262,144 | smallest footprint, native context without any overlay |

Official `Qwen/Qwen3.8-27B-FP8` is 28.51 GiB resident at **0.005294** on the v5 suite and
0.012798 on the corrected v3 subset, and runs on stock vLLM, which none of these do. On both
suites it is more faithful than this K4 build and less faithful than the other three.

## Why this shape

| role | this quant | `nvidia/Qwen3.6-27B-NVFP4` | `unsloth/Qwen3.8-27B-NVFP4` |
|---|---|---|---|
| MLP | **EXL3 K4, 4.004 bpw** | NVFP4 W4A16 gs16, **4.50 bpw** (4 b + FP8 scale per 16) | NVFP4 **W4A4** gs16 on L0-55; FP8 on L56-63 |
| attention | **BF16 on disk → K6 (6.0 bpw) in VRAM** | FP8 E4M3 W8A8, 8 bpw | FP8 W8A8 dynamic, 8 bpw |
| `lm_head` | **EXL3 K6** | NVFP4 (4 bpw) | FP8 (8 bpw) |
| `embed_tokens` | BF16 | BF16 | BF16 |
| vision tower | BF16 | BF16 | BF16 (explicit per-block ignore) |
| MTP head | BF16 | BF16 (`ignore: ["mtp*"]`) | BF16 (`re:^mtp.*`) |
| resident weights | **19.21 GB** | 21.92 GB | 23.42 GB |
| checkpoint size | 28.31 GB | 21.92 GB | 23.42 GB |

Trellis at K4 needs no per-group scale tensor, so 4-bit MLP costs 4.004 bpw here
against NVFP4's 4.50 bpw. That saved 1.07 GB, plus the 1.80 GB from serving
attention at K6 instead of FP8, is what pays for a *lower* footprint at *higher*
precision. The checkpoint is larger than the NVFP4 ones because attention ships
BF16 so the runtime can re-encode it — see *Tradeoffs*.

## Distribution fidelity — v5 held-out suite, 10,480,640 scored positions

This is the headline fidelity evidence for the family, and it is the one measurement where
this build loses to official FP8. Suite
[`receipts/kld5-suite-manifest.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-suite-manifest.json)
(schema `qwen38-distribution-fidelity/6`, suite token sha256
`510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88`): **5,120 contexts x
2,047 positions = 10,480,640 scored positions** over **842 source clusters**, from a corpus of
941 documents / 70,348,971 bytes fetched by `tools/fetch_corpus_v5.py`
([fetch log](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-corpus-fetch-log.json)).
All 941 documents were scanned at every position for exact normalized 12-token overlap with
exllamav3 calibration data *before* selection, and the **44** that hit (43 code, 1
encyclopedic) were excluded whole, leaving **897 eligible documents** — so contamination hits
are 0 by construction. The suite is token-disjoint from the v4 suite (0/160 prior context
hashes reachable), and its windows are exact-advance and non-overlapping — independently
verified at 5,120/5,120 unique token hashes and 0 overlapping windows. Every candidate is
scored body-only: both operands go through one shared BF16 LM head.

| candidate | mean KLD | bootstrap 95 % CI | top-1 | exact max single-position KLD |
|---|---:|---:|---:|---:|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | **0.002760** | [0.002540, 0.003020] | 97.70 % | 8.258 |
| [K5/K6 online K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 0.003210 | [0.002982, 0.003480] | 97.52 % | 22.241 |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 0.003509 | [0.003220, 0.003852] | 97.44 % | 5.557 |
| `Qwen/Qwen3.8-27B-FP8` | 0.005294 | [0.004927, 0.005728] | 96.79 % | 10.714 |
| **this quant (K4)** | **0.010604** | **[0.009640, 0.011746]** | **95.76 %** | 14.283 |

Paired per-context differences (source-cluster bootstrap, 10,000 resamples, seed 1, 842
clusters,
[`receipts/kld5-10M-paired.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-paired.json)):

| comparison | difference | 95 % CI | contexts won |
|---|---:|---:|---:|
| **this quant - FP8** | **+0.005310** | [+0.004710, +0.006019] | **7 / 5,120** |
| hydrated - FP8 | -0.002534 | [-0.002708, -0.002383] | 5,118 / 5,120 |
| online K5/K6 - FP8 | -0.002084 | [-0.002249, -0.001942] | 5,105 / 5,120 |
| context - FP8 | -0.001785 | [-0.001884, -0.001697] | 5,109 / 5,120 |
| hydrated - online K5/K6 | -0.000450 | [-0.000469, -0.000433] | 4,922 / 5,120 |

Stated plainly: **official `Qwen/Qwen3.8-27B-FP8` is markedly more faithful than this build**,
and it wins 5,113 of 5,120 contexts. That is the same direction the v3 receipts reported; ten
million positions did not overturn it, they narrowed the interval around it. If you want
distribution fidelity, take one of the other three builds or FP8; take this one for the
17.89 GiB footprint and native context without an overlay.

**The corpus mix moves absolute values for every candidate, so v5 numbers are not comparable
to v3 numbers.** This build reads 0.029679 on the corrected v3 subset and 0.010604 here; FP8
reads 0.012798 there and 0.005294 here. Nothing about the checkpoints changed between those
two runs — the text did. Only within-suite ordering and the paired differences transfer
across suites.

By stratum, this build's mean KLD is 0.007389 scientific, 0.008897 encyclopedic, 0.010356
multilingual, 0.011612 code and 0.014766 literary, over 1,024 contexts (2,096,128 positions)
each. The cumulative mean is stable along the ladder: for the hydrated candidate the 1M / 2M /
5M / 10M checkpoints read 0.002700 / 0.002759 / 0.002699 / 0.002760, so the 10M figures are not
a moving target.

Per-candidate receipts are
[`receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts)
(schema `qwen38-kld-ladder-cumulative/2`), built by `tools/kld_aggregate.py` from ten verified
per-shard reports produced by `tools/kld_ladder.sh`: capture six models over 512 contexts,
replay five candidates, verify, delete 64 GB of hidden states, move to the next shard.

**Distribution tail.** A mean and a top-1 rate can hide a tail, so here is the tail itself,
measured on **shard 0 of the same suite — 512 contexts, 1,048,064 scored positions** — the
identical contexts for all five candidates. Receipts
`receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json` (schema `qwen38-kld-ladder-cumulative/2`,
built by `tools/kld_aggregate.py`); this build's row is
[`receipts/kld5-1M-tail-k4.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-tail-k4.json).
Every `qwen38-fidelity-report/2` replay accumulates a **560-bin log-spaced histogram of
per-position KLD** (`KLD_HIST_LOG10_LOW=-12.0`, `KLD_HIST_LOG10_HIGH=2.0`,
`KLD_HIST_BINS_PER_DECADE=40` in `tools/fidelity.py`) whose bin counts add across shards,
which is what makes cumulative quantiles possible at all.

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | share of positions above 0.1 | above 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 0.002700 | 0.00109 | 0.0082 | 0.0276 | 0.1319 | 0.463 | 3.735 | 0.1534 % | 0.00219 % |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 0.003141 | 0.00128 | 0.0099 | 0.0321 | 0.1446 | 0.498 | 5.507 | 0.1820 % | 0.00200 % |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 0.003409 | 0.00135 | 0.0107 | 0.0357 | 0.1642 | 0.587 | 3.749 | 0.2287 % | 0.00305 % |
| `Qwen/Qwen3.8-27B-FP8` | 0.005197 | 0.00202 | 0.0167 | 0.0531 | 0.2438 | 0.812 | 5.296 | 0.3912 % | 0.00592 % |
| **this quant (K4)** | **0.010345** | **0.00320** | **0.0332** | **0.1194** | **0.5555** | **1.870** | **7.565** | **1.2604 %** | **0.03807 %** |

**Stated plainly: this build's tail is heavier than official FP8's at every measured
quantile.** p99 is 0.1194 against FP8's 0.0531, p99.9 is 0.5555 against 0.2438, p99.99 is
1.870 against 0.812 — roughly 2.3x at each depth — and the exact worst position on this shard
is 7.565 against 5.296. The exceedance counts say the same thing without any binning: **1.2604 %**
of positions exceed 0.1 against FP8's 0.3912 % (3.2x), and **0.03807 %** exceed 1.0 against
FP8's 0.00592 % (6.4x, 399 positions against 62). The tail ordering at p50, p95, p99, p99.9
and p99.99 is the same as the ordering of the means, so the mean was not flattering this
build: it is last on the mean and last at every quantile, and all three K5/K6 builds are below
FP8 throughout. Take this build for the 17.89 GiB footprint and native context, not for
distribution fidelity — including at the tail.

Three limitations, stated rather than hidden:

- **Not comparable across suites.** Absolute KLD is suite-specific; see above.
- **Cumulative percentiles come from one shard, not from all ten.** The ten shard reports of
  the 10M run carry no token-level KLD histogram, so nothing could be recombined across them.
  The tail table above closes that gap on **shard 0** (`receipts/kld5-1M-tail-*.json`), with
  bin-bounded quantiles — each receipt carries `lower` / `upper` / `estimate`, relative bin
  width about 5.6 % — and **exact** maxima and exceedance counts. Across all 10,480,640
  positions only the means, the intervals, the paired results and one exact global maximum
  per candidate exist.
- **No published captures.** The hidden-state captures were deleted shard by shard to fit 135
  GB of scratch, so unlike the v3 dataset this run is reproducible from the pinned corpus
  fetch log and suite manifest rather than from published captures.

## Against GGUF, measured on our suite

The comparator set used to stop at official FP8, which is a throughput format whose quality is
Q4-to-Q5 class, so llama.cpp's `Q8_0` and `Q6_K` are the honest bar. They have now been measured
on our own suite, and for this build the result is unambiguous: **K4 is the weakest point in the
table.**

Three GGUFs from `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` were
captured under **llama.cpp pinned at commit `ece963f41b0b02d7a0d61436ae365762c073a4c8`** with
[`tools/gguf_capture.cpp`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/gguf_capture.cpp),
which reads the **post-final-norm** state — the same mathematical point the vLLM hook takes, with
bf16 rounding verified bit-identical to torch on 2,012,449 probe values — and scored against the
same BF16 teacher through **the same shared BF16 head**, on **shard 0 of the v5 suite: the same
512 contexts and the same 1,048,064 scored positions every row below saw**. Manifests come from
[`tools/gguf_manifest.py`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/gguf_manifest.py)
and each one carries the GGUF blob digest and the llama.cpp identity; the build script is
[`tools/build_llamacpp.sh`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/build_llamacpp.sh).
Receipt
[`receipts/cross-engine-comparator.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/cross-engine-comparator.json),
per-candidate reports
[`receipts/gguf-report-{q8_0,q6_k,q5_k_xl}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/kld-family-comparison-dark.svg">
  <img alt="Quantization families: every family we have measured on one protocol with two size axes on the left, and one published protocol we have never run on the right; the columns are not interchangeable and no ratio between them is meaningful. Left column is shard 0 of our v5 held-out suite — the same 512 contexts, the same 1,048,064 scored positions and the same 330 source clusters for every candidate, both operands through one shared BF16 head — split into two sub-panels that share a logarithmic y-axis and deliberately do not share an x-axis. Upper sub-panel, x is weights measured resident under vLLM, with no GGUF point because llama.cpp resident weights were never measured: hydrated K5/K6 0.002700 at 20.31 GiB, online K5/K6 0.003141 at 20.32, context edition 0.003409 at 18.41, official Qwen FP8 0.005197 at 28.51, K4 0.010345 at 17.89, each mean marker joined by a vertical line to a hollow triangle at its p99.9 — 0.1313, 0.1447, 0.1632, 0.2440 and 0.5576 — and a printed value table repeating mean, p99.9, top-1 and GiB for all five, noting that a circle is captured under vLLM and carries no engine term. Lower sub-panel, same suite and same y-axis, x is serialized bytes on disk: filled squares for the three GGUFs measured under llama.cpp, Q8_0 0.001087 at 27.052 GiB, Q6_K 0.002035 at 21.313 and UD-Q5_K_XL 0.004444 at 18.830, each with a hollow square below it for its naive net-of-engine-floor estimate of 0.000579, 0.001528 and 0.003936, plus circles for the two builds of ours that have a published payload receipt, hydrated 0.002700 at 20.127 GiB and the context edition 0.003409 at 19.275; online K5/K6 and K4 are absent here because they ship BF16 attention quantized at load and have no payload receipt. A dashed line at 0.000507 marks the measured llama.cpp-versus-vLLM engine floor on the same unquantized BF16 weights and a dotted line marks that floor’s p99.9 of 0.0113; every square contains that term and no circle does. Two crossings are called out in boxes: at 6 bits GGUF Q6_K wins, 0.001528 net at 21.313 GiB against hydrated 0.002700 at 20.127, 43 percent lower KL for 1.186 GiB more weight; at 5 bits our context edition wins, 0.003409 at 19.275 GiB against UD-Q5_K_XL 0.003936 net at 18.830, 13 percent lower KL for 0.445 GiB more weight. A second printed value table repeats mean, net of floor, p99.9, top-1 and GiB for every point in this sub-panel. Right panel is a different protocol entirely: turboderp’s published chart labels on his own OpenWebText run, 8 x 8192 = 65,536 formatted positions against his own BF16 reference, x is quantized decoder weight with embeddings excluded and the output head included, his EXL3 bpw ladder, GGUF UD ladder, one GGUF-IQ point, Unsloth NVFP4 and Qwen FP8, his two synthetic noise floors at 0.0052 mean and 0.0007 median, and vertical markers where our context and hydrated builds fall on his size axis with no y-value because we have never run his protocol." src="assets/kld-family-comparison-light.svg">
</picture>

| candidate | engine | measured mean KLD | net of engine floor | top-1 | p99.9 | serialized |
|---|---|---:|---:|---:|---:|---:|
| GGUF `Q8_0` | llama.cpp | 0.001087 | ~0.000579 | 98.53 % | 0.0351 | 27.05 GiB |
| GGUF `Q6_K` | llama.cpp | 0.002035 | ~0.001528 | 97.98 % | 0.0794 | 21.31 GiB |
| hydrated | vLLM | 0.002700 | n/a, same engine | 97.80 % | 0.1313 | 20.12 GiB payload |
| online K5/K6 | vLLM | 0.003141 | n/a, same engine | 97.61 % | 0.1447 | — |
| context edition | vLLM | 0.003409 | n/a, same engine | 97.55 % | 0.1632 | 19.27 GiB payload |
| GGUF `UD-Q5_K_XL` | llama.cpp | 0.004444 | ~0.003936 | 97.20 % | 0.2144 | 18.83 GiB |
| official FP8 | vLLM | 0.005197 | n/a, same engine | 96.92 % | 0.2440 | 28.51 GiB resident |
| **this quant (K4)** | vLLM | **0.010345** | n/a, same engine as the reference | **95.91 %** | **0.5576** | **—** |

**The engine floor, measured and not assumed.** A GGUF row carries llama.cpp-versus-vLLM numerics
on top of quantization error, so that term was measured the same way: the unquantized **BF16
GGUF** against the vLLM BF16 reference, identical token ids, the same shared head, the same 512
contexts — **0.000507** mean, 99.07 % top-1, p99.9 0.0113
([`receipts/gguf-report-engine-floor.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gguf-report-engine-floor.json)).
Every GGUF row above contains that term; no vLLM row — ours or FP8's — does. **KL is not additive,
so the net column is an estimate, not an identity**: the measured GGUF value is an upper bound and
the net figure is the naive lower one. Note the direction of that asymmetry for this build — the
floor can only *inflate* a GGUF number, so subtracting it makes every GGUF row **better**, not
worse, and there is no reading of the cross-engine term under which this build stops being last.
The `—` cells are the two builds that ship BF16 attention for the runtime to encode at load,
including this one (28.31 GB download, **17.89 GiB resident**), so their disk bytes are not a
like-for-like payload against a GGUF file; the payload figures are `immutable_payload_bytes` from
[`receipts/collection-index.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/collection-index.json)
(hydrated 21,610,916,123 B = 20.127 GiB, context edition 20,696,033,532 B = 19.275 GiB; the table
truncates both to two decimals) and are serialized bytes, never VRAM. The FP8 figure is resident
weights and is labelled as such.

**The p99.9 column, and why it differs from the tail table above.** These p99.9 values are each
report's **exact** shard-0 p99.9 as the comparator receipt read them; the
[tail table above](#distribution-fidelity--v5-held-out-suite-10480640-scored-positions) quotes the
**bin-bounded cumulative estimate** from the 560-bin histogram, whose bins are about 5.6 % wide —
this build reads 0.5555 there and 0.5576 here, hydrated 0.1319 and 0.1313, and each exact value
lies inside the bin the estimate names. The two differ by construction, not by measurement.

**Where this build sits, stated plainly.** Last, on every column of the table. Its 0.010345 mean is
**2.6x** `UD-Q5_K_XL`'s net 0.003936, **6.8x** `Q6_K`'s net 0.001528, **17.9x** `Q8_0`'s net
0.000579 and **2.0x** official FP8's 0.005197; its top-1 is the lowest at 95.91 %, and its p99.9 of
0.5576 is **2.6x** `UD-Q5_K_XL`'s 0.2144, **7.0x** `Q6_K`'s 0.0794 and **2.3x** FP8's 0.2440. It is
beaten by every GGUF measured here, including `UD-Q5_K_XL`, the smallest of the three at 18.83 GiB
of serialized weight. This build's argument was never fidelity — it is the 17.89 GiB footprint and
native 262,144 context with no overlay — but that is
a capacity argument, and a reader choosing on distribution fidelity should take a GGUF or one of
the K5/K6 builds instead.

**The two conclusions for the family, neither of them about this build:**

1. **At the 6-bit operating point GGUF `Q6_K` is genuinely better than our best build** — 0.001528
   net at 21.31 GiB against the hydrated build's 0.002700 at 20.12 GiB of payload. It is the first
   measurement in this project where an off-the-shelf artifact beats the recipe, and it is
   published as such.
2. **At the 5-bit operating point our context edition wins** — 0.003409 at 19.27 GiB against
   `UD-Q5_K_XL`'s 0.003936 net at 18.83 GiB, about 13 % better fidelity for **0.445 GiB** more
   payload.

`Q8_0` is the fidelity leader at 0.001087 for 27.05 GiB, and its measured value is only about twice
the engine floor, so its own number sits near the resolution limit of any cross-engine comparison:
because the net column is an estimate and not an identity, **no ordering closer than a factor of two
should be pressed against `Q8_0`**. This build's distance from it is a factor of ten, so that
caution does not soften anything above.

Every GGUF point at or above 5 bits beats official FP8, which makes the family's "below FP8" claim
true and a weaker achievement than it sounds — and this build does not make that claim at all: it
loses to FP8 by a factor of two.

**What this comparison does not settle.** It is text-only teacher-forced fidelity on one shard of
ten. It says nothing about serving 262,144 tokens with vision and MTP on a 32 GB card, which is
where these artifacts actually differ and which is this build's own reason to exist, and llama.cpp
KV-quant behaviour, prefill and decode speed are separate axes that were not measured here. The
GGUF rows are a shard-0 ranking, not a paired per-context bootstrap against the ten-shard rows
above, because those were welded from a different position count. Shard 0 is one tenth of the suite, and it is close to it: over all 10,480,640 positions the five vLLM
means read 0.002760 / 0.003210 / 0.003509 / 0.005294 / 0.010604 — **1.9-2.9 % above** these shard-0
values, ordering unchanged (`receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`). The GGUFs have no
ten-shard equivalent; extending them is unrun.

**One protocol objection, bounded rather than argued.** `llama-perplexity` scores only the second
half of each window, so every position it scores has at least 256 tokens of left context, while our
suite scores from position 0. Re-scoring our own captures under that restriction lowers every
candidate's mean by **1.3-2.1 %** at a 256-token floor and **3.9-4.9 %** second-half-only,
uniformly enough to change no ordering — this build reads 0.010154 and 0.009876 respectively
([`receipts/scored-window-offset.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/scored-window-offset.json)).
The external protocol's scoring floor therefore explains at most about 5 % of any cross-protocol
gap, and none of this build's distance from the rest of the table.

## Post-selection qualification

The v3 numbers in this card come from the suite that guided recipe selection. This is the v4
test that did
not: **160 new contexts from 100 documents with zero intersection with the development suite**
(context token hashes 0/160, document names 0/100, content hashes 0/100), partitioned by whole
source cluster, run **once**, with no recipe changed afterwards.

The original 42-context table used a fixed-stride character overlap scan. A later,
offset-independent scan found exact 12-token calibration overlap in four qualification source
documents. Applying the same conservative rule to every candidate — exclude every context from
any source document with even one hit — leaves **36 contexts / 24 clusters**:

| candidate | mean KLD | 95 % CI | top-1 | paired vs FP8 |
|---|---:|---|---:|---|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | **0.003093** | [0.002577, 0.003684] | 97.63 % | −0.002798, **36/36** |
| [K5/K6 online K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 0.003455 | [0.002916, 0.004060] | 97.50 % | −0.002436, **36/36** |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 0.003990 | [0.003268, 0.004797] | 97.36 % | −0.001901, **36/36** |
| `Qwen/Qwen3.8-27B-FP8` | 0.005891 | [0.004901, 0.006985] | 96.72 % | — |

The correction changes no ordering or paired win: the three EXL3 builds remain 47 / 41 / 32 %
below FP8. The original 42-context figures and the candidate-independent correction are both
preserved in [docs/31](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/31-frozen-qualification.md).
Absolute magnitudes remain suite-specific.

## Public capability — MMLU-Pro, item-paired against BF16

70 MMLU-Pro questions, 14 official categories, 5 per category, official five-shot category
prefixes, pinned `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, greedy,
thinking at low reasoning effort, 5,120-token completion cap. The BF16 reference ran first; the
plan and its acceptance rule were frozen before any result was seen. All six models answered
the same 70 items in the same order through the same extractor, so every candidate row is
paired item-by-item against the BF16 control.

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.8-27B` BF16 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | [bf16](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16.json) |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 58/70 (82.9 %) | [72.4 %, 89.9 %] | 56/57 | **90.7 %** | 1 | 2 | 3 | [ctx](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-ctx.json) |
| **this quant (K4)** | **57/70 (81.4 %)** | **[70.8 %, 88.8 %]** | **55/57** | **88.1 %** | **2** | **2** | **4** | [k4](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k4.json) |
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 54/57 | 85.6 % | 3 | 2 | 4 | [hyd](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-hyd.json) |
| `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | [fp8](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-fp8.json) |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 55/70 (78.6 %) | [67.6 %, 86.6 %] | 54/57 | 85.6 % | 3 | 1 | 4 | [k5k6](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k5k6.json) |

**This quant's verdict: 57/70 matches BF16 absolutely, and it misses the pre-registered bar at
88.1 %.** Pass-outcome agreement is 66/70. Equal absolute scores with two regressions and two
improvements is exactly why the measurement is item-paired: the totals match, the per-item
behaviour does not, and the totals alone would have hidden both.

### The pre-registered bar, and who clears it

The frozen plan accepts a candidate when BF16-pass retention has a **Wilson 95 % lower bound at
or above 0.90** and **no category loses more than two BF16 passes**. The category clause is met
by all five candidates — the worst case is two passes in philosophy, for the hydrated build and
online K5/K6 — so the retention lower bound is the only clause that ever fails.

**Only the context edition clears the bar, at 90.7 %.** This quant reads 88.1 % (55/57) and
official `Qwen/Qwen3.8-27B-FP8` reads exactly the same 88.1 % (55/57); the hydrated build and
online K5/K6 read 85.6 %. Four of the five candidates, official FP8 included, are measured
shortfalls, published as measured with nothing retuned after the fact.

**No candidate here is shown to beat another.** Every interval in the table overlaps every
other interval, including the BF16 control's, so the matrix does not rank these models and this
card does not claim it does. This quant is not shown to be worse than the context edition or
better than the hydrated build or online K5/K6 on knowledge-and-reasoning tasks; on this suite
it and official FP8 land on identical retention numbers. Note the contrast with the v5 tail
result above, which is a distribution-fidelity measurement on 10,480,640 scored positions and
does separate this build from FP8: that is a different question, on a different metric, at a
different sample size, and it makes no capability claim.

### Why a 70-item suite cannot certify this bar

With 57 BF16 passes as the paired denominator, **56/57 is the smallest count whose Wilson 95 %
lower bound clears 0.90** (56/57 → 90.7 %; 55/57 → 88.1 %; 54/57 → 85.6 %). A single paired
regression is therefore the entire budget, and no result that gives up two can pass, however
sound the build. The suite has too few items to certify the bar it pre-registered, and at this
size it separates nothing — the point applies to official FP8 exactly as it applies to this
quant. Read the shortfall as a **power limitation of a 70-item draw, not as evidence that any
of these checkpoints is broken**.

Four caveats, all load-bearing:

1. **Four BF16 items hit the 5,120-token completion cap and are counted as failures** under the
   plan's frozen addendum (`finish_reason: length` with no letter emitted), so the BF16
   reference itself is depressed by the cap; this build hits the cap on four items too, and the
   per-model counts are in the table. The superseded 2,048-cap control, where BF16 lost 7/70 to
   truncation, is kept at
   [`receipts/public-capability-bf16-superseded-cap2048.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16-superseded-cap2048.json).
2. **Exact-answer agreement is 0/70** for this build and for every other EXL3 candidate, and
   1/70 for official FP8 (one math item, a 113-token answer both models pass). Long chains of
   thought differ token-wise on essentially every item, so the pairing is on pass/fail outcome,
   never on generated text.
3. **This is a measured shortfall against the pre-registered bar.** The plan required a Wilson
   95 % lower bound at or above 90 % on BF16-pass retention; 88.1 % is below it. It is
   published as measured, with nothing retuned after the fact.
4. **70 items is small.** The Wilson intervals are wide and mutually overlapping, so the full
   six-model matrix — now run for every candidate, not just this one — resolves no ordering.
   This is a **first public, licence-compatible, item-paired benchmark, not a leaderboard
   claim**. The honest next step is more items, which is the plan's own P1: HumanEval+/MBPP-style
   executable cases, IFEval-style constraint following, tool schemas, and a larger MMLU-Pro
   draw. No capability claim on this card graduates before that.

Receipts:
[plan](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-plan.json),
[suite](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-suite-mmlupro-70.json),
and the six per-model runs linked in the table
(`receipts/public-capability-{bf16,ctx,k4,hyd,fp8,k5k6}.json`); harness
[`tools/public_capability.py`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/public_capability.py),
sweep runner
[`tools/run_public_capability.sh`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/run_public_capability.sh).
Every receipt carries the per-item raw request, raw response, extracted letter, gold letter and
digests.

## Downstream task retention — 40-task smoke suite (prior, narrower evidence)

This ran before the MMLU-Pro suite above and is kept unchanged. It is the narrower evidence:
self-generated tasks with contract checks, not a public benchmark, and it says nothing about
the knowledge-and-reasoning behaviour MMLU-Pro probes.

On 40 deterministic generated tasks (10 each arithmetic, executable builtins-only code,
exact-list instruction following and tool-call schema), BF16 and every comparator scored
40/40. This build had **zero regressions** and matched BF16's exact final-answer text on
**33/40**; all seven differing answers still passed their contracts. Wilson 95 % lower bound
is 91.2 %. This is a transparent smoke suite, not a public leaderboard; full responses are in
the [run receipt](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/tasks-v2-k4.json);
its extracted-value agreement field is superseded by the
[strict rescore](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/task-retention-v2-strict-rescore.json).

## Serving

Requires the **Gilded Gnosis vLLM fork** — the EXL3 checkpoint loader, the B12X
Trellis kernels and the `exl3-b6` online overlay are not in upstream vLLM. The
public image is:

```
voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
registry digest sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

Its launcher scripts only dispatch GLM-5.2 and DeepSeek families, so call
`vllm serve` directly:

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro -v /cache:/cache \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
  -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-K4 \
    --served-model-name qwen38-k4 \
    --quantization exl3 \
    --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --host 0.0.0.0 --port 8000
```
The container listens on all interfaces internally, but Docker publishes the port to host
loopback only. For remote clients, keep that binding and put an authenticated TLS proxy in
front; do not expose this unauthenticated generation endpoint directly.


Four flags are load-bearing:

1. **`--quantization exl3` is mandatory.** Auto-detection only fires for the
   GLM-5.2 `r7_routed_experts` / `hybrid_tr3_tail` metadata; a dense
   `tensor_storage` checkpoint is not auto-detected.
2. **`--enforce-eager` is mandatory.** The loader refuses non-eager execution for
   any checkpoint without rank-sliced metadata, because `exl3_gemm` autotunes with
   timing launches. Expect no CUDA graphs.
3. **The `ignore` list is mandatory and its anchoring is subtle.** The overlay
   claims *every* BF16 `LinearBase` not present in `tensor_storage` — including
   the vision tower and the MTP head. The prefixes it matches have no leading
   `model.`, so `re:.*\.visual\..*` (dot before `visual`) silently fails to match
   while `re:.*visual\..*` works. With the wrong pattern the vision tower is
   claimed and startup **crashes** (`ValueError: MXFP8 requires
   input_size_per_partition (4304) to be divisible by 32`), reported upstream as
   [local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311)
   with a verified fix in
   [PR #312](https://github.com/local-inference-lab/vllm/pull/312), which degrades
   those shards to BF16 with a warning instead of aborting.
4. **`VLLM_EXL3_ONLINE_TRELLIS_BITS=6`** is what turns the overlay from MXFP8 into
   K6. Point `VLLM_EXL3_ONLINE_CACHE_DIR` at persistent storage: the first load
   encodes 208 attention projections (~16 min on one GPU here) and later loads
   reuse the cache.

Generation defaults from upstream `generation_config.json`: `temperature 1.0`,
`top_p 0.95`, `top_k 20`. Thinking control is upstream's
`chat_template_kwargs`: `{"enable_thinking": false}` or
`{"reasoning_effort": "xhigh"|"medium"|"low"}` (upstream raises on `high`). The chat template,
tokenizer, preprocessor configs and vocabulary (248320, untied head) are
upstream's, unmodified.

Context: 262144 native, verified here only to 8,192. Upstream's 1M procedure is static
YaRN (nested `rope_parameters` with `rope_type: yarn`, `factor: 4.0`,
`original_max_position_embeddings: 262144`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
`--max-model-len 1000000`), not a bare `max_position_embeddings` bump, and Qwen warns it
costs short-context quality. Untested on this runtime.

## Verification

All measured on 1x RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB), driver
595.58.03, TP1, with the r34 image above.

**Loads and serves.** Engine reports `quantization=exl3`, online K6 encoding for
every attention projection (proxy error ~3.2e-4 per projection), then:

```
Actual usage is 17.89 GiB for weight, 2.33 GiB for peak activation,
0.26 GiB for non-torch memory, and 0.0 GiB for CUDAGraph memory
GPU KV cache size: 736,109 tokens   (--max-model-len 8192, --gpu-memory-utilization 0.85)
```

**Text is coherent.** `"Name the three primary additive colors, comma separated."`
with `enable_thinking: false` → `Red, Green, Blue` (1.6 s, greedy).

**Vision works.** A 96x96 PNG, left half pure red, right half pure blue, with
`"Name the left colour then the right colour, comma separated."` → `red, blue`.

**Quantization error, per tensor, from the conversion log** (LDLQ proxy error):
`down_proj` is consistently the worst projection in every layer — about
`2.5e-3` versus `1.1e-3` for `gate_proj` and `1.0e-3` for `in_proj_qkv`. Whole-block
figures: `rfn ~0.0155`, `sqnr ~36.4 dB`.

### Distribution fidelity — v3 protocol, held-out corpus (prior receipt)

Superseded as the headline by the v5 suite above, which scores 10,480,640 positions against
this section's 278,392. Kept unchanged, including its own correction history.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fidelity-vs-size-dark.svg">
  <img alt="Overlap-corrected mean KL divergence from BF16 versus resident weight footprint. This quant at 19.2 GB and 0.0297; Qwen FP8 at 30.9 GB and 0.0128; Unsloth NVFP4 at 23.4 GB and 0.0927. Right panel shows top-1 agreement: 94.48, 96.18 and 90.49 percent respectively." src="assets/fidelity-vs-size-light.svg">
</picture>

**136 analysis contexts x 2047 positions = 278,392 scored positions** from separately
sourced Gutenberg, arXiv, Wikipedia and CPython documents. The original fixed-stride
160-character scan reported zero calibration hits; a later all-position 12-token scan found
exact overlap in 2/41 source documents. Exact full-vocabulary two-pass
`KL(BF16 reference || candidate)` through one shared BF16 LM head, float32 within each
vocabulary chunk and float64 across chunks, source-cluster bootstrap.

| candidate | weights | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 0.773 | 0.004528 | 96.22 % |
| **this quant** | **19.2 GB** | **0.030736** | [0.02238, 0.04073] | 0.004218 | 1.758 | 0.010051 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 4.509 | 0.028663 | 90.53 % |

**Overlap-corrected subset:** conservatively removing all nine analysis contexts from either
affected source document gives K4 **0.029679**, FP8 **0.012798**, and NVFP4 **0.092727**
over 127 contexts. The ranking and every conclusion survive; the table above is retained as
the original full-suite receipt, not described as contamination-free.

Paired over the same contexts: **-0.064242** versus NVFP4
(95 % CI [-0.08621, -0.04611], **136/136** contexts ours) and
**+0.017611** versus FP8 (95 % CI [0.01256, 0.02368], 136/136 contexts FP8).

**An earlier version of this card reported better numbers on a contaminated suite.**
The previous corpus was exllamav3's own calibration data — the text this quant was
tuned on, while the NVFP4 and FP8 candidates were calibrated elsewhere. Re-measuring
on held-out text moved ours from 0.026231 to 0.030736 (+17 %), NVFP4's from 0.073006
to 0.094978, and FP8's from 0.019309 to 0.013126 (-32 %). These are the honest
numbers; the correction is documented in the companion repo.

Controls shipped with the dataset: runtime-repeat noise floor **0.000000** across
three captures of the same runtime (this runtime is bit-deterministic, so every
difference above is far outside noise); harness self-check 0.000000; CUDA-graph
and a harness self-check of 0.000000. A third control, "CUDA-graph parity 0.000000", is
**withdrawn**: it captured a prefill forward, and `FULL_DECODE_ONLY` captures no prefill
graph, so it could not have measured the decode path. Re-measured on real decode steps,
graph and eager agree on 24/32 greedy 32-token sequences (mean |Δ logprob| 0.0118) and
unquantised BF16 on the same build drifts identically (24/32, 0.0128), so the drift is a
property of CUDA graphs here rather than of the quantisation.
**Replay qualification is the weak link at 6.54e-04** mean
`KL(live || replayed)` — 2 % of this candidate's KLD and 4 % of the gap to FP8, so no
ranking depends on it, but differences below ~1e-3 are not resolvable with these
artifacts.

### Head attribution: the K6 `lm_head` is nearly free

Replaying the identical stored hidden states through the BF16 head and through the
reconstructed K6 head (exllamav3's own `reconstruct_had_slice`, so it is the exact
serving matrix) isolates head error from body error:

| configuration | mean KLD | top-1 |
|---|---:|---:|
| head error alone (BF16 body, BF16 head vs K6 head) | 0.000367 | 99.31 % |
| body only (K4 body, same head both sides) | 0.026231 | 96.03 % |
| end to end, as served (K4 body + K6 head) | 0.026299 | 95.97 % |

The K6 head adds **6.78e-05** on top of the body
(95 % CI [4.63e-05, 9.01e-05]), i.e.
**0.26 % of total divergence**. Contrary to the common
assumption that `lm_head` is highly quantization-sensitive, at 6 bits on this model
it is not worth spending 1.6 GB to promote it to BF16 — that budget belongs to the
MLP stack, which owns the rest of the error.

### Single-window KLD, v1 protocol (kept for continuity)

This was the first measurement; the v2 protocol above supersedes it.

#### Teacher-forced KLD, full vocabulary

One frozen 2048-token window (exllamav3's bundled `wiki.utf8`, first 2048 tokens),
2047 scored positions, `KL(BF16 teacher || candidate)` across the entire
248320-token vocabulary with no top-k, 3 repeats, `--kv-cache-dtype auto` pinned
for every candidate, same teacher logits file for all of them. Protocol and
statistics follow the published Gilded Gnosis harness
(`rtx6kpro:scripts/glm52_exl3_shared_h_kld.py`): the headline value is the mean of
the per-run means and `run SD` is the sample SD across those means.

| candidate | mean KLD | run SD | SD across positions | resident weights |
|---|---:|---:|---:|---:|
| **this quant** (`--quantization exl3` + `exl3-b6` overlay) | **0.034030** | 0.000000 | 0.4628 | 19.21 GB |
| `unsloth/Qwen3.8-27B-NVFP4` control, same generation | **0.091457** | 0.000000 | 0.8036 | 23.42 GB |

**This quant is 2.7x closer to the BF16 teacher than the same-generation NVFP4
checkpoint, while holding 4.2 GB less VRAM.** That is the whole point of the
recipe: Trellis K4 spends 4.004 bpw where NVFP4 spends 4.50, and the savings buy
K6 attention instead of FP8.

`run SD = 0` for both candidates means the three repeats were bit-identical —
expected for the eager, `max_num_seqs=1`, prefix-caching-disabled configuration,
and a useful signal that the online-K6 cache reloads deterministically.

For scale, this project uses project-local, unvalidated descriptors (`<0.01` near-lossless,
`0.01-0.05` good, `0.05-0.1` noticeable, `>0.1` significant); they are not an external standard
and do not transfer across models, corpora or tokenizers. This quant sits in the "good" band; the
NVFP4 control sits in "noticeable".

Still measuring on the same window and teacher: this checkpoint, overlay off (attention stays BF16 in VRAM).

### Throughput — with CUDA graphs

Same GPU, `--max-num-seqs 8`, greedy, `ignore_eos`, 256 output tokens, warmup discarded.

| configuration | C1 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|
| **this quant + CUDA graphs** | **55.39** | **190.59** | **428.12** |
| `unsloth/Qwen3.8-27B-NVFP4` (Cutlass FP4 + graphs) | 49.09 | 171.78 | 371.06 |
| this quant, eager | 28.77 | 103.47 | 215.84 |
| `Qwen/Qwen3.8-27B` BF16 + graphs | 27.47 | 101.04 | 208.31 |
| `Qwen/Qwen3.8-27B` BF16, eager | 25.50 | 92.72 | 186.98 |

Graphs are worth **+92 % / +84 % / +98 %** here — roughly nine times what they buy the
BF16 model (+8-11 %), because eager EXL3 pays per-call dispatch on 193 quantized
matmuls. With graphs this quant is **both the smallest and the fastest** option
measured; the earlier claim that distribution parity against eager is exact has been **withdrawn** (it measured prefill, which FULL_DECODE_ONLY never captures) and replaced by a real decode probe: 24/32 exact sequences, with a BF16 control showing the same 24/32.

Graphs need the patch in
[local-inference-lab/vllm#312](https://github.com/local-inference-lab/vllm/pull/312)'s
sibling (autotune priming, filed separately) plus:

```bash
-e VLLM_EXL3_GRAPH_DECODE=1 ... --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Without that patch the loader refuses non-eager execution and you must pass
`--enforce-eager`, which costs the throughput above.

## Tradeoffs, stated plainly

- **The download is 28.31 GB for a 19.21 GB resident model.** Attention ships
  BF16 so the runtime can encode it at K6 (and, later, at another width) instead
  of being locked to a serialized choice. If you want download == VRAM, the
  `v-serialized-k6` variant is the one to ask for.
- **CUDA graphs need a patched loader** (see the throughput section). Unpatched, the
  loader refuses non-eager execution and you lose 46-50 % of decode throughput
  ([local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311)
  tracks the surrounding overlay work; the graph guard itself is next on the list).
  Decode is 58-60 % of the NVFP4 checkpoint's, dominated by the GEMM kernel rather
  than by graphs.
- **First load pays the K6 encode** (~16 min here) unless the cache directory is
  warm.
- **One runtime.** This checkpoint does not load in upstream vLLM, SGLang,
  transformers, TensorRT-LLM or llama.cpp. `exllamav3` itself can read the
  serialized K4/K6 halves, but it will not perform the runtime K6 encode of the
  BF16 attention.
- **KV cache** is left at engine default (`auto`); both NVFP4 references quietly
  ship FP8 KV schemes. Pin `--kv-cache-dtype` explicitly if you are comparing.

## Reproducing this quant

```bash
# 1. Convert everything at K4 (vision left BF16, head K6, mcg codebook).
python convert.py -i Qwen3.8-27B -o qwen38-k4 -w wd-k4 \
  -b 4 -hb 6 -mb 4 -vb 16 -cb mcg -d 0        # exllamav3 1.4.2 @ 5f3c537

# 2. Splice BF16 attention + MTP back over the K4 output; the converter cannot
#    emit BF16 for a decoder linear (load_fp16 forces float2half).
python splice_bf16_attn.py -q qwen38-k4 -s Qwen3.8-27B -o Qwen3.8-27B-K4

# 3. Regenerate metadata so tensor_storage describes the mix.
python util/add_safetensors_index.py -m Qwen3.8-27B-K4 --force
python util/add_quant_config.py -m Qwen3.8-27B-K4
```

`splice_bf16_attn.py`, the container-free runner used for all measurements here,
and the KLD harness are in the companion repo listed below.

## Prior art and credits

- [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — the base model
  (Apache-2.0). Architecture, chat template, tokenizer and generation defaults
  are theirs.
- [`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) —
  the recipe this one is modelled on: 4-bit MLP, 8-bit attention, BF16
  embeddings/vision/MTP. Built with NVIDIA TensorRT Model Optimizer.
- [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
  — second, independent confirmation of the same protection pattern, plus the
  last-8-layer MLP protection idea that the next iteration adopts.
- [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and the
  [vLLM recipe page](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — serving
  reference for context length, MTP and thinking modes.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3
  Trellis format, encoder and conversion pipeline (1.4.2 @ `5f3c537`).
- [Gilded Gnosis r34](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md)
  — the runtime that serves this: mixed-K EXL3 loader, B12X Trellis kernels, and
  the `exl3-b6` online overlay whose GLM-5.2 shared-expert pattern this quant
  borrows for dense attention. The K6-cache and KLD protocol documented there is
  what this quant is measured against.
- [malaiwah/progressive-tensors](https://github.com/malaiwah/progressive-tensors)
  — per-expert EXL3 segment provenance work; source of the per-bit error ladder
  that motivated K6 for the protected tensors.

## Companion repository

Recipe derivations, the measured composition of both NVFP4 references, the
runtime contract, the toolchain gaps, the KLD protocol and the iteration log:
**<https://github.com/malaiwah/qwen38-27b-exl3>**.

Successor checkpoints are published as separate repositories, each with its own
measurement receipts (`build-receipt.json`, `SHA256SUMS`, `quantization_manifest.json`).
