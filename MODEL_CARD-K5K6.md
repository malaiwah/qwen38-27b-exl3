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

# Qwen3.8-27B EXL3 K5/K6 — lower divergence than official FP8 at 71 % of its memory

> **Requires a custom runtime.** This checkpoint does **not** load in upstream vLLM,
> SGLang, TensorRT-LLM, llama.cpp, transformers, or stock exllamav3. It needs the
> Gilded Gnosis vLLM fork (public image below), explicit `--quantization exl3`, an exact
> `ignore` list, and — for CUDA graphs — an as-yet-unmerged patch
> ([PR #314](https://github.com/local-inference-lab/vllm/pull/314)). Treat it as an
> experimental, runtime-specific research artifact.

Mixed-precision EXL3 quant of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
@ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

**Headline evidence:** on a 10,480,640-position held-out suite (5,120 contexts, 842 source
clusters, calibration-overlapping documents excluded before selection), this build's mean
KL divergence from BF16 is **0.003210** [0.002982, 0.003480] against official FP8's
**0.005294**, with **97.52 %** top-1 agreement, and it diverges less on **5,105 of 5,120**
paired contexts — while holding 8.2 GiB less weight. Its
[hydrated sibling](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) is better
still, by 0.000450 on 4,922 of 5,120 contexts. [Details and limits](#fidelity).

| role | representation |
|---|---|
| MLP `gate_proj`, `up_proj` (64 layers) | EXL3 **K5**, `mcg` |
| MLP `down_proj` (64 layers) | EXL3 **K6**, `mcg` |
| attention: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48, `self_attn.{q,k,v,o}_proj` ×16 | **BF16 on disk**, encoded to **K6 at load** by the runtime's Trellis overlay — and **K5 or K4 instead, from the same download**, by one env var ([table](#attention-width-is-a-runtime-knob-and-native-context-needs-the-k4-build)) |
| `lm_head` | EXL3 **K6**, `mcg` |
| MTP draft head | **quantized** (attention K4, MLP K5/K6) |
| `embed_tokens`, vision tower (27 blocks), norms | BF16 |
| GatedDeltaNet `in_proj_a` / `in_proj_b` (96) | **FP16** passthrough (exllamav3 emits FP16 for unquantized linears; more mantissa than BF16, less range) |

`quantization_manifest.json` and `build-receipt.json` in this repo are authoritative for
composition. **`SHA256SUMS` covers the immutable payload only** — the four shards, the index,
the configs, the tokenizer and preprocessor files: 17 files, 30,597,223,337 bytes.
Documentation that changes with card edits (`README.md`, `.gitattributes`, `assets/**`) is
listed separately in `DOCS-SHA256SUMS`, because a hash map that claims to describe a build
must not move when prose does. An earlier revision mixed the two, listed a `crc32.txt` that
does not exist, and carried stale hashes for two files. The legacy `config.json → quantization_config` block keeps a single
`bits`/`codebook` pair for loader compatibility and **cannot** describe this mixed
checkpoint.

## Which of the four builds

Same architecture and tokenizer. The headline KLD column is the **v5 held-out suite
(5,120 contexts / 10,480,640 scored positions)**; the older overlap-corrected 127-context
v3 subset is kept beside it because the rest of this card's fidelity history is stated in it.
The two columns are different suites and are not comparable to each other.
Capacity uses each card's documented profile: hydrated, online and K4 are real RTX
5090 MTP-3 tests; context is MTP-3 with an 8.4 MP cap on a budget-capped RTX PRO 6000.
These profiles are not interchangeable
([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; the latter's hard-limit 5090 rerun is pending." src="assets/context-frontier-light.svg">
</picture>

*The figure plots the legacy v3 corrected means. The v5 ordering below is identical.*

| build | download | resident | v5 mean KLD | v3 corrected (legacy) | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | 0.007172 | ~180k | fidelity first, smallest download |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.57 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k | you want the attention width knob at launch |
| [-context](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 20.70 GB | **18.41 GiB** | 0.003509 | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window; hard-limit RTX 5090 check pending |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB | 17.89 GiB | 0.010604 | 0.029679 | 262,144 | smallest footprint, native context without any overlay |

Official `Qwen/Qwen3.8-27B-FP8` is 28.51 GiB resident at **0.005294** on the v5 suite
(0.012798 on the v3 subset) and runs on stock vLLM, which none of these do.

## Measured results

**30.60 GB download → 20.32 GiB (21.82 GB) resident weights**, measured from the engine's
own allocation log, versus 28.51 GiB for official FP8 and 21.34 GiB for Unsloth NVFP4
under identical flags.

## Fidelity

The headline evidence for this build is the **v5 held-out run: 5,120 contexts x 2,047
scored positions = 10,480,640 full-vocabulary positions**, with five candidates scored
against the same BF16 reference. It supersedes the 136-context v3 development suite and the
36-context v4 qualification as the primary result. Both are preserved below with their
numbers untouched — this section adds evidence, it does not revise theirs.

### Suite identity

`receipts/kld5-suite-manifest.json`, schema `qwen38-distribution-fidelity/6`, suite token
digest `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88`. 5,120 contexts,
2,047 scored positions each, **842 source clusters**. The corpus is 941 documents /
70,348,971 bytes fetched by `tools/fetch_corpus_v5.py`, logged in
[`receipts/kld5-corpus-fetch-log.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-corpus-fetch-log.json).

**Exclusion policy — contamination is zero by construction, not by audit.** Every discovered
document was scanned at every position for exact normalized 12-token overlap against the
exllamav3 calibration data **before** context selection, and any document with even one hit
was dropped whole: **44 of 941 documents excluded (43 code, 1 encyclopedic), 897 eligible**.
There is therefore nothing to correct after the fact and no "overlap-corrected subset" on
this suite — unlike the v3 numbers below, which needed one.

The suite is **token-disjoint from the v4 qualification suite** (0 of its 160 prior context
hashes is reachable), and its windows are exact-advance and non-overlapping: independently
verified as 5,120/5,120 unique context token hashes with 0 overlapping windows.

### Cumulative means at 10,480,640 positions

Body-only. Both operands replay through **one shared BF16 LM head**, so no candidate's head
quantization is counted and the five rows are comparable to each other.

| candidate | mean KLD | bootstrap 95 % CI | top-1 | max single position |
|---|---:|---|---:|---:|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | **0.002760** | [0.002540, 0.003020] | 97.70 % | 8.258 |
| **this build** (online K5/K6, attention K6) | **0.003210** | [0.002982, 0.003480] | **97.52 %** | 22.241 |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 0.003509 | [0.003220, 0.003852] | 97.44 % | 5.557 |
| `Qwen/Qwen3.8-27B-FP8` | 0.005294 | [0.004927, 0.005728] | 96.79 % | 10.714 |
| [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 0.010604 | [0.009640, 0.011746] | 95.76 % | 14.283 |

**This build sits 39 % below official FP8 on mean KL divergence and above it on top-1.**
Its maximum single-position divergence, 22.241, is the worst of the five — a tail property
that the mean does not show; the [distribution tail](#distribution-tail) below measures the
rest of that tail on one shard, and every quantile of it sits below official FP8's.

Per-candidate receipts are `receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`
([this build](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-k5k6.json)),
schema `qwen38-kld-ladder-cumulative/2`, built by `tools/kld_aggregate.py` from ten verified
per-shard reports produced by `tools/kld_ladder.sh`: capture six models over 512 contexts,
replay five candidates, verify, delete 64 GB of hidden states, next shard.

### Paired per-context differences

Bootstrap over 10,000 resamples, seed 1, 842 source clusters, receipt
[`receipts/kld5-10M-paired.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-paired.json).
Negative means the first term diverges less.

| pair | mean difference | 95 % CI | contexts where the first term is lower |
|---|---:|---|---:|
| **this build - official FP8** | **-0.002084** | [-0.002249, -0.001942] | **5,105 / 5,120** |
| hydrated - this build | **-0.000450** | [-0.000469, -0.000433] | **4,922 / 5,120** |
| hydrated - FP8 | -0.002534 | [-0.002708, -0.002383] | 5,118 / 5,120 |
| context edition - FP8 | -0.001785 | [-0.001884, -0.001697] | 5,109 / 5,120 |
| K4 - FP8 | +0.005310 | [+0.004710, +0.006019] | 7 / 5,120 |

**Versus official FP8 this is settled:** lower divergence on 5,105 of 5,120 contexts, with a
paired CI far from zero, while holding 8.2 GiB less weight.

**Versus the hydrated sibling it is also settled, and this build loses.** Same K5/K6 recipe,
attention encoded ahead of time instead of at load, 20.31 against 20.32 GiB resident:
hydrated is lower by **0.000450** on average
[-0.000469, -0.000433] and lower on **4,922 of 5,120 contexts (96.1 %)**. The margin is
small — about 14 % of this build's mean, and below the ~6.5e-04 live-vs-replayed
qualification floor quoted for the older harness, so it is a robust ordering of the replayed
captures rather than a promise of a perceptible difference in serving. But it is consistent,
one-directional and no longer in question. Choose this build for the **launch-time attention
width knob** and the ~206k demonstrated context that comes with K5 attention; choose
[hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) for the last
fraction of fidelity and a 9 GB smaller download. Nothing else separates them.

### Ladder stability

The run was aggregated at 1M / 2M / 5M / 10M scored positions. For the hydrated candidate,
carried through all four checkpoints, the cumulative mean reads **0.002700 / 0.002759 /
0.002699 / 0.002760** — a spread of 6.1e-05 across a tenfold increase in scored positions.
The headline numbers are not an artifact of where the run stopped.

### Distribution tail

A mean and a top-1 rate say nothing about the worst positions, and one exact maximum is not a
tail either. This is the whole right tail, measured on **shard 0 of the same suite — 512
contexts, 1,048,064 scored positions** — the identical contexts for all five candidates.
Receipts [`receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts),
schema `qwen38-kld-ladder-cumulative/2`, built by `tools/kld_aggregate.py`; this build's row is
[`receipts/kld5-1M-tail-k5k6.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-tail-k5k6.json).

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | share of positions above 0.1 | above 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hydrated | 0.002700 | 0.00109 | 0.0082 | 0.0276 | 0.1319 | 0.463 | 3.735 | 0.1534 % | 0.00219 % |
| **this build** (online K5/K6) | **0.003141** | **0.00128** | **0.0099** | **0.0321** | **0.1446** | **0.498** | **5.507** | **0.1820 %** | **0.00200 %** |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | 0.1642 | 0.587 | 3.749 | 0.2287 % | 0.00305 % |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | 0.2438 | 0.812 | 5.296 | 0.3912 % | 0.00592 % |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | 0.5555 | 1.870 | 7.565 | 1.2604 % | 0.03807 % |

**Method, in one sentence:** every `qwen38-fidelity-report/2` replay accumulates a **560-bin
log-spaced histogram of per-position KLD** (`KLD_HIST_LOG10_LOW=-12.0`,
`KLD_HIST_LOG10_HIGH=2.0`, `KLD_HIST_BINS_PER_DECADE=40` in `tools/fidelity.py`) whose bin
counts add across shards, which is what makes cumulative quantiles possible at all.

**What it says for this build.** The ordering at p50, p95, p99, p99.9 and p99.99 is the same
as the ordering of the means, so the mean is not hiding a worse tail. This build's tail is
**below official FP8's at every measured quantile** — 0.0321 against 0.0531 at p99, 0.1446
against 0.2438 at p99.9, 0.498 against 0.812 at p99.99 — and it has the **smallest share of
positions above 1.0 of the five**, 21 positions in 1,048,064 against FP8's 62. It stays
behind the hydrated sibling everywhere in the same ordering the means report. The exact
maximum is the one place it does not lead: 5.507 on this shard, just above FP8's 5.296, and
22.241 over the full ten-shard run — a single position, with 0.00200 % of positions above 1.0
behind it.

**Scope, stated exactly:**

- This is one 1,048,064-position shard, not the full 10,480,640-position run. The ten-shard
  run predates the histogram, so it could not be recomputed without re-running it.
- The quantiles are **bin-bounded, not exact**: each receipt carries `lower` / `upper` /
  `estimate` per quantile, with a relative bin width of about 5.6 %. The **maxima and the
  exceedance counts are exact**.
- The 10M receipts remain the source for the full-run means, intervals and paired results;
  nothing here replaces them.

### What the v5 numbers do not say

- **Absolute KLD is suite-specific.** v5 values are **not** comparable to the v3 values in
  the next section: the corpus mix differs, and K4 reads 0.029679 there against 0.010604
  here. Only within-suite ordering and paired differences transfer between the two.
- **Cumulative percentiles come from one shard, not from all ten.** The ten shard reports of
  the 10M run carry no token-level KLD histogram, so median/p95/p99/p999 could not be
  recombined across them. The [tail table above](#distribution-tail) closes that gap on
  **shard 0** (`receipts/kld5-1M-tail-*.json`); across all 10,480,640 positions only the
  means, the intervals, the paired results and the **exact global maximum** exist.
- **The captures are gone.** Hidden states were deleted shard by shard to fit 135 GB of
  scratch, so unlike the [v3 dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3)
  this run is reproducible from the pinned corpus fetch log and suite manifest, not from
  published captures.

## Prior receipt: v3 development suite (136 contexts, 278,392 positions)

Superseded as the headline by the v5 run above; kept because every earlier claim on this card,
including the as-served head cost and the attention-width ladder, is stated on this suite.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fidelity-vs-size-dark.svg">
  <img alt="Overlap-corrected mean KL divergence from BF16 versus resident weight footprint. This quant at 21.82 GB and 0.0079; Qwen official FP8 at 30.61 GB and 0.0128; the previous K4 iteration at 19.21 GB and 0.0297; Unsloth NVFP4 at 22.91 GB and 0.0927. Right panel: top-1 agreement, 96.95 / 96.18 / 94.48 / 90.49 percent." src="assets/fidelity-vs-size-light.svg">
</picture>



`KL(BF16 reference ‖ candidate)`, two passes, no top-k, float32 within vocabulary chunks
accumulated in float64 across chunks, one shared BF16 LM head for both operands,
source-cluster bootstrap. Corpus is separately sourced Gutenberg / arXiv / Wikipedia /
CPython. The builder requested nine Wikipedia languages but tolerated under-filled strata, so
the **frozen suite is English, German and Russian only** (the multilingual stratum is 6 German
and 1 Russian context). Its fixed-stride 160-character scan originally reported zero calibration
hits; an offset-independent 12-token scan later found exact overlap in 2/41 source documents.
The suite, shared BF16 head, sentinels, comparator captures and this checkpoint's captures are
published as a
[dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3).
The complete original 136-row report is under
`reports-k5k6/report-k5k6-online-k6-analysis.json`; its capture is under
`candidate-hidden/k5k6-online-k6/`.

| candidate | resident | mean KLD | bootstrap 95 % CI | median | p99.9 | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| **this quant** | **21.82 GB** | **0.008157** | [0.00607, 0.01067] | 0.001529 | 0.475 | **96.97 %** |
| `Qwen/Qwen3.8-27B-FP8` | 30.61 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 0.773 | 96.22 % |
| `malaiwah/Qwen3.8-27B-K4` (previous) | 19.21 GB | 0.030736 | [0.02238, 0.04073] | 0.004218 | 1.758 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 22.91 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 4.509 | 90.53 % |

**Overlap-corrected subset:** conservatively removing all nine analysis contexts from either
affected source document gives this quant **0.007945**, FP8 **0.012798**, K4 **0.029679**,
and NVFP4 **0.092727** over 127 contexts. The 38 % FP8 advantage and ordering survive. The
table above remains the original full-suite receipt, not a claim of zero lexical overlap.

Paired over the same contexts: **-0.004969** versus official FP8
(95 % CI [-0.00643, -0.00371], **136/136 contexts**), i.e.
**38 % lower mean KL divergence than FP8 while holding 8.8 GB less weight**;
and **-0.022579** versus the previous K4 release (136/136 contexts).

**Body-only versus as-served, measured on this v3 suite.** Every row above — and every row
in the v5 section — replays both operands through one shared BF16 head, so no candidate's
head quantization is counted; that is what makes them comparable. The only measurement of
what this build's own K6 head costs when served was made here, on the v3 suite, and has
**not** been repeated on v5. On the original 136-context receipt the K6 head adds +0.000127
(95 % CI [+0.000105, +0.000148]) for 0.008284 as served. On the overlap-corrected subset,
body-only is **0.007945** and the measured as-served result is **0.008078**
(+0.000132, 95 % CI [+0.000114, +0.000151], 7/127 contexts favour the quantized
head). Promoting the head to BF16 would cost **+1.589 GB**, so this checkpoint keeps K6.
That increment is a v3-suite quantity: it must not be added to the v5 means above, which
are a different suite.

Controls published with the dataset: runtime-repeat noise floor **0.000000** (three
captures of the same runtime) and harness self-check 0.000000. A third control,
"CUDA-graph parity 0.000000", was **withdrawn**: it captured a prefill forward, and
`FULL_DECODE_ONLY` captures no prefill graph, so it could not have measured the decode
path. Re-measured properly on real decode steps, graph and eager agree on 24/32 greedy
32-token sequences with mean |Δ logprob| 0.0118 on the chosen token; unquantised BF16
on this same build drifts identically (24/32, 0.0128), so this is a property of CUDA
graphs here and not of the quantisation.
**Weakest control:** live-vs-replayed logit qualification is 6.54e-04, so differences
below ~1e-3 are not resolvable with these artifacts. The FP8 gap is **7.6x** that floor
and the K4 gap **34.5x** it; the K6-head increment (0.000127) and the K5-vs-FP8 gap
(0.000991) are **at or below** it and are reported as unresolved point estimates, not
as established differences. The KLD magnitudes here are only comparable within this suite — thresholds from
other models, corpora or tokenizers do not transfer.

## Attention width is a runtime knob, and native context needs the K4 build

Attention weights ship in BF16 and are encoded to EXL3 Trellis **at load**, so the width is
a launch-time choice rather than a property of the download: `VLLM_EXL3_ONLINE_TRELLIS_BITS`
accepts 3-8. One checkpoint, several operating points.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-per-card-dark.svg">
  <img alt="GPU memory required versus context served for K5/K6 at attention K6 and K5 and for the K4 build, with lines at 16, 24 and 32 GB usable VRAM and a vision-safe 0.95 utilisation ceiling. Contexts actually run on an RTX 5090: 185,600 at K6, 206,400 at K5, 262,144 for the K4 build. Native 262,144 is reached only by the K4 build." src="assets/context-per-card-light.svg">
</picture>

**The width ladder was measured on the v3 suite only.** The v5 run scored this build at its
default K6 attention (0.003210); the K5 and K4 attention widths have not been rerun on v5, so
the three rows below are v3-suite quantities and belong beside 0.007945, not beside 0.003210.

| `VLLM_EXL3_ONLINE_TRELLIS_BITS` | resident weights | corrected v3 mean KLD | top-1 |
|---|---:|---:|---:|
| **6** (default) | 20.32 GiB | **0.007945** | 96.95 % |
| **5** | 19.82 GiB | 0.011801 | 96.28 % |
| **4** | 19.05 GiB | 0.026619 | 94.48 % |

On the same overlap-corrected 127 contexts, K5 costs **+0.003856** versus K6 and K4
costs **+0.018673**. K5's mean is 0.000997 below official FP8's corrected 0.012798 —
at this harness's ~1e-3 replay-resolution floor, so treat it as an unresolved point
estimate, not an advantage. The original 136-context width reports remain in the dataset.

### Measured on a real RTX 5090, by an independent tester

**Native 262,144 context does not fit this checkpoint on a 32 GB card.** An earlier revision
of this card claimed it did, from a memory simulation on a 96 GB card. That was wrong twice:
the simulation omitted MTP's KV (with `num_speculative_tokens: 3` the engine needs
**9.13 GiB** for 262,144 tokens, not 8.18) and it assumed 31.84 GiB usable where a 5090
reports **31.39**. Corrected, with hardware numbers (TP1, FP8 E4M3 KV, MTP-3, decode-only
CUDA graphs, vision enabled):

| attention | max seqs | util | KV attained | context | outcome |
|---|---:|---:|---:|---:|---|
| K6 | 8 | 0.95 | 187,050 tok / 6.71 GiB | **185,600 run** | stable, multimodal-safe |
| K6 | 8 | 0.98 | 202,185 tok / 7.55 GiB | — | text fine, **a 3,264-token image OOMed** with 33 MiB free |
| K5 | 8 | 0.95 | 212,255 tok / 7.55 GiB | **206,400 run, 205,021 tokens, exact retrieval passed** | best demonstrated |
| K5 | 8 | 0.95 | 7.33 GiB available vs 9.13 needed | 262,144 refused | short by 1.80 GiB |
| K5 | 4 | 0.96 | 7.99 vs 9.13 | 262,144 refused | short by 1.14 GiB |
| K5 | 4 | 0.97 | 8.30 vs 9.13 | 262,144 refused | short by 0.83 GiB |
| **K4 build**, K6 | 8 | 0.98 | 289,577 tok capacity | **262,144 configured** | native context fits |

So on a 32 GB Blackwell card with MTP-3: **use K5 attention for ~206k with retrieval
verified**, keep utilisation at **0.95** if you serve images (0.98 leaves no vision headroom),
and if you need hardware-qualified native 262,144 use
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) — it is the only
family member proven at the physical limit, at 3.8x the divergence. The context edition starts
at native length with MTP-3 and an 8.4 MP image ceiling under a capped 32 GB-equivalent engine
budget, but its hard-limit RTX 5090 rerun is pending. Closing this online-K6 build's last
0.83 GiB through utilisation alone would need ~0.997, with no runtime headroom. On 48 GB and
larger, native context fits at K6 with the best fidelity.

KV costs **37.4 KB/token with MTP-3** (16 full-attention layers, 4 KV heads, head_dim 256;
the other 48 layers are Gated DeltaNet and hold per-sequence state instead) and 33.5 KB/token
without it — turning MTP off is worth about 11 % more context if you would rather have length
than 2x decode.

**4-bit KV is not available on this architecture:** the runtime's generic NVFP4 KV path
requires SM100 trtllm-gen and is rejected on SM120, and GLM-5.2's `nvfp4_ds_mla` cache is
MLA-specific, which Qwen3.8 is not.

Thanks to the tester who ran this on real hardware and caught the overclaim.

## Prior receipt: v4 post-selection qualification (36 contexts)

The v3 numbers come from the suite that guided recipe selection. This was the first test that
did not: **160 new contexts from 100 documents with zero intersection with the development
suite** (context token hashes 0/160, document names 0/100, content hashes 0/100), partitioned
by whole source cluster, run **once**, with no recipe changed afterwards. The v5 suite above
is token-disjoint from this one too, is **32x its size** (5,120 contexts / 10,480,640 scored
positions against 160 / 327,520) with the same conservative rule applied at document-scan
time instead of afterwards, and supersedes it as the headline held-out evidence; the ordering
it found is unchanged there.

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
Absolute magnitudes remain suite-specific, and these 36 contexts (73,692 scored positions)
are not comparable to the v5 suite's 5,120 — where this build reads 0.003210 against FP8's
0.005294 and wins 5,105/5,120 paired contexts, on 142x as many scored positions.

## Public capability — MMLU-Pro, item-paired against BF16

70 MMLU-Pro questions, 14 official categories, 5 per category, pinned
`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, official five-shot category
prefixes, greedy, thinking at low reasoning effort, 5,120-token completion cap. The BF16
control ran first and the acceptance rule was frozen in
[`receipts/public-capability-plan.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-plan.json)
before any candidate result was seen. All six models answered the same 70 items in the same
order through the same extractor, so every candidate row is paired item-by-item against that
control.

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.8-27B` BF16 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | [bf16](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16.json) |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 58/70 (82.9 %) | [72.4 %, 89.9 %] | 56/57 | **90.7 %** | 1 | 2 | 3 | [ctx](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-ctx.json) |
| [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 57/70 (81.4 %) | [70.8 %, 88.8 %] | 55/57 | 88.1 % | 2 | 2 | 4 | [k4](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k4.json) |
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 54/57 | 85.6 % | 3 | 2 | 4 | [hyd](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-hyd.json) |
| `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | [fp8](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-fp8.json) |
| **this build (online K5/K6)** | **55/70 (78.6 %)** | **[67.6 %, 86.6 %]** | **54/57** | **85.6 %** | **3** | **1** | **4** | [k5k6](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k5k6.json) |

### The pre-registered bar, and this build's verdict

The frozen plan accepts a candidate when BF16-pass retention has a **Wilson 95 % lower bound at
or above 0.90** and **no category loses more than two BF16 passes**. The category clause is met
by all five candidates — the worst case is two passes in philosophy, for this build and for the
hydrated build — so the retention lower bound is the only clause that ever fails.

**Only the context edition clears the bar, at 90.7 %.** K4 and official
`Qwen/Qwen3.8-27B-FP8` read 88.1 %. **This build reads 85.6 % (54/57) and does not clear it**,
as does the hydrated build. Three of BF16's 57 passes flipped to failures here and one BF16
failure flipped to a pass, for 55/70 absolute. That is a measured shortfall, published exactly
as measured, with nothing retuned afterwards.

**What the numbers do not say.** 55/70 is the lowest absolute count in the matrix, and that is
**not** a ranking: every interval in the table overlaps every other interval, including the
BF16 control's and official FP8's. This build is not shown to be worse than official FP8, K4,
the hydrated build or the context edition on knowledge-and-reasoning tasks, and none of them is
shown to be worse than it. The KLD advantage this card reports over official FP8 is a
distribution-fidelity result on 10,480,640 scored positions; it makes no capability claim, and
this 70-item suite neither confirms nor contradicts it.

### Why a 70-item suite cannot certify this bar

With 57 BF16 passes as the paired denominator, **56/57 is the smallest count whose Wilson 95 %
lower bound clears 0.90** (56/57 → 90.7 %; 55/57 → 88.1 %; 54/57 → 85.6 %). A single paired
regression is therefore the entire budget, and no result that gives up two can pass, however
sound the build. The suite has too few items to certify the bar it pre-registered, and at this
size it separates nothing — the point applies to official FP8 exactly as it applies to the EXL3
builds. Read it as a **power limitation of a 70-item draw, not as evidence that any of these
checkpoints is broken**.

### Two protocol facts that bound the reading

- **Exact-answer agreement is 0/70 for every EXL3 candidate, and 1/70 for official FP8** (one
  math item, a 113-token answer both models pass). Long chains of thought differ token-wise on
  essentially every item, so pass/fail outcome is the only meaningful pairing unit; nothing
  here is a generated-text match claim.
- **Four BF16 items end at the 5,120-token completion cap with no letter emitted and are
  scored as failures** under the plan's frozen addendum, so the control itself is depressed by
  the cap; per-model counts are in the table (this build: 4). The earlier 2,048-cap control,
  where BF16 lost 7/70 to truncation, is retained unchanged at
  [`receipts/public-capability-bf16-superseded-cap2048.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16-superseded-cap2048.json).

### Status of this evidence

This is a **first public, licence-compatible, item-paired benchmark, not a leaderboard claim**.
The honest next step is more items, which is the plan's own P1: HumanEval+/MBPP-style
executable cases, IFEval-style constraint following, tool schemas, and a larger MMLU-Pro draw.
No capability claim on this card graduates before that.

Harness
[`tools/public_capability.py`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/public_capability.py),
sweep runner
[`tools/run_public_capability.sh`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/run_public_capability.sh),
suite
[`receipts/public-capability-suite-mmlupro-70.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-suite-mmlupro-70.json).
Every run receipt carries the per-item raw request, raw response, extracted letter, gold letter
and digests.

## Downstream task retention — 40-task smoke suite (prior, narrower evidence)

This ran before the MMLU-Pro suite above and is kept unchanged. It is the narrower evidence:
self-generated tasks with contract checks, not a public benchmark.

On 40 deterministic generated tasks (10 each arithmetic, executable builtins-only code,
exact-list instruction following and tool-call schema), BF16 and every comparator scored
40/40. This build had **zero regressions** and matched BF16's exact final-answer text on
**34/40**; all six differing answers still passed their contracts. Wilson 95 % lower bound
is 91.2 %. This is a transparent smoke suite, not a public leaderboard; full responses are in
the [run receipt](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/tasks-v2-k5k6.json);
its extracted-value agreement field is superseded by the
[strict rescore](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/task-retention-v2-strict-rescore.json).

## Throughput

Median of 3 runs, `--max-num-seqs 8`, greedy, 256 output tokens; prefill measured with
exact token-count prompts.

| configuration | TG C1 | TG C4 | TG C8 | PP 2k | PP 6k |
|---|---:|---:|---:|---:|---:|
| **this quant + graphs + prefill dispatch** | 56.6 | 199.6 | 404.6 | **5,050** | **5,146** |
| **+ MTP-3 speculative decoding** | **113.8** | 206.8 | — | 2,292* | — |
| this quant, graphs only (no prefill patch) | 56.5 | 199.6 | 402.7 | 2,369 | 2,362 |
| `unsloth/…-NVFP4` | 48.9 | 171.4 | 369.7 | 14,528 | 13,468 |
| `Qwen/…-FP8` | 46.3 | 163.3 | 342.5 | 10,667 | 10,474 |

\* MTP figure predates the prefill patch; the two are independent and compose.

**Best decode throughput of every candidate measured, and 2.3-2.5x the single-stream
rate of FP8/NVFP4 with speculative decoding on.** Speculative decoding uses this
checkpoint's **quantized** draft head: 58.2 % of drafted tokens accepted, **1.745 accepted
draft tokens per step**, so **2.745 output tokens per speculative iteration** once the
verifier's own token is counted (acceptance 77.5 / 57.2 / 39.8 % by draft position). The
comparators are measured **without** speculative decoding, so the 113.8 figure is this
checkpoint against itself, not a like-for-like format comparison.

**Prefill improved 2.1x** in [PR #316](https://github.com/local-inference-lab/vllm/pull/316)
plus [PR #318](https://github.com/local-inference-lab/vllm/pull/318): #316 adds the
reconstruct+`hgemm` path for rows >= 128, while #318 routes native K6/MCG B12X shards to
that path instead of the decode-shaped kernel. The change is slower below m=64 and
4.1-5.2x faster at m=2048; decode is untouched. It is **not** bit-exact — fp16 summation
order changes, costing +0.43 % measured divergence with top-1 unchanged to four decimals.
`VLLM_EXL3_PREFILL_RECONSTRUCT_M=0` restores the exact path.

**Prefill remains the weak axis, and it is now attributed rather than suspected.** A 2x2
over attention representation and MLP kernel isolates it: the MLP kernel is worth
**2.13-2.26x**, the online attention overlay only **1.05-1.11x** — so the overlay is not
the bottleneck, which refutes what this card previously said. `ext.hgemm` measures at
cuBLAS parity (0.92-1.06x) and a larger prefill chunk changes nothing, so the tuning
levers are spent. The residual is the GEMM's **dtype**: at these shapes an FP8 matmul runs
1.85-2.02x faster than fp16 on this card, which is almost exactly official FP8's prefill
lead. Closing it needs dequant emitted into an FP8 GEMM, not another dispatch tweak — the
work is tracked in [docs/26](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/26-prefill-attribution.md).

## Serving

**Two recipes, because the published image predates the patches.** The pinned digest below
was built on 2026-08-10 from vLLM `e2666d9a`; the maintained patch stack remains unmerged,
so **that image cannot contain it**. Recipe A is what the image runs unmodified. Recipe B
replaces one module before launch. The exact module used for the headline table is preserved
below; a later superseding module adds K6/MCG prefill routing. Anything claiming graph decode
or reconstructed prefill on recipe A is wrong.

### Recipe A — unmodified image, eager only

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro -v /cache:/cache \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6 \
    --served-model-name qwen38 --quantization exl3 --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --host 0.0.0.0 --port 8000
```

Measured on this path: **28.8 tok/s** decode at concurrency 1 and **2.4k tok/s** prefill —
that is the honest floor without the patches.

### Recipe B — headline configuration (graphs + prefill dispatch)

Replace one module inside the container, then launch. The **exact headline-table module** is
[`vllm-exl3-prefill-dispatch.py` at `21c2b6d`](https://github.com/malaiwah/qwen38-27b-exl3/blob/21c2b6d708de011c2d73fdd8b1806e2e49c0ed71/tools/vllm-exl3-prefill-dispatch.py),
`sha256:cb9e60024057e8097237a5518e6469b15f73e4139cc37f1f67e9c1485b44aedd`. It
incorporates [PR #314](https://github.com/local-inference-lab/vllm/pull/314)
(`7917c928`) and [PR #316](https://github.com/local-inference-lab/vllm/pull/316)
(`8451183e`). The current recommended
[`main` module](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/vllm-exl3-prefill-dispatch.py),
`sha256:2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee`,
also incorporates [PR #318](https://github.com/local-inference-lab/vllm/pull/318)
(`5da0bcda`) and is what the later native-context receipt tested. The 8,192-token
three-run table was not relabelled as a measurement of that later file.

```bash
set -euo pipefail
PATCH=$PWD/vllm-exl3-prefill-dispatch.py
printf '%s  %s\n' \
  2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee \
  "$PATCH" | sha256sum -c -

docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro -v /cache:/cache \
  -v "$PATCH:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
  -e VLLM_EXL3_GRAPH_DECODE=1 \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6 \
    --served-model-name qwen38 \
    --quantization exl3 \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --host 0.0.0.0 --port 8000
```

No published image digest contains this patch. Until one does, recipe B is a
content-verified source mount, not an immutable runtime artifact. The container listens on
all interfaces internally, but Docker publishes the port to host loopback only. For remote
clients, keep that binding and put an authenticated TLS proxy in front; do not expose this
unauthenticated generation endpoint directly.


Load-bearing details:

1. **`--quantization exl3` is mandatory** — auto-detection only fires for GLM-5.2
   metadata.
2. **Both performance features need unmerged patches**, which is why recipe A and recipe B
   are separate. `VLLM_EXL3_GRAPH_DECODE=1` + `cudagraph_mode: FULL_DECODE_ONLY` need
   [PR #314](https://github.com/local-inference-lab/vllm/pull/314) (without it the loader
   refuses non-eager execution and decode loses ~46-50 %). Reconstructed generic EXL3 prefill
   needs [PR #316](https://github.com/local-inference-lab/vllm/pull/316). The maintained
   module additionally uses [PR #318](https://github.com/local-inference-lab/vllm/pull/318)
   to route native K6/MCG B12X shards into that path at prefill row counts. Current upstream
   heads have no CI result: pre-run jobs are blocked by repository policy labels, not by a
   demonstrated failure.
3. **The `ignore` list is mandatory and its anchoring is subtle.** Prefixes carry no
   leading `model.`, so `re:.*visual\..*` matches while `re:.*\.visual\..*` silently does
   not — and the wrong pattern **crashes** startup
   ([#311](https://github.com/local-inference-lab/vllm/issues/311), fixed by
   [PR #312](https://github.com/local-inference-lab/vllm/pull/312)). The tested config also
   ignores `mtp.*` from the **generic BF16 online overlay**. EXL3 ownership is checked first,
   so the serialized quantized draft still uses the EXL3 loader; the ignore prevents accidental
   re-encoding only when a draft projection is not EXL3-owned.
4. **`--mm-processor-kwargs '{"truncation":false}'` is required for images** whose expanded
   token sequence exceeds 2,048, or requests fail with HTTP 400
   ([#313](https://github.com/local-inference-lab/vllm/issues/313)). Verified here: a
   2044×1622 image answers correctly with the flag.
5. First load encodes 208 attention projections to K6 (~16 min cold); point
   `VLLM_EXL3_ONLINE_CACHE_DIR` at persistent storage.

### Client contract, inherited unchanged from upstream

Chat template, tokenizer, preprocessors, generation defaults and vocabulary (248,320,
untied head) are byte-identical to upstream. `generation_config.json`: `temperature 1.0`,
`top_p 0.95`, `top_k 20`. Qwen's recommendation for non-thinking mode is
`temperature 0.7`, `top_p 0.8`, `top_k 20`, `presence_penalty 1.5`.

Thinking control is `chat_template_kwargs`: `{"enable_thinking": false}`, or
`{"reasoning_effort": "..."}` where the **only valid values are `xhigh` (default),
`medium` and `low`** — the upstream template raises on `high`.

Context: **262,144 native**; verified here only to 8,192. Upstream's 1M procedure is
static YaRN, not a bare `max_position_embeddings` bump: it needs nested `rope_parameters`
with `rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144`,
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and `--max-model-len 1000000`, and Qwen warns it costs
short-context quality. **Untested on this runtime.**

## Verification performed, and not performed

Done: structural audit (1,199 logical tensors reconstructed, matching upstream), serving
under the pinned image, greedy text, 96×96 and 2044×1622 image answers, MTP acceptance
from server counters, 3-run throughput with <1 % dispersion, prefill at exact token
counts, eager-vs-graph decode parity on real decode steps (24/32 exact sequences, with a
BF16 control showing the same 24/32), the three fidelity suites above (v5 held-out at
10,480,640 scored positions, v4 qualification, v3 development), and 40/40 deterministic
task-retention smoke with zero BF16 regressions.

**Context length is the sharpest gap between what is claimed and what is tested.** Fidelity
and functional tests ran at `--max-model-len 8192`. The 262,144 rows above are *engine
allocation and startup* at that length, not generation, retrieval or accuracy at length. No
native-262K generation, no YaRN-1M run, and no long-context retrieval measurement exists yet.

**Measured but not established:** the paired MMLU-Pro run above has now covered this build and
all five comparators, and this build is a **measured shortfall** against the pre-registered
bar — 54/57 BF16-pass retention, Wilson 95 % lower bound **85.6 %** against a required 90 %,
which only the context edition clears. At 70 items the suite cannot certify that bar for any
result that gives up more than one paired pass, and its intervals separate no two candidates,
so there is still no wider public task evidence for this build: no executable-code,
constraint-following, tool-schema or larger-draw result (the plan's own P1), and no
GPQA/HumanEval-style run. Also not done:
real OCR/chart/video evaluation, long-context retrieval or perplexity for this build,
native-262K or YaRN-1M generation, multi-GPU/TP>1, non-SM120 hardware, and quant-specific
safety testing. KLD and the small generated smoke do not establish broad task capability.

## Safety and intended use

Upstream does not disclose training-corpus composition, knowledge cutoff, safety
evaluation, or detailed intended-use limits, and this quant adds none. Quantization can
alter refusal behaviour and calibration even when average divergence is low, and **no
quant-specific safety regression testing has been performed**. Intended for research and
local inference evaluation. Inherits Apache-2.0 from upstream.

## Prior art and credits

- [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) and
  [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — base and
  official 8-bit comparator.
- [`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) — the
  role-aware recipe this design is modelled on (4-bit MLP, 8-bit attention, protected
  embeddings/vision/MTP), and the source of the memory ceiling used here.
- [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) —
  independent confirmation of the same protection pattern.
- [`malaiwah/GLM-5.2-EXL3-TR3-MTP78`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78)
  — prior art for quantizing the MTP draft head: 3 bpw matched BF16 acceptance length at
  one fifth the size.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) 1.4.2 @ `5f3c537`
  — EXL3 format, encoder and conversion pipeline.
- [Gilded Gnosis r34](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md)
  — the runtime, its online-K6 overlay, and the distribution-fidelity protocol
  ([Kimi-K3 1024×2048](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/distribution-fidelity-1024x2048.md))
  this evaluation follows.
- [malaiwah/progressive-tensors](https://github.com/malaiwah/progressive-tensors) —
  per-expert EXL3 provenance work and the per-bit error ladder.

## Companion repository

Recipes, receipts, the fidelity harness, the response to an independent review, and the
open items: **<https://github.com/malaiwah/qwen38-27b-exl3>**.

Upstream contributions from this work:
[#311](https://github.com/local-inference-lab/vllm/issues/311) /
[PR #312](https://github.com/local-inference-lab/vllm/pull/312) (overlay fallback crash),
[PR #314](https://github.com/local-inference-lab/vllm/pull/314) (CUDA-graph decode),
[#313](https://github.com/local-inference-lab/vllm/issues/313) (Qwen3.8 vision
truncation).
