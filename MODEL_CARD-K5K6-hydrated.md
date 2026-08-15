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

# Qwen3.8-27B EXL3 K5/K6, attention serialized on disk — 9 GB smaller, 5x faster cold start, and measurably closer to BF16

> **Requires a custom runtime.** Does **not** load in upstream vLLM, SGLang, TensorRT-LLM,
> llama.cpp, transformers, or stock exllamav3. It needs the Gilded Gnosis vLLM fork with
> explicit `--quantization exl3` and an exact `ignore` list. Treat it as an experimental,
> runtime-specific research artifact.

Companion to
[`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6).
Same MLP recipe; one difference: **attention is quantized to EXL3 K6 on disk by the
calibrating converter** instead of shipping BF16 and being encoded to K6 by the runtime at
every cold start. Measured consequences:

| | BF16 attention (sibling) | **this build** |
|---|---:|---:|
| download | 30.57 GB | **21.61 GB** (−29 %) |
| resident weights | 20.32 GiB | 20.31 GiB (unchanged) |
| first load, cold | 957 s (encodes 208 projections) | **178 s** (5.4x faster) |
| restart with a warm cache | 173 s | 178 s (same) |
| persistent encode cache | required, and must be writable | **not used** |
| **v5 held-out mean KLD from BF16** (body-only, 10,480,640 positions) | 0.003210 | **0.002760** (−14 %, paired −0.000450 on 4,922/5,120 contexts) |
| v3 mean KLD from BF16, body-only (prior receipt) | 0.007945 | 0.007172 (−9.7 %, but see the floor caveat) |
| v3 mean KLD, as served (prior receipt) | 0.008078 | 0.007300 (measured, not estimated) |
| attention width | K6 / K5 / K4, chosen at launch | fixed at K6 |

**Choose this build** for the smaller download, the best fidelity in the family, and a start
that needs no writable cache. Note the nuance: a *warm* online cache also starts in 173 s, so
the 5.4x is about first load and about environments where a persistent cache is awkward —
read-only images, ephemeral containers, many nodes. **Choose the sibling** if you need context: its runtime knob trades
fidelity for KV room, and on a 32 GB card that difference matters (see
[context](#context-capacity)).

## Which of the four builds

Same architecture and tokenizer. The headline KLD column is the **v5 held-out suite,
10,480,640 scored positions**; the narrower overlap-corrected 127-context v3 subset is kept
beside it as the prior receipt. Capacity uses each card's documented profile: hydrated,
online and K4 are real RTX 5090 MTP-3 tests; context is MTP-3 with an 8.4 MP cap on a
budget-capped RTX PRO 6000.
These profiles are not interchangeable
([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; the latter's hard-limit 5090 rerun is pending." src="assets/context-frontier-light.svg">
</picture>

The figure's axis is still the overlap-corrected v3 receipt, because that is what the plotted
asset was built from; the v5 ordering of the same five checkpoints is identical (see
[Fidelity](#fidelity)).

| build | download | resident | **v5 mean KLD** (10,480,640 pos) | corrected v3 mean KLD | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | **0.007172** | ~180k | fidelity first, smallest download |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.57 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k | you want the attention width knob at launch |
| [-context](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 20.70 GB | **18.41 GiB** | 0.003509 | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window; hard-limit RTX 5090 check pending |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB | 17.89 GiB | 0.010604 | 0.029679 | 262,144 | smallest footprint, native context without any overlay |

Official `Qwen/Qwen3.8-27B-FP8` is 28.51 GiB resident at **0.005294** on the v5 suite
(0.012798 on the v3 subset) and runs on stock vLLM, which none of these do. The two KLD
columns are not comparable to each other — absolute divergence is suite-specific — but they
rank the family identically.

## Recipe

| role | representation |
|---|---|
| MLP `gate_proj`, `up_proj` (64 layers) | EXL3 **K5**, `mcg` |
| MLP `down_proj` (64 layers) | EXL3 **K6**, `mcg` |
| attention: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48, `self_attn.{q,k,v,o}_proj` ×16 | **EXL3 K6 on disk**, `mcg`, calibrated (208 modules) |
| `lm_head` | EXL3 **K6**, `mcg` |
| MTP draft head | quantized (`fc` + attention K4, MLP K5/K6) |
| `embed_tokens`, vision tower (27 blocks), norms | BF16 |
| GatedDeltaNet `in_proj_a` / `in_proj_b` (96) | FP16 passthrough |

Composition, from the emitted manifest: `full_attention` 1.260 GB (64 EXL3 K6 + 32 BF16),
`linear_attention` 4.207 GB (144 EXL3 K6 + 96 FP16 + 192 BF16), `mlp_gate_proj` 3.568 GB,
`mlp_up_proj` 3.568 GB, `mlp_down_proj` 4.281 GB, `lm_head` 0.954 GB, `embed_tokens`
2.543 GB, `vision_tower` 0.921 GB, `mtp_draft` 0.283 GB, norms 0.001 GB — 21.61 GB over
three shards.

`quantization_manifest.json` and `build-receipt.json` are authoritative for composition;
`SHA256SUMS` covers the immutable payload. `config.json → quantization_config` keeps one
`bits`/`codebook` pair for loader compatibility and **cannot** describe this mixed
checkpoint.

Built from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
@ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with exllamav3
@ `5f3c537ca9d89893d771256f5c43c93656553fbb`, plus two allocator hooks upstream does not
provide: `EXL3_BITS_FIXED` (pins modules *before* budget allocation — the only way to hold
attention at K6 while the MLP keeps its own split) and `EXL3_BITS_OVERRIDE` (pins after
allocation). Both are in the
[companion repository](https://github.com/malaiwah/qwen38-27b-exl3).

**Verified after the build:** reconstructing every EXL3 module yields exactly upstream's
**1,199 logical tensor names with matching shapes**. The finalizer fails closed on any
missing, extra or mis-shaped tensor, so this is a check rather than a claim.

## Fidelity

Headline evidence is the **v5 held-out suite: 5,120 contexts x 2,047 positions =
10,480,640 full-vocabulary scored positions**, about 38x the position count of the
136-context development suite that used to carry this section (kept below as a prior
receipt). `KL(BF16 reference ‖ candidate)`, two passes, no top-k, float64 accumulation,
**both operands replayed through one shared BF16 LM head** — body-only, so no candidate's own
head quantization is counted — and a source-cluster bootstrap over 842 clusters.

### Suite identity

| property | value |
|---|---|
| manifest | [`receipts/kld5-suite-manifest.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-suite-manifest.json), schema `qwen38-distribution-fidelity/6` |
| suite token sha256 | `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88` |
| size | 5,120 contexts x 2,047 scored positions = **10,480,640** scored positions |
| source clusters | **842** — the bootstrap resampling unit |
| corpus | 941 documents / 70,348,971 bytes of held-out public text, fetched by `tools/fetch_corpus_v5.py` (log [`receipts/kld5-corpus-fetch-log.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-corpus-fetch-log.json), 0 failures) |
| exclusion policy | every document is scanned at **every position** for exact normalized 12-token overlap with exllamav3 calibration data, and any document with even one hit is dropped **before** context selection: **44 of 941 documents excluded (43 code, 1 encyclopedic), 897 eligible** |
| contamination result | **0 hits by construction**: `contexts_with_any_hit` 0, `total_hits` 0, 0 decoded candidates rejected against 859,426 calibration shingles |
| disjointness | token-disjoint from the v4 suite: all 160 prior context token hashes seeded, **0 reachable**, and 0 contexts emitted from a matching prior source document |
| windowing | exact-advance, adjacent and **non-overlapping**: 5,120/5,120 candidate windows tokenized with unique token hashes, 0 duplicates, 0 overlapping windows |

### Cumulative means at 10,480,640 positions

| candidate | resident | mean KLD | source-cluster bootstrap 95 % CI | top-1 | exact max single position |
|---|---:|---:|---|---:|---:|
| **this build** ([hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)) | 20.31 GiB | **0.002760** | [0.002540, 0.003020] | **97.70 %** | 8.258 |
| [online K5/K6 sibling](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 20.32 GiB | 0.003210 | [0.002982, 0.003480] | 97.52 % | 22.241 |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 18.41 GiB | 0.003509 | [0.003220, 0.003852] | 97.44 % | 5.557 |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.005294 | [0.004927, 0.005728] | 96.79 % | 10.714 |
| [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 17.89 GiB | 0.010604 | [0.009640, 0.011746] | 95.76 % | 14.283 |

This build is **48 % below official FP8** at 71 % of its resident weight, and it is the best of
the five on mean KLD, on both interval bounds and on top-1 agreement; the one column where
another checkpoint wins is the exact maximum, where the context edition's 5.557 beats this
build's 8.258.

Per-candidate receipts:
[`receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts),
schema `qwen38-kld-ladder-cumulative/2`, welded by `tools/kld_aggregate.py` from ten verified
per-shard reports produced by `tools/kld_ladder.sh` (per shard: capture six models over 512
contexts, replay five candidates, verify, delete 64 GB of hidden states, next shard). This
build's row is [`receipts/kld5-10M-hyd.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-hyd.json).

### Paired per-context differences

Same contexts, same reference, same shared head; source-cluster bootstrap with 10,000
resamples, seed 1, over 842 clusters. Receipt
[`receipts/kld5-10M-paired.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-paired.json):

| comparison | paired mean difference | 95 % CI | contexts won |
|---|---:|---|---:|
| **this build − official FP8** | **−0.002534** | [−0.002708, −0.002383] | **5,118 / 5,120** |
| online K5/K6 − FP8 | −0.002084 | [−0.002249, −0.001942] | 5,105 / 5,120 |
| context edition − FP8 | −0.001785 | [−0.001884, −0.001697] | 5,109 / 5,120 |
| K4 − FP8 | **+0.005310** | [+0.004710, +0.006019] | 7 / 5,120 (K4 is worse) |
| **this build − online K5/K6 sibling** | **−0.000450** | [−0.000469, −0.000433] | **4,922 / 5,120** |

**The offline-versus-online question is settled in direction.** The 136-context development
run could only show **124/136** contexts favouring calibrated offline encoding, with a 95 % CI
of [−0.000977, −0.000572]; at 5,120 contexts it is **4,922/5,120** with [−0.000469,
−0.000433] — an interval roughly 11x narrower (3.6e-05 wide against 4.05e-04) around a
smaller point estimate. Calibrating attention offline is consistently closer to BF16 than the
runtime's calibration-free online encoding, and that no longer rests on 136 contexts.

**What is still bounded, honestly:** −0.000450 is *below* this harness's 6.54e-04
live-versus-replay qualification floor, so the v5 run resolves the sign and the consistency
of the offline gain, not what it is worth in a live server relative to that floor. The FP8 gap
is 3.9x the floor and not in question.

### Ladder stability

`tools/kld_aggregate.py` welds the shards at 1M / 2M / 5M / 10M scored positions, and every
one of those checkpoints is recomputable from the per-context rows of
[`receipts/kld5-10M-hyd.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-hyd.json)
(each shard contributes 512 contexts / 1,048,064 positions). This build's mean stops moving
after the first million positions, which is why the run was stopped at ten shards:

| cumulative checkpoint | 1M (1,048,064) | 2M (2,096,128) | 5M (5,240,320) | 10M (10,480,640) |
|---|---:|---:|---:|---:|
| hydrated mean KLD | 0.002700 | 0.002759 | 0.002699 | **0.002760** |

The spread across a tenfold increase in positions is 6.1e-05 — more than an order of magnitude
below the 0.002534 gap to FP8.

### Distribution tail

A mean and a top-1 rate say nothing about the worst positions, so here is the whole right
tail. It is measured on **shard 0 of the same suite — 512 contexts, 1,048,064 scored
positions** — the identical contexts for all five candidates. Receipts
[`receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts),
schema `qwen38-kld-ladder-cumulative/2`, built by `tools/kld_aggregate.py`; this build's row is
[`receipts/kld5-1M-tail-hyd.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-tail-hyd.json).

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | share of positions above 0.1 | above 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **this build** (hydrated) | **0.002700** | **0.00109** | **0.0082** | **0.0276** | **0.1319** | **0.463** | **3.735** | **0.1534 %** | 0.00219 % |
| online K5/K6 | 0.003141 | 0.00128 | 0.0099 | 0.0321 | 0.1446 | 0.498 | 5.507 | 0.1820 % | **0.00200 %** |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | 0.1642 | 0.587 | 3.749 | 0.2287 % | 0.00305 % |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | 0.2438 | 0.812 | 5.296 | 0.3912 % | 0.00592 % |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | 0.5555 | 1.870 | 7.565 | 1.2604 % | 0.03807 % |

**Method, in one sentence:** every `qwen38-fidelity-report/2` replay accumulates a **560-bin
log-spaced histogram of per-position KLD** (`KLD_HIST_LOG10_LOW=-12.0`,
`KLD_HIST_LOG10_HIGH=2.0`, `KLD_HIST_BINS_PER_DECADE=40` in `tools/fidelity.py`) whose bin
counts add across shards, which is what makes cumulative quantiles possible at all.

**What it says for this build.** The ordering at p50, p95, p99, p99.9 and p99.99 is the same
as the ordering of the means, so the mean is not hiding a worse tail: this build is the lowest
of the five at **every** measured quantile, and on this shard it also has the smallest exact
maximum, 3.735. The one column where it is not first is the share of positions above 1.0,
where the online sibling's 21 positions edge this build's 23 — 0.00200 % against 0.00219 %,
both about a third of FP8's 0.00592 %. Over the full ten-shard run the exact-maximum column
goes the other way (8.258 here against the context edition's 5.557); a single worst position
is not a tail, which is the point of the table.

**Scope, stated exactly:**

- This is one 1,048,064-position shard, not the full 10,480,640-position run. The ten-shard
  run predates the histogram, so it could not be recomputed without re-running it.
- The quantiles are **bin-bounded, not exact**: each receipt carries `lower` / `upper` /
  `estimate` per quantile, with a relative bin width of about 5.6 %. The **maxima and the
  exceedance counts are exact**.
- The 10M receipts above remain the source for the full-run means, intervals and paired
  results; nothing in this subsection replaces them.

### What the v5 numbers do not say

- **Not comparable across suites.** Absolute KLD is suite-specific, so these are **not**
  comparable to the v3 numbers below: the corpus mix differs, and K4 reads 0.029679 there
  against 0.010604 here. Only within-suite ordering and paired differences transfer.
- **Cumulative percentiles come from one shard, not from all ten.** The ten per-shard reports
  of this run are `qwen38-fidelity-report/1`, which carries no token-level KLD histogram, so
  median/p95/p99/p999 could not be recombined across them. The
  [tail table above](#distribution-tail) closes that gap on **shard 0**
  (`receipts/kld5-1M-tail-*.json`); across all 10,480,640 positions only the means, the
  intervals, the paired results and the exact global maximum (8.258 for this build) exist.
- **Reproducible from the corpus, not from captures.** The hidden-state captures were deleted
  shard by shard to fit 135 GB of scratch, so unlike the v3 dataset this run is reproducible
  from the pinned corpus fetch log and suite manifest rather than from published captures.
- **Body-only.** Every v5 row replays both operands through one shared BF16 head. The
  as-served head increment for this build was measured on the v3 suite (+0.000125) and has not
  been re-measured at v5 scale.

## Prior receipt: v3 development suite (136 contexts, 278,392 positions)

This is the suite that guided recipe selection, and it was this card's headline until the v5
run above superseded it as the strongest evidence. Numbers unchanged. Held-out corpus, 136
analysis contexts, 278,392 full-vocabulary scored positions, `KL(BF16 reference ‖ candidate)`,
two passes, no top-k, one shared BF16 LM head for both operands, source-cluster bootstrap.
Same suite, reference and head as every comparator in the table:

| candidate | resident | mean KLD | bootstrap 95 % CI | median | top-1 |
|---|---:|---:|---|---:|---:|
| **this build** | 20.31 GiB | **0.007406** | [0.00543, 0.00978] | 0.001335 | **97.19 %** |
| BF16-attention sibling | 20.32 GiB | 0.008157 | [0.00607, 0.01067] | 0.001529 | 96.97 % |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 96.22 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 21.34 GiB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 90.53 % |

NVFP4 has no v5 row: the v5 ladder replayed five candidates and NVFP4 was not one of them, so
this table remains its only measurement here.

**Overlap-corrected subset:** a later all-position 12-token scan found exact calibration
overlap in 2/41 source documents that the original fixed-stride scan missed. Conservatively
removing their nine contexts gives this build **0.007172**, the online-K6 sibling **0.007945**,
official FP8 **0.012798**, and NVFP4 **0.092727** over 127 contexts. No ordering changes. The
v5 suite applies that same all-position rule *before* selection, which is why its contamination
count is zero by construction rather than by correction.

Paired on identical contexts:

- versus the BF16-attention sibling: **−0.000751**, 95 % CI [−0.000977, −0.000572],
  **124/136 contexts**. Calibrated offline encoding is consistently closer to BF16 than the
  runtime's calibration-free online encoding. While this was the headline it had to be read as
  encouraging rather than settled — the magnitude is only slightly above this harness's
  6.54e-04 live-versus-replay floor — and the direction is what the 4,922/5,120 v5 result
  above now confirms at roughly 38x the positions.
- versus official FP8: **−0.005719**, 95 % CI [−0.007323, −0.004353], **136/136 contexts** —
  44 % lower mean divergence at 71 % of its resident weight.

**Body-only versus as-served.** Every row above replays both operands through one shared BF16
head, so no candidate's own head quantization is counted — that is what makes the ranking fair,
since official FP8 serves a BF16 head. Measured directly on **this** checkpoint with asymmetric
heads (reference through the true BF16 head, candidate through this build's dequantized K6
head): the head costs **+0.000125** (95 % CI [+0.000107, +0.000144], 9/136 contexts favour it),
so **as served this build is 0.007532** with 97.08 % top-1. On the overlap-corrected
127-context subset, the measured result is **0.007300** with 97.05 % top-1; the head increment
is +0.000128 (95 % CI [+0.000110, +0.000146], 8/127 contexts favour it). The original
as-served result is still 1.74x better than FP8's
body-only 0.013126.

**Weakest control:** live-versus-replayed logit qualification is 6.54e-04 on this harness,
so differences below ~1e-3 are not resolvable. The −0.000751 offline-versus-online gap sits
**at that floor**: the bootstrap interval excludes zero and 124/136 contexts agree on
direction, but treat the magnitude as a point estimate. The FP8 gap, at 7.6x the floor, is
not in question.

**Development-set caveat:** the recipe was chosen with this 136-context suite visible. The
source-disjoint qualification below is the post-selection test, and the v5 suite above is a
second, much larger held-out run built after every recipe decision was frozen.

## Prior receipt: v4 post-selection qualification (36 contexts)

The v3 numbers above come from the suite that guided recipe selection. This is the test that
did not: **160 new contexts from 100 documents with zero intersection with the development
suite** (context token hashes 0/160, document names 0/100, content hashes 0/100), partitioned
by whole source cluster, run **once**, with no recipe changed afterwards. The v5 suite is
token-disjoint from this one as well, and supersedes it on size.

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

## Downstream task retention

On 40 deterministic generated tasks (10 each arithmetic, executable builtins-only code,
exact-list instruction following and tool-call schema), BF16 and every comparator scored
40/40. This build had **zero regressions** and matched BF16's exact final-answer text on
**35/40**; all five differing answers still passed their contracts. Wilson 95 % lower bound
is 91.2 %. This is a transparent smoke suite, not a public leaderboard; full responses are in
the [run receipt](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/tasks-v2-hydrated.json);
its extracted-value agreement field is superseded by the
[strict rescore](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/task-retention-v2-strict-rescore.json).

## Context capacity

This build carries the same resident weights as the sibling at attention K6 (20.31 vs
20.32 GiB), so its context behaviour is the sibling's K6 arm, which an independent tester
measured on a real **RTX 5090** (31.39 GiB usable, TP1, FP8 E4M3 KV, MTP-3, decode-only CUDA
graphs, vision enabled):

| configuration | KV attained | context | outcome |
|---|---:|---:|---|
| K6 attention, seqs 8, util 0.95 | 187,050 tok / 6.71 GiB | **185,600 run** | stable, multimodal-safe |
| K6 attention, seqs 8, util 0.98 | 202,185 tok / 7.55 GiB | — | text fine, **a 3,264-token image OOMed** with 33 MiB free |

**Native 262,144 does not fit this checkpoint with MTP-3 on a 32 GB card.** The engine needs
**9.13 GiB** of KV (37.4 KB/token: 16 full-attention layers, 4 KV heads, head_dim 256; the
other 48 layers are Gated DeltaNet and hold per-sequence state). Without MTP that falls to
33.5 KB/token and buys ~11 % more length. The smaller
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) is the only family
member hardware-qualified at native length on a real 32 GB card (289,577-token capacity).
The context edition starts natively with MTP-3 and an 8.4 MP image ceiling under a capped
32 GB-equivalent engine budget, but its physical RTX 5090 rerun remains pending. On 48 GB and
larger, native context fits here at the best fidelity.

Keep utilisation at **0.95** if you serve images: 0.98 leaves no vision headroom.

Numbers measured locally at a simulated 31.2 GiB budget (196,608 starts with 246,903 KV
tokens; 262,144 refused) agree with the hardware within a few percent, but the hardware
numbers above are the ones to trust — the simulation initially omitted MTP's KV and
overstated usable VRAM, which produced an overclaim that the tester correctly rejected.

## Serving

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6-hydrated \
    --served-model-name qwen38 --quantization exl3 --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 8 \
    --host 0.0.0.0 --port 8000
```
The container listens on all interfaces internally, but Docker publishes the port to host
loopback only. For remote clients, keep that binding and put an authenticated TLS proxy in
front; do not expose this unauthenticated generation endpoint directly.


No `VLLM_EXL3_ONLINE_TRELLIS_BITS`, no `VLLM_EXL3_ONLINE_CACHE_DIR`: nothing is encoded at
load, which is the point of this build. Three notes carry over from the sibling:

1. **`--quantization exl3` is mandatory** — auto-detection only fires for GLM-5.2 metadata.
2. **The `ignore` list is mandatory and its anchoring is subtle.** Prefixes carry no leading
   `model.`, so `re:.*visual\..*` matches while `re:.*\.visual\..*` silently does not, and the
   wrong pattern **crashes** startup
   ([#311](https://github.com/local-inference-lab/vllm/issues/311), fixed by
   [PR #312](https://github.com/local-inference-lab/vllm/pull/312)). The tested config ignores
   `mtp.*` from the generic BF16 online overlay; EXL3-owned draft projections are selected
   before that ignore check and still load from their serialized quantized tensors.
3. **`--mm-processor-kwargs '{"truncation":false}'` is required for images** whose expanded
   token sequence exceeds 2,048, or requests fail with HTTP 400
   ([#313](https://github.com/local-inference-lab/vllm/issues/313)).

The command above is what the **pinned image runs unmodified**, so it is eager-only. CUDA-graph
decode (+46-50 %) and the measured reconstructed-prefill path (+113 %) need patches still
open upstream ([#314](https://github.com/local-inference-lab/vllm/pull/314),
[#316](https://github.com/local-inference-lab/vllm/pull/316), and
[#318](https://github.com/local-inference-lab/vllm/pull/318)); the sibling card carries the
exact one-module patch recipe and current sha256. No published image digest contains it.

## What is not verified

The deterministic downstream smoke passes 40/40 with zero BF16 regressions, but **no public
task benchmark has run for this build**. A paired MMLU-Pro run (70 questions, 14 official
categories, official five-shot prefixes, greedy, pinned
`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`) exists so far only for BF16
(**57/70**, Wilson [70.8 %, 88.8 %]) and K4 (**57/70**, 55/57 BF16-pass retention, Wilson
lower bound 88.1 %), receipts
[`public-capability-{plan,suite-mmlupro-70,bf16,k4}.json`](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts)
with harness `tools/public_capability.py`; this checkpoint is one of the four candidates not
yet run. Also unverified: real OCR/chart/document/video quality, long-context
retrieval or perplexity **for this build**, native-262K or YaRN-1M generation, multi-GPU or
TP>1, non-SM120 hardware, and quant-specific safety regression testing.
Throughput was not re-measured here: this build shares the sibling's kernels and resident
footprint, so its decode and prefill figures should carry over — that is an inference, not a
measurement.

Evaluation captures and reports for the **v3** rows are published in the
[fidelity dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3),
so that row is independently recomputable without this checkpoint or a GPU. The v5 headline
run has no published captures — they were deleted shard by shard to fit 135 GB of scratch —
so it is instead reproducible from the pinned corpus fetch log and suite manifest listed under
[Suite identity](#suite-identity).

## Machine-readable evidence

[`release-evidence-hydrated.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/release-evidence-hydrated.json)
carries the whole chain in one file: shard and index SHA-256, upstream revision and the
verified 1,199-tensor topology, research and exllamav3 commits with tree-clean state, the
container digest and the patched module's hash, hardware and driver, suite token hash and
partition, every fidelity number with its interval, the controls including the replay floor,
and an explicit `not_verified` list. `SHA256SUMS` covers the immutable payload (16 files);
`DOCS-SHA256SUMS` covers card files, so a card edit can no longer invalidate the build hashes.

That file's fidelity chain is the v3/v4 evidence. The v5 headline numbers live in their own
receipts: `receipts/kld5-suite-manifest.json`, `receipts/kld5-corpus-fetch-log.json`,
`receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json` and `receipts/kld5-10M-paired.json`, each
carrying the suite token hash, the cluster partition and its own content digest.

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — EXL3 Trellis format,
  LDLQ calibration, MCG codebook, the conversion pipeline.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path this checkpoint
  needs, and the online-encoding overlay whose output this build replaces with a calibrated
  one.
- [Qwen](https://huggingface.co/Qwen) — the base model and its official FP8 derivative.
- An independent tester on an RTX 5090 — the context-capacity table, and catching a 262K
  overclaim that came from a memory simulation.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
