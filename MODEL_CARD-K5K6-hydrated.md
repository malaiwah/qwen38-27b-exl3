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
| download | 30.60 GB | **21.61 GB** (−29 %) |
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
online and K4 are real RTX 5090 MTP-3 tests; context is MTP-3 with an 8.4 MP cap, qualified
on a physical RTX 5090 at utilisation 0.955.
These profiles are not interchangeable
([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; that engine-budget star has since been superseded by a physical RTX 5090 qualification of the context edition at 265,122 KV tokens and utilisation 0.955." src="assets/context-frontier-light.svg">
</picture>

The figure's axis is still the overlap-corrected v3 receipt, because that is what the plotted
asset was built from; the v5 ordering of the same five checkpoints is identical (see
[Fidelity](#fidelity)).

| build | download | resident | **v5 mean KLD** (10,480,640 pos) | corrected v3 mean KLD | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | **0.007172** | ~180k | fidelity first, smallest download |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.60 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k | you want the attention width knob at launch |
| [-context](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 20.70 GB | **18.41 GiB** | 0.003509 | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window, hardware-qualified on a physical RTX 5090 |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB\* | 17.89 GiB | 0.010604 | 0.029679 | 262,144 | smallest footprint, native context without any overlay |

**Byte and memory conventions for this table.** The download column is whole-tree bytes —
every published file of the artifact as its release evidence counted it
([`receipts/collection-index.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/collection-index.json),
`serialized_bytes.whole_tree_bytes`: this build 21,610,933,884 B, K5/K6 30,597,231,933 B,
context 20,696,053,306 B) — and they are serialized bytes on disk, never resident memory.
\*The K4 release evidence records no tree count, so that one row is the sum of its safetensors
shards, 28,313,841,196 B, read from the published repository. The context edition's resident
weight is measured twice: **18.41 GiB** as run on the rental RTX PRO 6000 engine-budget proof
and **18.19 GiB** on the physical RTX 5090 at the qualified `0.955` profile. This table prints
the larger figure deliberately, because
[`receipts/vram-class-verdict.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/vram-class-verdict.json)
elects 18.41 GiB for every class prediction; the 0.22 GiB gap is the rental-versus-5090 delta,
not a change in the checkpoint.

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
2.543 GB, `vision_tower` 0.921 GB, `mtp_draft` 0.283 GB, norms 0.001 GB. Those roles sum to
21,586,964,548 B = **21.587 GB** of tensor payload over three shards; the 21.61 GB download
above is the whole published tree, 21,610,933,884 B, which is 24.0 MB larger because it also
carries the tokenizer, the configs and the chat template.

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

## Against GGUF, measured on our suite

The standing objection to this family's headline is that official FP8 is a throughput format
whose quality is Q4-to-Q5 class, so beating it is a weak claim, and that llama.cpp's `Q8_0` and
`Q6_K` are the honest bar. That is now measured rather than argued.

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
| **this build** (hydrated) | vLLM | **0.002700** | n/a, same engine as the reference | **97.80 %** | **0.1313** | **20.12 GiB payload** |
| online K5/K6 | vLLM | 0.003141 | n/a, same engine | 97.61 % | 0.1447 | — |
| context edition | vLLM | 0.003409 | n/a, same engine | 97.55 % | 0.1632 | 19.27 GiB payload |
| GGUF `UD-Q5_K_XL` | llama.cpp | 0.004444 | ~0.003936 | 97.20 % | 0.2144 | 18.83 GiB |
| official FP8 | vLLM | 0.005197 | n/a, same engine | 96.92 % | 0.2440 | 28.51 GiB resident |
| K4 | vLLM | 0.010345 | n/a, same engine | 95.91 % | 0.5576 | — |

**The engine floor, measured and not assumed.** A GGUF row carries llama.cpp-versus-vLLM numerics
on top of quantization error, so that term was measured the same way: the unquantized **BF16
GGUF** against the vLLM BF16 reference, identical token ids, the same shared head, the same 512
contexts — **0.000507** mean, 99.07 % top-1, p99.9 0.0113
([`receipts/gguf-report-engine-floor.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gguf-report-engine-floor.json)).
Every GGUF row above contains that term; no vLLM row — ours or FP8's — does. **KL is not additive,
so the net column is an estimate, not an identity**: the measured GGUF value is an upper bound and
the net figure is the naive lower one. The two `—` cells are the builds that ship BF16 attention
for the runtime to encode at load, so their disk bytes are not a like-for-like payload; the
payload figures are `immutable_payload_bytes` from
[`receipts/collection-index.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/collection-index.json)
(this build 21,610,916,123 B = 20.127 GiB, context edition 20,696,033,532 B = 19.275 GiB; the table
truncates both to two decimals) and are serialized bytes, never
VRAM; the FP8 figure is resident weights and is labelled as such.

**The p99.9 column, and why it differs from the tail table above.** These p99.9 values are each
report's **exact** shard-0 p99.9 as the comparator receipt read them; the
[tail table above](#distribution-tail) quotes the **bin-bounded cumulative estimate** from the
560-bin histogram, whose bins are about 5.6 % wide — this build reads 0.1319 there and 0.1313 here,
and the exact value lies inside the bin the estimate names. The two differ by construction, not by
measurement.

**Where this build sits, without spin.** This is the best-measuring build in the family, and it
**loses the 6-bit comparison**. `Q6_K` reads 0.002035 measured and ~0.001528 net of the floor, at
21.31 GiB, against this build's 0.002700 at 20.12 GiB of payload — about **43 % lower divergence
for 1.186 GiB more file** — and it is still **25 %** lower on the measured value that includes
the cross-engine term, so no treatment of that term rescues this build there. What it does win on
this shard: `UD-Q5_K_XL`, whose net 0.003936 at 18.83 GiB is **46 % higher divergence** than this
build's, and official FP8, which this build sits **48 % below** at 28.51 GiB resident. Its tail
agrees with its mean: p99.9 **0.1313**, lighter than `UD-Q5_K_XL`'s 0.2144 and FP8's 0.2440,
heavier than `Q6_K`'s 0.0794. `Q8_0` leads everything at 27.05 GiB.

**The two conclusions worth stating plainly:**

1. **At the 6-bit operating point GGUF `Q6_K` is genuinely better than this build** — 0.001528 net
   at 21.31 GiB versus 0.002700 at 20.12 GiB. This is the first measurement in this project where
   an off-the-shelf artifact beats the recipe, and it is published as such.
2. **At the 5-bit operating point our context edition wins** — 0.003409 at 19.27 GiB against
   `UD-Q5_K_XL`'s 0.003936 net at 18.83 GiB, about 13 % better fidelity for **0.445 GiB** more
   payload.

So the format advantage at this bitrate is real at 5 bits, negative at 6 bits, and far short of a
full bit. Two further readings that are not flattering: **`Q8_0` is the fidelity leader** at
0.001087 for 27.05 GiB, and its measured value is only about twice the engine floor, so its own
number sits near the resolution limit of any cross-engine comparison — the net column is an
estimate, not an identity, so **no ordering closer than a factor of two should be pressed against
`Q8_0`**; and **every GGUF point at or
above 5 bits beats official FP8**, which makes our published "34-48 % below FP8" true and a weaker
achievement than it sounds. **K4 is the weakest point in the table.**

**What this comparison does not settle.** It is text-only teacher-forced fidelity on one shard of
ten. It says nothing about serving 262,144 tokens with vision and MTP on a 32 GB card, which is
where these artifacts actually differ, and llama.cpp KV-quant behaviour, prefill and decode speed
are separate axes that were not measured here. The GGUF rows are also a shard-0 ranking, not a
paired per-context bootstrap against the ten-shard rows in
[Fidelity](#fidelity), because those were welded from a different position count. Shard 0 is one tenth of the suite, and it is close to it: over all 10,480,640 positions the five vLLM
means read 0.002760 / 0.003210 / 0.003509 / 0.005294 / 0.010604 — **1.9-2.9 % above** these shard-0
values, ordering unchanged (`receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`). The GGUFs have no
ten-shard equivalent; extending them is unrun.

**One protocol objection, bounded rather than argued.** `llama-perplexity` scores only the second
half of each window, so every position it scores has at least 256 tokens of left context, while our
suite scores from position 0. Re-scoring our own captures under that restriction lowers every
candidate's mean by **1.3-2.1 %** at a 256-token floor and **3.9-4.9 %** second-half-only,
uniformly enough to change no ordering — this build reads 0.002660 and 0.002580 respectively
([`receipts/scored-window-offset.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/scored-window-offset.json)).
The external protocol's scoring floor therefore explains at most about 5 % of any cross-protocol
gap, and nothing in the ordering above.

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

## Public capability — MMLU-Pro, item-paired against BF16

70 MMLU-Pro questions, 14 official categories, 5 per category, pinned
`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, official five-shot category
prefixes, greedy, thinking at low reasoning effort, 5,120-token completion cap. The BF16
control ran first and the acceptance rule was frozen in
[`receipts/public-capability-plan.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-plan.json)
before any candidate result was seen. Every candidate answered the same 70 items in the same
order through the same extractor, so each row below is paired item-by-item against that
control.

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.8-27B` BF16 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | [bf16](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16.json) |
| [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 58/70 (82.9 %) | [72.4 %, 89.9 %] | 56/57 | **90.7 %** | 1 | 2 | 3 | [ctx](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-ctx.json) |
| [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 57/70 (81.4 %) | [70.8 %, 88.8 %] | 55/57 | 88.1 % | 2 | 2 | 4 | [k4](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k4.json) |
| **this build (hydrated)** | **56/70 (80.0 %)** | **[69.2 %, 87.7 %]** | **54/57** | **85.6 %** | **3** | **2** | **4** | [hyd](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-hyd.json) |
| `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | [fp8](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-fp8.json) |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 55/70 (78.6 %) | [67.6 %, 86.6 %] | 54/57 | 85.6 % | 3 | 1 | 4 | [k5k6](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k5k6.json) |

### The pre-registered bar, and this build's verdict

The frozen plan accepts a candidate when BF16-pass retention has a **Wilson 95 % lower bound at
or above 0.90** and **no category loses more than two BF16 passes**. The category clause is met
by all five candidates — the worst case is two passes in philosophy, for this build and for
online K5/K6 — so the retention lower bound is the only clause that ever fails.

**Only the context edition clears the bar, at 90.7 %.** K4 and official
`Qwen/Qwen3.8-27B-FP8` read 88.1 %. **This build reads 85.6 % (54/57) and does not clear it**,
as does online K5/K6. That is a measured shortfall, published exactly as measured, with nothing
retuned afterwards: three of BF16's 57 passes flipped to failures here and two BF16 failures
flipped to passes.

**What the shortfall is not.** Every interval in the table overlaps every other interval,
including the BF16 control's and official FP8's, so the matrix does not rank these builds and
this card does not claim it does. This build is not shown to be worse than official FP8, K4 or
the context edition on knowledge-and-reasoning tasks, and it is not shown to be better than
any of them either.

### Why a 70-item suite cannot certify this bar

With 57 BF16 passes as the paired denominator, **56/57 is the smallest count whose Wilson 95 %
lower bound clears 0.90** (56/57 → 90.7 %; 55/57 → 88.1 %; 54/57 → 85.6 %). A single paired
regression is therefore the entire budget, and no result that gives up two can pass, however
sound the build. The suite simply has too few items to certify the bar it pre-registered, and
at this size it separates nothing: the point applies to official FP8 exactly as it applies to
the EXL3 builds. Read it as a **power limitation of a 70-item draw, not as evidence that any
of these checkpoints is broken**.

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
33.5 KB/token and buys ~11 % more length. Both KB/token figures are ratios measured at that
one window, not coefficients to extrapolate other windows with: the KV pool one request needs
is affine in the window — a per-token term plus a fixed per-request term — so dividing a KV
budget by a KB/token figure does not predict a context length. The smaller
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) reaches native
length on a real 32 GB card (289,577-token capacity) with no overlay at all.
The context edition is **hardware-qualified at native length on a physical RTX 5090**: 262,144
with MTP-3 and the full 8.4 MP image ceiling, **265,122 KV tokens** of capacity and 1.01x
concurrency at native length, measured at `--gpu-memory-utilization 0.955` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, all seven gates passing
([`receipts/qualification-5090-context.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-context.json)).
On 48 GB and larger, native context fits here at the best fidelity.

Keep utilisation at **0.95** if you serve images: 0.98 leaves no vision headroom. That 5090
qualification measured the same trap from the other side — at utilisation 0.97 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`bounded_negative_results` arm B2) the
context edition's vision tower died wanting 62.00 MiB with 26.50 MiB free, and at the same
0.97 with no allocator configuration at all (arm B) it died wanting the same 62.00 MiB with
34.56 MiB free; lowering `max_pixels`
instead of utilisation made it strictly worse, because the engine spends every freed byte on
KV. Utilisation is the knob, on that profile and on this one.

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

The paired MMLU-Pro suite above has now run for this build and all five comparators, and this
build is a **measured shortfall** against the pre-registered bar: 54/57 BF16-pass retention,
Wilson 95 % lower bound **85.6 %** against a required 90 %. At 70 items the suite cannot
certify that bar for any result that gives up more than one paired pass, and its intervals
separate no two candidates, so public capability here is **measured but not established** —
one 70-item multiple-choice draw, with no executable-code, constraint-following, tool-schema or
larger-draw evidence yet (the plan's own P1). Also unverified: real OCR/chart/document/video quality, long-context
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
