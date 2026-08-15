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
  - long-context
  - gilded-gnosis
---

# Qwen3.8-27B EXL3 K5/K6 context edition — native 262,144 under a 32 GB engine budget, at 34 % lower divergence than official FP8

> **Requires a custom runtime.** Does **not** load in upstream vLLM, SGLang, TensorRT-LLM,
> llama.cpp, transformers, or stock exllamav3. It needs the Gilded Gnosis vLLM fork with
> explicit `--quantization exl3` and an exact `ignore` list. Treat it as an experimental,
> runtime-specific research artifact.

The long-context member of the family: K5 attention plus an opt-in int8 input embedding
overlay starts at native 262,144 with MTP-3, decode graphs and an 8,388,608-pixel image
ceiling under a conservative 30.24 GiB vLLM budget. It retrieved a code from 261,794 text
tokens and from a 236,824-token prompt containing a seven-megapixel image. The proof server
was a capped 96 GB SM120, not a physically constrained RTX 5090; hard-limit 5090 validation
is still pending. On the v5 held-out suite — 5,120 contexts, **10,480,640 scored positions** —
it measures **0.003509** mean `KL(BF16 ‖ candidate)` against official FP8's **0.005294**, a
paired **−0.001785** that wins **5,109/5,120** contexts. It was already below FP8 on the older
v3 development and v4 source-disjoint suites.

## Which of the four builds

Same architecture and tokenizer. The headline KLD column is the v5 held-out suite
(5,120 contexts, 10,480,640 scored positions, body-only through one shared BF16 head); the
overlap-corrected 127-context v3 subset is kept beside it because absolute KLD is
suite-specific and the two columns cannot be compared with each other. The frontier figure
below still plots the v3 axis. The context edition's native result is MTP-3 with an 8.4 MP
image cap on a budget-capped RTX PRO 6000; the other rows are real RTX 5090 MTP-3 tests.
These profiles are not interchangeable.
([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; the latter's hard-limit 5090 rerun is pending." src="assets/context-frontier-light.svg">
</picture>

| build | download | resident | v5 held-out mean KLD | corrected v3 mean KLD | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | 0.007172 | ~180k, 5090 MTP-3 | fidelity first |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.57 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k, 5090 MTP-3 | you want the width knob at launch |
| **this build** | 20.70 GB | **18.41 GiB** | **0.003509** | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window plus speculative decode; 5090 check pending |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB | 17.89 GiB | 0.010604 | 0.029679 | **262,144, 5090 MTP-3** | native context is non-negotiable |

Official `Qwen/Qwen3.8-27B-FP8` reads **0.005294** on the v5 suite and 0.012798 on the
corrected v3 subset: the three K5/K6 rows are below it on both suites, K4 on neither.

## Recipe

| role | representation |
|---|---|
| MLP `gate_proj`, `up_proj` (64 layers) | EXL3 **K5**, `mcg` |
| MLP `down_proj` (64 layers) | EXL3 **K6**, `mcg` |
| attention: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48, `self_attn.{q,k,v,o}_proj` ×16 | **EXL3 K5 on disk**, `mcg`, calibrated |
| `lm_head` | EXL3 **K6**, `mcg` |
| MTP draft head | quantized (`fc` + attention K4, MLP K5/K6) |
| `embed_tokens`; vision tower (27 blocks), norms | BF16 on disk; input table **int8 at load** with the overlay; vision/norms BF16 |
| GatedDeltaNet `in_proj_a` / `in_proj_b` (96) | FP16 passthrough |

Built from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
@ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Verified after the build: reconstructing every
EXL3 module yields exactly upstream's **1,199 logical tensor names with matching shapes**; the
finalizer fails closed otherwise.

**The vision tower stays BF16 for a measured reason.** Quantizing it (`-vb 6`) saves 0.58 GB
but the converter splits upstream's fused `visual.blocks.N.attn.qkv` into separate q/k/v
projections, so the checkpoint stops matching the architecture's weight names — and the EXL3
loader excludes vision by design anyway. The topology check caught it; the build was redone.

## Fidelity

### v5 held-out suite — 10,480,640 scored positions

This is the headline fidelity evidence, and it replaces the 136-context / 278,392-position v3
table below as the number to quote. The v3 and v4 results stay published because they are what
the recipe was selected and then frozen against. Metric is `KL(BF16 ‖ candidate)` over the full
vocabulary with both operands replayed through one shared BF16 LM head, body-only; this build
is scored as its serialized checkpoint, with no int8 input overlay and no candidate-owned head.
Intervals are a source-cluster bootstrap over the 842 clusters, 10,000 resamples, seed 1.

| candidate | v5 mean KLD | 95 % CI | top-1 | paired vs official FP8 | exact worst position |
|---|---:|---|---:|---|---:|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 0.002760 | [0.002540, 0.003020] | 97.70 % | −0.002534 [−0.002708, −0.002383], **5,118/5,120** | 8.258 |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 0.003210 | [0.002982, 0.003480] | 97.52 % | −0.002084 [−0.002249, −0.001942], **5,105/5,120** | 22.241 |
| **this build, serialized, MTP off** | **0.003509** | [0.003220, 0.003852] | **97.44 %** | **−0.001785** [−0.001884, −0.001697], **5,109/5,120** | **5.557** |
| `Qwen/Qwen3.8-27B-FP8` | 0.005294 | [0.004927, 0.005728] | 96.79 % | — | 10.714 |
| [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 0.010604 | [0.009640, 0.011746] | 95.76 % | +0.005310 [+0.004710, +0.006019], K4 wins **7/5,120** | 14.283 |

0.003509 against FP8's 0.005294 is **34 % lower divergence**. The paired result is what
carries it: this build is closer to BF16 than official FP8 on **5,109 of 5,120** contexts, and
the interval on that difference never touches zero. The build that produces it serves at
**18.41 GiB** resident in its int8-overlay MTP-3 profile, 19.31 GiB with the BF16 input table,
against FP8's 28.51 GiB. Its exact maximum single-position divergence over all 10,480,640
positions is **5.557**, the lowest of the five candidates — below FP8's 10.714 and far below
the online-encoded sibling's 22.241, which is the one axis on which the context edition leads
the whole family. Ordering is unchanged from v3, and from v4 for the four builds that suite
covered: hydrated, online K5/K6, this build, then FP8, then K4.

**Suite identity.** [`receipts/kld5-suite-manifest.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-suite-manifest.json),
schema `qwen38-distribution-fidelity/6`, suite token SHA-256
`510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88`: 5,120 contexts ×
2,047 scored positions = **10,480,640**, drawn from **842 source clusters** over a corpus of
941 documents / 70,348,971 bytes fetched by `tools/fetch_corpus_v5.py`
([fetch log](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-corpus-fetch-log.json)).
Windows are exact-advance and non-overlapping; an independent re-check of the emitted suite
found **5,120/5,120 unique context token hashes and 0 overlapping windows**. It is
token-disjoint from the v4 qualification — 0 of the 160 prior context hashes is reachable.

**Exclusion policy: contamination is 0 by construction, not by luck.** Before a single context
is selected, every source document is scanned at every position for exact normalized 12-token
overlap with the exllamav3 calibration data, and any document with even one hit is dropped
whole: **44 of 941 documents excluded** (43 code, 1 encyclopedic), 897 eligible. No emitted
context can then contain calibration text, which is why the scan over the finished suite
reports zero hits. This is the same conservative rule that had to be applied retroactively to
v3 and v4, applied here before selection instead of after.

**Ladder stability.** `tools/kld_ladder.sh` walks the suite in ten 512-context shards —
capture six models, replay the five candidates, verify, delete 64 GB of hidden states, next
shard — and `tools/kld_aggregate.py` welds the verified shard reports into cumulative receipts
at the 1M / 2M / 5M / 10M checkpoints. The cumulative mean is flat along that ladder: hydrated
reads **0.002700 / 0.002759 / 0.002699 / 0.002760**, so the 10M figure is not a late-shard
artifact. Per-candidate receipts are `receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`
(schema `qwen38-kld-ladder-cumulative/2`); the paired differences are in
[`receipts/kld5-10M-paired.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-paired.json).

**Distribution tail.** A mean, a top-1 rate and one worst position do not describe a tail, so
here is all of it, measured on **shard 0 of the same suite — 512 contexts, 1,048,064 scored
positions** — the identical contexts for all five candidates. Receipts
`receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json` (schema `qwen38-kld-ladder-cumulative/2`,
built by `tools/kld_aggregate.py`); this build's row is
[`receipts/kld5-1M-tail-ctx.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-tail-ctx.json).
Every `qwen38-fidelity-report/2` replay accumulates a **560-bin log-spaced histogram of
per-position KLD** (`KLD_HIST_LOG10_LOW=-12.0`, `KLD_HIST_LOG10_HIGH=2.0`,
`KLD_HIST_BINS_PER_DECADE=40` in `tools/fidelity.py`) whose bin counts add across shards,
which is what makes cumulative quantiles possible at all.

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | share of positions above 0.1 | above 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 0.002700 | 0.00109 | 0.0082 | 0.0276 | 0.1319 | 0.463 | 3.735 | 0.1534 % | 0.00219 % |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 0.003141 | 0.00128 | 0.0099 | 0.0321 | 0.1446 | 0.498 | 5.507 | 0.1820 % | 0.00200 % |
| **this build** (context) | **0.003409** | **0.00135** | **0.0107** | **0.0357** | **0.1642** | **0.587** | **3.749** | **0.2287 %** | **0.00305 %** |
| `Qwen/Qwen3.8-27B-FP8` | 0.005197 | 0.00202 | 0.0167 | 0.0531 | 0.2438 | 0.812 | 5.296 | 0.3912 % | 0.00592 % |
| [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 0.010345 | 0.00320 | 0.0332 | 0.1194 | 0.5555 | 1.870 | 7.565 | 1.2604 % | 0.03807 % |

For this build the tail behaves exactly like the mean: the ordering at p50, p95, p99, p99.9
and p99.99 is the same as the ordering of the means, so the mean is not hiding anything worse
at depth. Its entire tail sits **below official FP8's at every measured quantile** — 0.0357
against 0.0531 at p99, 0.1642 against 0.2438 at p99.9, 0.587 against 0.812 at p99.99 — and
**0.2287 %** of positions exceed 0.1 against FP8's 0.3912 %. It is the highest of the three
K5/K6 builds at every quantile, the same third place its mean reports, and the price of the
262,144-token window. Its exact maximum on this shard is 3.749, essentially level with
hydrated's 3.735, which is consistent with this build owning the lowest exact maximum
(5.557) over the full ten-shard run.

**What this run does not give you.**

- Absolute KLD is suite-specific. The v5 numbers are **not** comparable with the v3 numbers
  below: the corpus mix differs, and K4 reads 0.029679 there against 0.010604 here. Only
  within-suite ordering and paired differences transfer.
- Cumulative token percentiles come from **one shard**, not from all ten: the ten shard
  reports of the 10M run carry no token-level KLD histogram, so nothing could be recombined
  across them. The tail table above closes that gap on **shard 0**
  (`receipts/kld5-1M-tail-*.json`), with bin-bounded quantiles — each receipt carries
  `lower` / `upper` / `estimate`, relative bin width about 5.6 % — and **exact** maxima and
  exceedance counts. Across all 10,480,640 positions only the means, the intervals, the
  paired results and one exact global maximum per candidate exist.
- The hidden-state captures were deleted shard by shard to fit 135 GB of scratch. Unlike the v3
  dataset, the v5 run is reproducible from the pinned corpus fetch log and suite manifest
  rather than replayable from published captures.

### v3 development suite — 136 contexts, 278,392 positions (superseded as headline, kept)

These are the measurements the recipe was selected against, and they remain the source of the
int8-input-overlay and K6-head attributions below. Every number in this subsection, including
both overlay rows, is a v3 measurement; none of it is comparable to the v5 magnitudes above.

Held-out corpus, 136 analysis contexts, 278,392 full-vocabulary scored positions,
`KL(BF16 ‖ candidate)` with one shared BF16 head for both operands, source-cluster bootstrap.
The first row is the serialized checkpoint with its BF16 input table; the second changes only
that table at load.

| candidate | resident profile | mean KLD | 95 % CI | token median | top-1 |
|---|---:|---:|---|---:|---:|
| this build, BF16 input, MTP off | 19.31 GiB | **0.009673** | [0.00711, 0.01275] | 0.001672 | 96.81 % |
| **this build, int8 input overlay, MTP off** | **18.13 GiB** | **0.009738** | [0.00716, 0.01284] | 0.001687 | 96.80 % |
| hydrated (attention K6) | 20.31 GiB | 0.007406 | [0.00543, 0.00978] | 0.001335 | 97.19 % |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 96.22 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 21.34 GiB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 90.53 % |

**Overlap-corrected subset:** a later all-position 12-token scan found exact calibration
overlap in 2/41 source documents that the original fixed-stride scan missed. Conservatively
removing their nine contexts gives this build **0.009378**, hydrated **0.007172**, official FP8
**0.012798**, and NVFP4 **0.092727** over 127 contexts. The int8 input-overlay row is
**0.009459** on that same subset. No ordering changes.

Paired on identical v3 contexts:

- baseline versus official FP8: **−0.003453**, 95 % CI [−0.004383, −0.002666],
  **135/136 contexts** — 26 % lower divergence on this suite at 68 % of its resident weight;
  the int8 overlay is 64 % of FP8 resident weight for +0.000065 KLD. The same comparison on
  the v5 held-out suite is **−0.001785 over 5,109/5,120 contexts**, 34 % lower divergence; the
  two percentages are not two measurements of one quantity, they are two suites.
- baseline versus the hydrated build: **+0.002266** (1/136). That is what K5 attention costs,
  and it buys 0.85 GiB and ~16k tokens of context before the embedding overlay.
- **Calibration beats the runtime overlay again:** serialized K5 attention measures 0.009673
  where the same family encoded K5 *at load* measures 0.012135 on the original v3 receipt;
  overlap-corrected values are **0.009378 versus 0.011801**.
- With its own K6 head the BF16-input baseline is 0.009795 on the original v3 receipt and
  **0.009503** on the corrected subset. The head and input-overlay deltas were measured
  separately; their combination is not presented as a measured number.

## Post-selection qualification (v4, source-disjoint)

The v3 numbers above come from the suite that guided recipe selection. This is the test that
did not: **160 new contexts from 100 documents with zero intersection with the development
suite** (context token hashes 0/160, document names 0/100, content hashes 0/100), partitioned
by whole source cluster, run **once**, with no recipe changed afterwards. The v5 held-out
suite is a third, much larger post-selection run, and it is token-disjoint from this one.

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
before any candidate result was seen. All six models answered the same 70 items in the same
order through the same extractor, so every candidate row is paired item-by-item against that
control.

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.8-27B` BF16 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | [bf16](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16.json) |
| **this edition (context)** | **58/70 (82.9 %)** | **[72.4 %, 89.9 %]** | **56/57** | **90.7 %** | **1** | **2** | **3** | [ctx](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-ctx.json) |
| [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 57/70 (81.4 %) | [70.8 %, 88.8 %] | 55/57 | 88.1 % | 2 | 2 | 4 | [k4](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k4.json) |
| [hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 54/57 | 85.6 % | 3 | 2 | 4 | [hyd](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-hyd.json) |
| `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | [fp8](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-fp8.json) |
| [online K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 55/70 (78.6 %) | [67.6 %, 86.6 %] | 54/57 | 85.6 % | 3 | 1 | 4 | [k5k6](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-k5k6.json) |

### The pre-registered bar, and this edition's verdict

The frozen plan accepts a candidate when BF16-pass retention has a **Wilson 95 % lower bound at
or above 0.90** and **no category loses more than two BF16 passes**. The category clause is met
by all five candidates — the worst case is two passes in philosophy, for the hydrated build and
online K5/K6 — so the retention lower bound is the only clause that ever fails.

**This edition is the only candidate that clears the bar: 56/57 BF16-pass retention, Wilson 95 %
lower bound 90.7 %.** Exactly one of BF16's 57 passes flipped to a failure and two BF16 failures
flipped to passes, for 58/70 absolute. K4 and official `Qwen/Qwen3.8-27B-FP8` read 88.1 %; the
hydrated build and online K5/K6 read 85.6 %. None of the other four candidates clears the bar,
and all four shortfalls are published as measured.

**Clearing the bar is not a claim that this edition beats anything.** Every interval in the
table overlaps every other interval, including the BF16 control's and official FP8's, so the
matrix does not rank these models and this card does not claim it does. 58/70 against BF16's
57/70 is one item inside intervals more than 15 points wide; it is **not** evidence that this
edition is better than BF16, than official FP8, or than its siblings.

### Why almost nothing clears this bar at 70 items

With 57 BF16 passes as the paired denominator, **56/57 is the smallest count whose Wilson 95 %
lower bound clears 0.90** (56/57 → 90.7 %; 55/57 → 88.1 %; 54/57 → 85.6 %). A single paired
regression is the entire budget, so no result that gives up two can pass, however sound the
build — the suite has too few items to certify the bar it pre-registered. This edition clears
it by spending exactly one regression and no more, which is a narrow pass rather than a
demonstrated margin. The same limit applies to official FP8 exactly as it applies to the EXL3
builds: what the matrix shows is a **power limitation of a 70-item draw, not evidence that any
of these checkpoints is broken**.

### Two protocol facts that bound the reading

- **Exact-answer agreement is 0/70 for every EXL3 candidate, and 1/70 for official FP8** (one
  math item, a 113-token answer both models pass). Long chains of thought differ token-wise on
  essentially every item, so pass/fail outcome is the only meaningful pairing unit; nothing
  here is a generated-text match claim.
- **Four BF16 items end at the 5,120-token completion cap with no letter emitted and are
  scored as failures** under the plan's frozen addendum, so the control itself is depressed by
  the cap; per-model counts are in the table (this edition: 3). The earlier 2,048-cap control,
  where BF16 lost 7/70 to truncation, is retained unchanged at
  [`receipts/public-capability-bf16-superseded-cap2048.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/public-capability-bf16-superseded-cap2048.json).

### Status of this evidence

This is a **first public, licence-compatible, item-paired benchmark, not a leaderboard claim**.
The honest next step is more items, which is the plan's own P1: HumanEval+/MBPP-style
executable cases, IFEval-style constraint following, tool schemas, and a larger MMLU-Pro draw.
No capability claim on this card graduates before that — including this edition's pass.

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
**32/40**; all eight differing answers still passed their contracts. Wilson 95 % lower bound
is 91.2 %. This is a transparent smoke suite, not a public leaderboard; full responses are in
the [run receipt](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/tasks-v2-ctx.json);
its extracted-value agreement field is superseded by the
[strict rescore](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/task-retention-v2-strict-rescore.json).

## Long context, verified by generation

Not an allocation test. A unique code is planted in held-out literary text at three depths and
the model is asked to return it, on a server capped to a 5090-sized budget:

| profile | prompt tokens | needle depth | retrieved exactly | wall | request-average prompt tok/s* |
|---|---:|---|---|---:|---:|
| BF16 input, MTP-3 | 28,613 | 0.1 / 0.5 / 0.9 | **3/3** | 6.3 s | ~4,500 tok/s |
| BF16 input, MTP-3 | 113,345 | 0.1 / 0.5 / 0.9 | **3/3** | 34.3 s | 3,301 tok/s |
| BF16 input, MTP-3 | 196,857 | 0.1 / 0.5 / 0.9 | **3/3** | 76.1 s | 2,588 tok/s |
| int8 input, MTP off | 227,334 | 0.1 / 0.5 / 0.9 | **3/3** | 94.8 s | ~2,400 tok/s |
| **int8 input, MTP-3, 8.4 MP cap** | **261,794** | 0.5 | **1/1** | 123.1 s | 2,127 tok/s |

**13/13 exact retrievals.** \*This is prompt tokens divided by total request wall time,
including decode, queueing and HTTP overhead — not an engine-timed prefill measurement.
The request-average prompt-token rate falls with length, consistent with increasing attention
cost; this receipt does not attribute the decline.

For text-only proof, the harness applies the chat template itself and posts to `/completions`,
so the server tokenizer can size the exact final prompt. The multimodal chat path is also
validated at length, but it **must** use
`--mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}'`: one combined request
preserved **236,824 prompt tokens**, retrieved the planted code and read the image correctly.
The text-only wrapper is:

```text
<|im_start|>user
{your long prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Native 262,144 under a 32 GB-equivalent engine budget

The opt-in `VLLM_EXL3_EMBED_BITS=8` overlay narrows the input embedding table. At
248,320 × 5,120, BF16 costs **2.543 GB resident**; the operation is a gather, so per-row
symmetric int8 halves it without putting another matmul on the serving path.

| | BF16 input, MTP off | int8 input, MTP off | **int8 input, MTP-3 + 8.4 MP cap** |
|---|---:|---:|---:|
| embedding table | 2.543 GB | **1.272 GB** | **1.272 GB, shared with draft** |
| resident weights | 19.31 GiB | **18.13 GiB** | **18.41 GiB** |
| largest configured context tested | 229,376 | **262,144** | **262,144 (native)** |
| KV allocated at that length | 240,080 tokens | 279,007 tokens | **266,612 tokens** |
| v3 full-suite mean KLD | 0.009673 | 0.009738 | **0.009738 (same target)** |
| multimodal smoke score | 24/30 | 24/30 | **24/30 (identical)** |

Fidelity cost of the input overlay, measured on the v3 suite: **+0.000065 mean KLD**, 95 % CI
[+0.0000046, +0.00013], 49/136 v3 contexts; corrected v3-subset point estimate +0.000082. The
v5 held-out run scored the serialized checkpoint only, so the overlay delta has not been
remeasured there and the v3 attribution stands as the only evidence for it.
MTP does not alter the accepted target distribution; its draft is separately quantized and
shares the target input table.

The native MTP profile is a served proof, not allocation arithmetic: 261,794 text tokens
retrieved exactly; a 3,072 × 2,304 image returned `red, blue`; and a single request combining
that seven-megapixel image with 229,910 measured text tokens produced a **236,824-token**
prompt and exact `1376346594 | red, blue`. Warmed 256-token single-stream decode measured
98.72 tok/s in one run.

Scope matters. The server was an RTX PRO 6000 with vLLM capped to **30.24 GiB**, below the
30.44 GiB budget of a 31.39 GiB RTX 5090 at utilisation 0.97. That proves the engine budget
and served path, but the physical GPU retained memory beyond the cap. A real 5090 rerun
remains pending.

The captured run named the pinned image digest but did not preserve a launch-time full-rootfs
manifest. The published rerun harness now verifies every extracted-rootfs entry and the three
installed patch hashes before launch; that stronger image check applies to reruns, not
retroactively to this historical result.


```bash
-e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
  ... --max-model-len 262144 --gpu-memory-utilization 0.97 --max-num-seqs 1 \
      --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
```

Two model files must pass their existing quant config into `VocabParallelEmbedding`:
`qwen3_5.py` and `qwen3_5_mtp.py`. Backends without an `embedding()` method retain BF16.
The exact patched files and SHA-256 digests are in the companion repository.

### Correction: no second resident MTP embedding

An earlier card revision said the draft kept a second resident table and that separately
quantizing it to int4 preserved acceptance. Runtime receipts contradict that:

```text
Detected MTP model. Sharing target model embedding weights with the draft model.
```

The draft table is materialized during load, then replaced by the target embedding. The
int8/int4 acceptance comparison therefore did **not** exercise int4 at inference; 56.1 %
versus 56.7 % was run noise. The separate draft-width environment variable and int4 method
were removed. PR #319 carries the same correction in its body and public comment.

### Why the image ceiling is load-bearing

The initial MTP-3 profile retained the model's 16,777,216-pixel image default. It needed
9.13 GiB of KV against 8.59 GiB available and refused native length. vLLM reserves activation
memory for the maximum allowed image at startup; that default, not a second draft embedding,
was the remaining lever.

At **8,388,608 pixels** and the stricter 30.24 GiB budget, peak activation is 1.78 GiB,
available KV is **9.31 GiB**, and the engine allocates **266,612 tokens**. The cap still
accepts the tested 7,077,888-pixel image. It does not claim that the original 16.8 MP ceiling
fits.

Reducing concurrency to one and capturing only the speculative row count are also part of this
profile. A compact KV-group experiment remains rejected: it reduced padding from three layers
to one but raised graph memory from 0.46 to 1.25 GiB and increased total required KV.
Machine-readable proof and exact artifact hashes:
[`receipts/native-mtp-8mp-amendment.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/native-mtp-8mp-amendment.json).

## Throughput

Median of 3 runs on one RTX PRO 6000 Blackwell, `--max-num-seqs 8`, greedy, 256 output tokens.

| configuration | TG C1 | TG C4 | TG C8 | PP 2k | PP 6k |
|---|---:|---:|---:|---:|---:|
| B12X everywhere (as published upstream) | 56.0 | 197.2 | 397.3 | 5,078 | 5,188 |
| **+ prefill routing (shipped here)** | 56.0 | 197.0 | 398.8 | **5,250** | 5,249 |
| + FP8 prefill (**rejected**, see below) | 56.7 | 199.5 | 401.6 | 6,650 | 6,285 |

The native MTP-3 / 8.4 MP / 262,144 profile measured **98.72 tok/s C1** in one warmed
256-token run. It is a capability receipt, not part of the three-run concurrency matrix above.

**A whole class of matrices was on the wrong kernel at prefill.** The serialized EXL3 path
routes every K6/MCG shard with 128-divisible dimensions to B12X's native kernel *before* the
reconstruct dispatch is consulted — on this build that is 208 attention projections, 64
`down_proj` and the head. B12X is the right choice at decode (measured ~5x faster at m≤8) and
the wrong one at prefill (reconstruct+GEMM wins 1.08-1.40x at m=2048). Routing by row count
inside the opaque op gains **+3.4 % prefill and costs +0.0000377 mean KLD** (95 % CI
[−0.00001, +0.00009], 59/136 v3 contexts — a coin flip, i.e. free).

**FP8 prefill is measured and rejected.** Emitting the reconstructed weight directly as E4M3
and using `torch._scaled_mm` gives **+31 % prefill (5,078 → 6,650)** but costs **+0.0141 mean
KLD**, which lands this build at 0.0237 — worse than official FP8. Row-wise scaling (per-token
activation, per-channel weight) did not help: the loss is FP8 *activations*, not scale
granularity. It stays off; `VLLM_EXL3_PREFILL_FP8=1` enables it for anyone who wants prefill
over fidelity.

## Multimodal

First quantitative multimodal check in this family, on deterministic synthetic images with
exactly known answers, scored by exact match (30 cases, greedy, thinking disabled):

| task | what it tests | score |
|---|---|---:|
| 6-digit code, 5×7 bitmap font | fine detail | **8/10** |
| bar chart, tallest bar by index | spatial ordering | **9/10** |
| colour grid, count of one colour | counting and localisation | **7/10** |
| overall | | **24/30 (80 %)** |

The generator is published (`tools/vision_eval.py`) so the same cases can be run against any
candidate. This build and its siblings carry the same BF16 vision tower; all score 24/30.
The native MTP profile also answered a 3,072 × 2,304 two-colour image exactly, both alone and
inside a 236,824-token combined prompt. This is still synthetic, not OCR/document/video quality.

## Serving

### Pinned image, unmodified fallback

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6-context \
    --served-model-name qwen38 --quantization exl3 --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 196608 --gpu-memory-utilization 0.97 --max-num-seqs 4 \
    --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 --port 8000
```

Nothing is encoded at load: no `VLLM_EXL3_ONLINE_TRELLIS_BITS`, no cache directory. This is
what the pinned image runs without modification. It does not include the input-table overlay,
graph decode or prefill routing, so it is not the native-window profile.

### Patched native-window profile

Clone the companion repository, verify the three executed files against the native-MTP
receipt, then mount them over the pinned image:

```bash
set -euo pipefail
git clone https://github.com/malaiwah/qwen38-27b-exl3
cd qwen38-27b-exl3
cat <<'SHA256' | sha256sum -c -
2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee  tools/vllm-exl3-prefill-dispatch.py
04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7  tools/vllm-qwen3_5-embed-quant-config.py
0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab  tools/vllm-qwen3_5_mtp-embed-quant-config.py
SHA256
PATCH=$PWD/tools
VLLM=/opt/venv/lib/python3.12/site-packages/vllm
docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro \
  -v "$PATCH/vllm-exl3-prefill-dispatch.py:$VLLM/model_executor/layers/quantization/exl3.py:ro" \
  -v "$PATCH/vllm-qwen3_5-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5.py:ro" \
  -v "$PATCH/vllm-qwen3_5_mtp-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro" \
  -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
  -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6-context \
    --served-model-name qwen38 --quantization exl3 \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
    --max-model-len 262144 --gpu-memory-utilization 0.97 --max-num-seqs 1 \
    --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 8000
```
The container listens on all interfaces internally, but Docker publishes the port to host
loopback only. For remote clients, keep that binding and put an authenticated TLS proxy in
front; do not expose this unauthenticated generation endpoint directly.


The changes remain open upstream:
[#314](https://github.com/local-inference-lab/vllm/pull/314),
[#316](https://github.com/local-inference-lab/vllm/pull/316),
[#318](https://github.com/local-inference-lab/vllm/pull/318), and
[#319](https://github.com/local-inference-lab/vllm/pull/319).

Load-bearing details, unchanged from the siblings: `--quantization exl3` is mandatory; the
`ignore` list is mandatory and its anchoring is subtle (`re:.*visual\..*` matches,
`re:.*\.visual\..*` silently does not and **crashes** startup,
[#311](https://github.com/local-inference-lab/vllm/issues/311)). The tested config ignores
`mtp.*` from the generic BF16 online overlay; EXL3-owned draft projections are selected first
and still load from serialized tensors. `truncation:false` is required for large images
([#313](https://github.com/local-inference-lab/vllm/issues/313)); `max_pixels:8388608` is
required for the measured native-MTP memory profile.

## What is not verified

The 40-case deterministic downstream smoke has zero regressions against BF16, but it is not a
public benchmark, and the 70-item MMLU-Pro pass above is a narrow pre-registered pass, not a
capability claim. Still unverified: real OCR/document/video quality, long-context reasoning
beyond planted-code retrieval, YaRN-1M, multi-GPU or TP>1, non-SM120 GPUs, and quant-specific
safety regression testing.

Recipe development used the v3 analysis partition. A frozen source-disjoint v4 qualification
was run once after selection; after the same conservative overlap correction all three EXL3
builds still beat FP8 on 36/36 paired contexts. The v5 held-out suite then re-ran the same
question at 10,480,640 scored positions and reproduced the ordering and the paired wins.
Absolute KLD still differs by suite, and the three sets of magnitudes may not be compared with
each other.

Public capability remains partial even though this edition clears its bar. The paired MMLU-Pro
matrix above now covers all six models, and this edition is the **only** candidate whose
BF16-pass retention lower bound reaches the pre-registered 90 % (56/57, 90.7 %); K4 and official
FP8 read 88.1 %, the hydrated build and online K5/K6 read 85.6 %. At 70 items that bar cannot be
certified by anything giving up more than one paired pass, and every interval overlaps every
other, so the matrix separates no two models and this pass is not a superiority result. Missing
entirely: executable-code, constraint-following, tool-schema and larger-draw evidence — the
plan's own P1.

Machine-readable base evidence is `release-evidence.json`; the native-MTP result is the
[amendment](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/native-mtp-8mp-amendment.json).
Comparator and sibling captures are published in the
[fidelity dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3).
This edition's v3 report is in the companion repository, but its replay capture is not yet
published; the 0.009738 / 0.009459 result is therefore not independently replayable from the
dataset snapshot. The v5 run publishes its manifest, corpus fetch log and cumulative receipts
(`receipts/kld5-*.json`) but no hidden-state captures, for the scratch reason stated above.

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — EXL3 Trellis format,
  LDLQ calibration, MCG codebook, conversion pipeline.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path and the B12X native
  K6 kernel this build routes around at prefill.
- [Qwen](https://huggingface.co/Qwen) — base model and official FP8 derivative.
- An independent RTX 5090 tester — the memory model these context numbers are calibrated on.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
