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
> crash-and-resume; the successor's command produces `mcg` throughout. That is a claim about
> the recipe, not about the bytes — see [Reproducing this quant](#reproducing-this-quant).

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

**Measured resident weights: 17.89 GiB (19.21 GB)** on a single RTX PRO 6000 Blackwell. The
one NVFP4 build of this architecture whose resident weights were measured here under identical
flags is `unsloth/Qwen3.8-27B-NVFP4` at **21.34 GiB (22.91 GB)**, so this build holds
**3.70 GB less** resident weight
([`docs/22-results-iteration-2.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/22-results-iteration-2.md)).
Those are load-time device allocations. They are **not** the same quantity as the **21.92 GB**
of `nvidia/Qwen3.6-27B-NVFP4` and the **23.42 GB** of `unsloth/Qwen3.8-27B-NVFP4` quoted
elsewhere: those two are checkpoint sizes — serialized bytes on disk, never resident memory —
and no resident figure has ever been measured here for the `nvidia` checkpoint, which is a
different model generation (Qwen3.6). Per-role bit widths are not directly comparable
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
>
> **That `0.955` is a per-card measurement, not a constant.** It is the value that qualified on
> one board, `GPU-506a575d` (32,607 MiB, 458 MiB of it held by the driver); a second physical
> RTX 5090 needed **`0.956`**, missing at 0.955 by about **0.01 GiB**
> ([`receipts/second-5090-datapoint.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/second-5090-datapoint.json)).
> Two nominally identical boards differ in exactly two quantities no configuration can move —
> the driver's framebuffer reserve and the CUDA context size — and a **68 MiB** perturbation in
> either was measured to be enough to flip a gate
> ([`receipts/qualification-24gib-capped.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-24gib-capped.json)
> → `residual_risk_versus_a_physical_board`), while one thousandth of utilisation is only about
> 32 MiB. **So if a card refuses to start or OOMs at startup, raise utilisation by `0.001` at a
> time** rather than dropping the window — and do not lower `max_pixels` to make room, because
> at fixed utilisation that enlarges the KV pool and makes the large-image case fail *sooner*.

## Which of the four builds

Same architecture and tokenizer. The first KLD column is the v5 held-out suite (10,480,640
scored positions); the second is the older overlap-corrected 127-context v3 subset, kept for
continuity. Absolute KLD is suite-specific, so the two columns are not comparable with each
other — only the ordering within a column is. Capacity uses each card's documented profile:
hydrated, online and K4 are real RTX 5090 MTP-3 tests; context is MTP-3 with an 8.4 MP cap,
qualified on a physical RTX 5090 at utilisation 0.955.
These profiles are not interchangeable
([collection](https://huggingface.co/collections/malaiwah/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Overlap-corrected v3 mean KL divergence versus demonstrated or configured context. Circles are real RTX 5090 MTP-3 results: hydrated and online K6 at 185,600, K4 at 262,144. Stars have generation proof: online K5 at 206,400 on the 5090, and the context edition at 262,144 with MTP-3 and an 8.4 MP image cap under a 30.24 GiB engine budget; that engine-budget star has since been superseded by a physical RTX 5090 qualification of the context edition at 265,122 KV tokens and utilisation 0.955." src="assets/context-frontier-light.svg">
</picture>

| build | download | resident | v5 mean KLD | corrected v3 mean KLD | context profile | pick it when |
|---|---:|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.002760** | **0.007172** | ~180k | fidelity first, smallest download |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.60 GB | 20.32 GiB | 0.003210 | 0.007945 | ~180k | you want the attention width knob at launch |
| [-context](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | 20.70 GB | **18.41 GiB** | 0.003509 | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | native window, hardware-qualified on a physical RTX 5090 |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB\* | 17.89 GiB | 0.010604 | 0.029679 | 262,144 | smallest footprint, native context without any overlay |

**Byte and memory conventions for this table.** The download column is whole-tree bytes —
every published file of the artifact as its release evidence counted it
([`receipts/collection-index.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/collection-index.json),
`serialized_bytes.whole_tree_bytes`: hydrated 21,610,933,884 B, K5/K6 30,597,231,933 B,
context 20,696,053,306 B) — and they are serialized bytes on disk, never resident memory.
\*This build's release evidence records no tree count, so its own row is the sum of its
safetensors shards, 28,313,841,196 B, read from the published repository. The context
edition's resident weight is measured twice: **18.41 GiB** as run on the rental RTX PRO 6000
engine-budget proof and **18.19 GiB** on the physical RTX 5090 at the qualified `0.955`
profile. This table prints the larger figure deliberately, because
[`receipts/vram-class-verdict.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/vram-class-verdict.json)
elects 18.41 GiB for every class prediction; the 0.22 GiB gap is the rental-versus-5090 delta,
not a change in the checkpoint.

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
| resident weights (measured) | **19.21 GB** (17.89 GiB) | not measured here — Qwen3.6 generation | 22.91 GB (21.34 GiB) |
| checkpoint size on disk | 28.31 GB | 21.92 GB | 23.42 GB |

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

**How closely these absolute numbers may be read.** Each mean is a body-only replay value: both
operands are projected through the one shared BF16 head, and the replay path is not the engine's
own logit path. Replaying the unquantized model against its own live logits measures
`KL(live ‖ replayed)` = **5.83e-04** — 32 v5 shard-0 contexts, 65,504 scored positions,
context-bootstrap 95 % CI [5.15e-04, 6.64e-04], top-1 99.10 %, on the **same suite, reference
capture and shared BF16 head as the means above**
([`receipts/replay-live-floor-v5.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/replay-live-floor-v5.json)),
superseding the six-context v3 derivation of 6.54e-04
([`receipts/v3-qualification-bf16.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/v3-qualification-bf16.json)),
which its interval contains —
and moving hidden-state storage from BF16 to fp32 moves a candidate's KLD by 5.6 %
([docs/24](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/24-p0-results.md)). Absolute
values are therefore **within-suite numbers**: they carry a ~6e-4 implementation offset plus a
~5 % storage systematic, and absolute differences below about 1e-3 are not resolvable. Both
offsets are **common-mode** — every candidate replays through the identical path — so **paired
differences and orderings are the resolvable quantity**: hydrated − online K5/K6 is −4.50e-04
[−4.69e-04, −4.33e-04] on 4,922 of 5,120 contexts
([`receipts/kld5-10M-paired.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-10M-paired.json)),
smaller than the replay floor and resolved *because* the floor cancels in the pairing. The floor is now derived
**inside the suite** rather than on six out-of-suite v3 contexts, and the rule it licenses is
unchanged. What it does **not** license: it is not a claim that candidate KLDs are 11 % smaller, and
it does not let any single absolute mean be read more finely — the 5.83e-04 figure is a mean over 32
contexts whose own means span 3.09e-04 to 1.63e-03 with a worst single position of 0.2534. It is also
not the cross-engine floor (0.000507), which is a different control. Method of
record:
[`docs/42-kld-method.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/42-kld-method.md).

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/kld-all-measurements-dark.svg">
  <img alt="Every KL divergence this project has measured, four panels on four protocols with deliberately no shared axis. Panel A, top left — our v5 held-out suite, shard 0: 512 contexts, 1,048,064 scored positions, 330 source clusters, identical for every candidate; y is KL(BF16 reference || candidate) in nats per token on a log scale, x is serialized bytes on disk in GiB and never VRAM, resident weights or KV. Ten candidates with source-cluster bootstrap 95 % intervals and their p99.9: GGUF Q8_0 0.001087, GGUF Q6_K 0.002035, hydrated K5/K6 0.002700, online K5/K6 0.003141, context edition 0.003409, GGUF UD-Q5_K_XL 0.004444, official Qwen FP8 0.005197, K4 0.010345, Unsloth NVFP4 0.030115 and gittensor NVFP4 (RTX5090) 0.062163. The measured llama.cpp-versus-vLLM cross-engine floor, 0.000507 mean at 99.07 % top-1, is drawn as a dashed reference line; filled squares are llama.cpp rows that contain that term, hollow squares subtract it naively, circles are vLLM rows that never carried it. The five candidates with no published serialized-byte receipt — online K5/K6, official FP8, K4, NVFP4 and gittensor NVFP4-5090 — sit in a narrow lane at the right of the same panel on the same y-axis, each labelled with the reason instead of being given an invented x. Panel B, top right — the same suite's ladder checkpoints, cumulative 1,048,064 to 10,480,640 scored positions across ten shards, 5,120 contexts and 842 source clusters at 10M, five vLLM builds only because no GGUF candidate ran all ten shards; every mean moves by less than 2.9 % of its own value across the tenfold increase and no ordering changes. Panel C, bottom left — the two prior suites, each on its own y-axis with a hatched 'NOT ONE AXIS' barrier between them: C1, the corrected v3 suite, 127 contexts and 259,969 positions, printing the measured ratio of each candidate's v3 mean to its own v5 shard-0 mean (official FP8 2.46x, online 2.53x, hydrated 2.66x, context 2.75x, K4 2.87x, NVFP4 3.08x — a 1.25x spread, so no single conversion factor exists, while the ordering is identical in both suites); and C2, the source-disjoint v4 qualification, 36 contexts and 73,692 positions. Panel D, bottom right — a protocol we have never run: turboderp's published chart labels read off his own images, OpenWebText 8 x 8192 = 65,536 formatted positions, his BF16 reference and his output head inside the measured path, with two of our builds present only as dashed vertical decoder-weight markers carrying no y-value. The footer states the two rules the figure exists to enforce: RULE 1, the engine term is not shared — every GGUF value in panel A contains the 0.000507 cross-engine floor and every vLLM value does not, so the squares are upper bounds and the floor-subtracted values are estimates and not identities because KL is not additive; RULE 2, no cross-panel ratio is meaningful, because the panels differ in corpus, context length, scored-position selection, reference numerics, vocabulary handling and head placement." src="assets/kld-all-measurements-light.svg">
</picture>

*The widest single view of the evidence: **A** is the only panel where every family appears
together (v5 shard 0, 512 contexts, 1,048,064 positions, ten candidates), **B** is the same
suite's 1M → 10M ladder (five vLLM builds, 5,120 contexts at 10M), **C1/C2** are the superseded
corrected v3 (127 contexts, 259,969 positions) and the source-disjoint v4 (36 contexts, 73,692
positions) on separate y-axes behind a barrier, and **D** is turboderp's own published protocol
(OpenWebText, 65,536 positions), which we have never run. Two rules travel with the figure: the
cross-engine floor belongs to the llama.cpp rows only, and no ratio across panels means anything.
Generated by
[`tools/make_master_kld_chart.py`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/tools/make_master_kld_chart.py),
which reads every one of our values from `receipts/` at runtime.*

**These numbers are re-derivable, not merely re-runnable.** 5,240,320 scored positions — five
candidates x 512 contexts x 2,047 positions — reproduce **bit-for-bit** across independent runs
with separate model loads and different harness generations: every measured field identical,
including the complete per-context arrays and the whole bootstrap block, with only the capture
directory paths and the additively-added tail histogram differing
([`receipts/capture-determinism.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/capture-determinism.json)),
and a third harness generation's unwindowed `--score-from 0` control returns the same shard-0
means to the last digit
([`receipts/scored-window-offset.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/scored-window-offset.json)).
The scope is part of the claim and travels with it: one GPU, one driver, one pinned rootfs,
`enforce_eager=True`, `max_num_seqs=1`, one context per forward, 512 MiB bf16 KV. It is **not** a
claim that vLLM is bitwise deterministic in general — nothing here covers CUDA graphs,
`max_num_seqs > 1`, chunked prefill with more than one chunk per context, other GPUs or drivers, or
anything downstream of the logits.

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
- **Captures: the reference survived, the candidates did not.** The five candidates' hidden
  states and the BF16 references for shards 1-9 were deleted shard by shard to fit 135 GB of
  scratch. The **shard-0 BF16 reference was kept and is published**, together with the suite, all
  ten shard views and 79 per-shard reports, so a new candidate can be scored against the identical
  contexts without recapturing the reference — see [Reproduce this](#reproduce-this).

## Against GGUF, measured on our suite

The comparator set used to stop at official FP8, which is a throughput format whose quality is
Q4-to-Q5 class, so llama.cpp's `Q8_0` and `Q6_K` are the honest bar. They have now been measured
on our own suite, and for this build the result is unambiguous: **every GGUF measured here beats
it, including the smallest of the three**. The one row it does beat is the checkpoint its readers
actually weigh it against — `unsloth/Qwen3.8-27B-NVFP4`, the other 4-bit-weight-class build for
this architecture, which has now been measured on the identical shard and is **2.9x** this
build's divergence.

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
| `unsloth/Qwen3.8-27B-NVFP4` @ `9c73e2da` | vLLM | 0.030115 | n/a, same engine | 93.16 % | 1.6228 | — |
| `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` @ `69274a0d` | vLLM | 0.062163 | n/a, same engine | 89.85 % | 2.5911 | — |

**The engine floor, measured and not assumed.** A GGUF row carries llama.cpp-versus-vLLM numerics
on top of quantization error, so that term was measured the same way: the unquantized **BF16
GGUF** against the vLLM BF16 reference, identical token ids, the same shared head, the same 512
contexts — **0.000507** mean, 99.07 % top-1, p99.9 0.0113
([`receipts/gguf-report-engine-floor.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gguf-report-engine-floor.json)).
Every GGUF row above contains that term; no vLLM row — ours or FP8's — does. **KL is not additive,
so the net column is an estimate, not an identity**: the measured GGUF value is an upper bound and
the net figure is the naive lower one. Note the direction of that asymmetry for this build — the
floor can only *inflate* a GGUF number, so subtracting it makes every GGUF row **better**, not
worse, and there is no reading of the cross-engine term under which any GGUF row here stops beating
this build.
Four cells read `—`, for two different reasons. Two of them are the builds that ship BF16
attention for the runtime to encode at load, including this one (28.31 GB download, **17.89 GiB
resident**), so their disk bytes are not a like-for-like payload against a GGUF file; the payload
figures are `immutable_payload_bytes` from
[`receipts/collection-index.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/collection-index.json)
(hydrated 21,610,916,123 B = 20.127 GiB, context edition 20,696,033,532 B = 19.275 GiB; the table
truncates both to two decimals) and are serialized bytes, never VRAM. The others are the NVFP4
builds, for which we publish no serialized-byte receipt of our own: unsloth's 22.91 GB is measured
**resident weights** and its 23.42 GB is a **checkpoint size**, gittensor's 18.77 GiB is likewise
measured resident weights, and none of these is the payload quantity this column holds, so they
get no x rather than an invented one. The FP8 figure is resident weights and is labelled as such.

**The p99.9 column, and why it differs from the tail table above.** These p99.9 values are each
report's **exact** shard-0 p99.9 as the comparator receipt read them; the
[tail table above](#distribution-fidelity--v5-held-out-suite-10480640-scored-positions) quotes the
**bin-bounded cumulative estimate** from the 560-bin histogram, whose bins are about 5.6 % wide —
this build reads 0.5555 there and 0.5576 here, hydrated 0.1319 and 0.1313, and each exact value
lies inside the bin the estimate names. The two differ by construction, not by measurement.

**NVFP4 on the identical shard — the comparison this build's readers actually make.**
`unsloth/Qwen3.8-27B-NVFP4` at revision `9c73e2da` is the other 4-bit-weight-class checkpoint for
this architecture, and it is served by **the same vLLM build as this row**, so unlike the GGUF rows
it carries **no cross-engine term at all** and is directly comparable to ours with nothing
subtracted or estimated. On the same 512 contexts, the same 1,048,064 positions and through the
same shared BF16 head, it measures **0.030115** mean KLD, 95 % CI [0.027637, 0.032965], **93.16 %**
top-1, median 0.009584, p95 0.10051, p99 0.33546, p99.9 **1.6228**, exact worst position 10.6285
and mean JSD 0.010104 bits
([`receipts/kld5-1M-nvfp4.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-nvfp4.json),
with the run's own account in
[`receipts/nvfp4-v5-measurement.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/nvfp4-v5-measurement.json)).
That is **2.9x this build's 0.010345 at the same 4-bit weight class**, 5.8x official FP8, 8.8x the
context edition, 11.2x the hydrated build and 27.7x `Q8_0` as measured; its p99.9 is 2.9x this
build's 0.5576. Its own histogram brackets that quantile at [1.5849, 1.6788] with a 1.6128 point
estimate, which contains the exact 1.6228 — the same construction difference described just above,
in the one row where both numbers are published.

**Paired per context, which is a stronger statement than any ratio of means: NVFP4 loses every one
of 512 contexts, against both comparators it was paired against.** +0.026706 against the context
edition (95 % CI [+0.024465, +0.029285], **0 wins to 512**) and +0.024918 against official FP8
([+0.022756, +0.027424], **0 wins to 512**) — not one context anywhere in the shard where it is the
better of the pair
([`receipts/kld5-1M-paired-nvfp4.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-paired-nvfp4.json)).
It was not paired against this build, so the 2.9x above stays a ratio of means and is not presented
as a win count.

**gittensor's "RTX5090" NVFP4, measured on the same shard because its card claims the 32 GB /
262,144-token axis by name.** `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` at revision
`69274a0d` (ModelOpt W4A4 body, MTP and vision left BF16, FP8 KV cache baked into its config) is
served by **the same vLLM build as our rows**, so it carries no cross-engine term. It measures
**0.062163** mean KLD, 95 % CI [0.058491, 0.066360], **89.85 %** top-1, p99.9 **2.5911** — the
weakest row on this table, at 2.1x unsloth's NVFP4 and 6.0x this build. **Paired per context it
loses every one of 512 contexts to this build** (+0.051818 in this build's favour, 95 % CI
[+0.048909, +0.055160], **512 wins to 0**), every one of 512 to official FP8, and **511 of 512 to
unsloth's NVFP4** (+0.032048, [+0.030711, +0.033583]) — the same weight format at 2.57 GiB less
measured resident weight (18.77 vs 21.34 GiB, identical flags, engine-reported), which prices that
memory saving honestly: roughly double the KLD. A bf16-KV control capture moves its mean by only
+0.000365 [+0.000058, +0.000679], so its baked FP8 KV cache explains about 1 % of the gap to
unsloth — the rest is the weight conversion itself. Its card's serving numbers (18.8 GB weights in
VRAM, 275,941-token FP8 KV pool, 80.6 tok/s decode, native 262,144 on one 5090) are **its own
claims, which we did not run**; our measured 18.77 GiB resident is consistent with the first of
them, and consistency is not verification. Its only published fidelity evidence is a 20-item smoke
that its own card says not to treat as scores
([`receipts/kld5-1M-gt5090.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-gt5090.json),
[`receipts/kld5-1M-paired-gt5090.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kld5-1M-paired-gt5090.json),
full account with the checkpoint's composition, digests and mirror in
[`receipts/gittensor-nvfp4-rtx5090.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gittensor-nvfp4-rtx5090.json);
archival mirror
[`malaiwah/Qwen3.8-27B-NVFP4-RTX5090-archival-69274a0d`](https://huggingface.co/malaiwah/Qwen3.8-27B-NVFP4-RTX5090-archival-69274a0d)).

**Where this build sits, stated plainly.** Third from last, and last among the rows a reader
choosing on fidelity would shortlist. Its 0.010345 mean is **2.6x** `UD-Q5_K_XL`'s net 0.003936,
**6.8x** `Q6_K`'s net 0.001528, **17.9x** `Q8_0`'s net
0.000579 and **2.0x** official FP8's 0.005197; its top-1 is the lowest of those rows at 95.91 %,
and its p99.9 of
0.5576 is **2.6x** `UD-Q5_K_XL`'s 0.2144, **7.0x** `Q6_K`'s 0.0794 and **2.3x** FP8's 0.2440. It is
beaten by every GGUF measured here, including `UD-Q5_K_XL`, the smallest of the three at 18.83 GiB
of serialized weight. What it is **not** beaten by is either 4-bit alternative: unsloth's NVFP4
measures 2.9x this build's divergence and gittensor's 6.0x, on the identical positions, in the
same engine, with no floor to argue
about. This build's argument was never fidelity — it is the 17.89 GiB footprint and
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

**Correction, 2026-08-16 — the byte axis in the table above is not one axis, and every mixed
comparison flattered us.** A GGUF row is the **whole file** of a **text-only** artifact; our row is
**tensor payload** of a **multimodal** tree that also carries an MTP draft. Read from each artifact's
own tensor table, without downloading any payload
([`cross-candidate-byte-accounting.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/cross-candidate-byte-accounting.json)):

| candidate | file | tensor total | `token_embd` | `output` | transformer body | multimodal deployed |
|---|---:|---:|---:|---:|---:|---:|
| GGUF `Q8_0` | 27.05 | 27.04 | 1.258 (Q8_0) | 1.258 (Q8_0) | **24.526** | 27.92 (+ `mmproj-BF16` 0.867) |
| GGUF `Q6_K` | 21.31 | 21.30 | 0.971 (Q6_K) | 0.971 (Q6_K) | **19.360** | 22.18 (+ `mmproj-BF16` 0.867) |
| GGUF `UD-Q5_K_XL` | 18.83 | 18.82 | 0.814 (Q5_K) | 0.971 (Q6_K) | **17.034** | 19.70 (+ `mmproj-BF16` 0.867) |
| online K5/K6 | 28.50 | 28.47 | 2.368 (BF16) | 0.889 (K6) | **24.119** | 28.50, vision inside |
| K4 | 26.37 | 26.37 | 2.368 (BF16) | 0.593 (K4) | **21.463** | 26.37, vision inside |
| hydrated K5/K6 | 20.13 | 20.10 | 2.368 (BF16) | 0.889 (K6) | **15.726** | 20.13, vision inside |
| context edition | 19.27 | 19.25 | 2.368 (BF16) | 0.889 (K6) | **14.886** | 19.27, vision inside |

All figures GiB. The embedding and head widths are **not** uniform across GGUF tiers, and the vision
encoder is **absent from every GGUF text file** - it ships separately as `mmproj-BF16.gguf`, which no
earlier comparison of ours counted. What that does to the four published claims, two against us and
two for us:

1. **6 bits, `Q6_K` against hydrated: our sentence understated their byte spend roughly threefold.**
   "+1.186 GiB more file" is +1.198 on tensors and **+3.634 GiB of transformer body (23.1 % more than
   ours)**. Their fidelity win at 6 bits stands exactly as published - it is the *price* we
   mis-stated, in our own favour.
2. **6 bits, deployed: a multimodal `Q6_K` deployment is 22.18 GiB against our 20.13 GiB whole tree,
   so ours is 2.053 GiB smaller** and needs no second file.
3. **5 bits, `UD-Q5_K_XL` against the context edition: this claim was wrong against us.** We do not
   pay "0.445 GiB more" for the win - on transformer body **they** carry 2.148 GiB more (+14.4 %),
   and our deployed multimodal artifact is 0.422 GiB **smaller**.
4. **8 bits, `Q8_0` against online K5/K6: the bodies agree to 1.7 %** (24.526 against 24.119), the
   most format-comparable pair on the table, and **they win it cleanly**. That row is the reason the
   others are worth reading: this is not a table where every axis favours the author.

**Rule from here on, and it is printed rather than footnoted:** "at equal file bytes", "at equal
tensor bytes", "at equal transformer body" and "as a deployed multimodal artifact" are four different
claims, and whichever one a sentence means is written into the sentence. No fidelity number changes.

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

### Cross-citation: the same three GGUFs under llama.cpp's own protocol

The rows above are those GGUFs on **our** axis. They have also been measured on **theirs**, run
exactly as its authors run it, so the two can be cited side by side without either being converted
into the other: `llama-perplexity --kl-divergence` on **WikiText-2 raw test**, `n_ctx` 512,
**147,900 scored positions**, `KL(BF16 GGUF ‖ candidate)` with both operands inside llama.cpp and
each candidate's **own output head inside the measured path**, base `Mean PPL 6.950230 ± 0.044933`
([`receipts/wikitext-kld-run-a.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/wikitext-kld-run-a.json);
full protocol, delta by delta, in
[`docs/35-external-protocol-comparability.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/35-external-protocol-comparability.md)).

| quant | their protocol, their corpus | their top-1 | ours, net of the engine floor | theirs ÷ ours-net |
|---|---:|---:|---:|---:|
| `Q8_0` | 0.000926 ± 0.000042 | 98.761 % | ~0.000579 | 1.60x |
| `Q6_K` | 0.002286 ± 0.000108 | 97.875 % | ~0.001528 | 1.50x |
| `UD-Q5_K_XL` | 0.004426 ± 0.000167 | 97.178 % | ~0.003936 | 1.12x |

**The ordering is identical on both axes**, and the level difference is protocol rather than
disagreement about which quantization is better: their number is pushed down by scoring only the
second half of each window, by a single English corpus and by dropping base-side terms below
`log p ≤ −16`, and pushed up by having the candidate's own output head inside the measured path,
while both of our operands go through one shared BF16 head. Only our number carries a cross-engine
term, which is why the honest comparison is against our net column. **Correction, 2026-08-16:** this
paragraph used to call the output head "the large one". It is now measured on our own corpus and it
is not: replaying each candidate through its own head over all 512 shard-0 contexts and 1,048,064
positions raises its mean by **at most 5.28 %** (hydrated 5.01 % of head-inclusive divergence,
context 4.06 %, K4 1.17 %, unsloth NVFP4 2.64 %, and exactly 0 % for the official FP8 export, whose
head is byte-identical to the shared one — the internal control), every interval excluding zero
([`receipts/head-attribution-v5.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/head-attribution-v5.json)).
Scoring geometry is worth ≤4.9 % by the same kind of control. So the two protocol terms we have
quantified are together far too small to explain a 1.1-1.6x level difference: **the level gap is
not decomposed**, the leading unmeasured candidates being their 512-token English-encyclopedic
windows against our 2,048-token five-strata ones, and the width of a GGUF's own `output.weight`,
which is a different tensor from any head measured above.

**Their harness's own floor, measured on our hardware instead of assumed.** The `Minimum KLD`
column is negative for all three — −0.000080, −0.000056, −0.000077, i.e. **5.6e-5 to 8.0e-5** — the
uint16 16-nat log-probability encoding showing through rather than a candidate beating its own
reference. The same term appears in the perplexity: 6.9525 in the capture log against 6.950230 in
the scoring runs, identical weights on identical tokens, differing only by that stored round trip.

**Tokenization is not part of the difference, and that is a measured null result.** llama.cpp's
GGUF BPE and our Hugging Face tokenizer produce **bit-identical 297,194-token streams** over this
corpus — same int32 digest, no first divergence index — and the 296,960-token prefix that
`llama-perplexity` actually scores is identical too
([`receipts/wikitext-kld-token-identity.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/wikitext-kld-token-identity.json)).

**And one finding worth its own line: perplexity does not reproduce the KLD ordering.** `Q6_K` has
the *smallest* PPL delta of the three, **+0.00079** against the 6.950230 base, while `Q8_0` — the
better quant by every divergence statistic, including a mean 2.5x lower and 0.9 points more top-1
agreement — is **+0.00467**. A quantization that shifts the distribution can shift it in the
direction that happens to flatter a corpus mean, which is an argument for the metric this whole
section is built on and against ranking quants by perplexity delta.

What this cross-citation cannot do is put **this** build on their axis: `llama-perplexity` cannot
read an EXL3 checkpoint. The table at the top of this section, where every candidate is scored by
one harness on one suite, stays the primary comparison.

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

**Prefix caching is on in the recipe below, and it is on here and not everywhere.** At an
8,192-token window the KV pool is roughly thirty times the window, so a single request comes
nowhere near the pool ceiling and the failures that stopped the native-context profile cannot
occur. That is not an argument, it is the reason this recipe was measured separately: it starts
healthy on the promoted image with `--enable-prefix-caching --mamba-cache-mode align`, answers
a text and an image request exactly, and reports `enable_prefix_caching: True` in the engine
banner
([`receipts/production-image.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/production-image.json)).
The context edition's native 262,144-token recipe does **not** enable it, and its card explains
why
([`receipts/qualification-5090-apc.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-apc.json)).

**The release unit moved on 2026-08-16, and #51113 is why.** Upstream vLLM #51113 (mamba
`align` prefill-chunk splitting: a chunk that ends mid-block leaves its slot holding a short
state, which a later chunk then publishes anyway — wrong tokens, HTTP 200, no crash) merged
2026-08-06, after the pinned public image was built, and is still absent from it and from fork
head `fa033bd4e`. Cherry-picks were requested upstream on 2026-08-16 ([issue
#392](https://github.com/local-inference-lab/vllm/issues/392), [PR
#393](https://github.com/local-inference-lab/vllm/pull/393)). Until they land it is carried as
`tools/vllm-mamba-align-scheduler.py` (`sha256 b431c106…`), and the image this project serves
is now the four-module `localhost/vllm:gg-r34-patched-apc`, manifest `sha256:16a936b877b90f…`,
promoted from the three-module `localhost/vllm:gg-r34-patched` (`sha256:6eca4c693f01b6…`)
([`receipts/production-image.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/production-image.json)).
With prefix caching off the two images are not merely similar but behaviourally identical,
because the added module's changed function is unreachable unless `mamba_cache_mode` is `align`
— so the earlier hardware qualification carries over unchanged. One warning if you inspect the
image yourself: its build-time label `io.malaiwah.image.qualified` still reads `false` and is
**superseded by the receipts named here** — it was written before the image could possibly have
been qualified, and correcting it would add a layer and change the very digest that was
measured. That digest is local to the build host, so the recipes here reproduce its content
with sha256-verified read-only mounts over the pullable public base.

**What prefix caching buys.** On disjoint documents, so the cold case is genuinely cold: a
32,842-token prefix went **12.07 s cold → 1.04 s warm (11.6×**, 2,442 of 32,842 prompt tokens
recomputed, 92.6 % hit rate) and a 131,146-token prefix went **67.60 s → 2.31 s (29.3×**, 3,146
of 131,146 recomputed, 97.6 %); a 38-request schedule ran 84.0 s with the cache against 144.4 s
without
([`receipts/apc-poison-repro.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/apc-poison-repro.json)).
**And what we actually know about its safety.** Correctness was probed adversarially before any
of this shipped: seven freshly started servers, 38 requests each, **266 scored requests**,
nested token prefixes so later requests hit blocks published by earlier ones, and no prompt
length a multiple of the measured 1,600-token mamba block, so prefill chunks end mid-block by
construction — **zero corrupted responses, zero wrong answers, zero acceptance collapses, on
the unpatched image as well as the patched one**. Thresholds were committed before the first
server started; the worst repeated block was 15 characters against an 80-character threshold,
with no U+FFFD anywhere. Greedy chosen-logprob drift with the cache on is 0.1063 mean absolute
against a measured run-to-run floor of 0.0823 — drift, never an answer change. So the module is
carried as **insurance backed by upstream's own regression file** — 14 failed / 6 passed
against the vendored scheduler, 20 passed against this one — and **not** by a reproduction of
our own: we tried hard to reproduce the reported corruption and could not
([`receipts/mamba-align-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/mamba-align-defect.json)).

**LMCache is unmeasured by us.** It is not part of any recipe on this card, this project has
never run it, and it is the outstanding suspect in the one user report of prefix-cache
corruption we have. Nothing here says LMCache is safe; the evidence above covers vLLM's own
prefix cache and nothing else.

**#51812 is now recommended if you serve this recipe concurrently with MTP, and it stays an
overlay.** Upstream #51812 (Qwen GDN speculative gate ordering: the vendored code gathers the
speculative Q/K/V rows but hands the recurrent update the ungathered `a`/`b` gate tensors, so gate
row *i* can belong to a different token than Q/K/V row *i*) merged 2026-08-11 and is absent from the
promoted image. Whether that path is ever entered is no longer an argument — it was counted. A
CPU-only instrument over the engine's GDN metadata builder ran the three flags **this card**
publishes for prefix caching (`--max-model-len 8192`, `--enable-prefix-caching --mamba-cache-mode
align`, `--gpu-memory-utilization 0.85`) at **eight** concurrent streams, with MTP-3, fp8 KV and
`--max-num-batched-tokens 2048` supplied by the arm, and the defective path **was entered: three
events in 5,825 metadata builds, 0.515 per thousand**, over 468 requests with zero errors and a
prefix-cache hit rate reaching 50.1 %
([`receipts/gdn-gate-concurrency.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-gate-concurrency.json)).
The mechanism, in one sentence you can act on: at eight streams with prefix caching on, a short
non-speculative request can land between speculative ones in the same batch, and the unpatched
gather then misaligns the gates. All three firings had one shape — six speculative decodes plus one
non-speculative request, that request beginning at token index 20 and displacing one four-token
speculative decode, whose four gate rows were read from the wrong tokens. Two boundaries on that
number. The same instrument saw **zero** events in 3,329 builds at eight streams with prefix caching
**off** (below 0.90 per thousand, 95 % upper bound) and zero in 8,065 builds at a quarter of the
token budget, so it is the cache path that opens this rather than concurrency alone. And the arm
served the context edition's weights: the counter reads only the scheduler's host-side arrays, so
what it measures is a function of flags and traffic, with the checkpoint entering only through the
size of the KV pool.

**Two conditions gate all of this, and the command below is missing one of them.** The changed code
runs only when a batch carries speculative tokens at all — the vendored file takes its speculative
branch only when `spec_sequence_masks is not None` — so with **no `--speculative-config`, as the
command below ships, the module is a no-op by construction**; and at `--max-num-seqs 1` no mixed
batch can form either. Add MTP to this recipe and the measurement above is what applies. The command
below ships `--max-num-seqs 4`, four streams rather than the eight that were measured, where the
same mechanism exists and its rate is unmeasured. When both conditions hold, mount it read-only:
`tools/vllm-qwen-gdn-spec-gates.py`
(`sha256 7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0`) over
`/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.
It is diff-identical to upstream's eight changed lines and `py_compile`-clean under the image's
Python 3.12.3, and it is an **overlay deliberately not part of the qualified digest**:
`sha256:16a936b877b90f…` is what was qualified, and a reachability count is not evidence that would
survive a re-qualification, so it is mounted over the vendored file rather than promoted into the
image.

**The effect on answers was not measured, and measuring it was declined on resolution grounds.**
Three events in 5,825 builds cannot move a statistic whose run-to-run floor is **0.0823** mean
absolute chosen-logprob error against a per-event effect of **0.002755** — a noise floor about
thirty times the size of one event — so an A/B would have returned its own noise and was
deliberately not run. Nothing here claims the overlay changes an answer, or that it does not. The
recommendation follows a rule fixed before the GPU window opened instead: a nonzero rate in a
regime we ship gets the free fix, because a silently miscomputed forward pass gives the operator no
signal at all. Traffic in that arm was adversarial by design — eight speculative streams plus
injected short prompts and full-cache-hit repeats — so 0.515 per thousand is an upper bound on a
shipped regime, not a forecast for your workload
([`receipts/gdn-spec-gate-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-spec-gate-defect.json)
is the source-level defect analysis).

**Scope on this build, stated narrowly.** What was measured on this profile is that the recipe
starts healthy on the promoted image with the cache on, answers a text and an image request
exactly, and reports the cache enabled in its banner. That is a serving smoke, not a gate
suite: this window has no long-needle, combined-image or decode-dispersion gate of its own, and
the 11.6× and 29.3× reuse figures above were measured on the context edition's much longer
prompts, not on 8,192-token ones. Expect the shape of the win — recomputing only what changed —
rather than those multiples.

Enabling it on this profile is one extra read-only mount and two flags:

```bash
set -euo pipefail
git clone https://github.com/malaiwah/qwen38-27b-exl3 && cd qwen38-27b-exl3
cat <<'SHA256' | sha256sum -c -
b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf  tools/vllm-mamba-align-scheduler.py
SHA256
SCHED=$PWD/tools/vllm-mamba-align-scheduler.py

docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \
  -v /models:/models:ro -v /cache:/cache \
  -v "$SCHED:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro" \
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
    --enable-prefix-caching --mamba-cache-mode align \
    --host 0.0.0.0 --port 8000
```

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

### KV-cache dtype: fp8 is the family's measured default

The recipe above leaves the KV cache at the engine default, as the Tradeoffs section notes; the
family's qualified long-context profiles pin **fp8**, and that default is now measured rather than
assumed. A five-arm sweep on the physical RTX 5090 served the **context edition** — same engine,
same flag surface — at its native-window profile with the KV dtype the only deliberate flag
change: **no arm beat fp8 on native-or-beyond context on 32 GB with retrieval intact**
([`receipts/kv-dtype-sweep-5090.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/kv-dtype-sweep-5090.json),
decision record
[`docs/38-kv-dtype-sweep.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/38-kv-dtype-sweep.md)).
The engine derives the attention backend from the KV dtype, so each arm is measured as it actually
serves:

| `--kv-cache-dtype` | backend (engine-chosen) | KV tokens at 262,144 | prefill, same 261,795-token prompt | top-1 / trunc. KL vs bf16-KV |
|---|---|---:|---:|---|
| **fp8** (family default) | FLASHINFER | 265,122 | **180.4 s** | 95.60 % / 0.001655 |
| int8_per_token_head | TRITON_ATTN | 272,453 | 544.3 s | 97.25 % / 0.000914 |
| fp8_per_token_head | TRITON_ATTN | 272,453 | 545.9 s | 98.84 % / 0.001284 |
| int4_per_token_head | TRITON_ATTN | **502,667** | 501.0 s | 94.29 % / 0.005948 |
| bfloat16 | FLASH_ATTN | 138,519, window capped at 131,072 | — | reference |

The per-token-head family is measured, not assumed: int8 and fp8 per-token-head each dominate fp8
on **both** capacity and closeness to the bfloat16-KV reference, but each pays **3.0× prefill**
because TRITON_ATTN is the only backend on this fork that accepts per-token-head scales — and the
capacity edge is TRITON_ATTN's smaller CUDA-graph pool (0.06 against 0.45 GiB), not cheaper bytes:
those arms cost *more* per token than fp8 (35,360 against 34,816 B/token). `int4_per_token_head`
is the real capacity lever — 502,667 tokens, 1.92× concurrency — at two named prices: **3.6× fp8's
distributional error** and **2.78× its prefill**. `nvfp4` and `nvfp4_ds_mla` do not start: no
attention backend on this fork advertises nvfp4 for a non-MLA decoder — all five candidates answer
`kv_cache_dtype not supported` — and the GLM-5.2-serves-nvfp4 precedent is the owner's claim about
a different model, unverified here. The fidelity column is a **bfloat16-KV-reference probe at a
98,304-token context** — truncated top-20 KL over 70–173 paired greedy positions, a lower bound;
it is **not the v5 KLD** and must never be differenced against any published KLD figure. Retrieval
was **44/44 exact** across the five arms, 4-bit included — retrieval is not fidelity, which is
exactly why the KL column exists.

### Reconstruct-scratch arena: +17,874 KV tokens, and it stays an overlay

**A 2-hunk fork patch to `exl3.py` buys 17,874 more KV tokens, measured, and it is an opt-in
overlay rather than part of any qualified digest.** The pinned r34 image keeps one persistent fp16
prefill-reconstruct scratch **per weight geometry** — 790 MiB across the eight geometries that
allocate at the qualified long-context profile (the head's 5120×32768 chunk needs ≥128 sampled
logit rows and never triggered). The patch (overlay `tools/vllm-exl3-scratch-arena.py`, sha256
`9aba06ebf60ca7665c0513752387c349240ab85e1ebc44d6ce8137ef157b6c15`; fork PR
[local-inference-lab/vllm#397](https://github.com/local-inference-lab/vllm/pull/397)) shares one
grow-to-max arena per device instead, sized by the largest live geometry (**170 MiB**), because
each reconstruct is written and consumed inside one eager call on one stream. The kernels see
identical operands — same shapes, strides and dtypes.

**Measured on the physical RTX 5090 A/B: engine-reported KV pool 265,122 → 282,996 tokens
(+17,874, +6.7 %, 9.28 → 9.88 GiB ≈ +0.60 GiB)**, reproduced identically across two server starts
per arm, the arena's own growth log ending at exactly the predicted 170 MiB
([`receipts/scratch-arena.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/scratch-arena.json)).
That A/B served the **context edition** at its qualified 262,144-token profile — same engine, same
flag surface — and the scratch geometries are shape-derived, so the mechanism and the 170 MiB arena
size are identical on this build while the pool figures above are the context edition's. Fidelity
was gated rather than assumed: the 30-case deterministic vision suite returned **byte-identical
answers on both arms** (24/30 each, equal to the rank-1 qualification reference), a full-window
needle (258,925 tokens, depth 0.5) retrieved exactly, and decode did not regress (109.2–109.7
against 108.5–108.9 tok/s over three warmed C1 runs). **Read the byte-identity claim narrowly** —
it covers that deterministic probe set, because the control shows two restarts of the *unpatched*
baseline differ on 7 of 8 long greedy continuations (`exl3_gemm` autotunes kernel configs by
measured time per process), so this stack is not restart-deterministic on long greedy text with or
without the patch, and every cross-restart pairing is 7 DIFF / 1 MATCH either way.

Like #51812, it is an **overlay deliberately not part of the qualified digest**: the pinned digest
is what was qualified, a larger KV pool is not evidence that would survive a re-qualification, so
it is mounted read-only over the vendored file rather than promoted into the image —
`-v tools/vllm-exl3-scratch-arena.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro`.
The static prediction had been +620 MiB / +18.7k tokens; the measured gain is **95.7 %** of it, and
the measured number is the one to quote. On the 24 GB class the same bytes put the published
24,576-token window at **42,450 raw token headroom, supporting 40,960 at the next window step —
arithmetic only**, pending a 24 GB-class boot
([`docs/34-vram-class-profiles.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/34-vram-class-profiles.md)
§10.2).

### Chat template

This repo ships `chat_template.jinja` byte-identical to `Qwen/Qwen3.8-27B` (sha256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`), and the same bytes again in
the `chat_template` key of `tokenizer_config.json`. Do not replace one without the other: under
transformers 5.15.0 the `.jinja` file takes priority, so editing only `tokenizer_config.json` is
a no-op. To override, pass `--chat-template <file>`.

Three upstream-template restrictions to code against. All three are Qwen's, unchanged by us, and
all three surface as HTTP 400:

- **`reasoning_effort` accepts only `xhigh` (default), `medium`, `low`, or `none`.** `high`,
  `minimal` and `max` are rejected even though vLLM's OpenAI surface advertises them. If your
  client hard-codes `high`, serve with `--default-chat-template-kwargs.reasoning_effort=xhigh`.
- **Exactly one `system` message, first.** Two `system` messages 400 with `System message must
  be at the beginning.` (vllm#41114, open; fix PR vllm#44505 closed unmerged). Merge them
  client-side. A `developer` role is safe - vLLM folds and consolidates it (vllm#43590).
- **Content blocks must be `text`, `image`, `image_url` or `video`.** Anything else, including
  an Anthropic `tool_result` carrying a `tool_reference` item on `/v1/messages`, 400s with
  `Unexpected item type in content.` (vllm#52489, open).

Echoing `reasoning_content` back on assistant history turns is what buys a full prefix-cache
hit: measured 100 % prefix reuse when the client returns it, 94.7 % at ten turns when it does
not. Community "fixed" Qwen templates are not recommended here: measured against
`--tool-call-parser qwen3_coder`, `qwen3.8-froggeric-v22` renders a tool call whose arguments
the parser recovers as `{}`. Details in
[`docs/39-chat-template-audit.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/39-chat-template-audit.md)
and [`receipts/chat-template-audit.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/chat-template-audit.json).


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
  <img alt="Two panels. Left: mean KL divergence from BF16 on a log scale against resident weight footprint in decimal GB, measured on the overlap-corrected 127-context v3 subset. Four candidates with bootstrap 95 % confidence bars: malaiwah/Qwen3.8-27B-EXL3-K5K6 (iteration 2) at 21.82 GB and 0.0079; Qwen/Qwen3.8-27B-FP8 at 30.61 GB and 0.0128; malaiwah/Qwen3.8-27B-K4 (iteration 1, this quant) at 19.21 GB and 0.0297; unsloth/Qwen3.8-27B-NVFP4 at 22.91 GB and 0.0927. A grey star marks the Qwen3.8-27B BF16 reference at 55.56 GB, KLD zero by definition. A grey series legended 'Qwen3.6-27B GGUF/FP8/NVFP4 (external, other protocol)' runs behind them, with UD-Q4_K_XL, Q8_0 and FP8 (vLLM) labelled: a different model generation measured by someone else under a different protocol, plotted as context only and not comparable with our points. A dashed vertical line in both panels marks the NVFP4-equivalent memory ceiling 21.9 GB. Right panel: top-1 agreement with BF16 against the same axis, 96.95 / 96.18 / 94.48 / 90.49 percent in that same candidate order, with a dashed line at BF16 = 100 percent. Title: Distribution fidelity vs memory — held-out corpus, 127 contexts, 259,969 positions. Those KLD and top-1 values are the overlap-corrected 127-context subset, which is why they differ from the full-suite 136-context table below this figure." src="assets/fidelity-vs-size-light.svg">
</picture>

*The figure's four points and four top-1 values are the overlap-corrected 127-context subset;
the table below is the original full 136-context receipt, which is why 94.48 / 96.18 / 90.49 %
there reads 94.50 / 96.22 / 90.53 % here.*

**136 analysis contexts x 2047 positions = 278,392 scored positions** from separately
sourced Gutenberg, arXiv, Wikipedia and CPython documents. The original fixed-stride
160-character scan reported zero calibration hits; a later all-position 12-token scan found
exact overlap in 2/41 source documents. Exact full-vocabulary two-pass
`KL(BF16 reference || candidate)` through one shared BF16 LM head, float32 within each
vocabulary chunk and float64 across chunks, source-cluster bootstrap.

| candidate | resident weights | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.61 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 0.773 | 0.004528 | 96.22 % |
| **this quant** | **19.21 GB** | **0.030736** | [0.02238, 0.04073] | 0.004218 | 1.758 | 0.010051 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 22.91 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 4.509 | 0.028663 | 90.53 % |

*The weight column prints the measured resident weights of
[`docs/22-results-iteration-2.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/22-results-iteration-2.md)
— 28.51 GiB = 30.61 GB for FP8, 17.89 GiB = 19.21 GB here, 21.34 GiB = 22.91 GB for unsloth
NVFP4. Earlier revisions of this card printed 30.9 / 19.2 / 23.4 GB in this column, the older
docs/18 presentation of the same three measurements. The KLD, interval, median, p99.9, JSD and
top-1 columns are unchanged.*

**Overlap-corrected subset:** conservatively removing all nine analysis contexts from either
affected source document gives K4 **0.029679**, FP8 **0.012798**, and NVFP4 **0.092727**
over 127 contexts. The ranking and every conclusion survive; the table above is retained as
the original full-suite receipt, not described as contamination-free.

**Never quote the v3 NVFP4 number as our current one.** NVFP4 reads **0.092727** on this corrected
v3 subset and **0.030115** on v5 shard 0 — same checkpoint, same revision, same flags, same
shared-head protocol. That gap is suite hardness, and it was measured for all six candidates rather
than argued: v3-corrected ÷ v5 shard 0 is **2.4625x** official FP8, **2.5293x** online K5/K6,
**2.6564x** hydrated, **2.7505x** the context edition, **2.8688x** this build and **3.0791x**
NVFP4 — a band spanning **1.2504x** end to end, with the **ordering identical in both suites**
([`receipts/nvfp4-v5-measurement.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/nvfp4-v5-measurement.json),
block `suite_comparability_v3_vs_v5`). A band of factors and not one factor is exactly why no
conversion between the suites exists: the ordering carries across, an absolute value never does, and
a v3 number must never appear in the same sentence as a v5 number.

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
**Replay qualification is the weak link at 5.83e-04** mean, re-derived inside the v5 suite (32
contexts, 65,504 positions, [5.15e-04, 6.64e-04], superseding the six-context v3 figure of 6.54e-04),
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
| `unsloth/Qwen3.8-27B-NVFP4` control, same generation | **0.091457** | 0.000000 | 0.8036 | 22.91 GB |

**This quant is 2.7x closer to the BF16 teacher than the same-generation NVFP4
checkpoint, while holding 3.70 GB less resident weight.** That is the whole point of the
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

### Concurrent serving: speculative depth is a concurrency-dependent choice

Measured on the **user's own physical GeForce RTX 5090** (32,607 MiB, driver 610.57.04) on the
immutable production image `localhost/vllm:gg-r34-patched`
(`sha256:6eca4c69…`) with no source bind mounts, on the **context edition** rather than this
checkpoint — 11 configurations, identical frozen token-id prompts, three warmed repeats
([`receipts/perf-sweep-5090.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/perf-sweep-5090.json),
decision record
[`docs/36-performance-levers-5090.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/36-performance-levers-5090.md)).
**No figure in this subsection may be differenced against the rental RTX PRO 6000 table above**,
per the receipt's own rule; it is carried here because the finding is a serving recommendation that
applies to every build in the family.

The decision metric is **accepted tokens per step ÷ step time**, never acceptance rate. The
reference row, MTP depth 3 at `--max-num-seqs 8`, reads **82.94 / 263.12 / 313.28** tok/s aggregate
at 1 / 4 / 8 concurrent streams with step times 25.72 / 30.64 / 49.52 ms. At one
stream, MTP depth 3 wins: 2.1429 accepted tokens per step over a 25.72 ms step = **83.31**
per-request tok/s, against depth 1's 1.6558 over 22.29 ms = 74.28. At **eight** streams the trade
reverses decisively: `num_speculative_tokens=1` drops accepted tokens per step by **22.98 %**
(2.1753 → 1.6754) while step time falls **37.25 %** (49.52 → 31.07 ms), so aggregate throughput
rises **30.67 %, 313.28 → 409.35 tok/s** (+22.75 % per request), and it holds **10,911 more KV
tokens** (283,481 against 272,570) — the only row whose needle ran at the full **261,794** tokens,
retrieved exactly, with the 30-case image suite unchanged at 24/30. Depth 2 is dominated at both
ends. So: keep depth 3 for interactive single-stream use, set depth 1 when the deployment really
runs concurrent streams, and note that at concurrency 4 the choice does not matter (±2.6 %). One
constraint travels with the 8-stream matrix: at 262,144 tokens it only starts at
`--gpu-memory-utilization 0.97`, which is a **text-only** profile because a large image OOMs in the
vision tower there, so a vision-capable deployment stays at one sequence.
The crossover exists because each drafted token costs a fixed slice of step time whose acceptance
does not improve with batch size, while a wider batch already fills the step: past some concurrency
the cheaper step buys more than the deeper draft.

**What that means for the recipe above**, which does not enable speculative decoding: if you add
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`, choose `N` = 3 for
interactive single-stream serving and `N` = 1 once the deployment really runs eight concurrent
streams. That is the only lever in the sweep worth changing.

**Closed avenues, so nobody re-runs them.** `--attention-backend FLASHINFER` is a **no-op**: the
engine already auto-selects FlashInfer on SM120 with fp8 KV and head_size 256. Forcing
`TRITON_ATTN`, which the live K4 service does, is up to **5.5 % worse** on step time at eight
streams and its apparent +5.5 % gain at temperature 0 becomes **−7.3 % at temperature 0.6**, so it
is acceptance noise rather than throughput. `custom_ops:["all"]` is **2.2-5.2 % worse** on step time
and is **not bit-exact**. Both dynamic speculative-decoding knobs are structurally unusable here:
either one downgrades `cudagraph_mode` from `FULL_DECODE_ONLY` to `PIECEWISE`, which `Exl3Config`
refuses, so the server does not start — and forced eager, the only form that runs, loses 48 % of
decode. **Dynamic speculative depth is no longer closed — it was a fixable bug, now fixed and
measured (2026-08-16).** The `cudaErrorIllegalAddress` that made `VLLM_USE_V2_MODEL_RUNNER=1`
unusable was a FlashInfer gate that admitted persistent (CUDA-graph) decode wrappers only for
`q_len == 1 + num_speculative_tokens`, while the speculator's draft steps run `q_len == 1`; the
draft-decode graph was therefore captured *and* replayed on the **dynamic** wrapper, whose plan
buffers move on every call. Keying wrappers by the shape capture actually planned fixes it
([local-inference-lab/vllm#398](https://github.com/local-inference-lab/vllm/pull/398), closes
[#396](https://github.com/local-inference-lab/vllm/issues/396); root cause proven with an
instrumented capture-versus-replay address log plus a control that changes nothing except pinning
the capture-time buffers alive). With the fix, one server running the per-batch-size schedule
(depth 3 at batch 1-2, depth 1 at batch 3-8) measures **87.4 tok/s at C1 and 416.3 tok/s at C8**,
**+38.0 % aggregate decode at C8** against an MRV1 baseline measured in the same window
(301.6 tok/s) with C1 held — depth-1 throughput at eight streams without giving up depth-3 latency
at one. Greedy outputs repeat token-for-token within an engine process in all four arms, and
acceptance does not drop, so the speed is not bought with looser verification. **The constraint
that decides whether you want it:** at `--gpu-memory-utilization 0.97` the V2 runner leaves 58.56 MiB
free and the EXL3 prefill reconstruct OOMs on the first 2,048-token prefill (both arms, schedule off
and on), so the win was measured at **131,072 context and 0.95 utilisation — 234,256 KV tokens
against 272,570 on this card's published profile, −14.1 %**. Serving the published 262,144/0.97
profile under the V2 runner on a 32 GB card is **not demonstrated**. Cross-restart bit-exactness
remains unclaimable on this stack (`exl3_gemm` autotune is per process). The static depths, 3
single-stream and 1 at eight streams, remain the published recipe until that KV concession is
either accepted or removed
([`receipts/v2-runner-depth-schedule.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/v2-runner-depth-schedule.json)).
**Prefill did not move on any lever** (3,374.4 tok/s at 2,048 and 3,255.4 at 6,144 prompt
tokens for the reference row, no graph-decode arm more than 4.0 % away), so the prefill deficit is
structural rather than untuned.

One reconciliation, because both numbers are published: the sweep's 82.94 tok/s at one stream sits
below the context edition's qualification median of 107.56 tok/s purely because of **acceptance**,
not speed — 2.14 accepted tokens per step here against 2.69 there, since these frozen prompts are
literary prose — while step time agrees to 2.6 % (25.72 ms against 25.05 implied). The two
measurements are consistent.

## Tradeoffs, stated plainly

- **The download is 28.31 GB for a 19.21 GB resident model.** Attention ships
  BF16 so the runtime can encode it at K6 (and, later, at another width) instead
  of being locked to a serialized choice. If you want the download to equal the
  resident weight figure, the `v-serialized-k6` variant is the one to ask for.
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

**What that command reproduces, and what it does not.** It reproduces the recipe — composition,
widths and byte budget — and not the checkpoint. The published bytes are the artifact. A fresh
conversion of the hydrated sibling's recipe, run on the same hardware with the same flags and
source, returned **13 of 16 pinned payload files identical** (every config, the tokenizer, the
index and both quantization descriptors) with byte-identical shard headers, the same tensor names,
dtypes, shapes and offsets, and the same per-role byte totals and assigned widths — while
**399 of the 409 quantized modules (97.6 %) differed inside their `.trellis` payloads, at 41-92 %
of the bytes each (mean 82 %)**. No scale, norm, embedding, vision or BF16 companion tensor moved.
The converter is nondeterministic, measured rather than inferred: two runs of one conversion,
minutes apart, agreed on every width and every global scale and disagreed on the converter's own
`proxy_err`
([`receipts/converter-determinism.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/converter-determinism.json)).
So what you get from the commands above is a **sibling**: a different valid artifact of the same
recipe, not a broken one. Every fidelity number on this card measures the published bytes a
downloader receives and is unaffected. And "rebuild this and you get these numbers" is no longer
an untested expectation: a third conversion of that hydrated recipe — a sibling again, differing
in 399 `.trellis` payloads and nothing else — was captured and replayed on the identical v5
shard-0 protocol (512 contexts, 1,048,064 scored positions, same shared head, same comparator),
and paired against the published checkpoint the difference is **−3.755e-06, 95 % source-cluster
bootstrap interval [−2.854e-05, +2.062e-05]**, which **brackets zero**, on **257 contexts to 255
with no ties**. Two controls make that attributable to the sibling's weights rather than to the
harness — replaying the published checkpoint against the same reference capture returned its mean
and all 512 per-context rows bitwise, and a fresh recapture reproduces that mean exactly — so the
comparison's floor is zero rather than a tolerance. **97.6 % of the quantized modules come back
with different bytes and the fidelity is the same to within our resolution**, which is what makes
"the recipe is the reproducible thing, the bytes are the artifact" a measured claim rather than a
hedge. It is one sibling, one recipe, one shard, at this protocol's resolution: it bounds the
converter's fidelity variance, it does not estimate it, and it is not a finding that converter
nondeterminism is fidelity-neutral in general
([`receipts/sibling-rebuild-fidelity.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/sibling-rebuild-fidelity.json)).
Check the digest of the published tree against `SHA256SUMS`
rather than against your own conversion, and read a byte diff below this floor as the converter
rather than as tampering, corruption or a changed recipe. The same run also showed the recorded
build environment is incomplete: the pinned image has no `marisa_trie`, which the conversion
imports on an unconditional path, so that image provably could not have finished the job. The
conversion-capable image is `gg-r34-convert`. This is the converter's sense of
reconstruction only — the runtime's `reconstruct_had_slice` / `reconstruct_fp8_slice` path, which
turns stored trellis bytes back into weights at load, is a different claim and is untouched here.

## Reproduce this

This section is about **the numbers, not the bytes**: a fresh conversion of the recipe produces a
sibling rather than this checkpoint, as recorded under
[Reproducing this quant](#reproducing-this-quant).

Everything the v5 numbers above were computed from is published as a dataset:
[`malaiwah/qwen38-27b-fidelity-suite-v5`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5)
— **5,835 files, 10,826,796,868 B (10.83 GB / 10.083 GiB)**, verified at revision `08bde6cc`. It
contains the 5,120 token-id files that are the authoritative evaluation input (retokenizing the
source text does not reproduce them), the parent suite manifest whose sha256 equals the ladder pin,
the ladder pin itself, all ten 512-context shard views with their capture and replay command lines,
the corpus fetch log, **the shard-0 unquantized BF16 hidden-state reference** (512 captures plus
manifest, 10.73 GB) and **79 per-shard reports** — 50 ladder, 10 tail, 15 scored-window, 4
cross-engine. Receipt
[`receipts/preserved-artifacts.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/preserved-artifacts.json);
5,326 of the 5,835 files were re-downloaded and re-hashed end to end, and the 10 GB hidden-state
tree was checked against the Hub's own LFS digests plus a three-file CDN spot check.

**What it costs to replay.** Because the suite *and* the shard-0 BF16 reference are both published,
a third party can score a **new** candidate against the identical contexts without recapturing the
reference: one candidate capture plus one replay, about 6 minutes of GPU for shard 0 on a single
RTX PRO 6000 Blackwell, instead of two model loads. That is exactly how the NVFP4 row above was
produced. Re-running the whole ten-shard ladder is a different bill — about 5 hours of GPU for the
fifty candidate captures, plus about 54 minutes for the nine BF16 shard references that were
deleted once their reports verified.

**Three archival mirrors keep the third-party citations resolvable.**
[`malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da`](https://huggingface.co/malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da)
is a **recovery** mirror: upstream super-squashed its history on 2026-08-15 and the Hub now answers
`Invalid rev id` for `9c73e2da…`, the revision every NVFP4 number on this card was measured
against, so the reviewed revision is otherwise unreachable.
[`malaiwah/Qwen3.8-27B-GGUF-archival-f1bfb127`](https://huggingface.co/malaiwah/Qwen3.8-27B-GGUF-archival-f1bfb127)
is **precautionary**: the five files the cross-engine table cites, at a revision that still resolves
upstream. [`malaiwah/Qwen3.8-27B-NVFP4-RTX5090-archival-69274a0d`](https://huggingface.co/malaiwah/Qwen3.8-27B-NVFP4-RTX5090-archival-69274a0d)
is likewise **precautionary**: the gittensor checkpoint measured above, at a revision that still
resolves upstream, deep-verified after upload — its 19.2 GB of weights cost essentially zero
transfer because the Hub already held every chunk. Said plainly, a mirror preserves the **citation** — a resolvable repo id, revision and
digest table that survive an upstream squash or delete — and is **not** independent byte-level
redundancy. Hub storage is content-addressed, so our copy and upstream's plausibly reference the
same underlying chunks; nobody should assume physical copies we do not hold. The measured cost of
the first two mirrors was 2.34 GB of transfer for 149.3 GB of content, about 1.6 %, which is that
content-addressing showing through.

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
