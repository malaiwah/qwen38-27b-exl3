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
  - research-artifact
  - negative-result
  - sub-4-bit
  - gilded-gnosis
---

# Research artifact, not a release: the first sub-4-bit build in this family, measured and rejected

**This build was measured and rejected, and here is by how much.** On shard 0 of the v5 held-out
suite — 512 contexts, 1,048,064 scored positions, 330 source clusters, one shared BF16 head, one
shared BF16 reference — it measures **0.045374 mean KL divergence, 95 % CI
[0.041959, 0.049351]**, with top-1 agreement of **91.73 %**. The decision rule was pre-registered
before the conversion ran and keyed a **NO** to any mean above **0.030**: this is **1.5x that
threshold**. Paired per context it loses **512 of 512 contexts to
[K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4)** — the worst previously published candidate —
at **+0.035028 [+0.032382, +0.038117]**, i.e. **4.39x** its 0.010345, and **512 of 512 to the
[context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context)** at **+0.041964
[+0.038790, +0.045661]**, i.e. **13.31x** its 0.003409. All five strata lose to both comparators
with every interval excluding zero. **Do not deploy this checkpoint.** It is published so that the
rejection is auditable, because a NO nobody can check is an assertion.

If you want a build that runs, take
[`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)
(the recommended build), the [context edition](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context)
for long windows, or [K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) if you need the smallest
*published* one.

## Who should download this

- **Anyone auditing our NO.** The 16 GB verdict in
  [docs/34](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/34-vram-class-profiles.md)
  rests on this number. With these bytes you can re-capture and re-replay it against the published
  shard-0 BF16 reference and check the interval, the tail row and the paired losses yourself,
  against the pre-registration that was committed and pushed before the conversion started.
- **Anyone extending the per-bit error law below 4 bits.** This is the only sub-4-bit artifact in
  this family: 336 modules realised at K3, with the converter's own proxy error at that width, and
  the first evidence that the 3.73x-per-bit law holds one rung below where it was fitted
  (§ [the law below its fit](#the-373x-per-bit-law-one-rung-below-where-it-was-fitted)).
- **Not anyone looking for a serving recipe.** This is not a small-VRAM build you should run. It is
  a rejected candidate, its fidelity is 4.4x worse than the worst thing we publish, and its tail is
  worse still. There is no configuration of it that we recommend, and none will be offered.

## What was built

The docs/34 §6.1 primary 16 GB candidate — S16-V — converted exactly as specified:

| role | modules | width | serialized bytes |
|---|---:|---|---:|
| `full_attention` q/k/v/o | 64 | **K4** `mcg` | 0.840 GB |
| `linear_attention` | 144 | **K3** `mcg` | 2.131 GB |
| `mlp_gate_proj` | 64 | **K3** `mcg` | 2.142 GB |
| `mlp_up_proj` | 64 | **K3** `mcg` | 2.142 GB |
| `mlp_down_proj` | 64 | **K3** `mcg` | 2.142 GB |
| `lm_head` | 1 | **K4** `mcg` | 0.636 GB |
| `mtp_draft` | 8 | **K4** `mcg` | 0.213 GB |
| `embed_tokens` | — | BF16 on disk | 2.543 GB |
| `vision_tower` | — | BF16 (`-vb 16`) | 0.921 GB |
| norms and small | — | BF16 | 0.001 GB |

409 modules quantized, 400 of them body modules with a realised proxy error. Against the nearest
measured neighbour, K4, this build is **one bit lower on each MLP projection, three lower on the
linear-attention stack** (online K6 → K3), **two lower on full attention** (online K6 → K4) and
**two lower on the head** (K6 → K4) — every one of them in the direction that had already made K4
the worst published candidate.

Realised per-role proxy error, median over the modules in the role, from the conversion log:
`full_attn` (K4) 0.000724, `linear_attn` (K3) 0.003008, `mlp_gate` (K3) 0.004563, `mlp_up` (K3)
0.006575, `mlp_down` (K3) 0.009271, `lm_head` (K4) 0.000816. The eight MTP draft modules quantize
through the uncalibrated fallback and report `rmse` (0.00238-0.00562) rather than this proxy.

## The bytes, predicted to the byte

`docs/34` §6.2 published a serialized payload prediction for this recipe **before it was built**:
**13,711,503,428 B**. The measured payload is **13,711,503,428 B** — **prediction error zero**, and
the first test of the affine byte law below 4 bits. Whole tree **as built**: **13,735,527,028 B over 21 files**
(12.7922 GiB), against the `payload + 24.0 MB` disk rule's 13,735,474,746 B, i.e. **52,282 B** of
tree-metadata slack on the disk figure and none on the payload. Every per-file digest is recorded in
[`receipts/sixteen-flip-kld.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/sixteen-flip-kld.json)
→ `build.tree` and in this repo's `SHA256SUMS`. The **published** repository carries **23 files / 13,735,576,299 B**: publishing adds `README.md` and `DOCS-SHA256SUMS`, which is the whole **+49,271 B** difference. The build figure is the one the byte law is tested against, so both are stated rather than one silently standing for the other.

## The loader term, measured for the first time below 4 bits

vLLM's own `gpu_model_runner` reported **12.81 GiB** of model-loading weights against this build's
**12.7698 GiB** payload, in 1.605 s: a loader delta of **+0.0402 GiB**, with the BF16 embedding
resident (the fidelity protocol runs no int8 overlay). **This is the first loader term ever measured
for a sub-4-bit build.** Beside the existing observations — +0.206 GiB on hydrated at 20.10 GiB,
+0.305 to +0.342 GiB on the context build, +0.1067 GiB on K6-parity at 21.45 GiB — the term clearly
**scales with payload rather than being flat**, so the +0.35 GiB planning allowance over-charges the
16 GB class by roughly 0.31 GiB, about a third of that class's entire KV pool.

## The 16 GB class fails on fidelity, not on bytes

With the loader term measured instead of predicted, the int8-embedding resident-weight figure for
this build is **11.626 GiB** (payload 12,440,105,028 B = 11.5858 GiB, plus the measured 0.0402)
against budgets of **12.70 GiB** (docs/34 §3 basis) and **12.49 GiB** (the qualified 0.955
utilisation basis) — **1.07 and 0.86 GiB of headroom. The weights fit, comfortably, on both bases.**

**So the 16 GB no-go does not rest on bytes; it rests on measured fidelity, and this card is that
measurement.** That is a harder and cleaner finding than the budget argument it replaces: the class
is not out of reach because the artifact is too large, it is out of reach because the artifact that
fits is 4.4x worse than the worst build we publish. What this does *not* settle: it is still a 32 GB
board. The activation, non-torch and CUDA-graph terms in the 16 GB budget are carried from a 32 GB
profile (docs/34 flip item 2), and no 16 GB board has been started in this project. Items (3) needle
retrieval and text-plus-image at the 4.2 MP cap on a K3 body, and (4) the weighted non-termination
check, are equally untouched.

## The measurement

Shard 0 of the v5 suite, 512 contexts, 1,048,064 scored positions, suite token digest
`caef8a4628d6c07c162100895096f890cdf9cafc8e4c48b3d66035d737ee7cf7`. Hidden state captured after the
final norm and replayed through the shared BF16 `lm_head`
(`25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff`) against the BF16 reference
capture. `token_mean_kld`, cluster bootstrap over 330 source clusters, 10,000 resamples.

| build | mean KLD | median | p95 | p99 | p99.9 | exact max | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| context edition | 0.003409 | 0.001351* | 0.010691* | 0.035699* | 0.164195* | 3.749189 | 97.55 % |
| K4 | 0.010345 | 0.003204* | 0.033157* | 0.119353* | 0.555521* | 7.564510 | 95.91 % |
| **S16-V (this build)** | **0.045374** | 0.013875 | 0.151928 | 0.543578 | **2.370397** | **12.503125** | **91.73 %** |

\* the two comparators' percentiles are the histogram-bounded estimates in
`receipts/kld5-1M-tail-{k4,ctx}.json` — exact to a bin width of 5.6 % — because no shard report
carries the token-level vector an exact cumulative percentile would need. This build's row is exact,
from its own shard-0 report; K4's exact shard-0 p99.9 is 0.5576.

**The tail degrades faster than the mean** — p99.9 of 2.3704 against K4's exact 0.5576 is 4.25x on a
mean ratio of 4.39x, and the maximum is 12.50 against K4's 7.56 — which is the direction docs/34
§6.4 predicted and the reason the tail row was made part of the pre-registered report rather than an
afterthought.

Paired per context, difference reported as comparator minus this build (positive means this build
carries more KLD):

| comparator | paired difference | 95 % CI | contexts lost | ratio |
|---|---:|---|---:|---:|
| K4 | **+0.035028** | [+0.032382, +0.038117] | **512 / 512** | 4.39x |
| context edition | **+0.041964** | [+0.038790, +0.045661] | **512 / 512** | 13.31x |

Per stratum, against the context edition: code **+0.0505**, literary **+0.0507**, multilingual
+0.0424, encyclopedic +0.0346, scientific +0.0313. Against K4: literary +0.0430, code +0.0418,
multilingual +0.0352, encyclopedic +0.0287, scientific +0.0265. Every interval excludes zero and no
context in any stratum goes the other way.

**One control closes the obvious escape.** These reports are pairable against the published shard-0
reports because the reference capture was checked, not assumed: re-scoring the published hydrated
checkpoint against this window's surviving BF16 reference reproduced the published mean **bitwise**
(0.002699883159684943) with a paired difference of exactly 0.0 on all 512 contexts
([`receipts/sibling-rebuild-fidelity.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/sibling-rebuild-fidelity.json)
→ `capture_noise_floor`). The comparison's floor is zero, not a tolerance.

## The pre-registration held, including a miss worth printing

The prediction, the thresholds and the decision rule were committed and pushed in
[`receipts/preregistration-kld9-window.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/preregistration-kld9-window.json)
(`0fa601b4fbc99bf9bb2d2a06f5941f52f8034a4d3858930d9fe3132a546f7c61`, commit `edced9a`,
2026-08-16T15:05:21Z) **before this conversion ran**, so the verdict cannot have been chosen after
the number was known.

| registered | value | contains the measured 0.045374? |
|---|---|---|
| docs/34 §6.4 range | [0.030, 0.100] | yes |
| independent `sqrt_energy` surrogate bracket | [0.0318, 0.0585] | yes |
| primary point estimate from the byte law | **0.0689** | **1.52x too high** |

The build is *better* than the byte-law point estimate and still nowhere near good enough. The
direction and the size of that miss are printed here rather than reframed: the uniform-bpw form
behind 0.0689 assumes error spreads in proportion to bytes, while this recipe moves five role groups
at once, three of them by three bits.

## The 3.73x-per-bit law, one rung below where it was fitted

The published ladder
([`receipts/error-driven-ladder.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/error-driven-ladder.json))
has **no K3 rung for any module below 52M parameters**, so the pre-registration extrapolated those
96 modules as `eps(3) = eps(4) · eps(4)/eps(5)`. Comparing the converter's *realised* proxy error at
the width it actually used against the laddered value for the same module at the same width:

| modules | rung | ratio realised / laddered (median) | min | max |
|---|---|---:|---:|---:|
| 304 | measured rung exists | 0.9957 | 0.9731 | 1.0093 |
| 96 | K3 extrapolated by the law | **1.0164** | 0.9915 | **1.0455** |

So the per-bit law holds one rung below where it was fitted, to within about **4.5 %** on the
extrapolated modules. That is the reusable part of this artifact, and it is the reason to keep the
bytes rather than only the receipt.

## Reproduce this

```bash
# 1. convert (~41 min on one RTX PRO 6000 Blackwell Server Edition)
export EXL3_BITS_FIXED='{"^.*self_attn\..*$": 4, "^.*linear_attn\..*$": 3}'
export EXL3_BITS_OVERRIDE='{"^model\.language_model\.layers\.[0-9]+\.mlp\.(gate|up|down)_proj$": 3}'
python convert.py -i /models/Qwen3.8-27B -o out/qwen38-s16v -w wd-s16v \
    -b 3 -hb 4 -mb 4 -vb 16 -cb mcg -cpi 1000000

# 2. finalize: index, quant config, verified topology, per-role bytes
python util/add_safetensors_index.py -m out/qwen38-s16v --force
python util/add_quant_config.py -m out/qwen38-s16v
tools/finalize_checkpoint.py -m out/qwen38-s16v --upstream /models/Qwen3.8-27B

# 3. measure against the published shard-0 BF16 reference (~6 min GPU to capture, ~3 to replay)
tools/fidelity.py capture --model out/qwen38-s16v --suite <suite>/shard-0000/suite \
    --out hidden/s16v --gpu-memory-utilization 0.85 --quantization exl3 --quantization-config "$QCFG"
tools/fidelity.py replay --reference <reference>/hidden-bf16 --candidate hidden/s16v \
    --head lm_head.safetensors --suite <suite>/shard-0000/suite --out reports/report-s16v.json
tools/fidelity.py paired --a reports/report-k4.json --a-label k4 \
    --b reports/report-s16v.json --b-label s16v --out reports/paired-k4-vs-s16v.json
```

The MLP override names the language-model body explicitly rather than the loose `^.*mlp\.` form,
because `EXL3_BITS_OVERRIDE` is matched against every key in `f_targets` and the MTP draft layer's
three MLP projections are in there; the loose form would carry the draft to K3 as well and land
33,423,360 B below the published byte prediction for a recipe reason rather than a byte-law one.
`-cpi 1000000` disables periodic checkpointing: a first attempt with the default 120 s interval died
at layer 4 with `OSError: [Errno 38] Function not implemented: '/'` from `os.makedirs` inside the
checkpoint save, because under `proot` a transient `stat` failure made `makedirs` recurse to `/` and
`mkdir('/')` returned `ENOSYS` instead of `EEXIST`. Checkpointing only writes and reads resume state,
so with no interruption the quantization path is unchanged; the cost is that a crash restarts from
scratch. Working directory peak: 12,790,298,239 B.

Converter: [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
`@5f3c537ca9d89893d771256f5c43c93656553fbb` (**1.4.2**) with this project's worktree diff
`578066cd…`. Hardware: 1x RTX PRO 6000 Blackwell Server Edition, SM120, driver 595.58.03 — **no
number on this card is comparable to one from a different card.**

**What that reproduces is the recipe, not this checkpoint.** The EXL3 converter is nondeterministic
in its `.trellis` payloads at identical widths, measured and quantified in
[`receipts/converter-determinism.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/converter-determinism.json)
and bounded in fidelity terms in
[`receipts/sibling-rebuild-fidelity.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/sibling-rebuild-fidelity.json).
Verify against this repo's `SHA256SUMS`, not against your own conversion, and read a byte difference
below that floor as the converter rather than as tampering.

## Files in this repository

- the checkpoint: two safetensors shards, index, configs, tokenizer, chat template
- `SHA256SUMS` — the immutable payload; `DOCS-SHA256SUMS` — every other file in the tree (this card,
  `.gitattributes`, `crc32.txt` and the `flip/` artifacts), so a card edit cannot invalidate the
  build hashes
- `quantization_manifest.json` and `build-receipt.json` — authoritative for composition, including
  the per-role byte totals quoted above; `config.json → quantization_config` keeps one
  `bits`/`codebook` pair for loader compatibility and **cannot** describe this mixed checkpoint
- `flip/build_s16v.sh` and `flip/convert-s16v.log` — the build script and the full 1,053-line
  conversion log, including the realised bpw and proxy error of every module at K3. Both are
  digested in the receipt (`toolchain.build_script`, `toolchain.convert_log`) and neither survives
  anywhere else: the rental box they were produced on is being decommissioned

## Evidence

- [`receipts/sixteen-flip-kld.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/sixteen-flip-kld.json)
  — the verdict, the pre-registration read back out of its own file, the score and tail row, both
  paired reports and all five strata, the realised width and proxy error of every module, the byte
  accounting, the measured loader term, the 16 GB arithmetic, and the per-file digests of this tree
- [`receipts/preregistration-kld9-window.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/preregistration-kld9-window.json)
  — the prediction, thresholds and decision rule, committed before the conversion
- [`docs/34-vram-class-profiles.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/34-vram-class-profiles.md)
  — §6 is the 16 GB design study this build was made to test, and §6.4 now carries the measured
  answer
- [`receipts/error-driven-ladder.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/error-driven-ladder.json)
  — the five-rung per-module error ladder this build extends one rung downwards
- [`malaiwah/qwen38-27b-fidelity-suite-v5`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5)
  — the suite, the shard-0 BF16 reference and the per-shard reports

## What is not verified

No capability, task-retention, throughput, context-capacity, vision or safety evaluation was run on
this build, and none will be: it is a rejected candidate kept for its measurement. The fidelity
evidence is shard 0 only (512 contexts), not the ten-shard 10,480,640-position suite the released
builds carry, and its 512-context hidden-state capture was released after replay, so re-checking the
number from these weights means re-capturing it. It has never been served through the runtime for
anything but hidden-state capture, and the 16 GB claims on this card are weight arithmetic on a
32 GB board, not a boot on a 16 GB one.

Like every EXL3 build in this family it requires the Gilded Gnosis vLLM fork with explicit
`--quantization exl3` and an exact `ignore` list; it does **not** load in upstream vLLM, SGLang,
TensorRT-LLM, llama.cpp, transformers or stock exllamav3.

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — the EXL3 Trellis format,
  LDLQ calibration, the MCG codebook, the conversion pipeline, and the `proxy_err` this build's
  ladder work is built on.
- [Qwen](https://huggingface.co/Qwen) — the base model.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path the capture ran through.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
