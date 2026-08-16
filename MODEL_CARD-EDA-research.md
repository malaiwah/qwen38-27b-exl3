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
  - gilded-gnosis
---

# Research artifact, not a release: an error-driven EXL3 bit allocation that was measured and lost

**This checkpoint is the output of an error-driven allocation experiment and is _not_ a release.**
Do not deploy it. The recommended build of this family is
[`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated),
which this candidate was built to beat and did not.

**The outcome, plainly.** Against the hand-designed hydrated recipe, on shard 0 of the v5 held-out
suite — 512 contexts, 1,048,064 scored positions, one shared BF16 head, one shared BF16 reference —
this build is **worse by +0.000366 mean KLD, 95 % CI [+0.000334, +0.000398]**, paired per context
over 330 source clusters. The interval excludes zero. Hydrated wins **470 of the 512 contexts**;
this build wins 42. Both checkpoints serialize to **21,586,964,548 bytes — identical to the byte**,
so there is no size excuse: this is a pure reallocation of the same bit budget, and it went
backwards.

## Why it is published anyway

Three reasons, in order of how much they are worth to someone else:

1. **The mechanism is the result.** The allocator did exactly what it was asked to do. Total
   equal-weighted relative proxy error over the 400 body modules fell **13.1 %** (0.075355 →
   0.065486) while mean KLD **rose 13.6 %**. That is a clean, paired, interval-bounded
   demonstration that `sum_m eps(m,K)` — exllamav3's own Hessian-weighted relative quantization
   error, summed with equal weights — **is not a monotone surrogate for KLD**. Minimising it moves
   fidelity the wrong way.
2. **The objective was disproven, not merely doubted.** A four-pair calibration disqualified two of
   four candidate weightings on *sign*, and left the best survivor a factor of 2.5 uncertain in
   magnitude — not good enough to authorise a 1e-4 fidelity claim. See
   [the calibration table](#the-calibration-that-kills-the-objective).
3. **The five-rung ladder is reusable.** The measurement pass behind this build produced a
   per-module error curve at five widths for all 409 quantized modules, and from it a law
   (§ [the 3.73x-per-bit law](#the-373x-per-bit-law)) and an exact closed-form allocation rule that
   between them let *anyone* quantizing this family skip the two-hour measurement entirely and
   allocate from an ordinary conversion log. That part stands whether or not this checkpoint does.

Publishing the loser is also what makes the winner's claim checkable. The hydrated recipe now
stands not because nothing better was tried, but because the best available alternative was built,
measured against it under a shared reference, and lost by a stated interval.

## What was built

| | hydrated (the incumbent) | **this build** |
|---|---|---|
| widths | hand-designed role split | solved per module by dynamic programming |
| `full_attention` q/k/v/o (64) | K6 × 64 | K4 × 5, K5 × 13, K6 × 17, K7 × 29 |
| `linear_attention` (144) | K6 × 144 | K5 × 61, K6 × 70, K7 × 13 |
| `mlp_gate_proj` (64) | K5 × 64 | K4 × 1, K5 × 51, K6 × 12 |
| `mlp_up_proj` (64) | K5 × 64 | K4 × 1, K5 × 27, K6 × 36 |
| `mlp_down_proj` (64) | K6 × 64 | K4 × 1, K5 × 3, K6 × 60 |
| `lm_head` | K6, `mcg` | K6, `mcg` (unchanged) |
| MTP draft, `embed_tokens`, vision, norms | quantized / BF16 / BF16 / BF16 | unchanged |
| body modules moved | — | **175 of 400** |
| average body width | 5.531 bits | 5.531 bits |
| **serialized payload** | **21,586,964,548 B** | **21,586,964,548 B** |

Everything else is held identical to the hydrated build: same source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, same converter, same `-b 4 -hb 6 -mb 4 -vb 16 -cb mcg`,
same head and MTP treatment, BF16 embeddings and vision tower, attention serialized on disk. The
direction the solver chose was "take bits off `q_proj` and `in_proj_qkv`, put them on `up_proj`,
`k_proj`, `v_proj` and `out_proj`".

Per-role serialized bytes, from the emitted `quantization_manifest.json`: `full_attention`
1.112 GB (hydrated 1.260), `linear_attention` 3.898 (4.207), `mlp_gate_proj` 3.691 (3.568),
`mlp_up_proj` 3.958 (3.568), `mlp_down_proj` 4.225 (4.281), `lm_head` 0.954 (0.954),
`embed_tokens` 2.543, `vision_tower` 0.921, `mtp_draft` 0.283, norms 0.001. Each of the ten
realised figures equals the pre-registered prediction **exactly**.

## The measurement

Shard 0 of the v5 suite, 512 contexts, 1,048,064 scored positions, hidden state captured after the
final norm and replayed through the shared BF16 `lm_head` against the BF16 reference capture.

| build | mean KLD | median | p99.9 | exact max | top-1 |
|---|---:|---:|---:|---:|---:|
| hydrated, published | 0.002699883159684943 | 0.001090403 | 0.131263 | 3.734847 | 97.797 % |
| hydrated, re-captured for this comparison | 0.002699883159684943 | 0.001090403 | 0.131263 | 3.734847 | 97.797 % |
| **error-driven allocation (this build)** | **0.003066179635178366** | 0.001341795 | 0.137506 | 5.115300 | 97.506 % |

**Paired, hydrated − error-driven: −0.00036630 [−0.00039779, −0.00033477]** over 512 contexts and
330 source clusters, i.e. this build is +0.000366 [+0.000334, +0.000398] *worse*.

Three controls close the obvious escapes:

- **Not the reference.** The published shard-0 hydrated report was replayed against a BF16 capture
  that has since been deleted, so hydrated was re-captured and re-replayed against the surviving
  reference with the current harness. It came back bit-identical — same mean to 16 digits, same
  median, same p99.9, same exact max — and the paired report between published and re-captured is
  0.0 with a [0.0, 0.0] interval.
- **Not the solver.** Realised widths match the solved map for all 400 body modules, and the byte
  model was required to reproduce both published manifests byte-for-byte before it was allowed to
  solve anything.
- **Not propagation.** The ladder is measured at fixed propagation, so accumulation the proxy
  cannot see is the obvious suspect. It is not: the proxy error the real conversion achieved at the
  solved widths, under genuinely changed propagation, matched the fixed-propagation ladder at a
  median ratio of **0.9997**. The propagation term is −0.000028 of a −0.009841 objective delta:
  **0.3 %**.

## The 3.73x-per-bit law

Measured on this model, 409 modules, five widths each, 2,000 module-quantizations in one pass of
2 h 04 (against ~35 min for a plain conversion) by re-using each module's captured Hessian across
all five widths. Fitting `log eps(m,K) = a_m + s(K)` over the 400 calibrated body modules, `s(5)`
pinned to 0:

| K | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| `s(K)` | +2.6834 | +1.3327 | 0 | −1.3150 | −2.6089 | −3.8783 |
| implied ratio `eps(K−1)/eps(K)` | — | 3.8602 | 3.7913 | 3.7247 | 3.6470 | 3.5590 |

The per-bit ratio is **not** constant — it declines smoothly from 3.860 to 3.559, so each further
bit buys slightly less than the last. As a single geometric constant the same data gives
**3.7294 ± 0.0536** over 400 per-module fits, or **3.730 ± 0.097** over all 1,600 individual
rung-pair ratios. Per class it ranges from 3.661 (`linear_attn.in_proj_z`) to 3.767
(`mlp_down_proj`), the three MLP classes being the tightest at sd 0.004; across the eight
layer-octets it is flat (3.7225 … 3.7365), so there is no "late layers behave differently" effect
in the *shape*, only in the constant `a_m`.

Held out: predicting an unseen rung from one anchor rung plus the universal shape has median error
**+0.00 %** and mean absolute **1.175 %** over 8,000 held-out predictions; the single constant
instead of the shape is 2.6x worse at 3.03 %. The worst cases are all long extrapolations (K8→K4,
K7→K3); adjacent-width prediction, which is what an allocator actually needs, is the ~1 % case.

**Why 3.73 and not 4.** A pure information argument — one extra bit halves the quantization step,
quartering a squared error — gives 4.0. Nothing measured here reaches it, and the deficit widens
monotonically with width. It is stable across classes and flat across depth, so it is a property of
the quantizer, not of this model's layers. **We do not know why**, and this card declines to guess.

Not measured for: `lm_head` and the eight MTP draft modules (recipe-pinned by `-hb`/`-mb`, one rung
each; the draft additionally quantizes through the uncalibrated fallback, which reports `rmse` and
not this proxy), and the vision tower (unquantized at `-vb 16`).

## The closed-form allocation rule

With `eps(m,K) = c_m · 3.73^−K`, equalising marginal error per byte at a fixed serialized-byte
budget has a closed form: **sort modules by `log_3.73(c_m / numel_m)` and cut at the budget.** The
dynamic program does not merely approximate this, it reproduces it — the DP's score ranges per
assigned width come out monotone and essentially disjoint (K4 −16.57…−15.77, K5 −15.63…−14.76,
K6 −14.79…−13.78, K7 −13.77…−12.86).

The byte model is `bytes = 2*(in+out) + 4 + numel*K/8` per module plus each role's BF16/F16
companions; every module's cost is an integer multiple of 655,360 B, so the budget is an exact grid
of 25,664 points and the optimum is provable rather than greedy. It solves in about a second.

**The consequence: the two-hour ladder pass never has to be paid again.** One rung per module is
enough, and that is exactly what an ordinary conversion prints:

| allocation solved from | agreement with the five-rung DP | objective improvement recovered | bytes |
|---|---|---|---|
| the five-rung ladder (2 h 04) | — | 100 % by definition | 21,586,964,548 |
| one K5 rung + shape | 396 / 400 modules | 99.98 % | 21,586,964,548 |
| one K5 rung + the constant 3.7294 | 395 / 400 | 100.0 % | 21,586,964,548 |
| **a pre-existing log from a _different_ recipe** | **396 / 400** | **100.0 %** | 21,586,964,548 |

Every disagreement is one bit at a threshold boundary.

## The calibration that kills the objective

`proxy_err` is *relative*, so summing it across modules needs an assumption about per-module
sensitivity, and `out_energy` spans 26,000x across roles, so that choice is not cosmetic. Four
weightings, each with one free scale parameter, scored against **four measured paired KLD deltas**
between published checkpoints that differ only in widths — two uniform role-group moves and both
reallocations:

| rule | weight `w_m` | sign right on all 4 pairs | implied-scale spread | worst leave-one-pair-out ratio |
|---|---|---|---:|---:|
| `rel` — **the rule this build used** | 1 | **no** — wrong sign on the reallocation | changes sign | −0.69x |
| `numel` | parameter count | **no** — wrong sign on the reallocation | changes sign | −0.87x |
| `abs` | `out_energy` | yes | 13.3x | 4.40x |
| `sqrt_energy` | `√out_energy` | yes | **2.51x** | 2.47x |

`rel` predicted **−0.000251** for the reallocation against a measured **+0.000366**. That sign error
is precisely why this build lost, and the pre-registered selection rule could not have caught it:
both validation deltas available beforehand were *uniform* role-group moves, and neither tested a
reallocation **between** roles — which is the only thing the optimizer does.

`sqrt_energy` is the least-bad and the only rule inside a factor of ~2.5 everywhere, but a rule that
is a factor of two-and-a-half uncertain cannot decide a 1e-4 fidelity claim, which is the decision
it would have to support. It was also selected knowing this run's answer, so it is a
*pre-registrable candidate*, not a result. If it is tried, its validation must include a
between-role reallocation delta — and this run is now that third calibration point.

One number bounds how much of fidelity any width rule can even address: hydrated versus the
published online-K5/K6 build is **+0.000441 [+0.000412, +0.000474]** at *identical widths* — pure
mechanism, offline calibrated K6 against the runtime's uncalibrated online K6. Every width-based
surrogate predicts exactly zero there.

## Reproduce this

The allocation needs no GPU and no ladder pass — a conversion log you already have is the input:

```bash
# 1. solve the allocation from any existing conversion log (~1 s, no GPU)
tools/allocate_bits.py --ladder-from-log /path/to/convert.log \
    --out plan.json --fixed-out solved-fixed.json --override-out solved-override.json

# 2. convert (~50 min on one RTX PRO 6000 Blackwell)
export EXL3_BITS_FIXED=$PWD/solved-fixed.json      # attention, pinned before allocation
export EXL3_BITS_OVERRIDE=$PWD/solved-override.json # MLP, applied after it
python convert.py -i /models/Qwen3.8-27B -o out/qwen38-eda -w wd-eda \
    -b 4 -hb 6 -mb 4 -vb 16 -cb mcg -cpi 1800

# 3. finalize: index, quant config, verified topology, per-role bytes
python util/add_safetensors_index.py -m out/qwen38-eda --force
python util/add_quant_config.py -m out/qwen38-eda
tools/finalize_checkpoint.py -m out/qwen38-eda --upstream /work/upstream-index

# 4. measure against the published shard-0 BF16 reference (~6 min GPU, no second model load)
tools/fidelity.py capture --model out/qwen38-eda --suite <suite>/shard-0000/suite \
    --out hidden/eda --gpu-memory-utilization 0.85 --quantization exl3 --quantization-config "$QCFG"
tools/fidelity.py replay --reference <reference>/hidden-bf16 --candidate hidden/eda \
    --head lm_head.safetensors --suite <suite>/shard-0000/suite --out reports/report-eda.json
tools/fidelity.py paired --a reports/report-hyd.json --a-label hydrated \
    --b reports/report-eda.json --b-label error-driven --out reports/paired-hyd-vs-eda.json
```

Converter: [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
`@5f3c537ca9d89893d771256f5c43c93656553fbb` (**1.4.2**) plus this repository's allocation patch
`tools/exllamav3-allocation-bits-override.py`. Hardware: 1x RTX PRO 6000 Blackwell Server Edition,
SM120, driver 595.58.03 — **no number on this card is comparable to one from a different card.**

**What step 2 reproduces is the recipe, not this checkpoint.** The published bytes are the
artifact. The EXL3 converter is nondeterministic, measured on the hydrated build's recipe with the
same converter, flags and box: a fresh conversion returned **13 of 16 pinned payload files
identical** — configs, tokenizer, index, `quantization_config.json`, `quantization_manifest.json` —
with byte-identical shard headers, the same tensor names, dtypes, shapes and offsets, and the same
per-role byte totals and assigned widths, while **399 of the 409 quantized modules (97.6 %)
differed inside their `.trellis` payloads at 41-92 % of the bytes each (mean 82 %)**; no scale,
norm, embedding, vision or BF16 companion tensor moved. Two runs of one conversion minutes apart
agreed on every width and global scale and disagreed on the converter's own `proxy_err`, so this
is the converter and not the toolchain
([`receipts/converter-determinism.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/converter-determinism.json)).
Run the commands above and you get a **sibling** of this build — same composition, same widths,
same byte budget, different trellis payloads, a different valid artifact of the same recipe rather
than a broken one. The numbers on this card measure the bytes published here, so they stand;
"rebuild it and you get this interval" is an untested expectation. Verify against this repo's
`SHA256SUMS` rather than against your own conversion, and read a byte diff below this floor as the
converter, not as tampering or corruption. Two footnotes from the same measurement: the recorded
build environment is incomplete — the pinned r34 image has no `marisa_trie`, which the conversion
imports on an unconditional path, so the conversion-capable image is
`docker/Dockerfile.gg-r34-convert` — and none of this touches the runtime's weight-reconstruction
path (`reconstruct_fp8_slice`, reconstructed prefill), which is a different sense of the word.

**You do not need this checkpoint to check the headline.** Its 512-context hidden-state capture and
the matched hydrated re-capture are published in the fidelity dataset (see below), so
`fidelity.py replay` reproduces the paired interval with no GPU and no weights at all.

## Files in this repository

- the checkpoint: three safetensors shards, index, configs, tokenizer, chat template
- `SHA256SUMS` — the immutable payload (16 files); `DOCS-SHA256SUMS` — every other file in the
  tree (card, `.gitattributes`, `crc32.txt` and the `allocation/` artifacts), so a card edit
  cannot invalidate the build hashes
- `quantization_manifest.json` and `build-receipt.json` — authoritative for composition;
  `config.json → quantization_config` keeps one `bits`/`codebook` pair for loader compatibility and
  **cannot** describe this mixed checkpoint
- `allocation/solved-fixed.json`, `allocation/solved-override.json` — the exact per-module width
  maps this build was converted from
- `allocation/plan-error-driven-allocation.json` — the pre-registered plan, written before the
  conversion ran
- `allocation/build_solved.sh`, `allocation/convert-eda.log` — the build script and the full
  conversion log, including the realised proxy error of every module at its solved width

## Evidence

- [`docs/37-error-driven-allocation.md`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/37-error-driven-allocation.md)
  — the write-up: the law, the solve, the outcome, and what the ladder says about the hand-designed
  recipe
- [`receipts/error-driven-allocation.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/error-driven-allocation.json)
  — the verdict, the pre-registered plan, every control, prediction-versus-realised for all 409
  modules, and the preservation ledger
- [`receipts/error-driven-ladder.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/error-driven-ladder.json)
  — the full five-rung ladder: 409 modules × up to five widths, with `out_energy`, `numel`, `qmap`,
  per-rung seconds and the global scale search. This is the reusable artifact
- [`receipts/error-driven-surrogate-calibration.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/error-driven-surrogate-calibration.json)
  — the four-rule, four-pair calibration behind the table above
- [`malaiwah/qwen38-27b-fidelity-suite-v5`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5)
  — the suite, the shard-0 BF16 reference, the per-shard reports, and under `captures/shard-0000/`
  this build's own hidden-state capture plus the matched hydrated re-capture

## What is not verified

No capability, task-retention, throughput, context-capacity, vision or safety evaluation was run on
this build, and none will be: it is a measured negative, kept for its mechanism. The fidelity
evidence is shard 0 only (512 contexts), not the ten-shard 10,480,640-position suite the released
builds carry. It has never been served through the runtime for anything but hidden-state capture.

Like every EXL3 build in this family it requires the Gilded Gnosis vLLM fork with explicit
`--quantization exl3` and an exact `ignore` list; it does **not** load in upstream vLLM, SGLang,
TensorRT-LLM, llama.cpp, transformers or stock exllamav3. If you want a build to actually run, take
[the hydrated one](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated).

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — the EXL3 Trellis format,
  LDLQ calibration, the MCG codebook, the conversion pipeline, and the `proxy_err` this whole
  experiment is built on. The upstream error-driven path (`util/measure.py` → `util/optimize.py`)
  measures real per-group `dkld/dbits` by splicing whole converted models; it is a better signal
  and a much larger bill, and it is not what was done here.
- [Qwen](https://huggingface.co/Qwen) — the base model.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path the capture ran through.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
