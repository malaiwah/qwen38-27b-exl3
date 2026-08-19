# EDA re-solve: solver validated exactly, and `sqrt_energy` is the WRONG candidate

**Date:** 2026-08-19 (goal session). Tool: `tools/eda-resolve.py`.
Artifacts: `receipts/eda-resolve/resolve-{rel,sqrt_energy,abs}.json`.
**No GPU used.**

## The input was recoverable after all — I was wrong to defer

I deferred this lane claiming the 409-module `(proxy_err, out_energy)` ladder was
unrecoverable without 2.08 h of GPU: the plan references it only as
`/var/tmp/work/kld6/ladder.json` (gone), and it is absent from the EDA-research
**model** repo's file list, which is where I looked. The maintainer asked whether I
had checked the private/other repositories. I had not. It is published, in a
different repo of the same account:

`malaiwah/qwen38-27b-fidelity-suite-v5` (dataset) ->
`captures/shard-0000/error-driven-ladder.json` — all 409 modules, `numel`,
`out_energy`, `recipe_bits`, and `ladder{width: proxy_err}` for five widths each.

Two lessons, the second worse than the first:

1. "Not in the model repo" is not "not published". Enumerating every repo of the
   account, both types, and grepping the file lists took **11 s**.
2. **The document I was editing already answered the question.** docs/57 §4 states
   the solve needs no ladder at all: the closed-form rule
   (`sort by log_3.73(c_m / numel_m)`, cut at budget) recovers 396/400 modules and
   100.0% of the objective from a pre-existing conversion log, so the solve is "~1 s
   from a log we already have". I asserted a 2.08 h GPU cost while editing a file
   that put it at ~1 s, because I had grepped that file for one section instead of
   reading it. Read the whole source before acting on part of it.

## The reconstruction is falsifiable, and it passes

`tools/eda-resolve.py` refuses to emit an allocation unless its byte model
reproduces published ground truth. Three independent checks, all exact:

1. **Byte law -> hydrated role totals.** `fixed(role)` is not published; derived it
   from the hydrated totals and `recipe_bits`, then reproduced all 10 role byte
   totals **exactly**.
2. **Byte law -> published SOLVED role totals.** Reproduced exactly, once both
   halves of the published solve are loaded (the original splits attention/GDN into
   `solved-fixed.json` (212 entries) and MLP into `solved-override.json` (195) —
   loading only the first triggered the guard, which is what a guard is for).
3. **Objective domain.** Recomputing the plan's own `objective_hydrated` over the
   **400 movable** modules (excluding `lm_head` + the 8 MTP modules, which the
   ladder carries but the solver pins) gives 0.07535511617344567 against the
   published 0.07535511617344577 — agreement to 15 significant figures.

Then the full `rel` solve reproduces the published solve **bit-for-bit**:
objective 0.0655141051 (published 0.06551410506951784), **175 modules moved**, and
all five role byte deltas identical (linear_attention −309,329,920;
full_attention −147,456,000; mlp_down −55,705,600; mlp_gate +122,552,320;
mlp_up +389,939,200).

## The result: docs/57's `sqrt_energy` recommendation does not survive its own solve

| weighting | attention+GDN bytes | MLP gate+up | MLP down | moved |
|---|---|---|---|---|
| `rel` (published, measured WORSE by +0.000366) | **−456,785,920** | +512,491,520 | −55,705,600 | 175 |
| **`sqrt_energy`** (docs/57's candidate) | **−155,975,680** | +980,418,560 | −824,442,880 | 236 |
| `abs` (= out_energy) | **+100,270,080** | +1,236,664,320 | −1,336,934,400 | 320 |

docs/57 reasoned `[INFERENCE]` that "a `sqrt_energy` solve would shift bits toward
high-`out_energy` modules — which, given attention's KV-compounding, is the
direction the attribution physics favours." **The actual DP says otherwise:**
`sqrt_energy` still strips 156 MB from attention and GDN. It is directionally the
same mistake as `rel`, merely smaller — so if the attribution physics holds, a
`sqrt_energy` rebuild should regress too, just less.

**Only `abs` moves bytes toward attention/GDN**, and even then modestly
(+100 MB = 0.46% of the 21.6 GB budget), funded almost entirely by gutting
`mlp_down_proj` (−1.34 GB). All three weightings agree on one thing: take bits off
`mlp_down_proj` and give them to `gate/up`.

**Corrected recommendation, with the tension stated.** `sqrt_energy` should not be
re-solved: its allocation is directionally the same error as `rel`. But `abs` is not
a clean replacement either — docs/57 §1 records that `abs` was *considered and
rejected at selection* because its implied KLD-per-objective-unit scale varies
**2.52x** across the two validation deltas versus **1.51x** for `rel`. The real
picture:

| weighting | attention+GDN direction | sign correct on all 4 pairs | scale consistency |
|---|---|---|---|
| `rel` (shipped, measured worse) | −456,785,920 | no | best (1.51x) |
| `sqrt_energy` | −155,975,680 | yes | intermediate (2.47x worst LOO) |
| `abs` | **+100,270,080** | yes | worst (2.52x) |

`abs` is the only candidate that is both sign-correct and right-direction, and it is
also the least scale-consistent. That is a tension, not a fix. Any `abs` re-solve
must be pre-registered with the between-role reallocation delta docs/57 §6.4
requires, and its 2.52x scale uncertainty accepted up front.

The deeper point stands regardless: every one of these objectives is first-order and
layer-local, so none can see the KV compounding the attribution work measured. The
right fix is a compounding-aware objective, not another reweighting of a blind one.

## What is NOT claimed

No KLD prediction is attached to `sqrt_energy` or `abs`. The plan's
`kld_per_objective_unit = 0.021423` was calibrated for **`rel` units**; objective
units are not comparable across weightings, so transferring that scale would be
meaningless arithmetic. Any of these allocations remains a pre-registrable
candidate requiring a **paired** shard-0 validation against hydrated, shipping only
if the paired interval excludes zero in its favour.

---

## v2: structured weight models — and the class model falsifies itself on first contact

**Added** (same day, on request): `class-kld` weighting (per-class KLD-per-eps-unit
scales fitted from the plan's own two measured single-class deltas: attn 0.017425,
mlp 0.026340 — the 1.51x spread the plan averaged away), `--max-width` (b12x
ANY_BITS prefill supports K3–K6 only, `patches/vllm-exl3-multiprecision.py:1644`;
the published solve put **42 modules at K7**, all off the fast path on our stack),
`--depth-form u` (both-ends weighting; prior art in llama.cpp `use_more_bits`,
`src/llama-quant.cpp:430`, and exllamav3's own `allocation.py` — docs/58),
`--budget-delta`, width heatmaps, and a `--compare` mode. Regression re-verified:
`rel` at defaults still replicates the published solve bit-for-bit.

### The held-out test: the class model has the WRONG SIGN too

The two class scales are fitted on single-class moves; the published solve is a
cross-class reallocation neither saw. Prediction for it:

| model | predicted ΔKLD | measured |
|---|---|---|
| plan's global scale | −0.000211 | +0.000366 |
| **2-class scales** | **−0.000328** | +0.000366 |

Wrong sign, and *further* from truth than the global scale. My hypothesis that
"the 1.51x spread IS the class signal" is **falsified**: whatever makes
cross-class reallocation regress is not expressible as any per-module reweighting
fitted from those two deltas. Candidates that survive: within-class heterogeneity
(the k/v-vs-q/o state-writer split), depth structure, and true non-additivity.
Consequence: every `predicted_kld_delta` this tool emits for class-kld solves
carries demonstrated wrong-sign risk for cross-class moves — the artifacts say so,
and the exp05 solve's −0.0059 "prediction" should be read as a red flag that
large-amp solves leave the calibrated regime, not as a forecast.

### What the matrix shows anyway (allocations, not predictions)

| | attn+GDN | MLP g+u | L00-15 | L48-63 | agree w/ rel |
|---|---|---|---|---|---|
| rel (published) | −457 MB | +512 MB | 0 | 0 | 100% |
| class-kld | **−657 MB** | +702 MB | +108 MB | −156 MB | 82.9% |
| class-kld K≤6 | −668 MB | +713 MB | +108 MB | −161 MB | 77.3% |
| class-kld K≤6 u=.05 | −579 MB | +724 MB | +212 MB | **+52 MB** | 68.9% |
| class-kld K≤6 exp=.05 | −490 MB | +724 MB | **+325 MB** | **−619 MB** | 55.5% |
| abs K≤6 | **−279 MB** | +1,281 MB | −460 MB | +330 MB | 29.6% |

- **class-kld doubles down on attention-robbing** (−657 MB): the fitted scales say
  an MLP eps-unit costs 1.51x more KLD than an attention one, so the solver strips
  attention harder than `rel` did. Given the held-out sign failure, this direction
  is exactly as suspect as `rel`'s.
- **The K≤6 serving cap is nearly free**: 93.9% agreement with the uncapped solve;
  role deltas barely move. There is no reason ever to allow K7/K8 for our stack.
- **`u` and `exp` depth forms are genuinely different allocations** (59.9%
  agreement): exp piles bytes into the first half; u protects both ends and strips
  the middle. Two buildable, discriminable hypotheses.

### Experiment upgrade

The queued depth calibration becomes **three arms**, same cost structure
(13 layers each, identical bytes and KV): `FP6_LAYER_RANGE=0-12` (early) vs
`26-38` (middle) vs `51-63` (late). Monotone-early predicts
KLD(early) > KLD(middle) > KLD(late); U-shape predicts middle < both ends...
inverted: converting the MIDDLE to FP6 should hurt LEAST under the U-hypothesis
(middle layers are least sensitive) and converting EARLY should hurt most under
the exp-hypothesis. Three arms discriminate the two forms; two arms cannot.

### How much of hydrated's KLD the objective can even see

Un-anchored absolute extrapolation (objective at hydrated widths x fitted scale)
vs the measured offline 0.002700: global scale gives **0.001614 (59.8%)**,
per-class scales **0.001851 (68.6%)**. A third to 40% of real KLD is invisible to
the module-local ladder under any scale fitted from it — that mass lives in KV
compounding, cross-module interaction and propagation, which the one-module-at-a-
time, frozen-propagation ladder structurally cannot contain. Delta accuracy by
regime: single-class ±19–23% (global scale); cross-class reallocation wrong-sign.
The tool is a ~±20% within-class ranking instrument, not a KLD estimator.
