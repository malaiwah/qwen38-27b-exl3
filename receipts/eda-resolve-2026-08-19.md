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

Lesson recorded: "not in the model repo" is not "not published". Enumerate every
repo of the account, both types, and grep the file lists — the scan that found it
took 11 s.

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

**Corrected recommendation:** if a trellis rebuild ever runs, the candidate to
solve is **`abs`**, not `sqrt_energy`. And the deeper point stands — every one of
these objectives is first-order and layer-local, so none of them can see the KV
compounding that the measured attribution work says dominates. The right fix is a
compounding-aware objective, not a different weighting of a blind one.

## What is NOT claimed

No KLD prediction is attached to `sqrt_energy` or `abs`. The plan's
`kld_per_objective_unit = 0.021423` was calibrated for **`rel` units**; objective
units are not comparable across weightings, so transferring that scale would be
meaningless arithmetic. Any of these allocations remains a pre-registrable
candidate requiring a **paired** shard-0 validation against hydrated, shipping only
if the paired interval excludes zero in its favour.
