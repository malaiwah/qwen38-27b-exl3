# 57 — Should the EDA allocation be folded into the next trellis build of the flagship K5K6?

**Date:** 2026-08-19. **Scope:** assessment only; no GPU work, no rebuild. Every
numeric claim below is traceable to a fetched file or an in-repo receipt. Items
not directly measured by us are labelled `[INFERENCE]` or attributed to their
source.

The question: should the error-driven-allocation (EDA) approach from
[`malaiwah/Qwen3.8-27B-EXL3-EDA-research`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-EDA-research)
be folded into the next trellis build of the flagship checkpoint
([`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated))?

**Short answer.** Fold the *methodology*, not the *solved allocation*. The
published EDA allocation regresses fidelity against hydrated by +0.000366 mean
KLD (our suite), and its bit-shift direction contradicts our attribution
physics. A future rebuild should re-solve with the sign-correct weighting
candidate, but should not be scheduled solely for EDA while the mixed-GPTQ
requant lane is unmeasured on our suite.

---

## 1. What EDA-research is

The EDA checkpoint is the output of a per-module bit-allocation solver applied
to the same Qwen3.8-27B base, at the same serialized-byte budget as the
hydrated recipe. It is published as a **measured negative**: it was built to
beat hydrated and lost. The mechanism and the reusable infrastructure are the
result; the checkpoint is not.

### Solver, inputs, constraints

From `allocation/plan-error-driven-allocation.json` (fetched from the HF repo):

- **Decision variables:** the trellis width K (bits) of each of the 409
  quantized modules, independently. Candidate widths are K3-K7 for "big"
  modules (numel > 52,000,000) and K4-K8 for "small" modules.
- **Error metric (the objective):** `proxy_err = tr(E^T H E) / tr(W^T H W)`,
  exllamav3's own per-module Hessian-weighted *relative* quantization error
  (`allocation/plan-error-driven-allocation.json` → `ladder.meta.metric`). The
  chosen objective is `rel`: **`sum_m eps(m,K)`** — the proxy error summed over
  modules with equal weight (`objective.chosen = "rel"`,
  `objective.definition.rel`). An alternative `abs`
  (`sum_m out_energy_m * eps(m,K)`) was considered and rejected at selection
  because its implied KLD-per-objective-unit scale varied 2.52x across two
  validation deltas, versus 1.51x for `rel`
  (`objective.validation.{rel,abs}.scale_ratio_across_the_two_deltas`).
- **Constraint:** a fixed serialized-byte budget of **21,586,964,548 B** —
  identical to the hydrated build's serialized payload
  (`budget_bytes`, `budget_source = "receipts/hydrated-quantization-manifest.json
  role total (serialized payload)"`). The byte model is
  `bytes(role,K) = fixed(role) + params(role)*K/8` per module; every module's
  cost is an integer multiple of 655,360 B, so the budget is an exact grid of
  25,664 points (`grid_unit_bytes`, `grid_points`).
- **Solver:** exact dynamic programming over the byte grid, not greedy
  (`solver = "exact dynamic programming over the byte grid (not greedy)"`).
  With `eps(m,K) = c_m * 3.73^(-K)` (the measured 3.73x-per-bit law), the
  optimum has a closed form: sort modules by `log_3.73(c_m / numel_m)` and cut
  at the budget. It solves in about a second (HF model card README, "The
  closed-form allocation rule").
- **Ladder input:** a five-rung per-module error curve measured for all 409
  modules in one 2 h 04 pass (7,473.2 s, `ladder.meta.elapsed_sec`), reusing
  each module's captured Hessian across all five widths. Propagation is fixed at
  the hydrated recipe throughout (`propagation_recipe = "hydrated"`), so the
  ladder is the module-local error of changing one module's width, not the
  sequential effect of changing many at once.

The solver predicted a **-0.009841** objective delta (0.075355 → 0.065514) and
a **-0.000211** KLD delta (0.002700 → 0.002489) for the solved allocation
versus hydrated (`predicted.*`). The prediction was first-order and
layer-local: the proxy sees no error accumulation between layers
(`predicted.note`).

### What was actually built and measured

From the HF model card README ("What was built" / "The measurement") and
confirmed against `allocation/solved-fixed.json` (attention) and
`allocation/solved-override.json` (MLP):

| role | modules | hydrated | EDA solved | serialized GB (hyd → EDA) |
|---|---|---|---|---|
| `full_attention` q/k/v/o | 64 | K6 × 64 | K4×5, K5×13, K6×17, K7×29 | 1.260 → 1.112 |
| `linear_attention` (GDN) | 144 | K6 × 144 | K5×61, K6×70, K7×13 | 4.207 → 3.898 |
| `mlp_gate_proj` | 64 | K5 × 64 | K4×1, K5×51, K6×12 | 3.568 → 3.691 |
| `mlp_up_proj` | 64 | K5 × 64 | K4×1, K5×27, K6×36 | 3.568 → 3.958 |
| `mlp_down_proj` | 64 | K6 × 64 | K4×1, K5×3, K6×60 | 4.281 → 4.225 |
| `lm_head` | 1 | K6, mcg | K6, mcg (unchanged) | 0.954 → 0.954 |

175 of 400 body modules moved; average body width is 5.531 bits in both. The
serialized payload is byte-identical (21,586,964,548 B). Per-role bytes are from
the emitted `quantization_manifest.json` (HF model card README).

The model card states the solver's direction plainly: "take bits off `q_proj`
and `in_proj_qkv`, put them on `up_proj`, `k_proj`, `v_proj` and `out_proj`."

On our shard-0 suite (512 contexts, 1,048,064 positions, shared BF16 head and
reference), the measured outcome was the **opposite** of the prediction:

| build | mean KLD | 95% CI |
|---|---|---|
| hydrated | 0.002700 | — |
| EDA (this build) | 0.003066 | — |

Paired, hydrated − EDA: **−0.000366** [−0.000398, −0.000334]. Hydrated wins
470 of 512 contexts; EDA wins 42. The interval excludes zero. (HF model card
README, "The measurement".) The proxy objective fell 13.1% while mean KLD rose
13.6% — a clean demonstration that `sum_m eps(m,K)` is not a monotone
surrogate for KLD.

---

## 2. The independent validation: 8.7% better on a third-party harness

An independent third-party harness (Discord leaderboard, protocol unknown)
measured both builds (`receipts/discord-leaderboard-2026-08-19.md`):

| model | scheme | KLD | prefill | decode |
|---|---|---|---|---|
| malaiwah EXL3-EDA-research | error-driven allocation | 0.007461 | 1508 | 21.0 |
| malaiwah EXL3-K5K6 | uniform K5/K6 (online) | 0.008170 | 1679 | 22.3 |

EDA ranks **8.7% lower KLD** than the online K5/K6 build on their harness, at
similar speed.

### Evidentiary weight and limits, stated plainly

- **What it validates:** within their harness, the EDA allocation beats the
  *online* K5/K6 build. Their "K5K6" row is the published
  `malaiwah/Qwen3.8-27B-EXL3-K5K6` (the online build with runtime-encoded
  attention), not the hydrated build with offline-serialized attention. This is
  the same direction we see on our own suite: EDA (0.003066) beats online
  (0.003141) but loses to hydrated (0.002700) (HF model card README; our
  `receipts/kld5-1M-tail-{hyd,k5k6}.json` via PROGRESS.md).
- **What it does NOT validate:** it does not show EDA beats the hydrated
  recipe. On our suite EDA is 13.6% worse than hydrated. The 8.7% figure is
  EDA-versus-online, a different and weaker comparison.
- **Why the direction still holds:** EDA serializes attention offline
  (calibrated), while the online build encodes attention at load (uncalibrated
  K6 overlay). Even with EDA's attention reallocated — some modules demoted to
  K4/K5 — offline calibration beats the runtime overlay on average. We already
  knew offline beats online: hydrated beats online by 0.000450 on our 10M
  suite (`README.md`, Evidence at a glance). The third-party result is
  consistent with that, not new.
- **Protocol caveat (theirs, restated):** their protocol (context length,
  positions, reference model) is unknown; absolute KLDs are not comparable to
  our 512-ctx suite. Only within-harness ordering is evidence
  (`receipts/discord-leaderboard-2026-08-19.md`, Caveats). Their prefill
  (1,508/1,679) matches our *pre-cure stock* numbers; none of our serving fixes
  are in their harness.

---

## 3. Does the solved allocation agree with our attribution physics?

This is the key analytical question. **No — it contradicts it, and the
contradiction is the most informative thing about the experiment.**

### Our attribution physics (measured, multiple receipts)

- **MLP error is nearly free / additive.** The additive KLD model validated on
  held-out predictions at -1.9% and -6.0% for MLP groups
  (`receipts/glm52-transfer-2026-08-19.md`; `receipts/balanced-profile-2026-08-19.md`
  §"Third validation of the additivity model"). Quantizing all 64 MLP layers to
  NVFP4 costs the third-party harness nothing (0.002666 vs 0.002670 attn-only,
  `receipts/discord-leaderboard-2026-08-19.md` §1).
- **Attention/GDN error dominates KLD via KV compounding.** `self_attn`
  under-predicted by +46% because attention error propagates through the KV
  cache and is re-read at every later position, compounding over the scored
  window instead of staying per-position
  (`receipts/glm52-transfer-2026-08-19.md`; `receipts/selfattn-fp4-additivity-failure-2026-08-19.md`).
  MTP acceptance also couples to attention fidelity
  (`receipts/discord-leaderboard-2026-08-19.md` §1).
- **The budget consequence:** with a measured floor of 0.003412, only
  `self_attn` (7% of parameters) fits the post-floor KLD budget;
  prefill-grade throughput and trellis-grade fidelity are mutually exclusive on
  31.4 GiB (`receipts/kld-axis-conclusion-2026-08-19.md`; PROGRESS.md
  2026-08-19).

### What the solver actually did

At the role level, the solver moved bytes **from attention to MLP**:

| role class | net byte change (hyd → EDA) |
|---|---|
| full_attention | −0.148 GB |
| linear_attention (GDN) | −0.309 GB |
| mlp_gate_proj | +0.123 GB |
| mlp_up_proj | +0.390 GB |
| mlp_down_proj | −0.056 GB |

Attention (full + linear) lost ~0.46 GB; MLP (gate + up) gained ~0.51 GB. This
is the **opposite** of what attribution physics predicts: the solver invested
bits where error is nearly free (MLP) and withdrew bits from where error
compounds (attention/GDN).

The within-attention split is nuanced — q_proj and in_proj_qkv were demoted
while k/v/o were promoted in some layers — but the net role flow is
unambiguous: attention shrank, MLP grew.

### Why: the `rel` objective is blind to KV compounding

The `rel` objective weights every module equally in *relative* error. It
cannot see that attention error compounds through the KV cache while MLP error
is per-position. The calibration in the HF model card ("The calibration that
kills the objective") makes this exact: the `rel` rule predicted **−0.000251**
for the reallocation against a measured **+0.000366** — a **sign error**. Two
of four candidate weightings (`rel` and `numel`) got the sign wrong on the
reallocation; only `abs` (out_energy) and `sqrt_energy` (sqrt(out_energy)) got
the sign right on all four calibration pairs.

`out_energy` spans 26,000x across roles (HF model card README), so the
weighting choice is not cosmetic. The `rel` rule, by giving every module equal
weight, systematically over-values large-MLP modules (high parameter count,
large absolute error term) and under-values the compounding effect of attention
error.

### The resolution: the two lines of evidence are in tension, and the tension resolves against the solver

The EDA experiment is, in effect, an unintended independent test of the
attribution physics. A solver that did not model KV compounding moved bits the
wrong way — from attention to MLP — and lost by a bounded, interval-excluding
margin. That is exactly what "attention error dominates, MLP error is nearly
free" would predict. The build's failure corroborates the attribution physics;
the solver's allocation direction does not.

---

## 4. What a next trellis build would look like

### Inputs needed

The allocation solve needs no GPU and no ladder pass. Per the HF model card and
`allocation/plan-error-driven-allocation.json`, the closed-form rule
(`sort by log_3.73(c_m / numel_m), cut at budget`) reproduces the full DP to
396/400 modules from a single K5 rung — and even from a pre-existing conversion
log from a *different* recipe (396/400, 100.0% objective recovered). So the
solve is ~1 second from a log we already have.

A rebuild then needs one conversion: ~50 min on one RTX PRO 6000 Blackwell
(`build-receipt.json` → `converter`; HF model card README reproduce step 2),
plus ~6 min for a shard-0 fidelity capture/replay. The ladder pass (2 h 04) is
no longer required. The payload is budget-neutral — identical serialized bytes
to hydrated — so there is no VRAM or download cost.

### Expected gain, bounded

The only same-harness ordering we have for EDA is the 8.7% win over the
*online* build on the third-party harness. Against hydrated (the actual
flagship), EDA is **worse** on our suite by 13.6%. So the current `rel`-solved
allocation has a negative expected gain against the incumbent.

The realistic opportunity is a **re-solve with the `sqrt_energy` weighting**
(w_m = sqrt(out_energy)), the only candidate that got the sign right on all
four calibration pairs. `[INFERENCE]` A `sqrt_energy` solve would shift bits
toward high-out_energy modules — which, given attention's KV-compounding, is
the direction the attribution physics favours. But `sqrt_energy` is a factor of
2.5 uncertain in magnitude (worst leave-one-out ratio 2.47x, HF model card
calibration table) and was selected knowing this run's answer, so it is a
*pre-registrable candidate*, not a result. Its validation must include a
between-role reallocation delta — and this experiment is now that third
calibration point. The expected gain from a `sqrt_energy` rebuild is therefore
bounded above by the 8.7% ordering (against online) and is of unknown sign
against hydrated until measured.

### Cost of a full requant

- GPU: ~50 min conversion + ~6 min fidelity = ~1 h of serial GPU time.
- No payload-size change (budget-neutral).
- No ladder pass needed (closed-form rule from an existing log).
- The EXL3 converter is nondeterministic at the trellis-payload level (399/409
  modules differ in bytes across runs, mean 82%), but a sibling rebuild scores
  indistinguishable from the published checkpoint (−3.755e-06, CI brackets
  zero), so the recipe determines fidelity (HF model card README; our
  `receipts/converter-determinism.json`, `receipts/sibling-rebuild-fidelity.json`).

---

## 5. Interaction with the mixed GPTQ requant lane

A mixed GPTQ requant lane (FP8-attn + NVFP4-MLP) is in flight and measured
**0.002666** on the third-party harness — better than EDA (0.007461) and better
than hydrated's third-party-equivalent online build (0.008170)
(`receipts/discord-leaderboard-2026-08-19.md`).

These two approaches are **not mutually exclusive but compete for serial GPU
time**:

- **EDA improves the trellis family.** It keeps the EXL3 trellis format, the
  same serving path, the same runtime, and the same payload size. Its gain
  (if any, after re-solving) is a pure reallocation within the existing format.
  It cannot change the format-level fidelity ceiling.
- **GPTQ-mixed is a different format family.** It replaces the trellis
  quantization procedure with Hessian-compensated GPTQ calibration from BF16,
  which the discord receipt identifies as the dominant fidelity lever: on our
  suite, runtime-cast MLP→FP4 costs 0.042 over trellis, while GPTQ-calibrated
  MLP→NVFP4 costs 0.000004 over its own baseline — a 10^4 ratio in damage from
  the procedure, not the format (`receipts/discord-leaderboard-2026-08-19.md`
  §2). GPTQ-mixed has its own serving profile (W4A16 MLP, 4,551 PP on the
  third-party harness, which fails our PP >= 7000 criterion) and is not yet
  measured on our own suite.

The two address different layers of the problem: EDA is an *allocation*
optimization within one quantization procedure; GPTQ-mixed is a *procedure*
change that makes allocation less critical (because per-module error is 10^4
lower to begin with). If GPTQ-mixed delivers on our suite, the marginal value
of a trellis-only allocation rebuild shrinks. If it does not (e.g. it fails PP
or context criteria as the third-party numbers hint), the trellis family
remains the serving path and an EDA re-solve is the best in-family lever.

---

## 6. Recommendation

1. **Do not fold the current EDA `rel`-solved allocation into a trellis
   rebuild.** It regresses by +0.000366 mean KLD against hydrated (our suite),
   its bit-shift direction (attention → MLP) contradicts the attribution
   physics, and the calibration diagnosed the `rel` objective's sign error as
   the cause. Shipping it would be a measured step backwards.

2. **Fold the EDA methodology into any future trellis rebuild that happens for
   other reasons.** If a rebuild is already scheduled (e.g. for a new base
   revision, a codebook change, or a ParoQuant pre-rotation trial per
   `docs/14-paro-assessment.md`), re-solve the allocation with the
   `sqrt_energy` weighting instead of `rel`, validate against the hydrated
   recipe on shard 0, and ship only if the paired interval excludes zero in
   EDA's favour. The solve costs ~1 s and no GPU; the conversion is the only
   marginal cost and would happen anyway.

3. **Do not schedule a rebuild solely for EDA while the GPTQ-mixed lane is
   unmeasured on our suite.** GPU time is serial. GPTQ-mixed measured 0.002666
   on the third-party harness — below hydrated's 0.002700 — and addresses the
   procedure-level error that EDA cannot. Measuring GPTQ-mixed on our own
   suite is the higher-value next GPU slot. If GPTQ-mixed fails our PP or
   context criteria, the trellis family remains the serving path and an EDA
   re-solve becomes the best in-family fidelity lever at that point.

4. **If a `sqrt_energy` re-solve is run, pre-register it with a between-role
   reallocation delta in the validation set.** The `rel` objective failed
   precisely because its two validation deltas were both uniform role-group
   moves and neither tested a reallocation between roles — the only thing the
   optimizer does. This experiment is now the third calibration point; a
   `sqrt_energy` run must include it.

### Conditions summary

| condition | action |
|---|---|
| rebuild already scheduled for another reason | re-solve with `sqrt_energy`, validate vs hydrated, ship only if paired CI excludes zero in favour |
| no rebuild otherwise scheduled, GPTQ-mixed unmeasured on our suite | do not schedule a rebuild for EDA; measure GPTQ-mixed first |
| GPTQ-mixed measured and fails PP/context on our suite | EDA `sqrt_energy` re-solve becomes the best in-family lever; schedule it |
| GPTQ-mixed measured and passes all criteria on our suite | trellis family is superseded for fidelity; EDA re-solve is lower priority |

---

## Sources

- HF model card: `malaiwah/Qwen3.8-27B-EXL3-EDA-research` README (allocation
  table, per-role bytes, measurement, calibration, 3.73x law, closed-form rule,
  sibling-rebuild control).
- `allocation/plan-error-driven-allocation.json` — solver, objective, budget,
  byte law, ladder, predicted deltas.
- `allocation/solved-fixed.json` — attention per-module width map.
- `allocation/solved-override.json` — MLP per-module width map.
- `build-receipt.json` — converter, immutable payload, recipe.
- `receipts/discord-leaderboard-2026-08-19.md` — third-party 8.7% ordering,
  speeds, attribution-recipe validation, procedure-vs-format.
- `receipts/glm52-transfer-2026-08-19.md` — MLP additivity -1.9%/-6.0%,
  self_attn +46%, KV compounding, MTP acceptance coupling.
- `receipts/balanced-profile-2026-08-19.md` — additivity holds for MLP, fails
  for attention.
- `receipts/kld-axis-conclusion-2026-08-19.md` — floor 0.003412, per-group
  contributions, mutual exclusivity on 31.4 GiB.
- `receipts/selfattn-fp4-additivity-failure-2026-08-19.md` — attention error
  compounding mechanism.
- `README.md` — current profiles (fidelity KLD 0.003405, hydrated 0.002700,
  online 0.003141), hydrated-vs-online paired −0.000450.
- `PROGRESS.md` 2026-08-19 — KLD proof, additive attribution, per-group budget.
- `docs/54-multi-precision-strategy.md` — multi-precision FP4/FP6 strategy
  context.

---

## Re-solve status (2026-08-19, goal session): DONE — refutes the `sqrt_energy` candidate, and corrects two errors of mine

The re-solve ran, with **no GPU**. Full account: `receipts/eda-resolve-2026-08-19.md`.
`tools/eda-resolve.py` reproduces the published `rel` solve bit-for-bit (objective
0.0655141051 vs 0.06551410506951784, 175 modules moved, every role byte delta
identical) after three exact validations: derived `fixed(role)` -> all 10 hydrated
role totals; -> the published solved role totals; and the objective domain pinned
over the 400 movable modules, reproducing this plan's own `objective_hydrated` to
15 significant figures.

### Two corrections to earlier notes of mine in this file

**1. I briefly deferred this lane claiming its input was lost. That was wrong
twice over.** The ladder *is* published — in a different repo of the same account
(`malaiwah/qwen38-27b-fidelity-suite-v5`, dataset ->
`captures/shard-0000/error-driven-ladder.json`) — and, more to the point, **§4 of
this document already said no ladder is needed at all**: the closed-form rule
(`sort by log_3.73(c_m / numel_m)`, cut at budget) recovers 396/400 modules and
100.0% of the objective from a pre-existing conversion log. I asserted a 2.08 h GPU
cost while editing a file that stated the cost was ~1 s. Read the whole document
before acting on part of it.

**2. `abs` is not a clean "corrected recommendation".** §1 records that `abs` was
*considered and rejected at selection*: its implied KLD-per-objective-unit scale
varies **2.52x** across the two validation deltas versus **1.51x** for `rel`. So the
honest position is a genuine tension, not a fix:

| weighting | direction on attention+GDN | sign correct on all 4 calibration pairs | scale consistency |
|---|---|---|---|
| `rel` (shipped, measured worse) | −456,785,920 | no (sign error on reallocation) | best (1.51x) |
| `sqrt_energy` (§4's candidate) | **−155,975,680** | yes | intermediate (2.47x worst LOO) |
| `abs` | **+100,270,080** | yes | worst (2.52x) |

### What the solve establishes

The `[INFERENCE]` in §4 — that `sqrt_energy` would shift bits toward attention —
**is false**. `sqrt_energy` still strips 156 MB from attention and GDN: the same
direction as the objective that already regressed, merely gentler. Of the three,
only `abs` moves bytes *toward* attention/GDN (+100 MB, 0.46% of budget), funded by
gutting `mlp_down_proj` (−1.34 GB) — and `abs` is precisely the weighting with the
worst scale consistency. All three agree on taking bits off `down_proj` for
`gate/up`.

So §6's recommendation 2 ("re-solve with `sqrt_energy`") should now read: **do not
re-solve with `sqrt_energy` — its allocation is directionally the same error as
`rel`.** If an in-family lever is ever needed, `abs` is the only sign-correct
*and* right-direction candidate, and it must be pre-registered with the between-role
reallocation delta demanded by §6.4, accepting its 2.52x scale uncertainty.

The deeper conclusion is that none of these weightings is adequate: every one is
first-order and layer-local, so all are structurally blind to the KV compounding the
attribution work measured. The next real step for this family is a
compounding-aware objective, not another reweighting of a blind one.

No KLD prediction is attached to either new allocation: the plan's
`kld_per_objective_unit` is calibrated in `rel` units and does not transfer across
weightings.
