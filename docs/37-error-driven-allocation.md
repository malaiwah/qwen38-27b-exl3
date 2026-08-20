# Error-driven allocation: a measured law for EXL3 proxy error, and one candidate built from it

This closes [29](29-plan-and-loose-ends.md) P2 / rank 9. The headline is not the checkpoint. The
headline is that EXL3's per-module quantization error follows one shape across every width, so the
information an error-driven allocator needs is **one number per module**, obtainable from an
ordinary conversion log — and that a byte-constrained allocation over it can be solved exactly
rather than greedily.

Terminology is the F4 convention: **serialized bytes** are what a checkpoint occupies on disk, and
none of the numbers here is called VRAM. Every measurement is SM120 on the rental RTX PRO 6000
(driver 595.58.03); nothing here is comparable to a number from AIBoss's 5090.

## 1. What the converter actually gives you

exllamav3 allocates bits by a static priority order, not by measured error — Gap 1 in
[04](04-exllamav3-toolchain.md). It does emit an error, once per module, at the width the
allocator already chose:

```
proxy_err = tr(EᵗHE) / tr(WᵗHW)
```

computed in `modules/quant/exl3_lib/quantize.py` and printed by
`conversion/convert_model.py:print_quantized_linear`. `H` is the module's captured input
second-moment matrix, so this is the Hessian-weighted relative output error of the quantization —
the same objective GPTQ-family methods minimise, normalised per module.

What it is **not** is a curve. One conversion gives one point per module, at one width. Two things
follow, and both were checked rather than assumed:

- **Nothing on disk held a curve.** The only surviving converter stdout in this project is
  `convert-ctx.log`, 519 single-width points from the context build. The four conversion working
  directories (`wd-hyd`, `wd-ctx`, `wd-k4`, `wd-v2`, 151 GB) held resume state and per-layer
  quantized tensors — no Hessians, no error records.
- **The repository's own extractor was broken.** `tools/ladder_from_log.py` matched only the
  `rmse` label, which the converter prints for *uncalibrated fallback* modules; all 400 calibrated
  body modules print `proxy_err` and were silently dropped. It extracted 118 modules — the vision
  tower and the MTP draft, i.e. exactly the modules an allocator has no use for. Fixed here; the
  same log now yields 519.

There **is** an upstream error-driven path, and it is not this one: `util/measure.py` →
`conversion/measure_model.py` measures per-*group* `dkld`/`dbits` by swapping module groups between
two or more already-converted whole models against a reference, and `util/optimize.py` spends a bit
budget greedily on that ratio before splicing a checkpoint together with
`VariantSafetensorsCollection`. Its signal is real KLD rather than a proxy, which is better; its
price is N full conversions of a 27 B model plus a splice-recompile, and its allocator is a greedy
`dkld/dbits` walk with an ad-hoc `adjust(dkld) = -(-dkld)**0.69` fudge for negative deltas.

## 2. One pass, five widths

`tools/ladder_pass.py` wraps the converter's own quantization entry points. For every budgeted
body module it

1. copies the unquantized weight and computes `tr(WᵗHW)/count` from the **raw** accumulated
   Hessian, before `finalize_capture_H` mutates `H` in place;
2. runs the converter's unmodified pass at the recipe width, so state propagation and the emitted
   checkpoint are exactly the hydrated build's, and records that width's error by wrapping the
   print;
3. re-quantizes the saved weight at every other candidate width against the **same finalized
   Hessian**, through `quantize_exl3_batch` — the same batched entry point the converter uses.

Reusing the Hessian is the whole trick: the ladder costs ~5× a conversion's quantization time
rather than 5 conversions. Measured: **2 h 04 for 409 modules and 2,000 module-quantizations**,
against ~35 min for a plain conversion. Widths are `{3,4,5,6,7}` for modules of ≥52 M parameters
and `{4,5,6,7,8}` below that — five rungs either way, K3 where a bit is worth 700 MB across a role
and K8 where a bit is nearly free.

The measurement's standing limitation, which a reader must be able to check: **every rung is
measured at fixed propagation.** One Hessian per module, captured with the hydrated recipe's
upstream widths, re-used for all five widths. So a rung is the module-local error of changing one
module's width, not the sequential effect of changing many at once, which would move every
downstream Hessian. §5 and §6 test that assumption instead of asserting it.

## 3. The law

### 3.1 One universal shape

Fitting `log eps(m,K) = a_m + s(K)` over all 400 body modules, with `s(5)` pinned to 0:

| K | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| `s(K)` | +2.6834 | +1.3327 | 0 | −1.3150 | −2.6089 | −3.8783 |
| implied ratio `eps(K−1)/eps(K)` | — | 3.8602 | 3.7913 | 3.7247 | 3.6470 | 3.5590 |

The per-bit ratio is **not** constant: it declines smoothly from 3.860 to 3.559, so each further
bit buys slightly less than the previous one. The decline is not noise and not a class artifact —
it appears in every class with very little scatter:

| rung pair | all classes | MLP (192 modules) | `linear_attention` (144) |
|---|---|---|---|
| K3→K4 | 3.8580 ± 0.0137 | 3.8618 ± 0.0064 | 3.8548 ± 0.0108 |
| K4→K5 | 3.7914 ± 0.0234 | 3.8000 ± 0.0026 | 3.7856 ± 0.0261 |
| K5→K6 | 3.7249 ± 0.0402 | 3.7390 ± 0.0077 | 3.7152 ± 0.0454 |
| K6→K7 | 3.6475 ± 0.0594 | 3.6677 ± 0.0111 | 3.6352 ± 0.0689 |
| K7→K8 | 3.5734 ± 0.0719 | — | 3.5650 ± 0.0782 |

A single geometric constant is the cruder summary of the same data: per-module fits give
**3.7294 ± 0.0536** over 400 modules, with the ratio varying by only 5.3 % (median) *within* a
module across five widths.

### 3.2 Scope

Per-class fitted constants, and the depth check:

| class | n | mean r | sd | min | max |
|---|---:|---:|---:|---:|---:|
| `mlp_down_proj` | 64 | 3.7674 | 0.0034 | 3.7589 | 3.7754 |
| `mlp_gate_proj` | 64 | 3.7669 | 0.0045 | 3.7541 | 3.7778 |
| `mlp_up_proj` | 64 | 3.7667 | 0.0046 | 3.7521 | 3.7785 |
| `linear_attn.in_proj_qkv` | 48 | 3.7508 | 0.0346 | 3.5261 | 3.7685 |
| `linear_attn.out_proj` | 48 | 3.6835 | 0.0572 | 3.3294 | 3.7194 |
| `linear_attn.in_proj_z` | 48 | 3.6614 | 0.0445 | 3.5402 | 3.7135 |
| `self_attn.v_proj` | 16 | 3.7027 | 0.0138 | 3.6628 | 3.7164 |
| `self_attn.k_proj` | 16 | 3.6947 | 0.0210 | 3.6518 | 3.7184 |
| `self_attn.o_proj` | 16 | 3.6776 | 0.0297 | 3.6137 | 3.7103 |
| `self_attn.q_proj` | 16 | 3.6682 | 0.0668 | 3.4940 | 3.7639 |
| **all** | **400** | **3.7294** | **0.0536** | 3.3294 | 3.7785 |

The three MLP classes are indistinguishable from each other and the tightest of all (sd 0.004);
attention projections sit slightly lower. By depth the constant is flat — 3.7272, 3.7365, 3.7348,
3.7293, 3.7254, 3.7302, 3.7225, 3.7291 across the eight layer-octets — so there is no "late layers
behave differently" effect in the *shape*, only in the constant `a_m`.

Two exclusions, stated because the law is not measured for them: **`lm_head` and the eight MTP
draft modules have one rung each** (they are recipe-pinned by `-hb`/`-mb`, so the pass does not
ladder them), and the vision tower is unquantized at `-vb 16`. The MTP draft additionally
quantizes through the uncalibrated fallback path, which reports an `rmse` and not this proxy at
all.

### 3.3 Does it predict a width it has not seen?

A law that only interpolates its own fit is worth little. Two held-out tests, both over the
measured five-rung ladder:

| test | median | mean abs | p95 abs | max abs |
|---|---:|---:|---:|---:|
| leave one rung out, refit `a_m` and `r` on the other four | +1.14 % | 2.85 % | 5.46 % | 12.5 % |
| **one anchor rung + the universal shape**, predict another | +0.00 % | **1.175 %** | 3.86 % | 47.1 % |
| one anchor rung + the single constant 3.7294 | +0.00 % | 3.03 % | 6.88 % | 55.3 % |

The shape beats the constant by 2.6× on mean absolute error and, unlike the constant, carries no
systematic bias per rung (the constant under-predicts K3 by 4.2 % and K8 by 4.7 % while
over-predicting K5/K6 by ~1.8 %, which is exactly the declining ratio showing up as curvature).

The max column is honest about where it fails: the worst cases are all **long extrapolations** —
K8→K4 and K7→K3, four and five bits from the anchor. Only 3 of 400 modules exceed 20 % error, and
the worst is `layers.1.linear_attn.out_proj` at 47 % predicting K4 from K8. Adjacent-width
prediction, which is what an allocator actually needs, is the ~1 % case.

**Out-of-sample, on modules the ladder never measured (2026-08-16).** Everything above is held out
*within* the five-rung ladder. The S16-V conversion supplied the first test outside it. The ladder
has **no K3 rung at all** for the 96 body modules below the 52M-parameter big-module threshold, so
the S16-V pre-registration extrapolated them as `eps(3) = eps(4)·eps(4)/eps(5)` and committed that
before converting. The converter's **realised** K3 proxy errors match that extrapolation at a
**median ratio of 1.0164 and a maximum of 1.0455**, while the 304 modules that *do* have a measured
rung match theirs at median 0.9957 (min 0.9731, max 1.0093). So the per-bit shape holds one rung
below its fit, on modules where it was never measured, to within about 4.5 %
([`sixteen-flip-kld.json`](../receipts/sixteen-flip-kld.json) → `ladder_extension`). It came free:
the conversion was run to answer the 16 GB question, not to test this law.

### 3.4 Why 3.73 and not 4

A pure information argument — one extra bit halves the quantization step, quartering a squared
error — gives 4.0 per bit. Nothing measured here reaches it: the best rung pair is 3.86 and the
deficit widens monotonically with width, to 3.56 by K7→K8. The deficit is stable across classes
and flat across depth, so it is a property of the quantizer, not of this model's layers.

**We do not know why.** Trellis coding with a fixed codebook does not have to achieve the ideal
rate-distortion slope, and the regularization the converter applies before quantizing (global
scale search, sign flips, Hadamard rotation) is width-dependent through the scale search, which is
one plausible mechanism among several. This document records the constant and its stability and
declines to explain it.

## 4. From a curve to an allocation

### 4.1 The objective problem, and how it was decided

`proxy_err` is *relative*, so summing it across modules needs an assumption about per-module
sensitivity. Two candidates:

- **`rel`** — `sum_m eps(m,K)`. Every module's relative output perturbation counts the same. This
  is the assumption behind uniform-bitrate quantization.
- **`abs`** — `sum_m out_energy_m * eps(m,K)`, where `out_energy_m = tr(WᵗHW)/count` is the
  module's mean per-row calibration output energy, so the product is absolute output error energy.
  This is the GPTQ objective under isotropic downstream sensitivity.

Neither is a law, and `out_energy` spans 26,000× across roles, so the choice is not cosmetic. It
was made **before** the conversion, by scoring both against two *measured* KLD deltas between
published checkpoints that differ in exactly one role group:

| measured delta | what changes | KLD delta | `rel` implied scale | `abs` implied scale |
|---|---|---:|---:|---:|
| hydrated → context | attention K6 → K5, everything else identical | +0.000709 | 0.017425 | 2.126e−6 |
| K5/K6 → K4 | MLP K5/K5/K6 → K4, identical online-K6 attention and K6 head | +0.007204 | 0.026340 | 5.353e−6 |
| | | **ratio across the two** | **1.512** | 2.518 |

A usable objective must imply the *same* error-to-KLD scale on both, since the two deltas move
different role groups — this is exactly a test of cross-role weighting. `rel` is 1.5× inconsistent,
`abs` 2.5×, so `rel` was selected by the pre-registered rule and its geometric-mean scale
(0.021423 KLD per objective unit) is what converts a solved objective delta into a predicted KLD.
`abs` is recorded rather than discarded.

The MTP draft is not in the scored forward pass, so the K5/K6→K4 pair is a clean MLP-only delta
despite those two builds differing in MTP treatment.

### 4.2 The bytes, and an exact solve

The byte law is [34](34-vram-class-profiles.md) §2 applied per module,
`bytes = 2*(in+out) + 4 + numel*K/8` plus each role's BF16/F16 companions, and
`tools/allocate_bits.py` refuses to run unless it reproduces **both** published manifests
byte-for-byte first — all ten roles, hydrated 21,586,964,548 B and context 20,672,081,988 B. A byte
model that cannot reproduce the two shipped checkpoints has no business proposing a third.

That is byte *budget*, not bytes. A conversion reproduces the manifest exactly and the shard
contents not at all: re-running the published hydrated recipe on the same box returns identical
per-role totals, identical widths and identical safetensors headers while 399 of the 409 quantized
modules differ inside their `.trellis` payloads, because the converter is nondeterministic
([`receipts/converter-determinism.json`](../receipts/converter-determinism.json)). The byte model
is validated against the composition a recipe fixes, which is exactly the part that reproduces.

Every module's byte cost is an integer multiple of 655,360 B (the smallest module's `params/8`), so
the budget is an exact grid of 25,664 points and the allocation is solved by **dynamic programming
rather than greedily** — the optimum is provable. It runs in about a second.

### 4.3 The closed form, and why the ladder pass is not needed again

With `log eps = a_m + s(K)`, equalising marginal error per byte gives a closed form: sort modules
by `log(c_m / numel_m)` and cut at the byte budget. The DP does not merely approximate it, it
reproduces it — score ranges per assigned width come out monotone and essentially disjoint
(K4 −16.57..−15.77, K5 −15.63..−14.76, K6 −14.79..−13.78, K7 −13.77..−12.86).

So the input an allocator needs is one rung per module, which is what an ordinary conversion prints:

| allocation solved from | agreement with the five-rung DP | objective improvement recovered | bytes |
|---|---|---|---|
| the five-rung ladder (2 h 04) | — | 100 % by definition | 21,586,964,548 |
| one K5 rung + shape | 396 / 400 modules | 99.98 % | 21,586,964,548 |
| one K5 rung + constant 3.7294 | 395 / 400 | 100.0 % | 21,586,964,548 |
| **`convert-ctx.log` + shape** — a log from a *different* recipe that existed before this work | **396 / 400** | **100.0 %** | 21,586,964,548 |

Every disagreement is one bit at a threshold boundary. `tools/allocate_bits.py --ladder-from-log
<log>` is that path, and it is the one to use:

```bash
tools/allocate_bits.py --ladder-from-log /work/kld3/convert-ctx.log \
    --out plan.json --fixed-out fixed.json --override-out override.json
```

~25 min of GPU for the conversion whose log you already keep, against ~2 h for the measurement
pass. The pass earned the law; it does not need to be paid for twice.

## 5. What the ladder says about the hand-designed recipe

The shipped recipe puts `mlp_down_proj` at K6 while gate/up sit at K5. The
original justification compared down at K6 with gate/up at K5, which confounded
tensor role and width. At equal width over all 64 layers:

| width | `gate_proj` | `up_proj` | `down_proj` | down/gate | down/up |
|---|---:|---:|---:|---:|---:|
| K4 | 1.2442e−03 | 1.7452e−03 | 2.2276e−03 | 1.790 | 1.276 |
| K5 | 3.2744e−04 | 4.5927e−04 | 5.8620e−04 | 1.790 | 1.276 |
| K6 | 8.7561e−05 | 1.2284e−04 | 1.5679e−04 | 1.791 | 1.276 |
| K7 | 2.3880e−05 | 3.3493e−05 | 4.2748e−05 | 1.790 | 1.276 |

`down_proj` has the largest equal-width **proxy** error. But §7 shows that this
proxy is not monotone with KLD across role reallocations, so the table does not
prove K6-down is fidelity-optimal. The shipped recipe's measured quality stands;
isolating the role choice would require a direct byte-controlled ablation.

## 6. The candidate

Pre-registered before the conversion ran, in `receipts/error-driven-allocation.json` →
`pre_registered_plan` (the plan file's own digest and mtime are recorded against the conversion
log's): at the hydrated build's exact serialized-byte budget, with `lm_head` K6/mcg, the hydrated
MTP treatment, BF16 embeddings and vision tower, attention serialized, same source revision, same
`-cb mcg`, and 175 of 400 body modules moved off the hydrated widths:

| role | hydrated | solved |
|---|---|---|
| `full_attention` q/k/v/o | K6 × 64 | K4 × 5, K5 × 13, K6 × 17, K7 × 29 |
| `linear_attention` | K6 × 144 | K5 × 61, K6 × 70, K7 × 13 |
| `mlp_gate_proj` | K5 × 64 | K4 × 1, K5 × 51, K6 × 12 |
| `mlp_up_proj` | K5 × 64 | K4 × 1, K5 × 27, K6 × 36 |
| `mlp_down_proj` | K6 × 64 | K4 × 1, K5 × 3, K6 × 60 |
| `lm_head`, `mtp_draft`, embed, vision | unchanged | unchanged |

Average body width is 5.531 bits in both, necessarily: the byte budget and the parameter count are
identical, so this is a pure reallocation. The direction is "take bits off `q_proj` and
`in_proj_qkv`, put them on `up_proj`, `k_proj`, `v_proj` and `out_proj`".

The predicted effect was **−0.000211 mean KLD** (objective delta −0.009841 × the validated scale
0.021423), i.e. 0.002700 → 0.002489 on shard 0, against a paired resolution of ±6.4e−5 measured on
the existing hydrated-vs-context pair. That is the number the measurement was allowed to refute.

## 7. Outcome: the candidate loses, and the objective is why

`receipts/error-driven-allocation.json`. Shard 0 of the v5 suite, 512 contexts, 1,048,064 scored
positions, replayed through the shared BF16 head against `/work/gguf/hidden-bf16`:

| build | mean KLD | median | p99.9 | exact max | top-1 |
|---|---:|---:|---:|---:|---:|
| hydrated, published | 0.002699883159684943 | 0.001090403 | 0.131263 | 3.734847 | 97.797 % |
| hydrated, re-captured for this comparison | **0.002699883159684943** | 0.001090403 | 0.131263 | 3.734847 | 97.797 % |
| error-driven allocation | **0.003066179635178366** | 0.001341795 | 0.137506 | 5.115300 | 97.506 % |

**Paired, hydrated − error-driven: −0.00036630 [−0.00039779, −0.00033477]**, 512 contexts, 330
source clusters. The interval excludes zero *in favour of hydrated*, which wins **470 of 512**
contexts to the candidate's 42. Realised serialized bytes are **21,586,964,548 — equal to the
hydrated payload to the byte**, and every one of the ten per-role realised figures equals the
pre-registered prediction exactly, so this is a pure reallocation with no budget excuse.

Three controls close the obvious escapes before anyone reaches for them:

- **The reference was not the problem.** The published shard-0 hydrated report was replayed against
  a BF16 capture that has since been deleted, so hydrated was re-captured and re-replayed against
  the surviving one with the current harness. It came out **bit-identical** — same mean to 16
  digits, same median, same p99.9, same exact max, same bootstrap block — and the paired report
  between published and re-captured is 0.0 with a [0.0, 0.0] interval. (That control reads
  `a_wins 0, b_wins 512` because `fidelity.py paired` counts a tie as a b-win; it is 512 ties, not a
  sweep. Corroborates `receipts/capture-determinism.json`, from a fresh capture made *after* that
  receipt's claim.)
- **The solver was not the problem.** The realised widths match the solved map for all 400 body
  modules, and the byte model reproduced both published manifests before it was allowed to solve.
- **Propagation was not the problem.** The ladder is measured at fixed propagation, so the obvious
  explanation is accumulation the proxy cannot see. It is not: the proxy error the real conversion
  achieved at the solved widths, under the genuinely changed propagation, matched the fixed-
  propagation ladder at a median ratio of **0.9997**, mean 1.000 in every class and every layer
  octet. The propagation term is −0.000028 of a −0.009841 objective delta: **0.3 %**.

So the machinery did exactly what it was built to do. **Total equal-weighted relative proxy error
fell 13.1 %** — 0.075355 → 0.065486 over the 400 body modules — **and mean KLD rose 13.6 %.** The
defect is the objective: `sum_m eps(m,K)` is not a monotone surrogate for KLD, and minimising it
moved fidelity backwards.

The pre-registered selection rule could not have caught this, and that is the transferable lesson.
Both validation deltas were *uniform* role-group moves; neither tested a reallocation **between**
roles, which is the only thing the optimizer does. `receipts/error-driven-surrogate-calibration.json`
scores four weightings against all four measured width-only pairs — the two uniform moves plus both
reallocations — each with one free scale parameter:

| rule | sign right on all 4 pairs | implied-scale spread | worst leave-one-pair-out ratio |
|---|---|---:|---:|
| `rel`, w = 1 (the rule that was used) | **no** — wrong sign on the reallocation | changes sign | −0.69× |
| `numel`, w = parameters | **no** — wrong sign on the reallocation | changes sign | −0.87× |
| `abs`, w = `out_energy` | yes | 13.3× | 4.40× |
| `sqrt_energy`, w = √`out_energy` | yes | **2.51×** | 2.47× |

`sqrt_energy` is the least-bad and the only rule inside a factor of ~2.5 everywhere, but a rule that
is a factor of two-and-a-half uncertain cannot decide a 1e-4 fidelity claim, which is the decision
it would have to support. It was also chosen with knowledge of this run's answer, so it is a
pre-registrable candidate and not a result; if it is tried, its validation must include a
between-role reallocation delta, and this run is now that third calibration point.

One more number bounds how much of fidelity a width rule can even address: hydrated versus the
published online-K5/K6 build is **+0.000441 [+0.000412, +0.000474]** at *identical widths* — pure
mechanism, offline calibrated K6 against the runtime's uncalibrated online K6. Any width-based
surrogate predicts exactly zero there.

**We have no calibrated surrogate for end-to-end fidelity yet, so the hand-designed 5/5/6 MLP split
with K6 attention stands — now not because nothing better was tried, but because the best available
alternative was built, measured, and lost by 0.000366 [0.000335, 0.000398] at byte-identical size.**
The checkpoint is published as a documented negative at
[`malaiwah/Qwen3.8-27B-EXL3-EDA-research`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-EDA-research);
the ladder it was built from is `receipts/error-driven-ladder.json`, and §3 is the part worth
reusing.

## 8. What was published, and what was released

`receipts/error-driven-publication.json` carries every revision and digest; the short version:

| artifact | destination | revision | files | bytes |
|---|---|---|---:|---:|
| the checkpoint, its card, the two width maps, the pre-registered plan and the full conversion log | model repo `malaiwah/Qwen3.8-27B-EXL3-EDA-research` | `bbbe6a7e` | 27 | 21,611,155,800 |
| the error-driven shard-0 capture | dataset `…-fidelity-suite-v5`, `captures/shard-0000/hidden-eda` | `99b7468d` | 513 | 10,732,324,101 |
| the matched hydrated re-capture | dataset, `captures/shard-0000/hidden-hyd-rematch` | `17777e7e` | 513 | 10,732,324,091 |
| the five-rung ladder | dataset, `captures/shard-0000/error-driven-ladder.json` | `27625702` | 1 | 284,807 |

Every upload was re-verified against the local bytes before the local copy was released — LFS
sha256 oids for the large files, CDN readback and re-hash for the rest and for a spread sample of
the large ones — and all four reported an empty problem list. The three local trees (21.6 GB of
checkpoint and 2 × 10.5 GB of capture) were then deleted; their recompute costs are ~50 min of GPU
for the conversion (~25 min via `--ladder-from-log`) and ~6 min per capture, and every one of them
is now a download rather than a GPU bill.

The pair of captures is the useful part for a reader: with the shard-0 BF16 reference already in
the same dataset, `fidelity.py replay` reproduces the headline paired interval with **no GPU, no
checkpoint and no conversion**.
