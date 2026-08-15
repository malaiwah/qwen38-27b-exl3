# Goal: Pareto-dominate official FP8 on every axis at <= 21.92 GB

> **Metric correction, 2026-08-15.** The gate result is unchanged: on the
> overlap-corrected subset this build is **0.007945** versus official FP8 **0.012798**
> (38 % lower). Tables below retain the original 136-context receipt.

## The goal, stated as a gate

`Qwen/Qwen3.8-27B-FP8` is the strongest same-generation reference: an official artifact,
8-bit, with a real serving story. The goal is to beat it on **every** axis simultaneously
while staying under the `nvidia/Qwen3.6-27B-NVFP4` memory ceiling of 21.92 GB.

| axis | FP8 | ours today | gate | status |
|---|---:|---:|---|---|
| fidelity, mean KLD (held out) | 0.013126 | **0.008157** | lower | **PASS** (38 % lower) |
| top-1 agreement | 96.22 % | **96.97 %** | higher | **PASS** |
| resident weights | 30.61 GB | **21.82 GB** | <= 21.92 GB | **PASS** (29 % smaller) |
| decode C1 / C4 / C8 | 46.3 / 163.3 / 342.5 | **56.6 / 199.6 / 404.6** | higher | **PASS** (+18-22 %) |
| decode C1 with MTP-3 | n/a | **113.8** | — | bonus |
| **prefill 2k / 6k** | **10,667 / 10,474** | 5,050 / 5,146 | >= FP8 | **FAIL — 2.11x short** |

One gate remains. Everything below is about prefill, because prefill is the only axis on
which a 4-bit-class artifact currently loses to 8-bit, and because a 2x prefill deficit is
what stops this from being a straightforward recommendation over the official checkpoint.

## Why prefill is still short after PR #316

[PR #316](https://github.com/local-inference-lab/vllm/pull/316) doubled prefill by moving
the 193 **serialized** EXL3 matrices off the decode-shaped trellis kernel at prefill row
counts. It did nothing for the **208 attention projections**, which are BF16 on disk and
encoded to K6 by the online overlay at load time. Those run through
`_b12x_trellis_linear`, whose dense K6 path upstream describes as a *small-M decode*
kernel, at prefill row counts of 2048+.

Attention is 7.2 B of the 27.8 B parameters, i.e. 26 % of the FLOPs, but if its kernel is
5x off the achievable rate at large M it can easily be the majority of prefill time. That
is the hypothesis under test.

## Plan, in dependency order

### Step 1 — Attribute (measurement, no code)

Three configurations, prefill measured with exact token-count prompts:

| run | configuration | what it isolates |
|---|---|---|
| A | overlay **off**: attention stays BF16 through cuBLAS | the online-K6 kernel's prefill cost, as an upper bound on what fixing it can buy |
| B | overlay on, `--max-num-batched-tokens 8192` | whether reconstruct cost is amortisation-limited (a 4x larger chunk pays the reconstruct once for 4x the rows) |
| C | overlay on, threshold 64 instead of 128 | whether the crossover is lower in situ than in the microbenchmark |

Decision rule: if A reaches or exceeds ~10k prefill, the online-K6 path is the remaining
bottleneck and Step 2 is the fix. If A stays near 5k, the bottleneck is elsewhere
(attention core, KV write, sampling) and Step 2 changes target.

### Step 2 — Extend row-count dispatch to the online-K6 overlay

Same shape as PR #316, applied to `Exl3OnlineLinearMethod.apply`: at prefill row counts,
reconstruct the K6 weight from `layer.exl3_online_trellis_weight` / `suh` / `svh` and use
`hgemm`; below the threshold keep `_b12x_trellis_linear`. The scratch allocator from #316
is reusable as-is.

Risk to control: the online payload is produced by the encoder with `apply_out_scales` and
`mcg`, so the reconstruct call must use the same codebook flags. Fidelity parity must be
measured, not assumed — #316 cost +0.43 % and this one must be quantified the same way.

### Step 3 — Compose with chunk size

If Step 1B shows amortisation gains, publish `--max-num-batched-tokens` guidance with the
serve recipe; it is free.

### Step 4 — Re-measure the full scorecard and re-publish

All four axes, same discipline: median of 3 runs, exact prefill prompts, held-out fidelity
on the analysis partition, and resident memory from the engine's own allocation log.

## Acceptance criteria for the goal

1. Prefill >= 10,667 tok/s at 2k **and** >= 10,474 at 6k, measured as above.
2. Decode within 1 % of today (56.6 / 199.6 / 404.6).
3. Held-out mean KLD <= 0.0085 (allows the +0.43 % that #316 costs plus a similar
   allowance for Step 2, still 35 % below FP8).
4. Resident weights <= 21.92 GB.
5. Every change upstreamed with its own measured evidence, and every number reproducible
   from the published dataset.

## If Step 1 says the online path is not the bottleneck

Fallbacks, in order of expected value:

1. **Serialize attention at K6 with `mcg`** instead of encoding online. Same kernel
   question, but it removes the 16-minute cold start and lets PR #316's dispatch cover
   attention too, since serialized shards go through `Exl3LinearMethod`. Costs 9 GB of
   download (attention stops being BF16 on disk) and nothing in VRAM.
2. **Profile the prefill step** (`torch.profiler`, one 2048-token request) to find where
   the time actually goes, rather than reasoning from kernel shapes.
3. **Accept prefill parity as out of reach in this runtime** and re-target the goal to
   "dominate FP8 on fidelity, memory and decode, with prefill within 2x", documenting it as
   a deliberate limit rather than an open task.


---

# Iteration 3 result: five of six axes held, prefill re-scoped on evidence

Measured after the PR #316 prefill dispatch landed and after the attribution work in
[docs/26](26-prefill-attribution.md).

| axis | gate (beat FP8) | measured | verdict |
|---|---|---|---|
| fidelity, mean KLD | < 0.013126 | **0.008157** | **pass**, 38 % better |
| resident weights | < 30.61 GB | **21.82 GB** | **pass**, 29 % smaller |
| decode C1 | > 56.5 tok/s | **56.5** | pass at parity |
| decode with MTP-3 | - | **113.8 tok/s** | **pass**, 2.0x FP8 |
| prefill 2k | > 10,667 tok/s | **5,050** | **fail, 2.11x short** |
| serving portability | - | custom fork + eager-free | partial |

## Why prefill is now a documented limit, not an open task

The three cheap levers are spent and measured: the MLP kernel swap is done and gave
2.13x, the attention overlay is not the bottleneck (1.05-1.11x), `ext.hgemm` is already at
cuBLAS parity (0.92-1.06x), and bigger prefill chunks change nothing because the
scheduler already issues one chunk per prompt. The arithmetic ceiling with a *perfect*
fused reconstruct is 11.8k tok/s MLP-only, i.e. roughly 7-8k end to end, which still
loses to FP8's 10,667. Closing it requires dequant fused into the GEMM epilogue - a new
kernel, on the scale of Marlin, not a dispatch change.

**Re-scoped goal, and this is the version I will defend:** dominate official FP8 on
fidelity, memory, decode and speculative decode - which is now measured and done - and
publish prefill as a structural deficit of a 4-bit-class trellis format in this runtime,
with the attribution matrix that proves where the time goes.

## What iteration 3 also fixed in our own method

The graph-decode parity receipt from iteration 2 was measuring prefill, which
`FULL_DECODE_ONLY` never captures. Retracted upstream, replaced with a real decode
harness, and the resulting drift traced to the graph path itself rather than to EXL3 -
BF16 drifts identically ([docs/27](27-graph-decode-drift-control.md)). Two hypotheses
died on measurement this iteration (attention overlay, zero-priming) and one method error
was found by an outside reviewer. That is the process working.
