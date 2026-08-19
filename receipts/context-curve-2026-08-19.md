# Prefill scales as t(D) = aD + bD^2, and quantization only ever moved `a`

**Date:** 2026-08-19. Data: `receipts/context-curve/{fidelity,balanced,throughput}.json`.
Harness: `tools/bench-context-curve.py`. Runner: `/tmp/ctxcurve4.sh`.

## Measured

| depth | fidelity PP | balanced PP | throughput PP |
|---:|---:|---:|---:|
| 2,049 | 3016.4 | 3905.5 | 9085.3 |
| 8,193 | 2949.3 | 3814.5 | 8700.3 |
| 32,769 | 2538.6 | 3157.8 | 5935.0 |
| 65,537 | 2139.0 | 2566.6 | 4153.4 |
| 131,073 | 1631.0 | 1870.2 | 2598.9 |
| 199,105 | 1304.8 | — (ctx 199,104) | 1859.2 |
| 238,401 | — (ctx 238,400) | — | 1593.4 |

Fitting `1/PP = a + b*D` (exact linearisation of `PP = PP0/(1+D/Dc)`):

| profile | PP0 | Dc | max fit err | n |
|---|---:|---:|---:|---:|
| fidelity | 3,098 | 145,260 | 1.28% | 6 |
| balanced | 4,048 | 112,927 | 1.81% | 5 |
| throughput | 10,146 | 44,657 | 6.77% | 7 |

## The result: `b` is format-invariant

| profile | a = 1/PP0 (s/tok) | b = 1/(PP0*Dc) (s/tok^2) | b vs mean |
|---|---:|---:|---:|
| fidelity | 3.228e-04 | 2.222e-09 | +0.7% |
| balanced | 2.470e-04 | 2.187e-09 | -0.8% |
| throughput | 9.857e-05 | 2.207e-09 | +0.1% |

**`a` spans 3.27x across these three profiles. `b` is constant to 1.6% spread.**

Attention prefill runs in BF16/FP8 regardless of weight format, so no weight
quantization can touch the quadratic term. Every kernel and format win this
project has shipped — FP4/FP6, fold-48, recon→hgemm, the K5 cure, b12x
dispatch, the power-cap removal — moved **only `a`**.

This was predicted before the throughput arm was measured: if FP4 accelerates
only the linear term, `Dc` must fall by the same factor `PP0` rises.
Predicted `Dc(throughput) = Dc(fidelity)/3.24 = 44,833`; **fitted from 7 points
= 44,657, error -0.4%**. Separability confirmed: PP0 ratio 3.27x, Dc ratio 3.25x.

## Consequence: our benchmark chose our optimisation targets

Quadratic share of prefill time, from the fitted curves:

| depth | fidelity | balanced | throughput |
|---:|---:|---:|---:|
| 2,049 | 1.4% | 1.8% | 4.4% |
| 32,769 | 18.4% | 22.5% | 42.3% |
| 131,073 | 47.4% | 53.7% | 74.6% |
| 238,400 | 62.1% | 67.9% | **84.2%** |

The 2,051-token benchmark sits where attention is 1.4–4.4% of prefill. Amdahl
on the fidelity curve: a 2x faster attention kernel is worth **1.01x** at 2k and
**1.45x** at 238,400; a 2x faster GEMM is worth 1.97x at 2k and 1.27x at 199k.
The two levers pay off in opposite regimes, and we have only ever pulled one.

**The faster the profile, the more attention-dominated its deep regime.** At
238,400 the throughput profile spends 84.2% of prefill in the term that no
weight format can improve — which makes `throughput`, not `fidelity`, the right
target for a deep-context nsys trace.

## Method: three attempts, two invalid

1. **Difference two non-streaming requests** (`max_tokens=1` vs `1+N`).
   Invalid past a few thousand tokens: at 32k, prefill is ~13 s while 64 decode
   tokens is ~0.3 s, so prefill jitter swamps the signal. Reported **1,127.9
   tok/s** TG on a profile whose ceiling is ~228. Receipt preserved at
   `receipts/context-curve/fidelity-INVALID-tg-subtraction.json`.
2. **Client-side SSE streaming**, timestamping chunk arrivals. Invalid in the
   opposite direction: Python SSE parsing costs more per token than decode, so
   it reported **36.4 tok/s** where the harness measures 215.6. It also returned
   zero text chunks at depths >= 32k.
3. **Server-side Prometheus histograms** around a plain non-streaming request —
   `vllm:time_to_first_token_seconds` and
   `vllm:request_time_per_output_token_seconds`, sum/count deltas. Neither
   prefill nor the client is in the path. **Validated against a known value:
   TG at 2k = 228.8 vs the harness's 228.3 fox.**

Method 3 still has a gap: the per-output-token histogram delta is 0 at depths
>= 32k (the observation is not yet posted when the response returns), so TG at
depth is `n/a` except two incidental points (throughput 131,073 = 143.5,
balanced 8,193 = 210.0). A bounded retry on the scrape fixes it; deferred to a
separate TG-at-depth pass so that one pass = one protocol.

## Infrastructure defect found and fixed

Two earlier sweeps lost arms to `Container died with state=stopping` with a
clean container log. Cause: the systemd unit is `Restart=always` with
`ExecStop=podman stop -t 30 qwen38-27b` and
`ExecStopPost=-podman rm -f qwen38-27b`, **both keyed on the container name**.
A sweep that issued `systemctl stop` and then created a new container of the
same name a few seconds later had systemd's late `ExecStop` land on the *fresh*
container. `/tmp/ctxcurve4.sh` fixes it by polling until `ActiveState` is
`inactive|failed` **and** the name is free before launching.

Also: an empty `hub jobs` list is not proof a process exited. The first lost arm
was a genuine two-sweep race — `ctxcurve` was still running when `ctxcurve2`
started. Verify with `pgrep`, not the job table.
