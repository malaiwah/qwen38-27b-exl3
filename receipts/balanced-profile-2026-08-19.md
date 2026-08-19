# `PROFILE=balanced`: a second 5/6 profile that fails a *different* criterion

**Date:** 2026-08-19
**Config:** all-trellis kernel stack + `mlp.gate_up_proj` in MXFP6
**Standing:** **5/6** — passes criteria 2, 3, 4, 6; fails **5** (context 199,104)
and **1** (PP 3266). `PROFILE=fidelity` is also 5/6 but fails **1** only.

## Measured (n=3 boots, `receipts/bench-balanced-2026-08-19.json`)

| | `fidelity` | **`balanced`** | `throughput` |
|---|---|---|---|
| PP, 2051-tok | 1965.4 ± 2.1 | **3266.3 ± 13.9** | 7694.9 ± 21.8 |
| TG-fox | 207.7 ± 0.1 [acc 1.000] | 203.3 ± 0.5 [acc 1.000] | 185.0 ± 0.6 [acc 0.930] |
| TG-essay | 93.1 ± 0.1 [acc 0.304] | **95.7 ± 0.2** [acc **0.324**] | 93.3 ± 0.1 [acc 0.298] |
| KLD mean | **0.003437** | 0.005672 [0.005302, 0.006087] | 0.063759 |
| KLD p95 / p99 | 0.010343 / **0.035204** | 0.017638 / 0.059908 | — / 0.7010 |
| max context | **238,400** | 199,104 | 250,000 |
| criteria met | 5/6 (fails PP) | **5/6 (fails ctx)** | 4/6 |

`balanced` buys **1.66× the prefill of `fidelity`** and posts the **best TG-essay
and the best MTP acceptance measured anywhere in this project** (95.7, 0.324),
while keeping KLD **2.1× inside** the 0.012 budget and p99 **2.5× inside** 0.15.
Gate: **9/9 PASS, exit 0** (`receipts/verify-balanced-2026-08-19.json`).

## What it costs, precisely

Converting `mlp.gate_up_proj` to MXFP6 makes the largest matrices in the model
GEMM-resident instead of trellis-decoded per chunk. MXFP6 for those 11.4e9
parameters is **~2.4 GiB larger** than their trellis form, which drops available
KV from 9.29 → 7.60 GiB and the context ceiling from 238,400 → **199,104**. At
238,400 the engine refuses to start with a clean KV-sizing `ValueError` naming
199,104 as the estimated maximum — so the failure is loud, not silent.

That is the entire tradeoff: **~17% of context for +66% prefill and +2.8%
TG-essay**, at 1.65× the KLD of `fidelity` (still far inside budget).

## Third validation of the additivity model — and it holds here

Predicted before measuring: floor 0.003437 + `gate_up` FP6 contribution
(0.025463 × 0.102) = **0.006034**. Measured **0.005672**, ci95
[0.005302, 0.006087] → **−6.0%**, inside the CI.

Running tally of held-out predictions:

| config | predicted | measured | error | verdict |
|---|---|---|---|---|
| `gate_up`+`down` FP4 | 0.042845 | 0.042044 | −1.9% | holds |
| **`gate_up` FP6** | **0.006034** | **0.005672** | **−6.0%** | **holds** |
| `self_attn` FP4 | 0.007903 | 0.011534 | **+46%** | **fails** |

This sharpens the scope claim rather than just repeating it: additivity is
reliable for **per-position MLP error in either FP4 or FP6**, and unreliable for
**attention**, whose error compounds through the KV cache. Two independent MLP
confirmations now bracket the one attention failure.

## Why it is offered, not made default

It fails a north-star criterion (context ≥ 238,400), so promoting it would be
silently choosing a criterion to drop — which the objective forbids. It is added
as a **third selectable profile** with its own gated baseline
(`tools/baseline-balanced.json`, whose `max_model_len` floor is honestly set to
199,104 with the shortfall documented in the metric description), so the choice
is one env var away and fully measured:

```
PROFILE=balanced ./run-qwen38-27b.sh
```

`PROFILE=bogus` still exits 4, and `throughput`/`fidelity` resolve unchanged
(verified by reading effective values back out of the launcher).

## Decision-relevant summary

The frontier is no longer a two-way choice. Three measured operating points:

- **`fidelity`** — best fidelity (0.003437) and full 238,400 context; prefill 1965.
- **`balanced`** — 1.66× prefill and best decode, KLD 0.005672; context 199,104.
- **`throughput`** — only profile ≥ 7000 prefill; KLD 0.063759.

No profile reaches 6/6, and the reason is unchanged and now measured from four
directions (`receipts/frontier-2026-08-19.md`,
`receipts/selfattn-fp4-additivity-failure-2026-08-19.md`).
