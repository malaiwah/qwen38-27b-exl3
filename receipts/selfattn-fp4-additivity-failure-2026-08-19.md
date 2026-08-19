# The additivity model fails for `self_attn`, and no FP4 subset reliably passes criterion 3

**Date:** 2026-08-19
**Prediction:** KLD 0.007903 (floor 0.003437 + attributed `self_attn` 0.004491)
**Measured:** **0.011534**, ci95 **[0.010944, 0.012203]**, p95 0.038786, p99 0.115940
**Error: +46%.** The model was wrong, and in the unsafe direction.

## Why this test was run

The corrected attribution
(`receipts/kld-axis-conclusion-2026-08-19.md`, CORRECTION section) concluded that
`self_attn` — 7% of parameters, attributed contribution 0.004491 — was the **only**
FP4 group that fits the post-floor budget. That was arithmetic, not a measurement.
Since it was the last remaining candidate for a KLD-passing quantized config, and
since a held-out prediction is the only honest way to test an additive model, it
was measured: 512 contexts, shard-0000, BF16 reference, current fidelity settings.

## Result

| | predicted | measured |
|---|---|---|
| KLD mean | 0.007903 | **0.011534** |
| ci95 | — | [0.010944, **0.012203**] |
| p95 | — | 0.038786 |
| p99 | — | 0.115940 |

Two consequences, both material:

1. **Criterion 3 is not reliably met.** The point estimate 0.011534 sits under
   0.012, but the **95% CI upper bound is 0.012203, above the threshold**.
   Reporting this as a pass would be reporting a coin-flip as a certainty.
2. **Criterion 4 passes but its margin collapses**: p99 0.115940 versus 0.15,
   where all-trellis has 0.035204. FP4 on 7% of parameters costs 3.3× the tail.

## Where the model broke, and why

Measured `self_attn` contribution = 0.011534 − 0.003437 = **0.008097**, versus the
**0.004491** attributed from the all-FP4 decomposition — a factor of **1.80**.

The mechanism is specific and, in hindsight, predictable: the attribution derived
`self_attn` **by residual** inside a mixture where *every* group was already FP4.
Attention error does not stay local — it propagates through the **KV cache** and is
re-read at every subsequent position, so over 2047 scored positions it compounds.
When the MLP is also quantized, that compounding overlaps with (and is partly
masked by) MLP error, so `self_attn`'s *marginal* contribution in the all-FP4
mixture is much smaller than its *standalone* contribution. Extracting a
standalone estimate from a marginal one therefore under-predicts.

Scope of the additivity model, restated honestly:

| groups | additivity | evidence |
|---|---|---|
| MLP (`gate_up` + `down`) | **holds** | held-out prediction 0.042845 vs 0.042044 measured, **−1.9%** |
| `self_attn` | **fails** | held-out prediction 0.007903 vs 0.011534 measured, **+46%** |

The model is usable for per-position MLP error and **must not** be used for
attention. `receipts/kld-axis-conclusion-2026-08-19.md` predates this and its
CORRECTION section is superseded on exactly this point.

## The practical conclusion is stronger, not weaker

The earlier conclusion — "only `self_attn` fits the budget" — is now **overturned
by measurement**. Every FP4 group has been measured or bounded:

| FP4 group | KLD contribution | fits 0.008563 budget? |
|---|---|---|
| `self_attn` | **0.008097 measured** | **no** (total 0.011534, CI crosses 0.012) |
| `down_proj` | 0.013970 | no |
| GDN | 0.016423 | no |
| `gate_up_proj` | 0.025463 | no |

**No FP4 subset reliably passes criterion 3.** The FP4 side of the
criterion-1-versus-criterion-3 deadlock is now closed by direct measurement at
every candidate rather than by arithmetic on a model that has now been shown to
fail for attention.

## Not adopted

Beyond the fidelity risk, this configuration is worse on the axes it was supposed
to help:

| | all-trellis (shipped) | + `self_attn` FP4 |
|---|---|---|
| PP | 1965.4 ± 2.1 | 2043.0 (+4.0%) |
| TG-fox | **207.7** [acc **1.000**] | 199.5 [acc 0.967] |
| TG-essay | **93.1** [acc **0.304**] | 91.0 [acc 0.293] |
| KLD mean | **0.003437** | 0.011534 |
| KLD p99 | **0.035204** | 0.115940 |

+4.0% PP for 3.4× the KLD, 3.3× the p99 tail, and a fox margin cut from 9.3% to
5.0%. Rejected.

It is a **fourth independent confirmation** of the acceptance coupling: FP4 on
just 7% of parameters drops fox acceptance 1.000 → 0.967 and essay 0.304 → 0.293.
Draft agreement tracks weight fidelity every time it has been measured.

## Verification

`tools/kld-run-selfattn-fp4.sh` (identical to the all-trellis runner except
`VLLM_EXL3_FP4_LAYERS="self_attn."` and the current fidelity kernel settings),
512/513 contexts captured, replayed against the BF16 reference →
`/tmp/kld-data/reports/report-selfattn-fp4.json`. Serving numbers from
`tools/bench_lib.py` with acceptance reported alongside every TG figure.
