# Fold FP32 chunk budget: 96 → 48 MB is +5.7% PP for zero fidelity cost

**Date:** 2026-08-19
**Axis:** PP on the fidelity profile
**Result:** PP **1857.7 ± 3.7 → 1966.8 ± 8.3** (+5.9%), TG-fox 210.5 → 211.1,
TG-essay 90.0, gate **9/9 exit 0**, and **bit-identical output** so criteria 3/4
are untouched by construction.

## Why this knob suddenly mattered

`VLLM_EXL3_FOLD_FP32_BUDGET_MB` sizes the FP32 working set of
`hadamard_fold_weight_chunked`. It was introduced to *reduce launch count* on the
per-call trellis prefill path, and at the time it produced **no measurable PP
change** — correctly, because prefill was 92% GPU-bound and the fold was running
on cached weights (`receipts/prefill-profile-2026-08-19.md`).

Enabling reconstruct→fold→hgemm with `CACHE=0`
(`receipts/recon-hgemm-fidelity-2026-08-19.md`) put the fold on the **per-chunk
hot path** for the two largest matrices in the model. That changed it from a
launch-count curiosity into a throughput parameter, so it was swept.

## Measured curve (2051-token bench, median of 5, single boot each)

| budget (MB) | PP |
|---|---|
| 24 | 1955.0 |
| 32 | 1960.4 |
| **48** | **1964.4** |
| 64 | 1864.7 |
| 96 (previous default) | 1857.7 |
| 192 | 1789.9 |
| 384 | 1711.0 |

Monotonically worse above the peak, with a **sharp 5.3% cliff between 48 and
64 MB** — consistent with the FP32 working set falling out of L2 (the 5090 has
128 MB of L2; three live FP32 tensors of a 64 MB budget is ~192 MB of traffic per
pass versus ~144 MB at 48 MB). Below the peak it flattens and then slowly
degrades as launch count grows again, which is the effect the parameter was
originally added to control. So the curve has two competing terms and 48 MB sits
at their crossover.

Bigger is not better here, which is the opposite of the intuition that motivated
the parameter.

## Fidelity: bit-identical, verified rather than argued

The fold works on independent 128×128 blocks, so grouping *should* not change any
value — but a PP default was about to change, so it was checked directly on real
`gate_proj` weights (`tools/fold-budget-bitidentity.py`):

| budget vs 96 MB | bit-identical | max abs diff |
|---|---|---|
| 24 MB | **True** | 0.000e+00 |
| 32 MB | **True** | 0.000e+00 |
| 48 MB | **True** | 0.000e+00 |
| 64 MB | **True** | 0.000e+00 |

Exactly zero difference, so no KLD re-run is required and criteria 3/4 keep the
values measured for the reconstruct→hgemm configuration (0.003437, p99 0.035204).
This is the rare case where a throughput knob is provably free.

## Scope

Set in the **fidelity profile only**. The throughput profile still resolves to
96 MB (verified by reading the effective value back out of the launcher), because
there the fold runs at load time during FP4 conversion rather than per chunk, so
the knob would trade a load-time cost for nothing.

## Verification

- n=3 harness: `receipts/bench-fidelity-fold48-2026-08-19.json` —
  PP 1966.8 ± 8.3, TG-fox 211.1 ± 1.0 [acc 1.000], TG-essay 90.0 ± 0.2 [acc 0.281].
- Gate: `receipts/verify-fidelity-fold48-2026-08-19.json` — 9/9 PASS, exit 0,
  including the 200k prompt and the vision fixture.
- Bit-identity: table above.
