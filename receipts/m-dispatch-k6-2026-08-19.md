# Route K6 by row count: B12X for prefill, fused exl3_gemm for decode

**Date:** 2026-08-19
**Axes:** TG-essay and MTP acceptance, at unchanged PP
**Result:** TG-essay **90.0 ± 0.2 → 93.1 ± 0.1** (+3.4%), MTP acceptance
**0.281 → 0.304** (+8.2%), PP 1966.8 ± 8.3 → 1965.4 ± 2.1 (unchanged),
TG-fox 211.1 ± 1.0 → 207.7 ± 0.1 (−1.6%, still 9.3% clear of its 190 threshold).
Gate **9/9 exit 0**.

## How this was found

While confirming that B12X is the right choice for the K6 matrices, the control
run — B12X disabled entirely, everything on reconstruct→hgemm — lost prefill
badly (PP 1504.3 vs 1966.8, so B12X stays) but came back **better at decode**:

| | B12X for K6 (all M) | no B12X |
|---|---|---|
| PP | **1966.8 ± 8.3** | 1504.3 |
| TG-essay | 90.0 | **93.3** |
| MTP acceptance (essay) | 0.281 | **0.304** |

`_b12x_trellis_k6_supported` takes no row count, so B12X was serving *both*
phases and dragging decode down with it. Splitting the dispatch by `m` captures
both wins. New knob `VLLM_EXL3_B12X_MIN_M` (default **0** = previous behaviour;
the fidelity profile sets **128**).

## Measured (n=3 boots, `receipts/bench-fidelity-minm128-2026-08-19.json`)

| metric | MIN_M=0 | MIN_M=128 | delta |
|---|---|---|---|
| PP, 2051-tok | 1966.8 ± 8.3 | 1965.4 ± 2.1 | −0.1% (tie) |
| TG-essay | 90.0 ± 0.2 [acc 0.281] | **93.1 ± 0.1** [acc **0.304**] | **+3.4% / +8.2%** |
| TG-fox | 211.1 ± 1.0 [acc 1.000] | 207.7 ± 0.1 [acc 1.000] | −1.6% |
| max context | 238,400 | 238,400 | — |

## The tradeoff, stated plainly

This is **not** a free win: essay gains 3.1 tok/s and fox loses 3.4 tok/s, so in
raw tok/s it is roughly a wash. It was adopted because:

- **Acceptance improves 8.2%**, i.e. less wasted speculative compute per accepted
  token, which is a structural gain rather than a benchmark artifact.
- The **essay prompt (500 tokens) is more representative** of real generation than
  the 200-token fox prompt.
- Both TG criteria keep comfortable margin: essay 93.1 vs 83 (+12.2%), fox 207.7
  vs 190 (**+9.3%**).
- PP is statistically unchanged, and the n=3 spread actually tightens
  (±2.1 vs ±8.3).

`VLLM_EXL3_B12X_MIN_M=0` flips it back in one env var if fox headroom is ever
preferred, which is the "known path back" the objective requires.

## Fidelity: unchanged, with the reason stated

No KLD re-run was performed, and that is a reasoned choice rather than an
omission:

1. The KLD capture runs at **prefill row counts**, which still take the *same*
   B12X path as before this change — the capture is bit-for-bit unaffected.
2. The two routes were already proven to agree on the real served tensors by
   `VLLM_EXL3_B12X_SELFTEST` at **both** m=4 and m=3072
   (cos ≥ 0.999999, max_rel ≤ 1.14e-03) — see
   `receipts/b12x-k5-parked-2026-08-19.md`.

Worth noting honestly: acceptance moving 0.281 → 0.304 proves the decode outputs
*do* differ observably, because a ~1e-3 relative difference is amplified by the
discrete draft/target comparison. So this is not "identical output", it is "both
routes within measured agreement, and the fidelity metric measures a phase this
change does not touch". Criteria 3 and 4 therefore stand at 0.003437 / p99
0.035204.

## Verification

- n=3 harness numbers above, acceptance reported with each TG figure.
- `tools/verify-profile.sh` → **9/9 PASS, exit 0**
  (`receipts/verify-fidelity-minm128-2026-08-19.json`), including the 200k prompt
  and the vision fixture.
- Throughput profile resolves to `min_m=0`, read back out of the launcher.
