# TG-fox is capped by MTP draft acceptance, not by step time — and acceptance is a fidelity effect

**Date:** 2026-08-19
**Axis:** TG (criterion 2b, ≥190 tok/s on the fox prompt)
**Outcome:** the throughput profile's 185.0 ± 0.6 is a **plateau**, and the cause
is that FP4 weights draft the fox prompt at **0.930** acceptance where trellis
weights draft it at **1.000**. Criterion 2b is therefore not a decode-speed
problem; it is the KLD axis showing up on the TG axis.

## What was tried: MTP draft depth

`MTP` was swept on the throughput profile, everything else held (n=1 boot each,
3 reps per TG measurement):

| MTP depth | TG-fox | fox acceptance | TG-essay |
|---|---|---|---|
| 4 | 185.3 | 0.930 | 93.6 |
| **6 (shipped)** | **185.0 ± 0.6** (n=3) | **0.930 ± 0.000** | 93.3 ± 0.1 |
| 8 | 185.2 | 0.930 | 93.6 |
| 10 | 184.8 | 0.930 | 93.5 |

**Flat.** A 2.5× change in draft depth moves TG-fox by 0.3%, and acceptance does
not move at all. So the draft loop's cost scales with depth almost exactly as
fast as its benefit, and depth is not the limiter. Criterion 2b (190) is 2.7%
away and this lever cannot reach it.

## What the sweep accidentally exposed: a reporting error of mine

Acceptance came back as **0.930** on fox, while every receipt so far had reported
**0.298** next to fox numbers. Both are correct — for different prompts:

| profile | TG-fox | fox acc | TG-essay | essay acc |
|---|---|---|---|---|
| throughput (all-FP4) | 185.0 ± 0.6 | **0.930 ± 0.000** | 93.3 ± 0.1 | 0.298 ± 0.000 |
| fidelity (all-trellis) | 210.1 ± 0.3 | **1.000 ± 0.000** | 89.8 ± 0.1 | 0.281 ± 0.000 |

The fox prompt is highly predictable, the essay prompt is not. The harness only
ever recorded `acceptance_essay`, so the objective's requirement — *MTP acceptance
reported alongside every TG number* — was being met for one of the two TG numbers
and quietly violated for the other, with the misleading figure printed next to it.

Fixed in three places:
- `tools/bench-profile.sh` now collects and aggregates `acceptance_fox`, and
  prints each acceptance inline with its own TG line
  (`TG-fox: 185.0+/-0.6 tok/s (n=3)  [MTP acc 0.930+/-0.000]`).
- `tools/verify-profile.sh` records and gates `acceptance_fox` as its own metric.
- Both baseline JSONs carry an `acceptance_fox` entry (`min_absolute` at 85% of
  observed) and refreshed n=3 numbers.

Both gates re-run and **PASS 9/9, exit 0**:
`receipts/verify-throughput-accfox-2026-08-19.json`,
`receipts/verify-fidelity-accfox-2026-08-19.json`.

## Why this matters more than the sweep

The two profiles differ by **7%** on fox acceptance (0.930 vs 1.000) and **13.6%**
on TG-fox (185.0 vs 210.1). Same runner, same MTP depth, same scheduler — only the
weight format differs. So FP4 quantization does not merely cost KLD; it degrades
the draft model's agreement with the target model, which costs *throughput*:
every rejected draft token is a wasted verify slot.

That closes a gap I could not previously explain — why the all-FP4 profile is
slower at decode despite FP4 GEMMs being faster than trellis GEMMs at large M.
It also means **criterion 2b and criterion 3 are not independent**: reaching
TG-fox ≥ 190 wants high-fidelity weights, which is the same resource the KLD
criteria want, and which the PP criterion spends. This is a third face of the
same tradeoff already documented from the PP and KLD directions in
`receipts/frontier-2026-08-19.md`, and it makes the fidelity profile's 5/6 more
robust: it passes 2b *because* it is faithful.

## Status of criterion 2b

- `PROFILE=fidelity`: **210.1 ± 0.3, PASS** (acc 1.000).
- `PROFILE=throughput`: 185.0 ± 0.6, FAIL by 2.7%, plateaued against draft
  depth, and not addressable without raising weight fidelity — which would cost
  the PP criterion that is the only reason to run this profile.

No further lever identified for TG-fox on the throughput profile. Recorded as a
measured negative result rather than left as an open "should be tunable" item.
