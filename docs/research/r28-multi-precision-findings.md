# R28: Multi-precision allocator with BiP and Hadamard as actions — Findings (v3)

## Claim

Including BiP as an allocator action in the 9-option alphabet {K, K+B, K+H} × {K4, K5, K6} improves the global R-D frontier by 69–71% (HWE, 12 tensors mixed) or 46–65% (4 real tensors) at production-scale sidecar rates. At 5.5 bpw, the mixed 9-option allocator achieves 51% of K6-uniform HWE at 89% of bytes — it BEATS K6-uniform.

**This is a POC against naive uniform quantization, not EXL3's existing incoherence. Local HWE can invert end-to-end KLD. KLD harness must authorize.**

## v3 changes (from GapAudit feedback)

1. Fixed aspect-ratio artifact: sidecar now priced at production scale (17408×5120), not 128×128 slice
   - BiP sidecar: 0.0083 bpw (production) vs 0.5156 bpw (slice) — 62× overpriced in v2
   - Had sidecar: 0.00025 bpw (production) vs 0.0156 bpw (slice)
2. Corrected BiP vs Had discrepancy explanation: fit scope (slice-local vs global scales), not metric
3. Added sidecar sweep from production rate to slice rate

## Key result: production-scale sidecar changes everything

At production scale, BiP's sidecar (0.0083 bpw) is negligible. The allocator can afford BiP on ALL tensors AND use higher K values:

| Budget | K-only D | Full 9-opt D | Improvement | BiP count |
|--------|----------|--------------|-------------|-----------|
| K5-uniform | 1.763e-5 | 5.519e-6 | +68.7% | 12/12 |
| 5.5 bpw | 1.078e-5 | 3.050e-6 | +71.7% | 12/12 |
| K6-uniform | 5.973e-6 | 1.711e-6 | +71.4% | 12/12 |

**At 5.5 bpw: mixed 9-option has 0.511× K6-uniform HWE at 0.893× bytes — BEATS K6.**

## Real-only analysis (4 real tensors, production sidecar)

| Budget | K-only D | Full D | Improvement | BiP |
|--------|----------|--------|-------------|-----|
| K5-uniform | 1.154e-6 | 6.193e-7 | +46.3% | 4/4 |
| 5.5 bpw | 8.694e-7 | 3.070e-7 | +64.7% | 4/4 |
| K6-uniform | 3.128e-7 | 1.582e-7 | +49.4% | 4/4 |

With production sidecar rates, ALL real tensors get BiP. Improvement is 46–65% (was 9–29% with slice-scale sidecar).

## BiP vs Hadamard

BiP (BiIP scaling + Hadamard) dominates Hadamard-only in original-basis HWE:
- BiP: 57–81% HWE reduction per tensor
- Had: -35% to +30% (often hurts, especially at K4)

**Explanation (corrected per GapAudit)**: Both R28 and R27 score original-basis HWE. The discrepancy is FIT SCOPE: R28 uses slice-local BiIP scales (128×128 diagonal), R27 uses global full-tensor scales. Slice-local scales may overfit. The per-tile finding (R27: scaling hurts) is about per-tile dynamic range, a different issue.

**Allocator preference**: 12/12 tensors choose BiP, 0 choose Hadamard at all sidecar rates. Hadamard only enters at 0.125 bpw (1 tensor).

## Sidecar rate sweep (oracle cost-sensitivity)

| Sidecar (bpw) | Improvement | BiP count | Had count |
|---------------|-------------|-----------|-----------|
| 0.0083 (production) | +71.7% | 12 | 0 |
| 0.0625 | +67.6% | 12 | 0 |
| 0.125 | +65.9% | 11 | 1 |
| 0.25 | +58.8% | 12 | 0 |
| 0.5156 (slice 128×128) | +43.6% | 12 | 0 |

Lower sidecar cost → higher improvement (monotonic). At production rate, improvement is 71.7% vs 43.6% at slice rate.

## K4+B vs K4+H vs K5

| Option | Total D | Total bytes | Ratio vs K5 |
|--------|---------|-------------|-------------|
| K4+B | 2.955e-5 | 101593 | 1.229x |
| K4+H | 9.028e-5 | 101394 | 3.754x |
| K5 | 2.405e-5 | 125964 | 1.000x |

K4+B uses 19% fewer bytes than K5 but has 23% higher HWE. K4+H is much worse (3.754x).

## Critical caveats

1. **Baseline is naive uniform, NOT EXL3's existing incoherence.** R26: stock EXL3+GPTQ achieves 93.4%.
2. **BiIP scaling discrepancy is fit scope** (R28 slice-local vs R27 global scales), not metric difference.
3. **8/12 tensors are SYNTHETIC.** Real-only results (46–65%) are more reliable.
4. **Sidecar sweep is ORACLE cost-sensitivity**, not realizable compressed sidecar.
5. **Local HWE can INVERT end-to-end KLD** (QSRT lesson). KLD harness must authorize.
6. **Sidecar rates: production BiP=0.0083 bpw, Had=0.00025 bpw.** Slice rates 62× higher (artifact).
7. **Future: QSRT activation-boundary transform** enables joint MLP gate/up/down rotation.
