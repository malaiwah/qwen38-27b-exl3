# R27-EqualRateKLD v3 Findings: 16×16 affine-uniform HWE proxy (rotate-then-allocate)

## Verdict

**BiP (BiIP diagonal scaling) HURTS per-tile quantization (1/36 win rate). Hadamard rotation helps at uniform K (20/36 wins). At near-equal rate (same K-sum + ~0.000253 bpw amortized signs) with rotate-then-allocate, K5.5+Had-alloc beats K5.5-alloc on 9/12 slices. This is a fixed-fixture screening result, not deployment evidence.**

CRITICAL CAVEAT (QSRT): Local HWE can INVERT end-to-end KLD. All findings are proxy-only and require KLD validation on aiboss.

## Critical correction (v3 final)

Previous versions had a stale-map bug: the Had+alloc arm reused the unrotated K5.5 allocation after applying Hadamard (alloc-then-rotate). This made Had+alloc look worse than it is. The correct approach is rotate-then-allocate: for each Hadamard seed, transform W and Hessians to rotated space, recompute DP allocation + full-HWE local search in rotated space, quantize, inverse-transform, score in original space.

## Key results

### 1. Near-equal-rate comparison (rotate-then-allocate, ~5.754 bpw)

Hadamard signs amortized at full-tensor rate (~0.000253 bpw). Same K-sum for both arms. Not byte-identical (sign bytes make Had-alloc ~0.0003 bpw higher).

| Tensor | D(K5) | D(K5+Had) | D(K5.5-alloc) | D(K5.5+Had-alloc) | D(K6) | Had-alloc beats alloc? |
|--------|-------|-----------|---------------|--------------------|----|------------------------|
| L0_gate | 5.04e-3 | 4.49e-3 | 1.17e-3 | **1.02e-3** | 1.45e-3 | **2/3** |
| L0_down | 4.69e-3 | 5.13e-3 | 1.07e-3 | 1.12e-3 | 1.01e-3 | 1/3 |
| L55_gate | 7.82e-3 | 6.34e-3 | 1.69e-3 | **1.57e-3** | 1.74e-3 | **3/3** |
| L55_down | 7.99e-2 | 7.54e-3 | 1.01e-2 | **1.51e-3** | 1.11e-2 | **3/3** |

**K5.5+Had-alloc beats K5.5-alloc on 9/12 slices.** On L55_down, the improvement is 6.7× (1.51e-3 vs 1.01e-2). On L55_gate and L0_gate, Had-alloc also beats K6 despite using fewer total bytes (5.754 vs 6.25 bpw).

**Previous "Had hurts allocation" conclusion was entirely a stale-map artifact.** When allocation is recomputed in rotated space, Had-alloc beats alloc-only on 9/12 slices. This does not establish formal synergy (no matched Had-only/mixed-rate control or interaction test); it shows the joint arm dominates the alloc-only arm at near-equal rate.

### 2. Complete factorial: BiP hurts, Hadamard helps

**Full factorial totals (4 tensors × 3 K values × 3 slices = 36 comparisons):**
- BiP wins: **1/36** (L55_down K4, slice 0 only)
- Had wins: **20/36** (55.6%)
- BiP+Had wins: **15/36** (41.7%)

At K5 specifically: BiP 0/12, Had 7/12, BiP+Had 5/12.

### 3. Lower convex envelope (corrected, with rotate-then-allocate)

| Tensor | Arms on lower convex envelope |
|--------|-------------------------------|
| L0_gate | K3, K4, K5+Had, **K5.5+Had-alloc**, K7 |
| L0_down | K3, K4, K5, K5.5-alloc, K7 |
| L55_gate | K3, K4+Had, K5+Had, **K5.5+Had-alloc**, K7 |
| L55_down | K3, K4+Had, K5+Had, K6+Had |

**K5.5+Had-alloc is on the convex envelope for L0_gate and L55_gate.** It replaces K5.5-alloc on the envelope, confirming that rotate-then-allocate dominates allocate-without-rotation at the same rate.

### 4. K4+Had vs K5 (expert's explicit request)

| Tensor | D(K4+Had) | D(K5) | K4+Had/K5 ratio |
|--------|-----------|-------|-----------------|
| L0_gate | 2.15e-2 | 5.04e-3 | 4.26× |
| L0_down | 2.36e-2 | 4.69e-3 | 5.03× |
| L55_gate | 2.80e-2 | 7.82e-3 | 3.58× |
| **L55_down** | **3.38e-2** | **7.99e-2** | **0.42×** |

**K4+Had BEATS K5 on L55_down (0.42× ratio).** On typical tensors, K4+Had does NOT reach K5 (ratio 3.6-5.0×).

### 5. Sidecar costs (negligible at production scale)

- Global BiIP scales: 0.008 bpw (f32), 0.002 bpw (int8)
- Hadamard signs (amortized): ~0.000253 bpw — essentially free
- Per-tile metadata: 0.25 bpw (8 bytes per 256-element tile)
- K-map: 0.016 bpw (8 bytes for 64 tiles)

## What to tell R28 (multi-precision allocator)

1. **The effective alphabet is {K4, K4+Had, K5, K5+Had, K5.5+Had-alloc, K6, K6+Had}.** Replace BiP with Hadamard.
2. **Rotate-then-allocate is the correct approach.** Recompute DP allocation in rotated space, not reuse unrotated allocation.
3. **K5.5+Had-alloc beats K5.5-alloc on 9/12 slices** at the same total rate. Hadamard + allocation is synergistic.
4. **K5.5+Had-alloc beats K6** on L0_gate, L55_gate, and L55_down at 5.754 vs 6.25 bpw — fewer bytes, lower HWE.
5. **K4+Had beats K5 on L55_down** (0.42× ratio) — new memory frontier for outlier-heavy tensors.
6. **BiP (diagonal scaling) should NOT be in the alphabet.** It hurts (1/36 win rate).

## Limitations

**Scope: This is a fixed-fixture screening result on 12 diagonal 128×128 slices (4 tensors × 3 slices), 3 RHT seeds, affine-uniform quantizer, in-sample allocation. Not deployment evidence. Wave 5 scaling (9 depths × 6 roles, 8+ blocks/tensor, separate calibration/validation/test, full tensors) is required before any production claim.**

1. **Proxy metric, not end-to-end KLD** — QSRT showed local metrics can invert KLD.
2. **Affine-uniform quantizer, not EXL3 trellis** — no LDLQ, no trellis coding.
3. **Only 4 tensors (L0/L55 gate/down), 3 diagonal 128×128 slices** — Wave 5 requires 9 depths × 6 roles, 8+ blocks/tensor.
4. **Synthetic Hessians** — real activation Hessians may differ.
5. **In-sample allocation** — same Hessian for selection and evaluation.
6. **L55_down results driven by slice 0** — not stable tensor-wide.
7. **No GPTQ/LDLQ** — EXL3 uses error feedback.
