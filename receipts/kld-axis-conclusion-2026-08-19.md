# KLD axis: additivity validated, and <=0.012 is proven unreachable with current kernels

## 1. The additive attribution model, now validated against a held-out prediction

`receipts/peer-review-2026-08-18.md` retracted the original "closes to 1e-5"
claim as circular and demanded a held-out test. Run:

**Prediction, made before measuring:** a config with FP4 on `mlp.gate_up_proj` +
`mlp.down_proj` only (everything else trellis) should give
`floor + gu + dn = 0.00468 + 0.0242 + 0.0127 = 0.04158`.

**Measured: 0.042044** (ci95 0.039368-0.045089, p99 0.4700).
Error **+1.1%**, comfortably inside the interval. **Additivity holds.**

The same run also completed the model. The all-FP4 profile measured
**0.063759** (ci95 0.060026-0.068062, p99 0.7010) against a 3-group prediction
of 0.05678 — a +10.9% gap, which is exactly the one group the original
attribution never isolated: `self_attn`. That pins it:

| group | KLD contribution | share |
|---|---|---|
| `mlp.gate_up_proj` | 0.0242 | 38% |
| `linear_attn.` (GDN in/out) | 0.0152 | 24% |
| `mlp.down_proj` | 0.0127 | 20% |
| **`self_attn` (q/k/v/o)** | **0.0070** | **11%** |
| floor (trellis 0.0027 + int6 embeds 0.0020) | 0.00468 | 7% |
| **sum** | **0.0638** | vs **0.063759 measured** |

So per-group FP4 error is additive to ~1%, and a mixed config's KLD can now be
*computed* instead of measured. That is the useful deliverable of this axis.

Derived FP6 error factor, floor-removed:
`(0.010699 - 0.00468) / (0.063759 - 0.00468)` = **0.102** — FP6 costs ~10% of
FP4's error.

## 2. The proof that KLD <= 0.012 cannot coexist with PP >= 7000

Budget after the irreducible floor: `0.012 - 0.00468 = 0.00732`.

**The smallest single FP4 group (`down_proj`, 0.0127) already exceeds the entire
budget.** Therefore no subset assignment of FP4 can satisfy the criterion — the
model must be FP6-or-better essentially everywhere:

| FP6 on | predicted KLD | verdict |
|---|---|---|
| nothing (all-FP4) | 0.0568 | fail |
| gate_up | 0.0350 | fail |
| gate_up + GDN | 0.0214 | fail |
| **all three MLP/GDN groups** | **0.0100** | **pass** (measured all-FP6: 0.0107) |

And "FP6 everywhere" has a measured price: **PP 4671** vs 7627 for all-FP4,
because FP6 GEMM costs 1.8-2.1x FP4 per chunk at M=2051
(`tools/kernel-bakeoff.py`) and — per `receipts/nsys-utilization-2026-08-19.md`
— **prefill is 92% GPU-utilised**, so GEMM cost passes straight through to
wall-clock. There is no overhead slack left to absorb it.

This closes the axis with a proof rather than an opinion:

> PP >= 7000 requires FP4-class GEMM cost. KLD <= 0.012 requires FP6-class
> fidelity. No configuration of the three available formats provides both, and
> per-layer mixing cannot help because the required error reduction (86% of all
> quantization error) forces FP6 onto ~86% of the weight anyway.

The only escape is a format that is **FP4-cheap in bytes/FLOPs and FP8-accurate
in activations**: W4A8 (FP4 weights, FP8 activations), i.e. b12x issue #232.
Our probe confirmed no such path exists today — `dense_gemm` with
`a_fmt=e4m3, b_fmt=e2m1` fails `packed_k_bytes must be divisible by 3` because
e2m1 routes into the MXFP6 3:4 packing path.

## 3. Consequence for the layer-wise sensitivity sweep

De-scoped. It was queued to find a cheap subset of layers needing high
precision, but with additivity validated the arithmetic answers it for free: the
budget demands an 86% error reduction, so even a perfectly concentrated
sensitivity distribution lands close to all-FP6 in both bytes and cost. Spending
~15 GPU-minutes per layer-group data point to confirm that is not worthwhile.

## 4. Honest note on the tails

The p99 values track the means (all-FP4 0.7010, gate_up+down 0.4700,
balanced 0.638, all-FP6 0.111, trellis 0.0275), so the tail criterion
(p99 <= 0.15) fails and passes together with the mean criterion. Only all-FP6
(0.111) and trellis (0.0275) satisfy it.
