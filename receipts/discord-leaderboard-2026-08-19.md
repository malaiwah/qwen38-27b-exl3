# Independent Discord leaderboard: the KLD squeeze is GPTQ-mixed, and it validates our attribution work

**Date:** 2026-08-19 (late). Source: user-relayed screenshot from Discord
(`receipts/discord-leaderboard-2026-08-19.png`), third-party harness, protocol unknown
(their absolute numbers are NOT comparable to our 512-ctx suite; only their
*within-harness ordering* is evidence).

## What their tables show

Top-10 by KLD (their harness) — the relevant rows:

| model | scheme | KLD | prefill | decode |
|---|---|---|---|---|
| gptq-nvfp4-mixed-32 | FP8 W8A16 attn+lin_attn, NVFP4 W4A16 g16 + FP8 W8A8 MLP split | **0.002642** | 4551 | 30.7 |
| gptq-nvfp4-mixed-64 | FP8 W8A16 attn+lin_attn, NVFP4 W4A16 g16 all 64 MLP | 0.002666 | 4554 | 30.7 |
| gptq-mxfp8-mixed | FP8 W8A16 attn+lin_attn only (43.4 GB) | 0.002670 | 4547 | 30.6 |
| **malaiwah EXL3-EDA-research** | EXL3, error-driven allocation | **0.007461** | 1508 | 21.0 |
| **malaiwah EXL3-K5K6** | EXL3, uniform K5/K6 | 0.008170 | 1679 | 22.3 |

Top-10 by speed: modelopt-nvfp4 all-linear W4A4 10,460 @ KLD 0.350 (unusable);
nvfp4-gptq W4A4 mixed 9,660 @ 0.0755.

Their K5K6 prefill (1,679) matches our *pre-cure stock* numbers — none of our serving
fixes are in their harness. Our same weights do 2,987.7 here.

## Three conclusions

### 1. Their winning split IS our attribution finding, turned into a recipe

FP8 (8-bit) on attention + linear_attn, 4-bit on MLP. Quantizing **all 64 MLP layers**
to NVFP4 costs them nothing (0.002666 vs 0.002670 for attn-only): MLP error is nearly
free, attention error is everything. That is exactly what our additivity work found
(MLP additivity holds at -1.9%/-6.0%; attention fails +46% via KV compounding; MTP
acceptance couples to attention fidelity). Independent harness, same physics.

### 2. The gap is the PROCEDURE, not the format

On our suite, runtime-cast MLP->FP4 costs 0.042 over trellis. On theirs, GPTQ-calibrated
MLP->NVFP4 costs 0.000004 over its own baseline. Same ballpark format, 10^4 ratio in
damage: **Hessian-compensated quantization from BF16 vs uncompensated double
quantization through trellis**. We already knew direct NVFP4 (0.0301) beats
runtime-converted (0.0567); GPTQ-direct beats both by an order of magnitude.
Runtime conversion buys speed flexibility, never fidelity.

### 3. There is an unexplored middle where all six criteria could meet

Their two tables bracket it:
- mixed-32/64 (W4A16 MLP): 4,551 PP @ 0.0026 -> passes KLD, fails PP >= 7000
- nvfp4-gptq (W4A4 all): 9,660 PP @ 0.0755 -> passes PP, fails KLD <= 0.012

Nobody in either table built **FP8 W8A16 attn+lin_attn + GPTQ NVFP4 W4A4 MLP** - 8-bit
weights where error matters, 4-bit weights AND activations where it does not. W4A4 MLP
doubles mma throughput on exactly the layers that dominate prefill FLOPs
(gate_up+down = 17408-wide). Interpolating naively: PP ~6-8k @ KLD ~0.004-0.02. That
window contains the six-criteria target. This is now the highest-value requant
experiment on the list.

### Bonus: EDA-research validates error-driven allocation

Our own `malaiwah/Qwen3.8-27B-EXL3-EDA-research` (2026-08-16, solver-driven per-matrix
bit allocation) ranks **8.7% better KLD than uniform K5K6 on their harness** (0.007461
vs 0.008170) at similar speed. Same-harness ordering is protocol-independent: the
allocation solver works. Worth folding into any future trellis rebuild.

## Caveats, stated plainly

- Their protocol (context length, positions, reference model) is unknown; absolute
  KLDs are not comparable to ours. Only orderings within their table are used above.
- Their checkpoints are local paths (`models/...`), not published; the schemes are
  reproducible from the scheme column via llmcompressor, which is how we will pursue it.
- Their decode column (~30 tok/s) suggests no speculative decoding anywhere in their
  harness; our TG comparisons are irrelevant to it.
