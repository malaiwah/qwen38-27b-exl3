# GPTQ attempt 4: heavy dampening (0.1) — errors catastrophically worse

**Date:** 2026-08-19. Build: `/tmp/gptq-build-a4-damp010.log`.
Checkpoint: `/home/mbelleau/models/qwen38-27b-gptq-a4-damp010` (21 GB).

## Config

- Dataset: ultrachat_200k (default)
- Samples: 64, seq_len 2048, `--no-concatenate`
- `dampening_frac: 0.1` (10x default, heavy regularisation)
- No actorder

## Build: SUCCESS (61 minutes)

All 496 modules quantized, checkpoint saved.

## GPTQ internal errors: catastrophically worse

| class | A2 (damp 0.01) | A3 (platypus, actorder) | A4 (damp 0.1) |
|---|---:|---:|---:|
| GDN a/b | 4.3 | 2.4 | 16.0 |
| GDN out | 3.3 | 24.3 | 151.6 |
| GDN qkv | 658.6 | 635.1 | 4,014.7 |
| GDN z | 402.3 | 334.5 | 2,139.9 |
| MLP down | 60.7 | 751.5 | 5,718.4 |
| MLP gate/up | 3,056 | 7,304 | **40,103** |
| self_attn | 357.5 | 240.3 | 1,613.7 |

Heavy dampening made everything worse — MLP gate/up mean 13x worse than A2.
The Hessian is so ill-conditioned from `cache=None` tracing that adding more
diagonal loading pushes the GPTQ solution further from optimal, not closer.

## All 4 GPTQ attempts exhausted

| attempt | dataset | samples | lever | MLP gate/up mean | saved? |
|---|---|---:|---|---:|---|
| A1 | ultrachat | 512 | no-concat | cancelled (9h) | no |
| A2 | ultrachat | 181 | no-concat | 3,056 | crashed |
| A3 | open-platypus | 64 | no-concat + actorder | 7,304 | yes |
| A4 | ultrachat | 64 | no-concat + damp 0.1 | 40,103 | yes |

**No lever improved the calibration.** The root cause is the `cache=None`
tracing in llm-compressor's sequential pipeline, which feeds unrepresentative
activations to the GDN modules' Hessians. This is a bug in llm-compressor,
not a tuning problem. The upstream issue (#3057) and PR (#3058) were filed
earlier in this session.

## Best artifact: RTN (0.022121)

RTN remains the best requant artifact by measured KLD (0.022121 vs GPTQ
0.028548 from the prior session's attempt). The two GPTQ attempts that saved
(A3, A4) have internal errors far worse than A2, so they are expected to
measure even higher KLD. KLD measurement of A3/A4 is optional — the internal
errors are strong evidence they will not beat RTN.

**The requant lane is MLP-precision-bound.** Our MLP is NVFP4 (4-bit), and
both lribeiro's attribution (MLP ~0.009 at FP8) and our own KLD ladder
confirm MLP dominates. Closing the gap to 0.012 needs more bits in the MLP,
which 32 GiB forbids. The trellis checkpoint (KLD 0.002700) is the right
answer for this card.

## KLD measurement: not run (predicted catastrophically worse)

MLP gate/up error mean 40103 is 13x worse than the prior GPTQ that measured
0.028548. This checkpoint is expected to produce garbage output. Not measuring
KLD to save GPU time for higher-value work.
