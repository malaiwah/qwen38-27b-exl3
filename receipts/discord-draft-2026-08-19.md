# Discord draft: Qwen3.8-27B EXL3 K5K6 on RTX 5090

**Committed artifact, not a pasted message.** This file exists so the numbers
can never go stale silently — edit it here, then paste. Last updated
2026-08-19. All numbers are our own 512-ctx shard-0 suite (KLD) and Phase-0
harness (PP/TG), n>=3 unless noted.

## The model

Qwen3.8-27B (hybrid: 16 full-attention + 48 GatedDeltaNet linear-attention
layers), served via vLLM with EXL3 trellis quantization on a single RTX 5090
(SM120, 31.40 GiB usable). K5K6-hydrated checkpoint: `mlp.gate_proj`/`up_proj`
at K5 (5 bits), everything else at K6 (6 bits). Trellis payload 16.82 GiB /
409 matrices. Vision tower + MTP head in BF16.

## Three profiles, one checkpoint

| profile | PP (tok/s) | TG-fox | TG-essay | KLD mean | KLD p99 | context |
|---|---:|---:|---:|---:|---:|---:|
| **fidelity** | 2,987.7 +/- 4.4 | **228.3** | **104.1** | **0.003405** | **0.034889** | 238,400 |
| **balanced** | 3,925.2 +/- 13.1 | 215.6 | 103.7 | 0.005672 | 0.059908 | 199,104 |
| **throughput** | **9,638.9 +/- 18.3** | 187.4 | 94.3 | 0.063759 | 0.7010 | **249,600** |

PP is 2051-token prompt, max_tokens=1, temperature=0, median of 5 reps after
warmup. TG is duration-based sustained decode, temperature=0. KLD is per-token
KL divergence vs BF16 reference, 512-context shard-0 suite, 1,048,064 scored
positions, full 248,320 vocab, body-only shared BF16 LM-head replay.

No single profile meets all six north-star criteria simultaneously. Fidelity
fails PP (2,988 vs 7,000); throughput fails KLD (0.064 vs 0.012). The trellis
checkpoint is the fidelity answer; the all-FP4 profile is the throughput
answer.

## What makes this work on 32 GiB

- **Trellis coding** (EXL3): 16.82 GiB for 27B at ~5.5 effective bpw, vs
  29.97 GB for uniform FP8. On a 96 GiB card, 8-bit is viable; on 32 GiB,
  4-bit-class weights are mandatory.
- **GDN hybrid KV**: 48 of 64 layers hold fixed-size recurrent state, not
  per-token KV. Measured 40.7 KB/token vs 256 KB/token for a pure-attention
  model — 6.3x more context per GiB of cache.
- **K5 corruption cure**: `_warm_b12x_trellis_device` keyed on `(device, bits)`
  — prefill PP jumped 52% (1,966 -> 2,988) once trellis warmup was fixed.

## Prefill scales as t(D) = aD + bD^2

Fitted across all three profiles (7-point sweep, server-side Prometheus
timings, validated against known TG values):

| profile | PP0 (tok/s) | Dc (tok) |
|---|---:|---:|
| fidelity | 3,098 | 145,260 |
| balanced | 4,048 | 112,927 |
| throughput | 10,146 | 44,657 |

The quadratic coefficient b = 1/(PP0 * Dc) = **2.21e-9 s/tok^2 is
format-invariant** (spread 1.6%) while the linear coefficient a spans 3.27x.
Every weight-format and kernel win moves only a; attention prefill runs in
BF16/FP8 regardless of weight precision, so quantization cannot touch b.

nsys at 131,072 tokens confirms this directly: **unified_attention is 74.7% of
prefill and 84.3% of decode** at that depth (vs 4.8% at 2,051 tokens). The
prediction (73.2% from the curve) was within 1.5pp.

## Prefix caching: 34x at 131k, bit-exact

| depth | cold TTFT | warm TTFT | speedup |
|---:|---:|---:|---:|
| 131,072 | 51.95 s | 1.53 s | **34.0x** |
| 32,768 | 5.75 s | 0.44 s | **13.1x** |

Mamba `align` mode (experimental) restores GDN recurrent state exactly:
max|dlogprob| = 0.000e+00 between computed and cache-restored prefill, at
both 32k and 131k. Cost: 4,800 tokens of context (249,600 -> 244,800).

## Depth calibration: late layers are most sensitive

Banded FP6 KLD, 13-layer bands:

| band | layers | KLD mean |
|---|---|---:|
| early | L0-12 | 0.003600 |
| mid | L26-38 | 0.003838 |
| late | L51-63 | **0.004395** |

Monotone late-heavy, early-vs-late CIs disjoint. This is opposite to
llama.cpp's U-shaped `use_more_bits` and exllamav3's `allocation.py`, and
confirmed independently by AtomicChat's imatrix (peak energy at L52-62).

## Requant lane (compressed-tensors, not trellis)

| artifact | KLD mean | KLD p99 | PP | note |
|---|---:|---:|---:|---|
| RTN FP8attn+NVFP4mlp | 0.022121 | 0.2450 | ~4,142 | best requant |
| GPTQ FP8attn+NVFP4mlp | 0.028548 | 0.3508 | — | loses to RTN (CIs disjoint) |
| W4A4-MLP RTN | — | — | 8,876.8 | clears 7k but garbage output (uncalibrated) |

The requant lane is MLP-precision-bound: our MLP is NVFP4 (4-bit), and both
lribeiro's attribution (MLP ~0.009 at FP8) and our own KLD ladder confirm MLP
dominates. Closing the gap needs more bits in the MLP, which 32 GiB forbids.

## Comparisons

- **vs lribeiro FP8-GPTQ v17** (96 GiB card): their best (v31, 0.01014 on
  their suite) rescales to ~0.0040 on ours via the official-Qwen-FP8 anchor.
  Our trellis: 0.002700. ~1.5x lower KLD at ~1.8x smaller.
- **vs AtomicChat GGUF** (our size range, measured vs BF16 on RTX 5090): no
  shared artifact, no rescale. Their 18.6 GB file reads 0.00730 on their suite;
  our 18.06 GB trellis reads 0.002700 on ours. Their allocation finding (GDN
  gate + state-out) replicates on our ladder (12.4% weights, 16.3% of error).
  Their depth finding (L52-62) corroborates ours.
- **vs turboderp EXL3**: same x-axis definition (excl. embeddings, incl. output
  head). Our trellis is ~1.5x lower KLD than his 6.00bpw run at comparable
  size. We reproduced his three chart types with our data.

## Artifacts

- Checkpoint: `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` (HF)
- Code: `malaiwah/qwen38-27b-exl3` (GitHub)
- KLD suite: `malaiwah/qwen38-27b-fidelity-suite-v5` (HF dataset)
- Charts: `charts/` (turbo-style KLD-vs-size, context curve, depth calibration)
