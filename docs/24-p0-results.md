# P0 results: prefill dispatch, and the fp32 replay negative result

> **Metric correction, 2026-08-15.** Performance results below are unchanged. The current
> overlap-corrected fidelity point is **0.007945** rather than the original 0.008157; FP8 is
> **0.012798** rather than 0.013126. See [docs/31](31-frozen-qualification.md).

Both P0 items from [23](23-next-attack-list.md) are done. One worked, one produced a
negative result that changes how the measurement floor should be described.

## P0.1 — Prefill dispatch: +113 % prefill, decode unchanged

**Patch:** inside the existing `vllm::exl3_gemm` custom op, dispatch on row count —
`exl3_gemm` below the threshold, reconstruct + `hgemm` at or above it. Default threshold
128 rows, `VLLM_EXL3_PREFILL_RECONSTRUCT_M` overrides, `0` disables.

### Measured end to end (median of 3 runs, dispersion <1 %)

| configuration | TG C1 | TG C4 | TG C8 | PP 2k | PP 6k |
|---|---:|---:|---:|---:|---:|
| before | 56.5 | 199.6 | 402.7 | 2,369 | 2,362 |
| **after** | 56.6 | 199.6 | 404.6 | **5,050** | **5,146** |
| change | +0.2 % | +0.0 % | +0.5 % | **+113 %** | **+118 %** |

Context: `Qwen/…-FP8` does 10,667 / 10,474 and `unsloth/…-NVFP4` 14,528 / 13,468, so this
closes roughly half the distance to FP8 and a third to NVFP4 in one patch.

### The trap that cost two attempts

A plain Python `if rows >= threshold` **around** the two calls looks correct and is not.
vLLM compiles the model over a single shape range, so the branch is resolved once at trace
time from the profile run's large row count; the decode CUDA graphs then capture
reconstruct+hgemm. Measured consequence: **decode collapsed from 56.5 to 22.6 tok/s at C1**
(0.26 ms per shard x 193 shards ~ 50 ms/token) while prefill improved — a silent,
plausible-looking regression. Two further traps on the way:

1. `torch.cuda.is_current_stream_capturing()` is not Dynamo-traceable
   (`torch.* op returned non-Tensor`), so it cannot be used as a guard inside the region.
2. Raw pybind extension calls are not traceable either
   (`Attempted to call function marked as skipped: exllamav3_ext…had_r_128`); the new path
   has to live inside a registered custom op, which is exactly why the existing
   `exl3_gemm` wrapper is one.

The working shape is therefore: **one opaque op, dispatch inside it at runtime.**

### Kernel-level evidence, with the extension the image actually loads

The image's baked extension does not export exllamav3's fused `reconstruct_had_slice`, so
the patch uses the unfused sequence (`had_r_128` -> `reconstruct[_slice]` -> `hgemm` ->
`had_r_128`). Speedup of reconstruct+hgemm over `exl3_gemm`, reconstruct cost included:

| geometry | m=1 | m=32 | m=64 | m=128 | m=512 | m=2048 |
|---|---:|---:|---:|---:|---:|---:|
| mlp.gate_proj | 0.16x | 0.30x | 0.55x | 1.06x | 2.61x | 4.10x |
| mlp.down_proj | 0.19x | 0.35x | 0.63x | 1.28x | 3.10x | 4.35x |
| lm_head | 0.16x | 0.32x | 0.62x | 1.19x | 3.23x | 5.21x |

Crossover is m=128 for all three geometries, which is where the default threshold comes
from. Scratch is one fp16 buffer per (device, K, N-chunk) reused across layers, bounded at
336 MB by 32768-column slicing, so peak memory does not scale with layer count.

### Fidelity cost: small, real, and disclosed

This patch changes fp16 summation order, so it is **not** bit-exact:

| comparison | mean KLD | top-1 |
|---|---:|---:|
| patched vs unpatched capture of the same checkpoint | 9.17e-04 | 98.81 % |
| vs BF16 reference, patch active | 0.008032 | 96.5987 % |
| vs BF16 reference, patch inactive | 0.007998 | 96.5972 % |

The patch costs **+0.43 %** of measured divergence and leaves top-1 agreement
unchanged to four decimals. For scale, the gap to official FP8 is 61 % and the storage-format
systematic below is 5 %. `VLLM_EXL3_PREFILL_RECONSTRUCT_M=0` restores the previous
prefill path; it does not establish bit-exact CUDA-graph decode.

## P0.2 — fp32 replay: negative result, and a better description of the floor

**Expectation:** BF16 storage of hidden states was the suspected cause of our 6.54e-04
live-vs-replay error (the reference protocol reports 1.23e-06). **Result: it is not.**

| measurement | BF16 storage | fp32 storage | change |
|---|---:|---:|---:|
| replay qualification, `KL(live ‖ replayed)` | 6.54e-04 | 6.25e-04 | **−4.5 %** |
| candidate KLD on the same 32 sentinels | 0.007998 | 0.007550 | −5.6 % |

BF16 operand storage explains only ~5 % of the live-versus-replay gap. The
remaining difference is consistent with the distinct vLLM-live and chunked-replay
logit implementations, but this experiment did not isolate one mechanism. The
patched-versus-unpatched 9.17e-04 result changes body summation order and is not
an independent measurement of the head-path difference.

### What that means for every number we publish

- **Paired replay comparisons avoid mixing live and replay kernels.** They define
  a consistent replay-domain metric on shared contexts. The qualification offset
  is not an additive constant, however, and model-dependent numerical effects
  need not cancel exactly; bootstrap intervals do not cover that systematic.
- **Absolute served-logit equivalence remains bounded by qualification.** Effects
  below roughly 1e-3 should not be presented as live-serving divergences without
  a same-kernel control.
- The iteration-1 head-attribution result (6.8e-05, CI
  [4.6e-05, 9.0e-05]) stands as a paired replay-domain result, not an absolute
  served-logit divergence.
- Reaching 1e-05 live-versus-replay agreement likely requires capturing live
  logits through the same projection implementation used by replay, or an
  explicit cross-kernel error model. Because current ranking margins are much
  larger, this is a protocol-hardening item rather than a ranking blocker.

## Net effect on the scorecard

| axis | before P0 | after P0 | best competitor |
|---|---|---|---|
| fidelity (mean KLD, held out) | 0.008157 | 0.008157 (+0.4 % if the prefill path is used during scoring) | FP8 0.013126 |
| resident weights | 21.82 GB | 21.82 GB | NVFP4 22.91 GB |
| TG C1 / C4 / C8 | 56.5 / 199.6 / 402.7 | 56.6 / 199.6 / 404.6 | NVFP4 48.9 / 171.4 / 369.7 |
| TG C1 with MTP-3 | 113.8 | 113.8 | — |
| **PP 2k / 6k** | 2,369 / 2,362 | **5,050 / 5,146** | NVFP4 14,528 / 13,468 |

## What now leads the prefill gap

The MLP GEMMs are no longer dominant, so the remaining 2-3x against NVFP4 is elsewhere.
The prime suspect is the **online-K6 attention path**: 208 projections go through
`_b12x_trellis_linear`, whose K6 dense kernel is described upstream as a *small-M decode*
kernel, at prefill row counts. Extending the same row-count dispatch to
`Exl3OnlineLinearMethod.apply` is the next measurement, and it is cheap to attribute:
serve once with the overlay off (attention stays BF16 through cuBLAS) and compare prefill.
