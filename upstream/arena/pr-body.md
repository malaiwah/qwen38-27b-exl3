# [EXL3] share one reconstruct-scratch arena across dense prefill geometries

Stacked on #318 (same branch base `codex/gg-exl3-r7-k345-20260810`; the first
commits are #314/#316/#318 — review only the last one, `96972e10f5bf`).
Addresses dense EXL3 prefill **memory**, the scratch introduced by #316.

## Problem

#316's prefill dispatch keeps one persistent fp16 `(K, min(N, 32768))`
scratch per distinct `(device, K, N-chunk)` geometry, never freed
(`_EXL3_RECONSTRUCT_SCRATCH`). One geometry is bounded at 336 MB, but a real
dense checkpoint has many: Qwen3.8-27B (409 EXL3 modules) has 9 distinct
geometries, and the 8 that run during vLLM's profiling prefill hold
**790 MiB** for the lifetime of the process:

| geometry (K × chunk) | MiB | modules sharing it |
|---|---:|---:|
| 5120×17408 (gate/up) | 170.0 | 130 |
| 17408×5120 (down) | 170.0 | 65 |
| 5120×12288 (q) | 120.0 | 17 |
| 5120×10240 (GDN in_proj) | 100.0 | 48 |
| 10240×5120 (mtp.fc) | 100.0 | 1 |
| 5120×6144 (GDN ba) | 60.0 | 48 |
| 6144×5120 (o/out) | 60.0 | 65 |
| 5120×1024 (k/v) | 10.0 | 34 |
| head 5120×32768 (only at ≥128 sampled logit rows) | +320.0 | 1 |

Because the buffers allocate during the profiling prefill, they are inside
the measured peak and shrink the automatically sized KV pool byte for byte.

## Change

Replace the per-geometry dict with **one byte arena per device**, grow-to-max
while eager, serving every geometry a `(K, chunk)` fp16 view at offset 0.
This is the sharing rule #203 prescribed for the MoE arenas ("keep separate
*plans*, share the *bytes*") applied to the dense path, and it is sound for
the same reason: the scratch is written (`ext.reconstruct[_slice]`) and fully
consumed (`ext.hgemm`, or the FP8 scale probe's `amax`) inside one eager
`_reconstruct_hgemm_into`/`_fp8_weight_scale` call on one stream, and no view
outlives the call — so at most one geometry's scratch is ever live, and the
arena needs only the largest geometry's bytes (170 MiB here; 320 MiB if the
head's reconstruct ever runs).

The dict already shared one buffer per geometry across all 130 same-geometry
layers and across target/MTP-draft — serialized execution is a load-bearing
assumption of the *existing* code. The arena extends that assumption across
geometries and adds no new one. If model layers ever run concurrently on
independent streams, both the old dict and this arena need owner-local or
synchronized storage.

## What it provably does not change

**The kernels see identical operands.** The arena view is
`arena[:2*k*n].view(float16).view(k, n)`: contiguous, offset 0, identical
shape/strides/dtype to the `torch.empty((k, n))` it replaces. Same kernels,
same launch order, same values — only the backing allocation is shared.

Measured on an RTX 5090 A/B (Qwen3.8-27B-EXL3, MTP-3, `FULL_DECODE_ONLY`,
greedy):

- a 30-case deterministic vision suite returned **byte-identical answers on
  both arms** (24/30 correct on each, equal to the checkpoint's
  qualification reference);
- a full-window needle (258,925-token prompt, depth 0.5) retrieved exactly
  on the arena arm, 176.7 s wall vs the baseline qualification's 175.9 s;
- decode throughput unchanged: three warmed C1 runs 108.51–108.94 tok/s
  (baseline) vs 109.16–109.66 (arena); prefill 3,053.6 vs 3,042.9 tok/s;
  TTFT proxy 54.4 vs 54.6 ms.

One honest caveat from the same A/B: an 8-prompt **long-continuation greedy
probe** (256 tokens each) differed between the arms on 7/8 prompts — and the
control showed **two restarts of the unpatched baseline also differ 7/8**,
(as do two restarts of the arena arm), while an in-process repeat is 8/8
identical. The stack is not restart-deterministic on long greedy
continuations regardless of this patch (`exl3_gemm`'s autotuner selects
kernel configurations by measured time per process), so cross-restart text
equality is not a usable bit-exactness probe here. Within that envelope the
arms are indistinguishable.

## CUDA-graph safety

- At production defaults the reconstruct branch cannot be captured: it needs
  `m >= VLLM_EXL3_PREFILL_RECONSTRUCT_M` (128) while `FULL_DECODE_ONLY`
  capture rows stay far below that, so the arena is eager-only.
- If an operator configures capture sizes ≥ 128 with
  `VLLM_EXL3_GRAPH_DECODE=1`, a capture can record the arena's address. The
  patch freezes the arena the moment a capture serves from it: a frozen
  arena is **never reallocated**, and any geometry that no longer fits falls
  back to a dedicated persistent per-geometry buffer — exactly the previous
  behaviour, so the worst case is the status quo, never a dangling pointer.
- Eager growth is safe for the same reason the old dict's cross-layer reuse
  was: the caching allocator's stream-ordered reuse, plus no retained views.

## Measured effect (RTX 5090, Qwen3.8-27B-EXL3-K5K6-context, 262,144-token window, util 0.955)

| arm | Available KV cache | GPU KV cache size | max concurrency at 262,144 |
|---|---:|---:|---:|
| baseline (#318 head) | 9.28 GiB | 265,122 tokens | 1.01× |
| arena (this PR) | 9.88 GiB | **282,996 tokens** | 1.08× |

**+17,874 tokens (+6.74 %), ≈ +0.60 GiB of KV pool**, reproduced identically
across two server starts per arm. The arena grew three times during the
profiling prefill (60 → 100 → 170 MiB) and stabilized at the predicted
170 MiB; the startup log carries one INFO line per growth. The static
derivation predicted +620 MiB / +18,676 tokens; the measured pool gain is
95.7 % of that.

## Validation

- `tests/quantization/test_exl3_reconstruct_arena.py` (new, CPU-only, no
  CUDA/b12x/exllamav3_ext): arena sizing, view layout parity with
  `torch.empty`, write→consume integrity under interleaving including the
  chunked head geometry, capture freeze + dedicated-buffer fallback, eager
  grow/reuse. 5/5 pass.
- Existing `test_exl3_prefill_plan.py` + `test_exl3_warmup.py`: 19/19 pass
  unchanged.
- A shape-simulation harness additionally walked the checkpoint's 9 real
  geometries through the verbatim allocation path in three call orders and
  asserted every view in-bounds, offset-0/aligned, and non-overlapping for
  every reachable concurrent set.

## AI assistance

An AI agent (Claude) assisted with implementation, tests, and the memory
derivation. Michel Belleau is the human submitter and reviewed the diff and
the GPU evidence before requesting merge.
