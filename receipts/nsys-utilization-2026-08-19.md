# nsys: prefill is GPU-bound (92%), decode is host-bound (2%)

First nsys timeline of this stack. **This receipt corrects a conclusion in
`receipts/peer-review-2026-08-19.md` / `prefill-profile-2026-08-19.md`**, which
said both prefill and decode were CPU/launch-bound. That is true for decode and
**false for prefill** in the current (all-FP4) profile.

## Method (reusable)

This fork's `ProfilerConfig` exposes `profiler:"cuda"`, which drives
`cudaProfilerStart/Stop` from the `/start_profile` and `/stop_profile`
endpoints — exactly the hook `nsys --capture-range=cudaProfilerApi` wants. Wired
into `run-qwen38-27b.sh` behind `NSYS_PROFILE=1`:

```
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     -o /cache/jit/nsys/<tag>  vllm serve ...
```
plus `--cap-add=SYS_ADMIN` on the container and
`VLLM_NVTX_SCOPES_FOR_PROFILING=1` for the phase ranges that make the trace
readable.

Two gotchas worth knowing:
- nsys leaves only a `.qdstrm` because its auto-importer (`QdstrmImporter`,
  under `nsight-compute/*/host/linux-desktop-*/`) fails in this image with
  `libdw.so.1: cannot open shared object file`. Bind-mounting the host's
  `libdw.so.1` + `libelf.so.1` into a throwaway container converts it.
- `nsys stats` then exports a `.sqlite` you can query directly, which is how the
  numbers below were produced (no GUI needed).

Captured window: 2 prefill requests (2051-token prompt, `max_tokens=1`) plus one
60-token decode, all-FP4 profile, 262,144 ctx, mnbt 3072.

## Whole-window totals

| metric | value |
|---|---|
| kernels launched | **115,620** |
| GPU busy | 801.0 ms |
| wall span | 1551.4 ms |
| **GPU utilization** | **51.6%** |
| idle gaps | 750.4 ms across 115,617 gaps, **mean 6.49 us** |
| mean kernel duration | **6.93 us** |
| `cudaLaunchKernel` | 32,848 calls, 135.6 ms host (avg 4.13 us, med 2.01 us) |
| `cudaMemcpyAsync` (API) | 3,344 calls, 210.1 ms host (med 6.28 us, max 19.05 ms) |
| memcpy (device activity) | 8,272 ops, 22.1 ms |
| `cudaEventSynchronize` | 43 calls, 57.2 ms |
| `cudaGraphLaunch` | 11 calls, 15.9 ms (avg 1.45 ms) |

The 6.49 us mean idle gap matching the 4.13 us mean `cudaLaunchKernel` cost is
the signature of launch starvation — but the aggregate hides that the two phases
behave completely differently.

## Per-phase, from NVTX scopes — the actual result

| NVTX range | GPU busy | wall | utilization |
|---|---|---|---|
| `execute_context_1(2051)_generation_0(0)` — **prefill** | 404.8 ms | 440.0 ms | **92.0%** |
| `execute_context_0(0)_generation_1(7)` — **decode step, 7 MTP tokens** | 7.0 ms | 344.8 ms | **2.0%** |
| `gpu_model_runner: forward` (all phases) | 427.7 ms | 627.0 ms | 68.2% |
| `gpu_model_runner: draft` (MTP draft) | 286.1 ms | 478.2 ms | 59.8% |
| `gpu_model_runner: preprocess` | — | 339.6 ms / 19 calls | **17.9 ms per call** |
| `gpu_model_runner: sample` | — | 23.2 ms / 14 | 1.7 ms per call |

### Prefill: 92% utilized — GPU-bound

There is no launch headroom left in prefill. This retroactively explains three
null results that previously looked mysterious
(`receipts/prefill-profile-2026-08-19.md`):

- disabling usage telemetry: no effect
- cutting the Hadamard fold from 2880 to 448 launches: no effect
- and why the only thing that ever moved PP was making the **kernels cheaper**
  (all-FP4: 6457 -> 7627)

Note the ordering: the earlier torch-profiler capture was taken on the
*balanced* profile, where `_exl3_gemm`'s per-call reconstruct+fold burned
4.72 ms of CPU per call. Converting attention to FP4 deleted that path, which
moved prefill from host-bound to **GPU-bound**. Both measurements are correct;
they describe different configs, and the fix worked.

Consequence: further PP gains must come from arithmetic (fewer/cheaper FLOPs or
fewer weight bytes), not from overhead engineering. Prefill CUDA graphs are now
worth at most the residual 8%.

### Decode: 2% utilized — 98% host-bound

A decode step that does **7.0 ms of GPU work takes 344.8 ms of wall time.** The
GPU is idle essentially the whole step. That is the TG ceiling, and it is
entirely host-side: `preprocess` alone is 17.9 ms per call, and the MTP draft
loop runs its steps with Python between them.

**This inverts the TG roadmap.** `receipts/profiles-2026-08-19.md` argued TG
needed the 16-row skinny-M FP4 tile (b12x issue #234) because FP4 decode wastes
63/64 rows of a 64-row tile. That is still true of the *kernel*, but it is
irrelevant at 2% utilization — making the idle 98% cheaper buys nothing. The
same applies to draft-vocab pruning and an FP8 draft head: both target GPU bytes
that are not the constraint.

The TG levers, in the order the trace supports:
1. **Collapse the per-step host work.** `preprocess` at 17.9 ms/call and
   `sample` at 1.7 ms/call, times ~11 steps, dominate the step.
2. **Capture the whole MTP draft loop in one CUDA graph** (all 6 draft steps +
   sampling + acceptance), not one graph per step with Python in between — only
   11 `cudaGraphLaunch` calls appear in the whole window.
3. Only after 1-2 land does kernel efficiency (tile shape, draft head width)
   become measurable.

## Status of the two b12x kernel asks

- **#234 (16-row FP4 tile)** — de-prioritised by this trace. Revisit only after
  decode utilisation is materially above 2%.
- **#232 (W4A8)** — unaffected; it is a *fidelity* lever (KLD), and prefill
  being GPU-bound actually strengthens the case for cheaper-per-FLOP formats.
