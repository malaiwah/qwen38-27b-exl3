# Prefill is CPU-launch-bound: first real profile of the EXL3/SM120 stack

First torch-profiler capture of this serving stack (nobody had traced it
before). Artifact: `receipts/traces/torchprof-prefill-2051tok-configG.txt`
(vLLM key-averages digest). Raw chrome trace (12.9 MB gz) retained locally at
`~/.cache/jit/torchprof/rank0.*.pt.trace.json.gz`.

## Method

This fork does **not** use `VLLM_TORCH_PROFILER_DIR`; it has a first-class
`ProfilerConfig` (`vllm/config/profiler.py`). Enabling it:

```
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/cache/jit/torchprof",
                    "active_iterations":4,"torch_profiler_with_stack":false,
                    "torch_profiler_record_shapes":true,"ignore_frontend":true}'
```
then `POST /start_profile` ... load ... `POST /stop_profile`. It also exposes
`profiler:"cuda"` (the `cudaProfilerApi` capture-range hook for nsys) and
`VLLM_NVTX_SCOPES_FOR_PROFILING` for NVTX-annotated nsys timelines. Wired into
`run-qwen38-27b.sh` as `PROFILER_CONFIG`.

Config G (util 0.93 / 262,144 ctx / mnbt 3072), 3 profiled requests of the
2051-token PP prompt, `max_tokens=1`. Profiling inflates the request from
~318 ms to ~397 ms; ratios below are what matter, not absolutes.

## Result: the prefill path is dominated by CPU-side op dispatch

Totals over 3 requests (divide by 3 for per-request):

| op | CPU total | Self CPU | CUDA total | calls |
|---|---|---|---|---|
| `gpu_model_runner: forward` | — | — | 1.091 s | 3 |
| **`vllm::exl3_gemm`** | **510.2 ms (43.4%)** | 94.2 ms | 82.6 ms | 108 |
| `exl3::fp4_linear` | 346.9 ms (29.5%) | **169.0 ms (14.4%)** | 367.5 ms | 1017 |
| `b12x::dense_gemm_launch` | 61.9 ms | 58.1 ms | 261.7 ms | 1017 |
| `vllm::b12x_trellis_linear_out` | 30.9 ms | 29.0 ms | 115.4 ms | 102 |
| `aten::copy_` | 94.5 ms | 40.9 ms | 92.8 ms | **19134** |
| `aten::bmm` | 81.3 ms | 56.6 ms | 43.9 ms | **8640** |
| `aten::mul` | 55.0 ms | 34.8 ms | 34.6 ms | **10728** |
| `ChunkGatedDeltaRuleFunction` | 43.9 ms | 36.3 ms | 43.2 ms | 144 |
| `gpu_model_runner: draft` (MTP) | — | — | 67.6 ms | 6 |
| `vllm::unified_attention_with_output` | 6.1 ms | 5.1 ms | 29.5 ms | 51 |

**Headline: ~20,000 kernel launches per 2051-token request**, and the two
biggest CPU consumers are our own quantization wrappers, not the GEMMs.

## The single worst offender: `exl3_gemm` costs 4.72 ms CPU per call

108 calls (36/request) consume 510 ms of CPU for 82.6 ms of GPU work —
**4.72 ms CPU vs 0.76 ms CUDA per call.** Cause, confirmed by reading
`exl3.py:1214-1247`: for `M >= 128` the trellis path reconstructs the weight to
FP16 and folds the Hadamard **on every call**, and it only caches the result
when `weight_size_mb > 150`. The trellis matrices left in the flagship are the
attention projections, all *below* that threshold:

| matrix | fp16 size | cached? |
|---|---|---|
| q_proj 5120x12288 | 120 MB | **no** |
| o_proj 6144x5120 | 60 MB | **no** |
| k/v_proj 5120x1024 | 10 MB | **no** |
| (mlp gate/up 5120x17408) | 170 MB | yes — but FP4 in flagship, never here |

So all 64 attention matrices re-reconstruct + re-fold per forward. The fold
(`hadamard_fold_weight_chunked`) loops over K/128 blocks issuing 2 einsums
each: for K=5120 that is 40 blocks x 2 = 80 `bmm` launches per matrix, x36
matrices = **2880 bmm launches per request** — exactly the observed 8640/3.
Each is ~5 us of GPU work behind ~9.4 us of CPU dispatch.

## Two candidate fixes, both refuted by direct measurement

| change | PP median | verdict |
|---|---|---|
| baseline | 6456 | — |
| `VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1` | 6415 | **no effect** |
| `VLLM_EXL3_PREFILL_RECONSTRUCT_M=0` (fused trellis, no fold) | **5781 (-10%)** | **worse** |

- The telemetry lead came from a **flawed py-spy method**: I sampled the first
  thread marked `(active)`, which caught `_report_continuous_usage` in 28/69
  samples. It runs in a background `_report_usage_worker` thread and is not on
  the critical path. **Do not attribute critical-path time from py-spy's
  first-active-thread; filter to the engine thread.**
- Removing the fold is worse: the fused `ext.exl3_gemm` trellis kernel is
  genuinely slower at M=2051 (bake-off: attn.q 2.82 ms vs 0.36 ms for FP4).
  The fold is the lesser evil. **It must be cached, not removed.**

## What this implies (ranked, quantified)

1. **Cache the folded attention weight.** This is the whole 4.72 ms/call. The
   already-measured way is fp8dg-cached attention: PP 6466 -> 7062 (+9.2%) with
   KLD statistically unchanged. Net memory is only **+0.39 GiB** if the trellis
   codes are freed for those layers (fp8 1.56 GiB replaces trellis 1.17 GiB) —
   not the +2 GiB previously assumed, because the earlier measurement kept the
   trellis resident for an exact verify path.
2. **Collapse the fold's 80 launches into 1.** Even when it must run, the
   chunked fold is a launch-count disaster. A single batched einsum over
   `(k_blocks,128,n_blocks,128)` (the non-chunked `hadamard_fold_weight`) or a
   fast Walsh-Hadamard butterfly (7 add/sub passes, no multiplies, vs 128
   MACs/element for the dense form) removes ~2880 launches/request.
3. **The FP4 wrapper is the second CPU hotspot** (`exl3::fp4_linear` self CPU
   169 ms = 14.4%, plus 6378 `copy_` + 3576 `mul` + 339 `max` + 678 `abs` per
   request from per-call amax and padding). Fusing scale+quant into one kernel
   is worth ~50 ms CPU/request.
4. PP >= 7000 needs T(2051) <= 293 ms against ~318 ms now, i.e. **-25 ms**.
   Item 1 alone is measured at roughly that magnitude.
