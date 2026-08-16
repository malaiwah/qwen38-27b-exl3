# [Bug] V2 model runner: replay of the speculator's captured draft-decode FULL graph raises `cudaErrorIllegalAddress` on a hybrid GDN+attention EXL3 model (MTP-3)

## Summary

With `VLLM_USE_V2_MODEL_RUNNER=1` on this branch's r34 image, an MTP-3 speculative config on a
hybrid GDN+attention EXL3 model **negotiates, loads, allocates KV and captures every CUDA graph
successfully** — and then dies deterministically (3/3 engine starts) in `warmup_kernels`, before
the server ever answers `/health`. Under `CUDA_LAUNCH_BLOCKING=1` the fault pins to the replay of
the V2 speculator's own captured draft-decode FULL graph:

```
self.decode_cudagraph_manager.run_fullgraph(batch_desc)
  File ".../vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py", line 537, in _multi_step_decode
self.graphs[desc].replay()
  File ".../vllm/v1/worker/gpu/cudagraph_utils.py", line 589, in run_fullgraph
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The same model on the same image serves fine on the V2 runner **without** a speculator, and
serves fine with V2 + MTP-3 **eager** — the breakage is exactly the captured-draft-decode-graph
path.

## Environment

- Build: `vllm 0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34`
  (dev/gilded-gnosis lineage)
- Image: `localhost/vllm:gg-r34-patched`, manifest digest
  `sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27` — the r34 release
  plus this fork's three baked model modules (`exl3.py`, `qwen3_5.py`, `qwen3_5_mtp.py`),
  verified in-image; **no source bind mounts**; podman 4.9.3
- Hardware: one physical NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.57.04
- Model: Qwen3.8-27B — hybrid GDN (linear attention) + full attention, EXL3 weights,
  `--kv-cache-dtype fp8`, MTP speculator (`{"method":"mtp","num_speculative_tokens":3}`)
- CUDA graphs: `{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}` with
  `VLLM_EXL3_GRAPH_DECODE=1`

## Reproduction

Container env:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_EXL3_EMBED_BITS=8
VLLM_EXL3_GRAPH_DECODE=1
VLLM_EXL3_PREFILL_RECONSTRUCT_M=128
VLLM_USE_V2_MODEL_RUNNER=1
```

vLLM argv (the `v2base` arm; the per-batch-depth arm below differs only in
`--speculative-config`):

```
vllm serve /models/ctx-repo/snapshots/c45c273b0d6ef2859cb2d85b36dd52253c80d878 \
  --served-model-name m \
  --quantization exl3 \
  --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[32]}' \
  --host 127.0.0.1 --port 8231
```

## What happens

Everything before warmup succeeds, in both arms:

- V2 banner prints (`V2 Model Runner`); no `Overriding cudagraph_mode ... to PIECEWISE` line in
  either arm — `FULL_DECODE_ONLY` is retained.
- EXL3 graph decode enabled; exl3_gemm primed for 14 row counts (m=1..32).
- KV cache allocated (274,059 tokens static-depth arm; 271,080 tokens schedule arm).
- All graphs capture: static arm 8/8; with
  `"num_speculative_tokens_per_batch_size":[[1,2,3],[3,8,1]]` added, **16/16 target FULL graphs
  (8 sizes × 2 depths) plus 8/8 speculator decode graphs** — the per-batch-size depth schedule
  negotiates correctly on V2.
- `Capturing model for speculator...` completes.

Then `warmup_kernels` (`vllm/v1/worker/gpu/warmup.py:317`) drives
`speculator.propose(...)` → `_multi_step_decode(...)` and the engine core dies with
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`. 3/3 engine
starts (static arm, schedule arm, and the launch-blocking diagnostic) die at the same place;
the API server then fails with `Engine core initialization failed`.

Without launch blocking, the asynchronous fault surfaces at the next sync point,
`self._seq_lens_cpu = self.seq_lens.to("cpu")` in `_build_draft_attn_metadata`
(`vllm/v1/worker/gpu/spec_decode/speculator.py:312`) — that line is the messenger, not the
culprit. With `CUDA_LAUNCH_BLOCKING=1` and an otherwise identical argv, the fault site is
`self.graphs[desc].replay()` inside `decode_cudagraph_manager.run_fullgraph`
(`vllm/v1/worker/gpu/cudagraph_utils.py:589`), i.e. the replay of the captured draft-decode
graph itself.

## Attribution

Same image, same model, same host; one delta per row:

| configuration | result |
|---|---|
| V2, **no speculator** (no `--speculative-config`), graph decode on, FULL_DECODE_ONLY [32] | serves; ready in 95 s; greedy probe returns |
| V2 + MTP-3, **`--enforce-eager`** (cudagraph_mode NONE, `VLLM_EXL3_GRAPH_DECODE` unset) | serves; ready in 49 s; greedy probe returns |
| V2 + MTP-3, **captured draft-decode graph** (either static depth 3 or the per-batch schedule) | `cudaErrorIllegalAddress` in warmup graph replay, 3/3 |

So V2 + hybrid GDN + EXL3 + fp8 KV is fine, and the V2 speculator's draft kernels are fine
eagerly; only the replay of the speculator's captured draft-decode graph faults.

## Why this path matters

Measured on this same image, profile, frozen prompts and harness (MRV1 runner,
`receipts/perf-sweep-5090.json`): static depth 3 wins single-stream decode (83.31 vs 74.28
per-request tok/s) while depth 1 wins at eight streams (409.35 vs 313.28 aggregate tok/s,
**+30.67 %**). A working `num_speculative_tokens_per_batch_size` would give both in one server.

On the MRV1 runner that knob is unreachable for this model: setting it downgrades
`cudagraph_mode` from `FULL_DECODE_ONLY` to `PIECEWISE` (`config/vllm.py:862-887`) and
`Exl3Config` refuses any mode with a non-`NONE` mixed mode, so the server fails to start. The V2
runner skips that downgrade — as this run confirms (16 per-depth FULL graphs captured, no
override line) — which makes V2 the only route to the schedule on this build. Eager is not a
salvageable fallback: graph decode is worth ~48 % of decode throughput on this model, so an
eager schedule loses more than the schedule gains.

## What we did NOT establish

- **No minimal reproducer outside our model family.** Every faulting run is this hybrid
  GDN+attention EXL3 checkpoint; we have not tried a small stock model under the V2 runner with
  MTP and a captured draft graph.
- **Not tested on stock vLLM.** The V2 model runner as shipped in this branch and the
  `num_speculative_tokens_per_batch_size` knob are what we exercised; we make no claim about
  upstream vLLM's V2 path.
- **The faulting buffer is not identified.** We pinned the replay call, not which captured
  address is illegal; nothing here distinguishes a stale capture-time pointer from an
  out-of-bounds draft-KV slot.

We can support everything above on the fork build named in Environment, and only there. If a
maintainer wants specific diagnostics on this model (e.g. compute-sanitizer over the warmup
replay, or a capture with depth 1), we can run them on request.

## Evidence

Full receipts, including every argv, engine banner, capture count, call chain and the sha256 of
every raw server log:

- `receipts/v2-runner-depth-schedule.json` (commit `4d3cade`) — the two faulting arms, the
  launch-blocking pin, and the no-MTP / eager attribution serves
- `receipts/perf-sweep-5090.json` (commit `b8c72cc`) — the MRV1 depth-3 vs depth-1 numbers cited
  above

both in <https://github.com/malaiwah/qwen38-27b-exl3>.
