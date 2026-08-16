Closes #396.

## Provenance

Reported and measured on a **downstream fork build**, not upstream: image
`localhost/vllm:gg-r34-patched` (manifest `sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27`,
vLLM `dev280+gilded.gnosis.v20`), model
`malaiwah/Qwen3.8-27B-EXL3-K5K6-context` — a **hybrid Qwen3.5** (GDN/mamba +
full-attention layers, EXL3-quantised linears, fp8 KV) with an **MTP-3**
speculator, one RTX 5090 (32,607 MiB, driver 610.57.04), FlashInfer decode
backend, `cudagraph_mode=FULL_DECODE_ONLY`, `VLLM_USE_V2_MODEL_RUNNER=1`.

## Symptom

With the V2 model runner and MTP-3, the engine dies deterministically (3/3
starts) inside `warmup_kernels`, before the server answers:

```
draft_tokens = self.speculator.propose(...)
  self._multi_step_decode(...)
    File ".../v1/worker/gpu/spec_decode/autoregressive/speculator.py", line 537
    self.decode_cudagraph_manager.run_fullgraph(batch_desc)
      File ".../v1/worker/gpu/cudagraph_utils.py", line 589
      self.graphs[desc].replay()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The same build serves fine with V2 and **no** speculator, and with V2 + MTP-3
in eager mode. Only the captured draft-decode graph's *replay* faults.

## Root cause

`vllm/v1/attention/backends/flashinfer.py:1590-1601` admitted a batch to a
persistent (CUDA-graph) FlashInfer decode wrapper only when its per-request
query length equalled one hardcoded scalar,
`self._planned_decode_q_len = 1 + num_spec_tokens` (line 741). Every other
FULL-captured decode shape was therefore captured **and** replayed on the
*dynamic* wrapper:

* the V2 speculator's draft-decode graphs, which run `q_len == 1` — one token
  per request per draft step (`spec_decode/autoregressive/speculator.py:345`
  dispatches them with `uniform_token_count=1`);
* under `num_speculative_tokens_per_batch_size`, every reduced-depth verify
  shape (this profile captures 8 sizes x 2 depths).

The dynamic wrapper is not safe to capture a graph around. For
`use_cuda_graph=False`, `fast_plan_decode` (flashinfer.py:2465) falls through
to the full `plan()`, and flashinfer's `fast_decode_plan`
(`flashinfer/decode.py:3830-3837`) *rebinds* `_paged_kv_indptr_buf` /
`_paged_kv_last_page_len_buf` and *reallocates* `_qo_indptr_buf` on every
plan:

```python
    else:
        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        self._paged_kv_last_page_len_buf = last_page_len
        if self.use_tensor_cores:
            self._qo_indptr_buf = qo_indptr_host.to(self.device, ...)
```

A captured graph bakes in the *addresses* its wrapper held at capture time.
Those capture-time tensors are dropped when the capture-time metadata is, so
their memory is recycled; the replay then reads freed memory and the attention
kernel indexes out of bounds → `cudaErrorIllegalAddress`. The
`seq_lens.to('cpu')` host sync at `spec_decode/autoregressive/speculator.py:312`
is only where the asynchronous fault surfaces.

## Instrumented evidence

A probe (not part of this PR) wrapped `FlashInferMetadataBuilder.build` and
`.build_for_cudagraph_capture` and logged, per build, the wrapper the build
resolves to and that wrapper's plan-buffer addresses. Raw logs:
`logs/fiprobe-{u_probe,u_probe_pin,f_probe}.log`.

**Before** — the draft builder gets the dynamic wrapper at capture *and* at
replay, and its plan buffers move on every replay-time build, while the target
builder keeps a persistent wrapper at stable addresses:

```
phase=capture builder=12640 layer=language_model.model.layers.3.self_attn.attn tokens=32 wrapper=3008  cudagraph=True  fixed_bs=8 q_len=4 _paged_kv_indptr_buf=0x14e0003000 ... _qo_indptr_buf=0x26a0000800
phase=capture builder=51568 layer=mtp.layers.0.self_attn.attn              tokens=8  wrapper=96848 cudagraph=False fixed_bs=0 q_len=1 _paged_kv_indptr_buf=0x4a20000800 ... _qo_indptr_buf=0x4a20000c00
phase=replay  builder=51568 layer=mtp.layers.0.self_attn.attn              tokens=8  wrapper=96848 cudagraph=False fixed_bs=0 q_len=1 vs_capture=MOVED:_paged_kv_indptr_buf,_paged_kv_last_page_len_buf,_qo_indptr_buf _paged_kv_indptr_buf=0x4a20000e00 ... _qo_indptr_buf=0x4a20000a00
```

Counts for that run: 65 dynamic-wrapper builds, 17 replay builds with `MOVED`
plan buffers — and the run dies with `cudaErrorIllegalAddress`.

**Causality, not correlation** — a control run that changes nothing except
keeping the capture-time plan buffers *alive* (a probe-held reference, still
unpatched) no longer faults: the engine becomes ready and serves 8 greedy
completions. The fault is the freed capture-time plan buffers, not a
wrong-shaped kernel.

**After** — the draft builder gets its own persistent wrapper for its captured
shape, and every replay-time build resolves to the same buffers:

```
phase=capture builder=91104 layer=mtp.layers.0.self_attn.attn tokens=8 wrapper=7280 cudagraph=True fixed_bs=8 q_len=1 _paged_kv_indptr_buf=0x14e0074e00 ... _qo_indptr_buf=0x4a20000800
phase=replay  builder=91104 layer=mtp.layers.0.self_attn.attn tokens=8 wrapper=7280 cudagraph=True fixed_bs=8 q_len=1 vs_capture=IDENTICAL _paged_kv_indptr_buf=0x14e0074e00 ... _qo_indptr_buf=0x4a20000800
```

Counts: **0** dynamic-wrapper builds, 396 persistent-wrapper builds, **268
replay builds `IDENTICAL`, 0 `MOVED`**.

## The fix

Stop deriving the admissible query length from `1 + num_spec_tokens`, and
record the shapes capture actually plans instead:

* `_decode_wrappers_cudagraph` is keyed `(num_decodes, q_len_per_req)`;
* `build_for_cudagraph_capture` is overridden to mark the capture-time build,
  which is what creates, plans and records a persistent wrapper;
* `persistent_decode_wrapper_eligible` admits a batch to the persistent
  wrapper only for a recorded shape.

Capture and replay therefore resolve to the same wrapper — hence the same plan
buffers — for the draft-decode shape and for every captured verify depth,
while shapes no graph replays (spec truncation near `max_tokens`, chunked
prefill tails fused with a spec step) keep the dynamic wrapper that replans per
call. That property was the point of the previous scalar gate and it is
preserved.

## Before / after

| | before | after |
|---|---|---|
| V2 + MTP-3 static depth 3 | dies in warmup, `cudaErrorIllegalAddress` at `cudagraph_utils.py:589` (3/3) | ready in **53.0 s**, GPU KV 271,080 tokens, serves |
| V2 + MTP-3 + `num_speculative_tokens_per_batch_size=[[1,2,3],[3,8,1]]` | dies in the identical warmup replay | ready in **55.1 s**, GPU KV 268,101 tokens, serves |
| draft-decode replay builds with moved plan buffers | 17 | 0 |
| dynamic-wrapper builds under FULL decode capture | 65 | 0 |
| `tests/v1/attention/test_flashinfer_decode_qlen_guard.py` | 14 cases (scalar contract) | **26 passed** in-image (recorded-shape contract, draft `q_len==1`, per-depth schedule, capture build) |

## Cost

One persistent decode wrapper per captured decode shape instead of per
captured batch size: on this profile, 8 additional wrappers for the draft
(~8 MiB device int workspace each, `flashinfer/decode.py:871`), and one set per
captured depth when a depth schedule is used. Measured peak on this box:
32,092 MiB (V2 + MTP-3) vs 31,310 MiB (MRV1 baseline, same utilisation) — the
V2 runner accounts for most of that delta, but the extra wrappers are part of
it, and at `--gpu-memory-utilization 0.97` the engine is left with tens of MiB
of headroom.

## Explicitly open (does not affect this fix)

* **A second, independent fault is under investigation on the same profile.**
  With this fix applied, both V2 arms serve, but the throughput probe's first
  2048-token prefill kills the engine with
  `torch.OutOfMemoryError: ... Tried to allocate 68.00 MiB ... 58.56 MiB is
  free` inside the EXL3 prefill reconstruct (`_reconstruct_hgemm_into` →
  `torch.empty_like(x)`), in **both** the static-depth and depth-schedule arms.
  That is a memory-budget problem at `--gpu-memory-utilization 0.97` under the
  V2 runner, not a plan-buffer problem; it is being measured separately.
* The per-batch-size speculative depth schedule this unblocks is **still under
  measurement**; no throughput number is claimed here.
* Greedy output on this stack is not reproducible across engine restarts
  (`exl3_gemm` autotune selects kernel configs by measured time per process;
  baseline-vs-baseline restarts differ 7/8, within-instance repeats are 8/8
  identical). Any fidelity comparison here is within-process by necessity.
