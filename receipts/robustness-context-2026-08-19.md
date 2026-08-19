# Long-prompt robustness + full native context (2026-08-19)

## The bug: the committed flagship could not serve a 6k-token prompt

The 238,400-token context was **hollow**. Every benchmark prompt in the
2026-08-18 session was <= 2561 tokens, which hid an engine-fatal failure:

| prompt | result |
|---|---|
| 2051 tok (PP bench) | ok, 6484 tok/s |
| 4001 tok | ok |
| **6000 tok** | **HTTP 500, EngineDeadError, `/health` 503** |

Root cause (full traceback):
```
qwen_gdn_linear_attn.py:283 forward_native -> fla_chunk_gated_delta_rule
  chunk.py:72 chunk_gated_delta_rule_fwd -> chunk_fwd_o
    chunk_o.py:168  o = torch.empty_like(v)
torch.OutOfMemoryError: Tried to allocate 96.00 MiB; 41.56 MiB free
```
The GDN (gated-delta-rule linear attention) prefill allocates full-chunk
intermediates (`g, o, A, w, h, v_new`). vLLM's profiled peak activation
(2.84 GiB at mnbt 8192) under-predicts the real peak, leaving ~40-100 MiB of
true headroom at util 0.93. See `receipts/memory-ledger.md` L1-L3.

## Two false leads, corrected

1. **"GDN can't be chunked, so prefill must be segmented with state carry."**
   Wrong. `ChunkGatedDeltaRule.forward_*` already takes `initial_state` and
   returns `final_state`, and vLLM *does* carry it across chunks. A fresh boot
   served a 12,000-token prompt fine. No kernel work was needed.
2. **"mnbt 2048 and 4096 behave identically, so chunk size is irrelevant."**
   Wrong — measurement artifact. `Restart=always` on the systemd unit
   restarted the service with **default** env after each manual instance died,
   so both runs were really mnbt 8192 (ledger L8). Fixed by
   `/tmp/boot_cfg.sh`, which stops the unit and asserts the effective args
   from the engine's own `non-default args:` dump.

## Controlled sweep (assertions passing, fresh boot per config)

Text ladder = 2k -> 6k -> 12k -> 24k -> 2k -> 12k, all in one engine.

| util | max-model-len | mnbt | KV avail | peak act | cudagraph | bench PP | verdict |
|---|---|---|---|---|---|---|---|
| 0.93 | 238,400 | 8192 | 8.89 GiB | 2.84 | 0.89 | 6484 | FRAGILE (dies @6k) |
| 0.90 | 210,000 | 8192 | 7.95 GiB | 2.84 | 0.89 | 6404 | FRAGILE (dies @24k) |
| 0.93 | 238,400 | 2048 | 9.93 GiB | 2.21 | 0.52 | 4590 | ROBUST |
| 0.93 | 238,400 | 4096 | 9.72 GiB | 2.42 | 0.52 | 6450 | FRAGILE (dies @12k#2) |
| 0.92 | 250,000 | 4096 | 9.40 GiB | 2.42 | 0.52 | 6443 | ROBUST |
| **0.93** | **262,144** | **3072** | **9.82 GiB** | **2.32** | **0.52** | **6460** | **ROBUST** |

Findings:
- **Bounding the prefill chunk is the fix; adding headroom is not.** util 0.90
  at mnbt 8192 still died, and cost 28,400 tokens of context.
- Lowering mnbt *frees* KV twice over: smaller profiled activation **and** a
  smaller CUDA-graph pool (ledger L6). 8.89 -> 9.82 GiB.
- 3072 beats 2048 on PP because the 2051-token bench stays in ONE chunk. At
  2048 it spills into a second engine step: 6460 -> 4590. That delta implies
  **~130-142 ms of fixed cost per engine step**, matching the independently
  fitted per-request overhead of 142.5 ms.

## Adopted profile (new default in patches/run-qwen38-27b.sh)

`util 0.93 / max-model-len 262144 / max-num-batched-tokens 3072`

Mixed stress gate (long prefills + vision + decode interleaved — the ordering
that actually broke things; pure text ladders pass where mixed fails):

```
prefill 2k      -> ok  6480 tok/s   alive
prefill 6k      -> ok  6157 tok/s   alive
prefill 12k     -> ok  5734 tok/s   alive
prefill 24k     -> ok  4912 tok/s   alive
vision          -> ok               alive
decode 500      -> ok    71 tok/s   alive
prefill 12k#2   -> ok  5748 tok/s   alive
vision #2       -> ok               alive
prefill 2k#2    -> ok  6915 tok/s   alive
STRESS VERDICT: ROBUST
```

## Net effect vs the committed flagship

| axis | before | after |
|---|---|---|
| max usable prompt | **~4k (engine died above)** | **>=24k verified, mixed-stress clean** |
| context served | 238,400 | **262,144 (full native, +10%)** |
| PP (2051-tok bench) | 6484 | 6460-6915 (unchanged/better) |
| TG fox / essay | 160.6 / 75.2 | 158.7 / 74.8 (unchanged) |
| vision + MTP | pass | pass |
| KV cache | 8.89 GiB | 9.82 GiB |

Zero cost, three axes improved, one engine-fatal bug removed.

## Harness bug found and fixed at the same time

`tools/bench_lib.py::vision_check` reported False on a working server: this is
a **reasoning** model, so with thinking enabled all 20 budgeted tokens went to
`message.reasoning` and `message.content` was `null` (`finish_reason: length`).
Fixed with `chat_template_kwargs={"enable_thinking": false}`, max_tokens 32,
and a defensive fallback to `reasoning`. Verified: `content='Red Blue'`.

## Upstream (to file)

1. `max_num_batched_tokens` default 8192 is unsafe for GDN/mamba-hybrid models
   on memory-tight configs; the profiled peak activation under-predicts the
   real GDN prefill peak.
2. Forward-pass OOM kills the engine instead of preempting/rejecting the
   request (ledger L3) — availability bug affecting any model.
3. Hybrid KV padding wastes up to 6.25% of the cache for a 16/48 split
   (ledger L5).
