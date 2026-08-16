# [Bug]: hybrid Mamba + `--mamba-cache-mode align`: a request near the KV-pool ceiling is admitted, prefilled to 98.9 %, requeued, and re-prefilled forever with zero output tokens and zero preemption accounting

<!--
Target repo: vllm-project/vllm (new issue).
FILED 2026-08-16 -> https://github.com/vllm-project/vllm/issues/52520 (approved by Main). Fork-side record of the same trace: local-inference-lab/vllm#394.
Approved and filed by Main's instruction; this body makes a claim on our behalf.
HOW TO POST: line 1 (without the leading "# ") is the ISSUE TITLE; everything below the marker
below is the ISSUE BODY, verbatim. Nothing else in this file is posted.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE ===== -->
### Your current environment

This is a **downstream fork build**, and I want that on the record before the trace, because
it determines what I am and am not claiming (see *Scope of the claim* at the bottom).

```text
vllm 0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34
  (local-inference-lab/vllm @ dev/gilded-gnosis, upstream integration tree 4d006a43928cdee0…,
   base image built 2026-08-10)
container image  localhost/vllm:gg-r34-patched-apc
                 sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218
  (built FROM the published base docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda…,
   adding exactly one file: see next paragraph)
torch 2.12.0+cu132, FlashInfer @ 1ac6942, python 3.12.3
GPU        1x NVIDIA GeForce RTX 5090, 32,607 MiB, SM 12.0, driver 610.57.04, CUDA UMD 13.3
host       Ubuntu 24.04.4, kernel 6.8.0-137-generic, podman 4.9.3 rootless
model      Qwen3.8-27B (Qwen3.5 architecture: hybrid GDN linear-attention + full attention),
           EXL3-quantised weights, --kv-cache-dtype fp8
```

**The scheduler in this image is upstream's own post-#51113 file.** `vllm/v1/core/sched/scheduler.py`
is the vendored r34 copy with exactly the two behavioural hunks of
[#51113](https://github.com/vllm-project/vllm/pull/51113) (`c56f169d9ae4`) applied,
`sha256 b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf`; upstream's own
`tests/v1/core/test_mamba_align_chunk_split.py` from that merge commit passes 20/20 against it
and fails 14/20 against the unpatched copy. So this is **not** #43559 or #51113 resurfacing, and
it is not the encoder-cap zero-collapse livelock of #51603 / #40757 either — see *Prior art* below.

### 🐛 Describe the bug

On a hybrid Mamba model with `--enable-prefix-caching --mamba-cache-mode align`, a request whose
length sits close to the KV pool is **accepted** by admission, prefilled to ~99 % of the pool,
dropped back to the waiting queue with the pool freed to 0 %, and prefilled again — forever.

The cycle has a fixed 30-second period. Prompt throughput pins at ~960 tok/s, generation
throughput is `0.0` for the whole run, and **not one output token is ever produced**. The HTTP
request never returns and the server never recovers. It is not wedged — it is busy — so it
presents as a very slow request rather than as a hang, and a liveness probe on `/health` keeps
passing.

Two properties make this worse than a plain admission misjudgement:

1. **`vllm:num_preemptions_total` stays at `0.0`.** Nothing is preempted, because nothing is ever
   admitted to `running` for long enough to count. The only observable is
   `vllm:num_requests_waiting_by_reason{reason="capacity"} = 1`, which is indistinguishable from a
   healthy server that is momentarily full. There is no counter that says "this request was
   descheduled after partial prefill", so no alert can fire.
2. **The work is discarded, not published.** Over the run the engine logged **261,794 prefix-cache
   queries and exactly 0 hits**. The cache was consulted on every retry and had nothing to offer, so
   every retry costs full price and no retry can ever cost less. That is what converts a one-off
   admission error into an unbounded one.

   *I now believe property 2 is an instance of the already-reported #45238 rather than a new
   finding — see Prior art. I am keeping it in the trace because it is the reason the loop is
   stable, but I am not claiming it as new.*

### Observed cycle

```text
INFO 08-16 05:37:17 [loggers.py:314] Engine 000: Avg prompt throughput: 800.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 98.9%, Prefix cache hit rate: 0.0%
INFO 08-16 05:37:27 [loggers.py:314] Engine 000: Avg prompt throughput: 959.9 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 08-16 05:37:37 [loggers.py:314] Engine 000: Avg prompt throughput: 960.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 08-16 05:37:47 [loggers.py:314] Engine 000: Avg prompt throughput: 799.9 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 98.9%, Prefix cache hit rate: 0.0%
INFO 08-16 05:37:57 [loggers.py:314] Engine 000: Avg prompt throughput: 960.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 08-16 05:38:07 [loggers.py:314] Engine 000: Avg prompt throughput: 960.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 08-16 05:38:17 [loggers.py:314] Engine 000: Avg prompt throughput: 800.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 98.9%, Prefix cache hit rate: 0.0%
```

63 such lines. Stopped by the operator after 656 s; nothing in the trace suggests it would ever
terminate. `/metrics` at the moment of teardown:

```text
vllm:num_requests_running                                   0.0
vllm:num_requests_waiting_by_reason{reason="capacity"}      1.0
vllm:num_preemptions_total                                  0.0
vllm:prefix_cache_queries_total                       261,794.0
vllm:prefix_cache_hits_total                                0.0
```

### Reproduction

One request, one server. `--max-num-seqs 1`, so there is no concurrency and no second request
involved in the failure.

Exact server argv as run (EXL3 quantisation flags are ours; the scheduler-relevant flags are the
last eight):

```sh
podman run --rm --device nvidia.com/gpu=all --ipc=host --network host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /opt/venv/bin/vllm localhost/vllm:gg-r34-patched-apc \
  serve /models/ctx-repo/snapshots/c45c273b0d6ef2859cb2d85b36dd52253c80d878 \
  --served-model-name m \
  --quantization exl3 \
  --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9585 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --host 127.0.0.1 --port 8251
```

Startup banner confirms the flags took effect, so this is not a silently-ignored flag:

```text
'enable_prefix_caching': True
'mamba_cache_mode': 'align'
Prefix caching in Mamba cache 'align' mode is currently enabled
Setting attention block size to 1600 tokens
Add 3 padding layers, may waste at most 6.25% KV cache memory
Available KV cache memory: 9.39 GiB
GPU KV cache size: 265,072 tokens
Maximum concurrency for 262,144 tokens per request: 1.01x
```

Then one request: a 261,794-token text prompt (a "needle" document with a 10-digit code at depth
0.5), `max_tokens=16`, `temperature=0`. It never returns.

The **shape** that matters, for anyone reproducing on stock vLLM with a Qwen3-Next / Qwen3.5-family
checkpoint and no EXL3 involved:

- hybrid Mamba/GDN model, `--enable-prefix-caching --mamba-cache-mode align`
- speculative decoding on (`mtp`, or anything that makes `use_eagle()` true), `num_speculative_tokens ≥ 1`
- `--gpu-memory-utilization` tuned so `GPU KV cache size` is a **few thousand tokens above**
  `--max-model-len` (here: pool 265,072 vs window 262,144, engine's own "Maximum concurrency" 1.01x)
- one request whose prompt is within ~1 % of `--max-model-len`
- `--max-num-seqs 1`, `--max-num-batched-tokens` well below the mamba block size, so the prefill is
  chunked and the failure lands on the final allocation rather than the first

### Why the startup check does not catch it, and the arithmetic

`Setting attention block size to 1600 tokens` — the mamba page size forces the attention block size
up from the default. With `align`, a request occupies **whole blocks**:

| term | value |
|---|---|
| prompt | 261,794 tokens |
| rounded to whole 1,600-token blocks | `cdiv(261794, 1600) = 164` blocks = 262,400 slots |
| plus MTP-3 lookahead slots | `num_lookahead_tokens = 3` (`scheduler.py:238-262`) |
| plus the mamba group's own running-state and speculative blocks | `1 + num_speculative_blocks` blocks, from the same pool (`single_type_kv_cache_manager.py:1532-1534`) |
| plus `BlockPool`'s permanently reserved null block | 1 block (this is #47272's point) |
| pool reported by the engine | 265,072 tokens |

The pool *looks* like it has 3,278 tokens of slack over the rounded prompt, and the startup check
passes. (Deliberately stated in tokens: `GPU KV cache size` on a hybrid model cannot be inverted into
a block count — see the note below — so I am not claiming that slack is worth any particular number
of blocks.) It is not enough once every term above is added, and the engine only discovers this at
the **final** allocation of the prefill — after ~260 k tokens of work have already been done and
thrown away.

The asymmetry between the two formulas is visible in the source. The startup sizer's per-request
bound for the attention group is:

```python
# vllm/v1/kv_cache_interface.py:291-298  AttentionSpec.max_memory_usage_bytes
return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

`cdiv(max_model_len, block_size)` pages and nothing else: no lookahead slots, no decode block, no
null block. Runtime admission for the same request goes through:

```python
# vllm/v1/core/sched/scheduler.py:958-959  (waiting queue)
new_blocks = self.kv_cache_manager.allocate_slots(
    ..., num_lookahead_tokens=effective_lookahead_tokens, ...)

# vllm/v1/core/kv_cache_manager.py:430-431
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)
```

and, for the mamba group in align mode, through:

```python
# vllm/v1/core/single_type_kv_cache_manager.py:1507-1509
num_required_blocks = cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
# :1532-1534, first prefill
num_new_blocks = 1 + self.num_speculative_blocks + int(has_partial_hit)
```

**What I have proven and what I have not.** The behaviour above is measured. The exact term that
tips the final allocation over is *not* proven by me: `GPU KV cache size` on a hybrid model is a
derived figure (`num_blocks // len(kv_cache_groups) * min_block_size`, the subject of #40384), so I
cannot recover the block-level counts from the log line, and the card this ran on is not available
for an instrumented rerun. What I can say is that the startup bound omits at least the null block
(#47272) and the lookahead slots, that admission demands both, and that at 1,600-token blocks each
omitted page is 0.61 % of a 262 k window — which is precisely the size of the gap. **If a maintainer
tells me which counters to dump, I will run an instrumented reproduction and post the block-level
numbers.**

### Measured band, one engine start per row, identical flags otherwise

| `--gpu-memory-utilization` | available KV | `GPU KV cache size` | max concurrency @262,144 | result |
|---|---|---|---|---|
| 0.955  | 9.28 GiB | — | — | **refuses to start**: needs 9.29 GiB, suggests max len 260,800 |
| 0.9555 | 9.30 GiB | 262,144 | 1.00x | starts, then **deadlocks** mid-prefill, `reason="capacity"`, 0 tok/s |
| 0.958  | 9.38 GiB | 263,608 | 1.01x | starts (no request sent) |
| 0.9585 | 9.39 GiB | 265,072 | 1.01x | starts, then **livelocks**: 30 s re-prefill cycle, 0 output tokens |
| 0.959  | 9.41 GiB | 265,072 | 1.01x | starts (identical pool) |
| 0.96   | 9.44 GiB | 265,072 | 1.01x | starts (identical pool) |

Two things fall out of the band:

- The pool is quantised: 0.9585, 0.959 and 0.96 all yield the same 265,072 tokens. So on this card
  **there is no `--gpu-memory-utilization` that makes a 262,144-token request schedulable under
  align mode.** The configuration is unserviceable, and every value of the knob either refuses to
  start or livelocks.
- The 0.9555 row (pool exactly `max_model_len`, "Maximum concurrency … 1.00x", then 0 tok/s) is,
  as far as I can tell, exactly the boundary case #47272 describes and fixes.

The same build with `--max-model-len 256000` at 0.9585 (pool 264,777, 1.03x) serves a 254,964-token
needle exactly in 179 s, then serves a second 254,967-token needle in the same process after the
first is released. So the feature itself works; it is specifically the
request-close-to-the-pool case that fails.

### Expected behaviour

In rough order of value:

1. **Reject at admission.** A request whose rounded, lookahead-inclusive, null-block-inclusive
   requirement exceeds the *whole pool* can never be scheduled at any future time, no matter what
   else drains. It should fail with a 400-class error naming the maximum servable length — the same
   information `_check_enough_kv_cache_memory` already prints at startup — rather than being
   accepted and retried forever. #35541 asked for exactly this and was closed as stale.
2. **Make the startup bound and the admission bound the same function**, the way #40946 did for
   SWA/chunked-local by introducing `max_admission_blocks_per_request` on the spec and calling it
   from both `max_memory_usage_bytes` and `get_num_blocks_to_allocate`. The align-mode rounding, the
   lookahead slots and the null block belong in that one function, so the two paths cannot drift.
   Today a configuration that cannot serve one native-length request still boots.
3. **Account for it.** A request descheduled after partial prefill should increment a preemption or
   restart counter. `num_preemptions_total = 0.0` through 63 full re-prefills of a 261 k prompt is a
   monitoring blind spot independent of the allocation bug: the only signal is a `waiting` gauge
   that healthy back-pressure produces too.
4. Optionally, bound the retries: a request that has been descheduled from the same admission
   decision N times without gaining ground should be failed rather than retried.

### Prior art I checked, and how this differs

Searched `vllm-project/vllm`, issues and PRs, open and closed (queries recorded in our receipt,
linked below).

- **#47272** *(open PR, "Reserve the KV null block when validating max_model_len")* — **the closest
  thing to a duplicate, for one of my six rows.** Its exact-boundary case ("passes the check,
  logs `Maximum concurrency … 1.00x`, is one usable block short at runtime, engine hangs at 0
  tok/s") is my 0.9555 row. It does not cover align-mode whole-block rounding or the lookahead
  slots, and it does not address the retry loop or the missing accounting.
- **#35541** *(closed as not-planned/stale)* — asked for exactly recommendation 1 ("vLLM should
  abort requests that can never fit in the KV cache block pool with a clear error, rather than
  hanging indefinitely"). Different mechanism: there the request never starts (`allocate_slots`
  returns `None` immediately). Here it prefills to 98.9 % first, which is what burns the GPU.
- **#45238** *(open)* — align mode registers only ~1–2 mamba block hashes per request, and
  `HybridKVCacheCoordinator` requires every group to hit, so a mamba miss vetoes the attention
  matches and the hit rate collapses to 0 %. I believe this **is** the explanation of my 261,794
  queries / 0 hits, so I am *not* offering that as a new finding.
- **#51603** *(merged)* and **#51582** / **#40757** / **#40707** — mamba-align livelocks where
  `_mamba_block_aligned_split` collapses a chunk to **zero** tokens because a multimodal encoder cap
  landed mid-block, so the request makes no progress at all. Different trigger (encoder cache, not
  the KV pool), different signature (0 tokens scheduled vs a full 260 k prefill each cycle), and
  text-only here — no multimodal item is in this request. #51603 is merged and is *not* in this
  build; it would not change this trace.
- **#40946** *(merged)* — same *class* of defect (runtime admission and startup pool sizing using
  different formulas) for SWA/chunked-local, and the design I am pointing at in recommendation 2.
- **#45387 / #45388** *(closed)* — permanent scheduler deadlock under KV-cache pressure, but the
  root cause is an unconditional `break` blocking `WAITING_FOR_REMOTE_KVS` promotion with a KV
  offloading connector enabled. No connector is enabled here.
- **#38516** *(closed as stale)*, **#50509** *(open)* — both about the startup check being
  unfriendly/overridable. Neither is about admission accepting what it cannot schedule.
- Nothing found for: an accepted request being re-prefilled on a fixed period with zero output;
  `num_preemptions_total` staying at 0 across repeated partial prefills; or align-mode rounding
  being absent from the startup bound.

### Scope of the claim

I can support this **on the fork build named above**, and I say plainly that I have **not**
reproduced it on stock vLLM and do not claim it there. What I can say about the transfer: the
scheduler file involved is upstream's own post-#51113 copy, the flags are upstream's, the
`kv_cache_interface.py` / `kv_cache_manager.py` / `single_type_kv_cache_manager.py` lines quoted
above are unmodified vendored upstream code in this image, and nothing in the mechanism touches
EXL3 weights, this fork's kernels, or the quantisation path — the failure is a block-count decision
taken before any kernel runs. So I would expect any hybrid-Mamba deployment running align mode with
speculative decoding and a request near its pool ceiling to hit it, but that is an expectation, not
a measurement. The card is a single physical RTX 5090 shared with other work; if a maintainer wants
a stock-vLLM reproduction (BF16 Qwen3-Next at a window sized to its own pool), say so and I will
schedule one.

### Evidence

Full artefacts — the 63-line timeline, every per-probe engine banner, every launch argv, the seven
server logs with their sha256s, and the `/metrics` scrape:
`receipts/qualification-5090-apc.json` and `receipts/qual-apc-raw/` in
<https://github.com/malaiwah/qwen38-27b-exl3>. The prior-art search log, with every query, is in
`receipts/upstream-issue-sweep.json` in the same repository. The fork-side record of this trace is
local-inference-lab/vllm#394.

### Before submitting a new issue…

- [x] Searched for relevant issues (queries recorded and published; see *Prior art*).
