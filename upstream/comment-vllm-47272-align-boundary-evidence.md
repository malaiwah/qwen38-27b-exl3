# POSTED comment on vllm-project/vllm#47272 — "Reserve the KV null block when validating max_model_len"

<!--
Target: https://github.com/vllm-project/vllm/pull/47272 (open PR, author 92hyungjun)
Status: POSTED 2026-08-16 -> https://github.com/vllm-project/vllm/pull/47272#issuecomment-5307330064 (approved by Main) — it reports our measurement and endorses a fix direction.
Why it is worth posting: this PR's exact-boundary case is one of our six measured rows, and our
band shows two more terms missing from the same formula. The author asked for nothing; this is
unsolicited corroboration, which is why Main approved it before it went out.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
The boundary case in this PR reproduces on a hybrid Mamba model, and there is a second row just
above it that this PR would not catch. Numbers, in case they are useful for the test matrix.

Build: a downstream fork of vLLM (`0.11.2.dev280+…20260810.r34`, upstream integration tree
`4d006a43`), one physical RTX 5090, Qwen3.5-architecture hybrid (GDN linear attention + full
attention), `--enable-prefix-caching --mamba-cache-mode align`, `--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`, `--max-model-len 262144`, `--max-num-seqs 1`,
`--max-num-batched-tokens 2048`. Align mode forces `Setting attention block size to 1600 tokens`.
One engine start per row, only `--gpu-memory-utilization` varied:

| `--gpu-memory-utilization` | available KV | `GPU KV cache size` | engine's max concurrency @262,144 | result |
|---|---|---|---|---|
| 0.955  | 9.28 GiB | — | — | refuses to start: 9.29 GiB needed, suggests max len 260,800 |
| 0.9555 | 9.30 GiB | **262,144** | **1.00x** | starts, then **0 tok/s forever**, `num_requests_waiting_by_reason{reason="capacity"}=1` |
| 0.958  | 9.38 GiB | 263,608 | 1.01x | starts (no request sent) |
| 0.9585 | 9.39 GiB | 265,072 | 1.01x | starts, then **re-prefills forever**: 30 s period, ~960 tok/s prompt, 0 output tokens |
| 0.959  | 9.41 GiB | 265,072 | 1.01x | starts (identical pool) |
| 0.96   | 9.44 GiB | 265,072 | 1.01x | starts (identical pool) |

The 0.9555 row is your case verbatim: pool exactly `max_model_len`, the check passes and logs
`Maximum concurrency for 262,144 tokens per request: 1.00x`, and the engine then never schedules the
request. So the failure is not specific to `num_gpu_blocks_override` or to `kv_cache_memory_bytes` —
ordinary `gpu_memory_utilization` profiling landed on it here, on a real card, which is the case your
PR description calls "uncommon".

The 0.9585 row is the part this PR would not fix, and it suggests two more terms are missing from
the same bound rather than one:

- `AttentionSpec.max_memory_usage_bytes` is `cdiv(max_model_len, block_size) * page_size_bytes`
  (`vllm/v1/kv_cache_interface.py:291-298`) — no null block (your point), and also no lookahead
  slots, while admission passes `num_lookahead_tokens` into `allocate_slots`
  (`vllm/v1/core/sched/scheduler.py:958-959` → `vllm/v1/core/kv_cache_manager.py:430-431`).
- In `align` mode the mamba group additionally takes `1 + num_speculative_blocks` blocks from the
  same pool on first prefill (`vllm/v1/core/single_type_kv_cache_manager.py:1507-1509`,
  `:1532-1534`), and with a 1,600-token block size each such block is 0.61 % of a 262 k window —
  the same order as the gap.

At 0.9585 the pool reports 265,072 tokens against a rounded prompt of 164 blocks = 262,400 slots —
3,278 tokens of apparent slack — and the request still cannot be placed. I am stating that in tokens
on purpose: `GPU KV cache size` on a hybrid model is a derived figure
(`num_blocks // len(kv_cache_groups) * min_block_size`, the subject of #40384), so it cannot be
inverted into a block count and I will not claim the slack is worth N blocks. Hence: the behaviour and
the candidate terms, not an assertion about which one tips it.

One suggestion, offered only because #40946 already did it for SWA/chunked-local: putting the bound
in a single `max_admission_blocks_per_request`-style method on the spec, called from both
`max_memory_usage_bytes` and `get_num_blocks_to_allocate`, would make the null block, the lookahead
slots and the align rounding impossible to add to one path and forget in the other.

Separately, and worse than the sizing itself: at 0.9585 the request is **accepted**, prefilled to
98.9 % of the pool, requeued with the pool freed, and re-prefilled on a 30-second period
indefinitely, with `vllm:num_preemptions_total` staying at `0.0` and `prefix_cache_queries_total`
reaching 261,794 against 0 hits. So the failure mode above the exact boundary is unbounded GPU burn
with no accounting, not a hang. Full trace, argv, banners and logs:
`receipts/qualification-5090-apc.json` in <https://github.com/malaiwah/qwen38-27b-exl3>.

Not reproduced on stock vLLM — this is a fork build, and I say so for the record; the
`kv_cache_interface.py` / `kv_cache_manager.py` / `single_type_kv_cache_manager.py` lines quoted are
unmodified vendored upstream code in that image, and the scheduler is upstream's own post-#51113
file.
