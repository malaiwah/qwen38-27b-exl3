# POSTED comment on local-inference-lab/vllm#394 — prior-art cross-references, and one retraction

<!--
Target: https://github.com/local-inference-lab/vllm/issues/394 (our own issue)
Status: POSTED by UpstreamIssueSweep, 2026-08-16. Permitted as a duplicate/prior-art
cross-reference comment; it retracts a claim of ours rather than making a new one.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
Prior-art sweep of `vllm-project/vllm` (issues and PRs, open and closed) against this report. One
part of it is not new and I am withdrawing it; the rest stands with cross-references. Every query is
recorded in `receipts/upstream-issue-sweep.json`.

**Withdrawn: expectation 1, "publish the partial prefill to the prefix cache before requeueing", and
the 261,794-queries/0-hits observation behind it.** That is upstream
[vllm-project/vllm#45238](https://github.com/vllm-project/vllm/issues/45238) (open, 2026-06-11),
"Hybrid-model prefix caching silently drops to 0 % when the align-mode Mamba checkpoint lands in
request-unique tokens". In `align` mode only ~1–2 mamba block hashes are ever registered per request,
and `HybridKVCacheCoordinator` requires every group to hit, so a mamba miss vetoes the attention
groups' matches and the hit rate is 0 % regardless of what the retry recomputes. So the zero hits are
not evidence that *this* defect discards work; they are that defect. I had it as "the more important
of the two defects" — that ranking was wrong, and the sentence should read that #45238 is what makes
this loop stable rather than that the requeue path fails to publish.

**Duplicate, for one row of the band: the 0.9555 deadlock.**
[vllm-project/vllm#47272](https://github.com/vllm-project/vllm/pull/47272) (open PR, "Reserve the KV
null block when validating `max_model_len`") describes our 0.9555 row exactly: pool sized to exactly
`ceil(max_model_len / block_size)` blocks, startup check passes and logs
`Maximum concurrency … 1.00x`, one usable block short at runtime because `BlockPool` permanently
reserves the null block, engine then hangs at 0 tok/s. It does not cover the align-mode whole-block
rounding, the lookahead slots, the retry loop, or the missing preemption accounting.

**Related, closed as stale, same ask as our expectation 2.**
[vllm-project/vllm#35541](https://github.com/vllm-project/vllm/issues/35541): "vLLM should abort
requests that can never fit in the KV cache block pool with a clear error, rather than hanging
indefinitely." Closed `not_planned` + `stale`. Different mechanism (there the request never starts;
`allocate_slots` returns `None` on the first attempt), same remedy.

**Related, same class, and the design our expectation 4 should point at.**
[vllm-project/vllm#40946](https://github.com/vllm-project/vllm/pull/40946) (merged 2026-04-27) fixed
precisely "startup pool sizing and runtime admission use different formulas and drift" for
SWA/chunked-local, by putting the bound in one `max_admission_blocks_per_request` method on the spec
and calling it from both `max_memory_usage_bytes` and `get_num_blocks_to_allocate`.

**Explicitly NOT the same, recorded so nobody merges them:**
[#51603](https://github.com/vllm-project/vllm/pull/51603) (merged 2026-08-10),
[#51582](https://github.com/vllm-project/vllm/pull/51582) (closed unmerged) and
[#40757](https://github.com/vllm-project/vllm/pull/40757) (open) are also called "mamba-align
livelock", but the trigger there is a multimodal **encoder cache** cap landing mid-block, which
collapses `_mamba_block_aligned_split` to **zero** tokens so the request makes no progress at all.
Our trace is text-only, has no multimodal item, and does a full ~260 k-token prefill every cycle.
#51603 is merged and is not in this build; it would not change this trace. Likewise
[#45387](https://github.com/vllm-project/vllm/issues/45387) /
[#45388](https://github.com/vllm-project/vllm/issues/45388) (permanent scheduler deadlock under
KV-cache pressure) root-cause to an unconditional `break` blocking `WAITING_FOR_REMOTE_KVS`
promotion with a KV offloading connector enabled; no connector is enabled here.

**Still net-new after the sweep**, i.e. no upstream issue or PR found for any of these:

- an admitted request being re-prefilled on a fixed period, indefinitely, with zero output tokens
  (as opposed to never starting, or stalling at zero progress);
- `vllm:num_preemptions_total` remaining `0.0` across 63 full re-prefills of a 261 k-token prompt,
  so the only observable is a `waiting` gauge indistinguishable from healthy back-pressure;
- align-mode whole-block rounding and the speculative lookahead slots being absent from the startup
  bound while admission demands them.

Two corrections to the body's arithmetic section, for honesty rather than because they change the
verdict: `GPU KV cache size` on a hybrid model is a derived figure
(`num_blocks // len(kv_cache_groups) * min_block_size`, the subject of
[#40384](https://github.com/vllm-project/vllm/pull/40384)), so the block-level counts cannot be
recovered from that log line, and the specific term that tips the final allocation is therefore
**not** proven — only that the startup bound omits at least the null block and the lookahead slots,
both of which admission demands. The measured behaviour is unaffected.
