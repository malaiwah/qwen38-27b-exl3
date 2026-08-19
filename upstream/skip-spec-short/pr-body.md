# PR: scheduler: skip speculative decoding when all scheduled requests need <=1 output token

- **PR URL:** https://github.com/local-inference-lab/vllm/pull/437
- **Branch:** `feature/skip-spec-decode-short-requests` on `malaiwah/vllm-voipmonitor`
- **Base:** `local-inference-lab/vllm:dev/gilded-gnosis`
- **Commit SHA:** `43e8693122113ee7deb1992919a0774b027f6240`
- **File:** `vllm/v1/core/sched/scheduler.py` (+9/-0)
- **ast.parse:** OK

## Motivation

Speculative decoding is useless for 1-token outputs (e.g. classification,
logprobs, embedding extraction). The draft generation + verification pipeline
adds measurable latency with zero benefit when every request in the batch needs
`max_tokens <= 1`.

## Change

In `vllm/v1/core/sched/scheduler.py`, after `num_spec_tokens_to_schedule` is
computed (including the dynamic-spec-decode lookup), add a guarded block:

```python
# Skip spec decode when all scheduled requests need ≤1 output token.
# Speculative decoding is useless for 1-token outputs and adds ~40ms
# of draft generation + verification overhead.
if num_spec_tokens_to_schedule > 0:
    if (scheduled_new_reqs
            and not scheduled_running_reqs
            and all(req.max_tokens <= 1 for req in scheduled_new_reqs)):
        num_spec_tokens_to_schedule = 0
```

The guard requires:
- `num_spec_tokens_to_schedule > 0` (spec decode is active)
- `scheduled_new_reqs` is non-empty (there are new requests this step)
- `scheduled_running_reqs` is empty (no running/decoding requests mixed in)
- **Every** new request has `max_tokens <= 1`

When all conditions hold, spec decode is skipped for this scheduling step by
setting `num_spec_tokens_to_schedule = 0`.

## Measured evidence (RTX 5090, 31.4 GB, MTP=6)

| Benchmark | Before | After | Delta |
|-----------|--------|-------|-------|
| 1-token-prompt request latency | 141 ms | 127 ms | **-14 ms** |
| 2000-token prefill | 7,445 tok/s | 7,635 tok/s | **+2.5%** |
| TG (normal requests) | 189.8 tok/s | 189.8 tok/s | unchanged |

The 1-token latency improvement comes from eliminating the draft-model forward
pass + verification overhead. The prefill improvement is a side effect of
reduced scheduling overhead when spec decode bookkeeping is skipped.

## Safety

- The guard is conservative: it only fires when **all** scheduled requests are
  new and **all** need <=1 token. Mixed batches (some running, some with
  `max_tokens > 1`) are unaffected.
- When the guard fires, `num_spec_tokens_to_schedule = 0` is the same state as
  if spec decode were disabled, so downstream code paths are already exercised
  in CI.
- No changes to request state, KV cache, or model runner — purely a scheduling
  decision.
