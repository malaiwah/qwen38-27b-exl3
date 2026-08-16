# [Bugfix][Core] Fail requests the KV cache pool can never hold instead of retrying them forever

<!--
Target repo: vllm-project/vllm (pull request, base main).
Head: malaiwah/vllm-voipmonitor, branch fix/kv-pool-unservable-request-admission.
Line 1 (without the leading "# ") is the PR TITLE; everything after the marker
below is the PR BODY, verbatim.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE ===== -->
## Purpose

Fixes #52520.

### The symptom

On a hybrid Mamba + full-attention model with `--mamba-cache-mode align`, a request
whose length is near the KV pool ceiling is **accepted**, chunk-prefilled to ~99 % of
the pool, descheduled when the last chunk cannot be allocated — and then never
scheduled again. Zero output tokens, the whole prefill discarded, and the only
observable is `vllm:num_requests_waiting_by_reason{reason="capacity"} = 1`, which is
indistinguishable from a healthy server that is momentarily full. The HTTP request
never returns.

The engine boots happily into this state: `_check_enough_kv_cache_memory` passes.

### The mechanism

Two bounds describe the same quantity — how many blocks one `max_model_len` request
can hold at once — and they are computed by different code that does not agree.

**Startup** sizes the pool from each spec's `max_memory_usage_bytes`
(`vllm/v1/core/kv_cache_utils.py:2189-2198`). For a hybrid Mamba model in align mode
that is `cdiv(max_model_len, block_size)` blocks for the attention group plus
`2 + num_speculative_blocks` for the Mamba group
(`vllm/v1/kv_cache_interface.py:687-696`). It compares that against **all** the
blocks the pool will have. But `BlockPool` permanently reserves one block as the null
block (`vllm/v1/core/block_pool.py:188-191`), so only `num_blocks - 1` are ever
available to a request. A pool built at exactly the startup minimum is one usable
block short of one `max_model_len` request, and startup does not notice.

**Runtime admission** (`vllm/v1/core/kv_cache_manager.py:475-491`, reached with
`scheduler_reserve_full_isl` — default `True`) asks the coordinator for the same
quantity, but through the per-step allocators. For the Mamba align group that path
returns `1 + num_speculative_blocks` on a first prefill
(`vllm/v1/core/single_type_kv_cache_manager.py:1529-1540`), one block below the peak
the spec itself states, because a later step allocates the next state block before
`remove_skipped_blocks` retires the one from two steps ago. So admission asks for one
block *less* than the pool is short by, and admits.

The request then chunk-prefills until the final chunk needs the block that does not
exist. It is the only running request, so it preempts itself
(`vllm/v1/core/sched/scheduler.py:676-688`): `num_computed_tokens` back to 0, all
blocks freed, requeued. On the next step the same gate now also has to count the
blocks the discarded prefill left in the prefix cache as evictable, so it refuses —
and keeps refusing. The request is stuck, and the prefill is gone.

Measured on unmodified `main` at `4d2a68d64d9e05921ed5c4099146e768a92d71d5`, CPU only,
no model weights: hybrid full-attention + Mamba(align), `block_size=16`,
`max_model_len=1024`, `num_speculative_blocks=3`, `max_num_batched_tokens=256`. The
startup bound is `cdiv(1024,16)=64 + (2+3)=5 = 69` blocks, and
`get_kv_cache_configs` accepts a 69-block pool:

```text
pool of 69 blocks: startup check accepts=True (num_blocks=69)
  pool=69 blocks, free at start=68 (one block is BlockPool's null block)
    step   0 scheduled=  256 computed=  256 free= 48 status=RUNNING preemptions=0
    step   1 scheduled=  256 computed=  512 free= 31 status=RUNNING preemptions=0
    step   2 scheduled=  256 computed=  768 free= 15 status=RUNNING preemptions=0
    step   3 scheduled=  240 computed= 1008 free=  0 status=RUNNING preemptions=0
    step   4 scheduled=    0 computed=    0 free= 68 status=PREEMPTED preemptions=1
    step   5 scheduled=    0 computed=    0 free= 68 status=PREEMPTED preemptions=1
    ... 19 more identical steps ...
  VERDICT prompt=1023: finished=False computed=0 preemptions=1 status=PREEMPTED

pool of 70 blocks: startup check accepts=True (num_blocks=70)
    step   3 scheduled=  240 computed= 1008 free=  1 status=RUNNING preemptions=0
    step   4 scheduled=   15 computed= 1023 free=  0 status=RUNNING preemptions=0
  VERDICT prompt=1023: finished=False computed=1023 preemptions=0 status=RUNNING
```

One block of headroom is the entire difference. Note that the prefill dies at exactly
1008 tokens, which is `(69 - 1 - 5) * 16` — the sequence length the pool can actually
serve.

### The fix

One function computes the bound, and it is the function the startup error message
already uses:

- `max_servable_num_tokens(vllm_config, kv_cache_groups, num_blocks)` in
  `kv_cache_utils.py` — the longest sequence a pool of `num_blocks` blocks can hold
  for one request. It is `_estimate_max_model_len_from_groups` (the helper behind
  "the estimated maximum model length" that `_check_enough_kv_cache_memory` prints)
  evaluated against `num_blocks - 1` blocks, i.e. against the capacity `BlockPool`
  will actually hand out. It returns `None` for group layouts that cannot be priced
  per block — exactly the layouts whose pool was not sized per block either — so it
  never invents a bound the startup check did not apply.
- `Scheduler` evaluates it once at construction. When it is below `max_model_len` the
  configuration cannot serve its own window, which is logged at `ERROR` with the
  number the operator needs. A request past the bound can never be scheduled at any
  future time, however much drains, so `add_request` fails it immediately instead of
  queueing it: no prefill is spent, nothing is preempted, and the client gets a
  terminal response rather than a request that never returns.

So there is one bound with two call sites, which is the point:
`_estimate_max_model_len_from_groups` is called by the startup check
(`kv_cache_utils.py:2234`, via `_check_enough_kv_cache_memory`) and now by
`max_servable_num_tokens`, which is what the scheduler admits against. Neither side
can grow its own formula without the other seeing it.

The failure uses `RequestStatus.FINISHED_IGNORED`, which `vllm/v1/request.py` already
documents as the status for "requests whose prompt lengths are longer than the
model's length cap", mapping to `finish_reason="length"`. `FinishReason.ERROR` is
documented as a *retryable* internal error always converted to a 500, which is the
wrong contract here. Returning a literal 400 would mean exposing the pool-derived
bound to `input_processor._validate_prompt_len`, where `max_model_len` is validated;
happy to do that in a follow-up if maintainers prefer it — it is a frontend/engine-core
plumbing change and I kept it out of this PR.

Behaviour is unchanged whenever the pool covers `max_model_len`, which is every
correctly-sized deployment: the bound is then `>= max_model_len`, `_is_unservable`
short-circuits on the first comparison, and nothing is rejected. The 70-block row
above is byte-for-byte identical before and after.

### Relationship to #47272 — scoped around it, not a duplicate

I read #47272 ("Reserve the KV null block when validating `max_model_len`") before
writing any code, and this PR is deliberately scoped around it.

- **#47272 owns the capacity side.** It subtracts one block from the memory the
  startup check validates against, so a pool that cannot serve `max_model_len` is
  refused at boot with the existing message naming the estimated maximum model
  length. That is the right place for it and this PR does not touch that code path.
  Applied to the reproduction above, #47272 makes the 69-block pool illegal at
  startup — which is a better outcome than anything the scheduler can do, and is why
  I am not proposing an alternative to it.
- **This PR owns the request side.** It makes the runtime bound the same function as
  the startup bound, and gives the engine an answer for a request it cannot serve
  other than retrying it. That is needed independently of #47272, because the
  scheduler currently has *no* concept of "this request can never be scheduled": it
  only distinguishes "not now". If #47272 lands, this PR's check becomes dormant on
  correctly-sized pools (bound `>= max_model_len`) and remains the backstop for the
  paths #47272 does not cover — notably `_auto_fit_max_model_len`, which runs before
  #47272's reservation and can still land on the boundary.
- The two compose cleanly and neither subsumes the other. **Please do not merge this
  instead of #47272.**

Other prior art I checked: **#40946** (merged) is the design precedent — it fixed the
same class of defect (startup pool sizing and runtime admission using different
formulas) for SWA/chunked-local by putting the bound in one method called from both
sites, and this PR follows that shape. **#35541** (closed not-planned/stale) asked for
"abort requests that can never fit in the KV cache block pool with a clear error
rather than hanging indefinitely" — this is that, for the mechanism above. **#40384**
is why the numbers here are stated in blocks and tokens rather than inverted from the
`GPU KV cache size` log line, which on a hybrid model is
`num_blocks // len(kv_cache_groups) * min_block_size`.

### Scope of the claim

The field trace in #52520 came from a **downstream fork build** on a hybrid Qwen3.5
model (261,794-token request, `--enable-prefix-caching --mamba-cache-mode align`, MTP
depth 3, pool within ~1 % of the window; 63 full re-prefills over 656 s with zero
output tokens). The mechanism above is upstream code, and everything in this PR is
demonstrated on **stock `main` at `4d2a68d64d9e05921ed5c4099146e768a92d71d5`** by the
CPU-only unit test below — no GPU, no model weights, no fork code involved.

One part of the original report I could **not** reproduce on `main` and am therefore
not claiming: `vllm:num_preemptions_total` staying at `0.0` across the re-prefills.
On `main` the self-preemption path does increment `request.num_preemptions` and record
`EngineCoreEventType.PREEMPTED`, which `IterationStats` counts into
`vllm:num_preemptions_total`. So there is no accounting fix here, and I am not filing
one. (With this PR the point is moot for this defect: nothing is preempted, because
nothing is admitted, and the operator gets two `ERROR` lines naming the limit.)

## Test Plan

```bash
pytest tests/v1/core/test_kv_pool_unservable_requests.py
# surrounding suite, unchanged behaviour:
pytest tests/v1/core/
```

`tests/v1/core/test_kv_pool_unservable_requests.py` is CPU-only
(`pytestmark = pytest.mark.cpu_test`), built the same way as
`tests/v1/core/test_mamba_align_chunk_split.py`: a real `Scheduler` over a hand-built
hybrid `KVCacheConfig`, driven with `schedule()` / `update_from_output()`. No model
weights, no GPU. Five cases:

1. `test_max_servable_num_tokens_reserves_the_null_block` — the bound is
   `(69 - 1 - 5) * 16 = 1008` at the startup minimum, and `>= max_model_len` with one
   block of headroom.
2. `test_unservable_request_is_failed_not_prefilled_and_retried` — the regression
   test. A 1023-token request on the 69-block pool: nothing is scheduled, it is
   finished as `FINISHED_IGNORED`, `num_computed_tokens == 0`,
   `num_preemptions == 0`, and no unfinished request is left behind.
3. `test_request_at_the_bound_is_failed_because_it_needs_an_output_slot` — the bound
   is a sequence length, so a prompt of exactly 1008 is already too long (it was the
   one length that prefilled fully on `main` and then stalled on its first decode
   step).
4. `test_servable_request_at_the_same_pool_still_runs` — 1007 tokens on the *same*
   69-block pool prefills fully, decodes, and finishes. The bound must not
   over-reject.
5. `test_pool_with_headroom_serves_the_whole_window` — 1023 tokens on a 70-block pool
   behaves exactly as before this PR.

## Test Result

The new file, checked out on top of unmodified `main` at `4d2a68d6` (the fix reverted,
the test kept), fails:

```text
$ git checkout HEAD -- vllm/ && pytest tests/v1/core/test_kv_pool_unservable_requests.py -q
FAILED test_max_servable_num_tokens_reserves_the_null_block
  - AttributeError: 'Scheduler' object has no attribute 'max_servable_num_tokens'
FAILED test_unservable_request_is_failed_not_prefilled_and_retried
  - AssertionError: assert not {'0': 256}
FAILED test_request_at_the_bound_is_failed_because_it_needs_an_output_slot
  - AssertionError: assert not {'0': 256}
3 failed, 2 passed, 14 warnings in 3.48s
```

`{'0': 256}` is `main` admitting the unservable request and scheduling the first
256-token prefill chunk of the sequence it will then throw away. The two that pass
before and after are the two "must still work" directions — they are there to catch
over-rejection, and they must never have gone red.

With the fix:

```text
$ pytest tests/v1/core/test_kv_pool_unservable_requests.py -q
.....                                                                    [100%]
5 passed, 14 warnings in 3.83s
```

Surrounding suite, same command on `4d2a68d6` and on this branch:

```text
main @ 4d2a68d6 : 2 failed, 495 passed, 2 errors in 348.75s
this branch     : 2 failed, 500 passed, 2 errors in 351.08s
```

The same two failures and the same two setup errors occur on both, all environmental
in my CPU-only sandbox and unrelated to this change: failures in
`test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders`
and `test_reset_prefix_cache_e2e.py::test_reset_prefix_cache_e2e`, and setup errors on
`test_concurrent_partial_prefill` and `test_prefix_cache_stats_is_recorded` — all four
need a GPU-backed engine core. The +5 are the new tests.

Two existing tests build a `Scheduler` with `object.__new__` and set its attributes by
hand; they each gain one line for the new `unservable_reqs` set
(`tests/v1/core/test_scheduler.py`, `tests/v1/core/test_async_scheduler.py`).

Lint gates on the changed files:

```text
$ ruff check <changed files>          # 0.14.0, as pinned in .pre-commit-config.yaml
All checks passed!
$ ruff format --check <changed files>
5 files already formatted
$ typos --force-exclude <changed files>
(clean)
$ python tools/pre_commit/mypy.py 3.12 <changed files>   # mypy 1.20.2
Success: no issues found in 2 source files
Success: no issues found in 3 source files
```

---

AI assistance was used to investigate and draft this change; every changed line was
reviewed, and the reproduction, the tests and the lint gates were run locally as
described above.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model. — none needed: no new model, flag, or user-facing configuration.
</details>
