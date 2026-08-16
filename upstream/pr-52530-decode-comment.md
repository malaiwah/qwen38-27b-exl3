Measured: it livelocks. The decode case is the same defect, not a deliberate exclusion, and it is now fixed in this PR.

## Reproduction on unmodified upstream `main` @ `4d2a68d64d9e05921ed5c4099146e768a92d71d5`, CPU only

Same harness this PR was gated on, with the sequence split across the prefill/decode boundary instead of living entirely in the prompt:

```
block_size=16  max_model_len=1024  num_speculative_blocks=3  max_num_batched_tokens=256
pool = 69 blocks == exactly what get_kv_cache_configs accepts;  max_servable_num_tokens = 1008

request: prompt=1000  max_tokens=24  ignore_eos
  frontend        1000 + 24 = 1024 <= 1024   accepted   (renderers/params.py:210, :442)
  _is_unservable  1000 +  1 = 1001 <= 1008   admitted
  decode wall     1008 is reached at output token 9
```

Your example, one order of magnitude down. The trace:

```
step  0 sched= 256 computed= 256 seq_len=1000 out= 0 free=48 RUNNING    preempt=0
step  3 sched= 224 computed= 992 seq_len=1000 out= 0 free= 1 RUNNING    preempt=0
step  4 sched=   8 computed=1000 seq_len=1001 out= 1 free= 0 RUNNING    preempt=0
step  5 sched=   1 computed=1001 seq_len=1002 out= 2 free= 1 RUNNING    preempt=0
...
step 11 sched=   1 computed=1007 seq_len=1008 out= 8 free= 1 RUNNING    preempt=0
step 12 sched=   1 computed=1008 seq_len=1009 out= 9 free= 1 RUNNING    preempt=0
step 13 sched=   0 computed=   0 seq_len=1009 out= 9 free=68 PREEMPTED  preempt=1
step 14 ... step 3999   sched=0  computed=0  out=9  free=68  PREEMPTED  preempt=1
```

**Observation window: 4000 scheduler steps.** 3987 of them scheduled zero tokens and produced zero output; the last 3986 also freed nothing, with the whole 68-block pool sitting idle. This is not slow-but-progressing, it is stopped.

Two controls in the same run, so the harness is not the thing being measured:

| run | pool | request | result |
|---|---|---|---|
| decode-over-ceiling | 69 blocks | prompt 1000, max_tokens 24 | **never finishes**, 9/24 tokens, stalled from step 13 to step 3999 |
| control, one spare block | 70 blocks | prompt 1000, max_tokens 24 | finishes 24/24 in 28 steps, 0 preemptions |
| control, under the ceiling | 69 blocks | prompt 1000, max_tokens 8 | finishes 8/8 in 12 steps, 0 preemptions |

One correction to the signature I expected. `vllm:num_preemptions_total` increments **once** and then stops — it does not climb. The request is preempted at token 1009, and 1009 > 1008 means it is now too long for its own re-prefill, so it collapses straight into the state #52520 describes: `scheduled=0` forever, `waiting` non-empty, pool idle, one `EngineCoreEventType.PREEMPTED` event ever. The decode case does not have a distinct failure mode; it walks into the prefill case.

## The fix, and why not the bound you suggested

`num_tokens + max_tokens` at admission does catch this request. It also refuses every request that asks for generous headroom and stops long before the wall — and the third row of the table above *is* that request: `prompt=1000, max_tokens=8` runs to completion on the very same 69-block pool. Rejecting on `max_tokens` trades a livelock for a false rejection on the common path, and callers routinely send `max_tokens` an order of magnitude above what the model actually emits.

So the bound belongs at the point of exhaustion, and it is one line:

```diff
-            stopped = check_stop(request, self.max_model_len)
+            stopped = check_stop(request, self.max_servable_num_tokens)
```

`max_servable_num_tokens <= max_model_len` holds by construction — `_estimate_max_model_len_from_groups` binary-searches with `right = original_max` (`kv_cache_utils.py:1960` on main) — and the scheduler already falls back to `max_model_len` itself when there is no per-block bound to price. So on every pool that covers its window this is byte-for-byte the call it was. Where they differ, the request stops on the pool's ceiling and returns what it produced with `finish_reason: length`, exactly as if `max_model_len` had been set to what the pool can actually serve. In the repro it now finishes 8/24 in 12 steps with zero preemptions instead of freezing at 9/24 forever.

I considered failing it with an error at exhaustion instead, and rejected that too: by the time the ceiling is hit the client has already been streamed 8 tokens, and `length` is the truthful reason — the sequence hit a context limit, just a lower one than advertised. The operator gets the actionable version once, at startup, from the `logger.error` that already fires; it now also names the generation ceiling.

That makes it one bound with three call sites — startup sizing, admission, and the length cap — which is the shape #40946 settled on.

## Gap width

You asked whether the gap can be wider than a block or two. I measured rather than argued, and the answer corrects something I wrote earlier in this thread.

**The gap is exactly one block, never more**, and it is `((max_model_len - 1) mod block_size) + 1` tokens — i.e. however much of its last block `max_model_len` occupies. It is also zero as soon as the pool has one block more than the startup minimum. Fourteen configurations, on unmodified `main`:

```
  block max_model_len nspec  need smallest_ok servable  gap_tok gap_blk | +1blk_servable +1blk_gap
     16          1024     3    69          69     1008       16    1.00 |           1024         0
     16          1024     0    66          69     1008       16    1.00 |           1024         0
     16          1000     3    68          68      992        8    0.50 |           1000         0
     16          1009     3    69          69     1008        1    0.06 |           1009         0
     16          4096     3   261         261     4080       16    1.00 |           4096         0
     32          1024     3    37          37      992       32    1.00 |           1024         0
     64          1024     3    21          21      960       64    1.00 |           1024         0
    128          1024     3    13          13      896      128    1.00 |           1024         0
    128          8192     3    69          69     8064      128    1.00 |           8192         0
    128          8192    16    82          78     8192        0    0.00 |           8192         0
     16          8192    16   530         526     8192        0    0.00 |           8192         0
    256          8192     8    42          38     8192        0    0.00 |           8192         0
    256          8192     0    34          37     7936      256    1.00 |           8192         0
    512         32768     8    74          70    32768        0    0.00 |          32768         0
```

Contributors, with what each one is actually worth:

- **The null block — the entire gap.** `BlockPool` pops one block at `core/block_pool.py:190`, and `_check_enough_kv_cache_memory` is handed the byte count for *all* blocks (`kv_cache_utils.py:2193`). Since that check also refuses any pool below the minimum, the pool is short by exactly one block or by nothing at all. One block, never two.
- **Block rounding — decides how much of that block is lost.** `_max_memory_usage_bytes_from_groups` sums `cdiv(spec_bytes, page_size)` per group (`kv_cache_utils.py:1933`) and `FullAttentionSpec.max_memory_usage_bytes` is `cdiv(max_model_len, block_size) * page_size` (`kv_cache_interface.py:305`), so the lost block costs whatever `max_model_len` occupies in its last block: the full `block_size` when it divides evenly, less otherwise. Rows 3 and 4 above are `max_model_len` 1000 and 1009 at `block_size` 16 — 8 tokens and 1 token.
- **The speculative reserve — not a contributor. I was wrong about this above.** `MambaSpec.max_memory_usage_bytes` charges `2 + num_speculative_blocks` blocks in align mode (`kv_cache_interface.py:694`), but it charges the same amount on both sides of the comparison, so it moves the requirement and the ceiling together. Rows 1 and 2 are `num_speculative_blocks` 3 and 0 at the same block size, and both gaps are 16.
- **One spare block closes it completely**, because the estimator's binary search is capped at `max_model_len` (`kv_cache_utils.py:1960`). Every row's right-hand columns are 0.

So the gap is bounded by `block_size` rather than unbounded — but `block_size` is 128 or 256 in plenty of real deployments, and the `block_size=128, max_model_len=8192` row lands at **8064 vs 8192**, which is within touching distance of the 8000/8192 you wrote by hand. Your read on the magnitude was right; the mechanism is one block, not an accumulation of three.

## One thing I checked and deliberately did not change

The running-request path asks the allocator for `1 + len(spec_token_ids)` slots, so a draft proposal can reach past the ceiling before the length cap can fire. I injected drafts into the same harness on stock `main`: the request preempts twice around the ceiling and then **finishes**, because preemption clears `spec_token_ids` (`sched/scheduler.py:1389`) and the retry decodes one token at a time straight into the cap. Bounded and self-correcting, identical before and after this PR. I left `scheduler.py:605` alone rather than widen the diff on something I could not show was broken.

## Tests

Two new cases in the existing file, same style:

- `test_generation_stops_at_the_bound_instead_of_stalling_mid_decode` — asserts the request clears both length checks, then must finish `FINISHED_LENGTH_CAPPED` at `SERVABLE_LEN` with `num_preemptions == 0` and nothing left unfinished. With only the `check_stop` line reverted: **1 failed, 6 passed**, `assert RequestStatus.PREEMPTED == RequestStatus.FINISHED_LENGTH_CAPPED`.
- `test_generation_is_not_capped_when_the_pool_covers_the_window` — the same request with one block of headroom must still produce all 24 tokens, so the cap cannot silently truncate generation on the common path. This one passes both before and after, by design.

On unmodified `main` the file is 5 failed, 2 passed; on this branch, 7 passed. `tests/v1/core/` is 2 failed, 502 passed, 2 errors — the same four GPU-requiring cases as before, unchanged.

Thanks again for tracing `unservable_reqs` through to `has_unfinished_requests`; the decode path deliberately does not use that deferred set — it finishes through the ordinary `stopped` path in `_update_request_with_output` — but knowing the first one was sound is what let me leave it alone and put this in a different place.
