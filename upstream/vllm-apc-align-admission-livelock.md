# [Bug] `--mamba-cache-mode align`: the scheduler admits a request it can never schedule, then re-prefills it forever with zero output

## Summary

On a hybrid Mamba model with `--enable-prefix-caching --mamba-cache-mode align`, a request
whose length is close to the KV pool is **accepted** by admission, prefilled to ~99 % of the
pool, dropped back to the waiting queue with the pool freed, and prefilled again — forever.

The cycle has a fixed 30-second period. Prompt throughput pins at ~960 tok/s, generation
throughput is `0.0` for the entire run, and **not one output token is ever produced**. The
HTTP request never returns and the server never recovers; it is not wedged, it is busy, which
makes it look like a very slow request rather than a hang.

Two things make this worse than a plain admission misjudgement:

1. **`vllm:num_preemptions_total` stays at `0.0`.** Nothing is preempted, because nothing is
   ever admitted. The only signal is
   `vllm:num_requests_waiting_by_reason{reason="capacity"} = 1`, which looks identical to a
   healthy server that is momentarily full.
2. **The partial prefill is discarded rather than published to the prefix cache.** Over the
   run the engine logged **261,794 prefix-cache queries and exactly 0 hits** — the cache was
   consulted the whole time and had nothing to offer. So every retry costs full price and no
   retry can ever cost less. This is what converts a one-off admission error into an
   unbounded one, and it is the more important of the two defects: admission accepting
   something it cannot schedule would show up as one failed request, but not publishing the
   partial work is what produces a server that never emits another token.

## Environment

- Build: `vllm 0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34`
- `vllm/v1/core/sched/scheduler.py` is upstream's post-#51113 file
  (`sha256 b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf`), i.e. **the
  mamba-align prefill-split fix is already applied**; this is not #43559 resurfacing.
- Model: Qwen3.8-27B (hybrid Mamba + attention), EXL3 weights, `--kv-cache-dtype fp8`
- Hardware: one physical NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.57.04, CUDA UMD 13.3
- Flags: `--max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --enable-prefix-caching --mamba-cache-mode align`
- Banner confirms the flags took: `'enable_prefix_caching': True`, `'mamba_cache_mode': 'align'`,
  plus `Prefix caching in Mamba cache 'align' mode is currently enabled`.

## The arithmetic, and why the admission check does not catch it

`Setting attention block size to 1600 tokens` (mamba page size forces it). With align mode a
request occupies **whole blocks**:

```
prompt                       261,794 tokens
rounded to whole blocks      ceil(261794 / 1600) = 164 blocks = 262,400 slots
plus MTP-3 draft slots       + a decode block
KV pool reported by engine   265,072 tokens   ("GPU KV cache size", 1.01x at 262,144)
```

The pool *looks* like it has 3,278 tokens of headroom, and startup's own check passes. It is
not enough once the request is rounded and the draft slots are added, and the engine only
discovers this at the final allocation of the prefill — after it has already accepted the
request and done ~260 k tokens of work.

The startup check is separately visible one notch lower. At `--gpu-memory-utilization 0.955`
the same engine **refuses to start**, with a message that quantifies the rounding exactly:

```
To serve at least one request with the model's max seq len (262144), 9.29 GiB KV cache is
needed, which is larger than the available KV cache memory (9.28 GiB). Based on the available
memory, the estimated maximum model length is 260800.
```

So the engine knows the requirement is `9.29 GiB` when it checks at startup, and yet at a
utilisation where startup passes, admission still accepts a request it cannot finish. The
startup check is **necessary but not sufficient**: it validates one request against the pool
without the draft slots and the decode block that admission will later demand.

## Measured band, one engine start per row, same flags each time

| `--gpu-memory-utilization` | available KV | GPU KV cache size | max concurrency @262,144 | result |
|---|---|---|---|---|
| 0.955  | 9.28 GiB | — | — | **refuses to start**; needs 9.29 GiB; suggests max len 260,800 |
| 0.9555 | 9.30 GiB | 262,144 | 1.00x | starts, then **deadlocks** mid-prefill, `reason="capacity"`, 0 tok/s |
| 0.958  | 9.38 GiB | 263,608 | 1.01x | starts (gates not run) |
| 0.9585 | 9.39 GiB | 265,072 | 1.01x | starts, then **livelocks**: 30 s re-prefill cycle, 0 output tokens |
| 0.959  | 9.41 GiB | 265,072 | 1.01x | starts (identical pool) |
| 0.96   | 9.44 GiB | 265,072 | 1.01x | starts (identical pool) |

The pool is quantised: 0.9585, 0.959 and 0.96 all yield the same 265,072 tokens, so **there is
no utilisation on this card that makes a 262,144-token request schedulable under align mode**.
The same build with `--max-model-len 256000` at 0.9585 (pool 264,777, 1.03x) serves a
255,000-token needle exactly in 180 s, three warmed decode runs, and a second long request
after release — so the feature itself is sound, and it is specifically the
request-close-to-pool case that fails.

## Observed cycle

```
05:37:17 pp=800.0 tg=0.0 run=1 wait=0 kv=98.9
05:37:27 pp=959.9 tg=0.0 run=0 wait=1 kv=0.0
05:37:37 pp=960.0 tg=0.0 run=0 wait=1 kv=0.0
05:37:47 pp=799.9 tg=0.0 run=1 wait=0 kv=98.9
05:37:57 pp=960.0 tg=0.0 run=0 wait=1 kv=0.0
05:38:07 pp=960.0 tg=0.0 run=0 wait=1 kv=0.0
05:38:17 pp=800.0 tg=0.0 run=1 wait=0 kv=98.9
```

63 such lines. Stopped by the operator after 656 s; nothing suggests it would ever terminate.
Metrics at the moment of teardown:

```
vllm:num_requests_running                                   0.0
vllm:num_requests_waiting_by_reason{reason="capacity"}      1.0
vllm:num_preemptions_total                                  0.0
vllm:prefix_cache_queries_total                       261,794.0
vllm:prefix_cache_hits_total                                0.0
```

## What we would expect instead

Any one of these would turn an unbounded failure into a bounded one, in rough order of value:

1. **Publish the partial prefill to the prefix cache before requeueing.** The blocks are
   computed and valid; discarding them is what makes the loop stable. With them published,
   the retry starts from a high hit rate and may well succeed.
2. **Reject at admission.** If the rounded, draft-inclusive requirement of a request exceeds
   the pool, fail the request with a 400-class error naming the maximum servable length —
   the same information the startup check already prints — rather than accepting it.
3. **Account for it.** A request that is descheduled after partial prefill should increment
   a preemption or restart counter. Right now the only observable is a `waiting` gauge that
   is indistinguishable from healthy back-pressure.
4. **Tighten the startup check** to include the align-mode rounding, the speculative draft
   slots and the decode block, so a configuration that cannot serve one native-length request
   is refused at boot rather than at request time.

## Scope of this report

We can support this **on the fork build above**. We have not reproduced it on stock vLLM and
do not claim it there — but the file involved is upstream's own post-#51113 scheduler, the
flags are upstream's, and nothing in the mechanism is specific to EXL3 weights or to this
fork's kernels, so we would expect any hybrid-Mamba deployment running align mode with a
request near its pool ceiling to hit it. If a maintainer wants a stock-vLLM reproduction we
will attempt one on request.

## Adjacent finding, from a different investigation

A colleague auditing the same scheduler bytes found a separate path in admission that also
inflates a request's cost: `scheduler.py:865-878` pads a request admitted needing exactly one
new token up to `1 + num_spec_tokens` and writes `[-1] * num_spec_tokens` into
`scheduled_spec_decode_tokens`, and `872-877` then declines to schedule at all if the inflated
count exceeds the budget, rather than scheduling it un-padded. A **full prefix-cache hit** is
precisely how `num_new_tokens == 1` arises.

This is reported as adjacent, **not** as the explanation of the trace above: in our livelock
the prefix cache never hit (261,794 queries, 0 hits), so `num_new_tokens` was never 1 and that
path cannot have fired. The two share the admission decision and nothing else, and they should
not be conflated. Details and its own evidence are in that investigation's receipt,
`receipts/gdn-gate-concurrency.json` (commit `cc12846`), not restated here.

## Evidence

Full artifacts, including the 63-line timeline, the per-probe engine banners, every launch
argv and every server log: `receipts/qualification-5090-apc.json` and
`receipts/qual-apc-raw/` in <https://github.com/malaiwah/qwen38-27b-exl3>.
