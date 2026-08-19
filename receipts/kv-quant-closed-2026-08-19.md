# Sub-8-bit KV cache: axis closed, with one catastrophic positive

**Date:** 2026-08-19 (late)
**Question:** can a cheaper KV cache buy back the context that `balanced` lacks
(199,104 served vs the 238,400 criterion), or the 2.87 GiB that blocks a resident
FP8 configuration?

**Answer: no.** Three of the four candidate dtypes cannot be selected at all, and the
one that works trades away 3.8x of decode throughput.

## The mechanism: two disjoint dtype sets

Read from the served container, not the host checkout:

`TritonAttentionBackend.supported_kv_cache_dtypes` (`v1/attention/backends/triton_attn.py:278`):
```
auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2,
int4_per_token_head, int8_per_token_head, fp8_per_token_head
```

`TurboQuantAttentionBackend.supported_kv_cache_dtypes` (`v1/attention/backends/turboquant_attn.py:101`):
```
turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc
```

**The two sets are disjoint.** Our model is hybrid (16 full-attention + 48
linear-attention layers), and that architecture selects TRITON_ATTN; the dtype is then
validated *against the already-selected backend* rather than re-routing the selection.
Hence the failure mode is identical for all three turboquant/nvfp4 attempts:

```
ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for
this configuration. Reason: ['kv_cache_dtype not supported']
```

`nvfp4` appears in **neither** list, so it is unavailable regardless of backend.

There is a second, independent reason the turboquant path was never viable here:
`turboquant_attn.py:205` initialises with `supports_spec_as_decode=False`. Even if the
backend were selectable, it would disable MTP-as-decode — the mechanism responsible for
our entire TG advantage. The axis was doubly closed.

## The one that boots: int4_per_token_head, and why it is unusable

| metric | balanced @ fp8_e4m3 | balanced @ int4_per_token_head | delta |
|---|---|---|---|
| max_model_len | 199,104 | **238,400** | criterion met |
| KV cache memory | — | 7.61 GiB | — |
| PP | 3925.2 | 3773.0 | −3.9% |
| **TG-fox** | **215.6** | **57.0** | **−73.6%** |
| **MTP acceptance (fox)** | **1.000** | **0.096** | **−90.4%** |
| vision | true | true | — |

It delivers exactly the context we wanted and is still useless.

**Prefill barely notices** (−3.9%) because prefill is compute-bound. **Decode
collapses** because acceptance collapses: at 0.096, roughly nine drafts in ten are
rejected, so the speculative path becomes pure overhead.

## This is the coupling law again, in its strongest form yet

Three independent confirmations now that **fidelity and decode speed are coupled
through MTP acceptance**, and this is the largest effect of the three:

| perturbation | acceptance (fox) | TG-fox |
|---|---|---|
| none (all-trellis) | 1.000 | 228.3 |
| self_attn weights to FP4 | 0.967 | — |
| all weights to FP4 | 0.930 | 187.4 |
| **KV cache to int4** | **0.096** | **57.0** |

Weight quantization degrades acceptance gently. **KV quantization degrades it
catastrophically**, and the reason is structural: weight error is fixed per matrix,
but KV error is *re-read on every subsequent token*, so it compounds along the
sequence exactly where the draft head is trying to predict. This is the same
compounding that made the additivity model fail for attention (+46% versus prediction)
while holding for MLP (−1.9%, −6.0%). The two anomalies have one cause.

## Consequences

1. **The 2.87 GiB shortfall blocking a resident FP8 configuration cannot be recovered
   from the KV cache.** It is real memory, not slack.
2. `balanced` cannot reach 238,400 by this route, and cannot pass all six criteria
   anyway (PP 3925.2 versus the 7000 bar).
3. Still untested and *not* a context play: `fp8_per_token_head` and
   `int8_per_token_head` are 8-bit, the same size as our current `fp8_e4m3`, so they
   offer no context gain — but per-token-head scaling could be *more accurate at
   identical memory*. That is a fidelity opportunity rather than a capacity one, and
   it is being measured now.
