# POSTED comment on vllm-project/vllm#46778 — "[Docs][Core] Document Mamba/hybrid prefix caching and surface how to enable it"

<!--
Target: https://github.com/vllm-project/vllm/pull/46778 (open docs PR, author alikhan126)
Status: POSTED 2026-08-16 -> https://github.com/vllm-project/vllm/pull/46778#issuecomment-5307331622 (approved by Main).
Value: this PR adds the "how to enable it" doc that did not exist. The one thing an operator hits
five minutes after following it, on a long-context deployment, is that the engine now refuses to
start at the window that worked yesterday. That is a documentation gap this PR could close in two
sentences, and we have the measured numbers for it.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
Strong +1 on the discoverability half; we lost time to exactly the stale note this PR rewrites.

One thing worth a sentence in `v1_guide.md`, because it is the first thing that happens after an
operator follows the new instructions on a long-context deployment: **turning on prefix caching
reduces the maximum servable context length**, and the error message does not say that prefix caching
is the reason.

Measured on a hybrid Qwen3.5-architecture model, one physical RTX 5090 (32,607 MiB), unchanged
profile, the only edit being `--enable-prefix-caching --mamba-cache-mode align` added to a
configuration that had been serving a 262,144-token window:

```text
To serve at least one request with the model's max seq len (262144), 9.29 GiB KV cache is needed,
which is larger than the available KV cache memory (9.28 GiB). Based on the available memory, the
estimated maximum model length is 260800.
```

Supply did not move: available KV is 9.28 GiB with the cache on and off, weights 18.19 GiB, peak
activation 1.78, non-torch 0.27, CUDAGraph 0.45 all unchanged. What moved is the requirement, because
`align` makes a request occupy whole mamba blocks and the mamba page size had already pushed the
attention block size to 1,600 tokens. The price on this card is either +0.0005 of
`--gpu-memory-utilization` or 1,344 tokens of window (262,144 → 260,800, 0.51 %). That is a fair
price and we paid it deliberately — but it is invisible until the engine refuses to boot, and the
message reads like an out-of-memory problem rather than "the mode you just enabled rounds requests up
to whole blocks".

Suggested addition, wherever the `align` paragraph lands:

> Because `align` mode allocates whole Mamba blocks per request, enabling prefix caching slightly
> increases the KV cache required to serve one `max_model_len` request. On a deployment already
> sized close to its limit, the engine may refuse to start with "…KV cache is needed, which is larger
> than the available KV cache memory…" even though nothing else changed. Either raise
> `--gpu-memory-utilization` a little or lower `--max-model-len` to the value the message suggests.

I would add a caution too, if you are open to it: on the same card there is a band just above that
boundary where the engine *does* start and then cannot serve a request at `max_model_len` at all —
one utilisation step above the refusal it hangs at 0 tok/s, and the step above that it re-prefills
the request forever with zero output tokens. Six engine starts, one per utilisation value, in
`receipts/qualification-5090-apc.json` at <https://github.com/malaiwah/qwen38-27b-exl3>. That is a
bug rather than a doc gap (closest existing report: #47272 for the exact-boundary row), so I would
not put the band in the docs — but "prefix caching plus a window at the pool ceiling is not a
supported combination yet" may be worth one line so nobody tunes into it.

Build for the numbers above: downstream fork of vLLM `0.11.2.dev280+…20260810.r34` (upstream
integration tree `4d006a43`); the config, sizing and scheduler files involved are unmodified vendored
upstream code in that image.
