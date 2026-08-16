# POSTED comment on vllm-project/vllm#45238 — "Hybrid-model prefix caching silently drops to 0% when the align-mode Mamba checkpoint lands in request-unique tokens"

<!--
Target: https://github.com/vllm-project/vllm/issues/45238 (open issue, author Sahil170595)
Status: POSTED 2026-08-16 -> https://github.com/vllm-project/vllm/issues/45238#issuecomment-5307331518 (approved by Main).
Value: an independent single-request corroboration on a different model and a 1,600-token block
size, plus a note that the failure is not only a performance loss — it is what makes an unrelated
admission retry loop unbounded. Also records that we are retracting our own claim of novelty.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
Independent corroboration on a different model, and a consequence of this bug that is worse than a
lost hit rate.

Build: downstream fork of vLLM (`0.11.2.dev280+…20260810.r34`), one physical RTX 5090,
Qwen3.5-architecture hybrid (GDN linear attention + full attention), `--enable-prefix-caching
--mamba-cache-mode align`, MTP-3, `--max-model-len 262144`, `--max-num-seqs 1`. Align mode set the
attention block size to 1,600 tokens here, i.e. coarser than any figure in your table, which is the
direction you predicted gets worse.

**Corroboration.** A single 261,794-token request, repeatedly re-prefilled by the scheduler in the
same process (for an unrelated admission reason), produced
`vllm:prefix_cache_queries_total = 261,794` against `vllm:prefix_cache_hits_total = 0.0` — the cache
was consulted on every retry, over a prompt that was byte-identical every time, and never returned a
single hit. That is consistent with your mechanism: only ~1–2 mamba block hashes are ever registered
per request, `HybridKVCacheCoordinator` requires every group to hit, and a mamba miss vetoes the
attention groups' matches. A hostile-prefix probe on the same image at an 8,192-token window (block
size small enough that the checkpoint lands inside the shared prefix) reached a 60 % hit rate, so the
zero is not "prefix caching is off"; it is geometry, exactly as you describe.

**Why it is not only a performance bug.** In our trace the 0 % hit rate is what makes an unrelated
admission failure *unbounded*. A request near the KV-pool ceiling was accepted, prefilled to 98.9 %
of the pool, requeued, and re-prefilled on a 30-second period forever, at ~960 tok/s with zero
output tokens. If any of the partial prefill had been reusable, the retry would have started from a
high hit rate and could have converged. Because nothing is ever published, every retry costs full
price and the loop is stable. So your option **(c)** — a counter for "attention-side prefix match
vetoed by missing mamba checkpoint" — would have named our failure in one line instead of the two
days it took to reach it, and I would argue it is worth landing on its own even before (a) or (b).

For the record, we had written the discarded-partial-prefill observation up as a separate finding in
our own tracker before finding this issue. It is not separate; it is this. Cross-referenced and
retracted on our side.

Trace, argv, engine banners and the `/metrics` scrape:
`receipts/qualification-5090-apc.json` in <https://github.com/malaiwah/qwen38-27b-exl3>.
Not reproduced on stock vLLM — fork build, stated for the record; the coordinator, block-pool and
mamba-manager files involved are unmodified vendored upstream code in that image.
