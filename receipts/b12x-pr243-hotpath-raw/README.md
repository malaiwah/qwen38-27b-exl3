# b12x PR 243 service artifacts

The authoritative no-MTP A-B-A artifacts are:

- `no-mtp-cacheproof-baseline-a1.json`
- `no-mtp-cacheproof-candidate-b1.json`
- `no-mtp-cacheproof-baseline-a2.json`

They use distinct exact-length token-ID prompts. Every warmup and timed prefill
request fails unless vLLM reports 2,051 prefix-cache queries, zero cache hits,
2,051 locally computed prompt and KV tokens, no external/local cached tokens,
and one completed prefill request.

The older `no-mtp-baseline-a1.json`, `no-mtp-candidate-b1.json`, and
`no-mtp-baseline-a2.json` files are retained as superseded audit artifacts.
Their decode samples remain raw historical observations, but their
approximately 17k prompt-token rates are not uncached prefill: the old harness
warmed and timed the same prompt with prefix caching enabled.

The MTP-3 reachability log and sanity response are unaffected by this
correction.
