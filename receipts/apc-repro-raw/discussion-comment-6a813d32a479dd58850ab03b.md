Follow-up with actual numbers, on the same card class as yours.

**Headline: I could not reproduce it.** I ran your exact serving condition on a physical RTX 5090 (32,607 MiB, driver 610.57.04) — `--max-model-len 196608 --gpu-memory-utilization 0.92 --max-num-seqs 1 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --enable-prefix-caching --mamba-cache-mode align`, MTP-3, `FULL_DECODE_ONLY` with capture size 4, and your three `VLLM_EXL3_*` variables — across seven freshly started servers, 38 requests each, 266 scored requests total. **Zero corrupted responses, zero wrong answers, zero acceptance collapses.**

That matters because of something in your last message: your decisive control removed **four** things at once — the LMCache wrapper and connector, the cleared L2 files, `--enable-prefix-caching`, and `--mamba-cache-mode align`. So it proved the set was guilty, not which member. My run is the missing half of that decomposition: **prefix caching + align mode, on the unpatched image, without LMCache, stayed clean.**

The probe was built to be hostile, not gentle. Every request is a nested token prefix of one document, so later longer requests hit blocks published by earlier shorter ones. No prompt length is a multiple of the 1600-token mamba block (measured on this model at startup), so between one and seven prefill chunk ends per request land mid-block — the exact window the missing fix fails to align. Four passes, including pairs whose shared prefixes are 2900/3500, 4500/5100, 6100/6700 and 7700/8300 tokens, so the second of each pair crosses precisely the boundary the first stopped short of. The condition was genuinely live: arm A's engine line records `'enable_prefix_caching': True, 'mamba_cache_mode': 'align'`, and it finished at a **62.9 % prefix cache hit rate**.

| arm | image | prefix caching | seqs | failing / 38 | corrupted | acceptance median | log hit rate |
|---|---|---|---|---|---|---|---|
| A | release, **#51113 absent** | on, align | 1 | **0** | 0 | 63.0 % | 62.9 % |
| B | release | off | 1 | 0 | 0 | 64.0 % | 0 % |
| Bn | release (null repeat of B) | off | 1 | 0 | 0 | 64.2 % | 0 % |
| C | superset, **#51113 present** | on, align | 1 | **0** | 0 | 64.3 % | 62.7 % |
| D | C + #51812 GDN module | on, align | 1 | 0 | 0 | 63.6 % | 62.9 % |
| A2 | release | on, align | 4 | 0 | 0 | 63.5 % | 61.8 % |
| B2 | release | off | 4 | 0 | 0 | 64.3 % | 0 % |

Corruption thresholds were fixed and committed to git *before* the first server started. Nothing came close: worst repeated block in the whole experiment 15 characters against a threshold of 80, zero U+FFFD, zero non-ASCII characters, 5-gram repetition ratio 0.00. Across 68 SpecDecoding windows acceptance ranged 56.0–94.7 % and **not one window read 0.0 %** — your 39.4 → 0.0 → 64.5 → 20.4 → 0.0 walk has no counterpart here.

I also measured below the visible level, since greedy decoding makes token ids comparable. Arm Bn is arm B repeated on a second fresh server with identical flags: it reproduced only 20 of 38 responses token-for-token, mean absolute chosen-logprob difference 0.0823. That is this build's run-to-run floor. Against it: prefix caching on = 0.1063 (1.29×), patched = 0.1053 (1.28×), concurrency-only-no-cache = 0.0821 (1.00×). So enabling the cache adds a little numerical drift, patching does not change it, and it never changed an answer.

**And prefix caching is worth a great deal** (patched superset image, disjoint documents so the long case is genuinely cold):

| prefix | cold TTFT | warm TTFT | speedup | prompt tokens recomputed when warm | hit rate |
|---|---|---|---|---|---|
| 32,842 | 12.07 s | 1.04 s | **11.6×** | 2,442 of 32,842 | 92.6 % |
| 131,146 | 67.60 s | 2.31 s | **29.3×** | 3,146 of 131,146 | 97.6 % |

The whole 38-request schedule ran in 84.0 s with the cache versus 144.4 s without.

Full receipt — pre-registered thresholds, all 266 per-request rows, acceptance windows, hit rates, both patch digests, and the raw per-arm server logs: <https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/apc-poison-repro.json>

**What this does not prove, plainly.** I did not run LMCache. Your stack has LMCache 0.5.2 in `kv_both` disk mode at chunk 1600 — a second, independent state-restore path layered on top of vLLM's own prefix cache, with its own chunking and its own idea of a reusable prefix. That is the single largest difference between your setup and this measurement, and after your control it is now the leading suspect. I also did not use your custom chat template, ran text only, and ran single-turn rather than multi-turn. And this is one workload on one card: a clean result bounds the blast radius, it does not prove the defect is unreachable. PR #51113 is still genuinely absent from the pinned image (scheduler `1ea341f4…` versus patched `b431c106…`) and upstream's own regression file still fails 14 of 20 against it, so the patch is worth having regardless.

So when you wake up, the cheapest informative order is to add things back **one at a time**, not as a set: first `--enable-prefix-caching --mamba-cache-mode align` alone — 266 requests here say that should hold — then LMCache last, on its own. If it garbles only when LMCache returns, we have the culprit, and I will happily chase it from there. If you can share your chat template I will re-run this probe with it.
