# External validation, and the four claims it corrected

> **Current status, 2026-08-15.** Sections 2–4 preserve the iteration-2 review state at the
> time the defects were found; their “being built”, “absent” and “not run” statements are
> historical, not current. Source-disjoint v4 captures and qualification are published, and
> the overlap-corrected subset preserves all **36/36** paired wins. K5/K6 captures and reports
> are published. The generated task smoke passes **40/40** for BF16 and every comparator with
> zero regressions; it is not a public benchmark. The native-context retraction below remains
> correct for online K5. A separate serialized-K5 context build with per-row int8 input
> embeddings reaches native 262,144 with MTP-3 under a 30.24 GiB engine budget when image
> preprocessing is capped; physical RTX 5090 validation remains pending. Current receipts and
> acceptance gates are in [docs/31](31-frozen-qualification.md),
> [docs/32](32-native-context-embedding-overlay.md), and
> [docs/29](29-plan-and-loose-ends.md).

Two independent reviews and one hardware test landed on iteration 2. All three found real
defects. This records what was wrong, what the corrected numbers are, and what changed in the
code so the same class of error cannot recur.

## 1. Native 262,144 context on a 32 GB card: retracted

**Claimed:** attention overlay K5 reaches native 262,144 on a 32 GB card (266,743 KV tokens),
"verified".

**Measured on a real RTX 5090** (31.39 GiB usable, TP1, FP8 E4M3 KV, MTP-3, decode-only CUDA
graphs, vision enabled):

| attention | seqs | util | KV | 262,144 | outcome |
|---|---:|---:|---:|---|---|
| K6 | 8 | 0.95 | 187,050 tok / 6.71 GiB | not attempted | 185,600 run, multimodal-safe |
| K6 | 8 | 0.98 | 202,185 tok / 7.55 GiB | not attempted | a 3,264-token image OOMed, 33 MiB free |
| K5 | 8 | 0.95 | 7.33 GiB avail vs 9.13 needed | **refused** | 206,400 run, exact retrieval passed |
| K5 | 4 | 0.96 | 7.99 vs 9.13 | **refused** | 1.14 GiB short |
| K5 | 4 | 0.97 | 8.30 vs 9.13 | **refused** | 0.83 GiB short |
| K4 build, K6 | 8 | 0.98 | 289,577 tok capacity | **fits** | native context, production-qualified |

**Two errors, both mine:**

1. **MTP's KV was not counted.** With `num_speculative_tokens: 3` the engine needs
   **9.13 GiB** for 262,144 tokens, not 8.18 GiB — 37,396 B/token instead of 33,505, a 11.6 %
   difference. My capacity runs had no speculative config while the recommended recipe does.
2. **Usable VRAM was overstated.** I modelled a 32 GB card as 31.84 GiB usable; a 5090 reports
   **31.39**. At utilisation 0.98 that is 0.44 GiB of budget that does not exist.

Recomputed with the hardware's own coefficients (KV 37,396 B/token, fixed overhead 2.75 GiB at
`--max-num-seqs 4`), the model reproduces the tester's table: K5 at 0.97/seqs 4 gives 238,313
against their 236,800 estimate, and only the K4 build (17.89 GiB resident) crosses native
context. The chart generator now carries those coefficients and per-card *usable* VRAM.

**Also learned:** utilisation 0.98 is not safe for a multimodal server — text works and then a
3,264-vision-token image OOMs. 0.95 is the vision-safe ceiling. And 4-bit KV is not an escape:
generic NVFP4 KV requires SM100 trtllm-gen and is rejected on SM120, while GLM-5.2's
`nvfp4_ds_mla` is MLA-specific and Qwen3.8 is not MLA.

> **Superseded, 2026-08-16, by measurement.** The 4-bit-KV sentence above was written from the
> flag surface before any 4-bit arm had run; the KV-dtype sweep
> ([docs/38](38-kv-dtype-sweep.md), `receipts/kv-dtype-sweep-5090.json`) has now run them on the
> physical 5090. **4-bit KV exists on this fork** — `--kv-cache-dtype int4_per_token_head` starts,
> serves the native window, and moves the class windows substantially: 502,667 KV tokens against
> fp8's 265,122 on the same 32 GB board, 1.92× concurrency, with the predicted (not started)
> 24 GB MTP-3 window going 24,576 → 53,248. But it is an escape with two named prices, not a free
> one: **3.6× fp8's distributional error** against a bfloat16-KV reference (0.005948 vs
> 0.001655 nats mean truncated top-20 KL at a 98,304-token context — a KV-probe number, not a v5
> KLD, never to be differenced against one) and **2.78× fp8's prefill** (501.0 s against 180.4 s
> on the same 261,795-token prompt). The nvfp4 half of the sentence stands in its conclusion,
> though the measured mechanism is simpler than the reasons named: both nvfp4 forms still refuse
> at startup because no attention backend on this fork advertises nvfp4 for a non-MLA decoder —
> all five candidates answer `kv_cache_dtype not supported`.

**Word discipline adopted:** "starts" or "allocated" for an engine that reserved a cache and
completed startup; "run" only for a context actually generated; "retrieval passed" only where
an exactness check ran. The earlier card said "verified" for what was an allocation test.

## 2. Head attribution: measured wrong twice, then correctly

The reviewers asked whether the K6 `lm_head` costs anything on *this* checkpoint. First
attempt replayed both operands through the same head, so head error cancelled by construction
and returned `-0.000000` — the identical trap that produced a null result in iteration 1.

With asymmetric heads (reference through the true BF16 head, candidate through the
dequantized K6 head): the K6 head costs **+0.000127**, 95 % CI [+0.000105, +0.000148],
10/136 contexts favour it ([`receipts/v3-paired-head-asym-v2.json`](../receipts/v3-paired-head-asym-v2.json)).
That is 1.5 % of total divergence, and promoting the head to BF16
would spend **1.589 GB** to recover it. Rejected on that basis.

Consequence for the headline: 0.008157 is **body-only** (shared BF16 comparator head, the same
treatment every comparator gets, since official FP8 serves a BF16 head). As served with the K6
head the figure is **0.008284**
([`receipts/v3-report-v2-asym-k6head.json`](../receipts/v3-report-v2-asym-k6head.json)).
Both beat FP8's body-only 0.013126.

## 3. The "held-out" suite was not post-selection

The reviewer's measurement, reproduced here exactly: the v3 suite has 41 source clusters in
the analysis partition and 27 in qualification, and **all 27 also appear in analysis** — zero
qualification-only clusters. `suite3.py` shuffled *contexts* and sliced, so every document
family landed on both sides. No re-split of those 181 contexts can produce a post-selection
set, because the recipe was chosen with the analysis partition visible and every cluster is
represented there.

Fixed in the builder:

- partitions are assigned by **whole source cluster**, stratified by stratum, and the build
  aborts if any cluster appears on both sides;
- the manifest records `cluster_partition` (both sides plus an `overlap` list) and a
  `documents` map with each source's sha256, so group-disjointness is checkable without
  recomputation;
- `--exclude-suite` refuses documents and context hashes belonging to an earlier suite, which
  is how a later suite is made source-disjoint from the one that guided selection;
- an under-filled stratum now **fails** instead of silently shrinking. That tolerance is why
  the v3 suite advertised nine Wikipedia languages and actually held German and Russian only
  (6 German + 1 Russian in the multilingual stratum). The card now says English/German/Russian.

A source-disjoint v4 suite is being built from new documents; the frozen recipe will be
evaluated on it once.

## 4. Evidence chain: receipts split, build path fails closed

- **Release receipt.** `SHA256SUMS` mixed the immutable payload with documentation, carried
  stale hashes for `README.md` and `.gitattributes`, and listed a `crc32.txt` that does not
  exist. Now two scopes: `SHA256SUMS` covers the payload (17 files, 30,597,223,337 bytes,
  independently verified with `sha256sum -c`), `DOCS-SHA256SUMS` covers card files, and the
  file list is built by scanning the directory so a missing file is an error rather than a
  line.
- **Splice and finalize fail closed.** The splice refuses a non-empty output directory without
  `--force`, has no bare `assert` left (`python -O` disables those), enforces exact module
  counts (144 linear-attention + 64 full-attention = 208), BF16 dtype and per-tensor shape
  equality, and emits a `splice-receipt.json`. The finalizer requires `--upstream` and
  compares the reconstructed logical tensor **name set and shapes** against upstream, exiting
  non-zero on any difference; it also records `logical_parameter_count` and
  `physical_tensor_count`, because HF displays the packed storage-element count (15.3 G) and
  readers mistake it for the parameter count (27.78 G).
- **Reports are recomputable.** The replay report now emits `per_context_all` — every scored
  context, not the worst 20 — plus a `candidate_identity` block with the index sha256,
  optional shard hashes, and both the requested and engine-resolved KV dtype, since recording
  `auto` told a reader nothing.

## Still open

1. **K5/K6 captures are not in the published dataset.** Until they are, the headline is not
   independently recomputable. This is the top blocker.
2. **No image digest contains the graph and prefill patches.** The card now ships two recipes:
   what the pinned image runs unmodified (eager, 28.8 tok/s decode, 2.4k prefill) and the
   patched path with the module's sha256.
3. **No downstream task retention, multimodal quality, or long-context accuracy suite** beyond
   the tester's exact-retrieval check at 205,021 tokens.
4. **MTP is benchmarked asymmetrically** — 113.8 tok/s is this checkpoint with MTP-3 against
   comparators without it. The correct accounting, now on the card: 58.2 % of drafted tokens
   accepted, **1.745 accepted draft tokens per step**, **2.745 output tokens per iteration**.
