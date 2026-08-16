# The KLD method of record (v5)

This is the single document that describes how every KL-divergence number published for
the Qwen3.8-27B artifacts is produced, which floors bound it, and what it does and does
not support. It exists because the method was previously spread across four files and one
README: [`05-kld-protocol.md`](05-kld-protocol.md) describes the **superseded** iteration-1
logits protocol, [`14-fidelity-protocol-v2.md`](14-fidelity-protocol-v2.md) describes the
hidden-state replay design but on the 74-context **v2** suite,
[`33-evidence-volume-and-intervals.md`](33-evidence-volume-and-intervals.md) reports the v5
run, and [`35-external-protocol-comparability.md`](35-external-protocol-comparability.md)
states the protocol as one column of a comparison table. Where those disagree with this
file about the *current* protocol, this file is correct; where they add results or
comparisons, they stand.

Scope note that travels with everything below: this protocol measures **teacher-forced,
text-only, body-only distribution fidelity on a 2,048-token window**. It is not a quality
benchmark, not a generation benchmark, and not a long-context benchmark.

## 1. The metric

`KLD = KL(BF16 reference ‖ candidate)`, in nats, per scored position, over **all 248,320
vocabulary entries** — no top-k, no truncation, no probability-mass gating.

The comparator is exact and two-pass ([`tools/fidelity.py`](../tools/fidelity.py),
`normalizers_and_top1` and `context_metrics`): pass 1 accumulates `logsumexp` normalizers
and the argmax for both operands; pass 2 accumulates the KL term and Jensen-Shannon
divergence (reported in bits). Vocabulary is walked in chunks of 24,832 entries in
float32; accumulation across chunks is float64. Every report records that configuration in
its `comparator` block (`vocab_chunk: 24832`, `accumulation: "float64"`, `two_pass: true`).

Reported per report: `token_mean_kld` (mean over all scored positions),
`context_macro_mean_kld` (mean over contexts of each context's mean — identical to the
token mean here because every context contributes exactly 2,047 positions),
`top1_agreement`, `mean_jsd_bits`, exact `max_kld`, exact p50/p95/p99/p999, and the tail
histogram of §7.

## 2. The reference, and one shared BF16 head

The reference is unquantized BF16 `Qwen/Qwen3.8-27B`, served by vLLM. Its 18 weight shards,
`config.json` and `model.safetensors.index.json` digests are recorded in the
`reference_identity` block of every cumulative receipt (index
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`, config
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`).

Both operands are projected to logits by **one shared BF16 `lm_head`**:

| | |
|---|---|
| tensor | `[248320, 5120]` BF16, extracted from `Qwen/Qwen3.8-27B` |
| sha256 | `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff` |
| size | 2,542,796,896 B |
| published at | `malaiwah/qwen38-27b-fidelity-suite-v3`, path `lm-head/weight.safetensors` |

The digest above is the `head_sha256` of every published v5 report and of the ladder pin,
and it is the LFS object digest the Hub serves for that path — so the head every published
number used is downloadable and checkable.

**Why a shared head.** Head error and body error are otherwise entangled: each quant recipe
treats `lm_head` differently, so a head-inclusive number mixes "how good is the body" with
"which head did this packager keep in BF16". Replaying both operands through one BF16 head
makes every published number **body-only by construction**, and makes head cost a separate,
attributable measurement (§11, and [`16-head-attribution.md`](16-head-attribution.md)).
The cost of that choice is stated plainly: **these numbers do not measure the head a
downloader actually runs.**

## 3. The capture point

`fidelity.py capture` runs in-process vLLM and installs a forward hook, via
`collective_rpc` inside the worker, on the model's final norm module — matched as
`*language_model.norm`, `model.norm` or `*.model.norm` — i.e. the **post-final-norm**
hidden state, storing rows `0..2046` of each 2,048-token context as `[2047, 5120]` BF16 in
`hidden_NNNN.safetensors`, plus a `capture-manifest.json` binding the capture to the
suite's token digest. Row *r* is the state that predicts token *r+1*, so the scored
positions are exactly positions 0..2046 of the window — **including the low-context, high
entropy positions at the start.**

Storing hidden states rather than logits is a 96x storage reduction (21 MB versus 2.03 GB
per context) and is what makes offline replay of a new candidate possible. It also assumes
what follows the capture point is exactly one linear map, which for this architecture it
is (`norm` → `lm_head`); anything a serving stack does after the logits — sampling,
penalties, speculative acceptance — is outside the measurement.

## 4. The suite

Built by [`tools/suite3.py`](../tools/suite3.py) — schema `qwen38-distribution-fidelity/6`,
recorded in [`receipts/kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json).
The name is historical: `suite3.py` builds the v5 suite. `fidelity.py suite` emits schema
`/1` and did **not** build it.

| | |
|---|---|
| contexts x window | 5,120 x 2,048 tokens, exact-advance, non-overlapping |
| scored positions | 2,047 per context, **10,480,640** total |
| `suite_token_sha256` | `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88` |
| `manifest_sha256` | `c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019` |
| strata | code / encyclopedic / literary / multilingual / scientific, 1,024 contexts each |
| source clusters | 842, one document family per cluster |
| partitions | analysis 4,064, qualification 1,056, sentinels 32 |
| corpus | 941 documents, 70,348,971 B (Gutenberg, arXiv, Wikipedia en+de/fr/es/ja/zh/ru, CPython `Lib/`) |
| corpus provenance | [`receipts/kld5-corpus-fetch-log.json`](../receipts/kld5-corpus-fetch-log.json) — per-document URL, bytes, sha256, every skip and failure |

The **token id files are the authoritative evaluation input.** Retokenizing the source text
does not reproduce them; each context's `token_sha256` is the sha256 of `json.dumps(ids)`,
and the suite digest is derived from the ordered per-context digests. Every `capture`,
`replay` and `paired` command re-derives it and aborts on drift.

### Contamination exclusion

Whole-document pre-exclusion, **before** window selection, on any exact match of a 12-word
shingle against the exllamav3 calibration corpora:

* normalization Unicode NFKC + casefold, Unicode word tokens, one token per Han/Kana
  character; digest blake2b-128; shingle 12 words, stride 1
* 6 calibration files, 859,426 distinct calibration shingles; 10,772,868 corpus shingles
  scanned
* **44 of 941 documents excluded (43 code, 1 encyclopedic), 897 eligible**
* emitted contexts additionally rejected on any decoded-token match: 0 rejected, 0 hits

Contamination on this suite is therefore zero *by construction*, and there is no
"overlap-corrected subset" as there was for v3. The suite is also token-disjoint from the
v4 qualification suite (0 of its 160 context token hashes reachable).

The limitation is inherent to the method and is measured in §11: an exact 12-word shingle
catches verbatim reuse, not paraphrase.

## 5. The ladder

[`tools/kld_ladder.sh`](../tools/kld_ladder.sh) walks the parent suite in **ten 512-context
shard views**, and for each shard: captures BF16, captures each candidate, replays each
candidate against the BF16 reference through the shared head, verifies every report against
the shard view, then releases the hidden states before starting the next shard. Sharding is
a disk constraint, not a statistical one — one 512-context capture is 10.7 GB per model.

The whole run is pinned by `suite/ladder-pin.json` (published in the v5 dataset): harness
digest, shared-head digest, quantization config and its digest, shard size, parent suite
identity, and the runtime image digest.

Per-shard reports are welded into cumulative receipts by
[`tools/kld_aggregate.py`](../tools/kld_aggregate.py) `aggregate`, which recomputes every
headline from the per-context rows and cross-checks it against each shard report's own
summary. `paired` then compares two cumulative receipts per context.

`aggregate` fails closed: it refuses a shard set whose reports disagree on the shared head,
the reference identity, the scored-position window or the suite lineage, and it refuses to
mix a windowed report (schema `qwen38-fidelity-report/3`) with full-context reports (`/2`).

## 6. Determinism controls

`enforce_eager=True`, `enable_prefix_caching=False`, `max_num_seqs=1`, one context per
forward with `max_num_batched_tokens = context_length` (one prefill chunk per context),
`kv_cache_memory_bytes = 512 MiB` bfloat16, `gpu_memory_utilization = 0.85`, runtime image
`voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`.
Nothing is sampled, so there is no RNG seed in the measurement path; the only seed is the
bootstrap's.

Under exactly that configuration,
[`receipts/capture-determinism.json`](../receipts/capture-determinism.json) records
**5,240,320 scored positions** (5 candidates x 512 contexts x 2,047) reproducing
bit-for-bit across two independent runs with separate model loads and two different harness
files: every measured field identical including the whole per-context arrays and the
bootstrap block, with only capture directory paths and the additively-added tail histogram
differing.

That receipt states its own scope, and this document repeats it: it is **not** a claim that
vLLM is bitwise deterministic in general. It covers one GPU, one driver, one pinned image,
eager execution, one sequence, one chunk per context. It does not cover CUDA graphs,
`max_num_seqs > 1`, multi-chunk prefill, other GPUs or drivers, or anything downstream of
the logits.

## 7. Statistics

**Point estimate.** `token_mean_kld` over all scored positions.

**Interval.** Nonparametric percentile bootstrap over **source clusters**, not contexts and
not positions: the 842 clusters are resampled with replacement 842 at a time, 10,000
resamples, **seed 1**, and the interval is the 2.5th/97.5th percentile of the resampled
pooled means (`bootstrap()` in `fidelity.py`, reused by `kld_aggregate.py`). The cluster is
the independence unit because contexts drawn from one document family are not independent.

**Paired comparison.** The per-context difference `A - B` on the identical context set,
with the same cluster bootstrap and a win count. `a_wins` counts strict improvements and
`b_wins` is the remainder, so an exact tie is counted as a b-win — a convention that has
misled a reader before, kept unchanged so the published receipts stay reproducible, and
stated here rather than left in a receipt footnote.

**Tail.** Per-shard percentiles cannot be recombined, so `replay` also counts every scored
position into a **fixed 560-bin log-spaced histogram** — `log10` from -12.0 to 2.0, 40 bins
per decade, `edges[i] = 10 ** (log10_low + i / bins_per_decade)`, bucket
`bisect_right(edges, v)`, with explicit underflow bucket 0 (every exact zero) and overflow
bucket 561 — alongside its own exact max and exact quantiles. `kld_aggregate.py` sums those
counts across shards and derives **bin-bounded** cumulative quantiles, reported as a
`[lower, upper]` bracket with the bin's relative width, never as a false point value. The
three histogram constants are part of the report format: changing any of them means a new
schema, because two shards' counts may only be summed when their edges are identical.

## 8. Published artifacts, and which digest pins what

| artifact | where |
|---|---|
| suite manifest, shard views, ladder pin, 5,120 token id files | dataset `malaiwah/qwen38-27b-fidelity-suite-v5`, `suite/` |
| corpus fetch log | same, `corpus/corpus_fetch_log.json` |
| shard-0 BF16 hidden-state reference (512 captures + manifest) | same, `reference/hidden-bf16/` |
| 50 ladder + 10 tail + 15 scored-window + 4 cross-engine per-shard reports | same, `reports/` |
| four shard-0 candidate capture trees (nvfp4, eda, hyd-rematch, hyd-sibling) | same, `captures/shard-0000/` |
| the shared BF16 head | dataset `malaiwah/qwen38-27b-fidelity-suite-v3`, `lm-head/weight.safetensors` |
| cumulative and paired receipts, all analysis receipts | this repo, `receipts/` |
| harness, suite builder, ladder driver, aggregator, corpus fetcher | this repo, `tools/` |

The 15 scored-window reports and the 4 cross-engine reports in the dataset are
byte-identical to `receipts/kld5-window-*.json` and `receipts/gguf-report-*.json`; the 50
ladder reports match the `inputs` digests recorded in `receipts/kld5-10M-*.json`.

### Harness pin versus harness head — read this before comparing digests

`suite/ladder-pin.json` and `receipts/capture-determinism.json` pin the harness by content
digest, and **those digests are historical revisions of `tools/fidelity.py`, not the file
at the head of this repository**:

| digest | what |
|---|---|
| `804c20a6479df067b1dde0e34c3664f0a7d12d692d59cd5d99ed0e319257a7f6` | the harness that produced the ten-shard ladder (`ladder-pin.json`, `capture-determinism.json` `runs.published_ladder`) |
| `f8b1316f411a8ac02ff88144a6fedfeee56f0cc23c1b6887aa09ef9c7f842996` | the harness that produced the tail re-run (`capture-determinism.json` `runs.tail_rerun`) |
| `8ce8fe861c977c856f3263f509878e588caa2eaab29fe310f8c40223941e1294` | the file at repository head, and the harness of `receipts/nvfp4-v5-measurement.json` |

The three differ **only additively**, verified function by function: lazy `import torch` so
`--help` and `paired` work without torch installed; the tail histogram; and `--score-from`,
which defaults to 0. `normalizers_and_top1`, `context_metrics`, `bootstrap`,
`validate_capture_files` and `capture_identity` — everything in the measurement path — are
byte-identical across all three apart from that import line. So re-running `replay` with
the head of this repository computes the same statistic, but emits schema
`qwen38-fidelity-report/2` with a `kld_tail` block, which is a different file from a
published `/1` report and will not match its digest. Recover a pinned revision with
`git log --all -- tools/fidelity.py` and check the blob digest.

## 9. Reproduce a published number with no GPU

This is the shortest path from nothing to a published headline, and it is the one that was
run to produce this document's evidence. It needs 61 files and about 24 MB of downloads,
CPU only, no torch, no model weights, and it runs in seconds.

```bash
git clone https://github.com/malaiwah/qwen38-27b-exl3
D=https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5/resolve/main

# the 50 per-shard ladder reports, and the suite lineage the aggregator verifies against
for s in 0000 0001 0002 0003 0004 0005 0006 0007 0008 0009; do
  mkdir -p reports/kld5/shard-$s suite/shard-$s
  for c in hyd k5k6 ctx fp8 k4; do
    curl -sSL -o reports/kld5/shard-$s/report-$c.json "$D/reports/kld5/shard-$s/report-$c.json"
  done
  curl -sSL -o suite/shard-$s/suite-manifest.json "$D/suite/shard-$s/suite-manifest.json"
done
curl -sSL -o suite/suite-manifest.json "$D/suite/suite-manifest.json"
curl -sSL -o suite/ladder-pin.json    "$D/suite/ladder-pin.json"

# rebuild one cumulative receipt.  --allow-legacy-no-tail is required because the published
# ladder reports are schema /1: the ten-shard run predates the tail histogram, which is why
# receipts/kld5-10M-*.json carry "tail": null.
python qwen38-27b-exl3/tools/kld_aggregate.py aggregate \
  reports/kld5/shard-*/report-hyd.json \
  --suite suite --allow-legacy-no-tail \
  --candidate hyd --label 10M --out out-hyd.json
```

That prints, and this repository's
[`receipts/kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json) contains, the same
`token_mean_kld` `0.0027596250151498245`, the same interval
`[0.0025402354561764456, 0.003020322111388249]`, the same `top1_agreement`
`0.9770086559599414`, the same `max_kld` `8.257617292998475`, 5,120 contexts, 10,480,640
positions and 842 clusters. Repeat for `k5k6`, `ctx`, `fp8`, `k4` and then run `paired` over
the five outputs to re-derive `receipts/kld5-10M-paired.json`'s five differences, intervals
and win counts. The two tail shards reproduce the same way from
`reports/kld5-tail/shard-000{0,1}/`, including all 562 histogram counts and every
bin-bounded quantile, with no `--allow-legacy-no-tail`.

**What will not match is `content_sha256`.** That digest covers the whole receipt payload
except `generated_unix`, and the payload includes the `inputs` map keyed by absolute report
paths, the `shards[].report` paths and the aggregator's own schema string. A third party's
paths differ, so their digest differs even when every measured field is identical. Compare
the measured fields, not the receipt digest.

## 10. Reproduce with a GPU

Score a **new** candidate against the published shard-0 BF16 reference — one candidate
capture plus one replay, no BF16 recapture:

```bash
hf download malaiwah/qwen38-27b-fidelity-suite-v5 --repo-type dataset --local-dir v5
hf download malaiwah/qwen38-27b-fidelity-suite-v3 --repo-type dataset \
  --include 'lm-head/*' --local-dir v3
ln -s ../tokens v5/suite/shard-0000/tokens   # the shard view names its token files relatively

python tools/fidelity.py capture --model /path/to/candidate \
  --suite v5/suite/shard-0000 --out hidden-mine
python tools/fidelity.py replay --reference v5/reference/hidden-bf16 \
  --candidate hidden-mine --head v3/lm-head/weight.safetensors \
  --suite v5/suite/shard-0000 --out report-mine.json
python tools/fidelity.py paired --a report-mine.json \
  --b v5/reports/kld5/shard-0000/report-hyd.json --out paired-mine.json
```

`replay` is fail-closed on suite digest, context set, tensor shape, head digest and scored
window, so a report it accepts is comparable with the published ones by construction. The
full ten-shard ladder additionally needs the BF16 reference captured for shards 1-9, which
were deleted after their reports verified; the runtime image is public at the digest in §6.

## 11. Floors and systematics that bound every number

Nothing below is hypothetical; each is a measurement in this repository.

| term | size | what it bounds | receipt |
|---|---|---|---|
| runtime-repeat / capture-replay determinism | exactly 0 | re-running the same capture+replay changes nothing | `capture-determinism.json` |
| replay versus the engine's own logits | `KL(live ‖ replayed)` = 0.000654177503134738, top-1 0.9899853444064485, over **6 contexts** of the v3 suite | the **absolute** level of every number: our replay path is not vLLM's logit path | `v3-qualification-bf16.json` |
| hidden-state storage precision | fp32 storage moved that floor to 0.0006249375925433911 (-4.5 %) and a candidate's KLD by -5.6 % on 32 sentinels | ~5 % of the absolute level is BF16 storage; the rest is the two implementations' matmul ordering | `v3-qualification-fp32.json`, [`24-p0-results.md`](24-p0-results.md) |
| head exclusion | body-only versus body+K6-head, paired: -6.778619846896359e-05, CI [-9.014223019177851e-05, -4.632939580981435e-05], 74 contexts / 23 clusters, v2 suite | what the shared head leaves out, for one head recipe on a superseded suite | `paired-head-e2e.json`, [`16-head-attribution.md`](16-head-attribution.md) |
| cross-engine (llama.cpp vs vLLM) | 0.0005073550588083691 mean, top-1 0.9906713712139716, on identical unquantized BF16 weights | every GGUF row and no vLLM row | `gguf-report-engine-floor.json`, `cross-engine-comparator.json` |
| scored-position window | requiring 256 tokens of left context lowers each mean by 1.306-2.059 %; second-half-only by 3.876-4.874 % | how much of a gap to an external protocol is position selection | `scored-window-offset.json` |
| suite hardness | the same six checkpoints read 2.4624986758761565x to 3.0790651709420223x higher on v5 than on corrected v3, a 1.2503824676561468x band, ordering identical | absolute values do not travel between suites; ordering does | `nvfp4-v5-measurement.json` |

Consequences, stated once:

* **Differences below roughly 1e-3 in absolute terms are not resolvable; the same
  differences measured pairwise are**, because both arms traverse the identical capture and
  replay path and the implementation offset is common-mode. Every headline comparison in
  this project is a paired one for that reason.
* A GGUF-versus-EXL3 comparison carries the 0.000507 cross-engine term on the GGUF side
  only. Net-of-floor figures are estimates, not identities — KL is not additive — and no
  ordering closer than a factor of two should be pressed against `Q8_0`, whose measured
  0.001087 is about twice the floor.
* Volume does not buy precision here. Across a tenfold increase in positions every mean
  moved by less than 2.3 % of itself while relative interval width did not fall (hydrated
  14.6 % at 1,048,064 positions, 17.4 % at 10,480,640), because the estimator resamples
  documents. **More independent documents narrow the interval; more tokens do not**
  ([`33-evidence-volume-and-intervals.md`](33-evidence-volume-and-intervals.md)).

## 12. What these numbers support, and what they do not

They support: the **relative ordering** and **paired differences** of these specific
artifacts, on this corpus, at this window length, through this one BF16 head, under
teacher forcing, in this engine. That ordering is robust in every way it has been tested —
identical in all five strata, identical in all ten independent 512-context shards, and
unchanged by moving the scored window to 256 or 1,024 tokens of left context.

They do not support:

* **any absolute magnitude to better than ~1e-3**, per §11;
* **any cross-suite absolute comparison.** A v3 and a v5 number must never appear in the
  same sentence; the same checkpoint reads 2.46-3.08x apart between them;
* **any cross-protocol ratio.** Different corpus, window, reference numerics, head
  treatment and scoring floor — see
  [`35-external-protocol-comparability.md`](35-external-protocol-comparability.md) for the
  twelve enumerated deltas, of which two are measured;
* **generation quality.** Teacher forcing scores one fixed token sequence; it says nothing
  about what the model emits when it drives its own context. The task-level evidence for
  that is a separate, much smaller measurement (`public-capability-*.json`, 70 MMLU-Pro
  items) and is not interchangeable with KLD;
* **the head a downloader runs.** Body-only by construction;
* **long context.** Every window is 2,048 tokens. Fidelity at 32k/64k/128k/262k is
  unmeasured by this protocol;
* **anything downstream of the logits**, or any runtime configuration other than the one in
  §6.

## 13. Known gaps

Enumerated, with priorities, in
[`receipts/kld-method-reproducibility-audit.json`](../receipts/kld-method-reproducibility-audit.json),
the adversarial audit that produced this document.
