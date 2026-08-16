# What ten million scored positions bought, and what it did not

The published fidelity claim rested on 136 held-out contexts, 278,392 scored positions. The
objection was direct: *"Do I understand this correctly that you evaluated only on 136 traces
and 278k output tokens?"* This document records what happened when the same measurement was
repeated at 37.6x the volume, and it separates the part of the objection that was right from
the part that volume cannot fix.

## The run

| item | value |
|---|---|
| suite | [`receipts/kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json), schema `qwen38-distribution-fidelity/6`; `corpus_note` field amended by [`receipts/kld5-suite-manifest-corpus-note-amendment.json`](../receipts/kld5-suite-manifest-corpus-note-amendment.json) |
| contexts | 5,120 of 2,048 tokens, 2,047 scored positions each |
| scored positions | **10,480,640** |
| source clusters | **842** (one document family per cluster) |
| corpus | 941 documents, 70,348,971 bytes, [`receipts/kld5-corpus-fetch-log.json`](../receipts/kld5-corpus-fetch-log.json) |
| contamination | whole-document pre-exclusion on any exact normalized all-position 12-token overlap with exllamav3 calibration data: 44 of 941 documents excluded, 897 eligible, hits 0 by construction |
| prior-suite disjointness | 0 of the v4 suite's 160 context token hashes reachable |
| windows | exact-advance, non-overlapping; independently checked at 5,120/5,120 unique token hashes and 0 overlapping character spans |
| pipeline | [`tools/kld_ladder.sh`](../tools/kld_ladder.sh) in ten 512-context shards, [`tools/kld_aggregate.py`](../tools/kld_aggregate.py) for cumulative receipts |

Sharding is a disk constraint, not a statistical one: one 512-context capture is 10.7 GB per
model, six models are ~64 GB, and the scratch volume holds ~135 GB. Each shard captures six
models, replays five candidates against the BF16 reference, verifies every report, then
deletes the hidden states.

## Result

Body-only, both operands through one shared BF16 head, cumulative over all ten shards:

| candidate | mean KLD | bootstrap 95 % CI | top-1 | exact worst position |
|---|---:|---:|---:|---:|
| hydrated | **0.002760** | [0.002540, 0.003020] | 97.70 % | 8.258 |
| online K5/K6 | 0.003210 | [0.002982, 0.003480] | 97.52 % | 22.241 |
| context edition | 0.003509 | [0.003220, 0.003852] | 97.44 % | 5.557 |
| official FP8 | 0.005294 | [0.004927, 0.005728] | 96.79 % | 10.714 |
| K4 | 0.010604 | [0.009640, 0.011746] | 95.76 % | 14.283 |

**How closely these absolute numbers may be read.** Each mean is a body-only replay value, and
the replay path is not the engine's own logit path: replaying the unquantized model against its
own live logits measures 6.54e-04
([`receipts/v3-qualification-bf16.json`](../receipts/v3-qualification-bf16.json)), and BF16→fp32
hidden-state storage moves a candidate's KLD by 5.6 % ([docs/24](24-p0-results.md)). Absolute
values are within-suite numbers — a ~6e-4 implementation offset plus a ~5 % storage systematic,
so absolute differences below ~1e-3 are not resolvable. Both offsets are common-mode across
candidates, so paired differences are the resolvable quantity: the hydrated − online row below,
−0.000450, is smaller than the replay floor and resolved *because* the floor cancels in the
pairing. The floor was measured on six v3 contexts; its v5 re-derivation is an open measurement.
Method of record: [docs/42](42-kld-method.md).

Paired per context, 10,000 cluster resamples
([`receipts/kld5-10M-paired.json`](../receipts/kld5-10M-paired.json)):

| comparison | difference | 95 % CI | contexts won |
|---|---:|---:|---:|
| hydrated − FP8 | −0.002534 | [−0.002708, −0.002383] | 5,118 / 5,120 |
| online K5/K6 − FP8 | −0.002084 | [−0.002249, −0.001942] | 5,105 / 5,120 |
| context − FP8 | −0.001785 | [−0.001884, −0.001697] | 5,109 / 5,120 |
| K4 − FP8 | +0.005310 | [+0.004710, +0.006019] | 7 / 5,120 |
| hydrated − online K5/K6 | −0.000450 | [−0.000469, −0.000433] | 4,922 / 5,120 |

## The finding worth keeping

Volume did two useful things and one thing it is often assumed to do, which it did not.

**It confirmed the point estimate.** Across a tenfold increase in scored positions every
candidate mean moved by under 2.3 % of its own value:

| candidate | 1,048,064 | 2,096,128 | 5,240,320 | 10,480,640 |
|---|---:|---:|---:|---:|
| hydrated | 0.002700 | 0.002759 | 0.002699 | 0.002760 |
| online K5/K6 | 0.003141 | 0.003200 | 0.003144 | 0.003210 |
| context | 0.003409 | 0.003502 | 0.003423 | 0.003509 |
| official FP8 | 0.005197 | 0.005296 | 0.005186 | 0.005294 |
| K4 | 0.010345 | 0.010573 | 0.010334 | 0.010604 |

**It settled a marginal ordering.** Offline-serialized attention versus the online overlay was
124/136 contexts on the old suite — suggestive. It is 4,922/5,120 contexts and
−0.000450 [−0.000469, −0.000433] here.

**It did not narrow the interval.** Relative 95 % interval width by checkpoint
([`receipts/kld5-ladder-convergence.json`](../receipts/kld5-ladder-convergence.json)):

| candidate | 1M | 2M | 5M | 10M |
|---|---:|---:|---:|---:|
| hydrated | 14.6 % | 17.8 % | 15.8 % | 17.4 % |
| online K5/K6 | 13.4 % | 15.7 % | 14.0 % | 15.5 % |
| context | 15.1 % | 18.0 % | 16.4 % | 18.0 % |
| official FP8 | 13.0 % | 15.2 % | 13.5 % | 15.1 % |
| K4 | 16.3 % | 19.7 % | 17.8 % | 19.9 % |

The estimator resamples **source clusters**, so its variance is dominated by heterogeneity
between documents, not by how many positions are drawn inside each document. Every shard adds
positions *and* documents, and the newly admitted documents widen the population spread about
as fast as the extra positions tighten the estimate.

The lever that does work is document count, and it is visible across suites rather than
across checkpoints: the v3 suite bootstrapped 39 clusters and published relative widths near
60 % (K4 0.030736, interval [0.02238, 0.04073]); v5 bootstraps 842 clusters at 15-20 %.

So the honest reading of the original objection: 278,392 positions were **enough for the
mean** and not enough to make the result feel accountable. Ten million positions make it
accountable and confirm the mean. Anyone who wants a tighter interval should ask for more
independent documents, not more tokens.

## The tail

Mean and top-1 hide the positions that actually break a generation, which is the second
criticism the release thread raised. The replay harness now accumulates every scored
position into a 560-bin log-spaced histogram
(`KLD_HIST_LOG10_LOW=-12.0`, `KLD_HIST_LOG10_HIGH=2.0`, `KLD_HIST_BINS_PER_DECADE=40` in
[`tools/fidelity.py`](../tools/fidelity.py)), and the aggregator turns summed bins into
bin-bounded cumulative quantiles with exact maxima and exact exceedance counts. Measured on a
rerun of shard 0 — 512 contexts, 1,048,064 positions, the same contexts for every candidate
([`receipts/kld5-1M-tail-*.json`](../receipts)):

| candidate | p50 | p95 | p99 | p99.9 | p99.99 | exact max | above 0.1 | above 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hydrated | 0.00109 | 0.0082 | 0.0276 | **0.1319** | 0.463 | 3.735 | 0.1534 % | 0.00219 % |
| online K5/K6 | 0.00128 | 0.0099 | 0.0321 | 0.1446 | 0.498 | 5.507 | 0.1820 % | 0.00200 % |
| context | 0.00135 | 0.0107 | 0.0357 | 0.1642 | 0.587 | 3.749 | 0.2287 % | 0.00305 % |
| official FP8 | 0.00202 | 0.0167 | 0.0531 | **0.2438** | 0.812 | 5.296 | 0.3912 % | 0.00592 % |
| K4 | 0.00320 | 0.0332 | 0.1194 | 0.5555 | 1.870 | 7.565 | 1.2604 % | 0.03807 % |

The quantile ordering equals the mean ordering, so for these candidates the mean is not
concealing a heavier tail: every K5/K6-class build is lighter than official FP8 at every
measured quantile, and K4 is heavier than FP8 at every one. Quantiles are bin-bounded to about
5.6 % of their value; maxima and exceedance counts are exact.

## What this run still does not give

- **Absolute KLD is suite-specific.** v5 numbers are not comparable to v3 numbers: the same
  K4 checkpoint reads 0.029679 on the corrected v3 subset and 0.010604 here. Only ordering and
  paired differences transfer between suites.
- **No published captures.** The ten shards' hidden states were deleted as each shard verified,
  because keeping them all would need ~640 GB. Reproduction runs through the pinned corpus
  fetch log and suite manifest, not through downloadable captures as with the v3 dataset.
- **The tail is one shard of ten.** The ten-shard run predates the histogram; re-running the
  other nine with the newer harness is about six hours of GPU time.
- **Every window is 2,048 tokens.** Long-context fidelity at 32k/64k/128k is a separate suite
  and remains unmeasured.
- **Comparator breadth.** FP8 is a throughput-oriented format; GGUF `Q5_K_XL`/`Q6_K`/`Q8_0`
  and stock uniform-bitrate EXL3 controls are the comparisons that would settle where these
  builds really sit. See [29](29-plan-and-loose-ends.md) F2.
