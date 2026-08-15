# The frozen qualification: the ranking survives on sources nobody tuned against

Two independent reviews said the same thing: every number this project published came from the
v3 analysis partition, which guided recipe selection, and v3's own "qualification" partition
shared **all 27** of its source clusters with analysis — so it could never have been a
post-selection test. This is the test that was missing.

## The suite

`kld4`: **160 contexts, 2,048 tokens each, 327,520 scored positions, 100 source clusters**,
built from 100 documents that do not appear in v3 in any form. Three intersections with v3,
each measured rather than asserted: context token hashes **0/160**, document names **0/100**,
document content sha256 **0/100**. Partitions are assigned by whole cluster and
`cluster_partition.overlap` is empty: analysis 118 contexts / 72 clusters, qualification 42 /
28, sentinels 16. Contamination scan against exllamav3's calibration corpus: 0 hits.
`suite_token_sha256` `71f4fab2c951841c5c504b3ab723fd8d…`.

Coverage the v3 suite claimed and did not have: the multilingual stratum is genuinely six
languages (fr, es, it, ja, pt, zh), 32 contexts per stratum exactly, with the builder now
refusing an under-filled stratum instead of shrinking silently.

## The result

Run once. No recipe changed because of it. Future development uses the v4 **analysis**
partition only, so the qualification partition stays clean.

**Qualification partition, 42 contexts, 28 clusters:**

| candidate | resident | mean KLD | 95 % CI | top-1 | paired vs FP8 | contexts won |
|---|---:|---:|---|---:|---:|---:|
| hydrated (attention K6 offline) | 20.31 GiB | **0.003029** | [0.002572, 0.003536] | 97.68 % | −0.002691 | **42/42** |
| K5/K6 (attention online K6) | 20.32 GiB | 0.003395 | [0.002925, 0.003910] | 97.58 % | −0.002325 | **42/42** |
| context edition (attention K5 offline) | 19.56 GiB | 0.003900 | [0.003275, 0.004588] | 97.43 % | −0.001820 | **42/42** |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.005720 | [0.004849, 0.006680] | 96.82 % | — | — |

**Full v4 suite, 160 contexts** (also entirely unseen): hydrated 0.003863, K5/K6 0.004324,
context edition 0.004944, official FP8 0.007064.

## What it does and does not settle

**Settled:** the ordering is not an artefact of tuning against the measurement. Every build
beats official FP8 on every one of the 42 held-out contexts, and the *relative* advantage is
larger on the unseen suite than on the development one — 47 %, 41 % and 32 % below FP8 here
against 44 %, 38 % and 26 % on v3-analysis. Selection bias predicts the opposite.

**Not settled, and worth saying plainly:** absolute KLD is about 2.4x lower on v4 than on v3
for *every* candidate, official FP8 included (0.005720 against 0.013126). That is a property of
the corpus, not of the models — different documents, different tail behaviour. **KLD magnitudes
are only comparable within one suite.** Anyone quoting a number from this project must quote
the suite with it.

Also unchanged by this run: no downstream task retention, no multimodal quality beyond the
synthetic set, no long-context accuracy beyond needle retrieval, and the KLD protocol still
measures distribution divergence on a shared BF16 head rather than end-to-end served behaviour.
