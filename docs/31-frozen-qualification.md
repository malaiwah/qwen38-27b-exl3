# The frozen qualification: the ranking survives source-disjoint and overlap-corrected tests

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
28, sentinels 16. `suite_token_sha256`
`71f4fab2c951841c5c504b3ab723fd8d…`.

Coverage the v3 suite claimed and did not have: the multilingual stratum is genuinely six
languages (fr, es, it, ja, pt, zh), 32 contexts per stratum exactly, with the builder now
refusing an under-filled stratum instead of shrinking silently.

## The result

Run once. No recipe changed because of it. A later contamination-audit correction applies one
candidate-independent exclusion policy to the frozen outputs; no model was rerun.

The original 42-context result was hydrated 0.003029, K5/K6 0.003395, context edition
0.003900, and official FP8 0.005720, with all three builds winning all 42 paired contexts.
That table used a fixed-stride 160-character scan which reported zero hits but was not
offset-independent.

An all-position scan of every normalized 12-token n-gram (Unicode words plus one token per
Han/Kana character) found exact overlap with the quantizer calibration corpus in 10/100 source
documents, including two runs at least 32 tokens long. Four affected source clusters occurred
in qualification. Conservatively excluding every context from any source document with even
one exact 12-token hit leaves **36 contexts / 24
clusters**:

| candidate | resident | mean KLD | 95 % CI | top-1 | paired vs FP8 | contexts won |
|---|---:|---:|---|---:|---:|---:|
| hydrated (attention K6 offline) | 20.31 GiB | **0.003093** | [0.002577, 0.003684] | 97.63 % | −0.002798 | **36/36** |
| K5/K6 (attention online K6) | 20.32 GiB | 0.003455 | [0.002916, 0.004060] | 97.50 % | −0.002436 | **36/36** |
| context edition (attention K5 offline) | 19.56 GiB | 0.003990 | [0.003268, 0.004797] | 97.36 % | −0.001901 | **36/36** |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.005891 | [0.004901, 0.006985] | 96.72 % | — | — |

Receipts: `receipts/near-duplicate-v4.json` and
`receipts/qualification-v4-contamination-corrected.json`. `tools/suite3.py` now scans every
12-token position and emits schema v5; the old fixed-stride scan is retired.

## What it does and does not settle

**Settled:** the ordering is not an artefact of tuning against the measurement. On the
conservative clean subset, every build beats official FP8 on all 36 paired contexts and the
relative advantage is **47 %, 41 % and 32 %**. The contamination correction therefore changes
the magnitudes slightly but not one ordering, paired win, or conclusion.

**Not settled, and worth saying plainly:** absolute KLD is about 2.2–2.4x lower on v4 than on
v3 for every candidate, official FP8 included. That is a property of the corpus, not of the
models — different documents, different tail behaviour. **KLD magnitudes are only comparable
within one suite.** The clean subset was defined after the overlap bug was discovered, so a
new source-clean v5 qualification is still preferable to treating this correction as
preregistered evidence.

## Downstream smoke, added after qualification

A deterministic paired run now covers 40 generated tasks — 10 each arithmetic,
builtins-only code with three executable cases, exact-list instruction following, and
tool-call schema. BF16 solved 40/40; every EXL3 build and official FP8 retained **40/40 with
zero regressions**. A second offline pass hardened tool schemas, distinct-list requirements
and code-input immutability; all pass decisions remained unchanged. It also fixed the
diagnostic exact-output comparison: the original compared `3/3 cases` summaries for code
instead of final answer text. Correct exact-final-answer agreement with BF16 is hydrated
35/40, online K5/K6 34/40, FP8 34/40, K4 33/40 and context 32/40; every differing answer still
met its observable contract. Wilson 95 % lower bound for each 40/40 pass rate is 91.2 %.
This is a transparent smoke suite, not a leaderboard. Receipts:
`receipts/task-retention-v2-summary.json`, `receipts/task-retention-v2-strict-rescore.json`
and `receipts/tasks-v2-*.json`; the strict receipt supersedes only the original
exact-output-agreement diagnostic.

Still unsettled: representative public downstream tasks, OCR/document/video quality beyond
the synthetic multimodal set, and long-context accuracy beyond needle retrieval. KLD remains
distribution divergence through a shared BF16 head rather than an end-to-end capability score.
