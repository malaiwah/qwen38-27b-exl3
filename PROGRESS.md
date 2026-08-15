# Progress log

## 2026-08-14

**Artifacts staged.** Downloaded and byte-verified against the HF API at pinned
revisions (0 size mismatches, 0 missing across 121 files / 156.5 GB):
`Qwen/Qwen3.6-27B@6a9e13bd`, `nvidia/Qwen3.6-27B-NVFP4@0893e160`,
`Qwen/Qwen3.8-27B@1d4bf0f2`, `unsloth/Qwen3.8-27B-NVFP4@9c73e2da`. Stored on the
777 GB overlay at `/var/tmp/models` (the durable 100 GB volume cannot hold all
four); the overlay is container-local, so a rebuild means re-downloading.

**Reference recipes measured**, not read off cards — see
[docs/01-nvfp4-composition.md](docs/01-nvfp4-composition.md). Both vendors
4-bit only the MLP and keep attention at 8-bit; both keep the vision tower BF16.

**Recipe fixed** — [docs/02-recipe-k4.md](docs/02-recipe-k4.md). MLP K4,
attention BF16-on-disk for runtime K6, `lm_head` K6, MTP K6, embed/vision BF16.
18.75 GB VRAM vs NVIDIA's 21.92 GB, with the headroom exchange rates recorded
for a later `down_proj` promotion.

**Runtime contract established** — [docs/03-gg-runtime-contract.md](docs/03-gg-runtime-contract.md).
Four findings that change the plan: no Qwen `MODEL_FAMILY` (bypass the
launchers, call `vllm serve` directly), `--enforce-eager` is mandatory for a
dense EXL3 checkpoint, `--quantization exl3` must be explicit, and the
online-K6 `ignore` list must be written from scratch or the overlay will claim
the vision tower and silently degrade it to MXFP8.

**Toolchain gaps identified** — [docs/04-exllamav3-toolchain.md](docs/04-exllamav3-toolchain.md).
exllamav3 supports this architecture first-class but has no per-module bit
override and cannot emit BF16 for a decoder linear, so the plan is convert-then-splice.

**Baselines proven** — [docs/06-baseline-validation.md](docs/06-baseline-validation.md).
No container runtime is available (seccomp denies user/mount namespaces), so the
r34 image was pulled with an unprivileged OCI puller and is run through `proot`.
Both `unsloth/Qwen3.8-27B-NVFP4` and BF16 `Qwen/Qwen3.8-27B` serve and answer
requests under the official image, at full GPU speed.

**Card review landed** — [docs/07-serving-recommendations.md](docs/07-serving-recommendations.md)
and [docs/08-upstream-cards-digest.md](docs/08-upstream-cards-digest.md), plus
`https://recipes.vllm.ai/Qwen/Qwen3.8-27B`. Findings that changed our plan:
both NVFP4 vendors ship an undocumented FP8 KV-cache scheme (so KLD must pin one
KV dtype), both deliberately preserve the MTP head (so ours stays BF16 too), the
native context is 262144 with the 1M override nested under `text_config`, and
Unsloth's chat-template edits are partly undisclosed (we ship upstream's).

**exllamav3 1.4.2 installed inside the image rootfs** (`5f3c537`), because the
image's bundled copy is 0.0.43 and has no converter. The CUDA extension is being
JIT-built for `TORCH_CUDA_ARCH_LIST=12.0` against the image's torch 2.12.0+cu132.

**Iteration 1 built, served and measured** — [docs/10-results-iteration-1.md](docs/10-results-iteration-1.md).
K4 conversion took ~33 min on one GPU; the spliced checkpoint serves under the
official r34 image at **17.89 GiB resident weights** (predicted 19.28 GB, 0.4 %
error), answers text and image prompts correctly, and scores **mean KLD 0.034030
(run SD 0)** against the BF16 teacher. Published to
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4).

**Upstream defect filed** — [local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311):
the online overlay's MXFP8 fallback raises for shapes divisible by neither 128
nor 32, which is exactly this architecture's vision tower. Patch pushed to a fork
branch, PR opened after behavioural verification on this box.

**v3 held-out re-measurement.** The v2 suite came from exllamav3's calibration
corpora — the text our own quant was tuned on. Rebuilt the suite from Gutenberg,
arXiv, Wikipedia (9 languages) and CPython with a shingle contamination scan (0 hits),
181 contexts / 370,507 positions, 41 real source clusters, analysis/qualification
partitions, 32 sentinels. Ours moved 0.026231 -> 0.030736, NVFP4 -> 0.094978,
FP8 -> 0.013126. Advantage over NVFP4 grew to 3.09x (136/136 contexts);
deficit to FP8 grew to 0.0176. Noise floor measured at exactly 0.000000.

**CUDA graphs landed.** Autotune-priming patch verified end to end: three priming
lines, decode-only capture, +92 %/+84 %/+98 % decode throughput, now faster than the
NVFP4 checkpoint. [PR #314](https://github.com/local-inference-lab/vllm/pull/314).
The receipt published with it also claimed "exact distribution parity against eager
(KLD 0.000000)"; that claim was **withdrawn the next day** — it captured a prefill
forward, which `FULL_DECODE_ONLY` never graphs. See the 2026-08-15 entry and
[docs/27](docs/27-graph-decode-drift-control.md).

**Everything needed to contest our numbers is published** as
[`malaiwah/qwen38-27b-fidelity-suite-v3`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3):
tokens, reference and candidate hidden states, the shared LM head, sentinel repeats,
reports and checksums — recomputable without a GPU.

**Weakest control identified:** replay qualification 6.54e-04 versus the reference
protocol's 1.23e-06, which sets a ~1e-3 resolution floor and invalidates the
iteration-1 head-attribution figure until fixed. Top of the iteration-2 list.

## 2026-08-15

**Iteration 2 built, measured and published** — [docs/22](docs/22-results-iteration-2.md).
Gate K5 / up K5 / down K6, `mcg` throughout including `lm_head`, quantized MTP draft,
attention still BF16-on-disk for the runtime's K6 overlay. Enabled by a 30-line
`EXL3_BITS_OVERRIDE` addition to exllamav3's allocator, since upstream exposes only a
global `--bits`. Conversion 38m52s, whole-model 4.95 bpw, per-block `rfn 0.0066 /
sqnr 45.1 dB`. 30.60 GB download serves at **20.32 GiB resident weights** and measures
**mean KLD 0.008157** body-only (**0.008284** as served, since the K6 head adds
0.000127) with top-1 96.97 % on the held-out analysis partition: **-0.004969 versus
official FP8** (95 % CI [-0.00643, -0.00371], 136/136 contexts) and -0.022579 versus the
K4 release. Published as
[`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6).
Speculative decoding with the quantized draft works: 58.2 % acceptance, 2.745 tokens per
step, single-stream 56.5 -> **113.8 tok/s**.

**Prefill dispatch landed** — [docs/24](docs/24-p0-results.md). Row-count dispatch
*inside* the registered `vllm::exl3_gemm` op (`exl3_gemm` below 128 rows, reconstruct +
`hgemm` at or above it) took prefill from 2,369/2,362 to **5,050/5,146 tok/s** at 2k/6k
with decode unchanged. A Python `if` *around* the two calls does not work: vLLM resolves
the branch once at trace time and the decode graphs then capture reconstruct+hgemm,
collapsing C1 decode to 22.6 tok/s. The patch is not bit-exact (fp16 summation order):
it costs **+0.43 %** of measured divergence, disclosed, and
`VLLM_EXL3_PREFILL_RECONSTRUCT_M=0` restores the previous path.
[PR #316](https://github.com/local-inference-lab/vllm/pull/316).

**Prefill attributed, then bounded** — [docs/26](docs/26-prefill-attribution.md). Four
configurations over the two candidate causes: the **MLP kernel is the whole story**
(2.13x with the overlay on, 2.26x with attention BF16) and the **online-K6 attention
overlay is not the bottleneck** (1.05x / 1.11x) while saving 6.7 GB. That refutes the
hypothesis iteration 2 closed with. `ext.hgemm` is already at cuBLAS parity
(0.92-1.06x), and even a perfect fused reconstruct lands at 7-8k tok/s end to end
against FP8's 10,667. The goal is therefore re-scoped on evidence
([docs/25](docs/25-goal-pareto-dominate-fp8.md)): dominate FP8 on fidelity, memory, decode and
speculative decode — done and measured — and publish prefill as a structural deficit of a
4-bit-class trellis format in this runtime.

**Graph-vs-eager parity retracted and re-measured** —
[docs/27](docs/27-graph-decode-drift-control.md). An outside qualification of PR #314 on an
RTX 5090 found graph decode internally deterministic but not bit-identical to eager. My
own "KLD 0.000000" receipt could not have contradicted that: `fidelity.py capture` takes
one prefill forward and `FULL_DECODE_ONLY` captures no prefill graph, so it compared two
runs of the same eager prefill. Withdrawn upstream. The replacement probe
(`tools/decode_parity.py`, 32 prompts x 32 greedy tokens) measures decode directly:
**24/32** exact sequences, mean |delta logprob| **0.0118** on the chosen token, each mode
internally deterministic 32/32. Unquantised BF16 on the same build drifts identically
(**24/32**, **0.0128**), so the drift is ambient to this build's graph decode path and is
not caused by the quantisation. Randomised autotune priming changed nothing (25/32, same
deltas to five digits), killing the zero-priming hypothesis.

**Attention-overlay width is a runtime knob, and it is measured.** Same suite, same
reference and comparator head, varying only `ONLINE_QUANT` bits: K6 **0.008157** at
20.32 GiB, K5 **0.012135** at 19.82 GiB, K4 **0.027530** at 19.05 GiB. Paired against K6,
K5 costs +0.003978 and K4 +0.019373, both 0/136. K5 remains below official FP8's 0.013126.
The initial native-262,144 claim from a capped 96 GB simulation was later retracted after a
real RTX 5090 test; K5 reached **206,400** with exact retrieval. See
[docs/28](docs/28-external-validation-and-corrections.md).

**Offline attention beats the online overlay, slightly.** A "hydrated" build with the
same MLP recipe but attention **serialized at K6 by the converter** instead of encoded at
load measures **0.007406** — 0.000751 better than the published checkpoint (95 % CI
[-0.00098, -0.00057], 124/136 contexts) and 0.005719 better than FP8 (136/0). It costs
9 GB of download and nothing in VRAM, and removes the cold-start encode.

**fp32 replay was a negative result** — [docs/24](docs/24-p0-results.md). Storing hidden states
in fp32 moved the replay qualification only 6.54e-04 -> 6.25e-04 (-4.5 %), so operand
rounding is ~5 % of the floor and the rest is the implementation difference between vLLM's
logit path and our replay. Paired comparisons are unaffected (common-mode); only absolute
values below ~1e-3 stay unresolvable.

### Later that day: qualification, context edition, and corrected capacity accounting

The v4 source-disjoint suite was frozen before candidate scoring: 160 contexts from 100
documents, zero token/document/content overlap with v3. On its 42-context qualification
partition, hydrated scored **0.003029**, online K5/K6 **0.003395**, context edition
**0.003900**, and official FP8 **0.005720**. All five candidate capture sets and receipts are
published in the dataset, now 2,708 files / 51.0 GB.

A fourth checkpoint serialized attention at K5. Its BF16-embedding baseline is 19.31 GiB and
0.009673 v3 mean KLD. Per-row int8 input embeddings reduce resident weights to **18.13 GiB**
for **+0.000065 KLD**. Under a 30.24 GiB vLLM budget the server starts at native 262,144,
allocates 279,007 KV tokens, and retrieves 3/3 planted codes from 227,334-token prompts.
Because the host was a 96 GB SM120, this is an engine-budget proof rather than a hard physical
RTX 5090 proof; that rerun remains open.

An MTP follow-up also corrected a false lead. vLLM aliases the draft embedding to the target
after loading, so the claimed int4 draft acceptance comparison never exercised int4. The
method and environment variable were removed; PR #319 and all cards now record the correction.
At concurrency one with one decode graph, MTP-1 still needs 8.83 GiB against 8.59 available
(estimated max 255,024), while MTP-3 needs 9.13 GiB. Compact KV grouping reduced padding but
raised graph memory and lost overall.

### Final audit pass: overlap correction, task retention, native MTP

The original fixed-stride contamination scanner was offset-sensitive. An all-position
normalized 12-token audit (Unicode words plus Han/Kana characters) found overlap in two v3 source documents and four v4 qualification
documents. Conservatively excluding every context from those sources changes v3 K5/K6
0.008157 → **0.007945**, hydrated 0.007406 → **0.007172**, K4 0.030736 →
**0.029679**, context 0.009673 → **0.009378**, FP8 0.013126 → **0.012798**, and
NVFP4 0.094978 → **0.092727**. On corrected v4 qualification, hydrated / online /
context / FP8 are 0.003093 / 0.003455 / 0.003990 / 0.005891; all three EXL3
profiles still win 36/36 paired contexts. Both original and corrected receipts remain.

Capture/replay now fails closed on interrupted or modified data, and the frozen-suite builder
fails on language shortfalls. The corrected deterministic downstream matrix then completed:
BF16, all four EXL3 profiles and official FP8 each passed **40/40**, zero regressions.
A hardened offline rescore preserves every pass and corrects exact-final-answer agreement to
32/40–35/40; the original 37/40–40/40 diagnostic compared only `3/3 cases` summaries for
code tasks.

The remaining native-MTP gap was multimodal activation profiling, not another model weight.
Setting the image ceiling to 8,388,608 pixels under the exact 30.24 GiB budget reduced peak
activation to 1.78 GiB and allocated **266,612 KV tokens** with MTP-3 and decode graphs.
Generation retrieved exactly from 261,794 text tokens. A second request combined 229,910
measured text tokens with a 3,072 × 2,304 image: the server reported 236,824 prompt tokens
and returned exact `1376346594 | red, blue`. The synthetic vision score stayed 24/30 and
one warmed C1 decode run measured 98.72 tok/s. This remains an engine-budget proof on the
96 GB SM120; physical RTX 5090 qualification is P0.

### Next

Run the exact profile on a physical RTX 5090, publish an immutable runtime image containing
the open patches, then extend the paired smoke test to public capability and real multimodal
sets. Current ranked work and acceptance gates are in
[docs/29](docs/29-plan-and-loose-ends.md).

## 2026-08-15 (continued): the 10,480,640-position held-out rerun

**The objection was fair, and it is now answered.** Every fidelity claim up to this point
rested on the v3 analysis partition — 136 contexts / **278,392** scored positions from a
suite of 41 source clusters ([docs/18](docs/18-results-fidelity-v3.md),
`receipts/v3-report-v2-analysis.json`, tracked as F1 in
[docs/29](docs/29-plan-and-loose-ends.md)). The objection from readers was that this is too
thin to carry a "38 % lower divergence than official FP8" headline, and it was a good
objection: the bootstrap had ~41 independent units. This session re-measured at **37.6x the
volume** — **10,480,640 scored positions** over **842 source clusters**. Nothing older was
rewritten. The v3 and v4 receipts stand exactly as published and remain valid within their
own suites; they are superseded only as *the headline number*.

**v5 corpus, pinned by digest.** `tools/fetch_corpus_v5.py` assembled a five-stratum corpus
of **941 documents / 70,348,971 bytes** (code 241, encyclopedic 323, literary 59,
multilingual 276, scientific 42) from Project Gutenberg, arXiv, Wikipedia in **21 languages**,
and CPython **v3.13.1** plus NumPy sources. `receipts/kld5-corpus-fetch-log.json`
(`kld5-corpus-fetch-log/1`) enumerates every document with its URL and sha256 — the final
top-up pass fetched 79 CPython files and re-verified the 862 already present, with 0 failures,
0 skips, 0 unmet stratum targets, and the v3 and v4 corpus roots (`/var/tmp/work/kld3/corpus`,
`/var/tmp/work/kld4/corpus`) explicitly excluded as sources. Note for whoever reads both
files: the manifest's prose `corpus_note` still says "CPython v3.12.8", which is stale
carried-forward text — the fetch log's per-document `codeload.github.com/.../tags/v3.13.1`
URLs and digests are the authority, and 79/79 fetched code documents are v3.13.1.

**Contamination was handled before selection, not audited afterwards.** The all-position
12-token normalized scan (NFKC, casefold, one token per Han/Kana character, stride 1,
blake2b-128; 10,772,868 corpus shingles against 859,426 exllamav3 calibration shingles) ran
over each *complete* document before any window was cut. **44 of the 941 discovered documents**
— 43 code, 1 encyclopedic — were excluded whole on any match, leaving **897 eligible**
(`receipts/kld5-suite-manifest.json` `/document_scan`). Contamination hits in the emitted
suite are therefore 0 *by construction*: `contexts_with_any_hit` 0, `total_hits` 0,
`decoded_candidates_rejected` 0. (An earlier plan number of "17 excluded" came from an aborted
build against the smaller pre-top-up corpus and is wrong; 44 -> 897 is what the manifest says.)

**Suite.** `receipts/kld5-suite-manifest.json`, schema `qwen38-distribution-fidelity/6`,
`suite_token_sha256` `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88`:
**5,120 contexts** (1,024 per stratum) x 2,047 scored positions = **10,480,640**, drawn from
**842 source clusters**. Windows advance by exact tokenizer character offsets and never
overlap; an independent check of the built suite found **5,120/5,120 unique context token
hashes and 0 overlapping windows**. All **160** v4 context token hashes were seeded as
exclusions and **0** were reachable, so v5 is token-disjoint from the frozen v4 suite as well
as calibration-disjoint.

**Why the run is sharded — 64 GB against 135 GB.** One 512-context capture is
512 x 2,047 x 5,120 in bf16 = **10.7 GB per model**, so a shard covering all six models is
**~64 GB of hidden states** while `/var/tmp` scratch holds **~135 GB**. `tools/kld_ladder.sh`
therefore walks one shard at a time: capture six models over 512 contexts, replay the five
candidates against the BF16 reference through **one shared BF16 LM head** (so every number
below is body-only, both operands through the same head), verify every report against the
pinned suite / head / comparator / model identities, delete that shard's 64 GB, then start the
next. Resume is fail-closed in both directions: an already-verified shard is skipped without
recapturing, and a report describing a different suite, shard, head or model identity aborts
the run rather than being silently reused. `tools/kld_aggregate.py` welds the ten verified
per-shard reports into `receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`
(schema `qwen38-kld-ladder-cumulative/2`).

**Cumulative results, 10,480,640 positions each.** 95 % CI is a source-cluster bootstrap
(10,000 resamples, 842 clusters); "max" is the exact worst single position in the whole run.

| candidate | mean KLD | 95 % CI | top-1 | max position |
|---|---|---|---|---|
| hydrated [`…-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | **0.002760** | [0.002540, 0.003020] | 97.70 % | 8.258 |
| online K5/K6 [`…-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | **0.003210** | [0.002982, 0.003480] | 97.52 % | 22.241 |
| context [`…-EXL3-K5K6-context`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context) | **0.003509** | [0.003220, 0.003852] | 97.44 % | 5.557 |
| official [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | **0.005294** | [0.004927, 0.005728] | 96.79 % | 10.714 |
| K4 [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | **0.010604** | [0.009640, 0.011746] | 95.76 % | 14.283 |

**Paired per-context differences** — the comparison that actually transfers across suites,
bootstrap 10,000 resamples, seed 1, 842 clusters, `receipts/kld5-10M-paired.json`:

| pair | difference | 95 % CI | contexts won |
|---|---|---|---|
| hydrated - FP8 | **-0.002534** | [-0.002708, -0.002383] | **5,118 / 5,120** |
| online K5/K6 - FP8 | **-0.002084** | [-0.002249, -0.001942] | **5,105 / 5,120** |
| context - FP8 | **-0.001785** | [-0.001884, -0.001697] | **5,109 / 5,120** |
| K4 - FP8 | **+0.005310** | [+0.004710, +0.006019] | 7 / 5,120 (K4 wins) |
| hydrated - online K5/K6 | **-0.000450** | [-0.000469, -0.000433] | **4,922 / 5,120** |

So the three K5/K6-class profiles beat official FP8 on essentially every one of 5,120
held-out contexts, offline-serialized attention still edges the runtime overlay by a small
but unambiguous 0.000450, and K4 loses to FP8 on 5,113 of 5,120.

**The ladder was stable long before it finished.** Cumulative hydrated mean at the
1M / 2M / 5M / 10M checkpoints: **0.002700 / 0.002759 / 0.002699 / 0.002760** (recomputable
from the `shards[]` array of `receipts/kld5-10M-hyd.json`: 1,048,064 positions per shard).
The estimate never moved by more than **6.1e-05** (2.2 % relative) after the first million
positions, which is the useful result about evidence volume — the 278,392-position estimate
was not *wrong*, it was under-supported, and 10.48 M positions mostly buys a defensible
interval and 842 bootstrap units instead of 41.

**Tail statistics: a real gap, and the fix landed after the run.** The ten shard reports were
`qwen38-fidelity-report/1`, which carries no KLD histogram, and per-shard percentiles cannot
be recombined into a cumulative percentile. So for this run only **per-shard p50/p95/p99/p999**
and the **exact global maximum** exist; the cumulative receipts record that honestly in
`not_aggregable.token_percentiles` rather than printing a recombined number that would be
arithmetic fiction. The upgrade shipped immediately afterwards: replay now counts every scored
position into a fixed log-spaced histogram with explicit zero/underflow and overflow buckets
(`KLD_HIST_*` in `tools/fidelity.py`), stores edges plus counts plus the exact max, and
`tools/kld_aggregate.py` derives bin-bounded cumulative quantiles from the summed counts.
That bumps the report schema to `qwen38-fidelity-report/2` with every `/1` field unchanged —
so the *next* run answers "what does the 99.9th percentile look like over 10 M positions",
and this one cannot be retrofitted because its hidden states are gone.

**First public-benchmark numbers, not a private smoke test.** `tools/public_capability.py` ran
a paired MMLU-Pro subset — 70 questions, 14 official categories x 5, official five-shot
prefixes, pinned `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, greedy,
thinking at low effort, 5,120-token completion cap (`receipts/public-capability-plan.json`,
`receipts/public-capability-suite-mmlupro-70.json`). BF16 scores **57/70** (Wilson 95 %
[70.8 %, 88.8 %]) and K4 also **57/70**, with **BF16-pass retention 55/57** (Wilson lower
bound 88.1 %), 2 regressions, 2 improvements, pass-outcome agreement 66/70, and
**exact-output agreement 0/70** — long chains of thought never match token-for-token, which is
why exact-match agreement is a useless metric for a thinking model and retention is the one to
read. Four BF16 items hit the completion cap and are counted as failures, not excused. The
earlier 2,048-cap control is kept as `receipts/public-capability-bf16-superseded-cap2048.json`
(with its plan) rather than deleted. Receipts: `receipts/public-capability-bf16.json`,
`receipts/public-capability-k4.json`. The remaining four candidates are **not yet run**.

**Collection index.** `tools/collection_index.py` now emits `receipts/collection-index.json`
(schema `qwen38-collection-index/1`): one immutable row per published checkpoint — k4, k5k6,
hydrated, context — where every field carries the receipt path, that file's sha256 and the
RFC 6901 pointer it was read through, a `null` always carries a `not_verified` note, and an
optional field the receipt never wrote is *absent* rather than null. It also fixes the memory
vocabulary that kept leaking into cards: `whole_tree_bytes` / `immutable_payload_bytes` /
`tensor_payload_bytes` are disk sizes, `resident_weights` is the only memory figure and is
measured GPU allocation at load — serialized bytes are never VRAM. One `known_divergences`
entry is recorded instead of being fixed, because fixing it would mean rewriting a published
receipt: the 8.4 MP amendment pinned the context release evidence's digest at 11:11:08Z and
that file was then amended at 12:27:32Z to link back to the amendment, so the pinned digest
can no longer match.

**Operational lesson worth more than it cost: never edit a running `bash` script.** All ten
shards had already been captured, replayed, verified and released when `tools/kld_ladder.sh`
was edited in place to improve its summary output. `bash` reads a script incrementally from a
byte offset, so the *running* instance resumed at a shifted offset and died with **exit 127**
on the final summary line. **No measurement was lost** — the per-shard reports and the five
cumulative receipts were already on disk and verified, and the exit code refers to a summary
line, not to the ladder. The lesson is procedural, not technical: a long-running shell driver
is part of the running experiment, so edits go to a copy and land between runs. Recorded here
because a future reader who finds exit 127 in the shell history deserves to know it is a
cosmetic failure after a complete run, not a truncated measurement.

**Honest limitations of this run.**

- **v5 absolute KLD is not comparable to v3 absolute KLD.** The corpus mix differs, so the
  numbers move together: K4 reads **0.029679** on corrected v3 and **0.010604** here. Only
  within-suite ordering and paired differences transfer across suites; anyone quoting a v5
  mean beside a v3 mean is comparing two different measurements.
- **No cumulative percentiles for this run**, per the tail note above — per-shard p95/p99/p999
  and the exact global maximum only.
- **Not reproducible from published captures.** The hidden states were deleted shard by shard
  to fit 135 GB of scratch, so unlike the v3 dataset this run is reproducible from the pinned
  corpus fetch log and suite manifest plus a GPU, not from downloadable tensors.
- **Public capability is two of six candidates.** BF16 and K4 only; the three K5/K6-class
  profiles and official FP8 are pending on the same pinned suite.

### Still 2026-08-15: the 99.9th percentile, measured on a shard-0 rerun

**The histogram harness was exercised, not just shipped.** Rather than leave
`qwen38-fidelity-report/2` as an untested promise for the next volume run, **shard 0 of the
same v5 suite was re-captured and re-replayed with the `/2` harness** — the identical 512
contexts, 2,047 scored positions each, **1,048,064 positions per candidate**, every candidate
scored on the same contexts through one shared BF16 LM head. `tools/kld_aggregate.py` then
welded the five `/2` reports into `receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`
(schema `qwen38-kld-ladder-cumulative/2`), each carrying the summed 560-bin histogram, the
bin-bounded quantiles with their `lower` / `upper` / `estimate`, the exact maximum and the
exact exceedance counts.

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | above 0.1 | above 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| hydrated | 0.002700 | 0.00109 | 0.0082 | 0.0276 | **0.1319** | 0.463 | 3.735 | **0.1534 %** (1,608) | 0.00219 % (23) |
| online K5/K6 | 0.003141 | 0.00128 | 0.0099 | 0.0321 | **0.1446** | 0.498 | 5.507 | **0.1820 %** (1,907) | 0.00200 % (21) |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | **0.1642** | 0.587 | 3.749 | **0.2287 %** (2,397) | 0.00305 % (32) |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | **0.2438** | 0.812 | 5.296 | **0.3912 %** (4,100) | 0.00592 % (62) |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | **0.5555** | 1.870 | 7.565 | **1.2604 %** (13,210) | 0.03807 % (399) |

**What this settles.** The standing objection — "avg KLD and top-1 are meaningless, what does
the 99.9 % tail look like?" — now has an answer measured on the same held-out contexts as the
headline: the ordering at **p50, p95, p99, p99.9 and p99.99 is the same as the ordering of the
means**, so for these candidates the mean is not hiding a worse tail. Every EXL3 K5/K6-class
build has a lighter tail than official FP8 at every measured quantile — at p99.9 hydrated
**0.1319** against FP8 **0.2438**, and 0.1534 % of positions above 0.1 nats against FP8's
0.3912 % — and **K4 is worse than FP8 at every quantile**, including 1.2604 % of positions
above 0.1 nats against 0.3912 %.

**What it does not settle.** This is **one 1,048,064-position shard**, not the full
10,480,640-position run: the ten-shard ladder above predates the histogram and cannot be
retrofitted, since its hidden states are gone. Cumulative histograms over all ten shards
require re-running the other nine with the `/2` harness — roughly **6 hours of GPU time**, not
yet done. The quantiles here are **bin-bounded**, not exact: each is the log-spaced bin that
provably contains it, ~5.6 % wide, which is why every receipt publishes `lower` / `upper` /
`estimate` instead of a single fabricated digit. The **maxima and exceedance counts are exact**.
The 10 M receipts remain the authority for the full-run means, bootstrap intervals and paired
results; nothing above was rewritten.

### Still 2026-08-15: the public-capability sweep finished, and the bar it mostly does not clear

**All six models are now on the same pinned suite.** `tools/run_public_capability.sh` drove the
four remaining candidates — online K5/K6, hydrated, context edition, official FP8 — against the
frozen BF16 report, so the earlier note in this entry that public capability covers "two of six
candidates" is superseded (that text stays as written; nothing above was rewritten). Same 70
questions, 14 official categories x 5, official five-shot prefixes, pinned
`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, greedy, thinking at low effort,
5,120-token completion cap, every candidate scored item-paired against the BF16 control:

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---|---|---|---|---|---|---|---|
| BF16 `Qwen/Qwen3.8-27B` | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | `receipts/public-capability-bf16.json` |
| context edition | **58/70** (82.9 %) | [72.4 %, 89.9 %] | 56/57 | **90.7 %** | 1 | 2 | 3 | `receipts/public-capability-ctx.json` |
| K4 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | 55/57 | 88.1 % | 2 | 2 | 4 | `receipts/public-capability-k4.json` |
| hydrated | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 54/57 | 85.6 % | 3 | 2 | 4 | `receipts/public-capability-hyd.json` |
| official FP8 `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | `receipts/public-capability-fp8.json` |
| online K5/K6 | 55/70 (78.6 %) | [67.6 %, 86.6 %] | 54/57 | 85.6 % | 3 | 1 | 4 | `receipts/public-capability-k5k6.json` |

**Acceptance, per model, against the plan written before the runs.**
`receipts/public-capability-plan.json` pre-registered two conditions: BF16-pass retention with a
Wilson 95 % lower bound at or above **0.90**, and no category losing more than two BF16 passes.
The category condition is met by all five candidates — worst per-category loss is two passes
(hydrated and online K5/K6, both in philosophy). The retention condition is met by exactly one:

- **context edition — pass.** 56/57, Wilson lower **90.7 %**, and the only candidate above the
  absolute BF16 score (58/70 against 57/70).
- **K4 — shortfall.** 55/57, Wilson lower **88.1 %**.
- **official FP8 — shortfall.** 55/57, Wilson lower **88.1 %** — the vendor format misses the
  same bar, on the same items, under the same scorer.
- **hydrated — shortfall.** 54/57, Wilson lower **85.6 %**.
- **online K5/K6 — shortfall.** 54/57, Wilson lower **85.6 %**.

**Why four shortfalls are a statement about the suite, not the builds.** With 57 BF16 passes as
the paired denominator, **56/57 is the smallest count whose Wilson 95 % lower bound clears
0.90** — one paired miss already fails the bar. A 70-item suite simply has too few items to
certify what was pre-registered, and every candidate's absolute interval overlaps every other's,
so this matrix separates nothing at this size. It is a power limitation, and the cleanest proof
that it is a power limitation rather than a quantization verdict is that **official FP8 fails it
too**. Exact-output agreement is **0/70** for every EXL3 candidate and **1/70** for official FP8
(question 8278, math, a 113-token answer both models get right): long chains of thought differ
token-wise, so the pass/fail outcome is the only meaningful pairing. Four BF16 items hit the
completion cap and are counted as failures, not excused; the context edition hits it three
times. This is a first public, licence-compatible, item-paired benchmark, not a leaderboard
claim.

**Operational note: every EXL3 candidate needs `VLLM_EXL3_GRAPH_DECODE=1`.** The sweep serves
each model with `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, and the EXL3
loader is deliberately fail-closed: without the env var the pre-capture `exl3_gemm` priming pass
is disabled, the backend refuses CUDA-graph capture rather than capturing unprimed kernels, and
the server **exits during startup** instead of quietly falling back to eager. That was found the
hard way — the first candidate server of this sweep failed to start — which is why
`tools/run_public_capability.sh` now sets the environment per candidate: hydrated
`VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`, context the same plus
`VLLM_EXL3_EMBED_BITS=8`, K4 and online K5/K6 the same plus `VLLM_EXL3_ONLINE_TRELLIS_BITS=6`
and `VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online`, official FP8 nothing. Each receipt records
the exact string it ran under in `runtime.env`, so this is checkable and not folklore.

**New tool: `tools/run_public_capability.sh`.** One server at a time through the pinned rootfs;
it refuses to start unless the suite, plan, BF16 reference report and BF16 runtime record all
exist, because a candidate scored without the reference measures absolute accuracy and nothing
else. Readiness is verified on `/v1/models` (the served id must actually be present) before any
request, the server is stopped before the next model loads, and an already-written report is
skipped rather than overwritten. Release evidence is attached per candidate where it exists.

**Next step is items and task types, not another sweep of these 70 questions.** Re-running the
same suite cannot move a bound set by its denominator. The plan's own P1 is the fix: HumanEval+
/MBPP-style executable cases, IFEval-style verifiable constraints, schema-constrained tool
calls, and a larger MMLU-Pro draw. No capability claim graduates from "measured" to "certified"
before that.
