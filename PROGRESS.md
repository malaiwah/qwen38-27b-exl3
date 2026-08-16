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

## 2026-08-16: the context edition is hardware-qualified on a physical RTX 5090, at 0.955 rather than 0.97

**P0 rank 1 is closed.** The native-context claim no longer rests on an engine budget. The
context edition (`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`, revision
`c45c273b0d6ef2859cb2d85b36dd52253c80d878`) ran on **one physical NVIDIA GeForce RTX 5090**,
`GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a`, 32,607 MiB total and 32,149 MiB free with the card
idle, which vLLM sizes as **31.4 GiB usable**; driver 610.57.04, CUDA user-mode driver 13.3;
image `voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`
plus the three content-pinned patch modules, sha256-verified before every launch; vLLM
`0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34`. Receipt
`receipts/qualification-5090-context.json` (schema `qwen38-qualification-5090-context/1`), with
per-process server logs `receipts/qualification-5090-context-server-{B3,B4,C,D,E,F}.log`.

**The recipe is corrected, in exactly two places.** The qualified profile is
**`--gpu-memory-utilization 0.955` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**;
the cards printed 0.97 and set no allocator config. Nothing else moved: `--max-model-len 262144
--max-num-seqs 1 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 --speculative-config
'{"method":"mtp","num_speculative_tokens":3}' --mm-processor-kwargs
'{"truncation":false,"max_pixels":8388608}'`, `VLLM_EXL3_EMBED_BITS=8`,
`VLLM_EXL3_GRAPH_DECODE=1`, `VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`. Window, MTP depth, image
ceiling, KV dtype, sequence count and cudagraph mode are all unchanged from the published card.

**What the card measured at startup.** Engine budget **29.98 GiB** at utilisation 0.955; free
30.9 of 31.4 GiB; usage 18.19 weight + 1.78 peak activation + 0.27 non-torch + 0.45 CUDAGraph =
**20.69 GiB**; **available KV 9.28 GiB = 265,122 KV tokens**; maximum concurrency at 262,144
tokens **1.01x**; attention block size forced to 1600 tokens so the attention page is at least
the mamba page; mamba page padded 0.25 %; 3 padding layers, at most 6.25 % KV waste; startup
**55.7 s**; model load 18.19 GiB in 3.99 s. The 20,672,382,926 bytes of shards behind that load
are serialized bytes in the vault cache and are never called VRAM.

**All seven gates PASS.**

1. **Startup native allocation** inside the utilisation ceiling — the numbers above, 55.7 s.
2. **Long needle exact** — 261,794 prompt tokens, planted code `1376346594` retrieved exactly.
3. **Combined long text plus seven megapixels** — 236,824 prompt tokens with a 3,072 × 2,304,
   7,077,888-pixel fixture, answered `1376346594 | red, blue` exactly.
4. **Image suite 24/30** — digits 8/10, bars 9/10, grid 7/10, seed 20260815. Exactly on the
   threshold: a pass with no margin, per-case answers published.
5. **Three warmed 256-token C1 decode runs** — median **107.56 tok/s**, dispersion 0.60 %, MTP
   acceptance **56.48 %** at mean accepted length 2.69. Physical-card numbers; they are never
   differenced against the rental RTX PRO 6000 figures.
6. **Second native-length request after release**, in the same process — recovery, not one lucky
   allocation.
7. **Receipt identity complete** — GPU UUID and model, driver, image digest, patch hashes, full
   commands, four memory points, KV allocation, outputs and wall times.

**0.97 is published as a bounded negative, and the mechanism is the interesting part.** At 0.97
the engine starts and serves text fine (KV 272,570 tokens, 1.04x at native length), but the
combined text-plus-image request dies with `torch.OutOfMemoryError` inside
`vllm/v1/attention/ops/vit_attn_wrappers.py`, wanting 62.00 MiB with 26.50 MiB free.
`expandable_segments` does not save it: the allocator hands memory back and **vLLM immediately
spends it on more KV**, 272,570 → 280,017 tokens. Lowering the image ceiling instead of
utilisation is **strictly worse** — at 0.97 with `max_pixels` 4,194,304 the profiled peak
activation falls to 1.35 GiB, KV grows to 291,933 tokens, and it OOMs sooner with 6.56 MiB free.
So **the right knob is utilisation, not the image ceiling**, because the engine spends every
freed byte on KV; and seven megapixels is **not** a hard 32 GB ceiling, since the identical
request succeeds at 0.955 with the full 8,388,608-pixel ceiling. Lowering `max_pixels` would
also silently downscale large images, which 0.955 does not.

**Prefix caching was a non-issue here, and that is measured rather than assumed.** This build
lacks upstream vLLM #51113, so an explicit `--no-enable-prefix-caching` control arm was run. It
found the published profile already runs with prefix caching **off** — `enable_prefix_caching=False`
in the engine config with no flag from us, and 0.0 % hit rate on every scheduler line — because
the build disables it for this hybrid mamba model. The #51113 poisoning path is therefore never
exercised by this profile and no gate depended on it.

**The production image exists and is verified, but its serving gates are not ours to claim.**
`localhost/vllm:gg-r34-patched`, manifest
`sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27`, carries all three
patch modules **inside** the image, digest-verified there against the published map instead of
bind-mounted, with the import machinery proven to resolve to them and no stale bytecode
(`receipts/production-image.json`, schema `qwen38-production-image/2`) — which closes the build
half of P0 rank 2. The serving half is not closed and the receipt says so itself: no recipe has
served a request from the image, so every runtime gate — mounts, writes, endpoint exposure — is
recorded as `null` rather than passed. Those gates are owned by the current 5090 performance
window, so rank 2 stays open as built-and-verified-with-serving-gates-pending.

**Published surfaces updated:** `README.md` evidence list, `docs/29-plan-and-loose-ends.md`
(rank 1 CLOSED, its residual pixel-count and concurrency frontier moved to rank 11, rank 2
re-statused), `docs/32-native-context-embedding-overlay.md` (its engine-budget proof annotated
as superseded, every number kept), and the three sibling cards
`MODEL_CARD-K5K6{,-hydrated}.md` and `MODEL_CARD-K4.md`. No published receipt was rewritten:
`receipts/native-mtp-8mp-amendment.json` stands as the engine-budget proof it always was, and
the new receipt names it as superseded rather than editing it. The sibling cards' own profiles
were **not** given the qualified utilisation, because none of them has been qualified at any
value on this card; they now carry the context edition's measured result and the same
utilisation caution instead.

**Card consistency pass: all 16 audit findings resolved and the four cards re-published.**
`receipts/card-consistency-audit.json` found 4 blockers, 5 drifts and 7 cosmetics across the
four published cards; `receipts/card-consistency-audit-resolution.json` maps every id to what
changed, with before/after values and the receipt each corrected figure was read from. Every
value was read out of its authoritative receipt rather than out of the audit's transcription of
it, and every site was located by content match because the audit's line numbers had drifted.
The blockers: `MODEL_CARD-K5K6-context.md` no longer frames itself as engine-budget-proven with
a pending 5090 check — title, lede, four-builds intro, figure alt, table row and section
heading all now carry the hardware-qualified framing that its own qualification section already
had; the hydrated and K5K6 cards no longer contradict their own tables by calling the context
profile budget-capped; `MODEL_CARD-K4.md` no longer presents checkpoint bytes as resident
weights or as VRAM (21.92 / 23.42 GB are labelled disk, resident is 22.91 GB for unsloth NVFP4
and 30.61 GB for FP8, and the wrong 2.7 GB and 4.2 GB deltas are replaced by the measured
3.70 GB); and the K4 figure alt text is rewritten from the asset's own generator, which is the
only authoritative source for a matplotlib path-rendered SVG. The drifts: the two 0.97 arms are
named (34.56 MiB free without the allocator env, 26.50 MiB with it and KV 272,570 -> 280,017),
every card now states that the context edition has two measured resident figures and why 18.41
GiB is published rather than 18.19, the download column is whole-tree bytes with its convention
stated in a note, and the context card's rental and physical-5090 throughput are labelled with
their hardware and carry the receipt's own rule that the two are never differenced. Two things
found beyond the audit and fixed: `DOCS-SHA256SUMS` pins the README digest in all four repos, so
each was regenerated and uploaded in the same commit as its card, and the 37.4 / 33.5 KB/token
KV rates now say they are ratios at one window rather than extrapolation coefficients, because
the pool is affine in the window. Local and published are byte-identical, proven by re-fetch:
51,108 / 63,905 / 63,177 / 57,849 B. The verdict receipt's rental-driver defect was handed to
its owner rather than edited here.

**Prefix caching: the image is promoted, the flag ships on three recipes of four, and the
native window is a measured no.** The release unit is now the four-module superset
`localhost/vllm:gg-r34-patched-apc`, manifest
`sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218`, config
`5ce31638c0a5…`, promoted on Main's call and **not rebuilt**: promoting the exact bytes that
`receipts/apc-poison-repro.json` arms C and E measured is worth more than cosmetic accuracy in
a label, so the image's own `io.malaiwah.image.qualified="false"` is left stale and superseded
by receipt, with the precedence rule recorded in three places and the general lesson — a
boolean `qualified` label is unfixable by construction, carry a pointer or nothing — written
into `receipts/production-image.json` under `notes_for_the_next_image_build`. The receipt is
schema `qwen38-production-image/3`: `release_unit` is now a structured pointer, because the
release unit and the top-level build record are no longer the same image. The GDN gate module
(#51812) was **not** baked in; ImmutableImage and ApcPoisonRepro declined that coupling
independently and Main ruled the same way, so it stays a documented optional overlay for
concurrent serving.

**Nine gates, and the native 262,144 window failed three ways.** `receipts/qualification-5090-apc.json`
(schema `qwen38-qualification-5090-apc/1`), with per-process server logs and
`receipts/qual-apc-raw/`. The banner gate earns its place immediately: every launch really did
report `enable_prefix_caching: True` and `mamba_cache_mode: align`, so what follows is a
statement about prefix caching and not about a flag the engine ignored. At
`--gpu-memory-utilization 0.955` the engine **refuses to start** — `align` rounds a 262,144-token
request up to 164 whole 1,600-token blocks, needing **9.29 GiB** against an unchanged **9.28
GiB** supply, and it names 260,800 as the longest window it could serve. At 0.9555 (pool exactly
262,144, 1.00x) it starts and **deadlocks**. At 0.9585 (pool 265,072, 1.01x, within 50 tokens of
the cache-off baseline's 265,122) it **livelocks**: prefill to 98.9 % of the pool, requeue with
the pool freed, re-prefill, 30-second period, ~960 tok/s wasted, **zero output tokens** in 656 s,
`num_preemptions_total` stuck at `0.0`, and 261,794 prefix-cache queries against exactly **0**
hits — the partial prefill is discarded rather than published, which is what makes the loop
stable rather than self-correcting. The pool is quantised (0.9585, 0.959 and 0.96 all give
265,072), so no utilisation on this card makes it work, and 0.97 is barred anyway by gate 3.
Filed upstream as `local-inference-lab/vllm` **issue #394**, with #51113 already applied so it is
explicitly not a cherry-pick request; GdnGateAtConcurrency's `scheduler.py:865-878` padding path
is credited as an adjacent finding with a different trigger and, on its own initiative and mine,
recorded as **not** the cause of this trace.

**So the ship is scoped, which is a better answer than either extreme.** Prefix caching is on in
the `k4`, `k5k6` and `hydrated` recipes, where the pool is roughly 32x the window and none of
this can bite — evidenced on the promoted digest by the four-recipe smoke in
`receipts/production-image.json`: all three start healthy, answer a text and an image request
exactly, and report the cache enabled. It is **off** in the context edition's native-window
recipe, which is otherwise unchanged. And it is **offered** there as a measured option:
`--max-model-len 256000` at utilisation 0.9585, pool 264,777, 1.03x, which retrieves a
255,000-token needle exactly in 180 s, completes three warmed decode runs and answers a second
long request after release — for **6,144 tokens of context, 2.34 %**. The reuse win the cards
now print (11.6x at 32,842 prompt tokens, 29.3x at 131,146) and the 266-request zero-corruption
evidence are ApcPoisonRepro's and are cited, not restated; every card also says plainly that
**LMCache is unmeasured by us** and is the outstanding suspect in the one user report we have.
Two corrections landed in the same pass: the KV cost model on the K5K6 and hydrated cards is now
the measured affine law (34,816 B/token with MTP-3, 32,932 without, plus a **per-request** term
of 0.63 / 0.14 GiB that concurrency pays again per slot) rather than the 37.4 KB/token ratio that
folded the fixed term in and overstated the coefficient by 8 %; and every driver string was
audited against the receipt beside it and found already correct (610.57.04 for the physical 5090,
595.58.03 only where the rental RTX PRO 6000 is named). Local and published are byte-identical,
proven by re-fetch with `DOCS-SHA256SUMS` refreshed in the same commit: 92,326 / 90,916 / 78,707 /
85,741 B (`receipts/apc-card-publication.json`). The user's live `qwen38-27b` service was
restored and proven: systemd active, podman healthy, `/health` 200, `Qwen3.8-27B` served, GPU back
to 30,449 MiB, all twelve compared `podman inspect` fields identical to
`receipts/aiboss-live-service-snapshot.json`.

## 2026-08-16: a sibling rebuild scores the same as the published hydrated checkpoint

`receipts/converter-determinism.json` closed the byte question and deliberately left the
measurement one open: a rebuild of the published hydrated recipe reproduces the recipe exactly
and the weights not at all, so anyone who rebuilds gets a *sibling*, and **nobody had ever scored
one**. That is now measured, in `receipts/sibling-rebuild-fidelity.json`, and the answer is that
**the recipe determines fidelity to within our resolution**: paired over the 512 contexts of v5
shard 0, published minus sibling is **-3.755e-06 with a 95 % interval of [-2.854e-05,
+2.062e-05]**, which brackets zero. The sibling's own shard-0 mean is **0.002703638192873069**
against the published **0.002699883159684943**, and the per-context split is **257 to 255 with
zero ties** — a coin flip, with no direction to it. Both tools agree to every digit: report-level
`fidelity.py paired` and receipt-level `kld_aggregate.py paired` return the same difference,
interval and win counts.

**The siblinghood was verified, not assumed, and the check found a sibling rather than a twin.**
All ten per-role byte totals match (21,586,964,548 B of payload), as do the module composition and
formats, the declared widths, 2,426 physical and 1,199 logical tensors, all 715 `tensor_storage`
entries, and per shard the entire safetensors header — names, dtypes, shapes, data offsets — and
the shard sizes to the byte (8,351,643,585 / 8,442,742,474 / 4,792,879,451). 13 of the 16 pinned
files are byte-identical, `quantization_manifest.json` and `quantization_config.json` among them.
The three shards are not: of 2,426 tensors, **399 differ and every one is a `.trellis` payload**,
zero non-trellis tensors moved, and inside a differing payload 39.4-91.8 % of the bytes differ
(mean 82.3 %). That independently reproduces the earlier rebuild's 399-of-409 result on a third
distinct checkpoint, and it means a byte-identical rebuild — which would have been the bigger
finding — is not what happened.

**Two controls make the zero credible rather than merely convenient.** The published ladder
deleted its own BF16 capture, so this run replayed against the surviving `/work/gguf/hidden-bf16`;
`/work/kld6/reports/report-hyd-rematch.json` scored the *published* checkpoint against that same
capture and returned the published mean and **every one of its 512 per-context rows bitwise**, so
the reference swap is a measured no-op and not an assumption. And because recapture reproduces the
published number exactly (`receipts/capture-determinism.json`), this comparison's floor is exactly
zero rather than a tolerance — the -3.755e-06 is attributable to the sibling's weights and to
nothing in the capture or replay path.

**Scale, because "distinguishable" and "matters" are different questions.** At this same
resolution the tightest distinction we publish is context-vs-error-driven at 3.432e-04; the
sibling difference is **1.09 % of it**, 0.85 % of hydrated-vs-K5K6 and 0.05 % of K5K6-vs-K4. So no
published comparison is threatened, and no published interval needs a second variance term. What
this does *not* license is a distribution: it is one sibling, so it bounds converter fidelity
variance at this resolution without estimating it, and the reconstruction claim is now a **tested
expectation** rather than an identity — the bytes still differ.

The capture is preserved and deep-verified at
`malaiwah/qwen38-27b-fidelity-suite-v5:captures/shard-0000/hidden-hyd-sibling` (513 files,
10,732,324,117 B, revision `1afb136276a0efa94b68500e4878c0f1253b909f`), because recomputing it
needs the sibling checkpoint and a re-conversion yields a *different* sibling — its recompute cost
is not 40 minutes but infinite, and 40 minutes buys another draw. The 21.6 GB checkpoint is
**not** published: an artifact indistinguishable from the published one is 21.6 GB that teaches a
downloader nothing, and its evidential content — all 16 file digests and all 399 differing tensor
names with their byte fractions — is in the receipt. Two latent parse defects were fixed in
passing: `tools/kld_aggregate.py`'s `paired_window` refused to pair a receipt built from legacy
`/1` shards against a current one because the window block carries a null `positions_per_context`
(the published 10M paired receipt still reproduces exactly, and a windowed-vs-full pairing is
still refused), and `tools/preserve_artifacts.sh` parsed only the machine-readable shape of
`hf auth whoami` and `hf upload`, so it reported "not logged in" against an authenticated cache,
and discarded a *successful* 10.7 GB upload as unverified, whenever it ran outside an agent shell.

## 2026-08-16: the method audit's two P0s are closed — the corpus text is published, the resolution floor sits beside every v5 table

The adversarial method audit (`receipts/kld-method-reproducibility-audit.json`, landed this
session with `docs/42-kld-method.md` and `receipts/near-duplicate-v5.json` after verifying each
against the audit's own digest pins) found two P0 defects in the published claims. Both are now
closed. **G1**: 862 of the 941 rows in `corpus/corpus_fetch_log.json` carry `url: "preexisting"`,
so the dataset card's "refetch the corpus from this" sentence and its tier-3 "refetchable" ground
were false — the 69 MB corpus text itself is now published in the v5 dataset under `corpus/text/`
(941 documents verified bit-for-bit against the fetch log before upload and re-verified after by
a fresh unauthenticated download; per-document sha256 and per-stratum licence in
`corpus/text/manifest.json`), the card corrected, the corpus-note receipt amended additively, and
no stratum needed the shingle-digest fallback. **G2**: the ~6e-4 replay-vs-live floor and the
~5 % storage systematic from docs/24 now appear as one resolution paragraph immediately after the
v5 cumulative table on all four model cards and in docs/33 and docs/35 — absolute values are
within-suite, paired differences are the resolvable quantity because the floor is common-mode,
the floor's six-v3-context derivation is disclosed, and its v5 re-derivation is tracked open.
Also in this pass: the audit's G9 remainder (docs/35 D10 now cites docs/42), supersession banners
on docs/03/05/09/12/14/17/20 plus the docs/16 stale closing line and the docs/02 MTP drift, and
24 orphan-receipt links (startup-times.json now sits beside the K5K6 cold-start numbers). All six
repos' cards re-published byte-identical. Every edit maps to a gap id in
`receipts/method-hardening-pass.json`.

## gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090, measured on the identical protocol (GittensorCompare)

The checkpoint that attacks the 32 GB / 262,144-token headline by name — with a 20-item smoke as
its only fidelity evidence — is now on the suite and the cards. Identity first: revision
`69274a0d` pinned, every file hashed against the Hub's own LFS metadata, headers read directly
(400 body Linears NVFP4 W4A4 group 16, MTP 15 tensors and vision 333 tensors intact BF16,
lm_head/embed BF16, FP8 KV baked into config; its trailing BF16 shard `9ce944d5…` is bit-identical
to three other ModelOpt exports, but both quantized body shards are unique — not a re-upload, so
the measurement was necessary). Mirrored **before** any published number cited it
(`malaiwah/Qwen3.8-27B-NVFP4-RTX5090-archival-69274a0d@e85bcc97`, deep-verified, ~zero transfer via
Xet dedup). Measured on v5 shard 0 — same suite token `caef8a46…`, same 512 contexts and 1,048,064
positions, same BF16 reference capture (byte-identical to the one the unsloth row replayed
against), same shared head `25a30fd5…`, same pinned r34 engine, bootstrap 10,000/seed 1/330
clusters: **mean KLD 0.062163** [0.058491, 0.066360], top-1 **89.85 %**, p99.9 2.5911 — the
weakest row on the comparator, 2.1x unsloth's NVFP4 in the same weight format. Paired per context:
loses **512/512** to the context edition (+0.058754), **512/512** to official FP8 (+0.056966),
**512/512** to hydrated, online and K4, and **511/512 to unsloth's NVFP4** (+0.032048
[+0.030711, +0.033583]; its one win is context 58 by 0.00023). Resident weights measured under the
docs/22 protocol: **18.77 GiB** vs unsloth's 21.34 GiB on the same rental GPU with identical flags
— their 18.8 GB claim is consistent, and the receipt keeps measured (KLD, resident bytes) strictly
separated from their-claim (5090 serving envelope, KV pool, tok/s), which we did not run. The one
resolved-runtime difference — its baked FP8 KV — was bounded with a bf16-KV control capture:
+0.000365 [+0.000058, +0.000679], ~1 % of the gap to unsloth; the gap is the weight conversion.
So the two NVFP4 rows now bound a real trade inside one format: 2.57 GiB of resident memory bought
at roughly double the KLD. Artifacts: `receipts/gittensor-nvfp4-rtx5090.json` (identity, digests,
composition, protocol identity, diagnostic), `receipts/kld5-1M-gt5090.json`,
`receipts/kld5-1M-paired-gt5090.json` (six comparisons), comparator amended additively
(`nvfp4-rtx5090` row, every pre-existing value verified byte-identical against git HEAD before
writing), master chart regenerated (ten candidates) and uploaded to all four model repos with
DOCS-SHA256SUMS rebound and re-fetch digest verification, all four cards carry the row plus one
honest paired paragraph each (and the dead collection link is fixed), shard-0 capture preserved to
the v5 dataset (`captures/shard-0000/hidden-gt5090@ea1244fc`, deep-verified). Ladder map extension
(`gt5090`, `--quantization modelopt_fp4`) follows the nvfp4 pattern and is explicit-only.

## 2026-08-16: the KV-dtype lever is measured — fp8 is confirmed, and 4-bit is a small-class lever, not a free one

[docs/29](docs/29-plan-and-loose-ends.md) §F5 had carried this since the release megathread: the
pinned r34 build advertises seventeen `--kv-cache-dtype` values, **every number we publish used
`fp8`**, and no 4-bit scheme had ever been measured here for capacity, retrieval or fidelity. A
reader asked directly whether quantized KV destroys long-context quality. Both are now answered on
the physical RTX 5090, on the promoted four-module release image with no bind mounts, in 43 vLLM
launches over a 4 h 15 min window: [`docs/38`](docs/38-kv-dtype-sweep.md),
[`receipts/kv-dtype-sweep-5090.json`](receipts/kv-dtype-sweep-5090.json).

**The baseline reproduces, which is what licenses the rest.** The `fp8` arm was re-measured, not
cited, and printed 29.98 GiB budget, 18.19 / 1.78 / 0.27 / 0.45 usage, a 9.28 GiB pool, **265,122**
KV tokens, 1.01x at 262,144, block 1600 and 107.36 tok/s median decode — every figure identical to
the qualification's gate 1, which ran on the three-module image.

**Eight of seventeen dtypes start; nine refuse, with verbatim errors.** Two refusals retire plan
items. All four **TurboQuant** presets load the model and then die in KV cache creation with
`Unknown TurboQuant cache dtype: 'auto'` — the presets are not mapped into `KVQuantMode`, so the
spec reports `NONE`, the literal string `auto` reaches `TurboQuantConfig.from_cache_dtype`, and it
raises; §F5's `turboquant_4bit_nc` arm cannot exist in this image. **`nvfp4`** (added mid-run at
Main's request, on the owner's unverified report that this fork serves GLM-5.2 with nvfp4 KV)
fails in backend selection with all five candidates reporting `kv_cache_dtype not supported` — not
`head_size` 256, not sm120, not the GDN hybrid — and an off-GPU replay of the build's own
`get_valid_backends` finds no backend in any shape probed, MLA included.

**Fidelity is the part that decided it, and retrieval would have lied.** Graded against a
**`bfloat16` KV cache** at a 98,304-token context on byte-identical greedy prompts, `fp8` holds
0.9560 top-1 agreement and 0.001655 nats of truncated top-20 KL — so quantized KV does *not*
destroy long-context quality and the shipped default is not the weak link. `int4_per_token_head`
retrieved **10/10** needles exactly while carrying **3.6x** that error (0.005948). Forty-four
needle retrievals across five arms, forty-four exact: a needle table alone would have called 4-bit
free. Every arm's repeat run was byte-identical with a maximum logprob delta of exactly 0.0, so the
harness has no run-to-run noise. These are **not** v5 KLD numbers and may never be differenced
against them.

**Two arms dominate `fp8` on capacity and fidelity at once, and lose on prefill.**
`int8_per_token_head` and `fp8_per_token_head` both allocate **272,453** tokens against 265,122 and
both sit closer to the bfloat16 reference than `fp8` does — and both pay **3.0x prefill**, 544 s
against 180 s on the same 261,795-token prompt, because TRITON_ATTN is the only backend that
accepts them. Their capacity edge is not even a KV-byte effect: they cost *more* per token, and the
0.39 GiB comes from a 0.06 GiB CUDA-graph pool against `fp8`'s 0.45.

**The affine law is re-derived per dtype from twenty startup refusals**, and using the engine's own
block quantisation pins `a` to an integer count of charged layers — **17 with MTP-3, 16 with MTP
off**, for every dtype. `fp8` MTP-3 comes out at **34,816 B/token and 0.63 GiB**, digit for digit
the published law, from independent refusals on a different image. `M` turns out to be essentially
**dtype-independent** (0.619–0.631 GiB with MTP-3) across a 3.9x range of per-token cost, and
4-bit is 51.6 % of fp8 rather than 50 % because the dynamic scales add 32 B/token/layer back. One
sharpening is offered and **not** applied: the exact form makes MTP-off `a` exactly 32,768 rather
than the published 32,932.

**Class consequence, memory form of the ≥15 % rule, controlled by reproducing 24,576 and 28,672
exactly from the published law.** Exactly one arm moves them: `int4_per_token_head` takes 24 GB
MTP-3 to **53,248** and 16 GB MTP-off to **57,344**, roughly doubling both. The two 8-bit
per-token-head schemes change neither. And the **withdrawn 16 GB MTP-3 row stays withdrawn for
every dtype** — its pool allows 0.4809 GiB and the smallest fixed term any dtype achieves is
0.6188, so a free KV cache would not start it. Every class window here is a prediction; none was
started, and the pool itself moved with dtype on the 5090.

**Verdict: `fp8` is confirmed as the right default, and the alternatives are now measured rather
than assumed.** What would change it is one build: if a FLASHINFER build accepted per-token-head
scales and the 3.0x prefill went away, `int8_per_token_head` would be strictly better on this card.
The owner's `qwen38-27b` was restored and proven before hand-over — systemd active, podman healthy,
`/health` 200, `Qwen3.8-27B` at 262,144, an 8-token greedy completion on the expected fingerprint,
and all fifteen snapshot-enumerated `podman inspect` fields identical to
`receipts/aiboss-live-service-snapshot.json`. It was then stopped at 13:04:44Z by `V2RunnerDepth`
taking the next 5090 window, confirmed directly; the final restore belongs to `LMCacheTest`.

## 2026-08-16 — KV-dtype sweep landed on the cards and docs (KvOnCards)

The kv-dtype-sweep-5090 verdicts are now where readers act on them. Each of the four model cards
gained a `### KV-cache dtype` subsection in its serving area — fp8 as the measured default, the
per-token-head family with its 3.0x TRITON_ATTN prefill price, int4_per_token_head as the
capacity lever with its two named prices, the nvfp4 refusal, the bf16-KV-probe (not-v5-KLD)
framing, and 44/44 retrieval — plus, per Main's scope addition, the V2-runner dynamic-depth
unavailability sentence beside each static-depth recommendation
(receipts/v2-runner-depth-schedule.json). docs/34 s10.2 carries the per-dtype exact-form MTP-3
coefficients with fp8 named as the digit-for-digit control and the [P]-labelled class
consequences (int4_pth alone moves 24,576 -> 53,248 and 28,672 -> 57,344; 16 GB MTP-3 stays
withdrawn at every dtype); s4.1's a/2 guess is marked measured at 17,952 B/token. docs/28's
"4-bit KV is not an escape" sentence carries a dated superseding note: an escape with two named
prices, not a free one. Every restated number is mapped to its receipt field in
receipts/kv-on-cards.json; publication byte-proof in receipts/kv-on-cards-publication.json.

## 2026-08-16 — Scratch arena landed on the cards and docs (ArenaLanding2)

The rank-1 lever from docs/43 is now published where readers act on it, and it is published as an
opt-in overlay rather than as part of any qualified digest. Each of the four model cards gained a
`### Reconstruct-scratch arena` subsection directly after its `### KV-cache dtype` subsection: the
2-hunk `exl3.py` patch (overlay `tools/vllm-exl3-scratch-arena.py`, sha256 `9aba06eb...`, fork PR
local-inference-lab/vllm#397) replaces 790 MiB of per-geometry persistent fp16 reconstruct scratch
with one 170 MiB grow-to-max arena per device, and the physical-5090 A/B moved the engine-reported
KV pool 265,122 -> 282,996 tokens (+17,874, +6.7 %, 9.28 -> 9.88 GiB), reproduced across two
server starts per arm, with the 30-case deterministic vision suite byte-identical on both arms,
the 258,925-token needle retrieved exactly and decode unregressed
(receipts/scratch-arena.json). Two things are said on every card so the win cannot be over-read:
it is an overlay deliberately not part of the qualified digest, in the same words the #51812 GDN
overlay got, and the byte-identity claim covers the deterministic probe set only — the control
shows two restarts of the *unpatched* baseline differ on 7 of 8 long greedy continuations, so this
stack is not restart-deterministic on long greedy text with or without the patch. The static
prediction (+620 MiB / +18.7k tokens) is quoted only as the prediction; the measured 95.7 % of it
is the number. docs/43 s3 keeps its original [INFERENCE] row in place and adds the measurement
beside it — the inference figure is now labelled as this path's calibration datum — s7 ranks R1
built-measured-and-PR'd, and the document's "nothing here was run on a GPU" banner is amended
rather than quietly falsified. docs/34 s10.2 carries the 24 GB-class effect as [P] arithmetic
only: 24,576 + 17,874 = 42,450 raw headroom, 40,960 at the next 4,096 step (3.6 % headroom),
36,864 as the step that would clear s5.3's >=15 % envelope, with s8's physical-board gate restated
as open and no 24 GB board booted. Landing receipt receipts/arena-landing.json; publication
byte-proof receipts/arena-card-publication.json (six repos, all byte-identical). CPU only: no GPU
was taken and no service was touched.

## 2026-08-16 — Upstream PR filed for the admission livelock (UpstreamPr52520b)

`vllm-project/vllm#52520` — the hybrid-Mamba align stall the fork trace exposed — now has a fix
proposed upstream: **https://github.com/vllm-project/vllm/pull/52530**, one commit, +381/-2, base
`main`, head `malaiwah/vllm-voipmonitor:fix/kv-pool-unservable-request-admission`. The gate that
made the PR legitimate was met first: the defect **reproduces on unmodified upstream
`4d2a68d64d9e05921ed5c4099146e768a92d71d5`**, CPU only, no GPU and no model weights
(`upstream/repro-52520-stock-main.py/.out`). Startup sizes the pool from each spec's
`max_memory_usage_bytes` and compares against *all* blocks, but `BlockPool` permanently reserves
one as the null block, so a pool built at exactly the startup minimum — 69 blocks for
`max_model_len=1024`, `block_size=16`, mamba-align with 3 speculative blocks — is one usable block
short of one `max_model_len` request, and runtime admission asks for one block *less* than the
pool is short by. Measured: the 1023-token request is admitted, prefills to 1008 tokens
(`(69-1-5)*16`, the length the pool can actually serve), self-preempts, and is refused on every
step thereafter with zero output; at 70 blocks the identical request runs. The fix follows merged
**#40946**'s shape — one bound with two call sites: `_estimate_max_model_len_from_groups`, already
the helper behind the startup message, is now also called by a new
`max_servable_num_tokens(vllm_config, groups, num_blocks)` evaluated against `num_blocks - 1`, and
the scheduler fails an over-long request at admission (`FINISHED_IGNORED`) instead of re-prefilling
it forever. Correctly-sized pools are untouched: the bound is then `>= max_model_len` and the check
short-circuits. CPU-only regression file `tests/v1/core/test_kv_pool_unservable_requests.py`
**fails 3-of-5 on stock main and passes 5-of-5 on the branch**, with the two over-rejection guards
green on both; `tests/v1/core/` is 2 failed / 500 passed / 2 errors on the branch against 2 / 495 /
2 on stock main, the same four GPU-requiring cases. DCO signed, ruff 0.14.0 + ruff-format + typos +
mypy 1.20.2 all clean. The PR states its relationship to open **#47272** explicitly and asks not to
be merged instead of it: #47272 owns the capacity side (refuse the pool at boot), this PR owns the
request side (never admit what can never be scheduled), and it remains the backstop for
`_auto_fit_max_model_len`, which runs before #47272's reservation. Two things are deliberately
**not** claimed: the reporter's zero-preemption-counter observation does **not** reproduce on main
(the self-preemption path does increment `vllm:num_preemptions_total`), so no accounting fix is
filed; and the failure is `finish_reason="length"` rather than a literal HTTP 400, offered as a
follow-up rather than smuggled in. Receipt `receipts/upstream-pr-52520.json`. CPU only: no GPU was
taken and no service was touched.

**Terminal-Bench 2.1, CPU-side complete and pinned before the GPU window** — the marquee item's
infrastructure is finished, published and pushed; only the three passes themselves now wait on the
rental card. The container-runtime question is closed with evidence: the rental box **cannot run
containers at all** (it is itself a container whose seccomp profile denies
`clone(CLONE_NEWUSER|CLONE_NEWNS)` — `unshare -U` and `-m` both return `EPERM`, no docker/podman, no
systemd), so under the owner's route **(c)** the model serves on the rental while all 89 task
containers run on AIBoss under **rootless podman 4.9.3**, reached by a static user-local **docker
CLI 27.5.1 + compose 2.39.1** with `DOCKER_HOST` on the podman socket — which works unmodified
because harbor shells out to the `docker` CLI rather than using the Python SDK. Proven end-to-end
before spending any GPU time: an **oracle trial scored 1.0** through that stack, published at
`harness-validation/oracle-smoke`.
Two preflight findings changed the design. First, the tunnel's round-trip is **~581 ms per request**
(AIBoss→tunnel→rental 0.579–0.587 s TTFB vs 0.00072–0.00088 s on the rental's own loopback) and it
is **not** connection setup — with five requests on one reused connection `time_connect` was
0.0001 s while `time_starttransfer` stayed ~0.582 s, so LiteLLM's pooling cannot amortise it. The
cause is distance: the rental's TCP RTT to the jump host is **266 ms** median against AIBoss's
**12 ms**, and the rental has no direct route to AIBoss at all. It is a time-to-first-token cost,
not per-token; it is **common-mode across both arms** so it cannot corrupt the
capability-vs-quantisation split; and tok/s is therefore read from vLLM's own `/metrics` on the
rental loopback rather than from agent-side wall clock. Second, `ssh -R <port>` is **unsafe here**:
after an unclean client death sshd keeps the listener, which accepts and instantly resets every
connection (`curl rc=56`), and it belongs to sshd's **root** privsep process — `fuser -k` and
`lsof -iTCP` see nothing, so an unprivileged agent cannot reclaim it, and the port cannot be changed
either because harbor bakes `api_base` into the job config and resume requires an identical one.
Fixed by taking port ownership away from sshd: ssh binds a **unix socket** with
`StreamLocalBindUnlink=yes` and `tools/tb_tunnel_proxy.py` — our own process — owns
`127.0.0.1:18010`. Verified with `SIGKILL` rather than a polite shutdown: `attempt 1 → dropped
rc=137 → attempt 2`, same port serving `http=200 ttfb=0.581403` ~14 s later, proxy alive across the
drop; on deliberate stop it released the port at once while the old 18000 stayed held by root sshd.
Resume is delegated to harbor and **verified by source read** rather than assumed: a trial dir
without `result.json` is deleted and re-run (`job.py:258-260`), a completed trial is preserved and
skipped (`_init_remaining_trial_configs`, `job.py:327-357`), and the job config is identity-checked
(`job.py:246`) — which is why resume must go through `harbor job resume`. In-run retries are
**transport-only** so a tunnel drop can never score as a task failure, while `AgentTimeoutError` is
deliberately excluded because running out of task time is a real TB failure mode. CPU-only is
**asserted, not assumed**, by four independent checks (89/89 tasks pin `gpus = 0`; harbor's docker
environment contains no `DeviceRequests`/`--gpus`/`nvidia` code path at all; live TB containers are
inspected for devices; `nvidia-smi` process lists on both hosts) — and the two co-tenant containers
that *do* hold `/dev/nvidia*`, the owner's `qwen38-27b` service and an `nvidia-gpu-exporter`, are
named in the evidence rather than filtered out of it. Concurrency will come from a measurement, not
a guess: the 89 tasks' declared agent timeouts sum to **149,160 s = 41.4 h** serially, so a pass
must be concurrent, and since 83 of 89 tasks ask for a single core against AIBoss's 28, the binding
constraint is the single vLLM server — `headroom` sweeps 1/2/4/8/16 and records it.
Pins frozen in `receipts/terminal-bench-2.1-pins.json` + `-task-inventory.json`: harbor 0.21.0,
`terminal-bench-2-1@6` run from a **local tree pinned by sha256** so a registry change cannot alter
a pass mid-protocol, terminus-2 **2.0.0** with the explicit `model_info` that `hosted_vllm/` names
require, vLLM r34, 32k window + fp8 KV + MTP-3, and `generation_config` **byte-identical** across
the hydrated and BF16 checkpoints — which is what makes pass 3 like-for-like. `docs/45-terminal-bench.md`
frames every number as an **agent+model system** score, never model-only. Dataset
`malaiwah/qwen38-27b-terminal-bench-2.1` is live and **verified public unauthenticated**, and passes
publish as they complete rather than at the end. No GPU taken by this agent; the card is requested
by name from SixteenFlipCalib2.

## 2026-08-16 — Four rental measurements: head attribution, the v5 replay floor, the int8 embed overlay at 8k, and the live tool-call path (RentalFidelityBatch2)

One rental window, four receipts, each pushed as it landed. The batch resumed a predecessor killed
mid-flight: its shard-0 recaptures were complete for context, K4, FP8 and unsloth NVFP4 but the
hydrated capture had died at context 472 on `No space left on device`, and `fidelity.py capture`
verified the sha256 and shape of all 473 surviving files before finishing the last 39 in 19 s.

**Head attribution is now measured on our own corpus, not reasoned**
([`receipts/head-attribution-v5.json`](receipts/head-attribution-v5.json)). Each candidate's shard-0
capture was replayed twice against the published BF16 teacher capture — body-only, with the shared
BF16 head on both operands, and head-inclusive, with the candidate operand going through *that
checkpoint's own* serialized head — the bodies byte-identical between the arms. The three EXL3 heads
were reconstructed with exllamav3's own `reconstruct_had_slice` (`tools/dequant_head.py`, the docs/16
method; hydrated and context carry `mcg`, K4 carries `mul1`, all K=6, all finite at
`absmax = 0.34180`), the unsloth head from its FP8 values times its per-row BF16 scales, and the FP8
export's head straight out of the checkpoint. The control is exact: **all five body-only means
reproduce the published shard-0 numbers bitwise** (hydrated 0.002699883159684943, context
0.003409409957186559, official FP8 0.0051970635503104596, K4 0.01034534853668276, unsloth NVFP4
0.03011540756091421). Head-attributable delta, paired per context with a 10,000-resample
source-cluster bootstrap: hydrated **+1.425e-04** [1.354e-04, 1.496e-04], context **+1.444e-04**,
K4 **+1.225e-04**, unsloth NVFP4 **+8.161e-04**, and official FP8 **exactly 0** — not as a rounding
result but because that export's `lm_head.weight` is byte-identical to the shared head
(`tensor_sha256 d922b751f014...` on both), which is the internal control that turns the other four
into measurements. Every interval excludes zero; the candidate's own head is worse in 486-505 of 512
contexts. That is **1.17 %-5.01 % of head-inclusive divergence**, so docs/35's `[INFERENCE]` on D5
is resolved by measurement: the head is real and signed as stated, and far too small to be the
dominant term in a 1.1-1.6x level gap — our published ordering survives the objection instead of
depending on it.

**The replay-versus-live floor is re-derived on the suite the cards actually use**
([`receipts/replay-live-floor-v5.json`](receipts/replay-live-floor-v5.json)). The published 6.54e-04
came from six v3 contexts with no suite, head or capture digest recorded. Serving the unquantized
BF16 model on the rental and scoring its own full-vocabulary prompt logprobs against the same
contexts replayed from the published shard-0 capture, through the same shared head, gives
`KL(live ‖ replayed)` = **5.83e-04** over 32 analysis contexts and 65,504 scored positions,
context-bootstrap 95 % CI [5.15e-04, 6.64e-04], top-1 99.10 %, worst single position 0.2534. The
interval **contains** the old number, so the level has not moved materially and the cards' standing
rule is unchanged; what improves is provenance. `fidelity.py qualify` refuses to run unless the live
model's index/config/shard digests equal the capture's and the capture's recorded runtime equals the
requested one, so a mismatched pair cannot produce a number at all.

**The int8 embedding overlay is an opt-in lever on the non-context recipes, not a free switch**
([`receipts/embed-overlay-8k.json`](receipts/embed-overlay-8k.json)). Four served arms, each the
published 8,192-token card recipe with exactly one variable added (`VLLM_EXL3_EMBED_BITS=8`), each
answering the same 22 frozen probe requests greedily at concurrency 1. The engine logs the overlay
itself — `248320 x 5120 rows narrowed, 2.543 GB -> 1.272 GB` — and the accounting follows: hydrated
weights 20.31 → 19.14 GiB, available KV 66.45 → 67.62 GiB = **668,852 → 680,899 KV tokens
(+12,047, +1.80 %)**, concurrency at 8,192 81.65x → 83.12x; online K5-K6 weights 20.56 → 19.38 GiB,
**345,324 → 352,571 tokens (+7,247, +2.10 %)**, 42.15x → 43.04x. Correctness is untouched: 22/22
planted answers correct with zero corruption detectors on all four arms. Output identity is **not**:
13 of 22 (hydrated) and 10 of 22 (online K5-K6) greedy continuations diverge from the BF16-table arm
tens of tokens in, chosen-token logprob deltas up to 1.71. That is the expected consequence of the
+0.000065 mean KLD docs/32 already published rather than a new defect, but it is why the verdict is
opt-in with the output change stated plainly, on a profile whose KV is already 42x-82x the window.

**The template's `# Tools` block is finally exercised against a live server**
([`receipts/tool-calls-e2e.json`](receipts/tool-calls-e2e.json), driver `tools/tool_calls_e2e.py`).
Eight cases through `/v1/chat/completions` on the unmodified hydrated recipe, greedy, real `tools`
definitions: a single `get_weather(city=Paris, unit=c)` call with `finish_reason=tool_calls`; a
multi-turn round where the tool's 21C reaches the answer; whitespace-bearing arguments where
`'    if x:\n        return 1\n'` and its replacement survive render → generate → parse
**byte-for-byte**; **two genuinely parallel calls in one turn**, Paris and Lyon both intact; and the
docs/39 fixtures c17, c31 and c32 replayed as assistant history without mangling. **Zero structural
P0s.** One case is deliberately not a 200: docs/39's c16 shape, `function.arguments` as a JSON
object, violates the OpenAI request schema and is refused with a 400 validation error naming that
field. The harness now classifies that as conformance rather than as a defect — a clean refusal is
correct, while a 5xx, an unexplained refusal or acceptance-with-loss stays a P0 — and two passes on
two separate serves of the same build agree on every status and, modulo the server's random per-call
`tool_call` ids, on every byte of every response.

## 2026-08-16 — kld9 window: the 16 GB flip condition and K6-parity (SixteenFlipCalib2)

Two of three pre-registered conversions ran on the rental RTX PRO 6000; the third was handed on.
Everything was pre-registered before converting in `receipts/preregistration-kld9-window.json`,
which grew five pre-conversion addenda as the prep turned up problems.

- **S16-V, the first sub-4-bit width ever measured in this family: 0.045374 [0.041959, 0.049351].**
  docs/34 flip item (1) is closed and the answer is **NO**. Loses 512/512 to K4 (4.39x) and 512/512
  to the context edition (13.31x), every stratum with intervals excluding zero; p99.9 2.3704, max
  12.5031, top-1 91.73 %. The pre-registered range [0.03, 0.10] and the independent surrogate
  bracket [0.0318, 0.0585] both contained it; the registered point estimate 0.0689 was 1.52x too
  high. Receipt `receipts/sixteen-flip-kld.json`.
- **K6-parity: 0.001634 [0.001541, 0.001742] — MATCHES GGUF `Q6_K` at equal file bytes.** Between
  Q6_K net 0.001528 and measured 0.002035; better than hydrated by 0.001066 (511/512) and better
  than Q6_K measured by 0.000401 (493/512). docs/29 predicted 0.0016 in advance — within 2.1 %.
  The registered interval [0.001175, 0.001601] MISSED, 2.0 % low. Receipt
  `receipts/k6-parity-kld.json`, card `MODEL_CARD-K6-parity.md`.
- **The byte law is exact below 4 bits and at parity**: both payloads predicted to the byte
  (13,711,503,428 and 23,035,310,148, zero error).
- **The 3.73x-per-bit law holds one rung below its fit**: realised K3 proxy errors match the
  extrapolation at median 1.0164 for the 96 modules with no measured rung.
- **The 16 GB class was never lost on bytes.** With the first loader term ever measured for a
  sub-4-bit build (+0.0402 GiB) the resident figure is 11.626 GiB against budgets of 12.70 and
  12.49. It is lost on measured fidelity instead.
- **Four axis errors found in the published cross-engine comparison** and corrected in
  `receipts/cross-candidate-byte-accounting.json`: GGUF whole-file against our tensor payload,
  text-only against multimodal (the Q6_K GGUF has no vision tensors — mmproj ships separately),
  non-uniform embedding and head widths per tier, and body against body. Two move against us and
  two for us. `receipts/byte-law-recipe-audit.json` re-derived 25 published byte rows; none moved.
- Open, no GPU needed: upload both checkpoints (37 GB, digested in the receipts) and write the
  S16-V research-artifact card. Alt-calibration handed to `ShortlistScore` with
  `local://altcal-handover.md`.

## 2026-08-16 — The decode case on PR #52530, measured rather than argued (`Pr52530Decode`)

`brianosaurus` asked on the PR whether the decode case was deliberately out of scope: `_is_unservable`
bounds admission on `num_tokens + 1`, but a generate request grows to `prompt + max_tokens`, so a
request can clear both length checks and then need a block the pool does not have. Reproduced on
unmodified upstream `main` @ `4d2a68d6`, CPU only, on the gate-zero harness with the sequence split
across the prefill/decode boundary: `block_size=16`, `max_model_len=1024`, 69-block pool
(`max_servable_num_tokens=1008`), `prompt=1000, max_tokens=24`. **It livelocks.** The sequence reaches
1009 tokens at output token 9, is preempted there, is then too long to re-prefill, and the following
3987 scheduler steps schedule zero tokens, produce no output and leave all 68 blocks free. Two controls
in the same run finish normally (70 blocks: 24/24 in 28 steps; same pool with `max_tokens=8`: 8/8 in 12
steps), so it is the ceiling and nothing else. One correction to the expected signature:
`num_preemptions_total` increments **once** and then stops — the decode case collapses into the prefill
case rather than having a shape of its own.

Fixed at the point of exhaustion rather than at admission — `check_stop` now takes
`max_servable_num_tokens` instead of `max_model_len`, one line — because the reviewer's
`num_tokens + max_tokens` bound would refuse requests that stop before the wall, which the third row of
the control table literally is. Second commit `479413a` on the same branch; PR now 2 commits, +470/−3,
DCO green, `tests/v1/core/` 2 failed / 502 passed / 2 errors (the same four GPU-only cases).

Gap width, since he asked: **exactly one block, never more**, `((max_model_len - 1) mod block_size) + 1`
tokens, and 0 as soon as the pool has one spare block — measured over 14 configurations, largest gap 256
tokens at `block_size=256`. The speculative reserve is **not** a contributor: it is priced identically on
both sides, which retracts what our acknowledgement comment claimed. Reply posted at
`#issuecomment-5309046330`; receipt `receipts/upstream-pr-52520-decode-case.json`. No GPU at any point.

## 2026-08-16 late: the TB2.1 campaign image exists — vllm:gg-r34-tb21-sr1 (ImageForge)

Built on the driver VM from the promoted apc base (podman manifest `16a936b8…`, sourced by
`podman save | zstd` off AIBoss, 10.78 GB landing byte-exact at ~6.4 MB/s over the jump) with three
audited, fail-closed patch layers: PR #397 scratch-arena overlay, PR #398 FlashInfer decode-shape
keying, and a hand-port of upstream #52530 (admission gate + pool-bound length cap; one adaptation —
Request objects captured before the fork's tuple-returning, dict-freeing `finish_requests`). Every
layer asserts patch sha, post-patch sha, a sentinel symbol, and recompiles with the image's own
python; a wrong base or drifted file cannot build. LMCache: untouched, disabled by default, wheel
RECORD verified intact (LMCacheFix: not ready). Cross-runtime base identity proven by the diff_id
chain (identical on AIBoss/driver/endpoint, `886dbafc…`); the built image's manifest-list digest
`237a5025…` is identical on driver and endpoint after save/load. CPU smokes pass on the driver
(vllm --help SKIPPED there: CUDA-only build cannot infer a device GPU-less — base property) and
fully on the 1x endpoint under `--gpus all`. Canonical tar
`ubuntu@151.185.34.98:~/images/gg-r34-tb21-sr1.tar.zst` sha `1e5711f4…`; interim unpatched base also
loaded on the endpoint for gate rehearsal. Receipt `receipts/tb21-image-sr1.json`. No GPU used;
TB pass 2 untouched. Registry push (ghcr vs docker.io) awaits an owner token — Main's question.

## 2026-08-16 — LMCacheFix: root cause found, upstream PR #4600 filed, CPU gates green
The lmcache-reuse-test corruption is root-caused. Bounded arms (7/38+7/38): NOT wrong bytes from
LMCache — the GG vLLM scheduler's hybrid+connector path takes the full-attention group's cache hit
as computed for the whole model (`max(per_group_hits)`, scheduler.py:727-764 in the promoted image)
and delegates Mamba/GDN state restoration to the connector; only nixl fulfils that contract, so with
LMCacheMPConnector generation resumes on a stale/uninitialized recurrent state at the hit boundary.
All 76 bounded rows reconcile (hit = L0+1600 pattern, both failure signatures, 7/7 control, universal
logprob divergence); upstream vLLM fixed this class in PR #48425 (capability gate), our GG base lacks
it. L3restart 38/38 = poison propagation of bounded-arm stores + no local masking; p1s1-L3 zero-hit
failure stays a named blind spot. LMCache-side: the silent-corruption channel (failed retrieve ACKed
as loaded, #2865/#3388) is real upstream; filed https://github.com/LMCache/LMCache/pull/4600
(fail-closed error_block_ids propagation, DCO, fail-before 3F/pass-after 30P CPU tests; minimal
subset of stalled #2898, all prior art cited). Image wheel needs NO change — it already carries the
fork's equivalent guard (wheel == fork fix/lmcache-mp-retrieve-recovery-20260729). Campaign fix =
patches/gg-vllm-hybrid-divergent-hit-gate.patch (sha 5f9ad10b…, applies+compiles; routed to Main,
vLLM-side). GPU ladder re-run remains the gate for any LMCache default-config decision. No GPU used.
Receipt: receipts/lmcache-fix.json (+ receipts/lmcache-fix-raw/).
