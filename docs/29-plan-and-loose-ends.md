# What is left, ordered by evidence value

State on 2026-08-15 after four published builds, two independent reviews, one external
RTX 5090 test, a source-disjoint qualification, a contamination re-audit, and paired
downstream smoke testing. Priority is the claim each item can close, not implementation cost.

## Current measured frontier

KLD is `KL(BF16 || candidate)` through one shared BF16 head on the overlap-corrected
127-context v3 subset. Capacity profiles are deliberately not treated as interchangeable.

| profile | resident weights | corrected mean KLD | demonstrated/configured context | hardware |
|---|---:|---:|---:|---|
| hydrated K6 | 20.31 GiB | **0.007172** | 185,600, MTP-3 | physical RTX 5090 |
| online K6 | 20.32 GiB | 0.007945 | 185,600, MTP-3 | physical RTX 5090 |
| context + int8 input | **18.41 GiB** | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | 30.24 GiB engine budget on RTX PRO 6000 |
| K4 | 17.89 GiB | 0.029679 | **262,144, MTP-3** | physical RTX 5090 |
| official FP8 | 28.51 GiB | 0.012798 | not measured here | — |

The context profile now combines native context, speculative decode and multimodality:
266,612 KV tokens allocated; exact retrieval from 261,794 text tokens; and exact code plus
image answer from a 236,824-token prompt containing a 3,072 × 2,304 image. Its one-run
warmed C1 decode result is 98.72 tok/s. This is an engine-budget proof, not a physical
RTX 5090 proof.

The frozen post-selection ranking also survives the offset-independent correction. After
excluding every context from four qualification source documents with an exact calibration
12-gram, hydrated / online / context score 0.003093 / 0.003455 / 0.003990 against FP8
0.005891, each winning 36/36 paired contexts. Absolute KLD is suite-specific.

## Priorities added from public feedback, 2026-08-15

Four independent readers attacked the same weak point: the fidelity claim is strong but
thin, and the comparator set is narrow. These outrank everything below except the physical
RTX 5090 qualification.

### F1 — evidence volume: 278,392 scored positions is not enough

"Do I understand this correctly that you evaluated only on 136 traces and 278k output
tokens?" The headline KLD rests on 136 analysis contexts. Bootstrap intervals are computed
over 39 source clusters, so the interval is honest, but the absolute volume invites the
objection that the estimate is fragile.

Built for this: `tools/fetch_corpus_v5.py` (941 documents, 70.3 MB of held-out public text)
and a v5 suite of **5,120 contexts x 2,047 = 10,480,640 scored positions** over 842 source
clusters, token-disjoint from v4 and with zero calibration 12-gram overlap by construction
(whole documents with any hit are excluded before selection). `tools/suite3.py` now advances
each document cursor to the exact end of the emitted context, so corpus capacity is
token-bound rather than window-bound.

`tools/kld_ladder.sh` walks it in 512-context shards because one shard of six models is
~64 GB of hidden states and `/var/tmp` holds ~135 GB: capture six models, replay the five
candidates, verify, delete, next shard. `tools/kld_aggregate.py` welds the per-shard reports
into cumulative receipts at the **1M / 2M / 5M / 10M** checkpoints. Escalate until the run
stops being reasonable on this hardware, and publish the point at which it stopped.

### F2 — comparator breadth: FP8 is a throughput format, not the quality ceiling

"A comparison to Q8 instead of FP8 would've looked differently... someone would need to run
a KLD/top-1 test on the same dataset for all of them in the same size/performance range."
That is correct and currently unanswered. Required:

- GGUF `Q5_K_XL`, `Q6_K` and `Q8_0` in our suite, which means a second engine (llama.cpp)
  with the same tokenizer, same token IDs, same BF16 teacher and the same shared head;
- stock EXL3 uniform-bitrate controls, at least `turboderp/Qwen3.8-27B-exl3` 5.00bpw
  (`a35e75a7`) and 6.00bpw (`d32ba0bb`), to separate our role-aware allocation from EXL3
  itself;
- a cross-dataset run on the corpus other publishers use (OpenWebText-style, qbench's 8 x
  8,192-token protocol), because the same NVFP4 checkpoint reads 0.095 here and 0.016-0.068
  in Unsloth's own table. Whichever way that lands, both numbers get published.

Report confidence-conditioned buckets alongside the mean: a KLD below 0.01 is the threshold
readers already associate with "practically BF16", so the distribution shape matters as much
as the mean.

### F3 — VRAM-class SKUs: 24 GB, then 16 GB

Every profile we ship targets a 32 GB card. Two smaller classes are asked for and one of
them is explicitly framed as the hard one:

- **24 GB** (mid-point; the RTX PRO 4500/SFF-class 24 GB boards are not Blackwell-only, so
  the supported-hardware tuple has to be stated, not assumed);
- **16 GB** ("make a version for the 16GB of RAM or VRAM and you'll be a god"), which must be
  published with an honest KV budget: at 16 GB the usable context is a fraction of 262,144,
  and the card must say so in the same table as the fidelity number.

Bound already known from the external ladder: stock 3.50bpw is a 14.28 GiB tensor payload
with a 2.37 GiB BF16 embedding, so a 16 GB build needs either an int8/int4 input overlay or
CPU-resident embeddings, and its KV budget must be measured, not modelled.

### F4 — collection presentation

Adopted from the external ladder review: one immutable machine-readable collection index
(artifact commit, base commit, converter commit, whole-tree bytes, tensor-payload bytes,
measured resident bytes, per-role precision, runtime image and patch digests, suite digest),
the same compact ladder rendered in every card, and strict size terminology — serialized
bytes are never called VRAM.

### F5 — everything else the release megathread asked for

Harvested from 260 comments across the r/LocalLLaMA release megathread (`1voojjz`) and our
own KLD post (`1vp15wq`); the megathread and post are 403 to plain fetches, so the harvest of
record came from the Arctic Shift `comments/search` API, paginated. Items are grouped by what
they would change.

**Statistics.** "Avg KLD and top-1 are meaningless. What 99.9% tail looks like?" Publish
p99/p99.9/max per candidate, not only the mean. **Landed and measured.** `fidelity.py replay`
counts every scored position into a fixed log-spaced histogram — 560 bins from 1e-12 to 1e2
nats, plus explicit zero/underflow and overflow buckets — and stores it as `kld_tail` alongside
the shard's own exact p50/p95/p99/p999 and exact maximum, which bumps the report schema to
`qwen38-fidelity-report/2`. `tools/kld_aggregate.py` sums those counts across shards and
publishes cumulative p50/p95/p99/p999/p9999 as the bin interval that provably contains each one
(one bin, 5.6 % wide, wherever the tail is dense), plus the exact global maximum and exact
exceedance counts at 1e-4 … 1 nat. Counts recombine, percentiles do not — which is why the
histogram had to exist before any tail could be published at all.

**The tail is now on the record.** Shard 0 of the v5 suite was re-captured and re-replayed with
the `/2` harness — the same 512 contexts every candidate saw, **1,048,064 scored positions**
each — and welded into `receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`
(`qwen38-kld-ladder-cumulative/2`):

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | above 0.1 | above 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| hydrated | 0.002700 | 0.00109 | 0.0082 | 0.0276 | **0.1319** | 0.463 | 3.735 | **0.1534 %** | 0.00219 % |
| online K5/K6 | 0.003141 | 0.00128 | 0.0099 | 0.0321 | **0.1446** | 0.498 | 5.507 | **0.1820 %** | 0.00200 % |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | **0.1642** | 0.587 | 3.749 | **0.2287 %** | 0.00305 % |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | **0.2438** | 0.812 | 5.296 | **0.3912 %** | 0.00592 % |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | **0.5555** | 1.870 | 7.565 | **1.2604 %** | 0.03807 % |

The ordering at every measured quantile matches the ordering of the means, so for these
candidates the mean is not hiding a worse tail: every EXL3 K5/K6-class build has a lighter tail
than official FP8 at every quantile, and K4 is worse than FP8 at every quantile.

**Remaining gap, precisely.** The 10.48 M-position ladder was captured with the pre-histogram
harness and emits `/1` reports, which carry no histogram; its tail is published per shard with
the exact global maximum via `kld_aggregate.py --allow-legacy-no-tail`, and the receipt says in
`not_aggregable` why there is no cumulative percentile. A mixed `/1` + `/2` shard set is
rejected rather than summed into a tail that would silently describe only part of the run. Its
hidden states are deleted, so it cannot be retrofitted. **Cumulative histograms over all ten
shards therefore require re-running the other nine shards with the `/2` harness — about 6 hours
of GPU time, and not yet done.** Until that runs, the published tail is one shard of ten
(1,048,064 of 10,480,640 positions) with bin-bounded quantiles, and the 10 M receipts stay the
authority for full-run means, bootstrap intervals and paired results.

**Long context.** Every scored position today comes from a 2,048-token window. Two readers
asked for 64k/128k behaviour, and one reports measured instruction-following collapse at Q3
beyond 40k. Long-window KLD is a separate suite, not a rescale of this one.

**Degeneration.** Two independent Q4 reports describe looping until the context is exhausted.
Mean KLD cannot see it; the downstream suite needs an explicit repetition/non-termination
check.

**Sampler and effort pinning.** Measured in-thread: MTP acceptance moves with temperature and
with reasoning effort, and some front-ends silently downgrade the effort they were asked for.
Every fidelity and throughput figure we publish must carry temperature, top-p/top-k and
reasoning effort. `high` is not a valid effort on this model; only `low`, `medium`, `xhigh`.

**Serving answers we owe.** An unanswered question on our own thread asks whether the
Gilded Gnosis vLLM EXL3 path supports q4 KV cache. Read out of the pinned r34 build, the
accepted `--kv-cache-dtype` set is `auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2,
fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc,
turboquant_3bit_nc, int4_per_token_head, int8_per_token_head, fp8_per_token_head, nvfp4`, so
sub-8-bit KV exists in several flavours — but every number we publish used `fp8`, and none of
the 4-bit schemes has been measured here for KV capacity, retrieval or KLD. The honest card
answer is the flag list plus that gap, and the fix is a KV-dtype sweep (fp8 versus
`int4_per_token_head` versus `turboquant_4bit_nc`) reporting allocated KV tokens, needle
retrieval at native length and KLD delta on the same checkpoint. That sweep is also the
cheapest lever for the 16 GB and 24 GB classes, where KV is the binding constraint.
A second reader reverse-engineered the `-context` spec sheet
(fp8 KV, native 262,144, roughly 200-215k with MTP, LMCache-compatible) and asked for
confirmation: publish it as a table.

**Throughput next to fidelity.** The 5090 audience compares against NVFP4 and 5.5-bit vLLM
formats on tok/s. Each card needs single-request and concurrent throughput at its native
context with MTP on and off, measured on the same machine as its fidelity number.

**MTP acceptance regression.** Five independent reporters see 3.8 acceptance at roughly
60-70 % where 3.6 was 80-90 %, across llama.cpp and vLLM. Reproduce on our checkpoints at
pinned temperature and publish acceptance per SKU against the in-thread BF16 reference
(MTP-3 optimal, MTP-4 regressing).

**Prefill, not just decode.** A 196k-token agentic turn spends ~75 s in prefill; prefix-cache
reuse dominates that workload. Our LMCache path is real and unpublished.

**Per-tensor bit maps.** Readers now inspect how a quant spends its bit budget before
downloading. Mixed precision is our differentiator and it is currently only prose.

**Non-CUDA tiers.** MLX/Apple and ROCm requests recur; EXL3 serves neither. One explicit
unsupported line per card prevents the question repeating.

**Ship the client config.** Three readers hand-rolled an opencode/pi model block for this
model. Ship a correct one with the valid effort variants, 262,144 context and image modality.



## Closed by the audit

- The fixed-stride contamination claim was wrong. Every normalized 12-token position (Unicode
  word or Han/Kana character) is now scanned; v3 excludes two affected documents (nine
  analysis contexts) and v4 excludes four affected qualification documents (six contexts).
  A sliding five-token receipt measures
  lightly edited overlap. No ranking changes.
- Capture resume now fails closed on missing or modified files; replay refuses an incomplete
  selected context set. The frozen corpus builder fails on language shortfalls.
- The deterministic task harness now executes code in a bounded subprocess with a strict
  builtin allowlist and validates every hidden case. BF16, all four EXL3 profiles and official
  FP8 each pass **40/40 with zero regressions**. A strict final-answer rescore corrects exact
  BF16 agreement to hydrated 35/40, online K5/K6 34/40, FP8 34/40, K4 33/40 and context
  32/40; the original diagnostic compared extracted test summaries for code tasks.
- The draft-embedding int4 claim was invalid and is retracted. vLLM aliases the draft input
  table to the target after load; the earlier comparison never exercised int4 at inference.
- Native MTP was not fundamentally blocked by model weight. Halving the image profiler ceiling
  from 16,777,216 to 8,388,608 pixels lowers peak activation enough to fit MTP-3 and native
  262,144 inside the same 30.24 GiB engine budget while accepting a tested 7.08 MP image.
- PRs #312, #314, #316, #318 and #319 are open as of this state. Cards no longer imply that
  their patches exist in the pinned r34 image.

Receipts: `analysis-v3-contamination-corrected.json`,
`qualification-v4-contamination-corrected.json`, `near-duplicate-v3.json`,
`near-duplicate-v4.json`, `task-retention-v2-summary.json`,
`task-retention-v2-strict-rescore.json`, and `native-mtp-8mp-amendment.json`.

## Queue A — finish on this RTX PRO 6000 rental

Use paid time only for work that the 31.39 GiB RTX 5090 cannot perform, chiefly live BF16
Qwen3.8-27B controls. Preserve each control as a reusable reference so candidate-side runs
can move to the free local card.

| rental rank | global rank | investigation | why it belongs here | closeout before moving on |
|---:|---:|---|---|---|
| R1 | prerequisite | freeze and publish this session | `/var/tmp` models, rootfs, logs and caches are ephemeral | Git commit plus pushed receipts/cards/tools; no evidence only on rental disk |
| R2 | 4 | public capability BF16 baseline | the 55.6 GB BF16 model cannot reside on a 32 GB card | content-hashed prompts, raw BF16 outputs and scorer receipt |
| R3 | 5 | real multimodal BF16 baseline | full BF16 target plus large image activations need the 96 GB card | OCR/chart/document/video BF16 control with original media hashes |
| R4 | 6 | BF16 context-quality control | establish the longest same-model loss/reasoning curve BF16 can actually serve | 32k/64k/maximum-fit token-counted reports; never extrapolate to 262k |
| R5 | 13 | BF16 safety/control side | candidate regressions need a frozen base-model response, not a policy assumption | raw paired-set BF16 outputs and item labels |
| R6 | 9 | conversion error measurements | source BF16 plus converter workspaces are already present and proven here | immutable converter log, `args.json`, per-module adjacent-width errors |
| R7 | 10 | large-shape kernel prototypes | the 96 GB card can retain reconstructed weights and broad shape sweeps without displacing evidence | reproducible microbench plus end-to-end prefill/KLD gate |

Run R2–R5 from one immutable prompt/media bundle and one BF16 server identity where possible.
Do **not** spend rental time repeating quant-only 5090 tests. Before the rental ends, copy
every accepted or negative result into `receipts/`, update the corresponding section here and
`PROGRESS.md`, commit, and push `main` to
`https://github.com/malaiwah/qwen38-27b-exl3`.

## Queue B — free local RTX 5090 32 GB handoff

The local session starts here after pulling the pushed `main`; it must not infer state from
this rental's `/var/tmp` paths.

| local rank | global rank | investigation | first action | push target |
|---:|---:|---|---|---|
| L1 | 1 | physical RTX 5090 qualification | run the exact context/int8/MTP-3/8.4 MP profile and all seven gates below | receipt + raw logs to this repo; corrected context card to GitHub and its Hub README |
| L2 | 11 | image-cap/concurrency frontier | after L1, sweep 4.2/6.3/8.4/10.5 MP at sequence counts 1/2 | frontier receipt and card-supported ceiling |
| L3 | 7 | lifecycle/cache reliability | run cold/warm/corrupt/interrupted/restart cases against the L1 runtime | transition matrix and negative logs |
| L4 | 2 | immutable production image | build the reviewed patch stack into one OCI image, then repeat L1 smoke | Dockerfile/source map, OCI digest, SBOM and simplified card commands |
| L5 | 8 | fair speculative matrix | run every quant artifact at MTP off/depth 1/depth 3, C1/C4/C8 | raw three-run matrix and revised throughput claims |
| L6 | 4–6, 13 | candidate sides of paired quality tests | consume the content-hashed rental BF16 bundles; run quant profiles with identical prompts/scorers | candidate reports plus summaries bound to each BF16 receipt hash |
| L7 | 3 | clean-room reproduction | use a fresh checkout/cache and only public model/dataset revisions | independent data-only and runtime reproduction receipt |
| L8 | 12 | 5090 portability boundary | repeat the production smoke on the local driver/CUDA stack | explicit supported tuple; leave TP2/TP4/other architectures open |

**Local start sequence.** Run
`git clone https://github.com/malaiwah/qwen38-27b-exl3 && cd qwen38-27b-exl3`,
then `git fetch origin main && git checkout -B main origin/main`. Record `git rev-parse HEAD`,
verify the
three patch hashes in `receipts/native-mtp-8mp-amendment.json`, and download
`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`. Confirm `nvidia-smi` reports the physical card,
31.39 GiB-class usable memory and no competing process. Launch the content-verified command
in `MODEL_CARD-K5K6-context.md`; save GPU UUID/model/driver, full command and startup log
before sending requests. Run L1 in order: exact 261,794-token needle, 30-case image suite,
236,824-token combined seven-megapixel request, three warmed 256-token decode runs, then a
second long request after release. Do not tune after seeing a partial result; a failed gate
is published as a bounded negative result.

All local work lands in this repository: tools under `tools/`, immutable JSON/log evidence
under `receipts/`, and narrative in this file and `PROGRESS.md`; then run
`git push github main` (and mirror with `git push origin main` when the private Gitea remote
is reachable). Upload card-only changes to `malaiwah/Qwen3.8-27B-EXL3-K5K6-context`;
dataset captures and reports go to `malaiwah/qwen38-27b-fidelity-suite-v3`. Never compare
throughput across the two GPUs;
cross-machine pairing is allowed only for deterministic capability outputs whose prompt,
scorer, runtime and BF16 receipt hashes match.

## Contract for every investigation

Before the first candidate run, write a small machine-readable plan containing:

1. question, primary metric, paired unit, acceptance threshold and explicit failure outcome;
2. checkpoint index/config/release-evidence hashes, tokenizer revision, runtime image digest,
   patch hashes, command, environment knobs, GPU UUID/model, driver, CUDA and Torch versions;
3. immutable input identities and partition policy, including every exclusion decided before
   results are visible;
4. BF16 or current-profile control, warm-up policy, repeat count and randomness controls;
5. output schema with raw per-item results, not only means.

Afterward, publish the plan unchanged beside atomic per-run reports, stdout/stderr, a summary
derived only from those reports, and a SHA-256 inventory. Failed and rejected routes stay in
the record. Any post-hoc correction gets a new amendment that names the superseded receipt;
old evidence is never rewritten. A result closes a row only when a fresh checkout can run the
documented verifier without private paths, mutable branches or unrecorded model state.

## P0 / rank 1 — physical RTX 5090 qualification

This is the highest-value remaining test because it changes the context card from
“budget-proven” to “hardware-qualified”.

Run the exact native profile on a real 31.39 GiB RTX 5090:

- int8 input overlay, MTP-3, FP8 KV, decode graphs, one sequence;
- native 262,144;
- `max_pixels=8388608`;
- the three content-hashed runtime patches from the amendment receipt;
- no other process on the GPU, with `nvidia-smi` memory sampled before load, after startup,
  at peak prefill and after release.

Acceptance is all-or-nothing:

1. startup allocates at least one native-length request without exceeding utilisation 0.97;
2. the 261,794-token text needle is exact;
3. the combined 236,824-token / seven-megapixel request returns both code and colours exactly;
4. the 30-case image suite remains 24/30 or better;
5. three warmed 256-token C1 runs report median decode, MTP acceptance and dispersion;
6. a second long request succeeds after the first is released, proving recovery rather than
   one lucky allocation;
7. receipt includes GPU UUID/model, driver, image digest, patch hashes, command, peak memory,
   KV allocation, outputs and wall times.

Failure is still useful: report the exact shortfall and lower `max_pixels` before sacrificing
MTP or native length. A 4.2 MP trial fit at 30.44 GiB and the 8.4 MP profile fit at the exact
30.24 GiB budget, so this is a bounded frontier rather than an open-ended search.

## P0 / rank 2 — immutable production runtime

The published r34 image predates every required patch. Bind-mounting Python modules is
reproducible but not a production distribution.

Build one immutable image from the pinned r34 digest plus reviewed versions of:

- #312: BF16 fallback for unrepresentable online-overlay shapes;
- #314: EXL3 graph-decode autotune priming;
- #316 and #318: prefill reconstruction dispatch and B12X row-count routing;
- #319: pass the existing quant config to both Qwen3.5 input-embedding constructors.

Acceptance:

1. source modules in the image match the published SHA-256 map and a generated SBOM;
2. no bind mounts, startup file copies, mutable branch fetches or runtime package installs;
3. all four model-card recipes start, one text and one image request pass, and the native
   context profile repeats its memory allocation;
4. image digest, composed Git trees, package version and CUDA/Torch/driver compatibility are
   recorded;
5. model weights mount read-only, the service runs without host-root privilege, and only the
   content-addressed cache is writable;
6. the public recipe either binds loopback or requires an API key behind TLS; it never exposes
   an unauthenticated generation endpoint on every host interface;
7. cards collapse the current “unmodified” and “patched” recipes to one command only after
   that digest exists.

Wheel archives alone are insufficient: the prior audit found they were not bit-reproducible.
The OCI digest and source-tree identities are the release unit.

## P0 / rank 3 — clean-room release reproduction

The current receipts are internally cross-checkable, but most were produced on one
workstation. A second operator should begin from empty caches and only public URLs.

Two rungs are required:

1. **data-only:** verify Git/Hugging Face revisions and every release-evidence hash, download
   the published hidden-state captures, apply the declared v3/v4 exclusion policies, and
   reproduce every corrected mean, paired win count and interval used in the cards;
2. **build/runtime:** from pinned BF16 shards, converter source and immutable runtime, rebuild
   one representative checkpoint, compare logical tensor names/shapes/dtypes/storage metadata,
   then run deterministic text, image, graph/eager and KLD smoke.

Acceptance: a fresh-machine transcript contains no `/var/tmp/work` or unpublished artifact,
the data-only values match at full stored precision, every fetched blob is content-identified,
and build differences are either byte-identical or explicitly reduced to a measured numerical
equivalence contract. The independent operator signs the receipt or publishes it from a
separate account. A copy of this workstation is not an independent reproduction.

## P1 / rank 4 — public capability retention

The 40-task suite is a good regression smoke test, not evidence of broad capability. Extend
the same paired BF16/candidate design to established, licence-compatible task sets:

- code: HumanEval+/MBPP-style executable cases;
- reasoning/knowledge: a fixed MMLU-Pro or equivalent subset with exact prompt templates;
- instruction following: IFEval-style verifiable constraints;
- tools: schema-constrained multi-turn calls;
- multimodal: OCR, ChartQA and document question answering with original image hashes.

Run BF16, hydrated, online K5/K6, context, K4 and official FP8 with identical tokenizer,
template, greedy settings and runtime where formats permit. Publish every prompt, raw output,
scorer version and model identity. Primary statistic is paired pass retention versus BF16 with
bootstrap or exact intervals; absolute scores are secondary. No claim graduates from smoke to
benchmark until the public items and scorer are independently rerunnable.

## P1 / rank 5 — real multimodal quality

The current 24/30 suite is deterministic and useful, but synthetic. Add OCR, chart,
document-layout and video subsets with redistribution-safe fixtures. Run BF16 first so the
result measures retention rather than base-model capability. The 8.4 MP ceiling must be
reported as part of every result; include resize dimensions and visual-token counts.

Acceptance: paired score and latency, no hidden resize/truncation, at least one near-cap image,
and a long-text-plus-image case on the same production profile.

## P1 / rank 6 — context quality beyond retrieval

Needle retrieval proves access, not reasoning over the window. At 32k, 128k, 236k and the
largest physical-5090 length:

- two-hop retrieval with evidence in separated regions;
- aggregation over many records;
- passkey distractor count sweeps;
- perplexity or next-token loss on held-out continuous text;
- one combined image-plus-text reasoning task, not only colour recognition.

Use the server-reported token count, record TTFT/inter-token latency and fail on silent
truncation. Compare the context build with BF16 at the same lengths that BF16 can support;
do not extrapolate a short-window KLD to 262k.

## P1 / rank 7 — lifecycle and cache reliability

The online-attention cache and reconstructed-prefill scratch paths have been measured during
successful runs, not under service faults. Test the immutable image with:

- empty, warm and read-only caches; one deliberately truncated or hash-mismatched cache entry;
- termination during online encoding followed by restart;
- ten sequential native-length requests and an intentionally rejected over-budget request,
  followed by a valid request on the same server;
- concurrency at every documented profile limit, client cancellation during prefill, and
  clean shutdown/restart without orphan GPU memory or processes;
- disk-full and unwritable-cache failures that must fail closed rather than silently encode
  into an unidentified location.

Record cold-start/encode time, cache bytes and content identities, resident/peak GPU memory,
file-descriptor/process counts, request status and post-release memory after every transition.
Acceptance: no corrupt cache reuse, no silent precision fallback, no unbounded host/GPU growth,
all expected 4xx/5xx failures are explicit, and the final valid request matches the first.

## P1 / rank 8 — fair speculative-decoding matrix

The 113.8 tok/s headline compared our MTP-enabled path with FP8/NVFP4 target-only runs. It is
real throughput but not a fair model-to-model speculative comparison.

Run, for every artifact whose preserved MTP head loads:

- MTP off, depth 1 and depth 3;
- concurrency 1 / 4 / 8;
- 256 fixed output tokens, identical prompts, three warmed repeats;
- short and long configured windows separately;
- resident memory, KV allocation, drafted/accepted tokens, acceptance by position, TTFT and
  aggregate/per-stream throughput.

The closing result is a matrix, not a single favourable ratio. Unsupported comparator paths
must be reported as such rather than silently run without MTP.

## P2 / rank 9 — error-driven mixed-precision allocation

This is the last untried lever likely to improve fidelity without spending more memory.
The role split was hand-designed; exllamav3 already emits per-module proxy errors during
conversion, but the original logs were lost.

For the next conversion:

1. tee immutable converter stdout and preserve `args.json`, codebook and per-module errors;
2. measure each eligible module at two or more adjacent bit widths;
3. solve the byte-constrained benefit curve rather than assigning one width per role;
4. build one candidate only if predicted gains exceed the ~1e-3 replay-resolution caveat;
5. select on v4 analysis, then run one untouched source-disjoint qualification.

Expected improvement is a hypothesis, not a promised 10–30 %. The candidate must beat the
current profile at equal resident bytes with a paired interval excluding zero.

## P2 / rank 10 — prefill kernels

FP8 activations are closed: +31 % prefill cost +0.0141 KLD and made the context build worse
than official FP8. Remaining credible work:

1. fuse `gate_proj` and `up_proj` reconstruction/GEMM because they share input and shape;
2. prototype a Marlin-shaped fused dequant-in-epilogue kernel in exllamav3;
3. retain B12X for decode and dispatch by row count.

Acceptance is end-to-end prefill gain at unchanged decode and a measured fidelity delta below
the replay floor. Kernel microbenchmarks alone do not close the item. NVFP4's 4-bit tensor-core
prefill remains an architectural ceiling, not a tuning target for Trellis.

## P2 / rank 11 — image-cap and concurrency frontier

Eight megapixels is the first safe native-MTP point, not necessarily the maximum. On the
physical 5090, sweep 4.2 / 6.3 / 8.4 / 10.5 MP and sequence counts 1 / 2 without changing
other knobs. Report startup activation, KV tokens, actual resized dimensions and combined
request success. Select the largest cap with at least 256 KV-token margin after the full
262,144 request; do not publish a zero-headroom maximum.

## P2 / rank 12 — portability and package quality

- Test TP2 and TP4; the input table is rank-sliced and the overlay must prove correct shard
  scales and no non-local payload.
- Test one SM100 and one pre-Blackwell CUDA target or declare the support boundary.
- Publish explicit mixed-precision metadata; the legacy `bits: 4.0` key cannot describe
  gate/up K5, down K6 and per-row int8 input.
- Upstream the side-model `AttributeError` diagnostic and report the
  `torch._scaled_mm(..., out_dtype=float16)` row-wise-scale corruption separately.

## P2 / rank 13 — quant-specific safety stability

Low mean KLD does not guarantee unchanged refusals, calibration or tool-use boundaries. Freeze
a redistribution-safe paired set spanning self-harm, cyber misuse, privacy, medical/legal
over-reliance, jailbreak pressure and benign near-neighbours. Run BF16 and every shipped
profile with identical templates and decoding, retaining raw outputs and per-item policy labels.

Primary outcomes are BF16-safe-to-unsafe regressions, benign over-refusals and exact refusal
agreement; report categories and paired examples, never only an aggregate “safety score”.
Any severe BF16-safe-to-unsafe regression blocks a broad intended-use claim until reviewed.
A clean small set supports only “no regression observed on this set”, not general safety.

## Prior art that constrains the plan

- NVIDIA's Qwen3.6 NVFP4 is the memory/performance reference, not a drop-in recipe for this
  architecture. Its compressed-tensors runtime and calibration are different.
- Unsloth's Qwen3.8 NVFP4 retains higher precision in the last eight MLP layers and still
  measures 0.092727 on this protocol; “keep the last layers high” is not sufficient by itself.
- Gilded Gnosis GLM-5.2 demonstrates online K6 and quantized MTP for an MoE model. Its BF16
  shared-expert exception does not map onto this dense Qwen topology.
- exllamav3 supplies LDLQ/Trellis/MCG and proxy errors, but its public CLI exposes a global
  width; the regex per-module strategy used here remains a local extension.
- The Kimi-K3 protocol motivated hidden-state replay. Our 6.54e-4 live/replay discrepancy is
  roughly 500× its reported reference floor, so sub-1e-3 absolute conclusions remain out of
  scope even though paired common-head comparisons are stable.

## Do not reopen without new evidence

- Quantizing the vision tower: the converter changes fused qkv tensor topology and the loader
  intentionally excludes vision.
- A separate low-bit draft embedding: it is aliased away after load.
- Compact KV grouping: measured total memory increased.
- BF16 `lm_head`: costs 1.589 GB for a delta at or below the replay floor.
- More online attention bits: K6 is already the measured best point and attention is not the
  prefill bottleneck.
- FP8 prefill activations: fidelity failure is measured under both tensor and row-wise scales.
