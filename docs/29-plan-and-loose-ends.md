# What is left, ordered by evidence value

State at this point: three published builds, all headline numbers recomputable from the
dataset, two independent reviews and one hardware test addressed. What follows is ranked by
what each item would *prove*, not by how hard it is.

## P0 — DONE, and the goal it chased is now proven unreachable

Built and published as
[`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context):
attention serialized at K5, 19.31 GiB resident, **mean KLD 0.009673 — 26 % below official FP8,
135/136 contexts** — and long context verified by generation, **9/9 exact needle retrievals up
to 196,857 tokens**. Full write-up in [docs/30](30-iteration-4-context-edition.md).

The native-262,144 goal it was built for is **not reachable** and the gap is now exact rather
than modelled: 8.18 GiB of KV needed against **7.55 GiB available** on a 32 GB card with
multimodal profiling on. Smaller prefill chunks do not help (1024 and 512 both leave 7.54-7.55).
MTP's KV cost was measured for the first time — +0.65 GiB for the draft layer, only +0.30 more
from depth 1 to depth 3 — so speculative decoding is nearly free once you pay for it at all,
but it still costs 33k tokens of context.

The vision-tower lever died on measurement: `-vb 6` saves 0.58 GB but the converter splits
upstream's fused `visual.blocks.N.attn.qkv` into q/k/v, so the checkpoint stops matching the
architecture, and the loader excludes vision by design anyway.

**DONE — the embedding table was the answer.** Narrowed to per-row int8 behind
`VLLM_EXL3_EMBED_BITS=8`: resident 19.31 -> **18.13 GiB**, **native 262,144 now starts** with
279,007 KV tokens, verified by 3/3 exact needle retrievals from 227,334-token prompts, for
**+0.000065 mean KLD** and identical multimodal scoring. Two two-line model patches were needed
because `VocabParallelEmbedding` is constructed without a quant config. Write-up in
[docs/32](32-native-context-embedding-overlay.md).

Remaining on this axis, now with the lever eliminated: MTP and native context still do not
coexist (8.83 GiB needed at depth 1 against 8.36 available). int4 for the draft table is
implemented and **free** — acceptance 56.7 % against 56.1 %, throughput equal or better, 0.62 GB
saved — but the KV budget did not move, so **MTP's cost at native context is not its weights**.
It is the draft's own KV plus its share of the profiled peak. Anyone attacking this next should
look at the draft's KV allocation and the multimodal profiling peak, not at more weight
compression.

## P0 — DONE: the frozen qualification ran, and the ranking survived

Built and run once on **160 contexts from 100 documents with zero intersection with the
development suite** (token hashes 0/160, document names 0/100, content hashes 0/100), whole
cluster partitioning, `cluster_partition.overlap` empty. Full write-up in
[docs/31](31-frozen-qualification.md).

On the 42-context qualification partition: hydrated **0.003029**, K5/K6 0.003395, context
edition 0.003900, official FP8 0.005720 — **42/42 paired contexts for all three builds**, and
the relative advantage over FP8 is *larger* on unseen sources (47/41/32 %) than on the
development suite (44/38/26 %).

One caveat that must travel with every number: absolute KLD is ~2.4x lower on v4 than v3 for
every candidate including FP8, so magnitudes are comparable only within a suite.

Remaining discipline: development uses the v4 **analysis** partition only, so the qualification
partition stays clean for the next frozen run.

## P1 — RESOLVED: FP8 prefill measured, rejected on fidelity

The kernel works and is bit-exact; the serving result is **+31 % prefill (5,078 → 6,650 tok/s)
for +0.0141 mean KLD**, which lands the build at 0.0237 — worse than official FP8. Row-wise
scaling (per-token activation, per-channel weight) changed nothing, so the loss is FP8
*activations* rather than scale granularity. Shipped disabled behind
`VLLM_EXL3_PREFILL_FP8=1` for anyone who wants prefill over fidelity.

What did ship from that work: the B12X prefill routing, +3.4 % for no measurable fidelity cost.

Still open on the prefill axis, in order of remaining value: fuse `gate_proj` and `up_proj`
into one GEMM (same input, same shape, larger N measures slightly more efficient), and the
Marlin-shaped fused dequant-in-epilogue kernel that would reach FP8 parity — a real kernel,
weeks of work, and it belongs in exllamav3.

## P1 — earn the word "verified" for context, and for tasks

Two gaps where the cards currently say "not done":

- **Context exercises.** Prefill *and* generate at 32k / 128k / 196k / 262k on the 96 GB box,
  with needle retrieval, TTFT, inter-token latency and peak memory recorded per run. The
  external tester did exactly this at 205,021 tokens and it is the strongest single piece of
  evidence anyone has produced for this family.
- **Downstream retention.** Paired greedy runs against BF16 on the same prompts and seeds:
  code, grade-school reasoning, instruction following, tool-call schema conformance, and a
  multimodal set (OCR, chart, document). Report paired deltas with intervals, not absolute
  scores, because the point is retention rather than leaderboard placement.

## P2 — fairness and hygiene

New from iteration 4:

- **FILED: B12X prefill routing** as [PR #318](https://github.com/local-inference-lab/vllm/pull/318),
  carrying the int8 embedding overlay with it, and the model-side change as
  [PR #319](https://github.com/local-inference-lab/vllm/pull/319).
- **Report `torch._scaled_mm` row-wise + `out_dtype=float16`** returning silently wrong results
  (~6x relative error, no exception). bfloat16 output is correct.
- **Patch exllamav3's `quantize_side_model`**: it raises a bare `AttributeError` when a
  side-model Linear is consumed without being rebuilt, naming nothing. Local fix names the
  module; upstream it properly.
- **Keep `reconstruct_fp8_slice`** even though FP8 activations were rejected: it is bit-exact
  with per-column scales and is the right primitive if a path that keeps activations in fp16
  ever exists.


- **Symmetric MTP matrix.** MTP off/on x draft depth 1-4 x concurrency 1/4/8, for our builds
  *and* for FP8/NVFP4 using their own preserved MTP. The current 113.8 tok/s headline compares
  our speculative decoding against comparators without it, which the reviewers correctly called
  unfair. Accounting to keep straight: 58.2 % of drafted tokens accepted, 1.745 accepted draft
  tokens per step, 2.745 output tokens per iteration.
- **Support matrix**: tested (SM120, TP1, text+image), untested (TP>1, non-SM120, video),
  unsupported (generic NVFP4 KV on SM120 - the runtime requires SM100 trtllm-gen; GLM-5.2's
  `nvfp4_ds_mla` is MLA-only and Qwen3.8 is not MLA).
- **HF metadata**: `config.json` still carries a single `bits: 4.0` for loader compatibility, so
  HF tags a K5/K6 artifact "4-bit". Publish explicit mixed-precision fields and say in one line
  why the legacy key cannot describe the mix.
- **Error-driven allocation** - the original P1, still unmeasured. It needs per-module proxy
  error at two or more bit widths; the data existed in conversion stdout and was lost to log
  rotation. Next conversion must tee stdout to a file, then the ladder can be fitted and the
  allocation solved under a byte budget instead of hand-split by role. Expected 10-30 % KLD at
  equal bytes, and it is the last untried lever that improves fidelity without spending memory.
- **Embedding quantization** is worth 1.589 GB - more than the entire native-context gap - but
  exllamav3 quantizes `Linear`, not `Embedding`, so it needs new code. Park it behind the vision
  tower, which is free.
- **Near-duplicate contamination scanning.** Exact 160- and 80-character shingle scans find zero
  overlap with the calibration corpus; rolling n-gram or MinHash would test paraphrase overlap,
  which is currently unmeasured.
- **Capture resume must fail closed.** A resumed capture records only newly written files in the
  replacement manifest, and paired analysis silently intersects available contexts. Both should
  refuse rather than proceed for a publication receipt.

## Owned elsewhere

- **An immutable image containing PRs #312/#314/#316** is being built by voipmonitor (Martin),
  so it is off this list. Until that digest exists the cards ship two recipes: what the pinned
  r34 image runs unmodified (eager, 28.8 tok/s decode, 2.4k prefill) and the patched path with
  the module's sha256. When the digest lands, replace both recipes with one and drop the patch
  instructions.

## Closed while writing this plan (card hygiene, found by auditing the live repos)

Auditing the three published cards against the hub API rather than trusting them turned up
three defects, all now fixed:

- the **K5/K6 card had lost its `library_name`**, so the hub inferred `library_name: trellis`
  for a checkpoint that only runs under vLLM. Restored to `vllm`; verified through the API on
  all three repos.
- the **K4 card still asserted the withdrawn CUDA-graph parity control as live fact**, in two
  places, while every other surface had been corrected. Now carries the same retraction and the
  real decode probe (24/32, BF16 control identical).
- the **K4 card had a duplicated `### Head attribution` heading** from an earlier edit, and no
  cross-links. It is now explicitly the **capacity edition** — the only build in the family that
  fits native 262,144 on a 32 GB card — and all three cards carry the same three-build choice
  table and collection link.

The lesson worth keeping: card claims must be re-read from the hub after every edit round. Three
surfaces drifted apart even though each individual edit was correct.

## Deliberately not doing

- More bits on attention. Measured: online K6 attention already beats BF16 attention on this
  architecture, and the overlay is not the prefill bottleneck (1.05-1.11x).
- Chasing NVFP4's prefill. Even a perfect fused FP8 kernel reaches FP8 parity, not NVFP4's
  14.5k tok/s, because that needs 4-bit tensor cores. Prefill is the one axis where the honest
  ceiling is a draw.
