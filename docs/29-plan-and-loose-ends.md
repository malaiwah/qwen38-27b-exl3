# What is left, ordered by evidence value

State at this point: three published builds, all headline numbers recomputable from the
dataset, two independent reviews and one hardware test addressed. What follows is ranked by
what each item would *prove*, not by how hard it is.

## P0 — a build that reaches native context on a 32 GB card while beating FP8

This is the one genuinely new capability within reach, and the arithmetic says it is one
conversion away.

The RTX 5090 test showed K5/K6 short of native 262,144 by 0.83 GiB with MTP-3. Two levers
close it, and only one needs new code:

| configuration | resident weights | KV available at util 0.97, seqs 4 | KV needed at 262,144 | verdict |
|---|---:|---:|---:|---|
| K5 attention, MTP-3 | 19.40-19.82 GiB | 8.30 | 9.13 | short 0.83 |
| K5 attention, **MTP off** | 19.40-19.82 GiB | 8.30 | **8.18** | marginal: fits on their footprint, 0.30 short on mine |
| K5 attention, MTP off, **vision tower K6** | ~18.8-19.3 GiB | **8.41-8.83** | 8.18 | **fits** |

Turning MTP off is free to try (it is a launch flag) and costs 2x decode throughput. The vision
tower is 0.921 GB of BF16 that the converter can already quantize with `-vb 6`, saving
~0.58 GB, and nothing in the recipe justifies BF16 there beyond "untested".

**Build it:** `-vb 6` plus attention serialized at K5 (`EXL3_BITS_FIXED` attention=5), MLP
unchanged. Expected: ~18.8 GiB resident, ~19.9 GB download, mean KLD near 0.0125 — still below
official FP8's 0.013126 — and native 262,144 on a 32 GB card.

**Acceptance:** starts at `--max-model-len 262144` inside a 30.44 GiB budget with `--max-num-seqs 4`;
a 262,144-token prefill plus generation completes; needle retrieval passes at 262k; a
3,264-vision-token image still answers correctly; mean KLD on the v4 suite below FP8's.

**Why it matters:** it would make the family's fidelity leader *and* its capacity leader the
same artifact, which is what the K4 build currently wins by default.

Also chase, in the same pass: **our K5 footprint is 19.82 GiB and theirs is 19.40** on the same
checkpoint and overlay width. A 0.42 GiB unexplained difference is a real finding either way -
either their flags free memory we are wasting, or one of us is measuring a different thing.

## P0 — the frozen, source-disjoint qualification

The v3 suite cannot serve as a post-selection test: all 27 qualification clusters also appear
in the analysis partition that guided every recipe choice. A v4 suite is being built from new
documents (six Wikipedia languages, new Gutenberg works, arXiv categories and code sources not
used in v3), with whole-cluster partitioning and `--exclude-suite` refusing every v3 document
and context hash.

**Then:** capture K5/K6-online-K6, K5-overlay, hydrated and official FP8 on it, replay **once**,
publish whatever it says. No recipe changes may follow from it, or it stops being a test.

**Acceptance:** zero intersection with v3 on document sha256 and on context token hashes;
`cluster_partition.overlap` empty; one published result per candidate with paired intervals.

## P1 — the FP8 prefill path, end to end and upstreamed

The kernel exists and is verified: `reconstruct_fp8_slice` emits E4M3 transposed straight from
the shared-memory tile, **bit-identical to a fp32 reference across 89 M elements on three
shapes**, and makes the prefill primitive **1.76-1.98x** faster. The dispatch is already wired
behind `VLLM_EXL3_PREFILL_FP8=1`.

What remains is the part that turns a microbenchmark into a claim:

1. serve with the flag, measure PP at 2k/6k and TG at C1/C4/C8 (expect PP 5,050 -> 7,500-8,300);
2. measure the fidelity cost on the sentinel set - both operands narrow to E4M3 on the prefill
   path only, so this must clear the same bar as the +0.43 % that the fp16 dispatch costs;
3. PR the kernel to exllamav3 and the dispatch to the vLLM fork, with the numbers and the
   negative result that motivates it (the Python-only route is 2-4x *slower* because the layout
   conversion costs more than the GEMM saves).

Then the cheap adjunct: fuse `gate_proj` and `up_proj` into one GEMM. Same input, same shape,
and the larger N measures slightly more efficient.

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

## Blocked on someone else

- **An immutable image containing the patches.** PRs #312/#314/#316 are open; the pinned r34
  digest predates them. This box has no container builder (the rootfs is extracted with an
  unprivileged puller and run through proot), so the honest options are to ask the image owner
  for an r35 build or to publish the patched module with its sha256, which the cards now do.

## Deliberately not doing

- More bits on attention. Measured: online K6 attention already beats BF16 attention on this
  architecture, and the overlay is not the prefill bottleneck (1.05-1.11x).
- Chasing NVFP4's prefill. Even a perfect fused FP8 kernel reaches FP8 parity, not NVFP4's
  14.5k tok/s, because that needs 4-bit tensor cores. Prefill is the one axis where the honest
  ceiling is a draw.
