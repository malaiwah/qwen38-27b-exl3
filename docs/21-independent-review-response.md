# Response to the independent review (2026-08-14)

The review audited research-repo commit `7bb8127` and model-repo revision `49f6d5ab`.
Several findings were already being fixed while it was written; several were real and
are fixed now; a few are correct and remain open. Nothing in it was rejected as wrong.

Legend: **FIXED** (done and verifiable), **FIXED-IN-V2** (fixed in the iteration-2
artifact), **OPEN** (accepted, not yet done), **CLARIFIED** (no change needed, wording
tightened).

## Critical

| id | finding | status | evidence / action |
|---|---|---|---|
| C1 | v2 headline used prompts from the quantizer's own calibration corpus | **FIXED, THEN HARDENED** | Re-measured on separately sourced text in [18](18-results-fidelity-v3.md). A later all-position 12-token audit caught overlap that the fixed-stride “0 hits” scan missed; conservatively excluding both affected source documents changes K4 0.030736 -> 0.029679, K5/K6 0.008157 -> 0.007945, and FP8 0.013126 -> 0.012798. Every ranking survives; both receipts remain published |
| C2 | `splice_bf16_attn.py` raised `NameError: ATTN` | **FIXED** | `SPLICE.match` restored, regex selection parameterised (`--include-mtp`), docstring corrected. Hardening (clean-destination refusal, exact-count assertions, build receipt) landed in [`tools/finalize_checkpoint.py`](../tools/finalize_checkpoint.py) which now also verifies the reconstructed logical-tensor count (1199, matching upstream) |
| C3 | published codebook mix (`mcg` MLP + `mul1` head) is not reproducible from the documented `-cb mcg` | **FIXED-IN-V2** | Confirmed the anomaly on v1 by reading the shards. The v2 conversion, run in a single process with `-cb mcg`, produced `mcg` **everywhere including `lm_head`**, so v2 *is* reproducible from its published command. v1's `mul1` head is documented as an artefact of that run's crash-and-resume; v1 is superseded rather than re-derived |
| C4 | held-out v3 results absent; qualification code broken | **FIXED** | v3 suite, captures, reports, paired receipts and the noise floor are committed and published as a [dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3). The two qualification bugs (undefined `dense_prompt_logprobs`; passing a `[pos, vocab]` matrix where `[pos, hidden]` was expected) are fixed with a dedicated `qualification_metrics()`; result 6.54e-04, reported as our weakest control. The `run_v3.sh` overlay-env mismatch was real: it caused an `mm_mxfp8` crash, and K6 env is now set for every K4 capture including the primary |

## High priority

| id | finding | status | evidence / action |
|---|---|---|---|
| H1 | stale `config.json` quantization block (`bits 4.0`, `codebook mul1`, `mtp_bits 4`) | **FIXED-IN-V2** | v2 ships `quantization_manifest.json` with per-role truth (K5 gate/up, K6 down, K6/mcg head, quantized MTP, BF16 embed/vision, FP16 `in_proj_a/b`), plus `build-receipt.json` and `SHA256SUMS`. The legacy block is retained for loader compatibility and explicitly annotated as non-authoritative |
| H2 | `reasoning_effort="high"` invalid; 1M context recipe wrong | **FIXED** | Card now lists only `xhigh`/`medium`/`low` and replaces the `max_position_embeddings` override with Qwen's YaRN procedure, marked unverified on this runtime |
| H3 | `library_name: vllm` implies upstream compatibility | **FIXED** | Runtime warning moved above the first performance claim |
| H4 | "every role at equal or higher effective precision" not defensible | **FIXED** | Removed. Replaced with per-role facts and measured fidelity |
| H5 | KLD methodology overstatements | **PARTLY FIXED** | Fixed: reversed head-ablation win count (K6 head is worse in **64/74**, not 10/74 - see [16](16-head-attribution.md)); "MLP owns ~99 %" softened; float64 wording corrected to "float32 within chunk, float64 across chunks"; KLD bands labelled project-local and unvalidated; KV dtype now pinned explicitly rather than `auto`; a deterministic 40-task smoke shows zero BF16 regressions. **OPEN**: end-to-end (own-head) numbers for NVFP4 and FP8, and representative public task evaluation |
| H6 | v3 corpus not yet a frozen, reproducible benchmark | **FIXED-IN-V4** | v4 adds whole-source partitions, exact requested strata, source hashes, tokenizer identity, and source-disjoint qualification. The fixed-stride contamination claim was later corrected: all-position 12-token scanning (Unicode words plus Han/Kana characters) found overlap in 10/100 source documents; a conservative 36-context clean qualification preserves 36/36 wins. Schema v5 uses the corrected scan. Semantic paraphrase detection remains out of scope |
| H7 | receipts are not a chain of custody | **PARTLY FIXED** | Build receipts, SHA256SUMS, quantization manifests, suite/head hashes, model index identities and container digests now exist. Capture writes and manifests are atomic; resume validates contract, file set, shape and digest; replay refuses incomplete or modified captures. Older captures still lack shard hashes inside every individual report |
| H8 | performance evidence is a probe, not a suite | **PARTLY FIXED** | v2 numbers are the median of 3 repeated runs (dispersion <1 %), prefill throughput is now measured separately with **exact** token-count prompts built from the frozen suite, and MTP acceptance is measured from the server's own counters. **OPEN**: streaming TTFT/ITL, randomised A/B ordering, more context lengths, profiler evidence for the kernel-bound claim |
| H9 | multimodal / MTP / long context assumed | **PARTLY FIXED** | Fixed: a 2044x1622 image answers correctly (and reproduces [#313](https://github.com/local-inference-lab/vllm/issues/313), which needs `--mm-processor-kwargs '{"truncation":false}'`); MTP speculative decoding measured end to end with the **quantized** draft (58.2 % acceptance, 2.745 tokens/step, +101 % single-stream throughput); the capped RTX PRO 6000 profile starts native 262K with MTP-3, retrieves exactly from 261,794 text tokens, and retrieves plus reads an image in a 236,824-token request. **OPEN**: the physical RTX 5090 rerun, YaRN-1M, real OCR/chart/document/video evaluation, and reasoning beyond planted-code retrieval |
| H10 | charts mix measured resident memory with file-derived size | **FIXED** | Comparator resident memory is now collected from each engine's own allocation log under identical flags, so every plotted point is the same kind of measurement |

## Medium

| id | finding | status |
|---|---|---|
| M1 | OCI puller hardening; read-only model mount; `VLLM_ALLOW_INSECURE_SERIALIZATION` trust boundary | **OPEN** (documented as a known gap; the serialization switch is required only by our capture RPC and is now called out) |
| M2 | documentation inconsistencies (stale checklists, broken link, superseded ignore-regex, branch claim) | **FIXED** — README status table corrected, `PROGRESS.md` "Next" replaced with pointers, [03](03-gg-runtime-contract.md) marks the crashing regex as superseded, branch claim removed until branches exist |
| M3 | HF-local reproducibility incomplete | **FIXED-IN-V2** via `build-receipt.json`, `SHA256SUMS`, `quantization_manifest.json`, pinned base revision |
| M4 | no safety / intended-use section | **FIXED** — added to the card, distinguishing inherited upstream unknowns from the absence of quant-specific safety testing |
| M5 | research repo lacks licence, dependency lock, CI, tags | **PARTLY FIXED** — licence and requirements added; CI and signed tags OPEN |

## Findings we consider the most valuable

Two of the review's catches changed real conclusions rather than wording:

1. **The calibration-overlap bias (C1).** Fixing it moved every number and *reversed* one
   conclusion: on held-out text official FP8 is much stronger than the contaminated
   suite suggested (0.013126 vs 0.019309), which set the real target for iteration 2.
2. **The reversed head-ablation win count (H5).** The direction of the aggregate was
   right, but 64/74 versus 10/74 is the difference between "noise" and "consistently
   but slightly worse", and it matters for whether the head is worth revisiting at
   lower bit widths.

## What the review could not see

The audit predates the iteration-2 artifact. On the held-out suite that artifact
(gate K5 / up K5 / down K6, `mcg` throughout, quantized MTP) measures **0.008157**
mean KLD — **38 % lower than official FP8** at 21.82 GB resident versus 30.87 GB — with
CUDA graphs working, MTP speculative decoding working, and the metadata/receipt gaps
above closed. Details in [22](22-results-iteration-2.md).
