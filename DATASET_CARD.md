---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - de
  - fr
  - es
  - ja
  - zh
  - ru
  - it
  - pt
tags:
  - quantization
  - evaluation
  - kl-divergence
  - distribution-fidelity
  - exl3
  - nvfp4
  - fp8
  - qwen3.8
pretty_name: Qwen3.8-27B distribution-fidelity suite v3
size_categories:
  - n<1K
---

# Qwen3.8-27B distribution-fidelity suite v3 (held-out, 181 x 2048)

The frozen evaluation suite and the captured hidden states that let anyone **recompute or
contest** the KL-divergence numbers published for
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) — without a
GPU, without downloading any model, and without trusting the publisher.

**Scope, stated up front.** The captures in this snapshot are the **iteration-1** set:
`malaiwah/Qwen3.8-27B-K4`, `unsloth/Qwen3.8-27B-NVFP4` and `Qwen/Qwen3.8-27B-FP8` against
the BF16 reference. The **current** published checkpoint,
[`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6)
(mean KLD 0.008157), was measured on **this same suite and this same LM head**, but its
captures are **not in this dataset yet** — see [What is missing](#what-is-missing).

Protocol adopted from
[Kimi-K3 distribution fidelity](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/distribution-fidelity-1024x2048.md):
capture the BF16 hidden state after the final RMSNorm and before the LM head, then
reconstruct complete full-vocabulary distributions offline through **one shared BF16
LM head**. Storage is 21 MB per context instead of 2.03 GB of fp32 logits, which is
what makes many-context evaluation affordable.

## What is measured

`KL(BF16 reference || candidate)` over the entire 248,320-token vocabulary, no top-k,
two passes (log-sum-exp normalisers and argmax first, then divergence), float32 within a
vocabulary chunk accumulated in float64 across chunks, source-cluster bootstrap. Also
Jensen-Shannon divergence, top-1 agreement, tail quantiles, and per-stratum means.

Because both operands go through the same shared head, no candidate's own head
quantization is counted. That is what makes the three candidates comparable, and it is why
these are **body-only** numbers.

## Headline results reproducible from these files

Analysis partition, 136 contexts, 278,392 scored positions (iteration 1):

| candidate | weights | mean KLD | bootstrap 95 % CI | median | top-1 agreement |
|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 96.22 % |
| `malaiwah/Qwen3.8-27B-K4` | 19.2 GB | 0.030736 | [0.02238, 0.04073] | 0.004218 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 90.53 % |

Paired over the same contexts: K4 beats NVFP4 by 0.064242 (95 % CI
[-0.08621, -0.04611], **136/136 contexts**); FP8 beats K4 by 0.017611 (95 % CI
[0.01256, 0.02368], **136/136 contexts**).

## Layout

Total 17.75 GiB. Every count and size below is the actual content of this revision.

```
suite-manifest.json          181 contexts: stratum, source cluster, partition, sentinel
                             flag, token SHA-256, contamination-scan result;
                             partitions 136 analysis / 45 qualification / 32 sentinels
tokens/                      181 x context-NNNN.json, the exact token IDs (2048 each)
reference-hidden/            181 x hidden_NNNN.safetensors, BF16 [2047, 5120], from
                             Qwen/Qwen3.8-27B, plus capture-manifest.json (per-file
                             SHA-256 + the runtime config used)         3.53 GiB
candidate-hidden/            three candidates, same shape and manifest  3.53 GiB each
  k4-online-k6/              malaiwah/Qwen3.8-27B-K4: EXL3 K4 MLP + online-K6 attention
  qwen-fp8/                  Qwen/Qwen3.8-27B-FP8
  unsloth-nvfp4/             unsloth/Qwen3.8-27B-NVFP4
sentinel-hidden/             two extra captures of the K4 runtime over the 32 sentinel
  repeat-02/, repeat-03/     contexts, for the noise floor              0.62 GiB each
lm-head/weight.safetensors   BF16 [248320, 5120], extracted from
                             Qwen/Qwen3.8-27B@1d4bf0f2, SHA-256 25a30fd5…dee4cfff  2.37 GiB
reports/                     8 JSON receipts:
                             report-{k4,fp8,nvfp4}-analysis.json  the three candidates
                             paired-k4-vs-{fp8,nvfp4}.json        the paired comparisons
                             noise-r1-vs-r2.json, noise-r2-vs-r3.json  the noise floor
                             qualification-bf16.json              live-vs-replay error
checksums.txt                SHA-256 of all 985 data files (everything but itself and
                             this README)
```

`tokens/` is authoritative: **retokenising source text does not reproduce the
evaluation input.** Verify integrity with `sha256sum --check checksums.txt` before use.

## Reproduce a number

```bash
git clone https://github.com/malaiwah/qwen38-27b-exl3
hf download malaiwah/qwen38-27b-fidelity-suite-v3 --repo-type dataset --local-dir suite-v3

# recompute our published K4 result from the shipped captures (no model needed)
python qwen38-27b-exl3/tools/fidelity.py replay \
  --reference suite-v3/reference-hidden \
  --candidate suite-v3/candidate-hidden/k4-online-k6 \
  --head suite-v3/lm-head/weight.safetensors \
  --suite suite-v3 --filter analysis --out my-k4.json

# then compare receipts
python qwen38-27b-exl3/tools/fidelity.py paired \
  --a my-k4.json --b suite-v3/reports/report-nvfp4-analysis.json \
  --a-label mine --b-label nvfp4 --out my-paired.json
```

To score **your own** checkpoint, capture its hidden states over the same tokens
(`fidelity.py capture --suite suite-v3 ...`) and replay against `reference-hidden/`.

## Held-out corpus, and the contamination scan

| stratum | contexts | source |
|---|---:|---|
| literary | 52 | Project Gutenberg, public domain |
| code | 52 | CPython `Lib/` at tag `v3.12.8`, PSF licence |
| scientific | 52 | arXiv abstracts via the arXiv API |
| encyclopedic | 17 | Wikipedia extracts, English, CC BY-SA 4.0 |
| multilingual | 8 | Wikipedia extracts in de, fr, es, ja, zh, ru, it, pt, CC BY-SA 4.0 |

Why held out matters: the **previous** version of this suite was built from
exllamav3's bundled calibration corpora — the same text the EXL3 candidate was
calibrated on, while the NVFP4 and FP8 candidates were calibrated elsewhere. That
biased the comparison. Re-measuring on held-out text moved our own number from
0.026231 to 0.030736 (+17 %) and moved FP8's from 0.019309 to 0.013126 (-32 %). The
published v3 numbers are the honest ones.

A 160-character shingle scan of every context against every exllamav3 calibration
corpus reports **0 contaminated contexts, 0 hits** (67,818 calibration shingles).
Recorded in `suite-manifest.json -> contamination_scan`.

Token IDs are a lossless encoding of the source text, so the Wikipedia-derived
strata carry CC BY-SA 4.0 attribution requirements; the Gutenberg strata are public
domain; CPython is PSF-licensed; arXiv abstracts remain under their authors' terms.
`lm-head/weight.safetensors` is one tensor extracted from `Qwen/Qwen3.8-27B`
(Apache-2.0) and is redistributed under that licence.

## Controls shipped with the data

| control | receipt | result | why it matters |
|---|---|---|---|
| runtime-repeat noise floor | `reports/noise-r1-vs-r2.json`, `reports/noise-r2-vs-r3.json` | **0.000000** mean KLD between captures 1-2 and 2-3 of the same runtime over 32 sentinel contexts, top-1 1.0 | every reported difference is far above measurement noise; this runtime is bit-deterministic across process restarts, unlike the TP16 reference runtime whose floor was 0.0032 |
| harness self-check | reproducible with `fidelity.py replay` | reference scored against itself returns exactly 0.000000 | no bias in densification or window handling |
| replay qualification | `reports/qualification-bf16.json` | mean `KL(live served logits \|\| replayed logits)` = **6.54e-04**, top-1 98.999 % over 6 contexts | the replay path is not exact; see caveat 1 |

### The withdrawn CUDA-graph control

Earlier revisions of this card listed a fourth control, "CUDA-graph parity 0.000000
between graph and eager captures of the same checkpoint". **It is withdrawn and is not
shipped here.** `fidelity.py capture` takes a single **prefill** forward, and the graph
mode under test (`cudagraph_mode=FULL_DECODE_ONLY`) captures **no prefill graph**, so that
measurement compared two runs of the same eager prefill code and could not have measured
the decode path at all. The only claim it supports is "enabling graph decode does not
change prefill numerics".

Re-measured on real decode steps (32 prompts x 32 greedy tokens, `temperature 0`, fixed
seed, via the serving endpoint), graph and eager agree on **24/32** exact sequences with
mean `|delta logprob|` **0.0118** on the chosen token, and each mode is internally
deterministic (32/32 self-repeat). Unquantised **BF16** on the same build drifts by the
same amount (**24/32**, **0.0128**), so the drift belongs to this build's CUDA-graph
decode path and is **not** caused by the quantisation. Full result:
[docs/27-graph-decode-drift-control.md](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/27-graph-decode-drift-control.md).
These are decode-sequence receipts, not hidden-state captures, so they live in the
research repo (`receipts/decode-parity-*.json`) rather than in this dataset.

## What this snapshot now contains, and what it still does not

**Added (this revision):** the whole K5/K6 family's hidden-state captures, so every headline
in the family's model cards is independently recomputable without a GPU or a checkpoint
download:

| directory | build | mean KLD (analysis partition) |
|---|---|---:|
| `candidate-hidden/k5k6-hydrated-offline-k6` | attention serialized offline at K6 | 0.007406 |
| `candidate-hidden/k5k6-online-k6` | attention BF16 on disk, encoded K6 at load | 0.008157 |
| `candidate-hidden/k5k6-online-k5` | same download, overlay K5 | 0.012135 |
| `candidate-hidden/k5k6-online-k4` | same download, overlay K4 | 0.027530 |
| `candidate-hidden/k4-online-k6` | iteration-1 build | 0.030736 |
| `candidate-hidden/qwen-fp8` | official FP8 | 0.013126 |
| `candidate-hidden/unsloth-nvfp4` | Unsloth NVFP4 | 0.094978 |

`reports-k5k6/` carries the matching replay reports and every paired receipt, including the
two as-served (asymmetric-head) reports that attribute the K6 head: +0.000127 on the online
build, +0.000125 on the hydrated one.

**Still missing, stated plainly:**

- **A post-selection result.** Every number here is on the 136-context analysis partition,
  which guided recipe selection, and this suite's qualification partition is *not*
  source-disjoint from it: all 27 qualification clusters also appear in analysis, because the
  builder split contexts rather than source clusters. A v4 suite is being built from new
  documents with a group split; until it lands, treat these as development-set numbers.
- **Language coverage is narrower than the tags suggest.** The multilingual stratum is 6 German
  and 1 Russian context; the builder tolerated under-filled strata. It now fails instead, and
  v4 fetches more languages.
- **No downstream task, multimodal-quality or long-context accuracy data.** This dataset
  measures distribution divergence only.
- **No near-duplicate or semantic contamination scan.** The exact 160-character shingle scan
  found zero overlap with exllamav3's calibration corpus, and an independent reviewer
  reproduced that at 80 characters too, but near-duplicates are untested.

## Caveats, stated plainly

1. **Replay is not exact.** Our live-vs-replayed qualification is 6.54e-04, about
   500x worse than the 1.23e-06 the reference protocol reports on Kimi-K3. This was
   tested: storing the hidden states in **fp32 instead of BF16 moved it only to 6.25e-04**
   (-4.5 %), so operand rounding is ~5 % of the floor and the rest is the implementation
   difference between the serving runtime's logit path and our replay. Paired comparisons
   are unaffected because both arms use the identical replay path, but **absolute**
   differences below ~1e-3 are not resolvable with these artifacts.
2. 2048-token contexts only. Nothing here measures long context, and the model
   supports 262,144.
3. No dialogue/instruction, mathematics/reasoning, or structured/tool-call strata —
   exactly the distributions a thinking model is used for. 41 source clusters, versus
   827 in the reference artifact.
4. KLD ranks candidates **within this artifact's frozen identities**. It does not
   substitute for coding, reasoning, long-context, multimodal, tool-use or
   free-running generation evaluation, and thresholds from other models, corpora or
   tokenizers do not transfer.

## Provenance

| item | identity |
|---|---|
| reference model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| candidates | `malaiwah/Qwen3.8-27B-K4`; `unsloth/Qwen3.8-27B-NVFP4` @ `9c73e2da`; `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` |
| runtime | `voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b` (Gilded Gnosis r34), vLLM `0.11.2.dev280+gilded.gnosis.v20...r34` |
| hardware | 1x RTX PRO 6000 Blackwell Server Edition, SM120, driver 595.58.03, TP1 |
| capture | forward hook on `language_model.model.norm` via `collective_rpc`; `enforce_eager`, `max_num_seqs=1`, one prefill chunk per context |
| suite token SHA-256 | `3f9d17f1b55f64872ad3ac19c8711654e09ba70b7ca14b0851525088fe735691` |
| harness + docs | <https://github.com/malaiwah/qwen38-27b-exl3> |
