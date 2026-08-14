---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - exl3
  - exllamav3
  - trellis
  - mixed-precision
  - quantized
  - qwen3.8
  - vision-language
  - gilded-gnosis
---

# Qwen3.8-27B-K4 — EXL3 mixed-precision (superseded)

> ### Superseded by [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6)
>
> The successor spends the remaining memory budget on the MLP (gate/up K5, down K6) and
> measures **0.008157** mean KLD against this checkpoint's **0.030736** on the same
> held-out suite — and **38 % below official FP8** — at 21.82 GB resident. It also has a
> quantized MTP draft head that works with speculative decoding (58.2 % acceptance,
> +101 % single-stream throughput). Prefer it unless you specifically need the smaller
> 19.21 GB footprint.
>
> Two corrections to this card, from an independent review: the first published
> comparison used evaluation prompts drawn from the quantizer's own calibration corpus
> (re-measured held-out numbers are below), and `reasoning_effort` accepts only
> `xhigh`/`medium`/`low`. This checkpoint's `lm_head` also carries the `mul1` codebook
> rather than the `mcg` implied by the documented `-cb mcg`, an artefact of that run's
> crash-and-resume; the successor is `mcg` throughout and reproducible from its command.



Dense EXL3 quant of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
built on one principle: **spend 4 bits only where the two independent NVFP4
recipes for this architecture also spend 4 bits, and protect everything they
protect — but protect it with Trellis, not with FP8.**

- **MLP** (`gate_proj`/`up_proj`/`down_proj`, all 64 layers) → **EXL3 K4**,
  serialized, calibrated LDLQ.
- **Attention** (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` on the 48 linear
  layers, `self_attn.{q,k,v,o}_proj` on the 16 full-attention layers) → **BF16 on
  disk**, encoded to **K6 at load time** by the Gilded Gnosis vLLM fork's
  `ONLINE_QUANT=exl3-b6` overlay, cached content-addressed on disk.
- **`lm_head`** → **EXL3 K6** (6 bpw, serialized).
- **`embed_tokens`, vision tower (27 blocks), MTP draft head, norms** → **BF16**,
  untouched.

**Measured resident weights: 17.89 GiB (19.21 GB)** on a single RTX PRO 6000
Blackwell, versus **21.92 GB** for `nvidia/Qwen3.6-27B-NVFP4` and **23.42 GB**
for `unsloth/Qwen3.8-27B-NVFP4` on the identical architecture — a
**2.7 GB smaller** resident footprint. Per-role bit widths are not directly comparable
across Trellis, NVFP4 and FP8, so the comparison that matters is the measured fidelity
below, not a per-role precision claim.

Measured on a **held-out** corpus across **278,392 full-vocabulary positions in 136
stratified contexts**: mean KLD **0.030736** versus **0.094978** for
`unsloth/Qwen3.8-27B-NVFP4` (3.09x closer to BF16, **136/136 contexts**, CI excludes
zero) and **0.013126** for `Qwen/Qwen3.8-27B-FP8`, which is genuinely better
at 61 % more memory. With CUDA graphs enabled it also serves **faster than the NVFP4
checkpoint** (55.39 vs 49.09 tok/s at C1). Vision works. Every artifact needed to
recompute these numbers is published as a
[dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3).

## Why this shape

| role | this quant | `nvidia/Qwen3.6-27B-NVFP4` | `unsloth/Qwen3.8-27B-NVFP4` |
|---|---|---|---|
| MLP | **EXL3 K4, 4.004 bpw** | NVFP4 W4A16 gs16, **4.50 bpw** (4 b + FP8 scale per 16) | NVFP4 **W4A4** gs16 on L0-55; FP8 on L56-63 |
| attention | **BF16 on disk → K6 (6.0 bpw) in VRAM** | FP8 E4M3 W8A8, 8 bpw | FP8 W8A8 dynamic, 8 bpw |
| `lm_head` | **EXL3 K6** | NVFP4 (4 bpw) | FP8 (8 bpw) |
| `embed_tokens` | BF16 | BF16 | BF16 |
| vision tower | BF16 | BF16 | BF16 (explicit per-block ignore) |
| MTP head | BF16 | BF16 (`ignore: ["mtp*"]`) | BF16 (`re:^mtp.*`) |
| resident weights | **19.21 GB** | 21.92 GB | 23.42 GB |
| checkpoint size | 28.31 GB | 21.92 GB | 23.42 GB |

Trellis at K4 needs no per-group scale tensor, so 4-bit MLP costs 4.004 bpw here
against NVFP4's 4.50 bpw. That saved 1.07 GB, plus the 1.80 GB from serving
attention at K6 instead of FP8, is what pays for a *lower* footprint at *higher*
precision. The checkpoint is larger than the NVFP4 ones because attention ships
BF16 so the runtime can re-encode it — see *Tradeoffs*.

## Serving

Requires the **Gilded Gnosis vLLM fork** — the EXL3 checkpoint loader, the B12X
Trellis kernels and the `exl3-b6` online overlay are not in upstream vLLM. The
public image is:

```
voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
registry digest sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

Its launcher scripts only dispatch GLM-5.2 and DeepSeek families, so call
`vllm serve` directly:

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 8000:8000 \
  -v /models:/models -v /cache:/cache \
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
  -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-K4 \
    --served-model-name qwen38-k4 \
    --quantization exl3 \
    --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --host 0.0.0.0 --port 8000
```

Four flags are load-bearing:

1. **`--quantization exl3` is mandatory.** Auto-detection only fires for the
   GLM-5.2 `r7_routed_experts` / `hybrid_tr3_tail` metadata; a dense
   `tensor_storage` checkpoint is not auto-detected.
2. **`--enforce-eager` is mandatory.** The loader refuses non-eager execution for
   any checkpoint without rank-sliced metadata, because `exl3_gemm` autotunes with
   timing launches. Expect no CUDA graphs.
3. **The `ignore` list is mandatory and its anchoring is subtle.** The overlay
   claims *every* BF16 `LinearBase` not present in `tensor_storage` — including
   the vision tower and the MTP head. The prefixes it matches have no leading
   `model.`, so `re:.*\.visual\..*` (dot before `visual`) silently fails to match
   while `re:.*visual\..*` works. With the wrong pattern the vision tower is
   claimed and startup **crashes** (`ValueError: MXFP8 requires
   input_size_per_partition (4304) to be divisible by 32`), reported upstream as
   [local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311)
   with a verified fix in
   [PR #312](https://github.com/local-inference-lab/vllm/pull/312), which degrades
   those shards to BF16 with a warning instead of aborting.
4. **`VLLM_EXL3_ONLINE_TRELLIS_BITS=6`** is what turns the overlay from MXFP8 into
   K6. Point `VLLM_EXL3_ONLINE_CACHE_DIR` at persistent storage: the first load
   encodes 208 attention projections (~16 min on one GPU here) and later loads
   reuse the cache.

Generation defaults from upstream `generation_config.json`: `temperature 1.0`,
`top_p 0.95`, `top_k 20`. Thinking control is upstream's
`chat_template_kwargs`: `{"enable_thinking": false}` or
`{"reasoning_effort": "xhigh"|"medium"|"low"}` (upstream raises on `high`). The chat template,
tokenizer, preprocessor configs and vocabulary (248320, untied head) are
upstream's, unmodified.

Context: 262144 native, verified here only to 8,192. Upstream's 1M procedure is static
YaRN (nested `rope_parameters` with `rope_type: yarn`, `factor: 4.0`,
`original_max_position_embeddings: 262144`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
`--max-model-len 1000000`), not a bare `max_position_embeddings` bump, and Qwen warns it
costs short-context quality. Untested on this runtime.

## Verification

All measured on 1x RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB), driver
595.58.03, TP1, with the r34 image above.

**Loads and serves.** Engine reports `quantization=exl3`, online K6 encoding for
every attention projection (proxy error ~3.2e-4 per projection), then:

```
Actual usage is 17.89 GiB for weight, 2.33 GiB for peak activation,
0.26 GiB for non-torch memory, and 0.0 GiB for CUDAGraph memory
GPU KV cache size: 736,109 tokens   (--max-model-len 8192, --gpu-memory-utilization 0.85)
```

**Text is coherent.** `"Name the three primary additive colors, comma separated."`
with `enable_thinking: false` → `Red, Green, Blue` (1.6 s, greedy).

**Vision works.** A 96x96 PNG, left half pure red, right half pure blue, with
`"Name the left colour then the right colour, comma separated."` → `red, blue`.

**Quantization error, per tensor, from the conversion log** (LDLQ proxy error):
`down_proj` is consistently the worst projection in every layer — about
`2.5e-3` versus `1.1e-3` for `gate_proj` and `1.0e-3` for `in_proj_qkv`. Whole-block
figures: `rfn ~0.0155`, `sqnr ~36.4 dB`.

### Distribution fidelity — v3 protocol, held-out corpus

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fidelity-vs-size-dark.svg">
  <img alt="Mean KL divergence from BF16 versus resident weight footprint. This quant at 19.2 GB and 0.0307; Qwen FP8 at 30.9 GB and 0.0131; Unsloth NVFP4 at 23.4 GB and 0.0950. Right panel shows top-1 agreement: 94.50, 96.22 and 90.53 percent respectively." src="assets/fidelity-vs-size-light.svg">
</picture>

**136 analysis contexts x 2047 positions = 278,392 scored positions** from a corpus
that no candidate was calibrated on (Gutenberg, arXiv, Wikipedia in 9 languages,
CPython), verified by a 160-character shingle scan against every exllamav3
calibration corpus: **0 contaminated contexts**. Exact full-vocabulary two-pass
`KL(BF16 reference || candidate)` through one shared BF16 LM head, float64
accumulation, source-cluster bootstrap.

| candidate | weights | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 0.773 | 0.004528 | 96.22 % |
| **this quant** | **19.2 GB** | **0.030736** | [0.02238, 0.04073] | 0.004218 | 1.758 | 0.010051 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 4.509 | 0.028663 | 90.53 % |

Paired over the same contexts: **-0.064242** versus NVFP4
(95 % CI [-0.08621, -0.04611], **136/136** contexts ours) and
**+0.017611** versus FP8 (95 % CI [0.01256, 0.02368], 136/136 contexts FP8).

**An earlier version of this card reported better numbers on a contaminated suite.**
The previous corpus was exllamav3's own calibration data — the text this quant was
tuned on, while the NVFP4 and FP8 candidates were calibrated elsewhere. Re-measuring
on held-out text moved ours from 0.026231 to 0.030736 (+17 %), NVFP4's from 0.073006
to 0.094978, and FP8's from 0.019309 to 0.013126 (-32 %). These are the honest
numbers; the correction is documented in the companion repo.

Controls shipped with the dataset: runtime-repeat noise floor **0.000000** across
three captures of the same runtime (this runtime is bit-deterministic, so every
difference above is far outside noise); harness self-check 0.000000; CUDA-graph
parity 0.000000. **Replay qualification is the weak link at 6.54e-04** mean
`KL(live || replayed)` — 2 % of this candidate's KLD and 4 % of the gap to FP8, so no
ranking depends on it, but differences below ~1e-3 are not resolvable with these
artifacts.

### Head attribution### Head attribution: the K6 `lm_head` is nearly free

Replaying the identical stored hidden states through the BF16 head and through the
reconstructed K6 head (exllamav3's own `reconstruct_had_slice`, so it is the exact
serving matrix) isolates head error from body error:

| configuration | mean KLD | top-1 |
|---|---:|---:|
| head error alone (BF16 body, BF16 head vs K6 head) | 0.000367 | 99.31 % |
| body only (K4 body, same head both sides) | 0.026231 | 96.03 % |
| end to end, as served (K4 body + K6 head) | 0.026299 | 95.97 % |

The K6 head adds **6.78e-05** on top of the body
(95 % CI [4.63e-05, 9.01e-05]), i.e.
**0.26 % of total divergence**. Contrary to the common
assumption that `lm_head` is highly quantization-sensitive, at 6 bits on this model
it is not worth spending 1.6 GB to promote it to BF16 — that budget belongs to the
MLP stack, which owns the rest of the error.

### Single-window KLD, v1 protocol (kept for continuity)

This was the first measurement; the v2 protocol above supersedes it.

#### Teacher-forced KLD, full vocabulary

One frozen 2048-token window (exllamav3's bundled `wiki.utf8`, first 2048 tokens),
2047 scored positions, `KL(BF16 teacher || candidate)` across the entire
248320-token vocabulary with no top-k, 3 repeats, `--kv-cache-dtype auto` pinned
for every candidate, same teacher logits file for all of them. Protocol and
statistics follow the published Gilded Gnosis harness
(`rtx6kpro:scripts/glm52_exl3_shared_h_kld.py`): the headline value is the mean of
the per-run means and `run SD` is the sample SD across those means.

| candidate | mean KLD | run SD | SD across positions | resident weights |
|---|---:|---:|---:|---:|
| **this quant** (`--quantization exl3` + `exl3-b6` overlay) | **0.034030** | 0.000000 | 0.4628 | 19.21 GB |
| `unsloth/Qwen3.8-27B-NVFP4` control, same generation | **0.091457** | 0.000000 | 0.8036 | 23.42 GB |

**This quant is 2.7x closer to the BF16 teacher than the same-generation NVFP4
checkpoint, while holding 4.2 GB less VRAM.** That is the whole point of the
recipe: Trellis K4 spends 4.004 bpw where NVFP4 spends 4.50, and the savings buy
K6 attention instead of FP8.

`run SD = 0` for both candidates means the three repeats were bit-identical —
expected for the eager, `max_num_seqs=1`, prefix-caching-disabled configuration,
and a useful signal that the online-K6 cache reloads deterministically.

For scale, this project uses project-local, unvalidated descriptors (`<0.01` near-lossless,
`0.01-0.05` good, `0.05-0.1` noticeable, `>0.1` significant); they are not an external standard
and do not transfer across models, corpora or tokenizers. This quant sits in the "good" band; the
NVFP4 control sits in "noticeable".

Still measuring on the same window and teacher: this checkpoint, overlay off (attention stays BF16 in VRAM).

### Throughput — with CUDA graphs

Same GPU, `--max-num-seqs 8`, greedy, `ignore_eos`, 256 output tokens, warmup discarded.

| configuration | C1 tok/s | C4 tok/s | C8 tok/s |
|---|---:|---:|---:|
| **this quant + CUDA graphs** | **55.39** | **190.59** | **428.12** |
| `unsloth/Qwen3.8-27B-NVFP4` (Cutlass FP4 + graphs) | 49.09 | 171.78 | 371.06 |
| this quant, eager | 28.77 | 103.47 | 215.84 |
| `Qwen/Qwen3.8-27B` BF16 + graphs | 27.47 | 101.04 | 208.31 |
| `Qwen/Qwen3.8-27B` BF16, eager | 25.50 | 92.72 | 186.98 |

Graphs are worth **+92 % / +84 % / +98 %** here — roughly nine times what they buy the
BF16 model (+8-11 %), because eager EXL3 pays per-call dispatch on 193 quantized
matmuls. With graphs this quant is **both the smallest and the fastest** option
measured, and distribution parity against eager is exact (KLD 0.000000, top-1
1.000000 over 32 contexts).

Graphs need the patch in
[local-inference-lab/vllm#312](https://github.com/local-inference-lab/vllm/pull/312)'s
sibling (autotune priming, filed separately) plus:

```bash
-e VLLM_EXL3_GRAPH_DECODE=1 ... --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Without that patch the loader refuses non-eager execution and you must pass
`--enforce-eager`, which costs the throughput above.

## Tradeoffs, stated plainly

- **The download is 28.31 GB for a 19.21 GB resident model.** Attention ships
  BF16 so the runtime can encode it at K6 (and, later, at another width) instead
  of being locked to a serialized choice. If you want download == VRAM, the
  `v-serialized-k6` variant is the one to ask for.
- **CUDA graphs need a patched loader** (see the throughput section). Unpatched, the
  loader refuses non-eager execution and you lose 46-50 % of decode throughput
  ([local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311)
  tracks the surrounding overlay work; the graph guard itself is next on the list).
  Decode is 58-60 % of the NVFP4 checkpoint's, dominated by the GEMM kernel rather
  than by graphs.
- **First load pays the K6 encode** (~16 min here) unless the cache directory is
  warm.
- **One runtime.** This checkpoint does not load in upstream vLLM, SGLang,
  transformers, TensorRT-LLM or llama.cpp. `exllamav3` itself can read the
  serialized K4/K6 halves, but it will not perform the runtime K6 encode of the
  BF16 attention.
- **KV cache** is left at engine default (`auto`); both NVFP4 references quietly
  ship FP8 KV schemes. Pin `--kv-cache-dtype` explicitly if you are comparing.

## Reproducing this quant

```bash
# 1. Convert everything at K4 (vision left BF16, head K6, mcg codebook).
python convert.py -i Qwen3.8-27B -o qwen38-k4 -w wd-k4 \
  -b 4 -hb 6 -mb 4 -vb 16 -cb mcg -d 0        # exllamav3 1.4.2 @ 5f3c537

# 2. Splice BF16 attention + MTP back over the K4 output; the converter cannot
#    emit BF16 for a decoder linear (load_fp16 forces float2half).
python splice_bf16_attn.py -q qwen38-k4 -s Qwen3.8-27B -o Qwen3.8-27B-K4

# 3. Regenerate metadata so tensor_storage describes the mix.
python util/add_safetensors_index.py -m Qwen3.8-27B-K4 --force
python util/add_quant_config.py -m Qwen3.8-27B-K4
```

`splice_bf16_attn.py`, the container-free runner used for all measurements here,
and the KLD harness are in the companion repo listed below.

## Prior art and credits

- [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — the base model
  (Apache-2.0). Architecture, chat template, tokenizer and generation defaults
  are theirs.
- [`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) —
  the recipe this one is modelled on: 4-bit MLP, 8-bit attention, BF16
  embeddings/vision/MTP. Built with NVIDIA TensorRT Model Optimizer.
- [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
  — second, independent confirmation of the same protection pattern, plus the
  last-8-layer MLP protection idea that the next iteration adopts.
- [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and the
  [vLLM recipe page](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — serving
  reference for context length, MTP and thinking modes.
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3
  Trellis format, encoder and conversion pipeline (1.4.2 @ `5f3c537`).
- [Gilded Gnosis r34](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md)
  — the runtime that serves this: mixed-K EXL3 loader, B12X Trellis kernels, and
  the `exl3-b6` online overlay whose GLM-5.2 shared-expert pattern this quant
  borrows for dense attention. The K6-cache and KLD protocol documented there is
  what this quant is measured against.
- [malaiwah/progressive-tensors](https://github.com/malaiwah/progressive-tensors)
  — per-expert EXL3 segment provenance work; source of the per-bit error ladder
  that motivated K6 for the protected tensors.

## Companion repository

Recipe derivations, the measured composition of both NVFP4 references, the
runtime contract, the toolchain gaps, the KLD protocol and the iteration log:
**<https://github.com/malaiwah/qwen38-27b-exl3>**.

Successor checkpoints are published as separate repositories, each with its own
measurement receipts (`build-receipt.json`, `SHA256SUMS`, `quantization_manifest.json`).
