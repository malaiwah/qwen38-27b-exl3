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

# Qwen3.8-27B-K4 — EXL3 mixed-precision, NVFP4-class footprint

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
**2.7 GB smaller** resident footprint with every role at equal or higher
effective precision than the NVFP4 recipes.

Vision works. Text is coherent. See *Verification* for the receipts.

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
   [local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311).
4. **`VLLM_EXL3_ONLINE_TRELLIS_BITS=6`** is what turns the overlay from MXFP8 into
   K6. Point `VLLM_EXL3_ONLINE_CACHE_DIR` at persistent storage: the first load
   encodes 208 attention projections (~16 min on one GPU here) and later loads
   reuse the cache.

Generation defaults from upstream `generation_config.json`: `temperature 1.0`,
`top_p 0.95`, `top_k 20`. Thinking control is upstream's
`chat_template_kwargs`: `{"enable_thinking": false}` or
`{"reasoning_effort": "low"|"medium"|"high"|"xhigh"}`. The chat template,
tokenizer, preprocessor configs and vocabulary (248320, untied head) are
upstream's, unmodified.

Context: 262144 native. For ~1M, upstream's override nests under `text_config`:
`--hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'`.

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

<!-- KLD_TABLE -->

## Tradeoffs, stated plainly

- **The download is 28.31 GB for a 19.21 GB resident model.** Attention ships
  BF16 so the runtime can encode it at K6 (and, later, at another width) instead
  of being locked to a serialized choice. If you want download == VRAM, the
  `v-serialized-k6` variant is the one to ask for.
- **No CUDA graphs**, by loader requirement (see flag 2). Decode throughput is
  below what an NVFP4 checkpoint achieves on the same GPU.
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

Variants are published as branches of this repo, each with its own measurement
receipt. `main` is the recommended one.
