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
  - long-context
  - gilded-gnosis
---

# Qwen3.8-27B EXL3 K5/K6 context edition — 196,857-token retrieval verified on a 32 GB budget, at 26 % lower divergence than official FP8

> **Requires a custom runtime.** Does **not** load in upstream vLLM, SGLang, TensorRT-LLM,
> llama.cpp, transformers, or stock exllamav3. It needs the Gilded Gnosis vLLM fork with
> explicit `--quantization exl3` and an exact `ignore` list. Treat it as an experimental,
> runtime-specific research artifact.

The long-context member of the family: attention is serialized at **K5** on disk, which frees
0.85 GiB against the K6 builds and buys about 16k tokens of context, while still measuring
**below official FP8**. It is the only build here whose long context is verified by
generation rather than by allocation.

## Which of the four builds

Same architecture, same tokenizer, same held-out suite — they differ in where the bits go.
Contexts are what the engine serves on a 32 GB card with MTP-3, vision enabled, at
utilisation 0.97 ([collection](https://huggingface.co/collections/qwen38-27b-mixed-precision-exl3-measured-6a7fe0cb27817c23e4a57025)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-frontier-dark.svg">
  <img alt="Mean KL divergence versus context served on a 32 GB card. Hydrated 0.007406 at 180k, online K6 0.008157 at 180k, context edition 0.009673 at 196k with needle verification, online K5 0.012135 at 196k, K4 0.030736 at 262k. Official FP8 is 0.013126." src="assets/context-frontier-light.svg">
</picture>

| build | download | resident | mean KLD | context on 32 GB | pick it when |
|---|---:|---:|---:|---:|---|
| [-hydrated](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | 21.61 GB | 20.31 GiB | **0.007406** | ~180k | fidelity first |
| [-EXL3-K5K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6) | 30.57 GB | 20.32 GiB | 0.008157 | ~180k | you want the width knob at launch |
| **this build** | 20.70 GB | **19.56 GiB** | 0.009673 | **196,608** | long context that still beats FP8 |
| [-K4](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) | 28.31 GB | 17.89 GiB | 0.030736 | **262,144** | native context is non-negotiable |

## Recipe

| role | representation |
|---|---|
| MLP `gate_proj`, `up_proj` (64 layers) | EXL3 **K5**, `mcg` |
| MLP `down_proj` (64 layers) | EXL3 **K6**, `mcg` |
| attention: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48, `self_attn.{q,k,v,o}_proj` ×16 | **EXL3 K5 on disk**, `mcg`, calibrated |
| `lm_head` | EXL3 **K6**, `mcg` |
| MTP draft head | quantized (`fc` + attention K4, MLP K5/K6) |
| `embed_tokens`, vision tower (27 blocks), norms | BF16 |
| GatedDeltaNet `in_proj_a` / `in_proj_b` (96) | FP16 passthrough |

Built from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
@ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Verified after the build: reconstructing every
EXL3 module yields exactly upstream's **1,199 logical tensor names with matching shapes**; the
finalizer fails closed otherwise.

**The vision tower stays BF16 for a measured reason.** Quantizing it (`-vb 6`) saves 0.58 GB
but the converter splits upstream's fused `visual.blocks.N.attn.qkv` into separate q/k/v
projections, so the checkpoint stops matching the architecture's weight names — and the EXL3
loader excludes vision by design anyway. The topology check caught it; the build was redone.

## Fidelity

Held-out corpus, 136 analysis contexts, 278,392 full-vocabulary scored positions,
`KL(BF16 ‖ candidate)` with one shared BF16 head for both operands, source-cluster bootstrap.

| candidate | resident | mean KLD | 95 % CI | median | top-1 |
|---|---:|---:|---|---:|---:|
| **this build** | 19.56 GiB | **0.009673** | [0.00711, 0.01275] | 0.001929 | 96.81 % |
| hydrated (attention K6) | 20.31 GiB | 0.007406 | [0.00543, 0.00978] | 0.001335 | 97.19 % |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 96.22 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 21.34 GiB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 90.53 % |

Paired on identical contexts:

- versus official FP8: **−0.003453**, 95 % CI [−0.004383, −0.002666], **135/136 contexts** —
  26 % lower divergence at 69 % of its resident weight.
- versus the hydrated build: **+0.002266** (1/136). That is what K5 attention costs, and it
  buys 0.85 GiB and ~16k tokens of context.
- **Calibration beats the runtime overlay again:** serialized K5 attention measures 0.009673
  where the same checkpoint family encoded to K5 *at load* measures 0.012135 — 20 % better for
  the same bit width.
- As served with its own K6 head: **0.009795** (head costs +0.000122, 95 % CI [+0.000103,
  +0.000142], 13/136), still 25 % below FP8's body-only figure.

## Long context, verified by generation

Not an allocation test. A unique code is planted in held-out literary text at three depths and
the model is asked to return it, on a server capped to a 5090-sized budget:

| prompt tokens | needle depth | retrieved exactly | wall | prefill |
|---:|---|---|---:|---:|
| 28,613 | 0.1 / 0.5 / 0.9 | **3/3** | 6.3 s | ~4,500 tok/s |
| 113,345 | 0.1 / 0.5 / 0.9 | **3/3** | 34.3 s | 3,301 tok/s |
| **196,857** | 0.1 / 0.5 / 0.9 | **3/3** | 76.1 s | 2,588 tok/s |

**9/9 exact retrievals.** Prefill throughput falls with length as attention cost grows, which
is expected and now quantified.

**Long-context requests must not use the chat endpoint on this VLM.** Its multimodal processor
truncates chat text to about 2,048 tokens — a 32,768-token request came back with
`prompt_tokens: 1909` — the same defect family as
[#313](https://github.com/local-inference-lab/vllm/issues/313). Apply the template yourself and
post to `/completions`:

```text
<|im_start|>user
{your long prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Memory: what fits on a 32 GB card

Measured with the engine budget capped to 30.44 GiB, which is what a 5090 gives vLLM at
utilisation 0.97, `--max-num-seqs 4`, fp8 KV, CUDA graphs, vision enabled.

| MTP draft depth | KV needed at 262,144 | max `--max-model-len` that starts | KV allocated |
|---|---:|---:|---:|
| off | 8.18 GiB | **229,376** | 240,080 tokens |
| 1 | 8.83 GiB | — | — |
| 2 | 8.98 GiB | — | — |
| 3 | 9.13 GiB | **196,608** | 205,346 tokens |

Two things worth knowing, both measured here for the first time in this family:

- **MTP's KV cost is nearly all fixed.** The draft layer's own cache is +0.65 GiB; going from
  depth 1 to depth 3 adds only 0.30 GiB more. If you are paying for speculative decoding at
  all, pay for depth 3.
- **Native 262,144 does not fit any ~19-20 GiB build on a 32 GB card**, with or without MTP:
  8.18 GiB of KV against 7.55 GiB available. Smaller prefill chunks do not help — the
  activation peak is multimodal profiling, not text chunking. Only the
  [K4 build](https://huggingface.co/malaiwah/Qwen3.8-27B-K4) reaches native context there, at
  3.2x this build's divergence. Closing the last 0.63 GiB needs the embedding table quantized
  (2.543 GB BF16), which the runtime does not support today.

## Throughput

Median of 3 runs on one RTX PRO 6000 Blackwell, `--max-num-seqs 8`, greedy, 256 output tokens.

| configuration | TG C1 | TG C4 | TG C8 | PP 2k | PP 6k |
|---|---:|---:|---:|---:|---:|
| B12X everywhere (as published upstream) | 56.0 | 197.2 | 397.3 | 5,078 | 5,188 |
| **+ prefill routing (shipped here)** | 56.0 | 197.0 | 398.8 | **5,250** | 5,249 |
| + FP8 prefill (**rejected**, see below) | 56.7 | 199.5 | 401.6 | 6,650 | 6,285 |

**A whole class of matrices was on the wrong kernel at prefill.** The serialized EXL3 path
routes every K6/MCG shard with 128-divisible dimensions to B12X's native kernel *before* the
reconstruct dispatch is consulted — on this build that is 208 attention projections, 64
`down_proj` and the head. B12X is the right choice at decode (measured ~5x faster at m≤8) and
the wrong one at prefill (reconstruct+GEMM wins 1.08-1.40x at m=2048). Routing by row count
inside the opaque op gains **+3.4 % prefill and costs +0.0000377 mean KLD** (95 % CI
[−0.00001, +0.00009], 59/136 — a coin flip, i.e. free).

**FP8 prefill is measured and rejected.** Emitting the reconstructed weight directly as E4M3
and using `torch._scaled_mm` gives **+31 % prefill (5,078 → 6,650)** but costs **+0.0141 mean
KLD**, which lands this build at 0.0237 — worse than official FP8. Row-wise scaling (per-token
activation, per-channel weight) did not help: the loss is FP8 *activations*, not scale
granularity. It stays off; `VLLM_EXL3_PREFILL_FP8=1` enables it for anyone who wants prefill
over fidelity.

## Multimodal

First quantitative multimodal check in this family, on deterministic synthetic images with
exactly known answers, scored by exact match (30 cases, greedy, thinking disabled):

| task | what it tests | score |
|---|---|---:|
| 6-digit code, 5×7 bitmap font | fine detail | **8/10** |
| bar chart, tallest bar by index | spatial ordering | **9/10** |
| colour grid, count of one colour | counting and localisation | **7/10** |
| overall | | **24/30 (80 %)** |

The generator is published (`tools/vision_eval.py`) so the same cases can be run against any
candidate. This build and its siblings all carry a BF16 vision tower, so these numbers
characterise the model rather than the quantization — they are the baseline for a future
paired comparison.

## Serving

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 8000:8000 \
  -v /models:/models:ro \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6-context \
    --served-model-name qwen38 --quantization exl3 --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","lm_head"]}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 196608 --gpu-memory-utilization 0.97 --max-num-seqs 4 \
    --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 --port 8000
```

Nothing is encoded at load: no `VLLM_EXL3_ONLINE_TRELLIS_BITS`, no cache directory. The command
above is what the pinned image runs **unmodified**, so it is eager-only. CUDA-graph decode
(+46-50 %) and the prefill routing above need patches still open upstream
([#314](https://github.com/local-inference-lab/vllm/pull/314),
[#316](https://github.com/local-inference-lab/vllm/pull/316)); the sibling card carries the
exact patch recipe with its sha256.

Load-bearing details, unchanged from the siblings: `--quantization exl3` is mandatory; the
`ignore` list is mandatory and its anchoring is subtle (`re:.*visual\..*` matches,
`re:.*\.visual\..*` silently does not and **crashes** startup,
[#311](https://github.com/local-inference-lab/vllm/issues/311)); `mtp.*` must **not** be
ignored; `--mm-processor-kwargs '{"truncation":false}'` is required for large images
([#313](https://github.com/local-inference-lab/vllm/issues/313)).

## What is not verified

No downstream task benchmarks, no OCR/chart/document evaluation beyond the synthetic set above,
no video, no YaRN-1M, no multi-GPU or TP>1, no non-SM120 GPU, and no quant-specific safety
regression testing. The needle test proves retrieval, not reasoning, at length.

Recipe development used the 136-context analysis partition, so these are development-set
numbers. A source-disjoint v4 suite (160 contexts, 100 clusters, zero overlap with v3 on
document hashes and token hashes) is built and awaiting a single frozen qualification run.

Machine-readable evidence: `release-evidence.json`. Captures and per-context reports for the
family are published in the
[fidelity dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3).

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — EXL3 Trellis format,
  LDLQ calibration, MCG codebook, conversion pipeline.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path and the B12X native
  K6 kernel this build routes around at prefill.
- [Qwen](https://huggingface.co/Qwen) — base model and official FP8 derivative.
- An independent RTX 5090 tester — the memory model these context numbers are calibrated on.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
