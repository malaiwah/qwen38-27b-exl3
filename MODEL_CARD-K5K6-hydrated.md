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

# Qwen3.8-27B EXL3 K5/K6, attention serialized — 9 % lower divergence than the BF16-attention build, 9 GB smaller, 5x faster cold start

> **Requires a custom runtime.** Does **not** load in upstream vLLM, SGLang, TensorRT-LLM,
> llama.cpp, transformers, or stock exllamav3. It needs the Gilded Gnosis vLLM fork with
> explicit `--quantization exl3` and an exact `ignore` list. Treat it as an experimental,
> runtime-specific research artifact.

Companion to
[`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6).
Same MLP recipe; one difference: **attention is quantized to EXL3 K6 on disk by the
calibrating converter** instead of shipping BF16 and being encoded to K6 by the runtime at
every cold start. Measured consequences:

| | BF16 attention (sibling) | **this build** |
|---|---:|---:|
| download | 30.57 GB | **21.61 GB** (−29 %) |
| resident weights | 20.32 GiB | 20.31 GiB (unchanged) |
| first load, cold | 957 s (encodes 208 projections) | **178 s** (5.4x faster) |
| restart with a warm cache | 173 s | 178 s (same) |
| persistent encode cache | required, and must be writable | **not used** |
| mean KL divergence from BF16 | 0.008157 | **0.007406** (−9.2 %) |
| attention width | K6 / K5 / K4, chosen at launch | fixed at K6 |

**Choose this build** for the smaller download, the best fidelity in the family, and a start
that needs no writable cache. Note the nuance: a *warm* online cache also starts in 173 s, so
the 5.4x is about first load and about environments where a persistent cache is awkward —
read-only images, ephemeral containers, many nodes. **Choose the sibling** if you need context: its runtime knob trades
fidelity for KV room, and on a 32 GB card that difference matters (see
[context](#context-capacity)).

## Recipe

| role | representation |
|---|---|
| MLP `gate_proj`, `up_proj` (64 layers) | EXL3 **K5**, `mcg` |
| MLP `down_proj` (64 layers) | EXL3 **K6**, `mcg` |
| attention: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` ×48, `self_attn.{q,k,v,o}_proj` ×16 | **EXL3 K6 on disk**, `mcg`, calibrated (208 modules) |
| `lm_head` | EXL3 **K6**, `mcg` |
| MTP draft head | quantized (`fc` + attention K4, MLP K5/K6) |
| `embed_tokens`, vision tower (27 blocks), norms | BF16 |
| GatedDeltaNet `in_proj_a` / `in_proj_b` (96) | FP16 passthrough |

Composition, from the emitted manifest: `full_attention` 1.260 GB (64 EXL3 K6 + 32 BF16),
`linear_attention` 4.207 GB (144 EXL3 K6 + 96 FP16 + 192 BF16), `mlp_gate_proj` 3.568 GB,
`mlp_up_proj` 3.568 GB, `mlp_down_proj` 4.281 GB, `lm_head` 0.954 GB, `embed_tokens`
2.543 GB, `vision_tower` 0.921 GB, `mtp_draft` 0.283 GB, norms 0.001 GB — 21.61 GB over
three shards.

`quantization_manifest.json` and `build-receipt.json` are authoritative for composition;
`SHA256SUMS` covers the immutable payload. `config.json → quantization_config` keeps one
`bits`/`codebook` pair for loader compatibility and **cannot** describe this mixed
checkpoint.

Built from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
@ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with exllamav3
@ `5f3c537ca9d89893d771256f5c43c93656553fbb`, plus two allocator hooks upstream does not
provide: `EXL3_BITS_FIXED` (pins modules *before* budget allocation — the only way to hold
attention at K6 while the MLP keeps its own split) and `EXL3_BITS_OVERRIDE` (pins after
allocation). Both are in the
[companion repository](https://github.com/malaiwah/qwen38-27b-exl3).

**Verified after the build:** reconstructing every EXL3 module yields exactly upstream's
**1,199 logical tensor names with matching shapes**. The finalizer fails closed on any
missing, extra or mis-shaped tensor, so this is a check rather than a claim.

## Fidelity

Held-out corpus, 136 analysis contexts, 278,392 full-vocabulary scored positions.
`KL(BF16 reference ‖ candidate)`, two passes, no top-k, one shared BF16 LM head for both
operands, source-cluster bootstrap. Same suite, reference and head as every comparator.

| candidate | resident | mean KLD | bootstrap 95 % CI | median | top-1 |
|---|---:|---:|---|---:|---:|
| **this build** | 20.31 GiB | **0.007406** | [0.00543, 0.00978] | 0.001335 | **97.19 %** |
| BF16-attention sibling | 20.32 GiB | 0.008157 | [0.00607, 0.01067] | 0.001529 | 96.97 % |
| `Qwen/Qwen3.8-27B-FP8` | 28.51 GiB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 96.22 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 21.34 GiB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 90.53 % |

Paired on identical contexts:

- versus the BF16-attention sibling: **−0.000751**, 95 % CI [−0.000977, −0.000572],
  **124/136 contexts**. Calibrated offline encoding beats the runtime's calibration-free
  online encoding, and the interval excludes zero.
- versus official FP8: **−0.005719**, 95 % CI [−0.007323, −0.004353], **136/136 contexts** —
  44 % lower mean divergence at 71 % of its resident weight.

**Body-only, stated plainly:** every row replays both operands through one shared BF16 head,
so no candidate's own head quantization is counted — that is what makes the ranking fair,
since official FP8 serves a BF16 head. On the sibling, the same K6 head was measured with
asymmetric heads at **+0.000127** (95 % CI [+0.000105, +0.000148]), so expect an as-served
figure near 0.00753 here.

**Weakest control:** live-versus-replayed logit qualification is 6.54e-04 on this harness,
so differences below ~1e-3 are not resolvable. The −0.000751 offline-versus-online gap sits
**at that floor**: the bootstrap interval excludes zero and 124/136 contexts agree on
direction, but treat the magnitude as a point estimate. The FP8 gap, at 7.6x the floor, is
not in question.

**Development-set caveat:** the recipe was chosen with these 136 contexts visible, and the
suite's source clusters are shared with the partition that guided selection. A
source-disjoint qualification suite is being built to test the frozen recipe once.

## Context capacity

This build carries the same resident weights as the sibling at attention K6 (20.31 vs
20.32 GiB), so its context behaviour is the sibling's K6 arm, which an independent tester
measured on a real **RTX 5090** (31.39 GiB usable, TP1, FP8 E4M3 KV, MTP-3, decode-only CUDA
graphs, vision enabled):

| configuration | KV attained | context | outcome |
|---|---:|---:|---|
| K6 attention, seqs 8, util 0.95 | 187,050 tok / 6.71 GiB | **185,600 run** | stable, multimodal-safe |
| K6 attention, seqs 8, util 0.98 | 202,185 tok / 7.55 GiB | — | text fine, **a 3,264-token image OOMed** with 33 MiB free |

**Native 262,144 does not fit any 20 GiB-class build on a 32 GB card.** With MTP-3 the engine
needs **9.13 GiB** of KV for native context (37.4 KB/token: 16 full-attention layers, 4 KV
heads, head_dim 256; the other 48 layers are Gated DeltaNet and hold per-sequence state).
Without MTP that falls to 33.5 KB/token and buys ~11 % more length. For native context on
32 GB, the smaller [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4)
build is the only one in this family that fits (289,577-token capacity measured), at 3.8x the
divergence. On 48 GB and larger, native context fits here at the best fidelity.

Keep utilisation at **0.95** if you serve images: 0.98 leaves no vision headroom.

Numbers measured locally at a simulated 31.2 GiB budget (196,608 starts with 246,903 KV
tokens; 262,144 refused) agree with the hardware within a few percent, but the hardware
numbers above are the ones to trust — the simulation initially omitted MTP's KV and
overstated usable VRAM, which produced an overclaim that the tester correctly rejected.

## Serving

```bash
docker run --rm --gpus '"device=0"' --ipc host -p 8000:8000 \
  -v /models:/models:ro \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b \
  serve /models/Qwen3.8-27B-EXL3-K5K6-hydrated \
    --served-model-name qwen38 --quantization exl3 --enforce-eager \
    --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","lm_head"]}' \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 8 \
    --host 0.0.0.0 --port 8000
```

No `VLLM_EXL3_ONLINE_TRELLIS_BITS`, no `VLLM_EXL3_ONLINE_CACHE_DIR`: nothing is encoded at
load, which is the point of this build. Three notes carry over from the sibling:

1. **`--quantization exl3` is mandatory** — auto-detection only fires for GLM-5.2 metadata.
2. **The `ignore` list is mandatory and its anchoring is subtle.** Prefixes carry no leading
   `model.`, so `re:.*visual\..*` matches while `re:.*\.visual\..*` silently does not, and the
   wrong pattern **crashes** startup
   ([#311](https://github.com/local-inference-lab/vllm/issues/311), fixed by
   [PR #312](https://github.com/local-inference-lab/vllm/pull/312)). `mtp.*` must **not** be
   ignored: the draft head is quantized and belongs to the EXL3 loader.
3. **`--mm-processor-kwargs '{"truncation":false}'` is required for images** whose expanded
   token sequence exceeds 2,048, or requests fail with HTTP 400
   ([#313](https://github.com/local-inference-lab/vllm/issues/313)).

The command above is what the **pinned image runs unmodified**, so it is eager-only. CUDA-graph
decode (+46-50 %) and the prefill dispatch (+113 %) need patches still open upstream
([#314](https://github.com/local-inference-lab/vllm/pull/314),
[#316](https://github.com/local-inference-lab/vllm/pull/316)); the sibling card carries the
exact patch recipe with its sha256. There is no published image digest containing them.

## What is not verified

No downstream task benchmarks, no OCR/chart/document/video evaluation, no long-context
retrieval or perplexity **for this build**, no native-262K generation, no YaRN-1M run, no
multi-GPU or TP>1, no non-SM120 GPU, and no quant-specific safety regression testing.
Throughput was not re-measured here: this build shares the sibling's kernels and resident
footprint, so its decode and prefill figures should carry over — that is an inference, not a
measurement.

Evaluation captures for this build are being published in the
[fidelity dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3);
until they land, the
[receipts](https://github.com/malaiwah/qwen38-27b-exl3/tree/main/receipts) let a third party
check the arithmetic but not recompute the row independently.

## Prior art and credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) (Turboderp) — EXL3 Trellis format,
  LDLQ calibration, MCG codebook, the conversion pipeline.
- Gilded Gnosis vLLM fork (Josh Cartu / jcartu) — the EXL3 serving path this checkpoint
  needs, and the online-encoding overlay whose output this build replaces with a calibrated
  one.
- [Qwen](https://huggingface.co/Qwen) — the base model and its official FP8 derivative.
- An independent tester on an RTX 5090 — the context-capacity table, and catching a 262K
  overclaim that came from a memory simulation.
- Research, recipe, harness and receipts:
  [malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3).
