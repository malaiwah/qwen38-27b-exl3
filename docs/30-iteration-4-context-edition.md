# Iteration 4: the context edition, and two kernels on the wrong side of a threshold

> **Later result.** This document records the pre-overlay iteration. Per-row int8
> input embeddings subsequently reduced resident weights, and the physical RTX
> 5090 qualification then passed native 262,144 with MTP-3, the 8.4 MP ceiling,
> and `--gpu-memory-utilization 0.955`. See
> [docs/32](32-native-context-embedding-overlay.md),
> [docs/29](29-plan-and-loose-ends.md), and
> `receipts/qualification-5090-context.json`.

Goal for this iteration, from [docs/29](29-plan-and-loose-ends.md): a build that reaches
native context on a 32 GB card while still beating official FP8. That goal was **not** met,
and the arithmetic that says why is now measured rather than modelled. Four other things were
found on the way, three of them worth shipping.

## The build

`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`: attention serialized at **K5** instead of K6, MLP and
head unchanged. 20.70 GB on disk, **19.31 GiB resident** (19.56 with MTP-3), 1,199 logical
tensors verified against upstream.

| | value | against |
|---|---:|---|
| mean KLD, body-only | **0.009673** | FP8 0.013126, hydrated 0.007406 |
| paired vs official FP8 | **−0.003453**, 135/136 | 26 % lower at 69 % of its resident weight |
| paired vs hydrated | +0.002266, 1/136 | what K5 attention costs |
| as served (own K6 head) | 0.009795 | head costs +0.000122, CI [+0.000103, +0.000142] |

**Calibration beats the runtime overlay a second time.** Serialized K5 attention measures
0.009673 where the same family encoded to K5 *at load* measures 0.012135 — 20 % better at the
same bit width, mirroring the 9.2 % the hydrated build won at K6. Offline calibration is worth
roughly a fifth of the attention error budget.

## Native 262,144 on 32 GB: not reachable, and here is the exact gap

Measured on a budget of 30.44 GiB, which is what a 5090 gives vLLM at utilisation 0.97, with
`--max-num-seqs 4`, fp8 KV, CUDA graphs and vision enabled:

| MTP depth | KV needed at 262,144 | max length that starts | KV allocated |
|---|---:|---:|---:|
| off | 8.18 GiB | 229,376 | 240,080 tokens |
| 1 | 8.83 GiB | — | — |
| 2 | 8.98 GiB | — | — |
| 3 | 9.13 GiB | 196,608 | 205,346 tokens |

- **MTP's KV cost is almost entirely fixed**: the draft layer's own cache is +0.65 GiB, and
  depth 1 → 3 adds only 0.30 GiB. If you pay for speculative decoding at all, take depth 3.
- **The gap is 0.63 GiB** (8.18 needed against 7.55 available with multimodal profiling on).
  Smaller prefill chunks do not close it — 1024 and 512 tokens both leave availability at
  7.54-7.55 GiB, because the activation peak is the vision profile, not the text chunk.
- The only remaining lever large enough is the **embedding table**: 2.543 GB of BF16, which at
  FP8 would free 1.19 GiB. exllamav3 quantizes `Linear`, not `Embedding`, and the runtime has
  no quantized-embedding path, so this is now the single blocking item for native context on a
  32 GB card. It is a loader feature, not a recipe change.

## Long context, verified by generation rather than allocation

9/9 exact needle retrievals, three depths at each length, on the 5090-sized budget:
28,613 tokens (6.3 s), 113,345 (34.3 s), **196,857 (76.1 s)**
([`receipts/needle-32768-0.1.json`](../receipts/needle-32768-0.1.json),
[`-0.5`](../receipts/needle-32768-0.5.json), [`-0.9`](../receipts/needle-32768-0.9.json);
[`receipts/needle-131072-0.1.json`](../receipts/needle-131072-0.1.json),
[`-0.5`](../receipts/needle-131072-0.5.json), [`-0.9`](../receipts/needle-131072-0.9.json);
[`receipts/needle-225000-0.1.json`](../receipts/needle-225000-0.1.json),
[`-0.5`](../receipts/needle-225000-0.5.json), [`-0.9`](../receipts/needle-225000-0.9.json)). Prompt tokens divided by
total request wall time fall from ~4,500 to 2,588 tok/s as length grows. Those quotients
include decode and HTTP overhead; they are not engine-timed prefill measurements and do not
by themselves attribute the decline.

Two harness defects had to be fixed before those numbers meant anything, and both would have
produced a *false pass*:

1. **The chat endpoint truncates long text to ~2,048 tokens on this VLM.** A 32,768-token
   request returned `prompt_tokens: 1909`. Same defect family as issue #313. The harness now
   applies Qwen's template by hand and posts to `/completions`.
2. A corpus path that did not exist on the host silently degraded into a prompt of newlines,
   which tokenises to a fraction of the requested length. `load_filler` now refuses an empty
   corpus instead of padding.

## Two kernels were on the wrong side of a row-count threshold

**B12X.** The serialized EXL3 path sends every K6/MCG shard with 128-divisible dimensions to
B12X's native kernel *before* the reconstruct dispatch from PR #316 is consulted. On this build
that is 208 attention projections, 64 `down_proj` and the head — the majority of the model —
running a decode kernel through prefill. Measured per matrix at m=2048, B12X against
reconstruct+GEMM ([`receipts/b12x-vs-reconstruct.json`](../receipts/b12x-vs-reconstruct.json)):
`down_proj` 1.11x, `lm_head` 1.40x, attention `in_proj_qkvz` 1.08x in favour
of reconstruct; at m≤8 B12X wins by ~5x. Routing by row count inside the opaque op gains
**+3.4 % prefill** (5,078 → 5,250 tok/s) and costs **+0.0000377 mean KLD**, CI
[−0.00001, +0.00009], 59/136 contexts — a coin flip. Shipped.

**FP8 prefill: measured and rejected.** Emitting the reconstructed weight straight to E4M3 from
the kernel and using `torch._scaled_mm` is **+31 % prefill (5,078 → 6,650 tok/s)** and costs
**+0.0141 mean KLD**, landing at 0.0237 — worse than official FP8. Row-wise scaling did not
help (+0.0141 either way), so the loss is FP8 *activations*, not scale granularity. Off by
default behind `VLLM_EXL3_PREFILL_FP8=1`.

Two traps found inside that work, both worth knowing:

- The Python-only FP8 route is **2-4x slower** than fp16, because `_scaled_mm` needs a
  column-major weight and transposing a reconstructed buffer costs more than the GEMM saves.
  That is why the FP8 store had to go inside the reconstruct kernel.
- `torch._scaled_mm` with **row-wise scales and `out_dtype=float16` returns silently wrong
  results** — no error, ~6x relative error, and in serving it produced KLD 10.8 with 0.1 %
  top-1. With `out_dtype=bfloat16` it matches an fp32 reference to 0.040, the same as
  per-tensor. Worth reporting upstream.

## Multimodal, measured for the first time

30 deterministic synthetic images with exactly known answers, scored by exact match: 6-digit
bitmap codes **8/10**, tallest-bar-by-index **9/10**, colour-grid counting **7/10**,
**24/30 overall**. The generator is published so any candidate can be scored on the same cases.
All tested builds retain a BF16 vision tower, but their quantized language bodies
can still change multimodal answers. This 24/30 result is a baseline for a paired
comparison, not evidence that quantization is irrelevant to multimodal quality.

Also measured and abandoned: **quantizing the vision tower** (`-vb 6`) saves 0.58 GB but the
converter splits upstream's fused `visual.blocks.N.attn.qkv` into q/k/v, so the checkpoint no
longer matches the architecture's weight names — and the loader excludes vision by design
anyway. The finalizer's upstream topology check caught it before publication, which is exactly
what it was hardened for.

## Loose ends this iteration created

- The B12X prefill routing was filed upstream as
  [PR #318](https://github.com/local-inference-lab/vllm/pull/318).
- `torch._scaled_mm` row-wise + fp16 silent corruption deserves an upstream report.
- The `reconstruct_fp8_slice` kernel (bit-exact, per-column scales) is useful even though the
  serving path rejects FP8 activations: it is the right primitive if a future path keeps
  activations in fp16.
- exllamav3's `quantize_side_model` crashes with `AttributeError` when a side-model Linear is
  consumed without being rebuilt; patched locally to name the module instead.
