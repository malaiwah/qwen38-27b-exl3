# `VLLM_EXL3_FP6_LAYER_RANGE`: precision as a continuous memory dial — and why 13 layers is not shipped

**Date:** 2026-08-19
**Added:** `VLLM_EXL3_FP4_LAYER_RANGE` / `VLLM_EXL3_FP6_LAYER_RANGE` (`"lo-hi"`,
inclusive, empty = no restriction), 16/16 unit checks in
`tools/test-layer-range.py`.
**Shipped default: unchanged.** The measured operating point it unlocks is
recorded below and deliberately **not** promoted.

## Motivation

`PROFILE=balanced` converts `mlp.gate_up_proj` to MXFP6 and wins 1.66× prefill,
but costs ~2.4 GiB and so caps context at 199,104 instead of 238,400
(`receipts/balanced-profile-2026-08-19.md`). Precision was an all-or-nothing
switch, so the only choices were "all 64 layers, lose 39k of context" or "none".
A layer-index range makes it a **dial**: convert exactly as many layers as the
memory budget allows.

## Calibration (each measurement is one boot at `MAX_MODEL_LEN=238400`)

| FP6 layers | available KV | engine's max context |
|---|---|---|
| 0 (`fidelity`) | 9.29 GiB | ≥ 238,400 |
| 13 (`0-12`) | **8.93 GiB** | **238,400** |
| 29 (`0-28`) | 8.46 GiB (implied) | 226,848 |
| 64 (all) | 7.60 GiB | 199,104 |

Cost is linear at **~0.029 GiB of KV per converted layer**, which predicted the
13-layer budget before it was tried — and the prediction held.

## The 13-layer point, measured

| | `fidelity` (shipped) | FP6 layers 0-12 |
|---|---|---|
| PP, 2051-tok | 1965.4 ± 2.1 | **2145.3** (+9.2%) |
| TG-fox | 207.7 ± 0.1 [acc 1.000] | 207.9 [acc 1.000] |
| TG-essay | 93.1 ± 0.1 [acc 0.304] | 92.2 [acc 0.298] |
| max context | 238,400 | **238,400** |
| 200k prompt | 200 OK | **200 OK** |
| vision + MTP | pass | pass |
| KLD mean | **0.003437 measured** | ~0.003891 **predicted, not measured** |

## Why it is not the default: a 0.04 GiB boot margin

At 13 layers the engine reports **8.93 GiB available against 8.89 GiB required** —
a **40 MiB** margin on a 31.40 GiB device, i.e. **0.13%**. Anything that shifts the
profiled peak by more than that (a driver update, a different CUDA context, an
allocator change, a slightly different capture set) turns a working profile into
one that **refuses to boot**. The failure would be loud but total.

`fidelity` carries 9.29 vs 8.89 = **0.40 GiB (4.5%)** of margin. Trading a 10×
reduction in boot headroom for **+9.2% prefill**, on the profile whose entire
selling point is *reliable* service at full context, is a bad trade. So the dial
ships, the operating point is documented, and the default stays where the margin
is.

Anyone who wants it, with eyes open:

```
PROFILE=balanced MAX_MODEL_LEN=238400 VLLM_EXL3_FP6_LAYER_RANGE=0-12 ./run-qwen38-27b.sh
```

Note also that its KLD is **predicted (0.003891), not measured**. The prediction
rests on a model that is validated for MLP-in-FP6 (−6.0%,
`receipts/balanced-profile-2026-08-19.md`) but it is still a prediction, and this
project's rule is that shipped profiles carry measured fidelity. That is a second,
independent reason not to promote it without a KLD run.

## What the dial is actually good for

It converts "which precision?" into "how much memory do you have?", which is the
question that actually binds on a 31.40 GiB card. Concretely it lets a future
operator target a specific context and spend whatever is left on prefill, rather
than choosing between two fixed points 39,296 tokens apart. It also applies to
FP4 (`VLLM_EXL3_FP4_LAYER_RANGE`) should a partial-FP4 point ever be wanted.

## Verification

- `tools/test-layer-range.py` → **16/16 PASS** (host-only, no GPU/torch/vLLM):
  bounds inclusive on both ends, reversed ranges normalised, bare index = single
  layer, malformed spec warns and disables rather than raising, and prefixes with
  no layer index (`lm_head`, vision tower, MTP) stay unrestricted so a range
  narrows only the decoder stack.
- Serving numbers above from `tools/bench_lib.py`, acceptance reported alongside
  every TG figure; `long_ctx_check(200000)` → `True`.
- `EXL3_PATCH_SHA256` re-pinned; `throughput`/`fidelity`/`balanced` all resolve to
  an empty range, so no existing profile changes behaviour.
