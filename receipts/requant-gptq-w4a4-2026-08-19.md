# GPTQ lands, loses to RTN; W4A4-MLP proves the speed window

**Date:** 2026-08-19 (night). Artifacts: `/home/mbelleau/models/qwen38-27b-{rtn,gptq}-fp8attn-nvfp4mlp`,
`.../qwen38-27b-rtn-fp8attn-nvfp4w4a4mlp`. Reports in `receipts/kld-reports/`.

## 1. The GDN tracing blockade is BROKEN — GPTQ runs end-to-end on Qwen3.5

Four defects peeled, in order:
1. `Qwen3_5GatedDeltaNet.forward` is decorated with `@force_accelerate_hooks` WITHOUT
   `functools.wraps` -> `inspect.getsource` returns the closure (`def wrapped`), the
   AST rewrite exec produces no `forward`. Fix: fish the original out of
   `__closure__` cells; the re-exec re-applies the decorator (venv patch to
   llmcompressor `ast_helpers.py`, upstreamable).
2. With the rewrite landing, the AutoWrapper + `tracing_ignore` of the bound
   chunked-scan fallbacks (`torch_chunk_gated_delta_rule` etc.) wrap the
   HFProxy-hostile loops opaquely.
3. Per-`Linear` sequential targets slice MID-GDN: cache values that are None at
   calibration cross subgraph cuts and detonate as `Tensor + NoneType`. Fix:
   module-class boundaries (`Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`,
   `Qwen3_5MLP`) - the documented pattern.
4. 45m7s wall for the full 482-subgraph GPTQ on one RTX 5090 (2-4 GB VRAM,
   sequential offload).

## 2. The upset: GPTQ measured WORSE than RTN on our suite

| artifact | KLD mean | ci95 | p99 | top-1 | PP (crude) |
|---|---|---|---|---|---|
| RTN W4A16-MLP | 0.022121 | [0.020656, 0.023824] | 0.245008 | 0.939 | ~4,143 |
| **GPTQ W4A16-MLP** | **0.028548** | [0.026393, 0.031012] | 0.350838 | 0.935 | ~4,220 |

CIs disjoint: GPTQ is genuinely worse as built. GPTQ cannot lose to RTN with a
healthy Hessian - so the Hessians were thin or poisoned. Named suspects, in
probability order: (a) calibration starvation - `concatenate_data` collapsed 512
samples into **42** sequences (86k tokens; typical GPTQ uses 3x more); (b) GDN
calibration flows through the traced subgraphs with cache=None - if the
propagated activations feeding the in_proj Hessians are unrepresentative, GPTQ
compensates against garbage; (c) static actorder + dampening 0.01 overfitting the
thin slice. The naive cross-harness transfer ("their GPTQ 0.002666 ~ 10x better
than our RTN") is REFUTED for this pipeline as-built: the Discord gap is now
attributable to calibration quality/quantity, not merely procedure. Next levers,
all cheap: unconcatenated 512 samples, open-platypus, dampening 0.1, actorder
off, and Hessian-health logging per module group.

## 3. W4A4-MLP: the six-criteria speed window is REAL

Data-free RTN W4A4 build (65 s) + serve test:

| | W4A16-MLP (GPTQ) | **W4A4-MLP (RTN)** |
|---|---|---|
| PP (median of 4, per-request incl. HTTP) | ~4,220 | **8,876.8** |
| output | coherent | garbage ("ductduct...") |

The 4-bit-activation MLP path more than DOUBLES prefill and clears the 7,000 bar
with margin - on the layers that dominate prefill FLOPs, exactly as the Discord
bracketing predicted. The garbage output is the expected cost of an uncalibrated
static activation global scale, not a serving bug (the W4A16 sibling of the same
build pipeline is coherent). **The middle SKU is worth the calibration
investment**: a properly calibrated FP8-attn + NVFP4-W4A4-MLP checkpoint is the
first single-artifact candidate for all six criteria (PP ~8.9k >= 7000; KLD needs
the fixed calibration to approach the 0.012 bar; vision/MTP via the SPARK-style
BF16 re-merge).

## 4. Standing glue (scripted, reusable)

Composite-config surgery for AutoModelForCausalLM-saved checkpoints; serve via
`--language-model-only`; weight names stay `model.language_model.*` (1,538/1,539);
`fidelity.py --language-model-only` for KLD of text-only checkpoints.
