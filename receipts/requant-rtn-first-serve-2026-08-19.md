# Mixed FP8-attn + NVFP4-MLP: first checkpoint built AND served, same evening

**Date:** 2026-08-19 (night). Checkpoint: `/home/mbelleau/models/qwen38-27b-rtn-fp8attn-nvfp4mlp` (21 GB).

## What was proven tonight

The Discord leaderboard's winning scheme SHAPE — FP8 W8A16 on attention+GDN, NVFP4
W4A16 g16 on all 64 MLP layers — was reproduced as a compressed-tensors checkpoint
via llmcompressor 0.13.0 and **served on our fork's stock compressed-tensors path**:

| check | result |
|---|---|
| boot (`--language-model-only`, 32k, GMU 0.90) | **healthy** |
| sanity | " Paris." |
| PP (crude warm single-request, incl. HTTP) | **~4,142.6 tok/s** |
| TG (no speculative decode) | 74.5 tok/s |
| GDN fused-name hazard (SPARK bug #3) | **did not bite** — vLLM stacked the unfused FP8 `in_proj_qkv/z/a/b` into `in_proj_qkvz/ba` correctly |

PP lands in the Discord table's ballpark for this scheme (4,551 on their harness) and
above our balanced profile (3,925.2) — with ZERO serving-stack tuning.

## Two honest limitations of this artifact

1. **RTN, not GPTQ.** llmcompressor's GPTQ requires fx-tracing decoder layers; the
   Qwen3.5 GDN mixer is untraceable (`HFProxy cannot be interpreted as an integer`).
   The Hessian compensation — the thing that made the Discord scheme's MLP-to-NVFP4
   step cost ~nothing — is MISSING here. KLD of this artifact is NOT a bound on the
   scheme; it is the floor. GPTQ needs a traceable model definition port
   (llmcompressor documents the pattern; scoped next).
2. **Text-only, no MTP.** llmcompressor loads via `AutoModelForCausalLM` -> vision
   tower and `mtp.*` head silently dropped (also why the transformers-5.14 re-saved
   config broke the fork's composite-config expectation; fixed by rebuilding the
   composite shell around the saved text config — `config.json.textonly-orig` keeps
   the original). MTP re-merge from BF16 is a known, proven pattern
   (SPARK_Qwen3.5-122B repo did exactly this).

## The six startup defects fixed to get here (all committed)

recipe path resolution, multimodal processor init (explicit tokenizer), dataset
REGISTRY naming (`ultrachat-200k`, not HF repo id), `dataset_config_name` misuse,
whole-layer sequential OOM (per-Linear granularity), and — for the GPTQ path — the
tracer autowrap KeyError (venv-local skip-and-warn patch) which then exposed the
real GDN untraceability.

## Queue for the fidelity version

1. Traceable Qwen3.5 definition -> GPTQ run of the same recipe (the actual Discord
   winner). 2. KLD both artifacts on the 512-ctx suite (RTN floor vs GPTQ).
3. MTP BF16 merge + spec-decode re-check. 4. W4A4-MLP GPTQ variant — the
   six-criteria middle SKU. 5. If GPTQ lands near 0.0026-equivalent on our suite at
   ~4.5k PP: it becomes the fourth profile and the balanced successor.
