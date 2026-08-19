# BF16 MTP + vision merged into a requant checkpoint: decode 2.9x, vision passes

**Date:** 2026-08-19 (goal session). Tool: `tools/merge-mtp-vision.py`.
Artifact: `/home/mbelleau/models/qwen38-27b-rtn-mixed-mtp-vision` (23 GB).

## What this unblocks

Every requant artifact so far was text-only: `llmcompressor` saves via
`AutoModelForCausalLM`, silently dropping the vision tower (333 tensors) and the
MTP draft head (15 tensors). Two promotion-gate criteria — `acceptance_fox >= 0.85`
and the vision red/blue check — were therefore **unmeasurable** for the whole
requant family, not merely unmet. Both weight groups were never quantized, so
re-attaching the BF16 originals is lossless.

| check | text-only RTN mixed | **+ MTP + vision** |
|---|---|---|
| boot | ok (`--language-model-only`) | **ok, full multimodal** |
| sanity | `' Paris.'` | `' Paris.'` |
| vision red/blue | impossible (no tower) | **PASS** (`left: red / right: blue`) |
| TG fox | 74.0 tok/s (no draft head) | **215.3 tok/s (2.9x)** |
| served context | 32,768 tested | 49,152 (KV 4.54 GiB) |
| weights resident | 20.45 GiB | 22.51 GiB |

MTP-6 restores decode to within 6% of the trellis `fidelity` profile's fox rate
(215.3 vs 228.3) on a checkpoint whose KLD is 6.5x worse — decode speed rides on
the draft head, not on weight fidelity, exactly as the acceptance-coupling work
predicted.

## Two defects found and fixed in the merge path

1. **Missing processor configs.** vLLM refused the merged checkpoint with
   `OSError: Can't load image processor` — a text-only save has no
   `preprocessor_config.json` / `video_preprocessor_config.json`, and the vision
   tower cannot initialise without them. The tool now copies the multimodal
   processor configs from the base snapshot (`--base-snapshot`), so this cannot
   recur.
2. **Ignore-list coverage.** The re-attached modules must be in
   `quantization_config.ignore` or the compressed-tensors path looks for
   quantized parameters that do not exist. Six regexes added (97 -> 103 entries).
   This is the same failure the SPARK_Qwen3.5-122B deployment hit from the other
   direction, where an ignore list that missed vLLM's fused GDN names produced
   silently zero-loaded layers.

The tool refuses to leave a broken checkpoint: it verifies every tensor in the
merged index resolves inside the shard the index names (1,887 tensors, 3 shards,
all resolve) and rejects key clashes.

## The binding constraint is now context, and it is arithmetic

23 GB of weights on a 31.40 GiB card leaves **4.54 GiB** for KV, which is 49,152
tokens at the measured 39.0 MiB/1k rate. The trellis `fidelity` profile serves
238,400 because its weights are 2 GiB smaller *and* it needs no BF16 vision/MTP
resident. So the merged requant family trades ~189k of context for its
throughput profile — it cannot reach criterion 5 (>= 238,400) in this shape.

Recovering it means shrinking the re-attached weights (quantizing the vision tower
and MTP head rather than restoring them BF16) — which is exactly the tradeoff the
hydrated recipe already makes, and a separate experiment.
