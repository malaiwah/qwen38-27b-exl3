# Comparison: lribeiro/Qwen3.8-27B-nvfp4-v17 (all-FP8 W8A8 GPTQ)

**Date:** 2026-08-19. Source: HF model card + `config.json` via API, read in full.

## The repo name is a misnomer

Despite `nvfp4` in the repo id, this is **not** a 4-bit artifact.
`quantization_config` reads `"weights": {"num_bits": 8}`, `format: float-quantized`,
`quant_method: compressed-tensors`. The card title is
"Qwen3.8-27B-FP8-GPTQ-v17 (Vision)". It is **uniform FP8 E4M3 W8A8 with GPTQ**
Hessian correction, per-channel weights, per-token dynamic activations.
The `nvfp4` string is legacy from an earlier phase of their 38-run sweep.

## What it is

| Field | Value |
|---|---|
| Format | FP8 E4M3 W8A8 (weights + dynamic per-token activations), GPTQ |
| Size | 29.97 GB, 15 shards (BF16 base 55.6 GB → 1.85x smaller) |
| Architecture | full `Qwen3_5ForConditionalGeneration` — vision + LM + MTP |
| BF16 kept | `visual.*` (333 tensors), `mtp.*` (15), `linear_attn.norm`, `in_proj_a`, `in_proj_b` |
| KV cache | FP8 E4M3, static per-tensor |
| Calibration | **`malaiwah/qwen38-27b-fidelity-suite-v3`** — our published dataset — 181 contexts x 2048 |
| Benchmark HW | **RTX PRO 6000 Blackwell, 96 GiB**, 86,606 / 97,887 MB used (88.5%) |
| Benchmark ctx | 8,192 |

They calibrate on our published suite and credit the rtx6kpro Kimi-K3 KLD
protocol plus "the Gilded Gnosis EXL3 model cards" for the error-ladder and
validation-tier framework. Their card states outright: "KLD far from
malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated".

## Their measured numbers

KLD protocol: hidden-state capture after final RMSNorm → shared BF16 LM-head
replay → full-vocabulary `KL(ref||cand)`, two-pass LSE, body-only.
Suite **136 contexts x 2048 tokens**, 278,392 scored positions.
Same *methodology* as ours, **different suite** → not directly comparable.

| Model | KLD | prefill 2k | decode C1 | note |
|---|---:|---:|---:|---|
| v31 (FP8 MLP + **W8A16 weight-only attn**) | 0.01014 | 6,385 | 49.9 | their best composite |
| v18 | 0.00903 | 6,023 | 40.8 | their lowest KLD |
| **v17** (this repo, uniform FP8) | **0.01231** | **7,306** | **49.8** | p99 0.1911, top-1 96.17% |
| v6 (best NVFP4 W4A4) | 0.01597 | 7,756 | 52.3 | their best 4-bit |
| official Qwen FP8 | 0.0134 | — | — | **cross-protocol anchor** |

Their KLD attribution: MLP FP8 **~0.009 (dominant)**, linear_attn activation
quant ~0.0025, self_attn ~0.0006.

## Cross-protocol anchoring

Both projects measured the **same third artifact**, official Qwen FP8:
theirs 0.0134, ours 0.005294 (v5 suite). Ratio **2.53x** — their suite reads
~2.5x higher on identical weights. Rescaling their numbers by /2.53 gives an
*approximate* v5-equivalent (single multiplicative factor; their literary
stratum at 0.02224 dominates their mix, so this is an anchor, not an identity):

| Model | their scale | ~v5-equivalent | ours (v5) |
|---|---:|---:|---:|
| their v31 (best) | 0.01014 | ~0.0040 | — |
| their v17 | 0.01231 | ~0.0049 | — |
| their v6 (4-bit) | 0.01597 | ~0.0063 | — |
| **our K5K6 trellis** | — | — | **0.002700** |

Our flagship is **~1.5x lower KLD than their best config** at **16.82 GiB**
trellis payload vs their 29.97 GB — roughly **1.8x smaller and better**.

## The decisive difference is the card, not the method

Their frontier is computed on a **96 GiB** GPU. Ours is a **31.40 GiB** 5090.
Their 29.97 GB checkpoint is 27.91 GiB, leaving **~3.5 GiB** on our card for
KV + activations + graphs. Our fidelity profile spends ~9.25 GiB of KV to reach
238,400 tokens, so their artifact would serve roughly **60–90k context** on a
5090 with no vision headroom — it cannot meet our >=238,400 criterion at all.

**8-bit is viable for them because they have 3x our VRAM. On 32 GiB, 4-bit-class
weights are mandatory, and that is exactly where trellis coding beats NVFP4.**
Their own best 4-bit config (v6, 0.01597 → ~0.0063 v5-equiv) is 2.3x worse than
our trellis at ~5 bits. Two different Pareto frontiers, set by the hardware.

## Where they beat us

- **Portability.** Stock `compressed-tensors`; no b12x, no 10 bind-mounted
  patches, no custom image, no `EXL3_PATCH_SHA256` pin. Runs on Ada/Hopper too.
- **Prefill vs our fidelity profile.** 7,306 vs 2,987.7 tok/s at 2k (2.4x).
  Our *throughput* profile beats them: 9,638.9 (1.3x).
- **Breadth of ablation.** A documented 38-run sweep with a published composite
  score. Our sweep is deeper per-axis but narrower in config count.

## What this changes for us (verified against our artifacts)

1. **Their MTP defect does not apply to us — verified.** Their `config.json`
   omitted `re:^mtp.*` from `ignore`, so vLLM applied dynamic FP8 activation
   quant to BF16 MTP weights and acceptance collapsed to ~0.3%; adding it
   restored ~70–77% acceptance and **+54% decode** (106.9 vs 69.4 tok/s).
   Our merged checkpoint already carries `re:^mtp\..*` **and**
   `re:.*\bmtp\b.*` among 103 ignore entries. Consistent with our measured
   fox 215.3 post-merge.
2. **Their best lever is already ours — verified.** v31's −18% KLD comes from
   W8A16 *weight-only* attention (no activation quant). Our
   `recipe-fp8attn-nvfp4mlp-rtn.yaml` already sets `input_activations: null`
   with the comment "weight-only (W8A16)". Independent convergence; the lever
   is spent, not available as a future gain.
3. **Our requant lane is MLP-precision-bound, and their attribution says so.**
   They measure MLP as the dominant term (~0.009) *at FP8*. Our MLP is NVFP4
   (4-bit), which is why our mixed RTN sits at 0.022121 against a 0.012 bar.
   Closing that gap needs **more bits in the MLP**, which our VRAM forbids.
   This is independent corroboration that the requant lane cannot reach the
   flagship on this card — the trellis checkpoint is the right answer.
4. **Two of our planned GPTQ levers are weakened by their data.** They tested
   `dampening_frac` 0.001–0.1 → all 0.0124 +/- noise ("irrelevant for FP8"),
   and 512 vs 181 calibration samples → within noise. Caveat: both results are
   **FP8-specific**; our failure mode is 4-bit with Hessians built from only 42
   sequences, where conditioning matters more. Net effect on our plan: keep
   attempt 2 at **181 unconcatenated 2048-token sequences** (their
   proven-adequate count, ~2.5–3 h), and drop the dedicated dampening sweep
   unless attempt 2 shows Hessian-conditioning is still the binding constraint.
5. **Two serving knobs we do not use** appear in their compose file:
   `--enable-prefix-caching` and `--async-scheduling`. Prefix caching is
   independently our top-ranked deep-context lever (see the context-curve work).

## Honest gaps in this comparison

- Their KLD suite (136x2048) is not our suite; the /2.53 rescale is an anchored
  approximation from one shared artifact, not a measurement.
- Their throughput is on a 96 GiB RTX PRO 6000 at ctx 8,192; ours is a 5090 at
  238,400–249,600. Neither throughput column is transferable.
- Their decode C1 49.8 excludes MTP; the card's 106.9 tok/s MTP figure is from
  a separate concurrency-1 run. Our fidelity essay 104.1 / fox 228.3 use our
  own harness. No like-for-like decode comparison exists.
- The rigorous fix for all of the above is to run **their checkpoint on our
  suite**. Feasible for short-context KLD capture (~3.5 GiB headroom), and it
  would replace every rescaled number here with a measured one.
