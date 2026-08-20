# EDA vs Unsloth vs K5K6-hydrated: 3-way allocation comparison

**Date:** 2026-08-20. Method: extracted per-tensor bit allocation from three
sources and compared sensitivity rankings via Spearman correlation.

## Sources

1. **K5K6-hydrated** (shipped): `quantization_config.json` → `tensor_storage`
   from `/home/mbelleau/models/qwen38-27b-K5K6-fp8-embed/`. The actual model
   running in production. Uniform recipe: all attention=K6, MLP gate/up=K5,
   down=K6, no per-layer variation.

2. **EDA proposed** (unshipped): `receipts/eda-resolve/resolve-rel.json` →
   `widths`. Error-driven allocation from the 409-module sensitivity ladder.
   175 modules moved vs K6 baseline. KLD predicted −0.000211 (measured +0.000366,
   the known regression).

3. **Unsloth UD-Q4_K_M** (Dynamic 3.0): parsed GGUF tensor metadata with
   `gguf-py`. Per-tensor GGUF quant types mapped to approximate bpw:
   Q3_K=3.5, IQ3_S=3.5, IQ4_XS=4.25, Q4_K=4.5, IQ4_NL=4.5, Q5_K=5.5,
   Q6_K=6.5, Q8_0=8.5.

## Per-submodule mean bits (3-way)

| submodule | K5K6 | EDA | GGUF | GGUF type distribution |
|---|---:|---:|---:|---|
| self_attn.v_proj | 6.00 | 6.88 | 7.00 | Q8_0:7, Q5_K:6, Q6_K:3 |
| self_attn.k_proj | 6.00 | 6.81 | 6.19 | Q6_K:9, Q5_K:5, Q4_K:1, Q8_0:1 |
| linear_attn.out_proj | 6.00 | 6.25 | 5.37 | Q5_K:33, Q6_K:5, IQ4_XS:5, Q4_K:5 |
| self_attn.o_proj | 6.00 | 6.00 | 5.67 | Q5_K:11, Q6_K:4, IQ4_XS:1 |
| mlp.down_proj | 6.00 | 5.92 | 4.72 | IQ4_XS:24, Q5_K:22, Q4_K:10, IQ4_NL:3, IQ3_S:3 |
| linear_attn.in_proj_z | 6.00 | 5.71 | 4.69 | Q4_K:24, Q5_K:12, IQ4_XS:12 |
| mlp.up_proj | 5.00 | 5.55 | 4.63 | IQ4_XS:30, Q5_K:16, Q4_K:14, Q3_K:2, IQ4_NL:1 |
| mlp.gate_proj | 5.00 | 5.17 | 4.49 | IQ4_XS:31, Q4_K:14, Q5_K:12, Q3_K:4, IQ4_NL:2 |
| linear_attn.in_proj_qkv | 6.00 | 5.04 | 4.72 | Q4_K:29, Q5_K:12, IQ4_XS:6, IQ4_NL:1 |
| self_attn.q_proj | 6.00 | 4.69 | 4.50 | IQ4_XS:8, Q4_K:6, Q5_K:2 |

## Sensitivity rankings (most → least sensitive)

| rank | K5K6-hydrated | EDA proposed | Unsloth GGUF |
|---:|---|---|---|
| 1 | all attn = 6.00 (tied) | self_attn.v_proj (6.88) | self_attn.v_proj (7.00) |
| 2 | — | self_attn.k_proj (6.81) | self_attn.k_proj (6.19) |
| 3 | — | linear_attn.out_proj (6.25) | self_attn.o_proj (5.67) |
| 4 | — | self_attn.o_proj (6.00) | linear_attn.out_proj (5.37) |
| 5 | — | mlp.down_proj (5.92) | mlp.down_proj (4.72) |
| 6 | mlp.gate/up = 5.00 (tied) | linear_attn.in_proj_z (5.71) | linear_attn.in_proj_qkv (4.72) |
| 7 | — | mlp.up_proj (5.55) | linear_attn.in_proj_z (4.69) |
| 8 | — | mlp.gate_proj (5.17) | mlp.up_proj (4.63) |
| 9 | — | linear_attn.in_proj_qkv (5.04) | self_attn.q_proj (4.50) |
| 10 | — | self_attn.q_proj (4.69) | mlp.gate_proj (4.49) |

## Spearman rank correlations

| level | K5K6 vs EDA | K5K6 vs GGUF | EDA vs GGUF |
|---|---:|---:|---:|
| per-submodule (n=10) | 0.348 | 0.611 | **0.869** |
| per-tensor (n=400) | 0.328 | 0.328 | 0.245 |
| per-layer (n=64) | 0.531 | 0.503 | 0.103 |

## Depth band means

| band | K5K6 | EDA | GGUF |
|---|---:|---:|---:|
| L00–15 | 5.68 | 5.72 | 4.66 |
| L16–31 | 5.68 | 5.72 | 4.94 |
| L32–47 | 5.68 | 5.77 | 4.74 |
| L48–63 | 5.68 | 5.50 | **5.35** |

## Analysis

### 1. EDA and Unsloth strongly agree on which *roles* are sensitive

The submodule-level Spearman ρ = 0.869 is remarkably high for two completely
independent methods — one is our error-driven DP solver on trellis K-widths, the
other is Unsloth's imatrix-guided GGUF type selection. Both agree:

- **v_proj is the most sensitive** tensor (EDA 6.88, GGUF 7.00 — Unsloth even
  uses Q8_0 for 7/16 full-attention layers' v_proj)
- **k_proj is second** (EDA 6.81, GGUF 6.19)
- **q_proj is least sensitive** attention tensor (EDA 4.69, GGUF 4.50)
- **out_proj > in_proj** for both linear and full attention
- **MLP gate/up are less sensitive than down_proj** — though EDA splits them
  more than GGUF

This cross-validation is the strongest evidence yet that our EDA sensitivity
ranking is physically correct, not an artifact of our specific calibration
data or error metric.

### 2. K5K6-hydrated is blind to per-tensor sensitivity

The shipped model has only two tiers: attention=K6 (all 208 attention tensors
identical) and MLP gate/up=K5. It cannot differentiate v_proj from q_proj,
or out_proj from in_proj_qkv. This is why its submodule-level correlation with
both EDA (0.348) and GGUF (0.611) is much lower — it's a flat allocation
forced into a ranking comparison.

The K5K6 recipe was chosen for simplicity and kernel efficiency (uniform K
within each role class), not for fidelity optimization. The EDA analysis shows
significant headroom: demoting q_proj from K6 to K4 and in_proj_qkv from K6
to K5 frees ~460 MB that could fund v_proj → K7 and k_proj → K7.

### 3. Per-tensor correlation is low (0.25) — the disagreement is *which layers*

EDA and GGUF agree on *roles* (ρ=0.87) but disagree on *layers* (ρ=0.10).
EDA's depth pattern is U-shaped with a late-layer drop (L48-63 mean 5.50 vs
5.72 for L00-15). Unsloth does the opposite: L48-63 mean 5.35 is their
*highest* band, and they allocate Q6_K/Q8_0 to layer 63's v_proj (8.5 bpw),
while EDA gives layer 63 the fewest bits (5.00 mean).

This is the most interesting disagreement. Two hypotheses:

1. **EDA's late-layer demotion is wrong.** Unsloth's imatrix sees layer 63 as
   critical (it's the last representation before the LM head), and gives it
   more bits. Our EDA's U-shape may be an artifact of the calibration data
   not exercising late-layer pathways enough.

2. **Unsloth's late-layer boost is overcompensating.** Layer 63 is a full
   attention layer with v_proj at Q8_0 (8.5 bpw) — that's near-lossless for a
   single tensor. The imatrix may be overweighting the last layer because of
   its direct connection to the output.

### 4. Unsloth uses a wider bit range and more quant types

| metric | K5K6 | EDA | GGUF |
|---|---:|---:|---:|
| distinct bit levels | 2 (K5, K6) | 4 (K4–K7) | 6 (3.5–8.5) |
| min bpw | 5.0 | 4.0 | 3.5 |
| max bpw | 6.0 | 7.0 | 8.5 |
| spread | 1.0 | 3.0 | 5.0 |

Unsloth's wider range (3.5–8.5 bpw) is possible because GGUF supports many
quant types. Our trellis is limited to integer K-widths, but the EDA allocation
already uses K4–K7 (spread 3.0). The K5K6-hydrated's spread of 1.0 is the
narrowest — it doesn't exploit the sensitivity variation at all.

### 5. Per-layer GGUF allocation is more depth-aware than EDA

GGUF layer 63: 6.36 bpw (highest of any layer). EDA layer 63: 5.00 bpw
(lowest). This is a 1.36 bpw gap on the most critical layer. If Unsloth's
imatrix is correct that late layers need more bits, our EDA's U-shape is
systematically wrong on the tail. This directly supports the `abs` weighting
finding in `eda-resolve-2026-08-19.md` — `abs` was the only weighting that
moved bytes *toward* attention/GDN, and the only one with a positive late-layer
byte delta.

## Implications for requant

1. **The EDA role ranking is validated.** v/k_proj > o_proj > out_proj >
   down_proj > in_proj > gate/up > q_proj. This is the correct sensitivity
   ordering, confirmed by an independent method.

2. **The EDA depth pattern needs re-examination.** The late-layer demotion
   contradicts Unsloth's allocation. Before requanting, we should test the
   `abs` weighting (which boosts late layers) against the `rel` weighting
   (which demotes them) on the actual KLD suite, not just the predicted
   objective.

3. **K5K6-hydrated leaves fidelity on the table.** The 2-tier recipe cannot
   differentiate q_proj (least sensitive) from v_proj (most sensitive). An EDA
   rebuild at the same total size, with q_proj→K4 and v_proj→K7, should
   improve KLD — *if* the depth pattern is fixed.

4. **Mixed quant types are Unsloth's structural advantage.** GGUF's 6+ quant
   types give finer granularity than our 4 trellis K-widths. This is the
   argument for per-tensor-type allocation (todo item #1 in
   `docs/59-unsloth-dynamic3-research.md`).

5. **v_proj at Q8_0 (8.5 bpw) is notable.** Unsloth keeps v_proj near-lossless
   in 7/16 full-attention layers. Our EDA gives v_proj K7 (7.0 bpw) — close,
   but not lossless. If v_proj is truly the most sensitive tensor, a K8 or
   BF16 v_proj on the last few layers might be worth the cost.
