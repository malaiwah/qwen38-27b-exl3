# Depth weighting: the published objective family robs early layers

**Date:** 2026-08-19 (goal session). Tool: `tools/eda-resolve.py --depth-amp`.
Artifacts: `receipts/eda-resolve/resolve-{abs,sqrt_energy}-amp{0,0.02,0.05}.json`.
**No GPU used.** Prompted by the maintainer's question: does the solver allocate
more bits to earlier layers, since error accumulates through the stack?

## Answer: no — and worse than neutral

The solver had **no layer-position term at all**. Weight depended only on
`out_energy`. That is faithful to the published objective family (`rel`, `abs`,
`sqrt_energy` are all depth-blind, and the ladder's `proxy_err` is module-local,
measured under fixed hydrated propagation — plan `predicted.note`, docs/57 §1),
but "faithful" turns out not to mean "neutral".

Byte delta versus hydrated, by depth quartile, for the depth-blind solves:

| weighting (amp=0) | L00-15 | L16-31 | L32-47 | L48-63 |
|---|---|---|---|---|
| `abs` | **−779,223,040** | −102,236,160 | +95,027,200 | **+786,432,000** |
| `sqrt_energy` | **−551,813,120** | +24,248,320 | +152,043,520 | **+375,521,280** |

Both **strip bits from the first 16 layers and hand them to the last 16**. In
`abs`'s case that transfer (±780 MB) is *larger* than its attention/GDN shift
(+100 MB). So the depth blind spot has a direction, and if error accumulates
with depth it is the harmful one — a second directional error stacked on the
attention→MLP error docs/57 already diagnosed.

Mechanistically this is unsurprising in hindsight: late-layer modules have no
downstream layers to amplify their error, so a depth-blind objective sees them as
equally worth protecting, while `out_energy` (which grows toward the readout)
actively favours them.

## A one-parameter depth term, and where it breaks even

Added `--depth-amp`: module weight `*= (1+amp)**(n_layers-1-layer_index)`. The
reading is physical — if each downstream layer amplifies injected error by a mean
factor `(1+amp)`, error injected at layer L is amplified `(1+amp)**(63-L)` times
before the readout, so early layers earn more protection. `amp=0` reproduces the
depth-blind behaviour exactly (verified: the `rel` solve at `amp=0` still
replicates the published solve bit-for-bit, 175 modules, identical role deltas).

| weighting | amp | L00-15 delta | moved |
|---|---|---|---|
| `abs` | 0 | −779,223,040 | 320 |
| `abs` | 0.02 | −631,111,680 | 325 |
| `abs` | 0.05 | −302,120,960 | 345 |
| `sqrt_energy` | 0 | −551,813,120 | 236 |
| `sqrt_energy` | 0.02 | −218,234,880 | 249 |
| **`sqrt_energy`** | **0.05** | **+145,489,920** | 251 |

`sqrt_energy` crosses over near **amp ≈ 0.05**: a 5%-per-layer downstream
amplification is roughly where early layers stop being robbed. `abs` is still
negative at 0.05 because its `out_energy` term pulls hardest toward the readout.

## `amp` is NOT calibrated — and the experiment that would calibrate it is cheap

Nothing in this repository fixes `amp`. Every layer-range fidelity experiment we
have converted ranges **starting at layer 0** (`0-12`, `0-28`, all 64 —
`receipts/layer-range-dial-2026-08-19.md`), so there is no early-vs-late contrast
to fit against, and that receipt's 13-layer KLD was **predicted (0.003891), not
measured**. Every `amp > 0` allocation above is a sensitivity probe, not a result,
and the artifacts record `depth_amp_calibrated: false`.

The calibrating experiment needs no new tooling, because the dial already exists:

```
PROFILE=balanced MAX_MODEL_LEN=238400 VLLM_EXL3_FP6_LAYER_RANGE=0-12    # early 13
PROFILE=balanced MAX_MODEL_LEN=238400 VLLM_EXL3_FP6_LAYER_RANGE=51-63   # late 13
```

Thirteen layers each: **identical converted-module count, identical byte cost,
identical ~0.029 GiB/layer KV cost** — so a KLD capture/replay of both arms
isolates layer position as the only variable. The sign of
`KLD(early) − KLD(late)` tests the accumulation premise directly, and its
magnitude gives a first empirical anchor for `amp`. Cost: 2 boots + 2 captures,
roughly 30–40 min of GPU. Queued.

If `KLD(early) > KLD(late)`, early layers matter more, accumulation is real, and
depth weighting is justified with a measured exponent. If they are
indistinguishable, the depth term should stay at `amp=0` and this receipt is the
reason why.
