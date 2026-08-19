# Depth calibration: LATE layers are the most precision-sensitive — my model had the sign backwards

**Date:** 2026-08-19 (goal session). Reports:
`receipts/kld-reports/report-depth-{early,mid,late}.json`.

## The experiment

Three matched arms, `mlp.gate_up_proj` -> MXFP6 over a 13-layer band, everything
else identical (all-trellis, ANY_BITS, fp8 KV off, same 512-context shard-0 suite,
same BF16 head). **13 layers each means identical converted-module count,
identical byte cost and identical ~0.029 GiB/layer KV cost** — layer position is
the only variable. n=1 per arm is sufficient because this pipeline was previously
shown bit-reproducible (run-to-run SD = 0).

| arm | FP6 layers | mean KLD | ci95 | p99 | top-1 |
|---|---|---|---|---|---|
| baseline | none | 0.003405 | — | 0.034889 | — |
| **early** | 0-12 | **0.003600** | [0.003354, 0.003876] | 0.037236 | 0.9753 |
| **mid** | 26-38 | **0.003838** | [0.003565, 0.004140] | 0.039935 | 0.9744 |
| **late** | 51-63 | **0.004395** | [0.004127, 0.004693] | 0.045253 | 0.9721 |

**Monotone increasing with depth.** Early-vs-late CIs are disjoint
([.., 0.003876] vs [0.004127, ..]); mid sits between, overlapping each slightly.
Converting the LAST 13 layers costs 2.3x the KLD delta of converting the FIRST 13
(+0.000990 vs +0.000195 over baseline).

## Three things this refutes, including my own

1. **My `exp` depth form is backwards.** I implemented weight
   `(1+amp)^(n-1-L)` — early-heavy — reasoning that error injected early is
   amplified by all downstream layers. The measurement says the opposite: precision
   matters MORE near the readout. Mechanistically, downstream layers appear to
   *attenuate* injected error (residual + normalisation are contractive on
   average) while error injected near the readout reaches the logits with no
   remaining opportunity to be absorbed. `depth-form=late` (`(1+amp)^L`) is now
   the default; `exp` is retained only to reproduce the superseded sweep.

2. **My "depth-blind objectives rob early layers, which is harmful" claim is
   refuted.** `abs` at amp=0 moved −779 MB out of L00-15 and +786 MB into L48-63.
   Given this measurement, **that is the correct direction.** The depth-blind
   objective was accidentally right on depth; my correction would have made it
   worse. The `out_energy` term — which grows toward the readout — was doing real
   work, not introducing a bias.

3. **The U-shape prior art does not describe this architecture.** llama.cpp's
   `use_more_bits` promotes the first *and* last n/8 of layers
   (`src/llama-quant.cpp:430`) and exllamav3's allocator weights by
   `min(layer, stack_max - layer)` (`allocation.py:63`, docs/58). Both protect
   both ends. We measure no early-end penalty at all: early is the *cheapest*
   band to degrade. Either the hybrid GDN stack differs from the dense decoders
   those heuristics were tuned on, or the heuristics carry an untested assumption.

## Scope and limits

This measures **FP6 conversion of `gate_up` in a 13-layer band**, not trellis-width
reallocation. The transferable claim is about *which layers tolerate precision
loss*, and it should hold across perturbation types, but a trellis-width version of
the same experiment would be the strict test. Only the **ordering** is measured;
the amplification *magnitude* (`--depth-amp`) is still uncalibrated — fitting it
properly needs more than three points.

## Consequence for the solver

`--depth-form late --depth-amp 0.05` (with `--max-width 6`) moves bytes toward the
readout as the physics now demands: L48-63 **+345 MB**, L32-47 +296 MB,
L00-15 −568 MB, 174 modules moved, 0 modules above K6. Artifact:
`receipts/eda-resolve/resolve-classkld-k6-late05.json`.
