# Peer review of the full project arc, prompted by the GLM-5.2 cross-model exercise

**Date:** 2026-08-19 (evening)
**Method:** the cross-model write-up (`receipts/glm52-transfer-2026-08-19.md`) forced every
finding into its general form. Generalising exposes asymmetries: checks made in one
direction but not the other, defaults inherited without measurement, and criticisms of
others that apply to ourselves. Each was then verified against the current repo/host
state, not memory. Three were confirmed and acted on immediately; twelve are registered
as revisit items.

## Found and fixed this session

1. **KV-dtype protocol gap in our own fidelity evidence** — the exact flaw I criticised
   on the GLM page ("quality test used FP8 KV, serving uses NVFP4 DS-MLA... likely
   understated"). Verified: `run-qwen38-27b.sh:293` serves with `--kv-cache-dtype
   fp8_e4m3`; **no KLD capture runner sets any KV dtype**, so every KLD we quote
   (0.003437, 0.005672, 0.063759...) was measured at the engine's default KV, not the
   served fp8. Direction of bias: served fidelity is likely slightly *worse* than
   measured. Margin is ~3.5x on criterion 3, so the pass probably survives — but that is
   a prediction, not a measurement. Registered: rerun one capture at fp8 KV parity.
2. **Published overclaim**: the HF cards said `balanced` (0.005672) is "statistically
   level with official FP8" (0.005294). Not supportable: different-sized suites, and
   balanced's CI low end (0.005302) sits *above* FP8's point estimate. Corrected on all
   three cards within the hour, with the reason in the commit message.
3. **Primary evidence not archived**: the 14 KLD report JSONs — the raw evidence behind
   criteria 3/4 — existed only in `/tmp/kld-data/reports/`. Receipts quoted numbers; the
   artifacts were one tmpfs clear from gone. Now in `receipts/kld-reports/`.

## What survives scrutiny

- The three-profile frontier, all gated 9/9 at 600 W with n=3 CIs.
- The mutual-exclusivity proof for criteria 1 vs 3 (measured from four directions; the
  power-cap removal *widened* the gap, 3.9x -> 4.9x).
- The additivity model's scope split (MLP holds at -1.9 %/-6.0 %; attention fails +46 %),
  because it was validated on held-out predictions rather than fitted.
- The trellis-ceiling numbers (2,458 bake-off, 80 % saturation): trellis measured
  power-insensitive (+0.0 %), so 400 W-era values remain valid for that path.
- Self-corrections were made in place with provenance (L9, roofline, suspect-2, additivity).

## Registered revisit items and why (15, phased)

**Evidence integrity** — the FP4/FP6 bake-off columns, the 43.5 % kernel-efficiency
figure, the "26.9 % of silicon" headline and the criterion-1 wall table were all derived
at 400 W; only the profile baselines were re-measured at 600 W. The VRAM overclock
(`mem_clock_offsets: 6000`) was disclosed as a parenthetical while the power cap got a
receipt — asymmetric treatment for two machine-state deviations of the same kind. And
every KLD is n=1 capture: bit-reproducibility was claimed for the *old* stack, never
re-verified with b12x FP32 accumulator arenas in the loop (GLM's protocol runs 3 repeats
and reports run SD ~0.002 — the same order as deltas we have interpreted).

**Fidelity gaps** — criteria 3/4 are certified at 2,047 positions while the product
story is 238,400 context, and our own +46 % result proves attention error *compounds
with position*. Long-context fidelity is unmeasured; "returns 200 at 200k" is not
"works at 200k".

**Performance revisits** —
(a) The FP4 draft head A/B (negative, 2026-08-18) predates the V2 runner *and* the
acceptance-coupling insight; the mechanism says matching the draft to an FP4 *target*
should raise acceptance — TG-fox needs +1.4 % to flip criterion 2b on the throughput
profile.
(b) `VLLM_EXL3_PREFILL_CAPACITY` was never touched on our own stack; our b12x #235
comment tells GLM to check whether the scratch cap makes their capacity dial a no-op —
we should eat our own dog food on the memory-starved fidelity profile.
(c) The GLM doc shows b12x graphs its MoE path with pre-initialised scratch;
`_warm_b12x_trellis_device` short-circuits after ONE shard per device, so bits=4/5
shapes first allocate lazily — possibly inside capture. That is a *new,
externally-informed* hypothesis for the parked K5 corruption (+51 % PP if fixed), not a
seventh attempt at an old one.

**Serving realism** — every published TG number is C1; GLM publishes C1/C4/C8, and our
own earlier work found speculative depth is concurrency-dependent. The HF cards present
single-stream numbers without saying so prominently. The 8 MP vision OOM from early
phases was never resolved and the stress gate only exercises the small fixture — with
engine-fatal OOM (#52871) unfixed upstream, a large image may still kill the engine.
Boot pays the full FP4 conversion every start (~56 s under Restart=always); GLM's
content-addressed K6 cache is the template for fixing that.

**Upstream hygiene** — PR #436 still carries 28/30 unsigned commits and an offer of a
clean replacement branch that was never delivered.

## Honest bottom line

The load-bearing conclusions stand. The review found no error that flips a criterion
verdict today — but it found one published sentence that overstated evidence (fixed),
one protocol gap that makes our strongest numbers slightly optimistic in an unmeasured
way (queued, margin likely absorbs it), and a body of derived tables quietly mixing
400 W and 600 W provenance (queued). The cross-model exercise generated five of the
fifteen items directly; that is the measurable value of writing findings down for
someone else's hardware.
