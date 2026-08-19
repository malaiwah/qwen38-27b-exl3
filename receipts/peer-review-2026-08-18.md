# Peer review of the 2026-08-18 optimization session (self-audit)

Reviewing our own evidence. Three claim-level corrections, one code defect
(fixed), and a list of methodology gaps that bound what we can currently
conclude.

## A. Claim corrections (retractions / qualifications)

### A1. We never reported CIs or tails. The mean hid both.
`fidelity.py` emits `context_bootstrap`, `p95/p99/p999/max_kld`, `kld_tail`
and `mean_jsd_bits`. No 2026-08-18 receipt used any of them. Full picture:

| profile | mean | ci95 | p95 | p99 | p999 | max |
|---|---|---|---|---|---|---|
| trellis (hyd) | 0.002700 | 0.00251-0.00291 | 0.0082 | 0.0275 | 0.131 | 3.73 |
| all-FP6 | 0.010699 | 0.01007-0.01140 | 0.0344 | 0.111 | 0.526 | 5.66 |
| flagship FP4 | 0.056732 | 0.05311-0.06087 | 0.1927 | 0.638 | 2.525 | 13.01 |
| flagship+fp8dg | 0.057621 | 0.05399-0.06182 | 0.1960 | 0.638 | 2.567 | 13.43 |
| gate_up only | 0.028874 | 0.02697-0.03102 | 0.0940 | 0.319 | 1.453 | 13.15 |
| down only | 0.017382 | 0.01625-0.01867 | 0.0570 | 0.192 | 0.853 | 11.48 |
| GDN only | 0.019835 | 0.01846-0.02140 | 0.0660 | 0.216 | 1.001 | 9.91 |

Consequences:
- **fp8dg's "+0.0009 KLD" is not a measurement.** CI95 [0.05399,0.06182]
  overlaps flagship [0.05311,0.06087] almost entirely. Correct statement:
  *statistically indistinguishable at 512 contexts*. (Favourable either way,
  but we cannot claim a cost, nor claim zero cost.)
- **The per-row-GS "0.8% KLD gain" (0.056251 vs 0.056732) was pure noise** —
  an order of magnitude inside the CI. Rejecting it on PP was right; the KLD
  half of that receipt is over-precise and is retracted.
- **Tails are where FP4 actually hurts.** Flagship p99 = 0.638 nats: 1 in 100
  scored positions has a materially different next-token distribution
  (trellis p99 = 0.0275, i.e. 23x lower). Mean-only reporting understates
  what a user perceives. The distribution shifts roughly proportionally
  (p99/mean ~ 11 for both trellis and FP4), so FP4 is not *pathologically*
  tailed — but absolute tail mass is what matters for reputation.

### A2. "Additive decomposition closes to 1e-5" is circular. RETRACTED.
t+e was derived *from* the four measurements, then checked against the same
four equations — it closes by construction, not as validation. With ±7% CIs
per run, the propagated uncertainty on each derived contribution is ~±10%.
What survives:
- gate_up is the largest contributor: **robust** (CI 0.02697-0.03102 does not
  overlap either other group).
- GDN (0.01846-0.02140) vs down (0.01625-0.01867) **overlap at ~0.0185** —
  their ordering is NOT statistically established. Report as tied.
- Real validation requires a held-out combination: gate_up+down FP4 predicts
  0.0416 under additivity. One unrun capture would falsify or support it.

### A3. "Trellis floor + int6 embeds" was inferred, not measured.
`attrib-none` (all-trellis + int6 embeds) OOMed at load (30.25 GiB allocated,
util 0.85). The 0.0027 floor comes from a *different* run with BF16 embeds.
e = 0.0020 is a subtraction, not an observation.

### A4. Upstream status overstated.
Both PRs are OPEN with **`pre-run-check` FAILING** (#436, #437) and issue #435
has no maintainer response. "Published upstream" is accurate; "landed" or
"accepted" is not. #436 also carries an unreviewed CodeRabbit auto-review.
Actionable, not yet done.

## B. Code defect found and fixed

`_fp8dg_prefill_apply` cached post-processed FP8 weights in a module-global
dict keyed by `(id(layer), shard_id)`. **CPython reuses object addresses after
GC**, so a freed-and-reallocated module could alias another layer's FP8
weights — silent wrong-weight inference. Layers live for process lifetime so
it was latent, not active. Fixed: cache now lives on the layer
(`layer._fp8dg_cache[shard_id]`), correct by construction. Re-verified:
PP=7078 (was 7062, same noise band), selftest cos=0.999512 unchanged.

Two further defects logged, not yet fixed:
- **Latent OOM**: the ~2 GB fp8dg cache is allocated lazily on the *first long
  prefill*, i.e. after vLLM has already sized the KV cache. It should be
  pre-allocated at load time (fail fast) and declared to the memory profiler.
- `_FP8DG_DISABLED` latches permanently on the first exception, including a
  transient OOM, with a single warning. Silent permanent slow-path.

## C. Methodology gaps that bound our conclusions

1. **The PP metric is overhead-dominated, not a prefill-throughput metric.**
   It is `prompt_tokens / wall_time` for one 2051-token request with
   `max_tokens=1`, including HTTP, tokenize, schedule, one decode step,
   sample, detokenize. Roofline (Section D) puts GEMM time at ~30-50% of that
   wall time. So headline PP mixes kernel speed with request overhead, is not
   comparable to published vLLM numbers, and compresses real kernel gains.
2. **The TG metric is an acceptance-rate artifact.** 161 (fox) vs 75 (essay)
   on the same config is a 2.1x spread from MTP acceptance alone; all-trellis
   hit **100% acceptance** on the repetitive prompt. Several TG comparisons
   across configs are therefore not like-for-like unless acceptance is
   reported with them (it now is, for the last few).
3. **No concurrency measurement at all.** Every number is single-request, yet
   we serve `max_num_seqs=4`. Aggregate throughput — the actual serving
   figure — is unmeasured and is probably materially higher.
4. **No variance / repeatability.** All headlines are one boot, median-of-5
   requests. Boot-to-boot spread is unknown but we observed ±2% within a
   boot, and we have made 2-3% calls (MTP-skip +2.5%, draft-head "no gain").
   Anything under ~5% in this session should be treated as unresolved.
5. **Uncontrolled clocks/thermals.** Sequential boots at 600 W with no
   cooldown or clock logging; later configs in a session may be
   systematically penalized.
6. **KLD suite is single-shard.** Prior receipts show shard-0 vs
   contamination-corrected multi-shard means differing by ~2.5x in absolute
   terms (hyd: 0.0027 vs 0.00717). Our deltas are internally valid; absolute
   values are suite-specific. `--score-from` (early-position exclusion) was
   never exercised, and the replay-vs-live floor (audit gap G2) is still
   unquantified — we do not know our own resolution limit.
7. **Vision is functionally verified only.** 256x128 PNG passes; 8 MP is a
   known engine-fatal OOM; no vision throughput, multi-image, or vision-KLD
   measurement. Marking the vision items "done" was generous.
8. **Banded-converter 0.70% nibble mismatch is explained, not proven.** The
   1-ulp-amax story is testable in 5 lines (pass the unbanded gs as override
   and expect exact equality); untested.
9. **No profile registry, no regression suite.** ~12 interacting env flags,
   no boot-time record of the resolved profile, no automated re-verification.
   This already caused one misattribution (the 154.5 draft-head number that
   was actually a fallback path).
10. **New reports not published.** report-attrib-*, report-fp8dg are local
    only; prior work's reproducibility standard (HF dataset + manifest) is
    not yet met for this session.

## D. Roofline reality check (why the above matters)

RTX 5090: 1792 GB/s; dense tensor ~1676 TFLOPS FP4, ~838 FP8, ~419 BF16.
27B params -> ~54 GFLOP/token prefill.

Prefill effective peak for our mixes (params-weighted):
- flagship (80% FP4 MLP+GDN, 20% attn trellis dequant->fp16): 1047 TFLOPS
  -> **19.4k tok/s** ceiling. Note attn is 20% of params but ~50% of compute
  time — exactly why fp8dg on attn helped.
- with fp8dg attn: 1397 TFLOPS -> **25.9k tok/s** ceiling.
Measured 7.1k = **27% of ceiling**. Of a 290 ms request, GEMMs need ~79 ms at
ceiling (~158 ms at 50% kernel efficiency) -> **130-210 ms is not GEMM**.

Decode bytes/step: 17.36 GiB weights + 6.37 GiB (7x lm_head, measured) + KV
~0.3 GiB @8k = ~24 GiB -> **74.7 steps/s** ceiling.
- essay: acceptance 51.7% -> 4.10 tok/step; 75 tok/s = 18.3 steps/s = 54.6
  ms/step vs 13.4 ms bandwidth-bound -> **4.1x off**, ~41 ms/step of overhead
  (~6 ms per each of the 7 sub-steps: graph replay + sample + accept in
  Python).
- fox: 23 steps/s -> **3.2x off**.

**Conclusion: both dimensions are 3-4x off roofline and in both cases the
deficit is CPU/launch/small-kernel overhead, not quantized GEMM throughput.**
The session optimized numerics (already near roofline inside the GEMM); the
remaining multiples are in the execution path.
