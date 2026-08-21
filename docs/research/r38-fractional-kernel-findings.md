# R38 — Conditional fixed-stride K5/K6 format and SM120 runtime

**Status:** final conditional stop, 2026-08-21. R37's frozen Wave-5 intake does not open R38, so no quality/runtime experiment or kernel is authorized. No untouched-test input has been opened.

## Decision question

R38 asks whether deterministic fractional K5/K6 *inside* an EXL3 shard adds enough fidelity beyond the free whole-module K5/K6 frontier to pay for a new checkpoint format and hot route. Whole-module mixing is the control, not an affine-uniform approximation. The stopping rule is deliberately asymmetric: encoding/search may be expensive, but a custom decode route must earn its permanent complexity.

The preregistered order is:

1. consume R37's validation-only, exact-file-byte best measured whole-module K5/K6 control and R33's QMM diagnostic;
2. only if that residual gate says `proceed`, encode actual R30 stock-EXL3 K5/K6 stripes and the selector oracle on calibration, then select on validation;
3. only if the fixed stripe beats module mixing by the engineering margin below, benchmark the honest unfused route on SM120;
4. author a fused one-launch kernel only after the quality gate passes.

No local MSE, OC-HWE, Fisher-HWE, block error, interpolation, or published Q-Palette result can open steps 2–4. Only validation full-vocabulary model-output KLD may authorize the kernel.

## Final gate and disposition

R37 froze `receipts/wave5/r37-slq-frontier.json` at SHA256 `c48969dec4f1a7a544d89f67ab606672ff5fde80ae5a31d3537c8618dda36c72`. Its final gate is:

- `proceed_r38=false`;
- `exact_target_bytes=null`;
- `selected_assignment=null`;
- measured assignment-bound QMM headroom is `null`;
- all R32–R36 lanes are closed with zero eligible full-vocabulary action rows.

R33's approved receipt SHA256 `f542a58aefab9d9164bf29182969fb67ed82f24c23b8d7705b679b179535876a` explicitly corrects its earlier `4.86` values: they are Gaussian SNR-equivalent achieved bits, not residual QMM headroom. They cannot open R38. With no exact target, no selected whole-module assignment, no eligible full-vocabulary row, and no measured residual-QMM premise, a stripe comparison would violate the preregistered control and selection contract.

R38 therefore stops after the reviewed theory/packer prototype. It did not request a GPU window, encode an actual stripe quality arm, run SM120 benchmarks, or author a kernel. The determination is deliberately narrow: **the current measured Wave-5 frontier does not justify a new K5/K6 fractional format/kernel**. This is not a claim that actual fixed stripes were measured and lost; they were correctly not run because the prerequisite gate failed.

## Fixed-stride format semantics

The prototype format ID is `exl3-fs1-input-h128-half-k5-prefix-k6-suffix`.
For a source matrix with stored shape `[out_features, in_features]`, it fixes equal input-coordinate halves at an H128 boundary:

- K5 stream: input columns `[0, in_features/2)`;
- K6 stream: input columns `[in_features/2, in_features)`;
- both streams cover every output coordinate;
- reconstruction is the source-basis concatenation of the two column stripes;
- inference evaluates the K5 stream on the input prefix and the K6 stream on the input suffix, then sums their output vectors.

The half ratio and phase are properties of the format ID, not per-shard choices. The header repeats the derived split for fail-closed validation and carries dimensions, fixed stream offsets/sizes, and both stream SHA256 values. The rule itself derives every coordinate's K, so selector-map bytes are exactly zero. The input must contain an even number of H128 blocks, and output width must be a multiple of 128. This alignment is load-bearing: stock EXL3's input/output H128 blocks cannot be cut in half. Each stream remains a complete actual R30 action with its own K, MCG/MUL1 marker, FP16 input/output scales, target, curvature/correction and legal trellis payload. Independent stream encoding may lose cross-stripe correction; that is a measured property of the candidate, not something the format hides.

The minimum hot route is **three launches**: one stock K5 EXL3 GEMM, one stock K6 EXL3 GEMM, and one elementwise sum/cast kernel. Both stock GEMMs return BF16 partial outputs. The sum kernel converts each partial to FP32, evaluates `float32(K5) + float32(K6)` in that order, and casts once to BF16 at the module boundary; Qwen's covered linears are bias-free, and any future biased unit must apply bias exactly once after the sum. If the stock kernels cannot consume the last-dimension half views with their original leading stride, activation gathers/materializations and their launches/bytes are additional measured work—the three-launch count is a lower bound, not a zero-copy claim. The design matches the algebra of Q-Palette's input-partitioned `CombtLinearTCQ` only as inspiration: at Q-Palette commit `025f27f311774d4618fcc0d47e4c8d33ee568a6c`, unequal input partitions call two kernels and add their outputs. That source supports neither stock EXL3 semantics nor a K5/K6 SM120 claim, and none is imported here.

## Packer and actual-byte contract

[`r38_fractional_kernel.py`](../../tools/research/wave5/r38_fractional_kernel.py) deliberately does not implement a quantizer or a clean-room trellis simulator. It treats two files from the pinned R30 actual-EXL3 harness as opaque byte streams and writes one deterministic container:

1. 256-byte fixed descriptor/header;
2. actual K5 stream file at a 256-byte boundary;
3. zero-filled alignment padding;
4. actual K6 stream file at a 256-byte boundary;
5. zero-filled final padding to a 256-byte file size.

Inspection re-derives canonical offsets and file size, rejects empty streams and noncanonical format/reserved/padding bytes, checks the format and stripe invariants, and verifies both embedded file digests. Extraction rejects resolved-path aliases and must reproduce two distinct input files byte-for-byte. Because the inputs are complete files, their real safetensors headers are counted rather than estimated. A real arm must additionally bind each embedded file hash, K, shape, source coordinates and complete action identity through R30/R31; opaque prototype bytes alone cannot claim actual EXL3 provenance. For a checkpoint claim, R31's canonical ordered actual-file manifest remains authoritative; the prototype container's `stat` size is not replaced with a formula.

For one R30 MCG/MUL1 stream, the returned-buffer law is

\[
B_K = \frac{mnK}{8} + 2(m+n) + 4,
\]

where `m=in_features` and `n=out_features`. For K5 prefix width `m_5` and K6 suffix width `m_6`, `m_5+m_6=m`, the two raw streams use

\[
B_{5\mid6} = \frac{n(5m_5+6m_6)}{8} + 2m + 4n + 8.
\]

Relative to the equal-fraction interpolated one-stream/module-mix law, the unavoidable raw duplication is exactly

\[
B_{5\mid6}-B_{\mathrm{module\ mix}} = 2n+4
\]

bytes: one extra FP16 output-scale vector and one extra int32 marker. The actual 256-byte descriptor, two real stream-file headers, and measured alignment are additional. The selector map is still zero bytes.

For R30's representative `[1024,5120]` tensor and a half split, the exact returned-buffer controls are K5 `3,289,092 B`, K6 `3,944,452 B`, and two-stream K5.5 `3,618,824 B`. The equal-fraction whole-module interpolation is `3,616,772 B`, so raw stripe overhead is `2,052 B` or `0.003131103515625` effective bpw before the actual container/file overhead. These are byte-law results, not an EXL3 quality or checkpoint-frontier claim.

## Controls if the residual gate opens

Every measured quality comparison must use actual R30 encode/serialize/decode actions and identical source, split, seed, codebook, target/correction family, and legal encoder-evaluation budget.

| Arm | Semantics | Byte treatment |
|---|---|---|
| Uniform K5 | one complete stock K5 action per legal unit | actual checkpoint files |
| Uniform K6 | one complete stock K6 action per legal unit | actual checkpoint files |
| Whole-module mix | R37 best measured complete K5/K6 assignment at the target | required primary control; best at `<=` stripe actual bytes, with slack explicit |
| Fixed stripe | the deterministic equal-half H128 input split above | two actual stream files, descriptor, both headers, all alignment |
| Selector oracle | choose K5/K6 independently per actual 16x16 trellis tile | nondeployable relaxed quality diagnostic; report selector bit/header/alignment and total bytes, but do not treat it as an exact-budget control |

The selector oracle is never called a format candidate and cannot promote or stop FS1. Its independently encoded per-tile feasible set is not proven to contain the two-complete-stream FS1 action, and counting its selector sidecar changes payload available under an exact cap; “quality ceiling” therefore means only the relaxed same-K6-tile-count diagnostic, with its larger actual bytes reported separately. The stripe ratio and phase are frozen by FS1. Every target, correction, codebook, seed and stopping rule is fitted/frozen on calibration before validation. Untouched test remains closed throughout Phase B.

## Preregistered engineering margin

A fused kernel is authorized only when a sequentially rebuilt, actual-file-byte fixed-stripe checkpoint satisfies all of the following against R37's best measured whole-module K5/K6 control at no more bytes:

- define the paired contrast as `delta = KLD_fixed_stripe - KLD_whole_module_mix`, so negative is better;
- every registered candidate/action and all parameters are frozen from calibration before any validation metric is read;
- if validation chooses among `N` registered contrasts, use R31's simultaneous paired source-cluster procedure or a Bonferroni upper bound at `alpha=0.05/N`; the multiplicity-adjusted 95% upper bound for `delta` must be at most `-0.00025`, not merely below zero;
- when the control is worse than F0-fresh, the fixed stripe's mean improvement must also be at least 5% of that excess;
- p99 and CVaR1% are noninferior within R31's frozen, multiplicity-adjusted limits, and EAR/top-1 do not regress beyond their frozen paired limits;
- exact candidate/action/checkpoint identities and actual-file byte manifests validate under R31.

The `0.00025` simultaneous-margin floor equals R31's frozen practical-equivalence mean band. It prevents selection on the same validation cohort from manufacturing a kernel authorization: unregistered tuning after validation is a failed run. If the fixed stripe is cheaper rather than byte-equal, it must meet R31's lower-byte noninferiority rule and the same simultaneous `0.00025` benefit before fused-kernel work. A local full-tensor reconstruction win is only a shortlist; it cannot satisfy this gate.

## Conditional SM120 benchmark route

If quality authorizes runtime work, the first implementation is the unfused route above using the actual eligible stock EXL3 K5 and K6 kernels. It freezes BF16 partial outputs, FP32 `K5+K6` accumulation order, and one BF16 boundary cast. The separately measured sum/cast kernel and any required activation gather/materialization are part of candidate time, resident/scratch bytes and graph capture; zero-copy is claimed only after the real stock kernels accept the strided half views. Candidate and same-route control must share image, extension/patch, GPU, driver, graph, KV, MTP, attention and routing identities, with zero fallback.

One warmup plus three measured trials are required for each row:

- decode GEMM `M=1` and `M=8`;
- prefill GEMM near `M=2,048` and `M=6,144`;
- CUDA graph capture and replay for every supported row;
- end-to-end codec-exact all-trellis qualification, then unchanged production materialization only if it has an explicitly qualified converter/route;
- startup, compile/conversion time, resident bytes, stream bytes, descriptor/header/alignment bytes, activation-slice gather/materialization bytes and launches, scratch/transient peak, and context capacity.

Every PP/TG row must retain at least 95% of its same-profile stock control; startup must be at most 110% of control and 360 seconds; graph capture must succeed; context must remain at least 238,400; fallback count must be zero. Production additionally retains PP near 2k at least 7,000 tok/s, TG-fox at least 180 tok/s, and TG-essay at least 90 tok/s. A compile-only CUDA result, CPU/MPS timing, Q-Palette RTX4090 coefficient, reconstructed-weight GEMM, or affine-uniform kernel is not an actual benchmark route.

## Verified prototype evidence

The targeted self-test performs a real file pack/inspect/extract cycle with deterministic 701-byte and 1,003-byte opaque streams. The actual container is 2,048 bytes: descriptor 256, K5 file 701, inter-stream alignment 67, K6 file 1,003, final alignment 21, selector map 0. Its deterministic SHA256 is `82458b0453cb22a98d806a902bb9e9598ef9968a2ca561a8e5470535e9a83718`. Twelve checks pass:

1. pack and independent inspect;
2. actual component-byte sum;
3. deterministic no-selector coordinate semantics;
4. byte-identical opaque stream extraction;
5. extraction output/container alias rejection;
6. corrupted K6 stream rejection;
7. noncanonical format-field rejection;
8. empty embedded-stream rejection;
9. non-H128 input-width rejection;
10. selectable/non-half stripe rejection;
11. non-H128 output-width rejection;
12. the R30 two-stream raw-byte law.

This proves the descriptor, fixed-stride semantics, file/header/alignment accounting and packer round trip. The synthetic stream contents are intentionally **not** evidence of EXL3 quality or runtime.

The required adversarial review used two rounds. Round 1 found canonical-header/nonempty-stream and extraction-alias bugs plus three scientific contract ambiguities (oracle comparability, validation multiplicity, and sum/copy semantics); all were corrected. Round 2 approved runner SHA256 `4b91e7ce4d495b334c14854c3e74ab1d58c50433b863b5492bf1cad031d38651` with zero remaining blockers. R37's later final freeze retained the same null gate fields and therefore does not introduce a new review requirement.

## Final scope boundary

R38 did not encode a quality arm, open validation outputs for R38 selection, use aiboss GPU, benchmark SM120, or author a CUDA kernel. `measured_quality_rows` and `measured_runtime_rows` are deliberately empty. No actual benchmark claim exists for a reviewer to mistake for the counterfactual route specification above. The final machine-readable disposition lives in `receipts/wave5/r38-fractional-kernel.json`.
