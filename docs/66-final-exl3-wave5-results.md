# 66 — Final Wave 5 Results: Actual-EXL3 Research Disposition

**Status:** final research-wave result, 2026-08-21. **No checkpoint promoted.**
**Postscript (2026-08-21):** the capture launcher was repaired and the
strength-zero control was captured and replayed — see §8.

## 1. Executive result

Wave 5 replaced the earlier affine/uniform proxies with an actual pinned EXL3
encoder/decoder, real Qwen3.8 activation/Fisher captures, exact payload bytes, and
a lean frozen fidelity contract.

The actual-stock screens did **not** produce a validation full-vocabulary action
row. Five mechanism lanes were falsified or too small locally; one scale lane was
locally positive but model-validation-blocked. The one-tensor capture launcher
materialized a valid sibling but did not complete hidden capture before the
first-bring-up retry cap. Therefore:

- no per-tensor KLD/EAR/p99 marginal was admitted;
- no action allocation/frontier was solved;
- no fixed-stripe/new-kernel work was authorized;
- the untouched test was never opened;
- the shipped checkpoint remains the production recommendation.

This is a successful negative research result: the large R1–R28 proxy gains do
not yet demonstrate incremental headroom over stock EXL3.

## 2. Foundation

### R29 real data/Fisher census — approved

- Actual 64-layer topology: 48 GDN + 16 gated full-attention blocks.
- 87 BF16 tensors at nine depths with shard/tensor/shape/offset/block hashes.
- BF16 shift-decode proved bit-exact; BF16-as-FP16 rejected.
- Source-disjoint split policy: six pinned EXL3 calibration documents, v5 shard0
  validation/common replay, sealed v5 confirmatory partitions, leakage audit.
- 36 real module-boundary BF16 and throughput-materialized captures.
- Real H_X, covariance/block statistics, and selected sequence-Fisher captures.
- Service restored healthy; 79GB remained free.

Canonical pins are in `receipts/wave5/data-manifest.json` and
`split-manifest.json`; findings in `docs/research/r29-data-fisher-findings.md`.

### R30 actual EXL3 action harness — codec-exact research approved

- Actual pinned EXL3 seeded signs, H128, stock scale search, `block_ldl(16)`/
  LDLQ, CUDA Viterbi, packing, FP16 `suh`/`svh`, LinearEXL3 source-basis decode.
- Full layer-3 K5 stock encode/decode finite; exact raw payload 3,289,092 bytes.
- A0 and strength-zero payload/reconstruction identity proved.
- K4/K5/K6 and MCG/MUL1 controls finite.
- Current operational extension/harness pins are recorded in the foundation and
  action receipts; historical stock receipt keeps its historical pins.
- Four production-validator/callback-loader hardening issues are documented as
  nonblocking for serial codec-exact research under doc 65.

### R31 fidelity protocol — lean approved

- Full-vocabulary forward `KL(BF16 || candidate)`, EAR=1-TV, global pooled tails,
  top1, clustered inference, body-only and candidate-own-head separation.
- One combined split manifest plus exact selection projections.
- Calibration/validation/untouched-test freeze semantics and one-use open receipt.
- Exact checkpoint byte manifests and screened-menu—not global-optimum—selection.
- Security-grade HMAC/process isolation/exhaustive frontier proof deferred.

### Traceability policy

[Doc 65](65-lean-quantization-traceability-manual.md) caps traceability near 5%
of experiment effort, limits review to two rounds, and retains only
conclusion-bearing identities/evidence. A four-retry first-bring-up exception
was used for runtime plumbing; untouched confirmation remained one-open only.

## 3. Phase-B mechanism results

### R32 zero-byte scale/path refit — local-positive, KLD-blocked

Actual full L55 gate, K5 MCG, all actions exactly 55,750,660 bytes:

| action vs A0-S | MSE ratio | OC-HWE ratio | result |
|----------------|----------:|-------------:|--------|
| A0 | 1.000055 | 1.015329 | stock variation |
| BiIP identity | 1.085412 | 1.813346 | falsified |
| BiIP Fisher | 5.500191 | 21.597850 | strongly falsified |
| finite-five scale family | 1.039172 | 0.766876 | HWE/MSE tradeoff |
| finite-two scale family | 1.036892 | 0.663799 | best HWE, worse MSE |
| path→Viterbi round 1 | 0.999677 | 0.998217 | tiny positive |
| path→Viterbi round 2 | 0.999456 | 0.995635 | tiny positive |
| path→Viterbi round 3 | 0.999143 | 0.992238 | +0.086% MSE, +0.776% HWE |

The finite scale family changes error direction but regresses MSE. The
same-byte path refit shows only sub-percent local headroom. Without an accepted
validation capture, no action was eligible for R37.

### R33 real covariance shrinkage — scientific negative

Actual stock full-tensor campaign with real-input H found stock rho=1/full
curvature best on L0/L55 under local HWE. No candidate advanced to KLD. The
reported 4.86-bit diagnostic is a Gaussian SNR-equivalent value, not measured
WaterSIC/QMM headroom; the preregistered QMM oracle was not run.

The nine-depth block screen stopped before its first arm on a block-digest
mismatch; it was invalid infrastructure and contributed no scientific row.

### R34 reconstructed-upstream down/seam — scientific negative

L55 gate/up K5 + down K6, same down payload 66,891,780 bytes:

- source seam: +7.80% validation block MSE regression;
- reconstructed down target: +23.10% regression;
- reconstructed + seam: +30.98% regression.

The preregistered block-output gate stopped before KLD. BF16 identity and fresh
stock strength-zero passed. Raw action role labels were schema-invalid, but no
row was promoted and the current runner is corrected. QSRT-style runtime boundary
transforms were stopped because BF16 inverse-H128 was not bit-exact and no fused
cost was available.

### R35 Schur legal-path refinement — scientific negative

- 144 actual-stock local screen rows; 125 strict local wins.
- Full L0 gate K4/K5/K6 dense-H changes: −0.0179%, −0.0598%, −0.1215%.
- Source-H changes remained below 0.45%.
- Fixed bytes and equal Viterbi-call controls passed.

Without an R34 target win or validation KLD, the tiny local signal was excluded.

### R36 final-KL hard legal paths — scientific negative

- Fresh stock K5 strength-zero payload/decode identity passed.
- Decoder-adjoint legal-displacement checks passed at ~1e-11 absolute error.
- Guided 20-tile hard screen: 11 negative, 7 positive, 2 unchanged final-KL
  linear terms; preregistered gate required at least 16 negative.
- Guided aggregate linear term was better than random, but block signs were
  unstable.

No validation KLD action row. Conditional BCJR/Gumbel/QES was correctly not run.

### R37 SLQ frontier — no eligible action rows

R32–R36 supplied no direct validation full-vocabulary action row. R37 retained
an interim/final scientific-negative receipt, produced no assignment or exact
frontier, and did not synthesize proxy values. Solver/tie oracle self-test passes.

### R38 fractional K5/K6 — conditional STOP

A canonical two-stream fixed-stripe packer/descriptor was implemented and
reviewer-approved. The honest minimum three-launch runtime contract (K5 GEMM +
K6 GEMM + sum/cast) was specified, not implemented or measured. R37's zero-row
gate did not authorize quality, GPU, runtime, fused-kernel, or format work.
No new format claim exists.

## 4. Capture launcher disposition

The launcher successfully:

- materialized a one-tensor layer55 K6 strength-zero sibling;
- preserved source/action/changed-shard/checkpoint lineage;
- kept disk above the floor and restored throughput after every attempt.

An initial failed rerun plus all four permitted first-bring-up retries failed
before hidden row 1 (five executions total):

1. missing `/opt/fp4/exl3_fp4_conversion.py`;
2. remote-shell brace expansion of service-only JSON;
3. OOM after about 88 cached ~170-MB gate/up tensors;
4. a second OOM after about 97 cached tensors, which identified and disabled the
   independent `VLLM_EXL3_PRE_RECONSTRUCT` hook;
5. vLLM `collective_rpc` hook callable not serializable (next known plumbing
   delta would be `VLLM_ALLOW_INSECURE_SERIALIZATION=1`).

A final full-mount rerun reached healthy engine warmup and frozen candidate
identity, then hit the hook-serialization failure. Under doc 65's retry cap, the
lane closed. No BF16 reference was recaptured, no legacy data deleted, temporary
siblings were cleaned, throughput returned healthy, and untouched test remained
closed.

## 5. What Wave 5 establishes

1. **Stock EXL3 is the required baseline; the tested Wave-5 arms did not
   reproduce the earlier large proxy gains as eligible validation evidence.**
2. **BiIP scaling is not an incremental default:** it regressed actual local MSE/HWE
   for the tested full tensor.
3. **On the tested full L55 gate tensor, zero-byte scale/path search showed
   sub-percent local headroom rather than the earlier proxy-scale gains.**
4. **Real full curvature beat shrinkage in the tested modules.**
5. **R34 reconstructed-down/seam failed its validation block-output gate; R35
   Schur refinement remained calibration-local and was excluded without
   validation evidence.**
8. **The validation KLD infrastructure is now operational** (§8). The
   strength-zero control measured mean KLD 0.003409, matching production
   0.003405. No Wave-5 candidate action has been replayed, but the lane is
   open for any future candidate that produces a local win large enough to
   justify the GPU time.

## 6. Production disposition

- Keep the shipped checkpoint/profile unchanged.
- Keep `frontier-g02-prep` research artifacts; do not merge a quantization method
  into production code.
- The capture launcher has been repaired (§8). The strength-zero control was
  captured and replayed, producing the first validation KLD row from the
  actual-EXL3 pipeline. R32's finite-two and scale-rerun-r3 candidates remain
  available for replay if future work warrants it, but the sub-percent
  one-tensor local gains make a full-model KLD improvement unlikely.

## 7. Artifacts

- Foundation: `tools/research/wave5/{data_capture,exl3_action,fidelity_gate}.py`,
  schemas/contracts, `receipts/wave5/{data,split,stock,fidelity}-*.json`.
- Phase B: `tools/research/wave5/r32_*.py` through `r38_*.py`, matching findings
  under `docs/research/`, receipts under `receipts/wave5/`.
- Lean process: [doc 65](65-lean-quantization-traceability-manual.md).
- Preregistered plan: [doc 64](64-final-exl3-wave5-research-plan.md).

## 8. Postscript: capture launcher repaired, strength-zero control validated

After the final research wave closed, the capture launcher's two infrastructure
bugs were fixed and the strength-zero control was run end-to-end through the
R31 validation gate.

### Fixes

1. **`VLLM_ALLOW_INSECURE_SERIALIZATION=1`** (commit `38ce6fe`): added the env
   var to the podman capture command in `candidate_capture.py:767`. Without it,
   vLLM's `MsgspecEncoder.enc_hook()` raises `TypeError` on `FunctionType` when
   `collective_rpc` ships the post-final-norm hook to the EngineCore process.
   With it, cloudpickle serializes the callable. Every other KLD capture script
   in the repo already set this env var; the Wave 5 launcher omitted it.

2. **Gate `max_batched_tokens=0` acceptance** (commit `e6e34d2`): the pinned v5
   BF16 reference manifest records `max_batched_tokens=0` (old fidelity.py
   default meaning "use ctx_len=2048"), but `fidelity_gate.py:778` required
   exactly `2048`. Changed `!= 2048` to `not in (0, 2048)`. The candidate
   manifest already had `2048`; only the reference failed the check.

### Strength-zero control results

The R30 strength-zero control (layer-55 `k_proj`, K6, strength=0.0 — a
codec-exact re-encoding of the production tensor) was materialized, captured,
and replayed through the R31 gate:

| metric | strength-zero control | production fidelity |
|---|---:|---:|
| mean KLD | 0.003409 | 0.003405 |
| p99 KLD | 0.03476 | 0.0349 |
| top-1 agreement | 0.9760 | — |
| EAR | 0.9804 | — |

CI95 (cluster bootstrap, 10,000 resamples, 330 clusters): mean KLD
[0.003168, 0.003677], p99 [0.031036, 0.039487].

512 contexts × 2047 positions = 1,048,064 scored positions. Capture took 347 s;
replay took ~300 s.

The strength-zero KLD (0.003409) matches the production fidelity profile KLD
(0.003405) within 0.000004, confirming the materialization pipeline is
codec-exact and the capture/replay infrastructure produces valid measurements.

### What this changes

The capture launcher lane is now open. R32's finite-two and scale-rerun-r3
candidate actions remain on aiboss and could be replayed if future work
warrants. However, the local gains were sub-percent on one tensor (L55 gate, 1
of 409 modules), so a full-model KLD improvement over the strength-zero control
is unlikely to be distinguishable from noise.

The research conclusion stands: no checkpoint promoted, stock EXL3 remains the
production recommendation.
