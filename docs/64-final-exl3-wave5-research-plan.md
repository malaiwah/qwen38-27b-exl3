# 64 — Final Wave 5: Actual-EXL3, Real-Activation, Full-KLD Research Plan

**Status:** preregistered final research wave, 2026-08-21.

## 1. Mission

Find the lowest-byte, highest-fidelity Qwen3.8-27B recipe using the **actual current
EXL3 encoder, payload, decoder, and production routes**. Wave 5 converts the
R1–R28 screening hypotheses into causal, converter-compatible evidence.

Wave 5 does **not** assume one method should apply everywhere. The output is a
heterogeneous action assignment: different legal modules, roles, depths, and
coupled blocks may select different K, scales, transforms, targets, corrections,
codebooks, or the stock/no-op action.

Encode time may be extremely expensive. Decode is always the hot path and must
remain fast, graph-capturable, byte-accounted, and free of runtime fallbacks.

## 2. Why Wave 5 is necessary

Four waves established mechanisms but not a production result:

- almost every percentage is against an affine/uniform proxy rather than stock EXL3;
- activations are synthetic, H_G is usually output covariance rather than Fisher;
- many experiments use 128×128 diagonal crops and slice-local sidecar rates;
- local HWE/SSE can invert model KLD (measured independently in QSRT and EDA work);
- stock EXL3 already implements signs, H128, FP16 `suh`/`svh`, scale search,
  BlockLDLQ, MCG/MUL1, actual Viterbi, and fixed-stride packing;
- per-tile K/entropy/learned-codebook proxies generally require new formats/kernels;
- the production throughput profile may materialize trellis weights to FP4/FP6,
  so an exact-trellis improvement can disappear in production.

No R1–R28 claim is promoted before Wave 5.

## 3. Non-negotiable contracts

### 3.1 Data and sampling

- Correct BF16 decode only: shift uint16 payload into float32 high 16 bits; never
  reinterpret BF16 as IEEE FP16.
- Depths: layers 0, 7, 14, 21, 28, 35, 42, 49, 55, or nearest
  topology-compatible layers.
- Roles: gate/up/down; GDN qkv/z/out plus real conv/a/b/state inputs; gated full
  attention q/k/v/o and output-gate coordinates.
- Screening: ≥8 preregistered blocks/tensor (first/middle/last diagonal, seeded
  random row/column, off-diagonal).
- Promotion: ≥20 blocks/tensor and full tensors where feasible.
- Common broad census: `cleanroom/qwen38_wave5_weights.npz` from R24 is screening
  input only; its policy/scoring must be rebuilt under the split below.

### 3.2 Three-way validation

1. **Calibration:** fit covariance, Fisher, scales, paths, targets.
2. **Validation:** select action, K, alpha, seed, shrinkage, stopping point.
3. **Untouched test:** opened once after freeze for reported evidence.

Split by complete documents/conversations and domain (code, prose, dialogue,
multilingual), never random activation rows. Hash every data manifest. Candidate
and control receive identical search/evaluation budgets and common seeds.

### 3.3 Stock EXL3 control

Every comparison includes a fresh stock current EXL3 control with identical:

- source tensor and K;
- sign streams/H128;
- `suh`/`svh`, scale search, damping/search budget;
- `block_ldl(b=16)` recurrence and actual `quantize_tiles`/Viterbi;
- MCG/MUL1 marker unless codebook is the declared variable;
- packing/alignment and fixed-stride decode route;
- calibration/validation/test sets and number of legal encoder evaluations.

A proxy uniform quantizer is never called EXL3.

### 3.4 Complete action schema

Each legal unit `g` gets a menu of complete actions `a`:

```text
Action = {
  unit/topology group,
  K,
  codebook marker,
  sign/scale/transform recipe,
  target recipe,
  curvature/correction recipe,
  Viterbi/refinement recipe,
  exact serialized buffers/bytes,
  decoder/hot route,
  startup/runtime contract,
  calibration/validation evidence
}
```

Correction, target, transform, scale, K, and codebook are **inside** the action.
Never allocate from RTN curves and append correction afterward.

### 3.5 Metrics and terminology

- `MSE`: source-basis weight error.
- `OC-HWE`: output-covariance proxy.
- `Fisher-HWE`: gradient-covariance weighted error.
- `block output error`: actual MLP/attention/GDN propagated output.
- `KLD`: only frozen-reference, shared-BF16-head, full-vocabulary model output.
- Also report p99/CVaR1%, top-1, worst contexts, and full-vocabulary
  `EAR = mean_i sum_v min(p_iv, q_iv)`.

Local metrics shortlist. Only untouched model KLD/EAR/p99 promote.

### 3.6 Bytes and runtime

- Actual serialized checkpoint bytes, sidecars, codebook IDs, headers, alignment,
  scale/path metadata.
- Resident memory, scratch/transient peak, cold start, conversion, compilation,
  graph capture, context capacity.
- Decode at production MTP/concurrency rows; prefill at M≈2k and 6k.
- Qualify twice:
  1. codec-exact all-trellis (`VLLM_EXL3_MULTIPRECISION=0`);
  2. unchanged production throughput profile with FP4/FP6 materialization.
- New runtime transforms/kernels must demonstrate no unacceptable TG/PP/graph regression.

## 4. Heterogeneous action optimization

Select one complete action per legal unit:

\[
\min_x \sum_{g,a} C_{g,a} x_{g,a}
\]

subject to:

- one action per unit;
- predicted/validated mean KLD, p99/CVaR, EAR/top-1 bounds;
- topology/fusion/route constraints;
- startup/runtime/context constraints;
- exact serialized byte budget.

The dual form minimizes fidelity loss at an exact byte/runtime budget. Use
multiple-choice DP/ILP plus targeted interaction terms and fresh sequential
rebuilds. Report action frequency by role/depth and explicit exceptions.

### Deployable default granularity

1. per topology/fused group;
2. per tensor/module/shard (integer K; current format);
3. fixed stripe only after a fused fixed-stride kernel exists;
4. per-tile K only after format/kernel qualification.

Per-tensor K5/K6 mixing already provides arbitrary whole-model fractional bpw and
is the required control for any Q-Palette/fixed-stripe proposal.

## 5. Phase A — shared prerequisites (three researchers, parallel)

### R29 — Real data, activation, and Fisher census

Deliver:

- correct weight manifest for nine depths/all roles with tensor hashes and source offsets;
- real FP-flow and running quant-flow activations from source-disjoint documents;
- gate/up hidden, down inputs/teacher outputs, attention/GDN topology traces;
- input H_X, output covariance diagonal, Fisher diagonal, selected block-16 Fisher;
- calibration/validation/untouched-test manifests;
- MPS screening dataset and aiboss capture scripts/receipts.

GPU discipline: stop the service for exclusive GPU work and restore healthy
`throughput` defaults afterward. Keep ≥60 GB free. KLD BF16 reference is reused;
no repeated BF16 capture.

### R30 — Exact stock EXL3 action/encoder harness

Deliver a reusable `EXL3Action` API that:

- invokes pinned actual EXL3 encoder/Viterbi/decoder;
- supports stock K4/K5/K6 and declared K7/control codebook only where qualified;
- exposes stock signs/scales/H128/BlockLDLQ/MCG/MUL1 controls;
- serializes and decodes full tensors in source basis;
- records exact bytes and decode route;
- accepts encode-only scale/target/curvature/path callbacks;
- proves strength-zero/stock action is bit-identical to fresh stock;
- provides broad MPS screens and aiboss exact callbacks to Phase B.

No cleanroom `chol(inv(H)).T` is substituted into stock block-LDLQ.

### R31 — Fidelity protocol, KLD/EAR, and statistical gate

Deliver:

- preregistered data/function split and arm registry;
- full-vocabulary KLD mean/CI/p99/CVaR/top1/EAR calculators;
- one-tensor replacement and full-checkpoint KLD workflow;
- paired document/context clustered bootstrap;
- equal-search/multiplicity policy;
- SLQ-style minimum-byte fidelity constraints;
- runtime/startup/context gate schema;
- immutable `F0` shipped and `F0-fresh` stock controls.

Phase B begins only when R29–R31 contracts pass independent review.

## 6. Phase B — seven final research axes (parallel after Phase A)

### R32 — Zero-byte scale/path co-refit (PiSO/Four-Over-Six/BiIP)

Actual-stock arms at fixed K/bytes:

- stock scales/search;
- BiIP magnitudes in existing `suh`/`svh`;
- path-frozen alternating input/output scale fit;
- scale-frozen equal-budget Viterbi rerun;
- 2–3 rounds scale→Viterbi with strict validation acceptance;
- finite selector-free scale family around stock `{0.8,0.9,1.0,1.1,1.2}` and
  `{s0,2s0/3}`;
- int8/int4 scale serialization only if existing decoder semantics permit.

Use actual FP16-in-loop scales. Zero incremental bytes/hot op is required for the
main arm. Falsifier: no untouched KLD/p99 benefit vs search-matched stock.

### R33 — Real covariance shrinkage and stock block-LDL variants (DASH/QMM)

Using real activations:

- `H_rho = diag(H) + rho(H-diag(H))`, rho `{0,.05,.1,.25,.5,.75,1}`;
- block-diagonal/banded stock recurrence at block sizes 16/32/64;
- Ledoit-Wolf control and sample-count sweep;
- Fisher-HWE/OC-HWE and untouched KLD selection;
- WaterSIC/QMM innovation-rate gap as an early stopping oracle.

Define truncation in the actual stock block recurrence, not a separate GPTQ proxy.
Falsifier: validation selects endpoints or interior rho fails untouched KLD.

### R34 — Candidate-conditioned dense block targets and seams (QSRT/ICBQ)

For every decoded gate/up K/transform action:

- form candidate-specific hidden `H_q` and teacher block output `Y`;
- fit ridge/no-ridge down target;
- stock-encode target at each K and discard continuous target;
- rebuild down covariance/target for every upstream action;
- compare source target vs reconstructed-upstream target;
- compare one sweep vs matched-compute block-seam reroll;
- test U_A-only/U_B-only legal activation-boundary H128 transforms, with U_R=I
  first and BF16 identity proof;
- include actual GQA/output-gate and full GDN topology where relevant.

Same payload mainline; QSRT runtime transforms are a separately timed custom arm.

### R35 — Schur-conditioned legal-path refinement inside every RD action

Start from valid stock BlockLDLQ/Viterbi payload. For each K/target/transform:

- derive exact conditional dense-H block target with suffix response;
- generate only legal paths with actual Viterbi;
- accept only strict reduction of the same full dense-H objective;
- compare stock vs one/two sweeps with matched Viterbi-call controls;
- factorial with source vs reconstructed-upstream target;
- rerun independently for every K before allocation.

Falsifier: local objective decreases but held-out block/KLD does not.

### R36 — Final-KL gradient and differentiable legal trellis (QSRT/BCJR/GSQ/QES)

Phase-gated:

1. hard final-KL gradient target shift at exact materialized anchor;
2. strength-zero payload identity and decoder-adjoint inner-product tests;
3. relinearize after each accepted hard payload;
4. random legal search equal-evaluation control;
5. low-temperature BCJR over actual EXL3 state graph if hard-path signal exists;
6. legal-transition Gumbel and zero-order QES ablations;
7. every result hard-projects to the stock payload/decoder.

Start on one high-attribution full K5 tensor. Stop if broad 20+ block screen has no
stable hard-path signal; QSRT's prior signal was tiny/uncertain.

### R37 — SLQ-style exact fidelity frontier and per-tensor action allocation

Using R30 complete actions and R31 metrics:

- measure full K4/K5/K6 curves with each action's own correction/target;
- direct single-group full-KLD/EAR marginals;
- targeted Shapley/interactions for fused/high-sensitivity groups;
- exact serialized-byte DP/ILP;
- mean-KLD, p99/CVaR, EAR/top1 constrained frontiers;
- hierarchical candidate menus coupling MLP triples, gated attention, and GDN;
- compare stock K5/K6 module mixing against every fractional/new-format claim.

Final selection uses fresh sequential conversions. Proxy/local EDA allocation is a
known negative control.

### R38 — Conditional fixed-stride fractional K5/K6 and runtime prototype (Q-Palette)

Only proceed if R37/QMM diagnostics show residual value beyond module mixing.

Compare exact bytes:

- best whole-module K5/K6 mixture;
- uniform K5 and K6;
- deterministic within-shard K5/K6 stripe (format ID, no selector map);
- per-tile selector oracle (nondeployable upper bound).

Prototype two packed fixed streams and honest two-launch+sum. Authorize a fused
one-launch kernel only if quality beats module mixing enough to justify it. Report
SM120 decode/prefill/graph behavior. Published Q-Palette K≤5 kernels are guidance,
not K5/K6 proof.

## 7. Phase C — exact candidates and deployment confirmation

Minimum full-checkpoint set:

1. `F0`: shipped incumbent, immutable.
2. `F0-fresh`: fresh stock conversion under Wave-5 data/search contract.
3. `F1`: best same-K/same-byte encoder action.
4. `F2`: stock encoder + per-tensor K allocation at `F0` exact bytes.
5. `F3`: heterogeneous best-action allocation at `F0` exact bytes.
6. optional `F4`: conditional fixed-stripe/custom-runtime arm.

For each: actual serialized bytes, exact and production profiles, startup, resident
memory, graph capture, context, PP/TG, full-vocabulary KLD/EAR/p99/top1. Only Phase
C can produce a promoted Qwen3.8-27B recipe.

## 8. Wave-5 stop and success criteria

### Stop local-optimization lane when

- post-stock QMM innovation gap ≤≈0.1 effective bit across the target strata;
- an oracle action cannot improve hard K5/K6 reconstruction under actual EXL3;
- proxy improvement fails block output or validation KLD;
- new format/runtime cannot beat free module K mixing at exact bytes/throughput.

### Success

At least one fresh sequential candidate must:

- improve mean and/or p99 KLD beyond the current same-engine frontier at equal or
  lower exact transformer-body bytes;
- retain acceptable PP/TG/context/startup/graph behavior;
- introduce no fallback or unqualified decode route;
- reproduce on untouched documents and the frozen fidelity suite.

North-star equal-body target: beat stock uniform EXL3 K6 mean KLD `0.001583` at
≤`17.0537 GiB` transformer-body bytes, while retaining the project's runtime gates.

## 9. Repository/worktree contract

Final Wave 5 uses ten isolated branches/worktrees `research/r29-*` through
`research/r38-*`. Phase A interfaces are shared by immutable receipts/hashes; Phase
B does not fork its own data, encoder, bytes, or metric conventions. Every researcher
spawns an adversarial reviewer. Main independently reviews any promoted claim and
publishes ntfy progress. Failed arms and exact falsifiers are documented so they are
not repeated.
