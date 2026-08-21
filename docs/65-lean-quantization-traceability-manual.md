# 65 — Lean Traceability Manual for Quantization Research

**Purpose:** good-enough scientific rerunability and honest comparison, not
security-grade provenance or formal verification.

## 1. The 5% rule

Traceability-specific manual work and automated overhead should each consume no
more than roughly **5%** of an experiment lane's researcher time and run time.

Stock controls, actual EXL3 encoding, exact-byte measurement, validation KLD, and
runtime measurements are scientific evidence—not traceability overhead. Receipt
plumbing, cryptography, duplicate validation, and formal attestations are overhead.

A reusable trace mechanism should normally cost:

- ≤4 engineer-hours once;
- ≤1 minute per run;
- no recurring reviewer loop.

If it exceeds those ceilings, defer it unless an executable counterexample shows
it can flip the ranking, cross a frozen threshold, change exact bytes, reveal split
leakage, or expose a different decode route.

## 2. Threat model

### In scope

Prevent accidental:

- wrong source tensor/model/revision/BF16 decode;
- wrong data split, capture flow, H basis, seed, control, or candidate;
- unmatched bytes/search budget;
- wrong K/codebook/transform/target/correction;
- wrong KLD direction/head/suite/window/aggregation;
- hidden fallback/materialization route;
- cherry-picking and loss of key negative evidence.

### Out of scope

- malicious tampering by the repository/machine owner;
- cryptographic nonrepudiation or hostile multi-user authorization;
- formal proof that every possible assignment was measured;
- transitive attestation of every helper/package/environment variable;
- bit-identical reproduction on arbitrary future hardware.

## 3. Normative levels

- **MUST:** omission can plausibly change/invalidate the scientific conclusion.
- **SHOULD:** default; may be waived with one sentence when cost exceeds value.
- **MAY:** useful when already automated/cheap; failure never blocks.
- **DEFER:** not a gate. A reviewer may promote it only with a concrete
  conclusion-changing counterexample.

## 4. One lean canonical receipt

Every scientifically meaningful run records:

```text
run_id, UTC, stage, claim_scope
code_commit + dirty_patch_or_runner_hash
actual_EXL3_commit/extension/image when used
source_model_revision, tensor/module key, source hash
split_manifest hash + split label/selection hash
capture/H artifact hash, flow, basis/layout/count when used
candidate action: K, codebook, scale/transform/target/correction/path recipe,
                  params, seed, legal encoder-evaluation count
matched stock/control ID and budget statement
payload/checkpoint hash, exact bytes/components, decoder route
metric direction/config, suite/token/head/window/aggregation hashes
key scalar/tail result + raw report locator
status: success / scientific-negative / invalid-infrastructure
promote/stop reason + key competing/falsifying arm IDs
```

Reference immutable manifests rather than rehashing their internals for every arm.

## 5. Four-stage workflow

### Stage 1 — 15-minute exploratory screen

**Goal:** kill bad ideas cheaply.

MUST:

- code/source/tensor/data/control/candidate IDs;
- calibration label (no confirmation access);
- H basis/layout/count if used;
- pinned actual EXL3 and same-K stock arm for an EXL3 claim;
- config, seed, actual legal encoder evaluation count;
- metric name/normalization/aggregation (never call a proxy KLD);
- actual payload bytes/hash for a rate comparison;
- append key result or scientific falsifier.

SHOULD:

- use MPS broad screen and common random numbers;
- save only decisive raw rows and failures.

No model KLD or runtime benchmark required unless the screen changes format/hot route.

### Stage 2 — promoted tensor/module arm

**Goal:** establish a full-tensor mechanism worth a checkpoint screen.

MUST:

- all Stage-1 fields;
- frozen calibration/validation manifests with document disjointness;
- complete tensor action identity and exact H/capture identity;
- matched stock seed/order/search cap for direct method claims;
- ≥20 frozen blocks and full tensor where feasible;
- source-basis finite decode, exact serialized bytes, payload hash;
- winning, stock, and conclusion-bearing losing/falsifying rows.

SHOULD:

- one-tensor replacement KLD for high-impact arms;
- paired block output and Fisher-HWE where available.

### Stage 3 — full checkpoint candidate

**Goal:** choose one candidate for untouched confirmation.

MUST:

- one selected action per legal unit in one assignment file;
- candidate and F0-fresh checkpoint hashes and exact byte manifests;
- measured candidate registry and validation selection rationale;
- state **best measured**, not global optimum;
- validation full-vocabulary forward KLD/EAR/tails/top1;
- candidate/control/config/thresholds frozen;
- codec-exact route decode/fallback check;
- one real positive replay→evaluate fixture plus six negative cases (below).

Full production benchmarking waits until fidelity/bytes pass.

### Stage 4 — publication/promotion

**Goal:** support the exact claim, no more.

For model fidelity:

- open one frozen confirmation once for candidate, F0, and F0-fresh;
- retain raw per-context rows and method-derived summaries.

For rate:

- exact checkpoint bytes and component basis.

For deployment:

- codec-exact and claimed production routes;
- raw-output-derived runtime rows and causal semantic provenance;
- no fallback.

Archive code/patch, manifests, configs, selected assignment, raw reports, key
negative evidence, and artifact locators. State limitations.

## 6. Holdout opening without a security project

MUST:

1. Freeze candidate/control/config/threshold hashes before confirmation.
2. Write one atomic single-use open receipt (for example `O_EXCL`).
3. Never tune after reading confirmation.
4. Report failure honestly; a later candidate is exploratory or needs a reserve holdout.

SHOULD: an independent operator/reviewer opens/runs the test when convenient.

DEFER: HMAC service, process isolation, hostile-user authentication. Same-owner
cryptography does not prove blindness.

## 7. Search and frontier policy

- Log actual legal encoder evaluations, seeds, and data order.
- Match search cap for direct head-to-head algorithm superiority.
- Otherwise preregister/disclose unequal compute and compare outputs honestly.
- Store the measured arms and selected per-unit assignment.
- Say **best among measured registered candidates**.
- Exhaustive global proof is required only for an explicitly claimed finite exact
  optimum. Never require the Cartesian product of hundreds of unit actions.

## 8. Exact bytes and runtime

MUST for rate claims:

- returned buffers, scales, marker, sidecars, headers/alignment;
- decoded payload and checkpoint hashes;
- actual checkpoint component byte manifest.

Formula estimates are labeled estimates.

Runtime is staged:

- exploratory: decoder round-trip/fallback only if format/route changes;
- tensor arm: identify route and finite source-basis decode;
- checkpoint: codec-exact route;
- deployment claim: candidate/control on both relevant routes, raw rows, semantic
  allowlist (checkpoint, route/profile, image, patch/extension, GPU/driver, MTP,
  graph, KV, attention and causal routing variables).

DEFER recursive/transitive attestations and equality of unrelated environment values.

## 9. Minimum end-to-end negative tests

After one real positive replay→evaluate fixture, require only six
conclusion-bearing mutations:

1. wrong split/open state;
2. wrong source/candidate/checkpoint;
3. wrong suite/head/window/KL direction or top-k report;
4. wrong payload/action/byte manifest;
5. mismatched control/selection budget;
6. route fallback/runtime mismatch for a deployment claim.

Additional fuzzing is MAY and cannot block.

## 10. Review and iteration stop rules

### Reviews

Maximum two rounds:

1. Round 1 enumerates every core scientific blocker.
2. One corrective pass.
3. Round 2 verifies only those blockers.

Round 2 cannot invent requirements unless a new executable counterexample can flip
ranking/threshold/bytes, reveal split leakage, or exercise an undeclared route.

After Round 2: approve for the stated stage with caveats, or reject for a named core
invariant. Security/formal hardening becomes DEFER.

### Experiment iterations

- Exploratory: one preregistered sweep + one bug-fix rerun.
- Promoted tensor: one validation run + one identical-config retry only for invalid execution.
- Full checkpoint: one frozen validation build + one identical-hash infrastructure retry.
- Confirmation: one candidate plus F0/F0-fresh, opened once.
- Runtime: one warmup + three measured trials per preregistered row.
- Stop a lane after two valid iterations fail its minimum effect or immediately when
  actual EXL3/bytes/validation KLD falsifies it.

### First-bring-up exception

For the **first** end-to-end Wave-5 stock capture/runtime bring-up, allow up to four
invalid-infrastructure retries (container mounts, JIT cache, libraries, service
startup, capture wiring). These retries:

- keep the candidate method, K, source tensor, data, thresholds, and search budget frozen;
- change plumbing only and record the failure plus the exact configuration delta;
- do not count as valid scientific iterations;
- end immediately once one accepted baseline path runs.

After baseline acceptance, the normal limits above resume. This exception never
permits another untouched-test opening or a results-dependent method change.

A valid negative is not rerun until positive.

## 11. Cost of omission

| Control | Omission risk | Decision |
|---------|---------------|----------|
| Code/data/control/candidate identity | Cannot rerun or attribute | MUST |
| Split manifest/document disjointness | Leakage/cherry-picking | MUST for promotion |
| H/capture basis/flow/count | Can reverse effect silently | MUST when used |
| Actual EXL3 + fresh stock control | Recreates proxy-baseline failure | MUST for EXL3 claim |
| Search counts/key failures | Hidden search advantage | MUST for method comparison |
| Exact payload/checkpoint bytes | Invalid rate frontier | MUST for rate claim |
| Method-of-record metric/raw rows | Wrong KL/head/window/tails | MUST for fidelity claim |
| Candidate freeze + open receipt | Confirmation becomes training | MUST |
| Selected assignment/measured registry | Cannot reconstruct heterogeneous model | MUST |
| Semantic runtime provenance | Wrong route/fallback | MUST only for deployment claim |
| HMAC/process isolation | No added scientific rerunability in same-owner setup | DEFER |
| Per-callback transitive hashes | Brittle, incomplete behavioral attestation | MAY |
| Exhaustive frontier proof | Combinatorially impossible, unnecessary for best-measured | DEFER |
| Hash-chain attempt ledger | Same-owner chain is not tamper proof | plain JSONL |
| Exhaustive adversarial fuzzing | Diminishing value/fixture maintenance | DEFER |

## 12. Reusable operator checklist

### Before run

- [ ] claim scope chosen: proxy / tensor / model fidelity / deployment;
- [ ] source, data split, control, candidate, K, bytes, search cap frozen;
- [ ] actual stock EXL3 baseline for an EXL3 claim;
- [ ] test not accessed.

### After run

- [ ] source-basis finite decode and exact bytes;
- [ ] raw result + canonical receipt;
- [ ] matched control and key falsifier retained;
- [ ] promote/stop decision recorded;
- [ ] no proxy mislabeled KLD.

### Before confirmation

- [ ] one selected assignment/checkpoint hash;
- [ ] F0/F0-fresh hashes and bytes;
- [ ] validation gates passed;
- [ ] one-use open receipt armed;
- [ ] no pending hyperparameter choice.

### Before publication

- [ ] claim-specific evidence only;
- [ ] untouched paired KLD/EAR/p99/top1 for fidelity;
- [ ] exact bytes for rate;
- [ ] raw runtime evidence for deployment;
- [ ] key negatives and limitations stated;
- [ ] no more than two review rounds.

## 13. Current Wave-5 foundation disposition

- R29 data/Fisher contract: core scientific gate passes; approved pins.
- R30 exact EXL3 harness: actual stock path and identity smoke pass. Remaining
  production-validator/callback-loader issues are documented caveats; Phase-B
  codec-exact research may proceed serially. Production qualification remains gated.
- R31 fidelity contract: metric/split/freeze core is sufficient once the combined
  split-manifest selector mismatch is corrected. Security/formal-frontier hardening
  is deferred under this manual.

Phase B opens when Main verifies those cheap compatibility fixes, not when all MAY/
DEFER controls are implemented.
