# R31 — Immutable fidelity, EAR, and deployment protocol

**Status:** APPROVED FOR RESEARCH — lean Phase-A contract schema v2, frozen before Wave 5 candidate results (2026-08-21).

## Decision

Wave 5 has two mandatory fidelity axes and two separate deployment axes. The
body-only axis is teacher-forced, text-only
$KL(p_{BF16}\Vert q_{candidate})$ on the exact stored token IDs of
`malaiwah/qwen38-27b-fidelity-suite-v5` revision
`7797fcce3ffed62b99871348887f4626dc9b2b3b`. Both operands use the one shared BF16
LM head (`sha256 25a30fd5...4cfff`). The served axis replays the identical candidate
capture through the candidate's own frozen BF16 head and independently applies
the same absolute and paired-noninferiority gates. The exact candidate head file
hash, tensor dtype/shape, checkpoint identity, capture hashes, report, and row
payload are bound together. A body-only win cannot satisfy the served-head gate.
Codec-exact and production runtime qualification remain separate from both
fidelity axes; none of the four axes may substitute for another.

The implementation and machine-readable thresholds are in
`tools/research/wave5/fidelity_gate.py` and
`tools/research/wave5/fidelity_contract.json`. The data/function split, access
grants, arm registry, and hash bindings are in `receipts/wave5/fidelity-prereg.json`.
Frozen artifact pins after the targeted self-test passed:
`gate=f4fc059c...e8482`, `contract=e8e1d476...66783`, and
`prereg=75a81665...8a784`.



This is a scientific rerunability contract, not a security boundary. Traceability
is intentionally limited to the evidence that can change a scientific conclusion:
code/data/control/candidate identities, exact candidate/control/action bytes, split
label and selector, metric configuration, thresholds, and key results/failures.
Security-grade provenance and formal verification are explicitly out of scope.

## Frozen data/function split

R29 supplies one combined immutable split manifest, not three different manifest
files. Its file SHA256 is `a7eab6e2...ee09cc` and canonical-content SHA256 is
`151c4115...96a2e`. The three labels are distinguished by exact selectors
`{field: split, op: eq, value: <label>}` and selection/projection SHA256 values:
calibration `490a1596...a259`, validation `4c5cf19a...a5de`, and untouched test
`4eaaa72d...bca`. Requiring different manifest-file hashes per label is a contract
error; requiring different selector/projection hashes is correct.

- **Calibration:** fresh, source-disjoint EXL3-standard and UltraChat-train
  documents. Methods may fit covariance/Fisher, scales, transforms, targets,
  paths, and callbacks here. Calibration cannot choose a reported arm.
- **Validation/model selection:** v5 shard 0, using its retained BF16 capture.
  It is the only split allowed to choose an action, K, alpha, seed, shrinkage,
  stopping point, interaction, or byte-frontier point. The retained BF16 shard-0
  reference is never recaptured.
- **Untouched confirmation:** the R29-frozen, hash-selected source-cluster-disjoint
  v5 contexts from shards 1–9. UltraChat test-gen documents in R29's broader split
  have no v5 context records and are mechanically excluded from the primary metric.
  Method researchers cannot open the split. The public CLI has one canonical
  candidate-freeze path, one canonical open-state path, and one canonical access
  ledger; it exposes no path override. Opening requires the owner-only capability,
  validates the complete candidate/control/action/budget/head hashes, and atomically
  creates an `O_EXCL` state containing its own hashed grant event. Exactly one
  matching grant must exist in the SHA256-chained canonical ledger. Denials after
  trust-root load enter that ledger, and a second use of the opening capability
  fails closed without printing or storing the secret. Missing shards1–9 BF16
  references may then be captured once, hashed, retained, and reused for candidate,
  `F0`, and `F0-fresh`.

The v5 suite has historical use. “Untouched” therefore means unopened during
Wave 5 method development under this new source-cluster policy, not never seen by
any prior project experiment. Existing reports are context only and do not
replace fresh candidate/control captures.

v3 revision `73252e77` remains an external multilingual/domain stress control;
its analysis and qualification sets are not source-disjoint and its absolute
values are not transferable to v5. `qwen38-frontier-g01-capability` and
`qwen38-27b-terminal-bench-2.1` are separate capability/long-context/multimodal
and agentic gates. GLM MTP78/Fruit data are not Qwen calibration.

## Full-vocabulary metric contract

For every scored position $i$ and every one of 248,320 tokens $v$:

$$
D_i = \sum_v p_{iv}\log\frac{p_{iv}}{q_{iv}}, \qquad
EAR_i = \sum_v \min(p_{iv},q_{iv}) = 1-TV(p_i,q_i).
$$

The method-of-record walks vocabulary in 24,832-token chunks, uses float32 inside
each chunk, and accumulates in float64 over two passes. It scores post-final-norm
rows 0–2046 of each 2,048-token context. Reports contain token and context-macro
mean KLD, source-cluster bootstrap CI, p50/p95/p99/p999, exact maximum, fixed
560-bin tail histogram, CVaR of the upper 1%, full-vocabulary EAR, top-1
agreement, JSD bits, worst contexts, and per-context/document/stratum rows.
Every bootstrap replicate retains all positions in each sampled source cluster
and recomputes global pooled per-position p99 and CVaR; per-context or per-shard
quantiles are never averaged.
Because every context has 2,047 scored positions, token and context-macro means
coincide; both labels remain explicit.

The promotable `replay` command does not accept probabilities or caller-authored
metric rows. It loads the hash-pinned method-of-record projector, validates every
v5 shard manifest/token/context record against R29, validates every hidden-state
capture manifest and file digest, verifies the BF16 source identity and candidate
checkpoint identity, loads the shared head at its exact digest/shape, and performs
the two-pass projection itself. It rejects partial contexts, wrong positions,
unknown strata, altered bootstrap settings, nonfinite metrics, and noncanonical
test state. The normalized-vector calculator is named `synthetic-metrics`, emits
`qwen38-wave5-synthetic-metrics/1` with `promotable=false`, and cannot satisfy the
evaluator.

MSE, OC-HWE, Fisher-HWE, local-softmax KL, hidden-state distance, block-output
error, and any top-k/truncated calculation are proxies. They are never labeled
“KLD” or “EAR”. Shared-head KLD is body-only and is never called served-logit
fidelity. The separately labeled `candidate-own-head/full-vocabulary` report is
capture-bound and promotable only when it independently passes every served-head
gate.

## Dependence and multiplicity

The source family/cluster is the independence unit. Each interval replicate
samples source clusters with replacement and retains every complete context and
scored position in a chosen cluster: 10,000 percentile resamples, seed 1.
Candidate and control use identical draws. Mean, global pooled p99, global pooled
CVaR1%, EAR, and top-1 are recomputed from every draw. Strict wins, strict losses,
and exact ties are reported separately. Per-shard or per-context quantiles are
never averaged; exact rows or mergeable fixed histograms are combined.

Tensor and block samples are not independent. A local screen first forms a
candidate/stock ratio within an identical block, then gives each tensor one
macro value, each tensor one vote in its role/depth cell, each depth one vote in
a role macro, and each role one vote globally. Raw-error pooling is forbidden.
Local screens only shortlist; only full-model output metrics can promote.

Every registered candidate arm and `F0-fresh` receives the same count of actual
legal EXL3 encoder/Viterbi evaluations and the same seed/document/block order.
The SHA256-chained attempt ledger reconciles arm, setting, ordinal, seed, split,
selection hash, and common-random-number hashes; untouched-test rows are rejected.
Unused budget is forfeited rather than donated. Validation binds one legal
assignment, one frontier point, one threshold digest, and one complete checkpoint
hash into the selection decision. That single frozen contrast is confirmatory on
the once-open test. Thresholds, seeds, arms, and stopping rules cannot change
after any test metric is read.

## Frozen controls and thresholds

`F0` is the immutable shipped hydrated revision
`ab3a91a13813df8096cb4c1d560ed3669035d0cf`. `F0-fresh` is a fresh current-stock
actual-EXL3 conversion from the frozen source under R30's stock action and the
same recipe/search budget. Practical equivalence requires the entire paired 95%
interval of `F0-fresh − F0` to lie within ±0.00025 mean KLD, ±0.003 global
per-position p99, ±0.006 global per-position CVaR1%, ±0.00025 EAR, and ±0.00025
top-1. If it fails, stock freshness must be investigated; the test is not reopened.

The preregistered absolute body-fidelity constraints are mean-KLD CI upper bound
≤0.012, p99≤0.12, CVaR1%≤0.25, EAR≥0.97, and top-1≥0.97. A lower-byte candidate
must also be paired-noninferior to `F0-fresh`; exact thresholds are in the JSON
contract. It must save at least max(1 MiB, 0.01% of fresh-stock bytes). An
equal-byte candidate promotes only if paired mean-KLD and EAR intervals establish
strict superiority while all noninferiority gates pass. These values were frozen
before Wave 5 results; measured replay/engine systematics are disclosed and never
subtracted as if KL were additive.

## SLQ-style primal and dual

The primal problem minimizes the sum of **exact integer serialized bytes** of one
complete R30 action per legal topology/fused unit, subject to the frozen mean,
tail, EAR/top-1, paired noninferiority, format/route, startup/runtime, graph, and
context constraints. Bytes include payload, headers, alignment, signs, scales,
codebook IDs, selectors/path metadata, sidecars, and every candidate-only decode
artifact. A proxy-additive solution must be rebuilt sequentially and remeasured
before confirmation.

The dual uses the hash-projected legal action menu and one canonical complete
action per unit. Each point binds a canonical assignment, its exact integer byte
sum, a validation-only full-vocabulary measurement, and an assignment equivalence
class. Equal lexicographic metric tuples are ordered by assignment SHA256. The
reachable measured budget set and threshold digest are frozen, every cheaper
point participates in the dominance check, and the selected point binds the
complete candidate checkpoint hash. A deterministic two-unit oracle independently
enumerates the Cartesian product and must exactly match solver output. Interaction
terms are themselves complete, rebuilt actions; correction is never appended
after allocation.

## One-tensor and full-checkpoint workflows

A one-tensor replacement starts from a new copy of `F0-fresh`, replaces exactly
one registered tensor's serialized buffers/metadata, proves every non-target hash
unchanged, loads the declared R30 route with fallback counters armed, captures
once, and replays through the retained BF16 reference/shared head. This is a
causal marginal screen, not promotion evidence.

A full candidate freezes its complete per-unit action registry, exact-byte
manifest, and checkpoint hash after validation. The once-only test opener then
permits fresh captures of candidate, `F0`, and `F0-fresh` on identical contexts.
Replay emits paired full-vocabulary metrics, strata, tails, and worst contexts.
The codec-exact and unchanged production profiles are then qualified separately.


The dual frontier is exact only over the complete **declared screened action
registry**. It does not claim a global optimum over unregistered actions. The
targeted exhaustive oracle checks solver semantics on its small screened menu;
larger experiments retain the declared menu, exact action bytes, validation
measurements, chosen assignment, ties, and failures.

## Deployment gate

Every report uses R30's exact route IDs, never an inference from K/codebook:

- `codec-exact/all-trellis-stock-exl3`, with
  `VLLM_EXL3_MULTIPRECISION=0`, compared with `F0-fresh` on the same route;
- `production/throughput-fp4-fp6-materialized`, unchanged throughput defaults,
  compared with shipped `F0`.

Candidate, F0-fresh, and shipped F0 report exact serialized/component bytes
through canonically content-identified manifests whose ordered files are re-hashed
and stat-sized by the evaluator; F0-fresh additionally binds the frozen stock
action. Runtime receipts separately report resident/scratch/transient peak, cold
start, conversion, compilation, graph capture, context capacity, PP near
M=2,048 and 6,144, TG/MTP at concurrency 1 and 4, and fallback details. Each
candidate/control runtime receipt binds the image digest, vLLM and build revisions,
GPU UUID/name, driver, attention backend, KV dtype, MTP count, graph mode, profile,
checkpoint, route, served-head hash, open-state hash, startup receipt, runtime
receipt, raw harness evidence, and an exact effective-environment object plus
receipt. Candidate and same-route control runtime-stack identities must match.
Missing, stale, or mismatched provenance fails closed. Every PP/TG row must retain
at least 95% of its same-profile control, cold start at most 110% of control and
360 seconds, context at least 238,400 tokens, graph capture must succeed, and
fallback count is zero. Production additionally requires PP(M≈2k)≥7,000 tok/s,
TG-fox≥180 tok/s, and TG-essay≥90 tok/s.

## Verification

`fidelity_gate.py self-test` deterministically verifies KL zero, positivity, and
direction; JSD zero; full-vocabulary EAR=$1-TV$; linear p99 and fractional-boundary
CVaR; seed-1 source-cluster draws; global pooled tail resampling; truncated
vocabulary rejection; and rejection after an actual checkpoint file is changed
behind its hashed byte manifest. Access tests verify split hashes, candidate/checkpoint/head
freeze fields, equal budgets, direct-test denial, once-only state/path behavior,
and hash-chain tampering. The test-mode opener deliberately bypasses the real
secret check; it verifies state mechanics, not isolated owner authentication.

Adversarial helper tests reject malformed replay headers (reverse KL, top-k,
mismatched suite/head/window/context, invalid split/tail labels, and body/served
conflation), stale runtime/environment summary pointers, threshold mutations, and
a missing served-head CLI argument. The small frontier helper independently
enumerates a two-unit Cartesian product. These are unit-level contract checks:
they do not invoke full `replay_command`, `_validated_report`,
`_validate_action_assignment`, or `evaluate_command` with positive and mutated
end-to-end fixtures.

The production evaluator does rerun the pinned projector for submitted
untouched-test fidelity reports and compares regenerated aggregates, breakdowns,
capture bindings, and metric rows. It also enforces separate body/served reports,
global tails, one legal action per unit, exact candidate bytes, and submitted
runtime/frontier hashes. The limitations below prevent calling the overall
promotion path approved.

## Review disposition and deliberate caveats

Final scoped independent `openai-reviewer` verdict: **APPROVED** against
`gate=f4fc059c...e8482`, `contract=e8e1d476...66783`, and
`prereg=75a81665...8a784`. The approval applies to the lean scientific contract
and its explicit caveats, not to the deferred security/formal hardening.

The first corrective review confirmed candidate-own-head promotion and global
pooled tails, then requested security-grade provenance, owner-keyed opening,
formal full-registry frontier derivation, and exhaustive full-path adversarial
fixtures. Those requests do not change the current scientific conclusion and
would exceed the project's traceability budget, so they are deferred rather than
misrepresented as implemented.

The implemented research gate remains fail-closed for the scientifically
load-bearing contract: final approved R29 file and canonical-content pins; one
combined split manifest plus exact per-label selectors/projections; frozen
candidate/control/action/head identities and thresholds; full-vocabulary forward
KL and EAR lineage; paired source-cluster inference with global position tails;
actual-file-rehashed serialized candidate/F0-fresh/F0 bytes and screened-menu action sums; separate
body-only, served-head, codec-exact, and production axes; one-use manual
untouched-test state; and recorded results/failures.

Deliberate caveats:

1. Runtime/startup/raw receipt files are hash-bound but not independently parsed
   into every promoted summary. Before a real promotion, an operator must inspect
   the raw receipt bundle and record any mismatch as a failed run.
2. Frontier optimality is limited to the complete declared screened menu. No
   global optimality claim is permitted.
3. The manual test-open capability is not an authenticated service boundary.
   Repository/host access control remains operational policy.
4. Adversarial self-tests exercise metric, access, header, tail, threshold,
   runtime-pointer, CLI, and small-oracle helpers rather than every full production
   path. A real promotion still requires the actual projector rerun already
   performed by `_validated_report`.

Under this lean scope, unresolved security/formal hardening is not a Phase-B
blocker. No candidate result, untouched-test opening, runtime qualification, or
promotion claim is made by R31 itself.
