# Retrospective peer review: Qwen3.8-27B EXL3 research record

**Review date:** 2026-08-20  
**Scope:** the complete repository research record through `docs/59-unsloth-dynamic3-research.md`, every receipt artifact, every materially supporting source file, all referenced Hugging Face model cards, and the project-authored upstream issue/PR set plus directly overlapping fixes.  
**Machine ledger:** [`docs/60-peer-review-ledger.json`](60-peer-review-ledger.json)  
**Deterministic checker:** [`tools/verify_peer_review_inventory.py`](../tools/verify_peer_review_inventory.py)

## Verdict

The repository contains a strong, unusually well-preserved empirical record, but the record is not uniform. Current v5 fidelity receipts, finalized checkpoint manifests, runtime-image identities, and narrow qualification gates are load-bearing. Several early or speculative conclusions were later superseded, and a few publication surfaces continued to repeat invalid interpretations after the underlying evidence had improved.

The most serious defect was the subtraction of a BF16 llama.cpp-to-vLLM KL value from GGUF candidate KL to infer a format-only “net” value. KL is neither additive nor a metric. The control proves engine confounding; it supplies no algebraic correction and no quantization-only upper or lower bound. That verdict is withdrawn everywhere current, including the K6-parity card and receipt generator. Historical receipts remain immutable and are identified as superseded interpretations.

Review outcome across **186 registered findings**: **141 fixed**, **24 later resolved by stronger evidence**, **6 dismissed after verification**, **5 open research/tooling items**, and **10 blocked publication/upstream actions**. Six findings were critical. No GPU run, credential use, model upload, issue edit, PR edit, or receipt overwrite was performed.

## Audit proof and coverage

The audit read the record rather than sampling it:

- **Markdown:** 186 pre-report files, 2,870,500 bytes, 42,865 lines: 67 under `docs/`, 79 receipt narratives, 22 under `upstream/`, 14 root documents, two tool bodies, one chart note, and one shared Hub-serving profile. Numbered documents were reviewed in filename order, including duplicate-number variants and the concurrent additions in docs 59 and the two 2026-08-20 comparison receipts.
- **Receipts:** 1,939 files, 292,133,494 bytes. SHA256 was streamed over every byte. The set contains 1,283 strict JSON documents, 113 valid JSONL streams, 31 valid gzip files, 492 other UTF-8 artifacts, and 12 binary artifacts. Duplicate JSON keys were not found.
- **Supporting source:** 243 code files under `tools/`, `patches/`, and `docker/`, 4,782,893 bytes, plus the remaining upstream and configuration support files. All 163 Python files parsed as Python AST and all 53 shell files passed `bash -n` at the audit snapshot. Material fidelity, aggregation, serving, chart, parity, YaRN, EDA, and upstream-patch paths received manual semantic review.
- **External cards:** 120 discovered Hugging Face identifiers. 112 current READMEs were fetched at exact 40-hex model revisions, decoded, hashed, and read in full; two accessible repositories had no README; four referenced cards were unavailable/private; two strings were false-positive non-card references. Rendered pages were also crawled for claims hidden by presentation.
- **Upstream:** 47 project-authored or directly overlapping issues and PRs were read at current state. The older 33-item audit was not treated as complete.

The ledger records every current repository path, byte size, SHA256, parse status where applicable, external-card revision and README hash, finding status, upstream verdict, and verification result. The only exclusion is the ledger's own digest, because a file cannot contain its own finite SHA256 fixed point.

## Chronological reconciliation

| Epoch | Documents | What survives | What later evidence changed |
|---|---|---|---|
| Checkpoint construction | 01–05 | Mixed EXL3 roles, BF16 overlays, final manifests, and fail-closed finalization remain the correct artifact model. | Early byte/headroom arithmetic and some width language were corrected; final manifests supersede estimates. |
| Runtime bring-up | 06–11 | Direct `vllm serve`, pinned rootfs/image identity, Qwen-specific patches, and bounded smoke gates remain supported. | Startup success is not throughput, long-context quality, or general portability. Quesma chart transcription was independently verified. |
| Fidelity protocol formation | 12–20 | Hidden-state capture through one shared BF16 head is a useful body-only, teacher-forced distribution comparison. | Early one-window, mean-only, and unpinned results are historical; v5 suite/head/candidate identities supersede them. |
| Allocation and causal probes | 21–29 | Held-out comparison, role interventions, and paired per-context analysis are the right direction. | Greedy allocation is not globally optimal; circular additive self-checks are not validation; calibration and evaluation must remain disjoint. |
| v5 method and landscape | 30–42 | The frozen v5 suite, source-cluster bootstrap, cumulative ladder, tails, own-head ablations, and exact artifact identities are the method of record. | Cross-engine “net” KLD is invalid; cross-suite and cross-estimator ratios are not measurements; the EDA sqrt-energy allocator failed independent re-solving. |
| Serving, tools, and performance | 43–52 | Qualification is multi-axis: startup, text, vision, tools, retention, long context, throughput, and resource behavior are separate gates. | Capability probes do not establish quality; timeouts do not identify causes; profiler conclusions are configuration-specific; plans are not results. |
| Cost, speculation, EDA, prior art | 53–59 | Cost arithmetic, speculative-decoding candidates, direct marginal KLD from exllamav3, and external landscape review provide useful decision inputs when scoped. | Hardware-independent speedup transfer, six-prompt fidelity conclusions, cross-model KLD, single-card training assumptions, and universal quantizer laws were retired. |

Later evidence resolves many early gaps without rewriting history. The correct reading order is: preserve the early hypothesis, read its later receipt, then use the current card/README conclusion. `PROGRESS.md` is chronology, not the current queue.

## Current load-bearing conclusions

1. **Artifact identity is strong.** Current finalized manifests separate logical from packed tensors, record payload and document digests, and fail closed on unsupported layouts. Published runtime claims still require the exact image digest and patch hashes named by their receipt.
2. **The v5 KLD ladder is the strongest fidelity comparison in the repo.** Over 10,480,640 scored positions and 842 source clusters, the same-engine/shared-head means are: hydrated 0.002760, online K5/K6 0.003210, context edition 0.003509, official FP8 0.005294, K4 0.010604, and NVFP4 0.031059 nats/token. These are body-only, teacher-forced, text-only values; they are not generation, vision, throughput, or long-context results.
3. **Tails matter.** Mean KLD alone hid meaningful p99/p99.9 behavior. Current publication charts now show means, clustered intervals, tails, top-1 agreement, protocol, and size-axis definitions.
4. **GGUF comparisons are complete-pipeline comparisons only.** The shared contexts and BF16 head make the observed rows comparable as delivered pipelines, but llama.cpp versus vLLM prevents format-only attribution. A same-engine capture or one common forward implementation is required for that claim.
5. **The output head is measured rather than assumed.** Own-head ablations show a modest but nonzero contribution for relevant candidates; the shared-head protocol intentionally isolates the body.
6. **The propagation-aware allocation direction survives; sqrt-energy does not.** Pinned exllamav3 source confirms that one candidate group is substituted and propagated through remaining reference modules before final KL is evaluated. That is a direct end-to-end marginal for one group, though greedy composition can still interact and needs held-out validation.
7. **Serving gates are orthogonal.** A 200 response, a needle retrieval, a cache integrity check, and a throughput number answer different questions. LMCache remains disabled because connector reuse corruption was measured and its open fixes have not requalified this stack.
8. **Static YaRN has a resolved short-context effect in the measured probe.** Native-262k versus 1M-YaRN is about 0.0106 top-20 KLD, roughly nine times the observed boot-associated background. The native-1M arm was separately booted and had a different KV pool, so it does not prove a zero window effect. No request beyond 262,144 tokens tested native extrapolation quality.
9. **The official FP8 identity is complete despite a null revision field in one receipt.** All 66 safetensor LFS SHA256 values and the index hash match the pinned official revision recorded in F081.

## Method review and better experimental design

### What was done well

- Immutable inputs, exact suite/head identities, source-cluster bootstrap, per-context rows, tails, and explicit candidate/runtime metadata.
- Paired comparisons for same-context candidates and fail-closed aggregate identity checks.
- Negative results and later corrections were usually preserved rather than erased.
- Source-level hypotheses were frequently converted into narrow intervention experiments.

### Recurring weaknesses

- **Estimand drift:** body-only KLD, own-head KLD, top-20 serving KLD, task accuracy, and complete-pipeline cross-engine KLD were occasionally ranked as if one scale.
- **Causal overreach:** a one-variable intent does not guarantee a one-variable execution when boots, KV pools, graph capture, scheduler state, or engines differ.
- **Power without paired variance:** within-arm CV does not estimate the variance of a paired difference.
- **Calibration leakage risk:** optimizing and evaluating on the same contexts rewards the search procedure rather than generalization.
- **Performance mechanism by arithmetic:** traffic/FLOP estimates narrow hypotheses; only profiles identify the active bottleneck.
- **Historical executability:** some old generators can still reproduce withdrawn semantics. Current generators now fail closed or emit corrected framing where changed, but historical preregistration source remains evidence of what was actually registered.

### Recommended protocol

1. Pre-register the **estimand**, causal contrast, identities, exclusion rules, and stopping rule—not only a threshold.
2. Freeze three disjoint sets: converter calibration, allocation/search, and final held-out evaluation.
3. For format claims, capture both candidates in one engine and numerical path. If impossible, publish only complete-pipeline results and the engine control separately.
4. Use per-context paired differences and source-cluster bootstrap; report strict wins, losses, and ties.
5. Validate allocator changes on a second held-out corpus and with direct propagation-aware marginals before composing multiple role moves.
6. Require strict JSON (`allow_nan=False`), atomic replacement, canonical path-independent content hashes, and explicit schema/version changes for corrected receipts.
7. Separate capability, fidelity, performance, memory, long-context, vision, and tool-use gates. Do not let one proxy close another axis.
8. Upstream one defect per clean branch. Attach the smallest reproducer and the measured correction; avoid stacked history that makes review provenance ambiguous.

## Receipt integrity findings

Eight `.json` paths are not strict single-document JSON:

| Path | Exact condition |
|---|---|
| `receipts/gdn-gate-raw/out/reorder-sim.json` | two concatenated JSON objects |
| `receipts/lever-ab-local-5090.json` | a valid JSON object followed by prose |
| `receipts/lmcache-raw/out/lmcache-status-L0-pre.json` | empty capture |
| `receipts/lmcache-raw/out/lmcache-status-L0-post.json` | empty capture |
| `receipts/prefill-pp-chunk8k.json` | multiple JSON documents |
| `receipts/scratch-arena-raw/bench-arena.json` | multiple JSON documents |
| `receipts/scratch-arena-raw/bench-baseline.json` | multiple JSON documents |
| `receipts/yarn-short-context-penalty.json` | non-standard `NaN` values; its causal prose also contains the superseded F186 interpretation |

There are 19 zero-byte raw captures in total: ten LMCache status/metric captures and nine APC qualification stdout/stderr captures. Zero bytes can be evidence of “no output,” but it must be described by a parent receipt rather than parsed as JSON. The ledger marks each path individually.

One immutable Markdown receipt, `receipts/nsys-utilization-2026-08-19.md`, links to nonexistent `receipts/peer-review-2026-08-19.md`; **[INFERENCE]** the likely intended earlier source is `receipts/peer-review-2026-08-18.md`, whose review contains the cited overhead conclusion. It is documented here instead of silently rewriting a published receipt.

No duplicate JSON keys were found; every JSONL and gzip artifact parsed. Raw multi-document streams are evidence, but their `.json` suffix is misleading. Future corrected successors should use JSONL or a containing array and preserve the originals.

## Source review and applied corrections

Critical source changes:

- `tools/fidelity.py` and `tools/kld_aggregate.py`: strict win/loss accounting plus explicit ties; schemas advanced. Identical inputs now report zero wins and all contexts tied.
- `tools/fidelity.py` and `tools/qwen38_kld.py`: removed undocumented `kv_cache_memory_bytes` overrides so fidelity launch behavior matches the documented runtime contract.
- `tools/kld9_receipt.py`, `tools/parity_card.py`, and `tools/parity-card-body.md`: withdrew the invalid K6 parity rule, derive publication values from receipts, and emit only a cross-engine complete-pipeline observation.
- `tools/yarn_penalty_analyze.py`: strict JSON, atomic output, null empty-disagreement summaries, direction-correct truncation language, and a schema-2 causal account that keeps boot/KV-pool confounding explicit.
- Chart generators: no KL subtraction, distinct resident and serialized axes, protocol-explicit legends, and corrected type annotations. All eight SVG/PNG variants regenerated.
- Publication cards, README, dataset cards, shared Hub profile, chronology, and late research notes: corrected arithmetic, causal scope, chat-template security constraints, EDA guidance, and supersession status.

Published receipt bytes were not overwritten. Corrections live in source, current publication surfaces, and this review.

## External model-card review

The current-card audit is exact-revision evidence, not a claim that a card never changed. The ledger records each model revision and README SHA256. Quesma chart values were independently extracted from the live DOM and match docs 11 row-for-row. Official FP8 payload identity was verified from Hugging Face LFS metadata, not inferred from the card prose.

All seven locally maintained publication cards reviewed here now differ from their current Hub READMEs after this correction pass: K4, K5K6, context, hydrated, K6 parity, S16 research, and EDA research. This is expected locally but unsafe as a publication state. F184 is blocked until an authorized owner deliberately syncs each Hub README and verifies the resulting revision.

## Upstream audit

No upstream mutation was made during this review. Current decisions:

- PR398 is **not** a duplicate of merged PR234. PR234 guards the one planned `q_len`; PR398 extends keying to the broader captured wrapper family, including reduced draft depths and multiple shapes.
- PR397 is retargeted to the right base but still carries a broad 15-file stack. PR319 remains on a stale base. PR436 is similarly stacked and quotes body-only 0.002700 as embedding evidence; the actual measured int8 embedding delta is about +0.000065.
- Issues 409 and 410 need follow-up corrections from later experiments: measured marginal for 409, measured null for 410. Issues 406 and 407 already received corrections.
- Upstream vLLM PR47272 merged on 2026-08-20 and fixes null-block startup capacity. PR52530 remains changes-requested; close it or retain only an independently justified request-side contract.
- LMCache issue 4247 has an active fix PR, but no merged fix or stack-specific requalification. Keep LMCache disabled.
- Two encountered defects were not filed: B12X trellis warmup keyed only by device instead of `(device, bits)`, and QKV launch fragmentation/tuple serialization from the kernel-gap work. They are F179 and F180.

| Item | Current state | Review verdict |
|---|---|---|
| <https://github.com/local-inference-lab/vllm/issues/311> | open | Valid online-overlay shape bug; no maintainer response recorded. |
| <https://github.com/local-inference-lab/vllm/pull/312> | open | Scoped companion fix; still awaiting maintainer review. |
| <https://github.com/local-inference-lab/vllm/issues/313> | open | Vision-token truncation report remains open and evidence-backed. |
| <https://github.com/local-inference-lab/vllm/pull/314> | open | Dense EXL3 graph-decode implementation remains open. |
| <https://github.com/local-inference-lab/vllm/pull/316> | open | Reconstruct-plus-hgemm prefill dispatch remains open. |
| <https://github.com/local-inference-lab/vllm/pull/318> | open | K6 MCG routing fix remains open. |
| <https://github.com/local-inference-lab/vllm/pull/319> | open | Stale codex-base branch; should be rebased or replaced before review. |
| <https://github.com/local-inference-lab/vllm/issues/392> | open | Cherry-pick request remains open; its user-report evidence was correctly retracted. |
| <https://github.com/local-inference-lab/vllm/pull/393> | open | Cherry-pick PR remains open with the regression evidence intact. |
| <https://github.com/local-inference-lab/vllm/issues/394> | open | Fork-side align-admission livelock report remains valid; stock reproduction moved to vLLM issue 52520. |
| <https://github.com/local-inference-lab/vllm/issues/396> | open | V2 replay fault report remains open. |
| <https://github.com/local-inference-lab/vllm/pull/397> | open | Base is now dev/gilded-gnosis, but the 15-file stacked diff still obscures the scratch-arena change. |
| <https://github.com/local-inference-lab/vllm/pull/398> | open | Not a duplicate of merged PR234; it extends shape keys to the wider captured wrapper family. |
| <https://github.com/local-inference-lab/vllm/issues/402> | open | Hybrid connector corruption report stands; LMCache remains disabled for published workflows. |
| <https://github.com/local-inference-lab/vllm/pull/403> | open | Necessary guard was measured insufficient and the PR says so; no merge recommendation. |
| <https://github.com/local-inference-lab/vllm/issues/405> | closed | Accidental prepared-drafts issue was closed; no technical claim should cite it as a filed report. |
| <https://github.com/local-inference-lab/vllm/issues/406> | open | K6 dispatch regression remains open; correction comment is present. |
| <https://github.com/local-inference-lab/vllm/issues/407> | open | in_proj_ba decode finding remains open; correction comment is present. |
| <https://github.com/local-inference-lab/vllm/issues/408> | open | Silent FP8-prefill no-op is a valid fail-open configuration defect. |
| <https://github.com/local-inference-lab/vllm/issues/409> | open | Copy-and-concatenate report needs the later measured marginal correction. |
| <https://github.com/local-inference-lab/vllm/issues/410> | open | Scratch overlap hypothesis needs the later measured null correction. |
| <https://github.com/local-inference-lab/vllm/issues/411> | open | Chunk-size reconstruction result is measured but configuration-specific. |
| <https://github.com/local-inference-lab/vllm/issues/412> | open | MTP head-stream accounting is evidence-backed and remains open. |
| <https://github.com/local-inference-lab/vllm/issues/435> | open | Row-indexed trellis reconstruct feature request remains the clean upstream dependency for embedding work. |
| <https://github.com/local-inference-lab/vllm/pull/436> | open | Embedding implementation works locally, but is stacked, history-heavy, and cites the wrong KLD quantity. |
| <https://github.com/local-inference-lab/vllm/pull/437> | closed | Superseded first speculative-skip PR; closed. |
| <https://github.com/local-inference-lab/vllm/pull/438> | closed | Superseded second speculative-skip PR; closed. |
| <https://github.com/local-inference-lab/vllm/pull/439> | open | Current speculative-skip PR includes CodeRabbit follow-up and remains open. |
| <https://github.com/local-inference-lab/vllm/issues/440> | open | V1 eager MTP-loop regression remains open. |
| <https://github.com/local-inference-lab/vllm/issues/442> | open | External draft architecture-routing bug remains open. |
| <https://github.com/local-inference-lab/vllm/pull/234> | merged | Narrow planned-q_len guard is merged; it does not cover every shape family addressed by PR398. |
| <https://github.com/local-inference-lab/b12x/issues/232> | open | W4A8 kernel request has no maintainer engagement. |
| <https://github.com/local-inference-lab/b12x/issues/233> | open | NVFP4 fused-quant request has no maintainer engagement. |
| <https://github.com/local-inference-lab/b12x/issues/234> | open | Skinny-M tile request is deprioritized until host-side decode overhead falls. |
| <https://github.com/local-inference-lab/b12x/issues/235> | open | Scratch-sizing contract issue remains open. |
| <https://github.com/local-inference-lab/rtx6kpro/issues/79> | open | Transfer report remains open without maintainer engagement. |
| <https://github.com/vllm-project/vllm/issues/52520> | open | Stock CPU reproduction confirms the admission livelock; the upstream accounting claim was corrected. |
| <https://github.com/vllm-project/vllm/pull/52530> | open | Changes requested; merged PR47272 handles the startup capacity defect, so this should close or retain only independently justified request-side scope. |
| <https://github.com/vllm-project/vllm/pull/47272> | merged | Merged null-block startup-capacity fix covers the exact-boundary boot defect. |
| <https://github.com/vllm-project/vllm/issues/52871> | open | Engine-fatal forward OOM report remains open without response. |
| <https://github.com/vllm-project/vllm/issues/52872> | open | Hybrid profiler peak report received and answered a maintainer response. |
| <https://github.com/vllm-project/llm-compressor/issues/3057> | open | Decorated-forward tracing bug is acknowledged. |
| <https://github.com/vllm-project/llm-compressor/pull/3058> | open | Fixes were pushed, CI reported green, and re-review is pending. |
| <https://github.com/LMCache/LMCache/issues/4247> | open | Silent fused-layout corruption report is active; published workflows must keep LMCache disabled. |
| <https://github.com/LMCache/LMCache/pull/4253> | open | Maintainer fix is open and review-blocked; it has not qualified this project's stack. |
| <https://github.com/LMCache/LMCache/issues/4492> | open | Cross-restart corruption report remains open. |
| <https://github.com/LMCache/LMCache/pull/4600> | open | Retrieve-failure handling fix remains open without review. |

## Current prioritized queue

This is the current queue; historical open lists elsewhere are append-only chronology.

### P0 — publication and correctness

1. **Sync all seven corrected local cards to their Hub READMEs** and verify exact new revisions (F184).
2. **Resolve runtime-source provenance** for the served EXL3 implementation; pinned bytes without a rebuildable public ancestor remain a release blocker (F063).
3. **Correct or clean the upstream record:** issues 409/410, unfiled B12X/QKV findings, PR52530 disposition, and clean replacements for PR319/397/436 where still valuable (F177–F183).
4. **If a format-only Q6_K claim is desired, run a same-engine experiment.** Current cross-engine results support only complete-pipeline comparison (F060/F167).
5. **Create versioned strict-JSON successors** for load-bearing invalid artifacts instead of modifying the originals (F163).

### P1 — evidence quality

1. Expand and refreeze missing v5 strata (F046).
2. Define and migrate to canonical path-independent content digests (F100).
3. Capture durable KV-pool provenance for the live serving claim (F142).
4. Validate the speculative-draft commands and EXL3 capture path on a compatible runtime (F150).
5. Finish repository-wide atomic receipt-writer migration (F176).

### P2 — optional research

- Independent-corpus validation of propagation-aware allocation.
- Task-paired reasoning-effort study with enough prompts and repeats.
- Profiler-first QKV fusion and MTP host-loop work after clean upstream scoping.
- Long-context quality, not only needle retrieval, above 262,144 tokens.

## Complete finding register

Statuses are terminal for this review: `open` and `blocked` mean the defect/gap is confirmed and intentionally remains in the queue, not that it was forgotten.

| ID | Severity | Status | Finding | Resolution |
|---|---|---|---|---|
| F001 | high | fixed | Correct document 01 four-bit contradiction | Corrected the prose so the stated width and recipe agree. |
| F002 | medium | later-resolved | Verify document 01 measurement provenance | Later manifests and build receipts pin the tensors, converter, and measured values; the early note remains historical. |
| F003 | medium | fixed | Correct recipe A stale headroom value | Replaced the stale headroom arithmetic with the receipt-backed value. |
| F004 | medium | fixed | Resolve recipe A disk arithmetic discrepancy | Aligned disk and payload arithmetic with the finalized manifest. |
| F005 | medium | later-resolved | Verify online K6 transfer rationale | Later online-K6 startup, fidelity, and manifest evidence supports the bounded transfer claim. |
| F006 | high | fixed | Verify in_proj_ba BF16 preservation | Restored the established ignore expression and verified the preservation path. |
| F007 | low | dismissed | Resolve redundant manual config repair | The historical manual step is redundant after finalization, but remains a valid external-toolchain safeguard rather than a second current convention. |
| F008 | medium | later-resolved | Track unhashed iteration-one KLD evidence | The v5 suite, reports, and cumulative receipts supersede the unhashed iteration-one figures. |
| F009 | medium | fixed | Correct unsupported proot performance claim | Removed performance attribution that the proot setup did not measure. |
| F010 | medium | later-resolved | Verify baseline startup provenance | Production-image, launch, and qualification receipts now pin the startup environment. |
| F011 | medium | fixed | Repair serving matrix eager row | Corrected the eager-mode row against the recorded launch configuration. |
| F012 | medium | fixed | Correct favorable benchmark count | Recomputed and corrected the favorable-result count. |
| F013 | medium | fixed | Separate disk ratio from VRAM claim | Removed the unsupported conversion from serialized bytes to unmeasured resident VRAM. |
| F014 | medium | later-resolved | Resolve serving card revision provenance | Publication receipts and the exact-revision external-card ledger now provide the missing provenance; Hub divergence remains F184. |
| F015 | medium | fixed | Qualify any-GPU W4A16 portability | Scoped portability to the tested runtime and hardware family. |
| F016 | high | fixed | Bound single-window KLD causality | Reframed one-window observations as bounded associations rather than isolated causes. |
| F017 | medium | later-resolved | Verify iteration-one evidence identities | Later suite, head, candidate, and runtime identities supersede the early unpinned evidence. |
| F018 | medium | fixed | Qualify one-shot throughput attribution | Scoped the throughput observation to its one-shot configuration and removed causal overreach. |
| F019 | low | fixed | Source or retire FP8 point-five claim | Retired the uncited 0.5 claim from current publication surfaces. |
| F020 | medium | dismissed | Verify Quesma protocol transcription | Browser extraction matched every chart row and the FP8 and NVFP4 values exactly. |
| F021 | medium | fixed | Reassess KLD closed-with-proof claim | Replaced closure language with the actual protocol scope and unresolved axes. |
| F022 | medium | later-resolved | Verify status-pass receipt gates | Later qualification receipts carry explicit identity and gate fields and supersede the early status shorthand. |
| F023 | high | fixed | Prevent calibration evaluation leakage | Separated calibration inputs from held-out evaluation and documented the prohibition. |
| F024 | medium | fixed | Correct greedy allocator optimality claim | Replaced global-optimum language with the measured greedy procedure and its limitations. |
| F025 | medium | fixed | Correct pure-upside calibration wording | Acknowledged tradeoffs and held-out uncertainty. |
| F026 | medium | later-resolved | Trace W4A8 design to outcome | Later measured W4A8 and allocation receipts close the design-to-outcome chain. |
| F027 | medium | fixed | Correct W4A8 shared-memory arithmetic | Fixed the shared-memory calculation and resulting feasibility statement. |
| F028 | medium | fixed | Separate static defects from GPU cycles | Split source-verifiable correctness defects from experiments that genuinely require GPU evidence. |
| F029 | low | fixed | Qualify optimal decode tile claim | Marked the decode tile as an unmeasured candidate, not an optimum. |
| F030 | low | fixed | Repair tautological alignment explanation | Replaced circular reasoning with the actual alignment invariant. |
| F031 | medium | fixed | Qualify learned-rotation runtime inference | Separated a representation hypothesis from unmeasured runtime cost. |
| F032 | medium | fixed | Correct calibration versus weight-tile terms | Used weight-tile and calibration terminology consistently. |
| F033 | medium | fixed | Bound one-matrix rotation no-go | Limited the conclusion to the tested matrix and transform. |
| F034 | medium | fixed | Assess learned-scale degeneracy | Documented the objective degeneracy and removed the unsupported recommendation. |
| F035 | medium | later-resolved | Verify live-versus-replay floor | receipts/replay-live-floor-v5.json measures 5.83e-4 with a 5.15e-4 to 6.64e-4 interval. |
| F036 | medium | fixed | Resolve v2 suite stratum imbalance | The v5 suite supplies the balanced, source-clustered replacement. |
| F037 | low | later-resolved | Reconcile PARO todo with pilot | Later pilot evidence resolves the standing item and bounds the negative result. |
| F038 | medium | fixed | Correct NVFP4 p99 transcription | Replaced the mistyped tail value with the receipt value. |
| F039 | low | fixed | Qualify ordering as validation | Ordering is now a consistency check, not independent validation. |
| F040 | medium | fixed | Reconcile stale MLP ownership consequence | Updated the consequence to match the final ownership and manifest. |
| F041 | high | fixed | Correct zero noise-floor sequencing | Removed the false zero-floor implication and used measured replay controls. |
| F042 | medium | fixed | Correct self-check validation overclaim | Classified algebraic self-consistency separately from held-out validation. |
| F043 | medium | later-resolved | Reconcile replay floor with paired head | Later paired-head and live-replay receipts explain the different estimands. |
| F044 | high | fixed | Retract graph-decode exact parity | Replaced exact-parity language with the measured tolerance and scope. |
| F045 | medium | fixed | Qualify graph speed mechanism | Removed the unprofiled causal mechanism. |
| F046 | medium | open | Reconcile missing-strata count | The current v5 corpus still lacks a completed rerun for every proposed stratum; a new frozen-suite expansion is required. |
| F047 | low | fixed | Correct KV-cache formula label | Corrected the quantity and units in the KV formula. |
| F048 | medium | fixed | Correct head-ablation rationale | Attributed the head exclusion to the shared-head protocol rather than a size assumption. |
| F049 | medium | fixed | Remove orders-above-zero noise claim | Replaced the claim with measured control magnitudes. |
| F050 | medium | fixed | Qualify unequal MTP throughput | Stopped comparing unequal MTP depth and acceptance configurations as one causal arm. |
| F051 | medium | fixed | Avoid K6-head extrapolation to K5 | Limited the head result to the measured K6 configuration. |
| F052 | medium | fixed | Remove replay-floor causal attribution | Kept the floor empirical and labeled mechanism speculation. |
| F053 | medium | fixed | Qualify common-mode cancellation | Stated the assumptions required for cancellation instead of treating it as guaranteed. |
| F054 | medium | fixed | Correct decode gate comparator | Used the matching baseline for the decode gate. |
| F055 | medium | fixed | Preserve scorecard memory ceiling | Restored the declared memory constraint in the scorecard decision. |
| F056 | low | fixed | Correct prefill decode-step wording | Separated prompt prefill from the one included decode step. |
| F057 | medium | fixed | Bound structural-limit claim | Scoped the limit to the existing implementation path. |
| F058 | medium | fixed | Bound ambient-drift claim | Limited drift conclusions to the observed measurement envelope. |
| F059 | high | fixed | Correct all-builds-beat-FP8 contradiction | Aligned the prose with the actual candidate ordering. |
| F060 | critical | fixed | Remove cross-engine KL subtraction | Withdrew every current net-of-floor and format-only bound; KL is neither additive nor a metric. |
| F061 | medium | fixed | Qualify MMLU shortfall cause | Removed the unsupported single-cause attribution. |
| F062 | high | fixed | Retract down-projection proxy correctness | The proxy result is now evidence for prioritization only, not correctness. |
| F063 | critical | blocked | Surface unrebuildable runtime source | The served exl3.py bytes are pinned, but no public source ancestor can rebuild them; release-source provenance remains blocked upstream. |
| F064 | high | fixed | Separate current queue from chronology | The retrospective now distinguishes append-only history from the current prioritized queue. |
| F065 | low | later-resolved | Update physical 5090 status | Later physical-host receipts supersede the earlier planned status. |
| F066 | medium | fixed | Qualify multimodal quantization cause | Separated measured multimodal behavior from a non-isolated quantization mechanism. |
| F067 | medium | fixed | Correct embedding versus attention size | Recomputed and corrected the byte comparison. |
| F068 | medium | fixed | Qualify paired replay volume systematic | Documented the shared systematic and what pairing does and does not cancel. |
| F069 | medium | fixed | Correct tail-breaks-generation claim | Tail divergence is no longer equated with observed generation failure. |
| F070 | high | fixed | Stop transferring paired values across suites | Current tables compare only identical suite and scoring identities. |
| F071 | low | later-resolved | Update v5 capture status | The complete v5 capture and cumulative receipts supersede the pending status. |
| F072 | low | later-resolved | Update two-shard tail status | Later tail receipts supersede the pending two-shard note. |
| F073 | low | later-resolved | Update S16 fidelity verdict | The measured S16 receipt replaces the preregistered prediction. |
| F074 | medium | fixed | Correct two-orders efficiency claim | Recomputed the ratio and removed the exaggerated order-of-magnitude statement. |
| F075 | medium | fixed | Separate replay floor and paired resolution | The two controls are reported as different estimands. |
| F076 | low | later-resolved | Update K3 and KV sweep status | Later K3 and KV receipts resolve the pending work. |
| F077 | low | later-resolved | Update tail-shard availability | The additional tail artifacts are now recorded. |
| F078 | medium | fixed | Qualify no-new-fidelity-risk | Changed an absolute claim to the tested gate result. |
| F079 | medium | fixed | Correct replay precision wording | Documented actual accumulation precision. |
| F080 | medium | fixed | Correct protocol accumulation comparison | Stopped comparing accumulation modes across non-identical protocols. |
| F081 | high | dismissed | Verify official FP8 identity | All 66 shard LFS digests and the index match Qwen/Qwen3.8-27B-FP8 at 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a. |
| F082 | medium | fixed | Qualify prompt-independent step time | Scoped the observation to measured prompts and operating point. |
| F083 | medium | fixed | Separate acceptance from noise | Acceptance changes are no longer described as measurement noise. |
| F084 | medium | fixed | Qualify unchanged-fidelity MTP | Limited the statement to the measured target path and metric. |
| F085 | low | later-resolved | Add V2-runner outcome | Later V2 evidence supersedes the earlier blocker status. |
| F086 | medium | fixed | Qualify down-projection proxy | Kept the result as a heuristic lead rather than a causal conclusion. |
| F087 | high | fixed | Fix paired comparator ties | Both paired tools now report strict wins and a separate ties count; identical inputs produce zero wins and all ties. |
| F088 | medium | fixed | Correct all-results-used-fp8 | Separated experiments that did not use FP8 KV. |
| F089 | high | fixed | Remove truncated-KL lower bound | Top-k renormalized KL is now described as direction-unbounded without tail assumptions. |
| F090 | medium | fixed | Correct order-of-magnitude arm claim | Recomputed the compared arms and corrected the magnitude. |
| F091 | high | fixed | Bound long-context quality | Needle retrieval and startup are no longer generalized to long-context quality. |
| F092 | medium | fixed | Correct token-head prefill claim | Separated per-token head cost across prefill, draft, and decode paths. |
| F093 | low | later-resolved | Update live tool-call status | Later end-to-end tool-call receipts replace the earlier gap. |
| F094 | high | fixed | Mitigate tool-delimiter injection | Cards now require delimiter rejection, untrusted tool-data treatment, and message-order constraints. |
| F095 | high | fixed | Verify card template constraints | K4, K5K6, context, hydrated, and shared serving profiles now carry the same chat-template contract. |
| F096 | medium | fixed | Qualify external validation | Shared harness evidence is not labeled independent validation. |
| F097 | high | fixed | Retract whole-prefill kernel-bound | Separated profiled all-FP4 prefill from other profiles and host overhead. |
| F098 | medium | fixed | Update fused-kernel-only lever | The roadmap now includes measured host and scheduler costs. |
| F099 | critical | fixed | Retire undocumented KV override | Removed kv_cache_memory_bytes from current fidelity and KLD launch paths. |
| F100 | medium | open | Canonicalize receipt content digests | Several historical content hashes remain path-dependent; a versioned canonical digest migration is still required. |
| F101 | medium | fixed | Remove replay-floor cause | Kept the cause as a hypothesis rather than a measurement. |
| F102 | medium | fixed | Qualify cross-suite prohibition | Explained which identities must match and when qualitative comparison remains permissible. |
| F103 | medium | later-resolved | Update head exclusion attribution | v5 own-head ablation receipts now quantify the excluded head contribution. |
| F104 | low | later-resolved | Update KV sweep and S16 | Later receipts close both status items. |
| F105 | medium | fixed | Qualify TP4 ceiling | Scoped the ceiling to the tested implementation and topology. |
| F106 | medium | fixed | Bound concurrency neutrality | Reported one bounded comparison rather than a universal no-cost claim. |
| F107 | high | fixed | Replace window-free claim | The 1M-native arm is now reported with boot and KV-pool confounding. |
| F108 | low | fixed | Scope completed task list | Separated completed historical work from present open work. |
| F109 | low | later-resolved | Update Terminal-Bench status | Later Terminal-Bench receipts supersede the pending card text. |
| F110 | medium | fixed | Qualify tunnel common mode | Removed unmeasured causal attribution to the tunnel. |
| F111 | medium | fixed | Stop equating timeout with speed | A timeout is no longer treated as proof that only latency changed. |
| F112 | low | later-resolved | Update passes two and three | Later receipts record the completed passes. |
| F113 | medium | fixed | Correct two-times-timeout capability | Removed capability inference from a timeout multiple. |
| F114 | low | later-resolved | Update speedrun status | The later speedrun record supersedes the plan. |
| F115 | low | fixed | Remove invalid high effort | Removed the unsupported effort option. |
| F116 | high | fixed | Retract zero quantization attribution | A null task result is not attributed to zero quantization effect. |
| F117 | medium | fixed | Qualify hybrid scaling cause | Marked the mechanism as consistent with, not isolated by, the measurement. |
| F118 | high | fixed | Replace fidelity-free window | Current surfaces say the native-1M comparison is unresolved at boot-scale variation. |
| F119 | high | fixed | Stop cross-estimator ranking | Removed quantitative ranking across incompatible estimators. |
| F120 | high | fixed | Bound concurrency no-cost | The card and docs now state only that no extra effect was resolved in one probe. |
| F121 | medium | fixed | Remove request-majority claim | Removed unsupported claims about where most traffic lives. |
| F122 | high | fixed | Bound six-prompt proxy | The six-prompt proxy is now exploratory and cannot establish high-fidelity superiority. |
| F123 | medium | fixed | Correct teacher-forced token label | Teacher-forced full-sequence KLD is no longer called a single-token test. |
| F124 | medium | fixed | Replace MTP KLD objective | MTP removal is evaluated on throughput, acceptance, context, and capability rather than body KLD. |
| F125 | critical | fixed | Remove pairwise KL error sum | Current surfaces reject additive KL decomposition; immutable historical receipts are explicitly superseded. |
| F126 | medium | fixed | Separate cache correctness and chat calibration | Cache integrity no longer stands in for chat-template calibration. |
| F127 | high | fixed | Use paired-difference variance for power | The unsupported power calculation from within-arm CV was retired. |
| F128 | high | fixed | Replace private source evidence | Durable repository paths replace inaccessible agent-session citations. |
| F129 | medium | fixed | Correct prefill chunk FLOPs | FLOP wording now names the per-2048-token chunk rather than the whole prompt. |
| F130 | medium | fixed | Distinguish tolerance and parity | A tolerance gate is no longer called numerical parity. |
| F131 | high | fixed | Separate traffic and scratch capacity | 48.65 GB of traffic is no longer presented as scratch allocation. |
| F132 | high | fixed | Redact identifiers | Removed account, host, network, and balance identifiers from current narrative surfaces; no credential value was found. |
| F133 | medium | fixed | Bound cost-model invariants | Empirical knees and coefficients are scoped to their measured setup. |
| F134 | medium | fixed | Bound reasoning effort | The recommendation is limited to the one-prompt, two-repeat probe. |
| F135 | low | fixed | Correct multiplier ranking | Reordered the candidates using the actual computed multipliers. |
| F136 | high | fixed | Retract timeout quantization attribution | Timeouts no longer prove zero quantization effect. |
| F137 | high | fixed | Correct DP8 price comparison | Corrected the C1 DP8 ratio from 49x to 8.2x. |
| F138 | high | fixed | Separate cache hit and request cost | Removed the invalid cached-input blend and corrected the warm-request denominator. |
| F139 | medium | fixed | Bound long-context mechanism | The observed scaling is separated from its unisolated architectural explanation. |
| F140 | medium | fixed | Supersede overnight plan | The plan is marked retrospective and no longer presents stale gains as current. |
| F141 | medium | fixed | Qualify TP8 ceiling | TP8 is identified as an implementation-specific bound. |
| F142 | medium | open | Commit KV-pool provenance | The referenced live KV-pool state remains outside a durable receipt and requires a new capture. |
| F143 | medium | dismissed | Verify external prices | OpenRouter, pricepertoken, Puter, and Jarvis values were refreshed; the unverified Puter cached-input price was not retained as fact. |
| F144 | medium | fixed | Require profiler for ALU claim | ALU bottleneck language now requires profiler evidence. |
| F145 | medium | fixed | Correct serial speedup arithmetic | Amdahl speedup is 3.077, not 3.4. |
| F146 | medium | fixed | Qualify plus-one fidelity rule | The plus-one rule and error percentages are labeled unmeasured heuristics. |
| F147 | high | fixed | Replace agent-session dependencies | Durable docs and receipts replace agent URI dependencies. |
| F148 | high | fixed | Reject hardware-independent speedup transfer | Speculative speedups are no longer transferred as universal ratios. |
| F149 | medium | fixed | Qualify draft acceptance | Better observed acceptance is not generalized beyond the tested draft and target. |
| F150 | medium | open | Verify speculator commands | The proposed speculator commands and EXL3 capture path are now labeled unvalidated; a compatible runtime and GPU run remain needed. |
| F151 | high | fixed | Correct 24GB training feasibility | The document no longer assumes the drafter trains on one 24 GB card. |
| F152 | medium | fixed | Qualify better-than-BF16 acceptance | The result is reported as an observation, not a general quality law. |
| F153 | medium | fixed | Reconcile MTP speedup | The speculative plan now includes the local measured acceptance and throughput constraints. |
| F154 | critical | fixed | Prevent EDA evaluation reuse | The EDA card and docs require independent held-out validation and forbid calibration-evaluation reuse. |
| F155 | high | fixed | Retire sqrt-energy recommendation | The refuted sqrt-energy allocator is replaced by propagation-aware objectives and validation. |
| F156 | medium | fixed | Qualify KV compounding | The KV mechanism is labeled an unisolated hypothesis. |
| F157 | medium | dismissed | Verify EXL3 allocation metric | Pinned exllamav3 source confirms direct one-group marginal end-to-end KLD after downstream propagation; greedy interactions remain possible. |
| F158 | high | fixed | Remove cross-model KLD suggestion | Removed invalid direct KLD comparison across different base models. |
| F159 | medium | fixed | Update upstream audit | This retrospective includes #407 through #412, #394, #435, PR319, PR436, stock vLLM, and LMCache items omitted by the prior audit. |
| F160 | high | fixed | Preserve receipts with corrections | Published receipts remain immutable; corrected cards, generators, and this review explicitly supersede invalid interpretations. |
| F161 | medium | dismissed | Resolve PR398 overlap | PR398 is not a duplicate of PR234: it keys the broader captured wrapper family, including draft shapes and depths. |
| F162 | medium | fixed | Correct dense-model OOM description | The session note now identifies the dense model and removes the MoE claim. |
| F163 | high | blocked | Disclose invalid JSON receipts | Eight strict-JSON failures and nineteen empty raw captures are inventoried; immutable artifacts require versioned corrected successors rather than edits. |
| F164 | low | fixed | Repair split issue links | Corrected the #392 and #393 links in publication cards. |
| F165 | low | fixed | Repair malformed heading | Fixed the legacy card heading. |
| F166 | high | fixed | Restore in_proj_ba ignore | All current standalone launch examples preserve in_proj_ba with the established ignore regex. |
| F167 | critical | fixed | Reframe K6 parity | The parity verdict is withdrawn; card and generator report only the cross-engine complete-pipeline observation. |
| F168 | medium | fixed | Correct S16 tail ratio | The card now distinguishes the 4.25x p99.9 ratio from the 4.39x mean ratio. |
| F169 | medium | fixed | Correct chart protocol claims | Charts now name exact protocols and treat the cross-engine BF16 value as diagnostic only. |
| F170 | medium | fixed | Qualify dataset hardware claims | Hardware and common-mode statements are limited to measured configurations. |
| F171 | high | fixed | Correct v3 paired sign | The v3 paired-difference direction and labels now agree. |
| F172 | high | fixed | Remove cross-protocol percent | Deleted the invalid percentage comparison between different scoring protocols. |
| F173 | medium | fixed | Bound quantizer law | A one-model EDA result is no longer promoted to a universal quantizer law. |
| F174 | high | fixed | Update README EDA guidance | README now recommends propagation-aware allocation and independent validation, not sqrt-energy. |
| F175 | high | fixed | Correct truncation bias | Top-k renormalization is documented as direction-unbounded without tail assumptions. |
| F176 | medium | open | Standardize atomic receipt writers | The modified YaRN and KLD9 writers are atomic and strict, but a repository-wide writer migration remains. |
| F177 | medium | blocked | Update issue 409 | The existing issue needs a public correction with the measured marginal; no upstream mutation was authorized in this review. |
| F178 | medium | blocked | Update issue 410 | The existing issue needs the measured null result; no upstream mutation was authorized. |
| F179 | high | blocked | File B12X warmup bug | Warmup is keyed only by device instead of device and bit width; locally cured, but no upstream issue was filed in this review. |
| F180 | medium | blocked | File QKV fragmentation finding | The measured QKV launch-fragmentation and tuple-serialization problem remains unfiled upstream. |
| F181 | medium | blocked | Close or rescope PR52530 | PR52530 is changes-requested and the null-block startup fix merged as PR47272; owner action is required to close or narrowly rescope it. |
| F182 | high | blocked | Replace stacked embedding PR | PR436 remains a large stacked diff and needs a clean branch scoped to the embedding change. |
| F183 | high | blocked | Correct PR436 KLD evidence | PR436 quotes body-only 0.002700 as embedding evidence; the measured int8 embedding delta is about 0.000065 and needs a public correction. |
| F184 | high | blocked | Reconcile Hub and local cards | All seven reviewed local publication cards now differ from their current Hub READMEs after corrections; publication credentials were not used and every card needs an explicit sync. |
| F185 | medium | fixed | Complete upstream inventory | The final audit and ledger include all project-authored items in scope plus the key overlapping upstream fixes. |
| F186 | high | fixed | Correct YaRN control contradictions | The analyzer schema and hydrated card now report boot and KV-pool confounding, remove the false bit-identical claim, and avoid claiming native 1M quality or a zero window effect. |

## Verification performed

- `python3 -m py_compile` passed for every modified Python tool.
- `tools/fidelity.py paired` on one report against itself emitted schema `/3`, 136 ties, and zero wins for either arm.
- `tools/kld_aggregate.py paired` on one cumulative receipt against itself emitted schema `/2`, 5,120 ties, and zero wins.
- `tools/yarn_penalty_analyze.py` replayed the committed raw artifacts into strict schema-2 JSON; the corrected output contains no false bit-identical/native-window claim.
- Both KLD chart generators ran successfully; regenerated light/dark SVG and PNG assets were visually inspected.
- `tools/parity_card.py` regenerated the K6 card from its fidelity and byte-accounting receipts; the result withdraws the historical parity verdict.
- LSP diagnostics are clean for the aggregate, chart, YaRN, parity-card, and parity-receipt changes. The fidelity and direct-KLD tools retain pre-existing diagnostics caused by unavailable pinned-runtime imports and dynamically supplied vLLM/torch types on the host.
- `python3 tools/verify_peer_review_inventory.py` passed: 108 documents, 1,939 receipts, 261 supporting files, 44 publication assets, two other files, 120 external-card records, 186 findings, and 47 upstream items; only the ledger self-digest is excluded.

## Final assessment

The project should continue to publish the v5 same-engine/shared-head ladder, finalized artifact identities, and narrowly scoped serving gates. It should stop publishing any “net-of-engine-floor” quantity, universal allocation law, or capability-to-quality transfer. The next gains come less from another broad experiment than from closing provenance, strict-receipt, card-sync, and upstream-scope debt.
