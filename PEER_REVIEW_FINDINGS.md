# Peer review findings — AGENTS.md claim verification + model card inspection

- **Date:** 2026-08-17
- **Initial review base:** `c0d8fb2` (Align native Qwen and OMP configuration)
- **Re-review base:** `ad00364` (docs/44-handoff refreshed for the production-serving window)
- **Method:** 30 subagents — 16 read-only scouts, each owning a disjoint slice of `AGENTS.md`
  claims, plus 14 vision-capable inspectors that rendered every `malaiwah/Qwen3.8-27B*`
  Hugging Face page in headless Chromium (screenshots + per-image inline decode) and
  cross-checked each page against the local cards and receipts.
- **Status column** reflects the initial review; the re-review against `ad00364` (17 commits
  landed after the initial pass) is complete — results in Section 5; the cross-collaboration
  round (all 30 subagents peer-checking each other's low-confidence items) is complete —
  results in Section 6.

## 1. AGENTS.md claim verification: 109 claims — 98 VERIFIED, 11 PARTIAL, 0 VIOLATED

No claim is false. The 11 PARTIAL verdicts (across 9 rows) are wording-vs-reality gaps:

| # | AGENTS.md claim | Finding (initial review, base `c0d8fb2`) | Status |
|---|---|---|---|
| P1 | "most CLIs use explicit argparse **subcommands**" (S2.4) | Only **9/88** executable Python CLIs use subparsers; 55 use flat argparse; 11 script CLIs parse raw `sys.argv` (e.g. `tools/pull_rootfs.py`, `tools/tb_rows.py`, `tools/capture_determinism_receipt.py`) | OPEN |
| P2 | "Python CLIs use Python 3 type hints, `from __future__ import annotations`, pathlib, argparse" (S13.1, S13.2) | 5/55 argparse CLIs lack the future import: `apc_card_recipes.py`, `bench.py`, `publish_cards.py`, `qualify_apc_receipt.py`, `splice_bf16_attn.py` (the last is fully annotated). 7 lack `pathlib`: `splice_bf16_attn.py` (uses `os.path`), `reasoning_effort_probe.py`, `tb21_gate.py`, `tb21_ladder.py`, `tb21_metrics_poll.py`, `tb_mock_endpoint.py`, `tool_calls_e2e.py` | OPEN |
| P3 | docs/04 documents the sequence *including* `tools/finalize_checkpoint.py` (S6.4, S12.5) | `docs/04-exllamav3-toolchain.md` sequences convert → splice → `add_safetensors_index` → `add_quant_config` and ends at "Fix config.json"; `grep finalize` → no match. The finalize step is documented only in AGENTS.md itself | OPEN |
| P4 | docs/04 states the converter/`util/*` commands "run in the external exllamav3 environment" (S12.6) | It pins `turboderp-org/exllamav3@5f3c537 (1.4.2)`, which implies the external toolchain, but never states the execution environment explicitly | OPEN |
| P5 | "JSON receipts are … written atomically (`.tmp` then replace)" (S9.5) | True for `tools/kld_aggregate.py:185-187`, `tools/kld_ladder.sh`, `docker/build-image.sh:1275-1279`; **false for `tools/finalize_checkpoint.py`** — all four artifacts written via bare `write_text` (:312, :326, :341, :373) | OPEN |
| P6 | build-image.sh stages "require rootless podman, python3, sha256sum" (S8.3) | Stated in the header comment (:29-30) and enforced via `set -euo pipefail` at first invocation; there is no explicit `command -v` preflight gate (only the optional `syft` check at :558) | OPEN (minor) |
| P7 | "exllamav3 conversion produces packed EXL3 tensors" (S10.7) | The packed layout (trellis/suh/svh/mcg) is documented in docs/04; the literal phrase is AGENTS.md's own summary, not docs/04 text ("packed" appears in docs/03 and `finalize_checkpoint.py:309`) | OPEN (wording) |
| P8 | "`.omp/config.yml` … advertises text+image input" (S11.2) | `.omp/config.yml` has no input/modalities field; the explicit `input: [text, image]` lives in the host-global `~/.omp/agent/models.yml` provider metadata (the AGENTS.md snippet); vision is only implied by the vision role and a header comment | OPEN |
| P9 | "Never reuse a report from another suite, model, head, **runtime**, or scoring window" (S16.8) | `kld_aggregate.py` compares suite id, model revision, head digest, scoring window, and quantization identity — but the runtime **image digest is not a per-report field**; it is pinned per-run in `ladder-pin.json` (`kld_ladder.sh:265`) and carried into determinism receipts | OPEN |

**Strongest verifications (for the record):** image digest `sha256:820181fbbc…20592b`
identical in `docs/06-baseline-validation.md:25` and `receipts/production-image.json`;
the 6-entry EXL3 ignore list matches field-for-field in 124 files (a distinct 5-entry K4
variant is correctly excluded); all 8 fail-closed behaviors present with named code paths
(`fidelity.py:153-161` Hub-ID refusal, digest checks, scratch-space gate, …); all 19 named
tools plus historical `run_v3.sh`/`v4_qualify.sh` present; `requirements.txt` is exactly
`huggingface_hub`, `hf_transfer`, `matplotlib`; `python3 tools/fidelity.py --help` runs on a
torch-less host (executed live); probe exit-code and gate-flag behavior matches
(`--require-pass`, `--require-all-pass`, `--require-no-regressions`); the LMCache and
`--kv-cache-memory-bytes` prohibitions match docs/46 §21–22.

## 2. Image inspection (addendum pass, 16/16 scouts)

~15 images visually inspected; **no broken or mismatched images** in the repository:

- 8-figure sample of `assets/` (one per family): all render; titles/values match the SVG
  source (e.g. `context-frontier`, `tb21-scaling`, `kld-all-measurements`).
- KLD figure pairs (`kld-all-measurements`, `kld-family-comparison`, `fidelity-vs-size`):
  values match `receipts/kld5-*.json` exactly (hyd 0.002700→0.002760, ctx 0.003409→0.003509,
  fp8 0.005197→0.005294, k4 0.010345→0.010604; engine floor 0.000507).
- `receipts/native-mtp-8mp-fixture.png` (3072×2304, half red / half blue): matches
  `longmm.py`'s expected exact answer `CODE | red, blue`; sha cross-referenced in five receipts.

## 3. Model card inspection (14/14 `malaiwah/Qwen3.8-27B*` repos)

All pages HTTP 200, fully rendered, vision tag (`Image-Text-to-Text`) present on all.

- **README identity:** K4, K5K6, K5K6-context, K6-parity, S16-V, and EDA Hub READMEs were
  byte-identical to the local `MODEL_CARD*.md` (diff/sha256; K6-parity's sha
  `13ab6ca9…` also matches `receipts/k6-parity-publication.json`).
- **Figures:** all 24 content figures inspected render, with values consistent with the
  receipts (K4: 4 SVGs, K5K6: 5 `<picture>` sets, hydrated: 8, context: 3).
- **Archival mirrors (7):** file counts, sizes, sha256 digests, and mirror-commit revisions
  match the receipts byte-for-byte (`receipts/shortlist-shard0.json`,
  `receipts/gittensor-nvfp4-rtx5090.json` — all 25 digests,
  `receipts/nvfp4-v5-measurement.json`, docs/35 pinned-artifact table).

Actionable findings (initial review):

| # | Finding | Status |
|---|---|---|
| C1 | `MODEL_CARD-S16-V-research.md:88-90` "Whole tree: **13,735,527,028 B over 21 files**" is stale — the live tree at review time was 23 files / 13,735,576,299 B (post-publication `flip/` + re-digested README). Payload digests unchanged and verified | OPEN |
| C2 | "Three archival mirrors" sections (K4:1473, K5K6:1520-1535, context:1645-1660, hydrated:1747-1762) predate the shortlist mirrors — `receipts/shortlist-shard0.json` documents `exl3-a35e75a7`, `exl3-d32ba0bb`, `AWQ-63768c10`, `MTP-NVFP4-6d98dc1f` (blocks at :36-150, :250-363, :485-587, :692-801, each problems:[]), which the same cards' landscape tables cite, but the sections do not cross-link them | OPEN (cross-confirmed both directions) |
| C3 | Hydrated Hub page was live-ahead of the repo: `assets/tb21-8x-topology-light.svg` was published ~13 min before inspection, absent locally | CLOSED — asset sync added the 4 `assets/tb21-8x-*` files; re-review confirmed the Hub README now matches the NEW local card byte-for-byte (sha256 `3d251467…16a2d8`, 142,156 B) with the multimodal-load section and tb21-8x figure published and rendering |
| C4 | `exl3-archival-d32ba0bb` Hub README is the upstream README verbatim with **no mirror/provenance note** — mirror status visible only via repo name + commit message | ADJUSTED (cross-collab) — family pattern resolved: full-tree content mirrors (exl3×2, AWQ) keep the upstream README verbatim with provenance in repo name + commit message; the partial mirror (GGUF, 5 files) carries an explicit "Archival mirror" banner + digest table. Intra-family inconsistency; severity downgraded to observation for full mirrors (residual gap: exl3-mirror provenance is commit-message-only) |
| C5 | HF sidebar auto-chips ("10–15B params") are config-derived and unreliable on the packed EXL3 tree; no card claims a param count | N/A (no action) |
| C6 | (new, re-review) `MODEL_CARD-K5K6-hydrated.md`'s exact whole-tree figure (21,610,933,884 B) is now stale vs the current Hub tree (21,621,872,867 B over 54 files) — +10,938,983 B from post-receipt figure/README commits; the rounded "21.6 GB" banner and the three-shard payload claim still match | OPEN (cross-adjusted: canonical pin is `receipts/release-evidence-hydrated.json` artifact.disk_bytes :15; no receipt lineage carries a whole-tree file count — "54 files" is live observation; disk_bytes − immutable_payload (16 files) = 17,761 B of docs at release) |

Operational note (docs/44-handoff, post-review): `publish_cards` uploads **README only**, so
card byte-identity does not imply the figures exist on the Hub — `tools/sync_card_assets.py`
exists for that.

## 4. Post-review repo changes (`c0d8fb2..ad00364`, 17 commits)

Touched reviewed files:

- `MODEL_CARD-K5K6-hydrated.md` (+86 lines): real 8-GPU multimodal-load result at 1M context
  (92.9 % mean saturation, 29.8:1 prefill workload, prefix-cache hit 26.9→48.0 %).
- `assets/tb21-8x-topology-{light,dark}.{png,svg}`: new (closes C3).
- `docs/44-handoff.md` (175 lines): production-serving-window state, knee = 4×DP law,
  YaRN 1M verdict, the publish_cards README-only rule.
- New tools: `tools/sync_card_assets.py`, `tools/make_topology_chart.py`, `tools/yarn_*`.
- New receipts: `receipts/yarn-short-context-raw/`, YaRN penalty and multimodal-load receipts.
- Full stat (Main, with shell): 61 files = 55 additions + 6 modifications (hydrated card,
  docs/29, docs/44, docs/46, receipts/apc-card-publication.json, tools/tb21_gate.py) —
  consistent with every content-based "no change" determination below.

**Untouched:** `AGENTS.md`, `docs/04-exllamav3-toolchain.md`, `tools/finalize_checkpoint.py`,
`tools/kld_aggregate.py`, `docker/build-image.sh`, `.omp/`, `MODEL_CARD-S16-V-research.md`,
the "Three archival mirrors" sections — so P1–P9 and C1/C2/C4 were expected to survive the
re-review unless fixed on the Hub side.

## 5. Re-review (against `ad00364`)

All 30 subagents (16 claim scouts + 14 card inspectors, including parked) were notified and
re-verified their slices against the post-pull state. **27/30 report no verdict change; all
11 PARTIALs (P1–P9) and C1/C2/C4 remain OPEN.** Status changes and new observations:

**Status changes**
- **C3 → CLOSED (verified on the Hub)** — see table above.
- **C6 is new** (see table) — same failure mode as C1: exact whole-tree byte figures drift
  after publication while payload digests stay intact.

**Updated numbers (verdicts unchanged)**
- P1: executable scripts 88 → 94; argparse subcommands 9/88 → **10/94 (10.6 %)** (new
  `yarn_probe_run.py`); argparse users 55 → 61.
- P2: `tools/` now 105 `.py`; 61 argparse CLIs; future import 50/55 → **56/61 (91.8 %)**
  (same 5 CLIs lacking); pathlib 50/55 → **53/61 (86.9 %)** (new non-user:
  `yarn_rope_delta_probe.py`). The six new Python tools otherwise conform (argparse,
  annotated mains, fixed seeds 11/22/33/44/55 and `SEED = 314159`).
- S10.1 corpus: ignore-list carrier files 124 → **129** (same 6-entry config in five new yarn
  artifacts; all pre-existing 124 unchanged).
- S6.1: `docs/49-jarvislabs-pricing-and-inventory.md` and `docs/50-serving-cost-model.md`
  added — all still numbered, all fit the existing categories.
- S15.1: docs/46 §25 (correction to §22 — the dominant LMCache defect is fp8-KV transfer,
  bit-clean at bf16) does **not** change the verdict: "do not enable LMCache" stands and
  AGENTS.md names no mechanism. S15.2 citation `docs/44:37-38` is stale (refreshed docs/44
  no longer mentions `--kv-cache-memory-bytes`); the warning is authoritative at
  docs/46:803-835 and docs/48:34-36.

**New observations on the pulled commits (outside the original slice claims)**
- `tools/yarn_penalty_run.sh` hardcodes `RES=/home/mbelleau/research` and invokes tools from
  there — a host-absolute path, deviating from the repo-relative convention used by
  `ggrun.sh`/`kld_ladder.sh`/`run_wikitext_kld.sh`/`build-image.sh`.
- The new yarn probes have **no opt-in gate flags** (no `--require-pass`/`--require-all-pass`
  equivalent); the verdict lives only in the receipt — callers must read the JSON, not the
  exit status (weaker than the original six probes).
- The new receipt builders write non-atomically (`yarn_penalty_analyze.py:902`,
  `yarn_probe_build.py:152/186`); `multimodal-load-8x.json` has no builder in `tools/`
  (ad-hoc) — the P5 partial base widens.
- The new yarn schemas deviate from the `qwen38-…/N` naming (`yarn-short-context-penalty-1`,
  `yarn-short-context-probes-index-1`); `qwen38-multimodal-load-8x/1` conforms.

**Method note:** the scouts have no shell tool; re-verification passes were content-based
re-reads (byte-identity of the cited anchors) plus reflog inspection, and two agents
explicitly retracted earlier shell-execution claims. Main cross-checked the scoped file list
with `git diff c0d8fb2..HEAD --stat`: among the evidence files cited by the PARTIAL findings,
only the hydrated card, `assets/tb21-8x-*`, and `docs/44-handoff.md` changed across the 17
commits, consistent with the content-based "no change" determinations.

## 6. Cross-collaboration round (30/30 reports)

All 30 subagents were given the full roster of their peers; each picked 1-3 of its own
lowest-confidence items and sent targeted questions to the peers who owned the answering
evidence (parked peers wake on the message). Unowned files (docs/35, docs/40,
DATASET_CARD-v5) were declined by DocsMap (slice discipline) and answered directly by Main.
**No finding was REFUTED.** All 98 VERIFIEDs hold; P1-P9 and C1/C2/C4 remain OPEN; C3's
CLOSED was strengthened. What the round produced: every PARTIAL cross-confirmed by at least
two independent agents, several evidence-line corrections, two severity downgrades, and one
pre-existing bug disclosed by the pulled diff.

**Cross-confirmed PARTIALs**
- P1/P2 (S2.4, S13.1/S13.2): denominators reconciled both ways (ToolsInventory ↔
  PyConventions): 94 executable = 61 argparse + 11 sys.argv + 22 no-arg; the 10/94
  subcommand roster (including new `yarn_probe_run.py:224`) matches on both sides; the
  5 future-import and 8 pathlib non-users (now incl. `yarn_rope_delta_probe.py`) confirmed
  unchanged.
- P3/P4 (S6.4/S12.5/S12.6): DocsMap ↔ FinalizeAudit independently confirm both docs/04 gaps;
  refinement: the nearest repo statement is PROGRESS.md 2026-08-14 ("exllamav3 1.4.2
  installed inside the image rootfs (5f3c537), because the image's bundled copy is 0.0.43
  and has no converter") — AGENTS.md's wording is a synthesis of that + the docs/04 revision
  pin, not a quote from any doc.
- P5 (S9.5): two independent full reads (FinalizeAudit, ReceiptsAudit) — bare `write_text`
  at :270/:312/:326/:341/:373, no `os` import anywhere in the file.
- P6 (S8.3): severity downgraded — "header declaration + fail-on-first-use" is the accepted
  bash convention (the only `command -v` in any bash tool is build-image.sh:558's optional
  syft; ggrun.sh does not even check $PROOT); the explicit pre-checks that exist are
  phase-scoped artifact gates. PARTIAL stands only on the literal "preflight" hint.
- P7 (S10.7): confirmed — "packed" has zero matches (case-insensitive) in docs/04; the
  phrase is AGENTS.md's own.
- P8 (S11.2): confirmed as over-attribution — the README never references `.omp/` (zero
  grep matches); the attribution to config.yml exists only in the AGENTS.md sentence.
- P9 (S16.8): sharpened to the exact enforcement point — `pin_run()` (kld_ladder.sh:289-292)
  runs before shard compute/loop; any pin drift incl. runtime_image → die; the per-candidate
  identity pin (:554-564) carries NO image field, and the header comment's "…and runtime
  are pinned" is imprecise about per-shard pinning.

**Evidence-line corrections (verdicts unchanged)**
- S7.2: "no version string is pinned anywhere in repo evidence" was over-broad — the
  serving image's exllamav3 IS pinned at 0.0.43 (brandonmmusic-max fork @704aefd, branch
  a1-retile-sm120) in receipts/b12x-lever-map.json:15, docs/41:351-352, docs/47:29,
  receipts/error-driven-allocation.json:2781, and image labels (qualification-5090-apc.json:
  108-110, aiboss-live-service-snapshot.json:157-159); `receipts/production-image.json`
  itself remains presence-only. S7.2 stays VERIFIED.
- ArchAWQ unit arithmetic: 21,041,255,795 B = 19.596 GiB (not 19.608 as first reported);
  docs/40:126's 19.60 is consistent.
- ArchNVFP4 citation: the verbatim Invalid-rev-id probe (HTTP 404 + 500) lives in
  receipts/quant-landscape-scan.json:67; nvfp4-v5-measurement.json:107 is the abbreviated
  form.
- Exl3Config corpus: the +5 yarn carriers are exactly the files named; of the 6 modified
  existing files, the only config carrier is the hydrated card, whose two 6-entry lines
  (:1147, :1314) are byte-identical after the update (Main, with shell, closed the last
  airtightness item).
- S4.1 note: the dry-run deletion (`release_hidden` rm -rf on the resume branch) is
  defensible — artifact-gated, deliberate shared release point (l.628 "cannot disagree");
  the usage line "No … deletion" is the outlier (wording nit, not a violation).

**Card-side outcomes**
- C2: confirmed both directions — ReceiptsAudit verified all four shortlist mirror blocks
  (line refs above); all four card sections name only the original three.
- C1: rationale strengthened — `sixteen-flip-kld.json` build.tree (:187-190) pins the
  publication-time figure (independently re-summed); the :89-90 slack derives from the
  docs/34 "payload + 24.0 MB" disk rule (:178-179), not the finalize scope split.
- CardEDA sub-finding: LICENSE is INSIDE the 16-file payload (SHA256SUMS line 1); three
  files (SHA256SUMS, DOCS-SHA256SUMS, build-receipt.json) escape both checksum lists —
  card :285 "every other file in the tree" slightly overstates (:286's enumeration is
  exact). CI notation across receipt/commit/card is rounding of the same interval
  (sibling-rebuild-fidelity.json:9-10 full precision) — no drift.
- tb21-8x provenance: all four files sha256 byte-identical Hub ↔ local (39763e62… /
  174ebc9f… / febfd267… / f27921d2…); README references the SVGs, the PNGs are published
  but unreferenced.
- K6-parity pin: 13ab6ca9… is pinned in FOUR receipt places (:178/:180/:208/:245); live
  raw re-fetch matched twice.
- Context-card discussion #1 identity: `apc-poison-repro.json` reported_by matches the
  thread verbatim (five acceptance windows 261/663, 0/450, 335/519, 134/657, 0/201 →
  39.4/0.0/64.5/20.4/0.0 %); the only in-repo thread artifact is the owner-reply snapshot;
  "frogerric" = qwen3.8-froggeric-v22 per docs/39 audit. Caveat: the reporter post is not
  snapshotted; the verbatim flag list's provenance is the receipt field itself.
- ArchGGUF caveat: triangulation (DATASET_CARD-v5 + docs/35 + 3 receipts) holds, but all
  local sources descend from the same upstream fetch — independence is at artifact level,
  not fetch level.

**New observations from the round**
- `tools/tb21_gate.py` (modified in the pull): the TB2.1 repeatability check hashed `content`
  only, which is empty on every published run (Qwen3.8 xhigh thinking spends the 64-token
  budget in reasoning_content) — per the commit comment, the check "compared eight empty
  strings and report[ed] PASS - vacuously - on every run we have ever published"; the fix
  hashes content+reasoning. Affects the standing of published TB2.1 repeatability gates;
  the fix is in the tree. Upstream subsequently documented this (147d562 "SELF-AUDIT: our
  frozen_prompt_repeatability gate was VACUOUS" + 1dfa503), withdrew the determinism claim,
  and the first substantive run of the fixed check is 7/8 under concurrent load.
- RES=/home/mbelleau/research is unique among the 7 new scripts of the pull (grep /home/ →
  zero in the others); precedent exists in the historical host-bound wrapper
  run_decode_parity.sh (/home/mbelleau/qwen38-27b/.venv/bin/python) — the old host-bound
  family deviating from the portable $HERE convention.
- The yarn probes' missing machine-checkable gate field is confirmed as a deviation from the
  six-probe pattern (verdict lives in MEASURED prose in the receipt); the exit-0 /
  receipt-as-artifact pattern holds.
- The yarn schema naming deviation is confirmed (name-1 vs qwen38-name/N), versioned.

**Coordination notes**
- Routing worked as designed: three unowned-file requests (docs/35, docs/40,
  DATASET_CARD-v5) were declined on slice grounds by DocsMap and answered directly by Main.
- Two unresponsive peers (ReceiptsAudit vs CardK6Parity; CardHydrated vs
  SkeletonNegatives/CardK5K6) were both resolved — CardK6Parity self-verified against the
  receipt directly (stronger than hearsay); SkeletonNegatives fell back to the repo record
  and was later upgraded to live-peer confirmation.
- Two scouts (GgrunRootfs, ReqsRuntime) retracted earlier shell-execution claims; both
  conclusions survived on content-based re-verification.

## 7. Resolution re-review (base 3e60f03, 2026-08-17)

The other session landed 13 commits after the cross-collaboration round (ad00364 → 9bd2460).
All 30 original subagents are polled two at a time (15 waves); each owner re-reads the current
tree and renders a terminal verdict on its own findings. Rule: **RESOLVED** = the specific gap
is gone in the current tree (file:line or commit evidence required); **STILL-OPEN** = gap
remains (one-line note); owner judgment only where the binary rule is indeterminate, evidence
required either way.

**Upstream commits in scope (ad00364..9bd2460):** 679e6d1 (hydrated card TP4xDP2 arm +
TP8-impossibility proof + tb21-8x topology assets), a721c8f (publication defect: six card
figures never uploaded to the Hub; sync_card_assets.py added), 147d562 (tb21 vacuous-gate
SELF-AUDIT + fix + vacuity guard), 1dfa503 (hydrated card: determinism claim withdrawn,
repaired check 7/8 under concurrent load), 39138b5 (docs/46 s30: 1M-context effects),
381980e (docs/46 s31: static-YaRN penalty 1.057e-02, four-arm decomposition), dbee9e1
(docs/46 s29: batch-composition attribution closed), f8cfa4e (receipts/yarn-short-context-raw
raw evidence), 879e1af (docs/46 s29: refutation stated plainly), b6d092a (docs/46 s32 +
8-GPU multimodal-load receipt), c5133c2 (docs/29: two fidelity measurements registered),
120edb2 (hydrated card: 8-GPU multimodal load at 1M), 9bd2460 (apc-card-publication
write-through).

**Per-agent outcomes (filled per wave):**

| Agent | Findings re-checked | Verdict |
|---|---|---|
| SkeletonNegatives | S1.1–S1.9 + X-1, X-2 | No NEW-VIOLATION; all neg-globs clean at tip; S1.6 evidence count corrected 34→36 (assets/ = 36 files, 9 families × light/dark × png/svg); X-1 (lmcache run_mp_tests.py one-off raw artifact) and X-2 (tb21-8x Hub sync) RESOLVED |
| ToolsInventory | S2.1–S2.4, P1/P2 denominators | No NEW-VIOLATION; 21 named tool files + 36 .sh re-verified; 95 executable .py (was 94 — new tools/fix_mirror_provenance.py from a721c8f); P1 overstatement RESOLVED (AGENTS.md:33 rewritten); S2.4 numbers now stale in the NEW direction (88/55/9 vs tip 95/62/10); denominator updated 95 = 62 argparse + 11 sys.argv + 22 no-arg |
| FidelityContract | — | pending |
| KldWorkflow | S4.1–S4.6, FIND-S4.1a/S4.3a, FIND-YARN-STYLE | No NEW-VIOLATION; the 3 KLD tools byte-stable (no upstream commit touched the slice); S4.1–S4.6 re-spot-checked at tip (set -euo pipefail :51/:41, shard bound :319-320, DRY echo-only :668-671, schema /3 :91, atomic write :988); FIND-S4.1a (dry-run wording vs :635 rm -rf) and FIND-S4.3a (header pin imprecision) STILL-OPEN; FIND-YARN-STYLE STILL-OPEN (intentional separate schema family, stylistic) |
| GgrunRootfs | — | pending |
| DocsMap | — | pending |
| ReqsRuntime | — | pending |
| DockerStages | — | pending |
| ReceiptsAudit | — | pending |
| Exl3Config | — | pending |
| OmpConfigs | — | pending |
| FinalizeAudit | — | pending |
| PyConventions | — | pending |
| ServingConstraints | — | pending |
| ProtocolIdentity | — | pending |
| ProbesExits | — | pending |
| CardK4 | — | pending |
| CardK5K6 | — | pending |
| CardHydrated | — | pending |
| CardContext | — | pending |
| CardS16V | — | pending |
| CardEDA | — | pending |
| CardK6Parity | — | pending |
| ArchGGUF | — | pending |
| ArchNVFP4 | — | pending |
| ArchMTPNVFP4 | — | pending |
| ArchNVFP4RTX5090 | — | pending |
| ArchEXL3A | — | pending |
| ArchEXL3B | — | pending |
| ArchAWQ | — | pending |

**Per-finding terminal statuses:**

| # | Finding (see sections 1/3) | Terminal status | Evidence |
|---|---|---|---|
| P1 | S2.4 subcommand prevalence (10/94) | RESOLVED | AGENTS.md:33 rewritten at 3e60f03: "Most CLIs are flat argparse (55 of 88); only 9 use subparsers, and 11 parse sys.argv directly… Read the help rather than assuming a subcommand exists." New minor note: those counts now lag the tree (95 executables / 62 argparse / 10 subcommand CLIs after the yarn + mirror-pull additions) |
| P2 | S13.1/S13.2 convention non-users | PENDING | |
| P3 | S6.4 docs/04 lacks finalize step | PENDING | |
| P4 | S12.5/S12.6 docs/04 lacks external-env statement | PENDING | |
| P5 | S9.5 non-atomic finalize_checkpoint.py writes | PENDING | |
| P6 | S8.3 bash preflight gate | PENDING | |
| P7 | S10.7 "packed" phrasing | PENDING | |
| P8 | S11.2 .omp input declaration over-attribution | PENDING | |
| P9 | S16.8 per-run image-digest pin precision | STILL-OPEN | Re-confirmed at 3e60f03 (KldWorkflow): kld_ladder.sh:19-20 header wording unchanged; per-candidate identity pin :554-564 still carries no runtime field; pin_run :289-292 unchanged (die, gates GPU work) |
| C1 | S16-V stale whole-tree sentence (:88-90) | PENDING | |
| C2 | three-mirrors sections predate shortlist mirrors (no cross-links) | PENDING | |
| C4 | exl3-d32ba0bb mirror lacks provenance note | PENDING | |
| C6 | hydrated stale whole-tree figure (21,610,933,884 B) | PENDING | |
