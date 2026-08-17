# Peer review findings — AGENTS.md claim verification + model card inspection

- **Date:** 2026-08-17
- **Initial review base:** `c0d8fb2` (Align native Qwen and OMP configuration)
- **Re-review base:** `ad00364` (docs/44-handoff refreshed for the production-serving window)
- **Method:** 30 subagents — 16 read-only scouts, each owning a disjoint slice of `AGENTS.md`
  claims, plus 14 vision-capable inspectors that rendered every `malaiwah/Qwen3.8-27B*`
  Hugging Face page in headless Chromium (screenshots + per-image inline decode) and
  cross-checked each page against the local cards and receipts.
- **Status column** reflects the initial review; the re-review against `ad00364` (17 commits
  landed after the initial pass) is in progress and will update Section 6.

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
| C2 | "Three archival mirrors" sections (K4:1473, K5K6:1520, context:1645, hydrated:1681) predate the shortlist mirrors — `receipts/shortlist-shard0.json` documents `exl3-a35e75a7`, `exl3-d32ba0bb`, `AWQ-63768c10`, `MTP-NVFP4-6d98dc1f`, which the same cards' landscape tables cite, but the sections do not cross-link them | OPEN |
| C3 | Hydrated Hub page was live-ahead of the repo: `assets/tb21-8x-topology-light.svg` was published ~13 min before inspection, absent locally | CLOSED — post-review asset sync added the 4 `assets/tb21-8x-*` files (light/dark × png/svg) |
| C4 | `exl3-archival-d32ba0bb` Hub README is the upstream README verbatim with **no mirror/provenance note** — mirror status visible only via repo name + commit message (unlike the other mirrors) | OPEN |
| C5 | HF sidebar auto-chips ("10–15B params") are config-derived and unreliable on the packed EXL3 tree; no card claims a param count | N/A (no action) |

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

**Untouched:** `AGENTS.md`, `docs/04-exllamav3-toolchain.md`, `tools/finalize_checkpoint.py`,
`tools/kld_aggregate.py`, `docker/build-image.sh`, `.omp/`, `MODEL_CARD-S16-V-research.md`,
the "Three archival mirrors" sections — so P1–P9 and C1/C2/C4 were expected to survive the
re-review unless fixed on the Hub side.

## 5. Re-review (against `ad00364`)

All 30 subagents (16 claim scouts + 14 card inspectors, including parked) were notified to
re-verify their slices against the updated repo and re-open the Hub pages. Results will be
appended here as they land; Section 1/3 status columns will be updated to
OPEN / FIXED (evidence) / NEW FINDING accordingly.

*Re-review results — pending.*
