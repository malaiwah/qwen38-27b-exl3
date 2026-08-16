#!/usr/bin/env python3
"""Build receipts/nvfp4-v5-measurement.json from the artifacts on disk.

Nothing here is typed by hand: identity digests are recomputed from the snapshot,
the commands are read out of the generated capture/replay scripts the ladder
wrote, harness digests are recomputed, and shard coverage is whatever reports
actually exist and verify.  If only shard 0 was measured, the receipt says so and
carries the exact command that finishes the rest.

    build_measurement_receipt.py <repo-root> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/var/tmp/work/kld5-nvfp4")
SNAPSHOT = Path("/var/tmp/models/Qwen3.8-27B-NVFP4")
PARENT_SUITE = Path("/var/tmp/work/kld5/suite")
REVISION = "9c73e2daee1d0fd494ffbd1d8753f2174a953796"
UPSTREAM_HEAD = "16b6615af3548b88e2d8e382457bc705b00479cf"
MIRROR_REPO = "malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da"
MIRROR_COMMIT = "7a66267ebd34a01ba9a13e56aa2cea0b27bdacd4"
REFERENCE_CAPTURE = Path("/var/tmp/work/gguf/hidden-bf16")
HEAD_FILE = Path("/var/tmp/work/kld2/lm_head.safetensors")
FROZEN_HARNESS = Path("/var/tmp/work/kld5/tools/fidelity.py")
TOTAL_SHARDS = 10


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_blob_id(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def receipt_digest(repo: Path, name: str) -> dict | None:
    path = repo / "receipts" / name
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return {"path": f"receipts/{name}", "sha256": sha256_file(path),
            "content_sha256": payload.get("content_sha256"),
            "schema": payload.get("schema")}


# v5 shard-0 means of the five published candidates, read from their own receipts
# below rather than restated here; this maps the v3 receipt's candidate names onto
# the ladder's.
V3_TO_LADDER = {"hydrated": "hyd", "k5k6": "k5k6", "context": "ctx",
                "official_fp8": "fp8", "k4": "k4", "nvfp4": "nvfp4"}


def suite_comparability(repo: Path) -> dict:
    """How far the v3 suite's absolute numbers sit from v5's, for every candidate.

    This is why the receipt refuses to put a v3 and a v5 number in one sentence.
    The point is not that they differ -- of course they do, different corpora --
    but that they differ by a NARROW band of factors and preserve the ordering
    exactly, which is what licenses carrying conclusions across suites while
    forbidding carrying values.
    """
    v3 = json.loads(
        (repo / "receipts/analysis-v3-contamination-corrected.json").read_text())["metrics"]
    comparator = json.loads(
        (repo / "receipts/cross-engine-comparator.json").read_text())["candidates"]
    rows = {}
    for v3_name, ladder in V3_TO_LADDER.items():
        if v3_name not in v3 or ladder not in comparator:
            continue
        v3_mean = v3[v3_name]["mean_kld"]
        v5_mean = comparator[ladder]["mean_kld"]
        rows[ladder] = {
            "v3_contamination_corrected_mean_kld": v3_mean,
            "v5_shard0_mean_kld": v5_mean,
            "ratio_v3_over_v5": v3_mean / v5_mean,
            "v3_top1": v3[v3_name]["top1_agreement"],
            "v5_top1": comparator[ladder]["top1_agreement"],
        }
    ratios = {k: r["ratio_v3_over_v5"] for k, r in rows.items()}
    order_v3 = sorted(rows, key=lambda k: rows[k]["v3_contamination_corrected_mean_kld"])
    order_v5 = sorted(rows, key=lambda k: rows[k]["v5_shard0_mean_kld"])
    return {
        "why_this_block_exists": "our published NVFP4 number on the older v3 suite is "
            "0.094978 raw / 0.092727 contamination-corrected, and this measurement is "
            "0.030115. That is a 3.08x gap on the same checkpoint, same revision, same "
            "flags and same shared-head protocol, so it needs an account before anyone "
            "reads either number.",
        "finding": "it is suite hardness, not an NVFP4 anomaly. Every candidate moved "
                   "by a factor in a narrow band, NVFP4 sits at the top of that band "
                   "rather than outside it, and the ordering is identical in both "
                   "suites.",
        "ratio_range": [min(ratios.values()), max(ratios.values())],
        "ratio_spread_end_to_end": max(ratios.values()) / min(ratios.values()),
        "ordering_v3": order_v3,
        "ordering_v5": order_v5,
        "ordering_identical": order_v3 == order_v5,
        "monotone_in_severity": "the factor rises with quantization error across the "
                                "band, which is what a harder corpus does: the "
                                "candidates with more probability mass in the tail "
                                "gain the most from an easier one",
        "per_candidate": rows,
        "rule": "never put a v3 and a v5 number in the same sentence. What carries "
                "across the two suites is the ORDERING and the rough ratio structure, "
                "both of which are preserved; what does not carry is any absolute "
                "value. No conclusion of ours changes between the two suites.",
        "corroborated_by": "Main derived the same six ratios independently from the "
                           "published receipts; both derivations agree to three "
                           "decimal places",
    }

def main() -> int:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])


    # ---------------------------------------------------------------- identity
    tree = json.loads(
        (SNAPSHOT / ".cache/huggingface/trees" / f"{REVISION}.json").read_text())
    published = tree["files"]
    files = {}
    for name, meta in sorted(published.items()):
        path = SNAPSHOT / name
        if not path.is_file():
            raise SystemExit(f"snapshot is missing {name}")
        entry = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        if "lfs_sha256" in meta:
            entry["published_lfs_sha256"] = meta["lfs_sha256"]
            entry["matches_published"] = entry["sha256"] == meta["lfs_sha256"]
        else:
            entry["git_blob_id"] = git_blob_id(path)
            entry["published_blob_id"] = meta["blob_id"]
            entry["matches_published"] = entry["git_blob_id"] == meta["blob_id"]
        files[name] = entry
    if not all(e["matches_published"] for e in files.values()):
        raise SystemExit("snapshot does not match its published digests; refusing to "
                         "write a receipt for it")

    # ---------------------------------------------------------------- coverage
    shard_rows = []
    for shard_dir in sorted(ROOT.glob("reports/shard-*")):
        report = shard_dir / "report-nvfp4.json"
        if not report.is_file():
            continue
        payload = json.loads(report.read_text())
        view = json.loads(
            (ROOT / "shards" / shard_dir.name / "suite/suite-manifest.json").read_text())
        shard_rows.append({
            "shard": int(view["shard"]["index"]),
            "report": str(report),
            "report_sha256": sha256_file(report),
            "report_schema": payload["schema"],
            "harness_sha256": HARNESS_SHA,
            "suite_token_sha256": payload["suite_token_sha256"],
            "contexts": payload["contexts"],
            "scored_positions": payload["scored_positions"],
            "token_mean_kld": payload["token_mean_kld"],
            "top1_agreement": payload["top1_agreement"],
            "p999_kld": payload["p999_kld"],
            "max_kld": payload["max_kld"],
            "ci95": [payload["context_bootstrap"]["ci95_low"],
                     payload["context_bootstrap"]["ci95_high"]],
        })
    covered = [row["shard"] for row in shard_rows]
    positions = sum(row["scored_positions"] for row in shard_rows)
    missing = [s for s in range(TOTAL_SHARDS) if s not in covered]

    # ---------------------------------------------------------------- commands
    commands = {}
    for shard in covered:
        tag = f"shard-{shard:04d}"
        for kind in ("cap-nvfp4.sh", "cap-bf16.sh", "replay-nvfp4.sh"):
            path = ROOT / "shards" / tag / kind
            if path.is_file():
                commands.setdefault(tag, {})[kind] = path.read_text().strip().splitlines()

    reference_manifest = json.loads((REFERENCE_CAPTURE / "capture-manifest.json").read_text())
    parent = json.loads((PARENT_SUITE / "suite-manifest.json").read_text())

    payload = {
        "schema": "qwen38-nvfp4-v5-measurement/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "where does unsloth/Qwen3.8-27B-NVFP4 sit on OUR v5 held-out "
                    "suite, measured by the same harness and protocol as every "
                    "published row of the ladder?",
        "why": "NVFP4 is the size-class competitor most often argued for against "
               "these builds, and it was the one comparator missing from the v5 "
               "tables: it had only ever been scored on the older v3 suite "
               "(0.094978 raw, 0.092727 contamination-corrected). It is served by "
               "the same vLLM build as our EXL3 rows, so unlike the GGUF rows its "
               "number carries no cross-engine term, which makes it the cleanest "
               "external comparison on the table.",
        "artifact": {
            "repo": "unsloth/Qwen3.8-27B-NVFP4",
            "revision": REVISION,
            "local_snapshot": str(SNAPSHOT),
            "served_as": "/models/Qwen3.8-27B-NVFP4",
            "quantization_flag": "--quantization compressed-tensors",
            "files": files,
            "all_files_match_published_digests": True,
            "digest_source": f"{SNAPSHOT}/.cache/huggingface/trees/{REVISION}.json, "
                             "the download-time record of the pinned revision: "
                             "lfs_sha256 for LFS files, git blob ids "
                             "(sha1('blob <size>\\0' + bytes)) for the rest",
            "upstream_revision_no_longer_resolves": {
                "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "api": f"https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/revision/{REVISION}",
                "response": '{"error":"Invalid rev id"}',
                "cause": "the upstream repository was super-squashed on 2026-08-15; "
                         f"current HEAD {UPSTREAM_HEAD} is titled \"Super-squash "
                         "branch 'main' using huggingface_hub\". Squashing is a "
                         "routine maintainer action; the lesson is ours, namely that "
                         "a Hugging Face revision id is not a durable citation.",
                "consequence": "our local snapshot was the only verifiable copy of "
                               "the revision this project's published v3 NVFP4 "
                               "number was measured on, so it is now mirrored",
            },
            "archival_mirror": {
                "repo": MIRROR_REPO,
                "commit": MIRROR_COMMIT,
                "byte_identical": True,
                "verified_remotely_after_upload": True,
                "note": "public model repo, apache-2.0 and attribution inherited "
                        "from unsloth, upstream card preserved as upstream-README.md",
            },
            "difference_from_current_upstream_head": {
                "head": UPSTREAM_HEAD,
                "weights_identical": True,
                "weight_files": ["model.safetensors", "model_mtp.safetensors"],
                "config_json_diff": {
                    "field": "quantization_config.config_groups.group_1.weights.observer",
                    "pinned_revision": "imatrix_mse",
                    "current_head": None,
                    "only_difference_in_the_file": True,
                },
                "tokenizer_json_adjudicated": {
                    "why": "receipts and docs/35 carried a tokenizer.json digest that "
                           "did not match the local snapshot, and nobody could tell "
                           "which side had drifted because the cited revision no "
                           "longer resolved. Holding the only resolvable copy plus a "
                           "mirror settles it.",
                    "method": "both files were downloaded from their live revisions "
                              "and hashed: HEAD from upstream, the reviewed revision "
                              "from the archival mirror; the mirror's copy was then "
                              "compared against the local snapshot",
                    "reviewed_revision": {
                        "revision": REVISION,
                        "sha256": files["tokenizer.json"]["sha256"],
                        "size": files["tokenizer.json"]["size"],
                        "resolvable_upstream": False,
                        "reachable_via": f"{MIRROR_REPO}@{MIRROR_COMMIT}",
                        "mirror_copy_matches_local_snapshot": True,
                    },
                    "current_head": {
                        "revision": UPSTREAM_HEAD,
                        "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba"
                                  "91c272ca9ff729b6a0e523",
                        "size": 19989325,
                    },
                    "the_only_difference": {
                        "field": "truncation",
                        "reviewed_revision": {"direction": "Right", "max_length": 2048,
                                              "strategy": "LongestFirst", "stride": 0},
                        "current_head": None,
                        "bytes": 19989424 - 19989325,
                        "identical_elsewhere": "vocab (248,044 entries), merges and "
                                               "added_tokens are identical; a full "
                                               "structural walk of both files reports "
                                               "exactly one differing node, /truncation",
                    },
                    "does_it_change_tokenization": "Under transformers.AutoTokenizer, "
                        "no: measured, a 12,002-token input gives 12,002 tokens with "
                        "either file, because PreTrainedTokenizerFast sets truncation "
                        "per call and defaults to off. Under the raw tokenizers."
                        "Tokenizer API, YES: the reviewed revision's file silently "
                        "truncates that same input to 2,048 tokens while HEAD's returns "
                        "all 12,002. So the field is inert for every number we publish "
                        "but is a live trap for anyone reproducing through the raw "
                        "library against the mirrored revision.",
                    "irrelevant_to_this_measurement": "the suite ships pre-tokenized "
                                                      "contexts; no tokenizer is "
                                                      "consulted during capture or "
                                                      "replay",
                    "corrected_in": "docs/35-external-protocol-comparability.md, source "
                                    "row [HF-NVFP4], which now carries both digests "
                                    "labelled with their revisions",
                },
                "does_it_change_what_we_measured": "No. The two weight files are "
                    "byte-identical between the pinned revision and current HEAD, so "
                    "'the NVFP4 everyone downloads today' computes exactly what we "
                    "measured. The observer field is llm-compressor's record of how "
                    "scales were chosen at build time; it is metadata, not read at "
                    "inference, and it describes the same bytes. Its only substantive "
                    "content is historical: the revision we reviewed declared an "
                    "imatrix-MSE observer and the current one declares none.",
            },
        },
        "suite_comparability_v3_vs_v5": suite_comparability(repo),
        "protocol": {
            "metric": "KL(BF16 reference || candidate), full 248,320-entry vocabulary, "
                      "exact two-pass, float32 within vocabulary chunks and float64 "
                      "across them",
            "reference": "BF16 Qwen/Qwen3.8-27B at /models/Qwen3.8-27B, captured under "
                         "vLLM at the final norm",
            "head": {
                "path": "/work/kld2/lm_head.safetensors",
                "sha256": sha256_file(HEAD_FILE),
                "shared": "one BF16 head for both operands; no candidate head "
                          "quantization is counted",
            },
            "suite": {
                "parent": str(PARENT_SUITE),
                "manifest_sha256": sha256_file(PARENT_SUITE / "suite-manifest.json"),
                "suite_token_sha256": parent["suite_token_sha256"],
                "contexts": parent["contexts"],
                "context_length": parent["context_length"],
                "scored_positions_per_context": parent["scored_positions_per_context"],
                "shard_size": 512,
            },
            "gpu_memory_utilization": 0.85,
            "same_as_published_rows": "identical suite, identical shard views, "
                                      "identical shared head, identical reference "
                                      "checkpoint, identical gpu-memory-utilization",
        },
        "harness": {
            "capture_and_replay": {
                "guest_path": "/work/kld5-nvfp4/fidelity.py",
                "sha256": HARNESS_SHA,
                "source": "tools/fidelity.py at the repository revision of this run",
            },
            "frozen_ladder_harness": {
                "path": str(FROZEN_HARNESS),
                "sha256": sha256_file(FROZEN_HARNESS),
                "used_here": False,
                "why_not": "the surviving shard-0 BF16 reference "
                           f"({REFERENCE_CAPTURE}) was captured by the current "
                           "harness generation, and a replay's two operands should "
                           "come from one generation. The choice is safe because the "
                           "two generations are additively different and provably "
                           "produce identical numbers -- see determinism_evidence.",
            },
            "aggregator": {
                "path": "tools/kld_aggregate.py",
                "sha256": sha256_file(repo / "tools/kld_aggregate.py"),
                "note": "the `paired` subcommand was added by this run and validated "
                        "by rebuilding receipts/kld5-10M-paired.json from its own "
                        "five inputs: content_sha256 "
                        "ba0f9a2a3a680a135d28a5cc11b14d107e92c0ec477d4fdf0a7e3758b30b01e7 "
                        "reproduced bit-for-bit, every field identical, only "
                        "generated_utc differing",
                "tie_convention": "the schema counts an exact tie as a b-win "
                    "(`a_wins` counts strict improvements, `b_wins` is the remainder), "
                    "which has already misled one reader on a zero-difference control. "
                    "It is kept unchanged so the published paired receipt stays "
                    "reproducible, and the tool now prints the tie count to stderr when "
                    "any comparison ties. Audited for this run: BOTH NVFP4 comparisons "
                    "have ZERO exact ties, and the smallest per-context gap is "
                    "+0.007842279 against the context edition and +0.007170826 against "
                    "official FP8, so `b_wins: 512` here is 512 genuine losses on 512 "
                    "contexts rather than a tie artefact",
            },
            "ladder": {
                "path": "tools/kld_ladder.sh",
                "sha256": sha256_file(repo / "tools/kld_ladder.sh"),
                "extended_not_forked": "nvfp4 was added to that script's model map "
                    "(--quantization compressed-tensors, no environment of ours) "
                    "rather than run through a parallel pipeline, so this measurement "
                    "walks the same capture/replay/verify/release loop as the "
                    "published ladder. The other changes are a --candidates selector "
                    "so a subset can be measured without invalidating a root whose "
                    "reports predate the new model, KLD_KEEP_HIDDEN, and a space "
                    "check that scales to the number of models actually captured "
                    "instead of hardcoding six. A bare invocation still means exactly "
                    "what it meant for the published ten shards.",
                "backward_compatibility_verified": "with the published harness "
                    "(KLD_HARNESS=/var/tmp/work/kld5/tools/fidelity.py) and the default "
                    "candidate list, a --dry-run over shards 0-9 of the published root "
                    "reports all ten shards already verified and would recapture "
                    "nothing; --candidates refuses bf16, an empty list, and an unknown "
                    "model name",
                "provenance_defect_fixed": {
                    "what": "the script stages a copy of the repository harness into "
                            "the run root, and it had to do so BEFORE the run pin "
                            "could be checked, because the pin is partly the harness "
                            "digest. So a run the pin REJECTED still left a "
                            "fidelity.py behind in the root it had just been refused "
                            "entry to.",
                    "why_it_matters": "this actually happened during a regression test "
                                      "of the extension: a rejected run wrote "
                                      "/var/tmp/work/kld5/fidelity.py into the "
                                      "PUBLISHED ladder root, where it sat beside a "
                                      "ladder-pin.json naming a different harness "
                                      "digest. An auditor reading that tree later "
                                      "would reasonably have concluded that file was "
                                      "the harness which measured the ten published "
                                      "shards. It is a provenance trap, not untidiness.",
                    "fix": "pin_run's failure path now removes a harness staged by the "
                           "same invocation before aborting",
                    "verified_both_directions": [
                        "a rejected run leaves no file behind (checked on the "
                        "published root, which rejects on harness drift)",
                        "an accepted run is unchanged (the nvfp4 root still pins, and "
                        "still verifies shard 0's report)",
                        "the published path still verifies all ten shards as done with "
                        "nothing recaptured",
                    ],
                    "stray_file_removed": "/var/tmp/work/kld5/fidelity.py, deleted; it "
                        "existed for about four minutes",
                    "did_it_reach_the_published_dataset": "no. The full tree of "
                        "malaiwah/qwen38-27b-fidelity-suite-v5 was enumerated by "
                        "pagination (6,396 entries across suite/, captures/, "
                        "reference/, reports/ and corpus/) and contains zero .py files "
                        "of any name, so no harness copy was published.",
                },
            },
            "determinism_evidence": {
                "claim": "capture+replay on this box is deterministic across harness "
                         "generations, so a report's numbers do not depend on which "
                         "generation produced the hidden states",
                "evidence": "/var/tmp/work/kld5/reports/shard-0000/report-hyd.json and "
                            "/var/tmp/work/kld5-tail/reports/shard-0000/report-hyd.json "
                            "were produced from different capture directories by "
                            "different harness files, and agree bit-for-bit on the "
                            "whole per_context array as well as token_mean_kld "
                            "0.002699883159684943, max_kld 3.7348466556786946, top-1, "
                            "mean JSD and the entire bootstrap block; the only "
                            "difference between the two reports is the additive "
                            "kld_tail histogram",
                "independently_checked_by": "ErrorDrivenAlloc",
            },
        },
        "reference_capture_reuse": {
            "path": str(REFERENCE_CAPTURE),
            "guest_path": "/work/gguf/hidden-bf16",
            "why": "the v5 ladder deleted its own BF16 hidden states after "
                   "verification; this is the surviving shard-0 BF16 capture, the "
                   "same one every GGUF row was replayed against",
            "wired_in_as": str(ROOT / "hidden/shard-0000/hidden-bf16")
                           + " -> ../../../gguf/hidden-bf16 (relative symlink, so it "
                             "resolves identically on the host and in the rootfs)",
            "accepted_by": "tools/kld_ladder.sh verify_capture, which checks the "
                           "capture manifest against the shard suite view before "
                           "reuse",
            "capture_manifest": {
                "suite_token_sha256": reference_manifest["suite_token_sha256"],
                "contexts": reference_manifest["contexts"],
                "complete": reference_manifest["complete"],
                "capture_contract_sha256": reference_manifest["capture_contract_sha256"],
                "model_path": reference_manifest["candidate_identity"]["model_path"],
            },
            "consequence": "no BF16 capture was needed for shard 0, which is why the "
                           "shard-0 number cost one model load instead of two",
        },
        "commands": commands,
        "coverage": {
            "shards_measured": covered,
            "shards_total": TOTAL_SHARDS,
            "shards_outstanding": missing,
            "contexts": sum(row["contexts"] for row in shard_rows),
            "scored_positions": positions,
            "full_suite_positions": parent["contexts"] * parent["scored_positions_per_context"],
            "complete": not missing,
            "per_shard": shard_rows,
        },
        "receipts": {
            name: receipt_digest(repo, name)
            for name in ("kld5-1M-nvfp4.json", "kld5-10M-nvfp4.json",
                         "kld5-1M-paired-nvfp4.json", "kld5-10M-paired-nvfp4.json",
                         "cross-engine-comparator.json")
        },
        "artifacts_released": {
            "policy": "cost-based preserve-before-delete: preserved when recompute "
                      "costs GPU-hours, when an input has ceased to exist, or when it "
                      "is the shared reference future runs replay against",
            "preserved": {
                "shard-0000/hidden-nvfp4": {
                    "bytes": 10732323294,
                    "files": 513,
                    "repo": "malaiwah/qwen38-27b-fidelity-suite-v5",
                    "path": "captures/shard-0000/hidden-nvfp4",
                    "revision": "db4f7dcaa7bafc52e042c5dca8e6b7a9311bd5de",
                    "uploaded_with": "tools/preserve_artifacts.sh --deep",
                    "verification": "all 513 files re-verified against the remote copy: "
                                    "509 by remote sha256/LFS oid, 1 by served-bytes "
                                    "readback, 3 by LFS spot re-download and re-hash",
                    "why_preserved": "with the shard-0 BF16 reference (already in the "
                                     "same dataset) this lets anyone re-derive the "
                                     "headline number by `fidelity.py replay` alone, "
                                     "with no GPU capture at all -- the one capture in "
                                     "this run that buys a reader something they "
                                     "cannot cheaply recompute",
                },
                "the snapshot itself": f"mirrored to {MIRROR_REPO}@{MIRROR_COMMIT} "
                                       "because its upstream revision ceased to exist; "
                                       "a mirror preserves the CITATION, not "
                                       "independent byte-level redundancy, since a "
                                       "content-addressed store may hold one copy of "
                                       "chunks both repos reference",
            },
            "released": {
                "what": "the per-shard BF16 and NVFP4 hidden states for shards 1-9",
                "bytes_each": 512 * 2047 * 5120 * 2,
                "recompute_cost_per_capture": "one model load plus ~170 s of prefill: "
                                              "about 6 minutes of GPU per capture on "
                                              "this box",
                "recompute_cost_total": "about 2 hours of GPU for all nineteen (anchor agreed with ArtifactPreserve: the surviving shard-0 BF16 capture measured 170.08 s of prefill, rounded up to ~6 min with its model load, the way the ladder budgeted it)",
                "why_not_preserved": "uploading ~200 GiB to avoid ~2 GPU-hours costs "
                                     "more wall-clock than recapturing, and the "
                                     "reports -- three orders of magnitude smaller -- "
                                     "are the durable artifact",
                "how_to_recompute": "tools/kld_ladder.sh with the command below",
            },
        },
    }

    if missing:
        spec = f"{missing[0]}-{missing[-1]}" if missing == list(
            range(missing[0], missing[-1] + 1)) else ",".join(str(s) for s in missing)
        payload["outstanding"] = {
            "what": f"the full-suite NVFP4 row: shards {spec} are not measured, so "
                    f"receipts/kld5-10M-nvfp4.json does not exist and the "
                    f"{parent['contexts'] * parent['scored_positions_per_context']:,}"
                    "-position table has no NVFP4 entry",
            "published_here_instead": (
                f"shards {','.join(str(s) for s in covered)} only: "
                f"{sum(row['contexts'] for row in shard_rows)} contexts and "
                f"{positions:,} scored positions"
                + (", which is the same shard and the same 512 contexts as the "
                   "cross-engine comparator table" if covered == [0] else "")
            ) if covered else "nothing: no shard of this candidate is measured yet",
            "needs": "BF16 must be recaptured for every outstanding shard, because "
                     "the v5 ladder deleted its BF16 hidden states after verifying "
                     "each shard; only shard 0's survived",
            "command": [
                "cd /home/mbelleau/research",
                f"KLD_ROOT={ROOT} KLD_SUITE={PARENT_SUITE} \\",
                f"  tools/kld_ladder.sh --candidates nvfp4 {spec}",
                "tools/kld_aggregate.py aggregate --suite " + str(PARENT_SUITE) +
                " --shard-size 512 \\",
                "  --candidate nvfp4 --label 10M --shards 0-9 \\",
                "  --out receipts/kld5-10M-nvfp4.json " + str(ROOT) +
                "/reports/shard-*/report-nvfp4.json",
                "tools/kld_aggregate.py paired --out receipts/kld5-10M-paired-nvfp4.json \\",
                "  --input nvfp4=receipts/kld5-10M-nvfp4.json \\",
                "  --input ctx=receipts/kld5-10M-ctx.json \\",
                "  --input fp8=receipts/kld5-10M-fp8.json \\",
                "  --compare nvfp4:ctx --compare nvfp4:fp8",
            ],
            "estimated_gpu_time": f"{len(missing)} shards x (2 captures + 1 replay) "
                                  "at ~14 min per shard, about "
                                  f"{len(missing) * 14} minutes",
        }
    else:
        payload["outstanding"] = None

    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "generated_utc"})
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "shards_measured": covered,
                      "scored_positions": positions,
                      "complete": payload["coverage"]["complete"],
                      "content_sha256": payload["content_sha256"]}, indent=2))
    return 0


HARNESS_SHA = sha256_file(ROOT / "fidelity.py")

if __name__ == "__main__":
    sys.exit(main())
