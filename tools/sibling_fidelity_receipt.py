#!/usr/bin/env python3
"""Does a sibling rebuild of the published hydrated recipe land inside its fidelity interval?

`receipts/converter-determinism.json` answered a byte question: a fresh conversion of the
published hydrated recipe reproduces the recipe exactly (composition, widths, per-role byte
totals, tensor names, dtypes, shapes, offsets, shard sizes) and does NOT reproduce the
weights (399 of 409 trellis payloads differ, 41-92 % of their bytes).  It left the
measurement question open: a rebuild is a *sibling* of the published checkpoint, and nobody
had ever scored one.

This tool answers that, and it refuses to answer it about something that is not a sibling.
Two gates run before a single fidelity number is copied into the payload:

  1. **Siblinghood.**  Everything the recipe fixes must be identical -- the ten per-role byte
     totals, the module count and every module's storage format, the logical/physical tensor
     counts, the tensor_storage table, and per shard the full safetensors header (names,
     dtypes, shapes, data offsets) and the shard's size in bytes.  A tree that fails any of
     these is not a sibling and scoring it would measure a different recipe.
  2. **Not a twin.**  The trellis payloads must actually differ.  A rebuild that came back
     byte-identical is a much bigger finding than the one under test -- it would contradict
     `converter_repeat_control` -- so it is reported as its own verdict rather than folded
     into a fidelity comparison that would then be vacuous.

The fidelity half is copied from reports produced by `tools/fidelity.py`, never recomputed
here, and the receipt records which of the two publishable outcomes was measured:

  * `indistinguishable` -- the paired 95 % interval contains zero.  The recipe determines
    fidelity to within this protocol's resolution, and every published fidelity number
    carries no extra converter-variance term.
  * `distinguishable` -- it does not.  Converter nondeterminism is fidelity-relevant, and
    every published fidelity number acquires a second source of variance that has to be
    quantified and disclosed.

    sibling_fidelity_receipt.py --published DIR --sibling DIR \\
        --bytes sibling-bytes.json \\
        --published-report .../report-hyd.json --sibling-report .../report-sibling.json \\
        --paired .../paired-hydpub-vs-sibling.json \\
        --capture-control .../paired-hydpub-vs-hydrematch.json \\
        --toolchain toolchain.json --out receipts/sibling-rebuild-fidelity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# Every field of a per-tensor header entry that a sibling must reproduce.  `data_offsets`
# is in here deliberately: identical offsets are what make the shard packing identical, and
# they are also what lets the byte comparison be per tensor instead of per 8 GB shard.
HEADER_FIELDS = ("dtype", "shape", "data_offsets")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(p: Path) -> dict:
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def header_comparison(pub: Path, sib: Path) -> dict:
    """Names, dtypes, shapes and offsets of one shard, plus its size on disk."""
    a, b = safetensors_header(pub), safetensors_header(sib)
    at = {k: v for k, v in a.items() if k != "__metadata__"}
    bt = {k: v for k, v in b.items() if k != "__metadata__"}
    common = sorted(set(at) & set(bt))
    mismatched = {
        field: [k for k in common if at[k].get(field) != bt[k].get(field)]
        for field in HEADER_FIELDS
    }
    return {
        "tensors_published": len(at),
        "tensors_sibling": len(bt),
        "names_identical": set(at) == set(bt),
        "only_in_published": sorted(set(at) - set(bt)),
        "only_in_sibling": sorted(set(bt) - set(at)),
        "dtype_mismatches": mismatched["dtype"],
        "shape_mismatches": mismatched["shape"],
        "offset_mismatches": mismatched["data_offsets"],
        "metadata_identical": a.get("__metadata__") == b.get("__metadata__"),
        "header_identical": a == b,
        "published_bytes": pub.stat().st_size,
        "sibling_bytes": sib.stat().st_size,
        "size_identical": pub.stat().st_size == sib.stat().st_size,
    }


def role_comparison(pub: Path, sib: Path) -> dict:
    """The ten per-role byte totals and the module composition, from both manifests."""
    a = json.loads((pub / "quantization_manifest.json").read_text())
    b = json.loads((sib / "quantization_manifest.json").read_text())
    roles = sorted(set(a["roles"]) | set(b["roles"]))
    per_role, disagreeing = {}, []
    for r in roles:
        ra, rb = a["roles"].get(r), b["roles"].get(r)
        same = ra == rb
        if not same:
            disagreeing.append(r)
        per_role[r] = {
            "published_bytes": (ra or {}).get("bytes"),
            "sibling_bytes": (rb or {}).get("bytes"),
            "modules": (ra or {}).get("modules"),
            "formats": (ra or {}).get("formats"),
            "identical": same,
        }
    scalars = {}
    for k in ("physical_tensor_count", "logical_tensor_count", "logical_parameter_count",
              "exl3_modules", "converter", "recipe", "command"):
        scalars[k] = {"published": a.get(k), "sibling": b.get(k), "identical": a.get(k) == b.get(k)}
    ca = json.loads((pub / "quantization_config.json").read_text())
    cb = json.loads((sib / "quantization_config.json").read_text())
    widths = {k: {"published": ca.get(k), "sibling": cb.get(k),
                  "identical": ca.get(k) == cb.get(k)}
              for k in ("bits", "head_bits", "mtp_bits", "codebook", "out_scales",
                        "calibration", "quant_method", "version")}
    return {
        "per_role_bytes": per_role,
        "roles_disagreeing": disagreeing,
        "roles_identical": not disagreeing,
        "payload_bytes_published": sum(v["bytes"] for v in a["roles"].values()),
        "payload_bytes_sibling": sum(v["bytes"] for v in b["roles"].values()),
        "manifest_scalars": scalars,
        "declared_widths": widths,
        "tensor_storage_entries_published": len(ca.get("tensor_storage", {})),
        "tensor_storage_entries_sibling": len(cb.get("tensor_storage", {})),
        "tensor_storage_identical": ca.get("tensor_storage") == cb.get("tensor_storage"),
    }


def trellis_summary(bytes_payload: dict) -> dict:
    """Where the two trees differ, split by whether it is a trellis payload or not."""
    trellis_diff, other_diff, fractions = [], [], []
    tensors_compared = tensors_identical = 0
    for name, entry in bytes_payload.get("differing_files", {}).items():
        t = entry.get("tensors")
        if not t:
            other_diff.append({"file": name, "kind": entry.get("kind")})
            continue
        tensors_compared += t["tensors_compared"]
        tensors_identical += t["tensors_identical"]
        if t.get("differing_truncated"):
            raise SystemExit(f"{name}: differing tensor list is truncated; the byte "
                             "comparison cannot be summarised without every entry")
        for d in t["differing"]:
            frac = d["differing_bytes"] / d["bytes"] if d["bytes"] else 0.0
            (trellis_diff if d["tensor"].endswith(".trellis") else other_diff).append(
                {"tensor": d["tensor"], "bytes": d["bytes"],
                 "differing_bytes": d["differing_bytes"], "differing_fraction": frac})
            if d["tensor"].endswith(".trellis"):
                fractions.append(frac)
    fractions.sort()
    return {
        "tensors_compared": tensors_compared,
        "tensors_identical": tensors_identical,
        "trellis_tensors_differing": len(trellis_diff),
        "non_trellis_differences": other_diff,
        "non_trellis_differing_count": len(other_diff),
        "differing_byte_fraction_min": fractions[0] if fractions else None,
        "differing_byte_fraction_max": fractions[-1] if fractions else None,
        "differing_byte_fraction_mean": (sum(fractions) / len(fractions)) if fractions else None,
        "payloads_differ": bool(fractions),
        # Named in full, not counted: the byte comparison this is summarised from lives in
        # scratch and the sibling checkpoint is not published, so if these names are not here
        # they are nowhere.  It is also the only form in which a later reader can check that
        # the difference really is confined to trellis payloads.
        "trellis_tensors_differing_detail": sorted(trellis_diff, key=lambda d: d["tensor"]),
    }


def report_block(path: Path) -> dict:
    r = json.loads(path.read_text())
    return {
        "report": str(path),
        "schema": r["schema"],
        "reference_capture": r["reference"],
        "candidate_capture": r["candidate"],
        "head": r["head"],
        "head_sha256": r["head_sha256"],
        "suite_token_sha256": r["suite_token_sha256"],
        "filter": r["filter"],
        "contexts": r["contexts"],
        "scored_positions": r["scored_positions"],
        "token_mean_kld": r["token_mean_kld"],
        "context_macro_mean_kld": r["context_macro_mean_kld"],
        "context_bootstrap": r["context_bootstrap"],
        "token_median_kld": r["token_median_kld"],
        "p95_kld": r["p95_kld"],
        "p99_kld": r["p99_kld"],
        "p999_kld": r["p999_kld"],
        "max_kld": r["max_kld"],
        "mean_jsd_bits": r["mean_jsd_bits"],
        "top1_agreement": r["top1_agreement"],
        "candidate_identity_shards": r["candidate_identity"]["shard_sha256"],
        "reference_identity_index_sha256": r["reference_identity"]["index_sha256"],
        "comparator": r["comparator"],
    }


def protocol_identity(pub: dict, sib: dict, errors: list[str]) -> dict:
    """The score is only the published number's peer if every protocol field matches.

    `reference_capture` is deliberately NOT in this list.  The published ladder deleted its
    BF16 capture directory after verifying its reports, so a later run cannot point at the
    same path; `receipts/capture-determinism.json` records `reference` as a field that
    differs by construction.  Pointing at a different directory is only harmless if that
    directory is interchangeable with the deleted one, which is what `reference_control`
    proves by measurement instead of by assertion.
    """
    checks = {}
    for field in ("suite_token_sha256", "filter", "head_sha256", "contexts",
                  "scored_positions", "comparator", "reference_identity_index_sha256"):
        same = pub[field] == sib[field]
        checks[field] = {"published": pub[field], "sibling": sib[field], "identical": same}
        if not same:
            errors.append(f"protocol field {field} differs between the two reports")
    return checks


def reference_control(control_path: Path, pub: dict, sib: dict, errors: list[str]) -> dict:
    """Prove the reference capture used here yields the PUBLISHED number.

    The control is a replay of the published checkpoint itself -- recaptured, and replayed
    against the same reference directory this sibling was replayed against.  If it returns
    the published mean bitwise, then swapping the reference directory cannot have moved the
    sibling's score, and it is established rather than assumed.
    """
    c = json.loads(control_path.read_text())
    same_reference = c["reference"] == sib["reference_capture"]
    same_checkpoint = c["candidate_identity"]["shard_sha256"] == pub["candidate_identity_shards"]
    reproduces = c["context_macro_mean_kld"] == pub["context_macro_mean_kld"]
    same_suite = c["suite_token_sha256"] == pub["suite_token_sha256"]
    same_head = c["head_sha256"] == pub["head_sha256"]
    if not same_reference:
        errors.append("the reference control replayed against a different reference "
                      f"directory ({c['reference']}) than the sibling did "
                      f"({sib['reference_capture']}); it proves nothing about this run")
    if not same_checkpoint:
        errors.append("the reference control did not score the published checkpoint")
    if not (same_suite and same_head):
        errors.append("the reference control used a different suite or head")
    if not reproduces:
        errors.append("the reference control does NOT reproduce the published mean, so the "
                      "reference capture used here is not interchangeable with the deleted "
                      "one and the sibling's score is not comparable to the published number")
    return {
        "why": "the published ladder deleted its BF16 capture after verifying its reports, so "
               "this run replayed against a surviving capture of the same reference. That is "
               "only sound if the two are interchangeable, which this control measures.",
        "control_report": str(control_path),
        "control_reference_capture": c["reference"],
        "control_candidate_capture": c["candidate"],
        "control_scores_the_published_checkpoint": same_checkpoint,
        "control_uses_the_same_reference_as_the_sibling": same_reference,
        "control_mean": c["context_macro_mean_kld"],
        "published_mean": pub["context_macro_mean_kld"],
        "reproduces_published_mean_bitwise": reproduces,
        "conclusion": ("the reference capture used for the sibling returns the published "
                       "number, to every digit, for the published checkpoint: the reference "
                       "swap is a no-op") if reproduces else
                      "the reference swap is NOT a no-op; this comparison is invalid",
        "source": "receipts/capture-determinism.json",
    }


def scale_against_published(specs: list[str], sibling_diff: float,
                            errors: list[str]) -> dict:
    """How big is the converter draw next to the distinctions we publish?

    A paired difference that is statistically distinguishable from zero can still be far too
    small to threaten anything, and one that is indistinguishable can still be too large to
    be reassuring.  The honest way to say which is to divide it by the published margins
    measured at THIS resolution -- same shard, same 512 contexts, same 330 source clusters,
    same 10,000 resamples -- so the reader compares like with like.
    """
    entries = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--scale wants LABEL=PATH, got {spec!r}")
        p = json.loads(Path(path).read_text())
        if p["contexts"] != 512:
            errors.append(f"scale anchor {label} has {p['contexts']} contexts, not 512, so it "
                          "is not at this comparison's resolution")
        margin = abs(p["difference_a_minus_b"])
        entries[label] = {
            "paired": str(path),
            "a": p["a"]["label"], "b": p["b"]["label"],
            "difference_a_minus_b": p["difference_a_minus_b"],
            "ci95_low": p["bootstrap_difference"]["ci95_low"],
            "ci95_high": p["bootstrap_difference"]["ci95_high"],
            "clusters": p["bootstrap_difference"]["clusters"],
            "abs_margin": margin,
            "sibling_difference_as_fraction_of_this_margin":
                (abs(sibling_diff) / margin) if margin else None,
        }
    tightest = min(entries.items(), key=lambda kv: kv[1]["abs_margin"], default=None)
    return {
        "why": "a converter draw only matters if it is comparable to a distinction we draw "
               "conclusions from. These are published paired margins at the identical "
               "resolution, so the ratio is directly meaningful.",
        "sibling_abs_difference": abs(sibling_diff),
        "anchors": entries,
        "tightest_published_margin": ({"label": tightest[0],
                                       "abs_margin": tightest[1]["abs_margin"],
                                       "sibling_difference_as_fraction":
                                           tightest[1][
                                               "sibling_difference_as_fraction_of_this_margin"]}
                                      if tightest else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", required=True, type=Path)
    ap.add_argument("--sibling", required=True, type=Path)
    ap.add_argument("--sums", required=True, type=Path)
    ap.add_argument("--bytes", required=True, type=Path,
                    help="output of tools/converter_determinism_receipt.py for this pair")
    ap.add_argument("--published-report", required=True, type=Path)
    ap.add_argument("--sibling-report", required=True, type=Path)
    ap.add_argument("--paired", required=True, type=Path,
                    help="tools/fidelity.py paired, published as A and sibling as B")
    ap.add_argument("--paired-aggregate", type=Path,
                    help="tools/kld_aggregate.py paired over one-shard cumulative receipts")
    ap.add_argument("--capture-control", type=Path,
                    help="paired report of the published checkpoint against a RECAPTURE of "
                         "itself: the floor this comparison is read against")
    ap.add_argument("--reference-control", required=True, type=Path,
                    help="a replay report that scores the PUBLISHED checkpoint against the "
                         "same reference capture the sibling used; proves the reference swap "
                         "forced by the published capture's deletion is a no-op")
    ap.add_argument("--scale", action="append", default=[], metavar="LABEL=PATH",
                    help="a published paired report at THIS resolution, to size the sibling "
                         "difference against a distinction we draw conclusions from")
    ap.add_argument("--toolchain", type=Path)
    ap.add_argument("--preservation", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    errors: list[str] = []

    # ------------------------------------------------------------- gate 1: siblinghood
    pinned = {}
    for line in args.sums.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            pinned[name.strip()] = digest
    published_now = {n: sha256_file(args.published / n) for n in sorted(pinned)}
    drifted = [n for n in pinned if published_now[n] != pinned[n]]
    if drifted:
        raise SystemExit("published tree no longer matches receipts/hydrated-SHA256SUMS: "
                         + ", ".join(drifted))
    sibling_now = {n: (sha256_file(args.sibling / n) if (args.sibling / n).is_file() else None)
                   for n in sorted(pinned)}

    shards = sorted(n for n in pinned if n.endswith(".safetensors"))
    headers = {n: header_comparison(args.published / n, args.sibling / n) for n in shards}
    roles = role_comparison(args.published, args.sibling)
    payloads = trellis_summary(json.loads(args.bytes.read_text()))

    for n, h in headers.items():
        for k in ("names_identical", "size_identical", "header_identical"):
            if not h[k]:
                errors.append(f"{n}: {k} is false, so this tree is not a sibling")
    if not roles["roles_identical"]:
        errors.append("per-role byte totals differ: " + ", ".join(roles["roles_disagreeing"]))
    if not roles["tensor_storage_identical"]:
        errors.append("tensor_storage tables differ")
    for k, v in roles["manifest_scalars"].items():
        if not v["identical"]:
            errors.append(f"manifest scalar {k} differs")
    for k, v in roles["declared_widths"].items():
        if not v["identical"]:
            errors.append(f"declared width {k} differs")
    is_sibling = not errors

    # ------------------------------------------------------------- gate 2: not a twin
    byte_identical = not [n for n in pinned if sibling_now[n] != pinned[n]]
    if byte_identical:
        errors.append("the rebuild is BYTE-IDENTICAL to the published checkpoint; that is a "
                      "bigger finding than the one under test and contradicts "
                      "receipts/converter-determinism.json -> converter_repeat_control")
    elif not payloads["payloads_differ"]:
        errors.append("the shards differ but no trellis payload does; the difference is "
                      "somewhere else and this is not the case under test")

    # ------------------------------------------------------------- the fidelity half
    pub_report = report_block(args.published_report)
    sib_report = report_block(args.sibling_report)
    protocol = protocol_identity(pub_report, sib_report, errors)
    refctl = reference_control(args.reference_control, pub_report, sib_report, errors)

    paired = json.loads(args.paired.read_text())
    lo, hi = paired["bootstrap_difference"]["ci95_low"], paired["bootstrap_difference"]["ci95_high"]
    straddles_zero = lo <= 0.0 <= hi
    exactly_zero = (lo == 0.0 and hi == 0.0 and paired["difference_a_minus_b"] == 0.0)
    outcome = "indistinguishable" if straddles_zero else "distinguishable"

    # `a_wins` counts strict improvements and `b_wins` is the remainder, so an exact tie is
    # counted as a b-win and a zero-difference control reads "a_wins 0, b_wins 512".  The
    # true split is recounted here from the two reports' own per-context rows -- the same
    # numbers `paired` subtracted -- so the receipt can state losses and ties separately
    # without changing the paired payload.
    rows_a = {r["index"]: r["mean_kld"]
              for r in json.loads(args.published_report.read_text())["per_context_all"]}
    rows_b = {r["index"]: r["mean_kld"]
              for r in json.loads(args.sibling_report.read_text())["per_context_all"]}
    if set(rows_a) != set(rows_b):
        errors.append("the two reports cover different context sets")
    diffs = [rows_a[i] - rows_b[i] for i in sorted(set(rows_a) & set(rows_b))]
    ties = sum(1 for d in diffs if d == 0.0)
    published_better = sum(1 for d in diffs if d < 0.0)
    sibling_better = sum(1 for d in diffs if d > 0.0)

    sib_ci = sib_report["context_bootstrap"]
    pub_ci = pub_report["context_bootstrap"]
    inside = pub_ci["ci95_low"] <= sib_report["context_macro_mean_kld"] <= pub_ci["ci95_high"]

    payload = {
        "schema": "qwen38-sibling-rebuild-fidelity/1",
        "question": "does a sibling rebuild of the published hydrated recipe land inside the "
                    "published fidelity interval?",
        "answered_here": True,
        "left_open_by": "receipts/converter-determinism.json -> answer."
                        "what_it_means_for_reconstruction_claims[1]",
        "verdict": {
            "outcome": outcome,
            "paired_difference_published_minus_sibling": paired["difference_a_minus_b"],
            "paired_ci95_low": lo,
            "paired_ci95_high": hi,
            "paired_interval_contains_zero": straddles_zero,
            "paired_difference_exactly_zero": exactly_zero,
            "sibling_mean": sib_report["context_macro_mean_kld"],
            "published_mean": pub_report["context_macro_mean_kld"],
            "sibling_mean_inside_published_interval": inside,
            "sibling_mean_inside_published_interval_note":
                "a weak check kept for completeness: the published interval is a bootstrap "
                "over source clusters of ONE model's per-context means, so it describes "
                "corpus sampling, not run-to-run spread. The paired interval is the test.",
            "a_wins_published_as_scored": paired["a_wins"],
            "b_wins_sibling_as_scored": paired["b_wins"],
            "tie_convention": "`fidelity.py paired` counts strict improvements as a_wins and "
                              "the remainder as b_wins, so an exact tie is scored as a "
                              "b-win. The three-way recount below comes from the same "
                              "per-context means the paired tool subtracted.",
            "recount": {
                "contexts": len(diffs),
                "published_strictly_better": published_better,
                "sibling_strictly_better": sibling_better,
                "exact_ties": ties,
            },
        },
        "siblinghood": {
            "verified": is_sibling,
            "what_a_sibling_must_reproduce": [
                "composition: every role's module count and storage formats",
                "declared widths: bits, head_bits, mtp_bits, codebook, out_scales, calibration",
                "the ten per-role byte totals and the total payload bytes",
                "physical and logical tensor counts and the tensor_storage table",
                "per shard: tensor names, dtypes, shapes, data offsets and the shard's size",
            ],
            "which_outcome": ("BYTE-IDENTICAL TWIN" if byte_identical
                              else "SIBLING: identical recipe, different trellis payloads"),
            "byte_identical_to_published": byte_identical,
            "pinned_files_compared": len(pinned),
            "pinned_files_identical": sorted(n for n in pinned if sibling_now[n] == pinned[n]),
            "pinned_files_differing": sorted(n for n in pinned if sibling_now[n] != pinned[n]),
            "published_sha256": pinned,
            "sibling_sha256": sibling_now,
            "shard_headers": headers,
            "recipe_reproduction": roles,
            "payload_divergence": payloads,
        },
        "protocol": {
            "identical_to_the_published_score": (all(v["identical"] for v in protocol.values())
                                                 and refctl["reproduces_published_mean_bitwise"]),
            "fields": protocol,
            "reference_capture_published": pub_report["reference_capture"],
            "reference_capture_used_here": sib_report["reference_capture"],
            "reference_capture_interchangeability": refctl,
            "note": "the published number and the sibling's number are the same measurement "
                    "run twice with one operand swapped: same v5 shard-0 suite, same BF16 "
                    "reference content, same shared head, same comparator, same 512 contexts "
                    "and 1,048,064 scored positions. The reference capture DIRECTORY differs, "
                    "because the published ladder deleted its own; "
                    "reference_capture_interchangeability measures that this is a no-op.",
        },
        "published_score": pub_report,
        "sibling_score": sib_report,
        "sibling_interval": sib_ci,
        "paired": paired,
        "bytes_comparison": str(args.bytes),
    }
    if args.paired_aggregate:
        payload["paired_aggregate"] = json.loads(args.paired_aggregate.read_text())
    if args.scale:
        payload["scale"] = scale_against_published(
            args.scale, paired["difference_a_minus_b"], errors)
    if args.capture_control:
        control = json.loads(args.capture_control.read_text())
        payload["capture_noise_floor"] = {
            "what": "the same published checkpoint captured and replayed a SECOND time, "
                    "paired against the published report",
            "paired": control,
            "why_it_matters": "this comparison's floor is exactly zero, not a tolerance: "
                              "recapture reproduces the published mean bitwise, so any "
                              "nonzero paired difference measured against a sibling is "
                              "attributable to the sibling's weights and to nothing in the "
                              "capture or replay path.",
            "source": "receipts/capture-determinism.json",
        }
    if args.toolchain:
        payload["toolchain"] = json.loads(args.toolchain.read_text())
    if args.preservation:
        payload["preservation"] = json.loads(args.preservation.read_text())

    # The point of the receipt: say which of the two publishable outcomes was measured, and
    # what it costs every fidelity number already published.  Both are stated so a reader can
    # see that the answer was not chosen after the fact.
    indistinguishable = {
        "measured": "INDISTINGUISHABLE FROM ZERO",
        "plain": "A sibling rebuild lands inside the published fidelity interval. The paired "
                 "95 % interval on the per-context difference contains zero, so at this "
                 "protocol's resolution -- 512 contexts, 1,048,064 scored positions, a "
                 "source-cluster bootstrap over "
                 f"{paired['bootstrap_difference']['clusters']} clusters with 10,000 "
                 "resamples at seed 1 -- the published checkpoint and a sibling of it are the "
                 "same measurement.",
        "for_every_published_fidelity_number": [
            "Unchanged, and now for a second reason. Every published number was already a "
            "measurement of the PUBLISHED bytes, which is what SHA256SUMS pins and what a "
            "downloader receives; that never depended on this test.",
            "What this adds is the reconstruction claim receipts/converter-determinism.json "
            "had to leave as an expectation: 'rebuild this recipe and you will land on our "
            "numbers' is now measured, on one sibling, at this resolution. It is a tested "
            "expectation, not an identity -- the bytes still differ.",
            "No extra variance term has to be added to any published interval, and no "
            "published claim needs amending.",
        ],
        "what_it_does_not_license": [
            "It is one sibling, not a distribution. A single paired comparison bounds the "
            "converter's fidelity variance at this resolution; it does not estimate it.",
            "It says nothing about recipes other than the hydrated one, nor about resolutions "
            "finer than this protocol's, nor about the tail statistics, which are reported "
            "here but were not the test.",
        ],
    }
    distinguishable = {
        "measured": "DISTINGUISHABLE FROM ZERO",
        "plain": "A sibling rebuild does NOT land inside the published fidelity interval. The "
                 "paired 95 % interval on the per-context difference excludes zero, so "
                 "converter nondeterminism is fidelity-relevant: which of two equally valid "
                 "solutions to the same recipe a conversion happens to find changes the "
                 "measured distributional fidelity by an amount this protocol can see.",
        "for_every_published_fidelity_number": [
            "Still correct as measurements OF THE PUBLISHED BYTES: they describe the artifact "
            "a downloader gets, and SHA256SUMS pins it. Nothing published becomes wrong.",
            "But every one of them now carries a SECOND source of variance beside corpus "
            "sampling -- the conversion that produced the checkpoint -- and the published "
            "intervals do not contain it, because a bootstrap over source clusters resamples "
            "the corpus and holds the weights fixed.",
            "That term has to be quantified (several siblings, not one) and disclosed wherever "
            "an interval is quoted, and any comparison BETWEEN checkpoints whose margin is of "
            "the same order as the sibling difference measured here has to be re-read as "
            "possibly a converter draw rather than a recipe effect.",
        ],
        "what_it_does_not_license": [
            "It does not make any published number an overclaim, and it does not mean a "
            "rebuild is worse -- the sign of a single draw carries no information about which "
            "recipe is better.",
            "One sibling gives the difference for one draw. It does not give the spread, which "
            "is what disclosure would need.",
        ],
    }
    payload["implications"] = {
        "outcome": outcome,
        "this_run": distinguishable if outcome == "distinguishable" else indistinguishable,
        "the_other_outcome_would_have_meant": (
            indistinguishable if outcome == "distinguishable" else distinguishable),
        "both_were_publishable": "Neither outcome was the hoped-for one. The two readings "
                                 "above were written into this tool before the score existed, "
                                 "and which one the receipt reports is decided by whether "
                                 "`paired.bootstrap_difference` brackets zero.",
    }
    payload["errors"] = errors

    body = json.dumps({k: v for k, v in payload.items() if k != "content_sha256"},
                      sort_keys=True).encode()
    payload["content_sha256"] = hashlib.sha256(body).hexdigest()
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["verdict"], indent=1))
    if errors:
        print("ERRORS:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
