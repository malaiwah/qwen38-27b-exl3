#!/usr/bin/env python3
"""Cumulative KLD receipt from per-shard fidelity reports.

The ladder measures one candidate against the BF16 reference in shards of 512
contexts, because six models' hidden states for a shard already fill 64 GB of
scratch.  Each shard produces an ordinary `qwen38-fidelity-report/1` from
`fidelity.py replay`; this tool welds a set of those shard reports into one
receipt that states what the whole run measured.

What is aggregable and what is not:

  * position-weighted token mean KLD, top-1 agreement and scored positions are
    exact sums over the per-context rows, so they are recomputed here rather
    than averaged from the shard summaries -- and then cross-checked against
    each shard's own `token_mean_kld` / `top1_agreement`;
  * the macro mean over contexts and the source-cluster bootstrap are computed
    over the *union* of per-context rows, so the confidence interval reflects
    every cluster the ladder touched, not a mean of per-shard intervals.  The
    resampler is byte-for-byte the one in `fidelity.py`, so a single-shard
    aggregate reproduces that shard's own `context_bootstrap` exactly;
  * token-level percentiles (p95/p99/p999/median) are NOT aggregable from
    per-context rows and are deliberately absent.  `max_kld` is aggregable and
    is carried as the max over shards.

Fail-closed rules -- any of these aborts without writing:

  * a report that is not `qwen38-fidelity-report/1`, or lacks a required field;
  * disagreeing candidate/reference model identity, head digest, comparator,
    filter, vocabulary or hidden size across reports;
  * a context index that appears in more than one report (overlap);
  * with `--shard-size` (default 512): a partially covered shard, a gap in the
    covered shard set, or a covered set that differs from `--shards`;
  * with `--suite`: a report whose `suite_token_sha256` is not the digest of
    exactly the contexts it scored, taken in parent-suite order, or a context
    index that the parent suite does not contain;
  * without `--suite`: reports that declare different `suite_token_sha256`;
  * a shard summary that disagrees with its own per-context rows by more than
    `--tolerance` (relative).

Usage:

    tools/kld_aggregate.py aggregate --out receipts/kld5-k5k6-2M.json \\
        --suite /var/tmp/work/kld5/suite --shards 0-1 \\
        /var/tmp/work/kld5/reports/shard-0000/report-k5k6.json \\
        /var/tmp/work/kld5/reports/shard-0001/report-k5k6.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "qwen38-kld-ladder-cumulative/1"
REPORT_SCHEMA = "qwen38-fidelity-report/1"

# Cumulative scored-position checkpoints of the published ladder.
LADDER_TARGETS = (
    ("1M", 1_000_000),
    ("2M", 2_000_000),
    ("5M", 5_000_000),
    ("10M", 10_000_000),
)

# Report fields that must be identical across every shard of one candidate.  The
# `reference`/`candidate` capture directories are deliberately absent: a ladder
# stores each shard's hidden states in its own directory (and deletes them), so
# those paths differ by construction.  What must not differ is the identity of
# the checkpoints they came from, which `IDENTITY_FIELDS` covers.
SHARED_FIELDS = (
    "head",
    "head_sha256",
    "candidate_head",
    "candidate_head_sha256",
    "filter",
    "vocab_size",
    "hidden_size",
    "comparator",
)

# Identity sub-fields that pin the operands to actual bytes on disk.
IDENTITY_FIELDS = (
    "model_path",
    "model_revision",
    "model_revision_source",
    "index_sha256",
    "config_sha256",
    "shard_sha256",
    "quantization",
    "quantization_config",
    "trust_remote_code",
    "kv_cache_dtype_requested",
    "kv_cache_dtype_resolved",
)

REQUIRED_REPORT_FIELDS = (
    "schema",
    "contexts",
    "scored_positions",
    "token_mean_kld",
    "top1_agreement",
    "suite_token_sha256",
    "per_context_all",
)

ROW_FIELDS = ("index", "source_cluster", "stratum", "positions_scored",
              "mean_kld", "top1_agreement")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a receipt atomically, so an interrupted run cannot bless partial work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def bootstrap(values: list[float], clusters: list[str], samples: int, seed: int) -> dict:
    """Source-cluster bootstrap of the macro mean over contexts.

    Resampling whole source clusters -- not contexts -- is what makes the
    interval honest: contexts cut from one document are not independent.  The
    resample, the cluster order, the RNG (`random.Random(seed)`,
    `rng.randrange(len(keys))` once per cluster slot) and the percentile index
    arithmetic are the ones in `fidelity.py`, so this reproduces a shard
    report's own interval.

    One deliberate difference: a resampled mean is `fsum(cluster sums) / total
    count` rather than `sum(flattened values) / len`.  Both are the same number
    in exact arithmetic, but the builtin `sum` changed float behaviour in
    CPython 3.12 (compensated summation), so the flatten form makes the published
    interval depend on which interpreter runs the tool.  `math.fsum` is exactly
    rounded on every interpreter; residual disagreement with a report computed by
    the flatten form is ~1e-16 relative.
    """
    import random
    by: dict[str, list[float]] = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    sums = [math.fsum(by[k]) for k in keys]
    counts = [len(by[k]) for k in keys]
    rng = random.Random(seed)
    nkeys = len(keys)
    means = []
    for _ in range(samples):
        picked = [rng.randrange(nkeys) for _ in range(nkeys)]
        means.append(math.fsum(sums[j] for j in picked)
                     / sum(counts[j] for j in picked))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return {"mean": statistics.fmean(values), "ci95_low": lo, "ci95_high": hi,
            "clusters": len(keys), "samples": samples}


def parse_index_list(spec: str) -> list[int]:
    """`0-3,7` -> [0, 1, 2, 3, 7]; rejects anything else."""
    out: set[int] = set()
    for piece in spec.replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            lo_s, hi_s = piece.split("-", 1)
            if not lo_s.isdigit() or not hi_s.isdigit():
                raise SystemExit(f"bad shard range {piece!r}")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise SystemExit(f"bad shard range {piece!r}")
            out.update(range(lo, hi + 1))
        else:
            if not piece.isdigit():
                raise SystemExit(f"bad shard index {piece!r}")
            out.add(int(piece))
    if not out:
        raise SystemExit("empty shard list")
    return sorted(out)


def relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


class Rejected(SystemExit):
    """Every violation found, reported together rather than one per run."""

    def __init__(self, errors: list[str]) -> None:
        body = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"REJECTED: {len(errors)} problem(s) in the shard set:\n{body}")


def load_report(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"{path}: missing report")
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: unreadable report ({exc})")
        return None
    if not isinstance(report, dict):
        errors.append(f"{path}: report is not an object")
        return None
    if report.get("schema") != REPORT_SCHEMA:
        errors.append(f"{path}: schema {report.get('schema')!r} != {REPORT_SCHEMA!r}")
        return None
    for field in REQUIRED_REPORT_FIELDS:
        if field not in report:
            errors.append(f"{path}: report lacks {field}")
            return None
    rows = report["per_context_all"]
    if not isinstance(rows, list) or not rows:
        errors.append(f"{path}: per_context_all is empty or not a list")
        return None
    for row in rows:
        if not isinstance(row, dict) or any(f not in row for f in ROW_FIELDS):
            errors.append(f"{path}: per_context_all row lacks {ROW_FIELDS}")
            return None
    return report


def identity_view(report: dict) -> dict:
    """The subset of a report that must agree across shards."""
    view = {f: report.get(f) for f in SHARED_FIELDS}
    for side in ("reference_identity", "candidate_identity"):
        block = report.get(side) or {}
        view[side] = {f: block.get(f) for f in IDENTITY_FIELDS}
    return view


def check_rows_against_summary(path: Path, report: dict, tolerance: float,
                              errors: list[str]) -> dict:
    """A shard's headline numbers must follow from its own per-context rows."""
    rows = report["per_context_all"]
    positions = sum(int(r["positions_scored"]) for r in rows)
    if positions <= 0:
        errors.append(f"{path}: non-positive scored positions")
        return {"token_mean_gap": 0.0, "top1_gap": 0.0, "positions": 0}
    weighted_kld = math.fsum(
        float(r["mean_kld"]) * int(r["positions_scored"]) for r in rows) / positions
    weighted_top1 = math.fsum(
        float(r["top1_agreement"]) * int(r["positions_scored"]) for r in rows) / positions
    if len(rows) != int(report["contexts"]):
        errors.append(
            f"{path}: contexts={report['contexts']} but {len(rows)} per-context rows"
        )
    if positions != int(report["scored_positions"]):
        errors.append(
            f"{path}: scored_positions={report['scored_positions']} but rows sum to {positions}"
        )
    kld_gap = relative_gap(weighted_kld, float(report["token_mean_kld"]))
    top1_gap = relative_gap(weighted_top1, float(report["top1_agreement"]))
    if not math.isfinite(float(report["token_mean_kld"])) or float(report["token_mean_kld"]) < 0.0:
        errors.append(f"{path}: token_mean_kld is not a finite non-negative number")
    if kld_gap > tolerance:
        errors.append(
            f"{path}: token_mean_kld {report['token_mean_kld']!r} disagrees with its rows "
            f"({weighted_kld!r}), relative gap {kld_gap:.3e} > {tolerance:.3e}"
        )
    if top1_gap > tolerance:
        errors.append(
            f"{path}: top1_agreement {report['top1_agreement']!r} disagrees with its rows "
            f"({weighted_top1!r}), relative gap {top1_gap:.3e} > {tolerance:.3e}"
        )
    return {"token_mean_gap": kld_gap, "top1_gap": top1_gap, "positions": positions}


def load_parent_suite(suite: Path, errors: list[str]) -> dict | None:
    path = suite / "suite-manifest.json"
    if not path.is_file():
        errors.append(f"{path}: missing suite manifest")
        return None
    manifest = json.loads(path.read_text())
    index = manifest.get("context_index")
    if not isinstance(index, list) or not index:
        errors.append(f"{path}: suite manifest has no context_index")
        return None
    order: dict[int, int] = {}
    digests: dict[int, str] = {}
    for position, row in enumerate(index):
        idx = row.get("index")
        digest = row.get("token_sha256")
        if not isinstance(idx, int) or not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{path}: context_index[{position}] has no valid index/token_sha256")
            return None
        if idx in order:
            errors.append(f"{path}: duplicate context index {idx} in the suite")
            return None
        order[idx] = position
        digests[idx] = digest
    whole = sha256_bytes("".join(digests[i] for i in sorted(order, key=order.__getitem__)).encode())
    if manifest.get("suite_token_sha256") != whole:
        errors.append(
            f"{path}: suite_token_sha256 does not match its ordered context hashes"
        )
        return None
    return {
        "path": str(suite),
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "schema": manifest.get("schema"),
        "suite_token_sha256": whole,
        "contexts": len(order),
        "context_length": manifest.get("context_length"),
        "scored_positions_per_context": manifest.get("scored_positions_per_context"),
        "total_scored_positions": manifest.get("total_scored_positions"),
        "order": order,
        "digests": digests,
    }


def shard_lineage_digest(parent: dict, indices: list[int]) -> str:
    """Digest a shard view declares: its own contexts, in parent-suite order."""
    ordered = sorted(indices, key=lambda i: parent["order"][i])
    return sha256_bytes("".join(parent["digests"][i] for i in ordered).encode())


def ladder_state(positions: int) -> dict:
    reached = [(label, need) for label, need in LADDER_TARGETS if positions >= need]
    upcoming = [(label, need) for label, need in LADDER_TARGETS if positions < need]
    checkpoint, checkpoint_positions = reached[-1] if reached else (None, None)
    nxt, nxt_positions = upcoming[0] if upcoming else (None, None)
    return {
        "checkpoint": checkpoint,
        "checkpoint_positions": checkpoint_positions,
        "next": nxt,
        "next_positions": nxt_positions,
        "positions_to_next": None if nxt_positions is None else nxt_positions - positions,
        "targets": {label: need for label, need in LADDER_TARGETS},
    }


def cmd_aggregate(args: argparse.Namespace) -> int:
    started = time.time()
    errors: list[str] = []
    tolerance = float(args.tolerance)
    paths = [Path(p) for p in args.reports]
    if len(set(str(p.resolve()) for p in paths)) != len(paths):
        errors.append("the same report file was passed more than once")

    parent = load_parent_suite(Path(args.suite), errors) if args.suite else None

    reports: list[tuple[Path, dict, dict]] = []
    for path in paths:
        report = load_report(path, errors)
        if report is None:
            continue
        stats = check_rows_against_summary(path, report, tolerance, errors)
        reports.append((path, report, stats))
    if not reports:
        raise Rejected(errors or ["no usable reports"])

    # ---- identity: one candidate, one reference, one head, one comparator
    base_path, base_report, _ = reports[0]
    base_view = identity_view(base_report)
    for path, report, _ in reports[1:]:
        view = identity_view(report)
        for key, expected in base_view.items():
            if view[key] != expected:
                errors.append(
                    f"{path}: {key} differs from {base_path} "
                    f"({json.dumps(view[key])} vs {json.dumps(expected)})"
                )

    # ---- suite identity: lineage against the parent, or one shared digest
    if parent is None:
        digests = {report["suite_token_sha256"] for _, report, _ in reports}
        if len(digests) > 1:
            errors.append(
                "reports declare different suite_token_sha256 "
                f"({sorted(digests)}); pass --suite to verify shard views against a parent suite"
            )

    # ---- coverage: every context exactly once
    owner: dict[int, Path] = {}
    rows_by_index: dict[int, dict] = {}
    for path, report, _ in reports:
        seen_here: set[int] = set()
        for row in report["per_context_all"]:
            idx = int(row["index"])
            if idx in seen_here:
                errors.append(f"{path}: duplicate context index {idx} inside one report")
                continue
            seen_here.add(idx)
            if idx in owner:
                errors.append(
                    f"overlapping context index {idx}: present in both {owner[idx]} and {path}"
                )
                continue
            owner[idx] = path
            rows_by_index[idx] = row
        if parent is not None:
            unknown = sorted(i for i in seen_here if i not in parent["order"])
            if unknown:
                errors.append(
                    f"{path}: context indices absent from {parent['path']}: {unknown[:16]}"
                )
            else:
                expected = shard_lineage_digest(parent, sorted(seen_here))
                if report["suite_token_sha256"] != expected:
                    errors.append(
                        f"{path}: suite_token_sha256 {report['suite_token_sha256']} is not the "
                        f"digest of the contexts it scored in parent order ({expected})"
                    )

    # ---- shard structure: complete shards, no gaps, exactly the expected set
    shard_size = int(args.shard_size)
    covered_shards: dict[int, set[int]] = {}
    if shard_size > 0:
        for idx in owner:
            covered_shards.setdefault(idx // shard_size, set()).add(idx)
        parent_total = parent["contexts"] if parent is not None else None
        for shard in sorted(covered_shards):
            want = shard_size
            if parent_total is not None:
                want = min(shard_size, max(0, parent_total - shard * shard_size))
            got = covered_shards[shard]
            expect_indices = set(range(shard * shard_size, shard * shard_size + want))
            if got != expect_indices:
                missing = sorted(expect_indices - got)
                extra = sorted(got - expect_indices)
                errors.append(
                    f"shard {shard} is not covered exactly: {len(got)}/{want} contexts"
                    + (f", missing {missing[:8]}" if missing else "")
                    + (f", unexpected {extra[:8]}" if extra else "")
                )
        if args.shards is not None:
            expected_shards = parse_index_list(args.shards)
            if sorted(covered_shards) != expected_shards:
                errors.append(
                    f"covered shards {sorted(covered_shards)} != requested {expected_shards}"
                )
        elif sorted(covered_shards) != list(range(len(covered_shards))):
            errors.append(
                f"covered shards {sorted(covered_shards)} are not a gap-free run from shard 0; "
                "pass --shards to aggregate a deliberate subset"
            )

    if errors:
        raise Rejected(errors)

    # ---- aggregation over the union of per-context rows, in suite order
    if parent is not None:
        order_key = lambda i: parent["order"][i]  # noqa: E731
    else:
        order_key = lambda i: i  # noqa: E731
    ordered_indices = sorted(rows_by_index, key=order_key)
    rows = [rows_by_index[i] for i in ordered_indices]

    positions = sum(int(r["positions_scored"]) for r in rows)
    means = [float(r["mean_kld"]) for r in rows]
    clusters = [str(r["source_cluster"]) for r in rows]
    token_mean = math.fsum(
        m * int(r["positions_scored"]) for m, r in zip(means, rows)) / positions
    top1 = math.fsum(
        float(r["top1_agreement"]) * int(r["positions_scored"]) for r in rows) / positions

    strata: dict[str, dict] = {}
    for row in rows:
        s = strata.setdefault(str(row["stratum"]),
                              {"scored_positions": 0, "means": [], "weighted": []})
        npos = int(row["positions_scored"])
        s["scored_positions"] += npos
        s["means"].append(float(row["mean_kld"]))
        s["weighted"].append(float(row["mean_kld"]) * npos)
    strata_out = {
        name: {
            "contexts": len(s["means"]),
            "scored_positions": s["scored_positions"],
            "macro_mean_kld": statistics.fmean(s["means"]),
            "token_mean_kld": math.fsum(s["weighted"]) / s["scored_positions"],
        }
        for name, s in sorted(strata.items())
    }

    shard_rows = []
    for path, report, stats in sorted(
        reports, key=lambda t: order_key(min(int(r["index"]) for r in t[1]["per_context_all"]))
    ):
        indices = sorted(int(r["index"]) for r in report["per_context_all"])
        shard_rows.append({
            "shard": indices[0] // shard_size if shard_size > 0 else None,
            "report": str(path),
            "report_sha256": sha256_file(path),
            "suite_token_sha256": report["suite_token_sha256"],
            "reference_capture": report.get("reference"),
            "candidate_capture": report.get("candidate"),
            "contexts": int(report["contexts"]),
            "scored_positions": int(report["scored_positions"]),
            "context_index_min": indices[0],
            "context_index_max": indices[-1],
            "token_mean_kld": float(report["token_mean_kld"]),
            "context_macro_mean_kld": report.get("context_macro_mean_kld"),
            "top1_agreement": float(report["top1_agreement"]),
            "max_kld": report.get("max_kld"),
            "row_consistency": {"token_mean_relative_gap": stats["token_mean_gap"],
                                "top1_relative_gap": stats["top1_gap"]},
        })

    max_kld = [float(r["max_kld"]) for _, r, _ in reports if isinstance(r.get("max_kld"), (int, float))]
    jsd = [(float(r["mean_jsd_bits"]), int(r["contexts"]))
           for _, r, _ in reports if isinstance(r.get("mean_jsd_bits"), (int, float))]

    payload = {
        "schema": SCHEMA,
        "tool": "tools/kld_aggregate.py",
        "generated_unix": int(started),
        "candidate": args.candidate or Path(
            (base_report.get("candidate_identity") or {}).get("model_path")
            or base_report.get("candidate") or "unknown"
        ).name,
        "label": args.label,
        "reference_identity": base_report.get("reference_identity"),
        "candidate_identity": base_report.get("candidate_identity"),
        "head": base_report.get("head"),
        "head_sha256": base_report.get("head_sha256"),
        "candidate_head": base_report.get("candidate_head"),
        "candidate_head_sha256": base_report.get("candidate_head_sha256"),
        "filter": base_report.get("filter"),
        "vocab_size": base_report.get("vocab_size"),
        "hidden_size": base_report.get("hidden_size"),
        "comparator": base_report.get("comparator"),
        "suite": {
            "mode": "parent-lineage" if parent is not None else "single-suite-digest",
            "shard_size": shard_size,
            "shards": sorted(covered_shards) if shard_size > 0 else None,
            "requested_shards": parse_index_list(args.shards) if args.shards else None,
            "suite_token_sha256": None if parent is not None else base_report["suite_token_sha256"],
            "parent": None if parent is None else {
                "path": parent["path"],
                "manifest_sha256": parent["manifest_sha256"],
                "schema": parent["schema"],
                "suite_token_sha256": parent["suite_token_sha256"],
                "contexts": parent["contexts"],
                "context_length": parent["context_length"],
                "scored_positions_per_context": parent["scored_positions_per_context"],
            },
        },
        "reports": len(reports),
        "contexts": len(rows),
        "scored_positions": positions,
        "source_clusters": len(set(clusters)),
        "token_mean_kld": token_mean,
        "context_macro_mean_kld": statistics.fmean(means),
        "context_bootstrap": bootstrap(means, clusters, args.bootstrap_samples,
                                       args.bootstrap_seed),
        "bootstrap_config": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                             "unit": "source_cluster", "over": "union of per-context rows"},
        "top1_agreement": top1,
        "max_kld": max(max_kld) if max_kld else None,
        "mean_jsd_bits": (math.fsum(v * n for v, n in jsd)
                          / sum(n for _, n in jsd)) if jsd else None,
        "strata": strata_out,
        "worst_contexts": sorted(
            ({"index": int(r["index"]), "stratum": r["stratum"],
              "source_cluster": r["source_cluster"], "mean_kld": float(r["mean_kld"]),
              "top1_agreement": float(r["top1_agreement"])} for r in rows),
            key=lambda r: -r["mean_kld"],
        )[:20],
        "ladder": ladder_state(positions),
        "shards": shard_rows,
        "consistency": {
            "tolerance": tolerance,
            "max_token_mean_relative_gap": max(s["token_mean_gap"] for _, _, s in reports),
            "max_top1_relative_gap": max(s["top1_gap"] for _, _, s in reports),
            "note": ("headline numbers are recomputed from per-context rows and cross-checked "
                     "against each shard report's own summary"),
        },
        "not_aggregable": {
            "token_percentiles": ("median/p95/p99/p999 need the token-level KLD vector; see the "
                                  "per-shard reports listed above"),
        },
        "inputs": {row["report"]: row["report_sha256"] for row in shard_rows},
    }
    if args.rows:
        payload["per_context_all"] = rows
    # Everything except the wall-clock stamp, so re-running the same shard set over
    # the same reports is a digest-comparable reproduction.
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "generated_unix"}
    )
    payload["content_sha256_excludes"] = ["generated_unix", "content_sha256"]

    out = Path(args.out)
    atomic_write_json(out, payload)
    print(json.dumps({
        "out": str(out),
        "candidate": payload["candidate"],
        "reports": payload["reports"],
        "contexts": payload["contexts"],
        "scored_positions": payload["scored_positions"],
        "source_clusters": payload["source_clusters"],
        "token_mean_kld": payload["token_mean_kld"],
        "context_macro_mean_kld": payload["context_macro_mean_kld"],
        "ci95": [payload["context_bootstrap"]["ci95_low"],
                 payload["context_bootstrap"]["ci95_high"]],
        "top1_agreement": payload["top1_agreement"],
        "ladder_checkpoint": payload["ladder"]["checkpoint"],
        "content_sha256": payload["content_sha256"],
    }), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="kld_aggregate.py",
        description="Weld per-shard fidelity reports into one cumulative KLD receipt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="build a cumulative receipt for one candidate")
    a.add_argument("reports", nargs="+", help="per-shard replay reports for ONE candidate")
    a.add_argument("--out", required=True, help="receipt path (written atomically)")
    a.add_argument("--candidate", default=None,
                   help="candidate label; default is the basename of its model path")
    a.add_argument("--label", default=None, help="free-text label stored in the receipt")
    a.add_argument("--suite", default=None,
                   help="parent suite directory; enables shard-view lineage verification")
    a.add_argument("--shard-size", type=int, default=512,
                   help="contexts per shard; 0 disables shard-structure checks")
    a.add_argument("--shards", default=None,
                   help="shard indices the set must cover exactly, e.g. 0-4 or 0,1,3")
    a.add_argument("--bootstrap-samples", type=int, default=10000)
    a.add_argument("--bootstrap-seed", type=int, default=1)
    a.add_argument("--tolerance", type=float, default=1e-6,
                   help="relative tolerance when re-deriving each shard's summary")
    a.add_argument("--rows", action=argparse.BooleanOptionalAction, default=True,
                   help="store the union of per-context rows in the receipt")
    a.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
