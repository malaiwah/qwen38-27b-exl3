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
  * token-level percentiles are NOT aggregable from per-context rows, and an
    average of per-shard percentiles means nothing.  What IS aggregable is the
    fixed log-spaced histogram of every scored position's KL that
    `qwen38-fidelity-report/2` carries in `kld_tail`: counts add.  This tool
    sums them and reports cumulative p50/p95/p99/p999/p9999 as the bin each
    quantile falls in -- a bounded interval, plus a log-uniform point estimate
    inside it -- together with exact exceedance counts at the decade edges.
    `max_kld` needs no interpolation at all: it is the exact max over shards.

Every receipt records the scored-position window its shards were replayed with
(`scored_position_window`), because a run that scores only late positions of each
context is a different measurement from a full-context one even when every other
operand is identical.  Carrying that block is what bumps the receipt schema to
`qwen38-kld-ladder-cumulative/3`: no earlier field changed its name or meaning,
but `content_sha256` covers the whole payload, so a `/2` and a `/3` receipt over
the same shard reports differ in digest by construction and the schema string has
to say which shape was digested.

Fail-closed rules -- any of these aborts without writing:

  * a report that is none of `qwen38-fidelity-report/2` (full context),
    `qwen38-fidelity-report/3` (a declared scored-position window) and the
    pre-histogram `qwen38-fidelity-report/1`, or that lacks a required field, or
    whose `kld_tail` counts do not sum to the positions it says it scored;
  * a mix of `/1` with either histogram generation: `/1` has no histogram, so a
    summed tail would silently describe only part of the run;
  * an all-`/1` set, unless `--allow-legacy-no-tail` says so explicitly, in
    which case the receipt carries no cumulative tail and records why;
  * histogram-carrying reports whose bin edges differ;
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
    `--tolerance` (relative);
  * reports whose scored-position window differs: `fidelity.py replay
    --score-from N` drops the first N scored positions of every context, so a
    windowed shard and a full-context shard cover different position sets and
    their means, summed histograms, top-1 rates and maxima cannot be combined.  A
    report that predates the block counts as `score_from = 0`; a windowed report
    is a `qwen38-fidelity-report/3`, and a report whose schema and whose
    `scored_position_window.score_from` disagree about being windowed is rejected
    as internally inconsistent.

Usage:

    tools/kld_aggregate.py aggregate --out receipts/kld5-k5k6-2M.json \\
        --suite /var/tmp/work/kld5/suite --shards 0-1 \\
        /var/tmp/work/kld5/reports/shard-0000/report-k5k6.json \\
        /var/tmp/work/kld5/reports/shard-0001/report-k5k6.json
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "qwen38-kld-ladder-cumulative/3"
REPORT_SCHEMA = "qwen38-fidelity-report/2"
# Same format as `/2`, different scope: its numbers cover a declared sub-window of
# each context (`fidelity.py replay --score-from N`).  Both carry `kld_tail`, and
# both are aggregable -- but never with each other.
WINDOWED_REPORT_SCHEMA = "qwen38-fidelity-report/3"
LEGACY_REPORT_SCHEMA = "qwen38-fidelity-report/1"
TAIL_REPORT_SCHEMAS = (REPORT_SCHEMA, WINDOWED_REPORT_SCHEMA)
SCORED_WINDOW_SCHEMA = "qwen38-scored-position-window/1"

# Cumulative quantiles read off the summed histogram.  p9999 is here and not in
# the shard reports because it only becomes meaningful at ladder scale: it needs
# 10^4 positions to exist at all and 10^6 to be worth printing.
TAIL_QUANTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99),
                  ("p999", 0.999), ("p9999", 0.9999))

# Exceedance rows, as powers of ten of KL in nats.  These are counted at a bin
# edge, so unlike the quantiles they are exact.
EXCEEDANCE_LOG10 = (-4, -3, -2, -1, 0)

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
    schema = report.get("schema")
    known = (REPORT_SCHEMA, WINDOWED_REPORT_SCHEMA, LEGACY_REPORT_SCHEMA)
    if schema not in known:
        errors.append(
            f"{path}: schema {schema!r} is none of "
            + ", ".join(repr(s) for s in known)
        )
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


def scored_window(path: Path, report: dict, errors: list[str]) -> dict | None:
    """The scored-position window a report declares, defaulted for older reports.

    A report written before `--score-from` existed carries no block and scored
    every position of every context, which is `score_from = 0` -- so it may be
    welded with today's unwindowed reports and never with a windowed one.  The
    schema and the block have to agree about being windowed: a `/3` without a
    window, or a `/2` with one, is a hand-edited or half-migrated report and no
    number in it can be trusted to mean what it says.
    """
    schema = str(report.get("schema"))
    block = report.get("scored_position_window")
    if block is None:
        if schema == WINDOWED_REPORT_SCHEMA:
            errors.append(
                f"{path}: schema {schema} promises a scored-position window but the "
                "report carries no scored_position_window block"
            )
            return None
        return {"score_from": 0, "declared": False, "windowed": False,
                "positions_per_context": None, "source_schema": schema}
    if not isinstance(block, dict):
        errors.append(f"{path}: scored_position_window is not an object")
        return None
    score_from = block.get("score_from")
    if not isinstance(score_from, int) or isinstance(score_from, bool) or score_from < 0:
        errors.append(
            f"{path}: scored_position_window.score_from {score_from!r} is not a "
            "non-negative integer"
        )
        return None
    if (score_from > 0) != (schema == WINDOWED_REPORT_SCHEMA):
        errors.append(
            f"{path}: scored_position_window.score_from is {score_from} but the report "
            f"is a {schema}; a windowed report must be a {WINDOWED_REPORT_SCHEMA} and a "
            f"{WINDOWED_REPORT_SCHEMA} must declare score_from > 0"
        )
        return None
    per_context = block.get("positions_per_context")
    if per_context is not None and (not isinstance(per_context, int)
                                    or isinstance(per_context, bool) or per_context <= 0):
        errors.append(
            f"{path}: scored_position_window.positions_per_context {per_context!r} is "
            "neither null nor a positive integer"
        )
        return None
    return {"score_from": score_from, "declared": True, "windowed": score_from > 0,
            "positions_per_context": per_context, "source_schema": schema,
            "policy": block.get("policy"),
            "min_left_context_tokens": block.get("min_left_context_tokens"),
            "dropped_positions_per_context": block.get("dropped_positions_per_context")}


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


def validate_tail(path: Path, report: dict, errors: list[str]) -> dict | None:
    """Structural check of a histogram-carrying report's `kld_tail` block.

    A histogram that does not account for exactly the positions its own report
    claims to have scored is worse than no histogram, because it would be summed
    into a cumulative tail without anyone noticing.
    """
    tail = report.get("kld_tail")
    if not isinstance(tail, dict):
        errors.append(f"{path}: {report.get('schema')} report has no kld_tail object")
        return None
    edges = tail.get("bin_edges")
    counts = tail.get("counts")
    if (not isinstance(edges, list) or len(edges) < 2
            or not all(isinstance(e, (int, float)) for e in edges)):
        errors.append(f"{path}: kld_tail.bin_edges is not a list of at least two numbers")
        return None
    if any(b <= a for a, b in zip(edges, edges[1:])):
        errors.append(f"{path}: kld_tail.bin_edges is not strictly ascending")
        return None
    if not isinstance(counts, list) or len(counts) != len(edges) + 1:
        got = len(counts) if isinstance(counts, list) else "no"
        errors.append(
            f"{path}: kld_tail.counts has {got} entries; {len(edges)} edges need "
            f"{len(edges) + 1} buckets (underflow, one per bin, overflow)"
        )
        return None
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in counts):
        errors.append(f"{path}: kld_tail.counts holds a negative or non-integer count")
        return None
    total = sum(counts)
    if tail.get("total_count") != total:
        errors.append(
            f"{path}: kld_tail.total_count {tail.get('total_count')!r} != {total} summed counts"
        )
    if total != int(report["scored_positions"]):
        errors.append(
            f"{path}: kld_tail counts {total} positions but the report scored "
            f"{report['scored_positions']}"
        )
    top = tail.get("max_kld")
    if not isinstance(top, (int, float)) or not math.isfinite(float(top)):
        errors.append(f"{path}: kld_tail.max_kld is not a finite number")
    elif isinstance(report.get("max_kld"), (int, float)) and float(report["max_kld"]) != float(top):
        errors.append(
            f"{path}: kld_tail.max_kld {top!r} != the report's own max_kld "
            f"{report['max_kld']!r}; one of them is not the exact maximum"
        )
    exact = tail.get("shard_exact_quantiles")
    if not isinstance(exact, dict) or any(k not in exact for k in ("p50", "p95", "p99", "p999")):
        errors.append(f"{path}: kld_tail.shard_exact_quantiles lacks p50/p95/p99/p999")
    return tail


def bucket_bounds(edges: list[float], bucket: int) -> tuple[float, float]:
    """The half-open KL interval, in nats, that a bucket index stands for."""
    lo = 0.0 if bucket == 0 else float(edges[bucket - 1])
    hi = float(edges[bucket]) if bucket < len(edges) else math.inf
    return lo, hi


def histogram_quantile(edges: list[float], counts: list[int], cumulative: list[int],
                       total: int, q: float, global_max: float) -> dict:
    """Bound one cumulative quantile by the histogram bin it falls in.

    Linear interpolation between order statistics -- what `torch.quantile`, and
    therefore every per-shard report, uses -- puts the exact q-th quantile
    between the k-th and (k+1)-th smallest scored positions, where
    `k = floor((n - 1) q) + 1`.  Bounding both of those observations bounds the
    exact value, and each bound is a bin edge, so `lower` and `upper` are a
    guarantee rather than an estimate.  `estimate` interpolates log-uniformly
    inside the lower observation's bin, which is the assumption the log-spaced
    binning already encodes; it is never outside `[lower, upper]`.

    The exact maximum tightens the upper bound: no quantile can exceed it, and
    for the overflow bucket it is the only finite bound there is.
    """
    h = (total - 1) * q
    k_lo = math.floor(h) + 1
    k_hi = k_lo if h == math.floor(h) else min(k_lo + 1, total)
    b_lo = bisect.bisect_left(cumulative, k_lo)
    b_hi = bisect.bisect_left(cumulative, k_hi)
    lo, hi = bucket_bounds(edges, b_lo)
    upper = bucket_bounds(edges, b_hi)[1]
    if not math.isfinite(upper) or upper > global_max:
        upper = global_max
    below = cumulative[b_lo - 1] if b_lo else 0
    inside = counts[b_lo]
    if b_lo == 0:
        kind, estimate = "underflow", None
    elif b_lo == len(edges):
        kind, estimate = "overflow", None
    else:
        kind = "log"
        frac = min(1.0, max(0.0, (h + 1.0 - below - 0.5) / inside))
        # Clamped into the guaranteed interval: interpolation inside the last
        # populated bin can otherwise land above the exact maximum, and an
        # estimate that contradicts a bound the same receipt states is worthless.
        estimate = min(max(lo * (hi / lo) ** frac, lo), upper)
    return {
        "quantile": q,
        "lower": lo,
        "upper": upper,
        "estimate": estimate,
        "bin": b_lo,
        "bin_kind": kind,
        "bin_lower": lo,
        "bin_upper": None if math.isinf(hi) else hi,
        "order_statistic": k_lo,
        "positions_below_bin": below,
        "positions_in_bin": inside,
        "relative_width": None if upper <= 0.0 else (upper - lo) / upper,
    }


def exceedance_row(edges: list[float], counts: list[int], total: int,
                   log10_threshold: int) -> dict:
    """Exact number of positions at or above a bin edge -- no interpolation.

    `threshold` is the edge actually used: the first one at or above
    `10 ** log10_threshold`, which is that power of ten itself whenever it is a
    bin edge, as it is for the published binning.
    """
    requested = 10.0 ** log10_threshold
    index = bisect.bisect_left(edges, requested)
    if index == len(edges):
        raise SystemExit(
            f"exceedance threshold {requested!r} is above the histogram's last bin edge"
        )
    above = sum(counts[index + 1:])
    return {"requested": requested, "threshold": float(edges[index]),
            "exact": float(edges[index]) == requested,
            "positions": above, "fraction": above / total}


def cumulative_tail(tails: list[dict]) -> dict:
    """Sum the shard histograms and read the cumulative tail off the sum.

    Callers must have checked that every histogram shares one set of bin edges;
    counts from different binnings are not summable.
    """
    edges = [float(e) for e in tails[0]["bin_edges"]]
    counts = [0] * (len(edges) + 1)
    for tail in tails:
        for i, c in enumerate(tail["counts"]):
            counts[i] += int(c)
    total = sum(counts)
    cumulative, running = [], 0
    for c in counts:
        running += c
        cumulative.append(running)
    global_max = max(float(t["max_kld"]) for t in tails)
    mins = [float(t["min_kld"]) for t in tails
            if isinstance(t.get("min_kld"), (int, float))]
    return {
        "source": "sum of the per-shard kld_tail histograms",
        "units": "nats",
        "shards": len(tails),
        "binning": tails[0].get("binning"),
        "bin_edges": edges,
        "counts": counts,
        "total_count": total,
        "underflow_count": counts[0],
        "overflow_count": counts[-1],
        "max_kld_exact": global_max,
        "min_kld_exact": min(mins) if len(mins) == len(tails) else None,
        "quantiles": {name: histogram_quantile(edges, counts, cumulative, total, q, global_max)
                      for name, q in TAIL_QUANTILES},
        "exceedance": [exceedance_row(edges, counts, total, d) for d in EXCEEDANCE_LOG10],
        "method": ("counts add across shards, percentiles do not: each quantile is reported "
                   "as the [lower, upper] bin interval that provably contains it, plus a "
                   "log-uniform point estimate inside that interval. max_kld_exact and the "
                   "exceedance counts are exact, not interpolated."),
    }


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
    tails: list[tuple[Path, dict]] = []
    windows: list[tuple[Path, dict]] = []
    for path in paths:
        report = load_report(path, errors)
        if report is None:
            continue
        stats = check_rows_against_summary(path, report, tolerance, errors)
        reports.append((path, report, stats))
        window = scored_window(path, report, errors)
        if window is not None:
            windows.append((path, window))
        if report["schema"] in TAIL_REPORT_SCHEMAS:
            tail = validate_tail(path, report, errors)
            if tail is not None:
                tails.append((path, tail))
    if not reports:
        raise Rejected(errors or ["no usable reports"])

    # ---- one report generation, so the summed tail covers the whole run
    schemas = sorted({str(report["schema"]) for _, report, _ in reports})
    if LEGACY_REPORT_SCHEMA in schemas and len(schemas) > 1:
        errors.append(
            f"mixed report schemas {schemas}: {LEGACY_REPORT_SCHEMA} predates the kld_tail "
            "histogram, so a summed tail would describe only the shards that carry one. "
            f"Re-run `fidelity.py replay` on the {LEGACY_REPORT_SCHEMA} shards, or aggregate "
            "the two generations into separate receipts"
        )
    elif schemas == [LEGACY_REPORT_SCHEMA] and not args.allow_legacy_no_tail:
        errors.append(
            f"all {len(reports)} reports are {LEGACY_REPORT_SCHEMA}, which carries no kld_tail "
            "histogram, so no cumulative percentile can be derived from them. Re-run "
            f"`fidelity.py replay` to produce {REPORT_SCHEMA}, or pass --allow-legacy-no-tail "
            "to build a receipt that states the cumulative tail is unavailable"
        )
    for path, tail in tails[1:]:
        if [float(e) for e in tail["bin_edges"]] != [float(e) for e in tails[0][1]["bin_edges"]]:
            errors.append(
                f"{path}: kld_tail bin edges differ from {tails[0][0]}; histograms with "
                "different binning cannot be summed"
            )

    # ---- one scored-position window, or the numbers are not the same measurement
    if windows:
        base_window_path, base_window = windows[0]
        for path, window in windows[1:]:
            if window["score_from"] != base_window["score_from"]:
                errors.append(
                    f"{path}: scored-position window differs from {base_window_path} "
                    f"(--score-from {window['score_from']} vs {base_window['score_from']}"
                    + ("; a report without a scored_position_window block scored every "
                       "position, which is --score-from 0"
                       if not (window["declared"] and base_window["declared"]) else "")
                    + "): a windowed shard and a full-context shard score different "
                    "position sets, so their means, summed histograms, top-1 rates and "
                    "maxima are not the same measurement. Aggregate each window into "
                    "its own receipt"
                )

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

    tail_summary = cumulative_tail([t for _, t in tails]) if tails else None

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
            "p999_kld": report.get("p999_kld"),
            "row_consistency": {"token_mean_relative_gap": stats["token_mean_gap"],
                                "top1_relative_gap": stats["top1_gap"]},
        })

    max_kld = [float(r["max_kld"]) for _, r, _ in reports if isinstance(r.get("max_kld"), (int, float))]
    jsd = [(float(r["mean_jsd_bits"]), int(r["contexts"]))
           for _, r, _ in reports if isinstance(r.get("mean_jsd_bits"), (int, float))]
    if tail_summary is None:
        not_aggregable = {
            "token_percentiles": (
                f"these reports are {LEGACY_REPORT_SCHEMA}, which carries no kld_tail "
                "histogram: median/p95/p99/p999 need the token-level KLD vector, so only "
                "each shard's own percentiles exist; see the per-shard reports listed above"),
        }
    else:
        not_aggregable = {
            "token_percentiles_exact": (
                "an exact cumulative median/p95/p99/p999 would need the token-level KLD "
                "vector, which no shard report carries. `tail.quantiles` bounds each one by "
                "the histogram bin it falls in, which is exact to the bin width; each "
                "shard's own exact percentiles are in its report"),
            "jsd_percentiles": (
                "shard reports carry only the position-weighted mean JSD, so its "
                "distribution cannot be recovered here"),
        }

    # Every shard shares one window (enforced above), so the receipt states it once.
    score_from = windows[0][1]["score_from"] if windows else 0
    declared_per_context = {w["positions_per_context"] for _, w in windows
                            if w["positions_per_context"] is not None}
    scored_window_payload = {
        "schema": SCORED_WINDOW_SCHEMA,
        "score_from": score_from,
        "windowed": score_from > 0,
        "policy": (
            f"every shard was replayed with `fidelity.py replay --score-from {score_from}`: "
            f"the first {score_from} scored positions of every context were dropped before "
            "any statistic was computed, so every number in this receipt -- scored "
            "positions, token mean, macro mean, bootstrap, top-1 agreement, summed "
            "histogram, cumulative quantiles, exceedances and exact maximum -- covers only "
            "the retained positions and is not comparable with a full-context receipt"
            if score_from else
            "every shard scored every position of every context; nothing is windowed"
        ),
        "positions_per_context": (next(iter(declared_per_context))
                                  if len(declared_per_context) == 1 else None),
        "min_left_context_tokens": score_from + 1,
        "declared_by_all_reports": bool(windows) and all(w["declared"] for _, w in windows),
        "report_schemas": schemas,
    }

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
        "scored_position_window": scored_window_payload,
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
        "tail": tail_summary,
        "ladder": ladder_state(positions),
        "shards": shard_rows,
        "consistency": {
            "tolerance": tolerance,
            "max_token_mean_relative_gap": max(s["token_mean_gap"] for _, _, s in reports),
            "max_top1_relative_gap": max(s["top1_gap"] for _, _, s in reports),
            "note": ("headline numbers are recomputed from per-context rows and cross-checked "
                     "against each shard report's own summary"),
        },
        "not_aggregable": not_aggregable,
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
    summary = {
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
        "cumulative_p999": None if tail_summary is None else [
            tail_summary["quantiles"]["p999"]["lower"],
            tail_summary["quantiles"]["p999"]["upper"]],
        "max_kld": payload["max_kld"],
        "ladder_checkpoint": payload["ladder"]["checkpoint"],
        "content_sha256": payload["content_sha256"],
    }
    if score_from:
        summary["score_from"] = score_from
        summary["positions_per_context"] = scored_window_payload["positions_per_context"]
    print(json.dumps(summary), flush=True)
    return 0

# ------------------------------------------------------------------- paired
# `receipts/kld5-10M-paired.json` was originally built by hand.  This subcommand
# is that receipt's tool, written afterwards and validated against it: rebuilding
# the published five comparisons from the same five cumulative receipts reproduces
# every field bit-for-bit, including `content_sha256`.  So "the same way the
# published paired receipt was built" is now something a reader can re-run.
PAIRED_SCHEMA = "qwen38-kld-ladder-paired/1"

# What two cumulative receipts must agree on before their per-context means may be
# subtracted.  Anything here differing means the two numbers were not measured
# against the same reference through the same head on the same positions, and the
# difference would be an artefact rather than a quantization comparison.
#
# `schema` is deliberately NOT in this list.  `/2` and `/3` differ only by whether
# the receipt carries a `scored_position_window` block, and no earlier field
# changed name or meaning, so refusing to pair them would refuse to compare a
# receipt built today against one built last week for no measurement reason.  What
# actually has to match is the window itself, which `paired_window` derives for
# both generations and `pair_one` compares.
PAIRED_SHARED_FIELDS = ("head", "head_sha256", "candidate_head",
                        "candidate_head_sha256", "filter", "vocab_size",
                        "hidden_size", "comparator", "reference_identity",
                        "contexts", "scored_positions", "source_clusters")

CUMULATIVE_SCHEMAS = ("qwen38-kld-ladder-cumulative/2", SCHEMA)


def load_cumulative(label: str, path: Path, errors: list[str]) -> dict | None:
    """One candidate's cumulative receipt, with the fields `paired` needs present."""
    if not path.is_file():
        errors.append(f"{label}: missing receipt {path}")
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: unreadable receipt {path} ({exc})")
        return None
    if payload.get("schema") not in CUMULATIVE_SCHEMAS:
        errors.append(f"{label}: {path} schema {payload.get('schema')!r} is none of "
                      + ", ".join(repr(s) for s in CUMULATIVE_SCHEMAS))
        return None
    rows = payload.get("per_context_all")
    if not isinstance(rows, list) or not rows:
        errors.append(f"{label}: {path} carries no per_context_all rows, so nothing "
                      "can be paired (re-aggregate with --rows)")
        return None
    for row in rows:
        if not isinstance(row, dict) or any(f not in row for f in ("index", "mean_kld",
                                                                   "source_cluster")):
            errors.append(f"{label}: {path} per_context_all row lacks index/mean_kld/"
                          "source_cluster")
            return None
    if len({row["index"] for row in rows}) != len(rows):
        errors.append(f"{label}: {path} repeats a context index")
        return None
    return payload


def paired_window(payload: dict) -> tuple[int, int | None]:
    """`(score_from, positions_per_context)` for either receipt generation.

    A `/2` receipt predates the block, and a receipt that predates the block
    scored every position of every context -- that is what `--score-from 0` means
    -- so it is read as `score_from = 0` over the parent suite's positions per
    context.  Comparing this pair, rather than the receipt schema string, is what
    lets a receipt built by today's tool be paired with a published one while
    still refusing to subtract a windowed run from a full-context run.

    `positions_per_context` falls back to the parent suite whenever the block does
    not carry one, not only when the block is absent entirely.  A `/3` receipt
    aggregated from `qwen38-fidelity-report/1` shards HAS the block -- the tool
    always emits it -- but leaves that field null, because null there means "no
    shard declared it", never "the window is unknown": `windowed` is derived from
    `score_from` alone, and every legacy report scored every position.  Returning
    the null would make a published receipt unpairable with one built today for a
    reason that is purely generational, which is the case this function exists to
    permit.
    """
    block = payload.get("scored_position_window")
    parent = (payload.get("suite") or {}).get("parent") or {}
    fallback = parent.get("scored_positions_per_context")
    if isinstance(block, dict):
        declared = block.get("positions_per_context")
        return (int(block.get("score_from") or 0),
                declared if declared is not None else fallback)
    return 0, fallback


def pair_one(a_label: str, a: dict, b_label: str, b: dict, samples: int, seed: int,
             errors: list[str]) -> dict | None:
    """One A-minus-B comparison over the contexts both receipts scored.

    The unit is a context, not a position: per-context means are already equal-
    weight (every context scores 2,047 positions), and a context is the level at
    which the two candidates saw literally the same tokens.
    """
    for field in PAIRED_SHARED_FIELDS:
        if a.get(field) != b.get(field):
            errors.append(f"{a_label} vs {b_label}: disagree on {field}")
            return None
    window_a, window_b = paired_window(a), paired_window(b)
    if window_a != window_b:
        errors.append(f"{a_label} vs {b_label}: different scored-position windows "
                      f"{window_a} vs {window_b}; a windowed run and a full-context "
                      "run cover different positions and cannot be subtracted")
        return None
    rows_a = {row["index"]: row for row in a["per_context_all"]}
    rows_b = {row["index"]: row for row in b["per_context_all"]}
    if set(rows_a) != set(rows_b):
        only_a = sorted(set(rows_a) - set(rows_b))[:8]
        only_b = sorted(set(rows_b) - set(rows_a))[:8]
        errors.append(f"{a_label} vs {b_label}: different context sets "
                      f"(a_only={only_a}, b_only={only_b})")
        return None
    shared = sorted(rows_a)
    mismatched = [i for i in shared
                  if rows_a[i].get("source_cluster") != rows_b[i].get("source_cluster")]
    if mismatched:
        errors.append(f"{a_label} vs {b_label}: source cluster differs at contexts "
                      f"{mismatched[:8]}")
        return None
    diffs = [rows_a[i]["mean_kld"] - rows_b[i]["mean_kld"] for i in shared]
    clusters = [rows_a[i]["source_cluster"] for i in shared]
    interval = bootstrap(diffs, clusters, samples, seed)
    # `a_wins` counts strict improvements and `b_wins` is the remainder, so an exact
    # TIE is counted as a b-win.  That is the convention `fidelity.py paired` has
    # always used and it is kept here so the published paired receipt stays
    # reproducible -- but it makes a zero-difference control read "a_wins 0, b_wins
    # 512" when the truth is 512 ties, which has already misled one reader.  The
    # field count is therefore left alone and the tie count is reported on stderr
    # instead, so a run that CAN tie says so out loud without changing the payload.
    wins_a = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0.0)
    if ties:
        print(f"note: {a_label} vs {b_label}: {ties} of {len(diffs)} contexts are exact "
              f"ties, and this schema counts a tie as a b-win, so b_wins "
              f"({len(diffs) - wins_a}) is not a count of losses", file=sys.stderr)
    return {
        "a": a_label,
        "b": b_label,
        "contexts": len(shared),
        "difference_a_minus_b": statistics.fmean(diffs),
        "ci95_low": interval["ci95_low"],
        "ci95_high": interval["ci95_high"],
        "a_wins": wins_a,
        "b_wins": len(shared) - wins_a,
        "clusters": interval["clusters"],
    }


def cmd_paired(args: argparse.Namespace) -> int:
    errors: list[str] = []
    receipts: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for spec in args.input:
        label, sep, raw = spec.partition("=")
        if not sep or not label or not raw:
            raise SystemExit(f"--input wants LABEL=path, got {spec!r}")
        if label in receipts:
            raise SystemExit(f"--input {label} given twice")
        path = Path(raw)
        payload = load_cumulative(label, path, errors)
        if payload is not None:
            receipts[label] = payload
            paths[label] = path
    if errors:
        raise Rejected(errors)

    comparisons: dict[str, dict] = {}
    for spec in args.compare:
        a_label, sep, b_label = spec.partition(":")
        if not sep or a_label not in receipts or b_label not in receipts:
            raise SystemExit(f"--compare wants A:B naming two --input labels, got {spec!r}")
        key = f"{a_label}_vs_{b_label}"
        if key in comparisons:
            raise SystemExit(f"--compare {spec} given twice")
        row = pair_one(a_label, receipts[a_label], b_label, receipts[b_label],
                       args.bootstrap_samples, args.bootstrap_seed, errors)
        if row is not None:
            comparisons[key] = row
    if errors:
        raise Rejected(errors)
    if not comparisons:
        raise SystemExit("no comparisons requested: pass --compare A:B")

    reference = receipts[next(iter(receipts))]
    parent = (reference.get("suite") or {}).get("parent") or {}
    for label, payload in receipts.items():
        other = (payload.get("suite") or {}).get("parent") or {}
        if other.get("suite_token_sha256") != parent.get("suite_token_sha256"):
            errors.append(f"{label}: parent suite token differs from "
                          f"{next(iter(receipts))}")
    if errors:
        raise Rejected(errors)

    # Name the in-repo copy of the suite manifest only when it really is the byte
    # sequence the receipts were measured against.
    manifest_name = parent.get("path")
    repo_manifest = Path(__file__).resolve().parent.parent / "receipts" / "kld5-suite-manifest.json"
    if repo_manifest.is_file() and sha256_file(repo_manifest) == parent.get("manifest_sha256"):
        manifest_name = "receipts/kld5-suite-manifest.json"

    clusters = max(row["clusters"] for row in comparisons.values())
    checkpoint = (reference.get("ladder") or {}).get("checkpoint")
    payload = {
        "schema": PAIRED_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": args.scope or (
            f"paired per-context comparison at the {checkpoint} ladder checkpoint "
            "of the v5 held-out suite"),
        "suite": {
            "manifest": manifest_name,
            "manifest_sha256": parent.get("manifest_sha256"),
            "suite_token_sha256": parent.get("suite_token_sha256"),
            "contexts": reference.get("contexts"),
            "context_length": parent.get("context_length"),
            "scored_positions": reference.get("scored_positions"),
            "source_clusters": reference.get("source_clusters"),
        },
        "inputs": {
            label: {
                "path": str(paths[label]),
                "sha256": sha256_file(paths[label]),
                "content_sha256": receipts[label].get("content_sha256"),
            }
            for label in receipts
        },
        "method": {
            "unit": f"one {parent.get('context_length', 0):,}-token context, "
                    f"{parent.get('scored_positions_per_context', 0):,} scored positions",
            "statistic": "mean over contexts of (candidate A mean KLD - candidate B mean KLD)",
            "interval": f"source-cluster bootstrap, {args.bootstrap_samples:,} resamples, "
                        f"seed {args.bootstrap_seed}, {clusters} clusters resampled "
                        "with replacement",
            "operands": "both candidates replayed through the same shared BF16 LM head; "
                        "body-only, no candidate head quantization counted",
        },
        "comparisons": comparisons,
    }
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "generated_utc"}
    )
    atomic_write_json(Path(args.out), payload)
    print(json.dumps({"out": args.out, "comparisons": {
        k: {"difference_a_minus_b": v["difference_a_minus_b"],
            "ci95": [v["ci95_low"], v["ci95_high"]],
            "a_wins": v["a_wins"], "b_wins": v["b_wins"]}
        for k, v in comparisons.items()},
        "content_sha256": payload["content_sha256"]}), flush=True)
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
    a.add_argument("--allow-legacy-no-tail", action="store_true",
                   help=f"accept a set that is entirely {LEGACY_REPORT_SCHEMA} (the "
                        "pre-histogram harness) and write a receipt with no cumulative tail")
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

    q = sub.add_parser("paired", help="paired per-context comparison of cumulative receipts")
    q.add_argument("--input", action="append", required=True, metavar="LABEL=PATH",
                   help="a cumulative receipt to compare, named; repeatable")
    q.add_argument("--compare", action="append", required=True, metavar="A:B",
                   help="one comparison between two --input labels; repeatable")
    q.add_argument("--out", required=True, help="receipt path (written atomically)")
    q.add_argument("--scope", default=None,
                   help="free-text scope; default names the inputs' ladder checkpoint")
    q.add_argument("--bootstrap-samples", type=int, default=10000)
    q.add_argument("--bootstrap-seed", type=int, default=1)
    q.set_defaults(func=cmd_paired)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
