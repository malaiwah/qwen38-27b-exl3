#!/usr/bin/env python3
"""Wave 5 immutable full-vocabulary replay, access, and promotion gate.

The promotable path projects cryptographically validated hidden-state captures
through the pinned shared BF16 head itself. The normalized-probability helper is
test-only and can never emit a promotable report.
"""
from __future__ import annotations

import argparse
import importlib.util
import fcntl
import hashlib
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

SCHEMA_REPORT = "qwen38-wave5-fidelity-replay/2"
SCHEMA_SYNTHETIC = "qwen38-wave5-synthetic-metrics/1"
SCHEMA_ROWS = "qwen38-wave5-fidelity-rows/2"
SCHEMA_COMPARISON = "qwen38-wave5-paired-comparison/2"
SCHEMA_ACCESS = "qwen38-wave5-split-access/1"
SCHEMA_FREEZE = "qwen38-wave5-candidate-freeze/2"
SCHEMA_TEST_OPEN = "qwen38-wave5-test-open/2"
SCHEMA_RUNTIME = "qwen38-wave5-runtime-profile/2"
SCHEMA_FRONTIER = "qwen38-wave5-exact-byte-frontier/2"
SCHEMA_SELECTION = "qwen38-wave5-selection-decision/2"
SCHEMA_BYTE_MANIFEST = "qwen38-wave5-checkpoint-byte-manifest/1"
METRIC_NAMES = ("mean_kld", "p99_kld", "cvar1_kld", "ear", "top1_agreement")
BOOTSTRAP_NAMES = METRIC_NAMES
REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_PREREG = REPO_ROOT / "receipts/wave5/fidelity-prereg.json"
CANONICAL_CONTRACT = REPO_ROOT / "tools/research/wave5/fidelity_contract.json"


class GateError(RuntimeError):
    """A fail-closed contract violation."""

def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{path} must contain a JSON object")
    return value
def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def content_sha256(value: dict[str, Any], field: str = "content_sha256") -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _validated_content_identity(value: dict[str, Any], label: str,
                                field: str = "content_sha256") -> str:
    recorded = value.get(field)
    actual = content_sha256(value, field)
    if not _is_sha256(recorded) or recorded != actual:
        raise GateError(f"{label} canonical content identity mismatch")
    return actual


def _load_typed_receipt(pointer: dict[str, Any], schema: str,
                        label: str) -> tuple[dict[str, Any], str]:
    path = Path(_required(pointer, "path"))
    digest = sha256_file(path)
    if digest != _required(pointer, "sha256"):
        raise GateError(f"{label} file digest mismatch: {path}")
    receipt = load_json(path)
    if receipt.get("schema") != schema:
        raise GateError(f"{label} has wrong schema: {path}")
    _validated_content_identity(receipt, label)
    return receipt, digest


def trusted_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the canonical preregistration and recompute every R29 trust root."""
    prereg = load_json(CANONICAL_PREREG)
    contract = load_json(CANONICAL_CONTRACT)
    if prereg.get("contract_sha256") != sha256_file(CANONICAL_CONTRACT):
        raise GateError("canonical contract digest differs from the preregistration")
    if prereg.get("gate_sha256") != sha256_file(Path(__file__)):
        raise GateError("running gate digest differs from the preregistration")
    bindings = prereg["r29_bindings"]
    for manifest_name, file_key, content_key in (
            ("split_manifest_path", "split_manifest_file_sha256",
             "split_manifest_content_sha256"),
            ("data_manifest_path", "data_manifest_file_sha256",
             "data_manifest_content_sha256")):
        path = REPO_ROOT / bindings[manifest_name]
        if sha256_file(path) != bindings[file_key]:
            raise GateError(f"R29 final approved file pin mismatch: {path}")
        manifest = load_json(path)
        actual_content = _validated_content_identity(manifest, f"R29 {manifest_name}")
        if actual_content != bindings[content_key]:
            raise GateError(f"R29 final approved canonical-content pin mismatch: {path}")
    if bindings != contract["r29_final_approved_bindings"]:
        raise GateError("contract and preregistration disagree on final approved R29 pins")
    split_contract = contract["split_identity_contract"]
    if (split_contract["manifest_path"] != bindings["split_manifest_path"]
            or split_contract["manifest_file_sha256"]
            != bindings["split_manifest_file_sha256"]
            or split_contract["manifest_content_sha256"]
            != bindings["split_manifest_content_sha256"]):
        raise GateError("split contract does not use the one final approved R29 manifest")
    split_path = REPO_ROOT / bindings["split_manifest_path"]
    for label, registered in prereg["split_registry"].items():
        expected = split_contract["labels"][label]
        if (registered.get("manifest_path") != bindings["split_manifest_path"]
                or registered.get("manifest_sha256")
                != bindings["split_manifest_file_sha256"]
                or registered.get("selector") != expected["selector"]
                or registered.get("selection_sha256") != expected["selection_sha256"]
                or selection_sha256(split_path, label) != expected["selection_sha256"]):
            raise GateError(f"{label} split identity does not match the combined R29 manifest")
    return prereg, contract


def canonical_state_path(prereg: dict[str, Any], name: str) -> Path:
    value = prereg["canonical_paths"][name]
    path = (REPO_ROOT / value).resolve()
    if REPO_ROOT.resolve() not in path.parents:
        raise GateError(f"canonical {name} escapes the repository")
    return path




def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise GateError(f"refusing to replace immutable file {path}") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)




def _expand_metadata(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 0:
        return np.repeat(value.reshape(1), rows)
    value = value.reshape(-1)
    if value.size == 1:
        return np.repeat(value, rows)
    if value.size != rows:
        raise GateError(f"{name} has {value.size} entries for {rows} probability rows")
    return value


def validate_probabilities(probabilities: np.ndarray, expected_vocab: int | None,
                           tolerance: float = 2e-5) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise GateError("probabilities must have shape [scored_positions, full_vocabulary]")
    if expected_vocab is not None and probabilities.shape[1] != expected_vocab:
        raise GateError(f"vocabulary axis is {probabilities.shape[1]}, expected full vocabulary "
                        f"{expected_vocab}; truncated/top-k input is forbidden")
    if probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise GateError("probability matrix is empty or has no meaningful vocabulary axis")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise GateError("probabilities must be finite and non-negative")
    sums = probabilities.sum(axis=1, dtype=np.float64)
    if np.any(np.abs(sums - 1.0) > tolerance):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise GateError(f"probability rows are not normalized (maximum error {worst:.3g})")
    return probabilities


def probability_metrics(reference: np.ndarray, candidate: np.ndarray,
                        expected_vocab: int | None = None) -> dict[str, np.ndarray]:
    """Return exact full-vector forward KL, JSD, overlap/EAR, and top-1.

    ``reference`` is p and ``candidate`` is q. This normalized-vector entrypoint
    exists for deterministic metric tests and already-projected full-vocabulary
    shards. Production hidden-state replay remains the method-of-record two-pass
    implementation in ``tools/fidelity.py``; no local-softmax or top-k input is
    accepted here.
    """
    p = validate_probabilities(reference, expected_vocab)
    q = validate_probabilities(candidate, expected_vocab)
    if p.shape != q.shape:
        raise GateError(f"reference/candidate shapes differ: {p.shape} versus {q.shape}")
    p64 = p.astype(np.float64, copy=False)
    q64 = q.astype(np.float64, copy=False)
    p64 = p64 / p64.sum(axis=1, dtype=np.float64)[:, None]
    q64 = q64 / q64.sum(axis=1, dtype=np.float64)[:, None]
    positive_p = p64 > 0
    positive_q = q64 > 0
    impossible = positive_p & (q64 == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(positive_p, np.log(p64), 0.0)
        log_q = np.where(positive_q, np.log(q64), 0.0)
        terms = np.where(positive_p, p64 * (log_p - log_q), 0.0)
    kld = terms.sum(axis=1, dtype=np.float64)
    finite_negative = np.isfinite(kld) & (kld < 0)
    if np.any(kld[finite_negative] < -1e-12):
        raise GateError("computed KL is materially negative; probability input is invalid")
    kld[finite_negative] = 0.0
    kld[np.any(impossible, axis=1)] = np.inf
    midpoint = 0.5 * (p64 + q64)
    log_midpoint = np.log(np.maximum(midpoint, 1e-300))
    jsd = (0.5 * np.where(positive_p, p64 * (log_p - log_midpoint), 0.0)
           + 0.5 * np.where(positive_q, q64 * (log_q - log_midpoint), 0.0)
           ).sum(axis=1, dtype=np.float64) / math.log(2.0)
    ear = np.minimum(p64, q64).sum(axis=1, dtype=np.float64)
    top1 = (np.argmax(p, axis=1) == np.argmax(q, axis=1)).astype(np.float64)
    return {"kld": kld, "jsd_bits": jsd, "ear": ear, "top1": top1}


def cvar_upper(values: np.ndarray, fraction: float = 0.01) -> float:
    """Finite-sample upper-tail expected shortfall with fractional boundary weight."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not (0 < fraction <= 1):
        raise GateError("CVaR needs non-empty values and a tail fraction in (0, 1]")
    ordered = np.sort(values)[::-1]
    mass = values.size * fraction
    whole = int(math.floor(mass))
    residual = mass - whole
    total = float(ordered[:whole].sum(dtype=np.float64))
    if residual > 0:
        total += residual * float(ordered[whole])
    return total / mass


def summarize_metrics(kld: np.ndarray, ear: np.ndarray, top1: np.ndarray) -> dict[str, float]:
    if not (len(kld) == len(ear) == len(top1) and len(kld) > 0):
        raise GateError("metric rows must be non-empty and aligned")
    return {
        "mean_kld": float(np.mean(kld, dtype=np.float64)),
        "p99_kld": float(np.quantile(kld, 0.99, method="linear")),
        "cvar1_kld": cvar_upper(kld, 0.01),
        "ear": float(np.mean(ear, dtype=np.float64)),
        "top1_agreement": float(np.mean(top1, dtype=np.float64)),
    }


def _cluster_map(cluster_ids: np.ndarray, context_ids: np.ndarray) -> dict[str, np.ndarray]:
    if len(cluster_ids) != len(context_ids):
        raise GateError("source-cluster/context IDs are not aligned")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        grouped[str(cluster)].append(index)
    if not grouped:
        raise GateError("no source clusters")
    return {cluster: np.asarray(indices, dtype=np.int64)
            for cluster, indices in grouped.items()}


def hierarchical_cluster_draws(cluster_ids: np.ndarray, context_ids: np.ndarray,
                               samples: int, seed: int) -> list[np.ndarray]:
    """Method-of-record source-cluster bootstrap; complete contexts stay together."""
    if samples < 1:
        raise GateError("bootstrap sample count must be positive")
    grouped = _cluster_map(cluster_ids, context_ids)
    clusters = list(grouped)
    rng = random.Random(seed)
    draws: list[np.ndarray] = []
    for _ in range(samples):
        pieces = [grouped[clusters[rng.randrange(len(clusters))]] for _ in clusters]
        draws.append(np.concatenate(pieces))
    return draws


def _percentile_interval(values: np.ndarray) -> list[float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return [float(ordered[int(0.025 * len(ordered))]),
            float(ordered[int(0.975 * len(ordered)) - 1])]


def _context_metric_table(kld: np.ndarray, ear: np.ndarray, top1: np.ndarray,
                          cluster_ids: np.ndarray, context_ids: np.ndarray) -> tuple[
                              np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, identity in enumerate(zip(cluster_ids, context_ids)):
        groups[tuple(map(str, identity))].append(index)
    clusters, contexts, metrics = [], [], []
    for (cluster, context), indices in groups.items():
        rows = np.asarray(indices, dtype=np.int64)
        summary = summarize_metrics(kld[rows], ear[rows], top1[rows])
        clusters.append(cluster)
        contexts.append(context)
        metrics.append([summary[name] for name in METRIC_NAMES])
    return np.asarray(metrics), np.asarray(clusters), np.asarray(contexts)


def _cluster_position_draws(cluster_ids: np.ndarray, samples: int,
                            seed: int) -> Iterator[np.ndarray]:
    """Yield source-cluster resamples while retaining all position rows."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        grouped[str(cluster)].append(index)
    if not grouped or samples < 1:
        raise GateError("global-tail bootstrap needs clusters and positive samples")
    clusters = list(grouped)
    rows = {cluster: np.asarray(indices, dtype=np.int64)
            for cluster, indices in grouped.items()}
    rng = random.Random(seed)
    for _ in range(samples):
        yield np.concatenate([rows[clusters[rng.randrange(len(clusters))]]
                              for _ in clusters])

def bootstrap_summary(kld: np.ndarray, ear: np.ndarray, top1: np.ndarray,
                      cluster_ids: np.ndarray, context_ids: np.ndarray,
                      samples: int, seed: int) -> dict[str, Any]:
    if not (len(kld) == len(ear) == len(top1) == len(cluster_ids) == len(context_ids)):
        raise GateError("bootstrap rows are not aligned")
    draws = _cluster_position_draws(cluster_ids, samples, seed)
    matrix = np.empty((samples, len(METRIC_NAMES)), dtype=np.float64)
    for row, indices in enumerate(draws):
        summary = summarize_metrics(kld[indices], ear[indices], top1[indices])
        matrix[row] = [summary[name] for name in METRIC_NAMES]
    return {
        "method": "source-cluster percentile bootstrap retaining complete position rows",
        "tail_interval_estimand": "global pooled per-position p99/CVaR1%",
        "samples": samples,
        "seed": seed,
        "clusters": len(set(map(str, cluster_ids))),
        "contexts": len(set(map(str, context_ids))),
        "ci95": {name: _percentile_interval(matrix[:, column])
                 for column, name in enumerate(METRIC_NAMES)},
    }


def paired_bootstrap(candidate: dict[str, np.ndarray], control: dict[str, np.ndarray],
                     samples: int, seed: int) -> dict[str, Any]:
    for key in ("cluster_id", "context_id", "position_id"):
        if key not in candidate or key not in control or not np.array_equal(
                candidate[key], control[key]):
            raise GateError(f"paired rows differ at {key}; common scored positions are mandatory")
    draws = _cluster_position_draws(candidate["cluster_id"], samples, seed)
    deltas = np.empty((samples, len(METRIC_NAMES)), dtype=np.float64)
    for row, indices in enumerate(draws):
        candidate_summary = summarize_metrics(
            candidate["kld"][indices], candidate["ear"][indices],
            candidate["top1"][indices])
        control_summary = summarize_metrics(
            control["kld"][indices], control["ear"][indices], control["top1"][indices])
        deltas[row] = [candidate_summary[name] - control_summary[name]
                       for name in METRIC_NAMES]
    observed_candidate = summarize_metrics(
        candidate["kld"], candidate["ear"], candidate["top1"])
    observed_control = summarize_metrics(
        control["kld"], control["ear"], control["top1"])
    context_candidate, clusters, contexts = _context_metric_table(
        candidate["kld"], candidate["ear"], candidate["top1"],
        candidate["cluster_id"], candidate["context_id"])
    context_control, control_clusters, control_contexts = _context_metric_table(
        control["kld"], control["ear"], control["top1"],
        control["cluster_id"], control["context_id"])
    if not np.array_equal(clusters, control_clusters) or not np.array_equal(
            contexts, control_contexts):
        raise GateError("paired context ordering differs")
    mean_deltas = context_candidate[:, 0] - context_control[:, 0]
    return {
        "schema": SCHEMA_COMPARISON,
        "sign": "candidate_minus_control",
        "estimand": "global pooled per-position metrics under source-cluster resampling",
        "samples": samples,
        "seed": seed,
        "clusters": len(set(map(str, clusters))),
        "contexts": len(contexts),
        "observed_delta": {name: observed_candidate[name] - observed_control[name]
                           for name in METRIC_NAMES},
        "context_outcomes": {
            "candidate_wins": int(np.count_nonzero(mean_deltas < 0)),
            "control_wins": int(np.count_nonzero(mean_deltas > 0)),
            "ties": int(np.count_nonzero(mean_deltas == 0)),
        },
        "ci95_delta": {name: _percentile_interval(deltas[:, column])
                       for column, name in enumerate(METRIC_NAMES)},
    }


def _aggregate_rows(values: dict[str, np.ndarray], keys: Sequence[np.ndarray],
                    names: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, identity in enumerate(zip(*keys)):
        grouped[tuple(map(str, identity))].append(index)
    result = []
    for identity in sorted(grouped):
        indices = np.asarray(grouped[identity], dtype=np.int64)
        summary = summarize_metrics(values["kld"][indices], values["ear"][indices],
                                    values["top1"][indices])
        result.append({**dict(zip(names, identity)), "scored_positions": len(indices),
                       **summary})
    return result


def load_metric_rows(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"].item())
            if schema != SCHEMA_ROWS:
                raise GateError(f"unexpected rows schema {schema!r}")
            return {key: np.asarray(data[key]) for key in
                    ("kld", "ear", "top1", "cluster_id", "context_id", "position_id")}
    except (OSError, KeyError, ValueError) as exc:
        raise GateError(f"cannot load metric rows {path}: {exc}") from exc


def kld_tail_histogram(values: np.ndarray) -> dict[str, Any]:
    edges = np.power(10.0, -12.0 + np.arange(561, dtype=np.float64) / 40.0)
    buckets = np.searchsorted(edges, values, side="right")
    counts = np.bincount(buckets, minlength=562)
    return {
        "log10_low": -12.0,
        "log10_high": 2.0,
        "bins_per_decade": 40,
        "counts": counts.tolist(),
        "underflow_exact_zero": int(np.count_nonzero(values == 0)),
        "overflow": int(counts[-1]),
    }


def compute_command(args: argparse.Namespace) -> None:
    contract = load_json(CANONICAL_CONTRACT)
    expected_vocab = int(contract["vocabulary"]["size"])
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    input_hashes = []
    next_position = 0
    for path in args.input:
        input_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        try:
            with np.load(path, allow_pickle=False) as data:
                required_head = contract["reference_semantics"]["shared_head_sha256"]
                if str(data["shared_head_sha256"].item()) != required_head:
                    raise GateError("probability shard does not pin the registered shared BF16 head")
                if bool(data["full_vocabulary"].item()) is not True:
                    raise GateError("probability shard is not declared full-vocabulary")
                if bool(data["body_only"].item()) is not True:
                    raise GateError("probability shard is not the registered body-only comparison")
                if str(data["kl_direction"].item()) != "reference||candidate":
                    raise GateError("probability shard has the wrong KL direction")
                if str(data["projection_method"].item()) != "tools/fidelity.py/two-pass":
                    raise GateError("probability shard was not emitted by the registered projector")
                p = np.asarray(data["reference_probabilities"])
                q = np.asarray(data["candidate_probabilities"])
                metrics = probability_metrics(p, q, expected_vocab)
                rows = p.shape[0]
                cluster = _expand_metadata(data["source_cluster_id"], rows, "source_cluster_id")
                context = _expand_metadata(data["context_id"], rows, "context_id")
                document = (_expand_metadata(data["document_id"], rows, "document_id")
                            if "document_id" in data else cluster)
                if "position_id" in data:
                    position = _expand_metadata(data["position_id"], rows, "position_id")
                else:
                    position = np.arange(next_position, next_position + rows, dtype=np.int64)
                domain = (_expand_metadata(data["domain"], rows, "domain") if "domain" in data
                          else np.repeat("unspecified", rows))
        except (OSError, KeyError, ValueError) as exc:
            raise GateError(f"invalid probability shard {path}: {exc}") from exc
        for name in ("kld", "jsd_bits", "ear", "top1"):
            arrays[name].append(metrics[name])
        arrays["cluster_id"].append(cluster.astype("U"))
        arrays["document_id"].append(document.astype("U"))
        arrays["context_id"].append(context.astype("U"))
        arrays["position_id"].append(position.astype("U"))
        arrays["domain"].append(domain.astype("U"))
        next_position += rows
    merged = {key: np.concatenate(parts) for key, parts in arrays.items()}
    identities = np.stack((merged["cluster_id"], merged["context_id"],
                           merged["position_id"]), axis=1)
    if len(np.unique(identities, axis=0)) != len(identities):
        raise GateError("duplicate source-cluster/context/position identity")
    point = summarize_metrics(merged["kld"], merged["ear"], merged["top1"])
    point.update({
        "mean_jsd_bits": float(np.mean(merged["jsd_bits"], dtype=np.float64)),
        "p50_kld": float(np.quantile(merged["kld"], 0.50, method="linear")),
        "p95_kld": float(np.quantile(merged["kld"], 0.95, method="linear")),
        "p999_kld": float(np.quantile(merged["kld"], 0.999, method="linear")),
        "max_kld": float(np.max(merged["kld"])),
    })
    bootstrap = bootstrap_summary(merged["kld"], merged["ear"], merged["top1"],
                                  merged["cluster_id"], merged["context_id"],
                                  args.bootstrap_samples, args.bootstrap_seed)
    per_context = _aggregate_rows(
        merged,
        (merged["cluster_id"], merged["document_id"], merged["context_id"], merged["domain"]),
        ("source_cluster_id", "document_id", "context_id", "domain"))
    per_document = _aggregate_rows(
        merged, (merged["cluster_id"], merged["document_id"]),
        ("source_cluster_id", "document_id"))
    per_domain = _aggregate_rows(merged, (merged["domain"],), ("domain",))
    worst = sorted(per_context, key=lambda row: (-row["mean_kld"], row["source_cluster_id"],
                                                 row["context_id"]))[:args.worst_contexts]
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.rows_out, schema=np.asarray(SCHEMA_ROWS),
                        kld=merged["kld"], ear=merged["ear"], top1=merged["top1"],
                        cluster_id=merged["cluster_id"], context_id=merged["context_id"],
                        position_id=merged["position_id"])
    report = {
        "schema": SCHEMA_SYNTHETIC,
        "promotable": False,
        "label": "synthetic/already-projected-vector-metrics",
        "candidate_id": args.candidate_id,
        "reference": {
            "semantics": "caller-supplied normalized vectors for metric tests only",
            "body_only": False,
            "full_vocabulary_width_checked": True,
        },
        "comparator": {
            "direction": "KL(reference || candidate)",
            "units": "natural-log nats",
            "input": "caller-supplied normalized vectors",
            "production_method": None,
        },
        "vocab_size": expected_vocab,
        "scored_positions": len(merged["kld"]),
        "source_clusters": bootstrap["clusters"],
        "contexts": bootstrap["contexts"],
        "metrics": point,
        "cluster_bootstrap": bootstrap,
        "kld_tail": kld_tail_histogram(merged["kld"]),
        "per_context": per_context,
        "per_document": per_document,
        "per_domain": per_domain,
        "worst_contexts": worst,
        "probability_shards": input_hashes,
        "row_metrics": {"path": str(args.rows_out), "sha256": sha256_file(args.rows_out)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))


def _load_method_projector(contract: dict[str, Any]) -> Any:
    path = REPO_ROOT / "tools/fidelity.py"
    expected = contract["reference_semantics"]["projector_source_sha256"]
    if sha256_file(path) != expected:
        raise GateError("method-of-record projector source digest changed")
    spec = importlib.util.spec_from_file_location("wave5_fidelity_method", path)
    if spec is None or spec.loader is None:
        raise GateError("cannot load method-of-record projector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_v5_contexts(prereg: dict[str, Any], split_name: str,
                          suite_root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    split_row = _registry_split(prereg, split_name)
    split_path = REPO_ROOT / split_row["manifest_path"]
    if sha256_file(split_path) != split_row["manifest_sha256"]:
        raise GateError("R29 split manifest digest changed")
    if selection_sha256(split_path, split_name) != split_row["selection_sha256"]:
        raise GateError("R29 selected document-set digest changed")
    split_manifest = load_json(split_path)
    selected_documents = [row for row in split_manifest["documents"]
                          if row.get("split") == split_name]
    requested: dict[tuple[int, int], dict[str, Any]] = {}
    for document in selected_documents:
        for context in document.get("contexts") or []:
            key = (int(context["shard"]), int(context["index"]))
            if key in requested:
                raise GateError(f"duplicate R29 evaluation context {key}")
            requested[key] = {
                "document_id": str(document["document_id"]),
                "document_sha256": document["document_sha256"],
                "r29_domain": document["domain"],
                "token_sha256": context["token_sha256"],
            }
    if len(requested) != split_row["primary_v5_contexts"]:
        raise GateError(f"{split_name} primary-v5 context count differs from preregistration")
    shard_registry = {int(row["shard"]): row
                      for row in split_manifest["published_suite"]["shards"]}
    result: list[dict[str, Any]] = []
    suite_by_shard: dict[int, dict[str, Any]] = {}
    for shard in sorted({key[0] for key in requested}):
        manifest_path = suite_root / f"shard-{shard:04d}" / "suite-manifest.json"
        if sha256_file(manifest_path) != shard_registry[shard]["manifest_sha256"]:
            raise GateError(f"v5 shard-{shard:04d} suite manifest digest changed")
        suite = load_json(manifest_path)
        if suite.get("suite_token_sha256") != shard_registry[shard]["suite_token_sha256"]:
            raise GateError(f"v5 shard-{shard:04d} token digest changed")
        suite_by_shard[shard] = suite
        contexts = {int(row["index"]): row for row in suite["context_index"]}
        for (requested_shard, index), r29 in requested.items():
            if requested_shard != shard:
                continue
            if index not in contexts:
                raise GateError(f"v5 shard-{shard:04d} lacks context {index}")
            suite_context = contexts[index]
            if suite_context["token_sha256"] != r29["token_sha256"]:
                raise GateError(f"R29/v5 token digest mismatch for shard {shard} context {index}")
            result.append({
                **r29,
                "shard": shard,
                "index": index,
                "context_id": f"shard-{shard:04d}:{index:04d}",
                "source_cluster": str(suite_context["source_cluster"]),
                "stratum": suite_context["stratum"],
            })
    exact_strata = {"code", "encyclopedic", "literary", "multilingual", "scientific"}
    if {row["stratum"] for row in result} != exact_strata:
        raise GateError(f"{split_name} does not contain the five registered v5 strata")
    return result, suite_by_shard


def _capture_shard(root: Path, shard: int, only_shard: bool) -> Path:
    nested = root / f"shard-{shard:04d}"
    if nested.is_dir():
        return nested
    if only_shard and (root / "capture-manifest.json").is_file():
        return root
    raise GateError(f"capture root {root} lacks shard-{shard:04d}")


def _checkpoint_identity_sha256(identity: dict[str, Any]) -> str:
    content = {
        key: identity.get(key) for key in
        ("model_revision", "index_sha256", "config_sha256", "shard_sha256")
    }
    if not content["model_revision"] and not content["shard_sha256"]:
        raise GateError("capture checkpoint identity is not content-pinned")
    return hashlib.sha256(canonical_json(content)).hexdigest()



def _project_context(ref_hidden: Any, candidate_hidden: Any, head: Any,
                     vocab_chunk: int, method: Any,
                     candidate_head: Any | None = None) -> dict[str, np.ndarray]:
    import torch

    candidate_projection = head if candidate_head is None else candidate_head
    ref_z, ref_top = method.normalizers_and_top1(ref_hidden, head, vocab_chunk)
    cand_z, cand_top = method.normalizers_and_top1(
        candidate_hidden, candidate_projection, vocab_chunk)
    rows = ref_hidden.shape[0]
    kl = torch.zeros(rows, dtype=torch.float64, device=ref_hidden.device)
    js = torch.zeros(rows, dtype=torch.float64, device=ref_hidden.device)
    ear = torch.zeros(rows, dtype=torch.float64, device=ref_hidden.device)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        ref_log = (ref_hidden @ head[start:end].T).float() - ref_z[:, None]
        candidate_log = (
            candidate_hidden @ candidate_projection[start:end].T).float() - cand_z[:, None]
        p, q = ref_log.exp(), candidate_log.exp()
        kl += (p * (ref_log - candidate_log)).sum(-1).double()
        midpoint = 0.5 * (p + q)
        log_midpoint = midpoint.clamp_min(1e-30).log()
        js += (0.5 * (p * (ref_log - log_midpoint)).sum(-1)
               + 0.5 * (q * (candidate_log - log_midpoint)).sum(-1)).double()
        ear += torch.minimum(p, q).sum(-1).double()
    return {
        "kld": kl.cpu().numpy(),
        "jsd_bits": (js / math.log(2.0)).cpu().numpy(),
        "ear": ear.cpu().numpy(),
        "top1": (ref_top == cand_top).double().cpu().numpy(),
    }


def replay_command(args: argparse.Namespace) -> None:
    """Run one capture-bound full-vocabulary body-only or served-head replay."""
    import torch
    from safetensors.torch import safe_open

    prereg, contract = _command_trust(args)
    open_state = None
    if args.split == "untouched_test":
        open_state = _validated_open_state(prereg, contract)
        expected_identity = {
            "candidate": open_state["complete_checkpoint_sha256"],
            "candidate-own-head": open_state["complete_checkpoint_sha256"],
            "F0": open_state["F0_checkpoint_identity_sha256"],
            "F0-fresh": open_state["F0_fresh_checkpoint_identity_sha256"],
        }[args.role]
        artifact_id = (open_state["candidate_id"]
                       if args.role in ("candidate", "candidate-own-head") else args.role)
    else:
        expected_identity = args.expected_checkpoint_identity_sha256
        if not _is_sha256(expected_identity):
            raise GateError("validation replay needs an exact checkpoint identity SHA256")
        artifact_id = args.candidate_id
        if not isinstance(artifact_id, str) or not artifact_id:
            raise GateError("validation replay needs a nonempty candidate ID")
    contexts, suites = _selected_v5_contexts(prereg, args.split, args.suite_root)
    method = _load_method_projector(contract)
    head_path = args.head.resolve()
    if sha256_file(head_path) != contract["reference_semantics"]["shared_head_sha256"]:
        raise GateError("shared BF16 head digest mismatch")
    device = torch.device(args.device)
    with safe_open(str(head_path), framework="pt", device="cpu") as file:
        key = "weight" if "weight" in file.keys() else file.keys()[0]
        raw_head = file.get_tensor(key)
    if str(raw_head.dtype) != "torch.bfloat16":
        raise GateError("shared head tensor is not BF16")
    head = raw_head.to(device)
    if list(head.shape) != contract["reference_semantics"]["shared_head_shape"]:
        raise GateError("shared BF16 head shape mismatch")
    vocab_chunk = int(contract["metric_contract"]["kld"]["vocab_chunk"])
    candidate_head = None
    candidate_head_sha256 = None
    candidate_head_dtype = None
    candidate_head_shape = None
    if args.role == "candidate-own-head":
        if args.candidate_head is None:
            raise GateError("candidate-own-head replay requires --candidate-head")
        candidate_head_path = args.candidate_head.resolve()
        candidate_head_sha256 = sha256_file(candidate_head_path)
        expected_head_sha256 = (open_state["candidate_head_sha256"] if open_state else
                                args.expected_candidate_head_sha256)
        if candidate_head_sha256 != expected_head_sha256:
            raise GateError("candidate head differs from the frozen head")
        with safe_open(str(candidate_head_path), framework="pt", device="cpu") as file:
            key = "weight" if "weight" in file.keys() else file.keys()[0]
            raw_candidate_head = file.get_tensor(key)
        candidate_head_dtype = str(raw_candidate_head.dtype)
        candidate_head_shape = list(raw_candidate_head.shape)
        if (candidate_head_dtype != contract["served_head_contract"]["dtype"]
                or candidate_head_shape != contract["served_head_contract"]["shape"]):
            raise GateError("candidate head dtype/shape differs from the frozen served-head contract")
        candidate_head = raw_candidate_head.to(device)
        if list(candidate_head.shape) != list(head.shape):
            raise GateError("candidate head shape differs from the shared head")
    shards = sorted(suites)
    selected_by_shard = {
        shard: [row for row in contexts if row["shard"] == shard] for shard in shards
    }
    capture_receipts = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    candidate_identity_digest = None
    for shard in shards:
        only_shard = len(shards) == 1
        reference_dir = _capture_shard(args.reference_root, shard, only_shard)
        candidate_dir = _capture_shard(args.candidate_root, shard, only_shard)
        selected_indices = {row["index"] for row in selected_by_shard[shard]}
        try:
            reference_manifest = method.validate_capture_files(
                reference_dir, suites[shard]["suite_token_sha256"], selected_indices)
            candidate_manifest = method.validate_capture_files(
                candidate_dir, suites[shard]["suite_token_sha256"], selected_indices)
            reference_identity = method.capture_identity(str(reference_dir), hash_shards=False)
            candidate_identity = method.capture_identity(str(candidate_dir), hash_shards=False)
        except SystemExit as exc:
            raise GateError(f"capture validation failed: {exc}") from exc
        if reference_identity.get("index_sha256") != contract["reference_semantics"][
                "source_model_index_sha256"]:
            raise GateError("BF16 reference capture has the wrong model index")
        for label, manifest, identity in (
                ("reference", reference_manifest, reference_identity),
                ("candidate", candidate_manifest, candidate_identity)):
            capture_contract = manifest.get("capture_contract", {})
            runtime = capture_contract.get("runtime", {})
            if (capture_contract.get("context_length") != 2048
                    or runtime.get("fp32") is not False
                    or runtime.get("chunk_accumulate") is not False
                    or runtime.get("max_batched_tokens") != 2048
                    or identity.get("kv_cache_dtype_resolved") not in
                    ("bfloat16", "torch.bfloat16")):
                raise GateError(f"{label} capture violates eager TP1/full-chunk/BF16-KV contract")
        if reference_identity.get("config_sha256") != contract["reference_semantics"][
                "source_config_sha256"]:
            raise GateError("BF16 reference capture has the wrong config")
        identity_digest = _checkpoint_identity_sha256(candidate_identity)
        if candidate_identity_digest is None:
            candidate_identity_digest = identity_digest
        elif candidate_identity_digest != identity_digest:
            raise GateError("candidate checkpoint identity differs between capture shards")
        binding_path = candidate_dir / "wave5-capture-binding.json"
        binding = load_json(binding_path)
        if (binding.get("schema") != "qwen38-wave5-capture-binding/2"
                or binding.get("role") != (
                    "candidate" if args.role == "candidate-own-head" else args.role)
                or binding.get("checkpoint_identity_sha256") != identity_digest
                or binding.get("capture_manifest_sha256") != sha256_file(
                    candidate_dir / "capture-manifest.json")
                or binding.get("capture_execution")
                != contract["capture_execution_contract"]):
            raise GateError("candidate capture binding is incomplete, stale, or noncanonical")
        execution_pointer = binding.get("capture_execution_receipt", {})
        execution_path = Path(_required(execution_pointer, "path"))
        if sha256_file(execution_path) != _required(execution_pointer, "sha256"):
            raise GateError("candidate capture execution receipt digest mismatch")
        execution_receipt = load_json(execution_path)
        if (execution_receipt.get("capture_execution")
                != contract["capture_execution_contract"]
                or execution_receipt.get("checkpoint_identity_sha256") != identity_digest
                or execution_receipt.get("capture_manifest_sha256")
                != binding["capture_manifest_sha256"]):
            raise GateError("candidate capture execution receipt is stale or mismatched")
        if args.role in ("candidate", "candidate-own-head") and binding.get(
                "action_registry_sha256") != (
                open_state["action_registry_sha256"] if open_state else
                binding.get("action_registry_sha256")):
            raise GateError("candidate capture uses another action registry")
        if args.role == "F0-fresh" and open_state and binding.get(
                "stock_action_sha256") != open_state["F0_fresh_stock_action_sha256"]:
            raise GateError("F0-fresh capture is not the frozen R30 stock action")
        if args.role == "F0" and candidate_identity.get(
                "model_revision") != prereg["controls"]["F0_model_revision"]:
            raise GateError("F0 capture is not the immutable shipped revision")
        capture_receipts.append({
            "shard": shard,
            "suite_manifest_sha256": sha256_file(
                args.suite_root / f"shard-{shard:04d}" / "suite-manifest.json"),
            "suite_token_sha256": suites[shard]["suite_token_sha256"],
            "reference_capture_manifest_sha256": sha256_file(
                reference_dir / "capture-manifest.json"),
            "candidate_capture_manifest_sha256": sha256_file(
                candidate_dir / "capture-manifest.json"),
            "reference_capture_contract_sha256": reference_manifest["capture_contract_sha256"],
            "candidate_capture_contract_sha256": candidate_manifest["capture_contract_sha256"],
            "candidate_capture_binding_sha256": sha256_file(binding_path),
            "candidate_capture_execution_sha256": hashlib.sha256(canonical_json(
                binding["capture_execution"])).hexdigest(),
            "candidate_capture_execution_receipt_sha256": execution_pointer["sha256"],
        })
        suite_contexts = {row["index"]: row for row in selected_by_shard[shard]}
        for index in sorted(selected_indices):
            with safe_open(str(reference_dir / f"hidden_{index:04d}.safetensors"),
                           framework="pt", device="cpu") as file:
                reference_hidden = file.get_tensor("hidden_states").to(device, torch.bfloat16)
            with safe_open(str(candidate_dir / f"hidden_{index:04d}.safetensors"),
                           framework="pt", device="cpu") as file:
                candidate_hidden = file.get_tensor("hidden_states").to(device, torch.bfloat16)
            expected_shape = [contract["benchmark_registry"]["primary"][
                "scored_positions_per_context"], contract["reference_semantics"]["hidden_size"]]
            if list(reference_hidden.shape) != expected_shape or list(
                    candidate_hidden.shape) != expected_shape:
                raise GateError("capture hidden-state shape differs from the registered window")
            metrics = _project_context(reference_hidden, candidate_hidden, head,
                                       vocab_chunk, method, candidate_head)
            context = suite_contexts[index]
            rows = expected_shape[0]
            for name in ("kld", "jsd_bits", "ear", "top1"):
                arrays[name].append(metrics[name])
            arrays["cluster_id"].append(np.repeat(context["source_cluster"], rows))
            arrays["document_id"].append(np.repeat(context["document_id"], rows))
            arrays["context_id"].append(np.repeat(context["context_id"], rows))
            arrays["position_id"].append(np.arange(rows, dtype=np.int64))
            arrays["domain"].append(np.repeat(context["stratum"], rows))
    if candidate_identity_digest != expected_identity:
        raise GateError("candidate capture checkpoint identity differs from the frozen identity")
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    expected_positions = len(contexts) * contract["benchmark_registry"]["primary"][
        "scored_positions_per_context"]
    if len(merged["kld"]) != expected_positions:
        raise GateError("replay did not produce the exact preregistered scored-row count")
    if not all(np.all(np.isfinite(merged[name])) for name in ("kld", "jsd_bits", "ear", "top1")):
        raise GateError("replay produced nonfinite metrics")
    bootstrap_samples = contract["statistical_contract"]["bootstrap"]["samples"]
    bootstrap_seed = contract["statistical_contract"]["bootstrap"]["seed"]
    point = summarize_metrics(merged["kld"], merged["ear"], merged["top1"])
    point.update({
        "token_mean_kld": point["mean_kld"],
        "mean_jsd_bits": float(np.mean(merged["jsd_bits"], dtype=np.float64)),
        "p50_kld": float(np.quantile(merged["kld"], 0.5, method="linear")),
        "p95_kld": float(np.quantile(merged["kld"], 0.95, method="linear")),
        "p999_kld": float(np.quantile(merged["kld"], 0.999, method="linear")),
        "max_kld": float(np.max(merged["kld"])),
    })
    per_context = _aggregate_rows(
        merged,
        (merged["cluster_id"], merged["document_id"], merged["context_id"], merged["domain"]),
        ("source_cluster_id", "document_id", "context_id", "domain"))
    point["context_macro_mean_kld"] = float(np.mean(
        [row["mean_kld"] for row in per_context], dtype=np.float64))
    if not math.isclose(point["context_macro_mean_kld"], point["token_mean_kld"],
                        rel_tol=0, abs_tol=1e-15):
        raise GateError("equal-length token/context macro means unexpectedly differ")
    bootstrap = bootstrap_summary(
        merged["kld"], merged["ear"], merged["top1"], merged["cluster_id"],
        merged["context_id"], bootstrap_samples, bootstrap_seed)
    per_document = _aggregate_rows(
        merged, (merged["cluster_id"], merged["document_id"]),
        ("source_cluster_id", "document_id"))
    per_domain = _aggregate_rows(merged, (merged["domain"],), ("domain",))
    worst = sorted(per_context, key=lambda row: (-row["mean_kld"],
                                                 row["source_cluster_id"],
                                                 row["context_id"]))[:20]
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.rows_out, schema=np.asarray(SCHEMA_ROWS), promotable=np.asarray(True),
        checkpoint_identity_sha256=np.asarray(candidate_identity_digest),
        split_selection_sha256=np.asarray(_registry_split(prereg, args.split)[
            "primary_v5_selection_sha256"]),
        role=np.asarray(args.role), candidate_head_sha256=np.asarray(
            candidate_head_sha256 or ""),
        kld=merged["kld"], ear=merged["ear"], top1=merged["top1"],
        cluster_id=merged["cluster_id"], context_id=merged["context_id"],
        position_id=merged["position_id"])
    report = {
        "schema": SCHEMA_REPORT,
        "candidate_id": artifact_id,
        "control_id": args.role,
        "candidate_checkpoint_sha256": candidate_identity_digest,
        "candidate_head": {
            "sha256": candidate_head_sha256,
            "dtype": candidate_head_dtype,
            "shape": candidate_head_shape,
        },
        "fidelity_axis": ("served-candidate-head" if args.role == "candidate-own-head"
                          else "body-only-shared-head"),
        "head_inclusive": args.role == "candidate-own-head",
        "split": args.split,
        "split_manifest_sha256": _registry_split(prereg, args.split)["manifest_sha256"],
        "split_selection_sha256": _registry_split(prereg, args.split)[
            "primary_v5_selection_sha256"],
        "provenance": {
            "reference_root": str(args.reference_root.resolve()),
            "candidate_root": str(args.candidate_root.resolve()),
            "suite_root": str(args.suite_root.resolve()),
            "head": str(head_path),
            "device": str(args.device),
            "candidate_head": str(args.candidate_head.resolve()) if args.candidate_head else None,
        },
        "reference": {
            "body_only": args.role != "candidate-own-head",
            "full_vocabulary": True,
            "source_model_index_sha256": contract["reference_semantics"][
                "source_model_index_sha256"],
            "source_config_sha256": contract["reference_semantics"]["source_config_sha256"],
            "reference_head_sha256": sha256_file(head_path),
            "candidate_projection_head_sha256": (
                candidate_head_sha256 if args.role == "candidate-own-head"
                else sha256_file(head_path)),
        },
        "comparator": {
            "direction": "KL(BF16_reference || candidate)",
            "units": "natural-log nats",
            "projector_source_sha256": contract["reference_semantics"][
                "projector_source_sha256"],
            "vocab_chunk": vocab_chunk,
            "within_chunk": "float32",
            "accumulation": "float64",
            "two_pass": True,
            "capture_point": "post-final-norm",
            "scored_positions": "rows 0..2046",
        },
        "vocab_size": int(head.shape[0]),
        "scored_positions": expected_positions,
        "source_clusters": len(set(map(str, merged["cluster_id"]))),
        "contexts": len(contexts),
        "strata": sorted(set(map(str, merged["domain"]))),
        "metrics": point,
        "cluster_bootstrap": bootstrap,
        "kld_tail": kld_tail_histogram(merged["kld"]),
        "per_context": per_context,
        "per_document": per_document,
        "per_domain": per_domain,
        "worst_contexts": worst,
        "captures": capture_receipts,
        "row_metrics": {"path": str(args.rows_out), "sha256": sha256_file(args.rows_out)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))


def compare_command(args: argparse.Namespace) -> None:
    _, contract = trusted_contract()
    candidate = load_metric_rows(args.candidate_rows)
    control = load_metric_rows(args.control_rows)
    bootstrap = contract["statistical_contract"]["bootstrap"]
    result = paired_bootstrap(candidate, control, bootstrap["samples"], bootstrap["seed"])
    result["candidate_rows_sha256"] = sha256_file(args.candidate_rows)
    result["control_rows_sha256"] = sha256_file(args.control_rows)
    result["gate_sha256"] = sha256_file(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result))


def append_hash_log(path: Path, event: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        previous = "0" * 64
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"access log is corrupt at line {number}") from exc
            if row.get("previous_event_sha256") != previous:
                raise GateError(f"access log hash chain breaks at line {number}")
            body = dict(row)
            recorded = body.pop("event_sha256", None)
            actual = hashlib.sha256(canonical_json(body)).hexdigest()
            if recorded != actual:
                raise GateError(f"access log event digest fails at line {number}")
            previous = recorded
        body = {**event, "previous_event_sha256": previous}
        digest = hashlib.sha256(canonical_json(body)).hexdigest()
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json({**body, "event_sha256": digest}).decode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return digest


def _registry_split(prereg: dict[str, Any], split: str) -> dict[str, Any]:
    try:
        row = prereg["split_registry"][split]
    except KeyError as exc:
        raise GateError(f"split {split!r} is not registered") from exc
    if not row.get("manifest_sha256") or len(row["manifest_sha256"]) != 64:
        raise GateError(f"split {split!r} has no frozen SHA256")
    return row
def selection_sha256(manifest_path: Path, split_name: str) -> str:
    manifest = load_json(manifest_path)
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise GateError("R29 split manifest has no documents array")
    selected = [row for row in documents
                if isinstance(row, dict) and row.get("split") == split_name]
    if not selected:
        raise GateError(f"R29 split manifest has no {split_name!r} documents")
    projection = {
        "schema": "qwen38-wave5-split-projection/1",
        "selector": {"field": "split", "op": "eq", "value": split_name},
        "document_sha256": sorted(row["document_sha256"] for row in selected),
        "context_token_sha256": sorted(
            context["token_sha256"] for row in selected
            for context in (row.get("contexts") or [])),
    }
    digest = hashlib.sha256(canonical_json(projection)).hexdigest()
    registered = manifest.get("split_contracts", {}).get(split_name, {}).get(
        "projection_sha256")
    if registered is not None and registered != digest:
        raise GateError(f"R29 embedded {split_name} projection digest is invalid")
    return digest




def _command_trust(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(args, "_test_mode", False):
        return load_json(args.prereg), load_json(args.contract)
    return trusted_contract()


def _command_path(args: argparse.Namespace, prereg: dict[str, Any], name: str) -> Path:
    if getattr(args, "_test_mode", False):
        return getattr(args, name)
    return canonical_state_path(prereg, name)


def authorize_command(args: argparse.Namespace) -> None:
    prereg, _ = _command_trust(args)
    log_path = _command_path(args, prereg, "access_log")
    outcome, reason = "denied", "unspecified"
    split_row: dict[str, Any] = {}
    manifest = Path(".")
    try:
        split_row = _registry_split(prereg, args.split)
        manifest = (REPO_ROOT / split_row["manifest_path"]
                    if not getattr(args, "_test_mode", False) else args.manifest)
        if sha256_file(manifest) != split_row["manifest_sha256"]:
            raise GateError("manifest SHA256 does not match the preregistration")
        if selection_sha256(manifest, args.split) != split_row["selection_sha256"]:
            raise GateError("selected document-set SHA256 does not match the preregistration")
        grants = split_row["access"]
        if not any(args.actor in grant["actors"] and args.phase in grant["phases"]
                   and args.purpose in grant["purposes"] for grant in grants):
            raise GateError("actor/phase/purpose is not preregistered for this split")
        if args.split == "untouched_test":
            raise GateError("untouched_test may only be opened through open-test")
        outcome, reason = "granted", "registered hash, selection, and grant match"
    except (OSError, KeyError, GateError) as exc:
        reason = str(exc)
    event = {
        "schema": SCHEMA_ACCESS, "operation": "authorize", "actor": args.actor,
        "phase": args.phase, "purpose": args.purpose, "split": args.split,
        "manifest_sha256": split_row.get("manifest_sha256"),
        "selection_sha256": split_row.get("selection_sha256"), "outcome": outcome,
        "reason": reason, "event_nonce": args.event_nonce,
    }
    digest = append_hash_log(log_path, event)
    if outcome != "granted":
        raise GateError(f"access denied and logged as {digest}: {reason}")
    print(json.dumps({"authorized": True, "event_sha256": digest,
                      "manifest_path": str(manifest)}, sort_keys=True))


def _validate_candidate_manifest(candidate: dict[str, Any], prereg: dict[str, Any],
                                 contract: dict[str, Any], test_mode: bool = False) -> None:
    required = prereg["candidate_freeze"]["required_manifest_fields"]
    missing = [field for field in required if field not in candidate]
    if missing:
        raise GateError(f"candidate manifest lacks freeze fields: {missing}")
    if not isinstance(candidate["candidate_id"], str) or not candidate["candidate_id"].strip():
        raise GateError("candidate_id must be nonempty")
    hash_fields = prereg["candidate_freeze"]["sha256_fields"]
    invalid_hashes = [field for field in hash_fields if not _is_sha256(candidate.get(field))]
    if invalid_hashes:
        raise GateError(f"candidate manifest has invalid SHA256 fields: {invalid_hashes}")
    candidate_head_path = Path(candidate["candidate_head_path"]).resolve()
    if sha256_file(candidate_head_path) != candidate["candidate_head_sha256"]:
        raise GateError("candidate head artifact digest mismatch")
    if (candidate.get("candidate_head_dtype") != contract["served_head_contract"]["dtype"]
            or candidate.get("candidate_head_shape")
            != contract["served_head_contract"]["shape"]):
        raise GateError("candidate head dtype/shape declaration differs from contract")
    if not test_mode:
        try:
            from safetensors.torch import safe_open
            with safe_open(str(candidate_head_path), framework="pt", device="cpu") as file:
                key = "weight" if "weight" in file.keys() else file.keys()[0]
                tensor = file.get_tensor(key)
        except (OSError, ValueError) as exc:
            raise GateError(f"cannot inspect candidate head tensor: {exc}") from exc
        if (str(tensor.dtype) != candidate["candidate_head_dtype"]
                or list(tensor.shape) != candidate["candidate_head_shape"]):
            raise GateError("candidate head declaration differs from tensor metadata")
    if candidate.get("selection_split") != "validation":
        raise GateError("candidate selection_split must be validation")
    if candidate.get("untouched_test_accessed") is not False:
        raise GateError("candidate must state untouched_test_accessed=false")
    validation = _registry_split(prereg, "validation")
    if candidate["validation_manifest_sha256"] != validation["manifest_sha256"]:
        raise GateError("candidate pins another validation manifest")
    if candidate["validation_selection_sha256"] != validation["primary_v5_selection_sha256"]:
        raise GateError("candidate pins another primary validation context set")
    if (not isinstance(candidate["exact_serialized_bytes"], int)
            or isinstance(candidate["exact_serialized_bytes"], bool)
            or candidate["exact_serialized_bytes"] < 0):
        raise GateError("exact_serialized_bytes must be a nonnegative integer")
    routes = contract["deployment_report_schema"]["profiles"]
    if candidate["codec_exact_route_id"] != routes["codec_exact"]["route_id"]:
        raise GateError("candidate codec-exact route ID is not registered")
    if candidate["production_route_id"] != routes["production"]["route_id"]:
        raise GateError("candidate production route ID is not registered")
    seeds = candidate["ordered_seed_list"]
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise GateError("ordered_seed_list must contain one or more integers")
    budgets = candidate["matched_search_budget"]
    if not isinstance(budgets, dict) or set(budgets) != {"candidate", "F0-fresh"}:
        raise GateError("matched_search_budget must contain candidate and F0-fresh")
    if budgets["candidate"] != budgets["F0-fresh"]:
        raise GateError("candidate and F0-fresh search budgets/common-random-number hashes differ")
    budget = budgets["candidate"]
    if (not isinstance(budget.get("legal_encoder_evaluations"), int)
            or budget["legal_encoder_evaluations"] < 1):
        raise GateError("legal encoder evaluation budget must be a positive integer")
    for field in ("seed_order_sha256", "document_order_sha256", "block_order_sha256"):
        if not _is_sha256(budget.get(field)):
            raise GateError(f"matched search budget lacks {field}")
    if budget["seed_order_sha256"] != hashlib.sha256(canonical_json(seeds)).hexdigest():
        raise GateError("ordered_seed_list does not match seed_order_sha256")
    for path_field, hash_name in (("document_order_path", "document_order_sha256"),
                                  ("block_order_path", "block_order_sha256")):
        order_path = Path(candidate[path_field]).resolve()
        if sha256_file(order_path) != budget[hash_name]:
            raise GateError(f"{path_field} does not match the common-random-number hash")
    artifact_fields = (
        ("action_registry_path", "action_registry_sha256"),
        ("dual_frontier_path", "dual_frontier_sha256"),
        ("selection_decision_path", "selection_decision_sha256"),
        ("validation_attempt_ledger_path", "validation_attempt_ledger_sha256"),
    )
    for path_field, hash_field in artifact_fields:
        artifact = Path(candidate[path_field]).resolve()
        if sha256_file(artifact) != candidate[hash_field]:
            raise GateError(f"candidate {path_field} digest mismatch")
    attempt_path = Path(candidate["validation_attempt_ledger_path"]).resolve()
    if not test_mode and attempt_path != canonical_state_path(
            prereg, "validation_attempt_log"):
        raise GateError("candidate validation attempts do not use the canonical ledger")
    attempts = _read_hash_log(attempt_path)
    if not attempts or not all(row.get("operation") == "validation-attempt"
                               for row in attempts):
        raise GateError("candidate validation-attempt ledger is empty or invalid")
    selected_arm = candidate["selected_arm_id"]
    registered_arms = {row["arm_id"] for row in prereg["arm_registry"]}
    if selected_arm not in registered_arms:
        raise GateError("selected arm is not preregistered")
    expected_evaluations = budget["legal_encoder_evaluations"]
    budgeted_arms = ["F0-fresh"] + [
        row["arm_id"] for row in prereg["arm_registry"] if row.get("test_eligible") is True]
    for arm in budgeted_arms:
        arm_rows = [row for row in attempts if row.get("arm_id") == arm
                    and row.get("infrastructure_replacement") is not True]
        if len(arm_rows) != expected_evaluations:
            raise GateError(f"{arm} validation attempts do not match the frozen budget")
        if [row.get("seed") for row in arm_rows] != seeds[:len(arm_rows)]:
            raise GateError(f"{arm} validation seed order differs")
        for ordinal, row in enumerate(arm_rows):
            if (row.get("split") != "validation"
                    or row.get("split_selection_sha256")
                    != validation["primary_v5_selection_sha256"]
                    or row.get("attempt_ordinal") != ordinal
                    or row.get("seed_order_sha256") != budget["seed_order_sha256"]
                    or row.get("document_order_sha256") != budget["document_order_sha256"]
                    or row.get("block_order_sha256") != budget["block_order_sha256"]
                    or not _is_sha256(row.get("setting_sha256"))):
                raise GateError(f"{arm} attempt is not validation-only/common-random-number bound")
    if any(row.get("split") == "untouched_test" for row in attempts):
        raise GateError("selection ledger contains untouched-test evidence")
    decision = load_json(Path(candidate["selection_decision_path"]))
    if (decision.get("schema") != SCHEMA_SELECTION
            or decision.get("selected_arm_id") != selected_arm
            or decision.get("selection_split") != "validation"
            or decision.get("validation_selection_sha256")
            != validation["primary_v5_selection_sha256"]
            or decision.get("validation_attempt_ledger_sha256")
            != candidate["validation_attempt_ledger_sha256"]
            or decision.get("dual_frontier_sha256") != candidate["dual_frontier_sha256"]
            or decision.get("action_assignment_sha256")
            != candidate["action_assignment_sha256"]
            or decision.get("candidate_checkpoint_sha256")
            != candidate["complete_checkpoint_sha256"]
            or decision.get("thresholds_sha256") != _thresholds_sha256(contract)):
        raise GateError("selection decision is not frozen to validation evidence/frontier/candidate")


def freeze_command(args: argparse.Namespace) -> None:
    prereg, contract = _command_trust(args)
    candidate = load_json(args.candidate_manifest)
    _validate_candidate_manifest(candidate, prereg, contract,
                                 test_mode=getattr(args, "_test_mode", False))
    output = _command_path(args, prereg, "candidate_freeze")
    record = {
        "schema": SCHEMA_FREEZE,
        "candidate_manifest_path": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "prereg_sha256": sha256_file(CANONICAL_PREREG) if not getattr(args, "_test_mode", False)
                          else sha256_file(args.prereg),
        "contract_sha256": sha256_file(CANONICAL_CONTRACT) if not getattr(args, "_test_mode", False)
                           else sha256_file(args.contract),
        **{field: candidate[field] for field in prereg["candidate_freeze"][
            "carry_into_freeze"]},
    }
    write_json_exclusive(output, record)


def _verify_open_capability(prereg: dict[str, Any]) -> str:
    token = os.environ.get("WAVE5_TEST_OPEN_CAPABILITY")
    if not token:
        raise GateError("WAVE5_TEST_OPEN_CAPABILITY is absent")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if digest != prereg["untouched_test_policy"]["capability_sha256"]:
        raise GateError("untouched-test capability digest mismatch")
    return digest


def open_test_command(args: argparse.Namespace) -> None:
    prereg, contract = _command_trust(args)
    test = _registry_split(prereg, "untouched_test")
    state_path = _command_path(args, prereg, "test_open_state")
    log_path = _command_path(args, prereg, "access_log")
    freeze_path = _command_path(args, prereg, "candidate_freeze")
    manifest = (REPO_ROOT / test["manifest_path"]
                if not getattr(args, "_test_mode", False) else args.manifest)
    outcome, reason = "denied", "unspecified"
    freeze: dict[str, Any] = {}
    capability_digest = None
    try:
        if not getattr(args, "_test_mode", False):
            capability_digest = _verify_open_capability(prereg)
        freeze = load_json(freeze_path)
        if freeze.get("schema") != SCHEMA_FREEZE:
            raise GateError("canonical candidate freeze is invalid")
        expected_prereg_hash = (sha256_file(CANONICAL_PREREG)
                                if not getattr(args, "_test_mode", False)
                                else sha256_file(args.prereg))
        expected_contract_hash = (sha256_file(CANONICAL_CONTRACT)
                                  if not getattr(args, "_test_mode", False)
                                  else sha256_file(args.contract))
        if freeze.get("prereg_sha256") != expected_prereg_hash:
            raise GateError("candidate freeze pins another preregistration")
        if freeze.get("contract_sha256") != expected_contract_hash:
            raise GateError("candidate freeze pins another fidelity contract")
        if sha256_file(manifest) != test["manifest_sha256"]:
            raise GateError("untouched-test manifest hash mismatch")
        if selection_sha256(manifest, "untouched_test") != test["selection_sha256"]:
            raise GateError("untouched-test document-set hash mismatch")
        grant_body = {
            "schema": SCHEMA_ACCESS, "operation": "open-test", "actor_uid": os.getuid(),
            "phase": "untouched_test", "purpose": "final_evaluation",
            "split": "untouched_test", "manifest_sha256": test["manifest_sha256"],
            "selection_sha256": test["selection_sha256"],
            "primary_v5_selection_sha256": test["primary_v5_selection_sha256"],
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "capability_sha256": capability_digest, "outcome": "granted",
            "event_nonce": args.event_nonce,
        }
        grant_digest = hashlib.sha256(canonical_json(grant_body)).hexdigest()
        record = {
            "schema": SCHEMA_TEST_OPEN,
            "grant_event": {**grant_body, "event_sha256": grant_digest},
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "test_manifest_sha256": test["manifest_sha256"],
            "test_selection_sha256": test["selection_sha256"],
            "test_primary_v5_selection_sha256": test["primary_v5_selection_sha256"],
            **{field: freeze[field] for field in prereg["candidate_freeze"][
                "carry_into_open_state"]},
        }
        write_json_exclusive(state_path, record)
        outcome, reason = "granted", "canonical once-only state contains the grant event"
    except (OSError, KeyError, GateError) as exc:
        reason = str(exc)
    event = {
        "schema": SCHEMA_ACCESS, "operation": "open-test", "actor_uid": os.getuid(),
        "phase": "untouched_test", "purpose": "final_evaluation",
        "split": "untouched_test", "manifest_sha256": test.get("manifest_sha256"),
        "selection_sha256": test.get("selection_sha256"),
        "candidate_freeze_sha256": sha256_file(freeze_path) if freeze_path.exists() else None,
        "outcome": outcome, "reason": reason, "event_nonce": args.event_nonce,
    }
    digest = append_hash_log(log_path, event)
    if outcome != "granted":
        raise GateError(f"untouched-test access denied and logged as {digest}: {reason}")
    print(json.dumps({"authorized": True, "event_sha256": digest,
                      "open_state_sha256": sha256_file(state_path)}, sort_keys=True))

def _required(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise GateError(f"required evaluation field {path!r} is missing")
        value = value[component]
    return value


def _read_hash_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    previous = "0" * 64
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("previous_event_sha256") != previous:
            raise GateError(f"access-log chain breaks at line {number}")
        body = dict(row)
        recorded = body.pop("event_sha256", None)
        if recorded != hashlib.sha256(canonical_json(body)).hexdigest():
            raise GateError(f"access-log event digest fails at line {number}")
        previous = recorded
        rows.append(row)
    return rows
def _validated_open_state(prereg: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    state_path = canonical_state_path(prereg, "test_open_state")
    freeze_path = canonical_state_path(prereg, "candidate_freeze")
    state = load_json(state_path)
    if state.get("schema") != SCHEMA_TEST_OPEN:
        raise GateError("canonical untouched-test state is invalid")
    if state.get("candidate_freeze_sha256") != sha256_file(freeze_path):
        raise GateError("open state does not bind the canonical candidate freeze")
    if (state.get("prereg_sha256") != sha256_file(CANONICAL_PREREG)
            or state.get("contract_sha256") != sha256_file(CANONICAL_CONTRACT)):
        raise GateError("open state trust root changed")
    grant = dict(state.get("grant_event") or {})
    recorded = grant.pop("event_sha256", None)
    if recorded != hashlib.sha256(canonical_json(grant)).hexdigest():
        raise GateError("open state grant event digest fails")
    test = _registry_split(prereg, "untouched_test")
    if (grant.get("outcome") != "granted"
            or grant.get("capability_sha256") != prereg["untouched_test_policy"][
                "capability_sha256"]
            or grant.get("manifest_sha256") != test["manifest_sha256"]
            or grant.get("selection_sha256") != test["selection_sha256"]
            or grant.get("primary_v5_selection_sha256")
            != test["primary_v5_selection_sha256"]
            or grant.get("candidate_freeze_sha256") != state[
                "candidate_freeze_sha256"]):
        raise GateError("open state has no exact capability-authorized grant")
    ledger = _read_hash_log(canonical_state_path(prereg, "access_log"))
    matching_grants = [
        row for row in ledger
        if row.get("operation") == "open-test" and row.get("outcome") == "granted"
        and row.get("manifest_sha256") == test["manifest_sha256"]
        and row.get("selection_sha256") == test["selection_sha256"]
        and row.get("candidate_freeze_sha256") == state["candidate_freeze_sha256"]]
    if len(matching_grants) != 1:
        raise GateError("canonical access ledger does not contain exactly one matching grant")
    return state


def _validate_report_header(report: dict[str, Any], expected_role: str,
                            prereg: dict[str, Any], contract: dict[str, Any],
                            open_state: dict[str, Any], label: str,
                            expected_split: str = "untouched_test",
                            expected_candidate_id: str | None = None) -> tuple[
                                dict[str, Any], bool]:
    expected_id = (expected_candidate_id if expected_candidate_id is not None else
                   open_state["candidate_id"]
                   if expected_role in ("candidate", "candidate-own-head")
                   else expected_role)
    if (report.get("schema") != SCHEMA_REPORT
            or report.get("control_id") != expected_role
            or report.get("candidate_id") != expected_id):
        raise GateError(f"{label} is not the registered {expected_role} replay report")
    if report.get("split") != expected_split:
        raise GateError(f"{label} is not a {expected_split} replay")
    split_row = _registry_split(prereg, expected_split)
    exact = {
        "split_manifest_sha256": split_row["manifest_sha256"],
        "split_selection_sha256": split_row["primary_v5_selection_sha256"],
        "vocab_size": contract["vocabulary"]["size"],
        "contexts": split_row["primary_v5_contexts"],
        "scored_positions": split_row["primary_v5_contexts"]
                            * contract["benchmark_registry"]["primary"][
                                "scored_positions_per_context"],
    }
    if not _is_sha256(report.get("split_selection_sha256")):
        raise GateError(f"{label} has a zero/invalid split selection hash")
    for field, expected in exact.items():
        if report.get(field) != expected:
            raise GateError(f"{label} has wrong {field}: {report.get(field)!r}")
    if report.get("strata") != ["code", "encyclopedic", "literary", "multilingual",
                                "scientific"]:
        raise GateError(f"{label} does not contain exactly the five v5 strata")
    reference = report.get("reference", {})
    if (reference.get("full_vocabulary") is not True
            or reference.get("reference_head_sha256") != contract[
                "reference_semantics"]["shared_head_sha256"]):
        raise GateError(f"{label} uses top-k output or another BF16 reference head")
    head_inclusive = expected_role == "candidate-own-head"
    expected_axis = "served-candidate-head" if head_inclusive else "body-only-shared-head"
    if (report.get("head_inclusive") is not head_inclusive
            or report.get("fidelity_axis") != expected_axis
            or reference.get("body_only") is head_inclusive):
        raise GateError(f"{label} conflates body-only and served-head fidelity")
    projection_head = reference.get("candidate_projection_head_sha256")
    if head_inclusive:
        candidate_head = report.get("candidate_head", {})
        if (candidate_head.get("sha256") != open_state["candidate_head_sha256"]
                or candidate_head.get("dtype") != open_state["candidate_head_dtype"]
                or candidate_head.get("shape") != open_state["candidate_head_shape"]
                or projection_head != open_state["candidate_head_sha256"]):
            raise GateError(f"{label} candidate head is absent, stale, or mismatched")
    elif (report.get("candidate_head") != {"sha256": None, "dtype": None, "shape": None}
          or projection_head != contract["reference_semantics"]["shared_head_sha256"]):
        raise GateError(f"{label} body-only replay does not exclusively use the shared head")
    comparator = report.get("comparator", {})
    required_comparator = {
        "direction": "KL(BF16_reference || candidate)",
        "units": "natural-log nats",
        "projector_source_sha256": contract["reference_semantics"]["projector_source_sha256"],
        "vocab_chunk": contract["metric_contract"]["kld"]["vocab_chunk"],
        "accumulation": "float64",
        "two_pass": True,
        "capture_point": "post-final-norm",
        "scored_positions": "rows 0..2046",
    }
    for field, expected in required_comparator.items():
        if comparator.get(field) != expected:
            raise GateError(f"{label} comparator differs at {field}")
    bootstrap = report.get("cluster_bootstrap", {})
    frozen_bootstrap = contract["statistical_contract"]["bootstrap"]
    if (bootstrap.get("samples") != frozen_bootstrap["samples"]
            or bootstrap.get("seed") != frozen_bootstrap["seed"]
            or bootstrap.get("tail_interval_estimand")
            != "global pooled per-position p99/CVaR1%"):
        raise GateError(f"{label} does not use the frozen global-tail bootstrap")
    return exact, head_inclusive


def _validated_report(path: Path, expected_role: str, prereg: dict[str, Any],
                      contract: dict[str, Any], open_state: dict[str, Any],
                      expected_split: str = "untouched_test",
                      expected_checkpoint: str | None = None,
                      expected_candidate_id: str | None = None,
                      expected_action_registry: str | None = None,
                      command_args: argparse.Namespace | None = None) -> tuple[
                          dict[str, Any], dict[str, np.ndarray]]:
    report = load_json(path)
    exact, head_inclusive = _validate_report_header(
        report, expected_role, prereg, contract, open_state, str(path),
        expected_split, expected_candidate_id)
    split_row = _registry_split(prereg, expected_split)
    bootstrap = report["cluster_bootstrap"]
    frozen_bootstrap = contract["statistical_contract"]["bootstrap"]
    row_path = Path(_required(report, "row_metrics.path"))
    if sha256_file(row_path) != _required(report, "row_metrics.sha256"):
        raise GateError(f"{path} row payload digest mismatch")
    import tempfile
    provenance = report.get("provenance", {})
    with tempfile.TemporaryDirectory(prefix="wave5-replay-verify-") as directory:
        generated_rows = Path(directory) / "rows.npz"
        generated_report = Path(directory) / "report.json"
        replay_args = {
            "reference_root": Path(_required(provenance, "reference_root")),
            "candidate_root": Path(_required(provenance, "candidate_root")),
            "suite_root": Path(_required(provenance, "suite_root")),
            "head": Path(_required(provenance, "head")),
            "candidate_head": (Path(provenance["candidate_head"])
                               if provenance.get("candidate_head") else None),
            "role": expected_role, "split": expected_split,
            "candidate_id": expected_candidate_id,
            "expected_checkpoint_identity_sha256": expected_checkpoint,
            "expected_candidate_head_sha256": (
                open_state.get("candidate_head_sha256") if head_inclusive else None),
            "expected_action_registry_sha256": expected_action_registry,
            "device": _required(provenance, "device"), "rows_out": generated_rows,
            "output": generated_report,
        }
        if command_args is not None and getattr(command_args, "_test_mode", False):
            replay_args.update({
                "_test_mode": True, "prereg": command_args.prereg,
                "contract": command_args.contract,
                "test_open_state": command_args.test_open_state,
                "candidate_freeze": command_args.candidate_freeze,
                "access_log": command_args.access_log,
            })
        replay_command(argparse.Namespace(**replay_args))
        regenerated = load_json(generated_report)
        for field in ("candidate_id", "control_id", "candidate_checkpoint_sha256",
                      "candidate_head", "fidelity_axis", "head_inclusive",
                      "split_manifest_sha256", "split_selection_sha256", "reference",
                      "comparator", "contexts", "scored_positions", "strata", "metrics",
                      "cluster_bootstrap", "kld_tail", "captures", "per_context",
                      "per_document", "per_domain", "worst_contexts"):
            if regenerated.get(field) != report.get(field):
                raise GateError(f"{path} differs from projector rerun at {field}")
        regenerated_rows = load_metric_rows(generated_rows)
        supplied_rows = load_metric_rows(row_path)
        for field in ("kld", "ear", "top1", "cluster_id", "context_id", "position_id"):
            if not np.array_equal(regenerated_rows[field], supplied_rows[field]):
                raise GateError(f"{path} row payload differs from projector rerun at {field}")
    with np.load(row_path, allow_pickle=False) as data:
        if bool(data["promotable"].item()) is not True:
            raise GateError(f"{path} row payload is not promotable")
        if str(data["checkpoint_identity_sha256"].item()) != report[
                "candidate_checkpoint_sha256"]:
            raise GateError(f"{path} row checkpoint identity mismatch")
        if str(data["split_selection_sha256"].item()) != split_row[
                "primary_v5_selection_sha256"]:
            raise GateError(f"{path} row split identity mismatch")
        if str(data["role"].item()) != expected_role:
            raise GateError(f"{path} row role mismatch")
        expected_head = open_state["candidate_head_sha256"] if head_inclusive else ""
        if str(data["candidate_head_sha256"].item()) != expected_head:
            raise GateError(f"{path} row head identity mismatch")
    rows = load_metric_rows(row_path)
    if len(rows["kld"]) != exact["scored_positions"]:
        raise GateError(f"{path} row count mismatch")
    if not all(np.all(np.isfinite(rows[name])) for name in ("kld", "ear", "top1")):
        raise GateError(f"{path} has nonfinite metric rows")
    context_groups: dict[str, list[int]] = defaultdict(list)
    for index, context in enumerate(rows["context_id"]):
        context_groups[str(context)].append(index)
    expected_positions = np.arange(
        contract["benchmark_registry"]["primary"]["scored_positions_per_context"])
    if len(context_groups) != exact["contexts"]:
        raise GateError(f"{path} context identity count mismatch")
    for context, indices in context_groups.items():
        if not np.array_equal(rows["position_id"][indices].astype(np.int64),
                              expected_positions):
            raise GateError(f"{path} context {context} does not contain rows 0..2046 once")
        if len(set(map(str, rows["cluster_id"][indices]))) != 1:
            raise GateError(f"{path} context {context} crosses source clusters")
    point = summarize_metrics(rows["kld"], rows["ear"], rows["top1"])
    for field in METRIC_NAMES:
        if not math.isclose(float(report["metrics"][field]), point[field],
                            rel_tol=0, abs_tol=1e-15):
            raise GateError(f"{path} metric {field} differs from its row payload")
    if not math.isclose(float(report["metrics"]["token_mean_kld"]), point["mean_kld"],
                        rel_tol=0, abs_tol=1e-15):
        raise GateError(f"{path} token mean differs from row payload")
    context_mean = float(np.mean([
        np.mean(rows["kld"][indices], dtype=np.float64)
        for indices in context_groups.values()], dtype=np.float64))
    if not math.isclose(float(report["metrics"]["context_macro_mean_kld"]),
                        context_mean, rel_tol=0, abs_tol=1e-15):
        raise GateError(f"{path} context macro mean differs from row payload")
    recomputed_bootstrap = bootstrap_summary(
        rows["kld"], rows["ear"], rows["top1"], rows["cluster_id"], rows["context_id"],
        frozen_bootstrap["samples"], frozen_bootstrap["seed"])
    for metric in BOOTSTRAP_NAMES:
        got, expected_ci = bootstrap["ci95"][metric], recomputed_bootstrap["ci95"][metric]
        if not np.allclose(got, expected_ci, rtol=0, atol=1e-15):
            raise GateError(f"{path} bootstrap interval differs for {metric}")
    frozen_identity = expected_checkpoint
    if frozen_identity is None:
        frozen_identity = {
            "candidate": open_state["complete_checkpoint_sha256"],
            "candidate-own-head": open_state["complete_checkpoint_sha256"],
            "F0": open_state["F0_checkpoint_identity_sha256"],
            "F0-fresh": open_state["F0_fresh_checkpoint_identity_sha256"],
        }[expected_role]
    if report["candidate_checkpoint_sha256"] != frozen_identity:
        raise GateError(f"{path} checkpoint identity is not frozen")
    return report, rows


def _load_profile_receipt(pointer: dict[str, Any], expected_route: str,
                          expected_id: str, expected_checkpoint: str,
                          expected_action_registry: str | None,
                          expected_head_sha256: str,
                          open_state_sha256: str) -> dict[str, Any]:
    path = Path(_required(pointer, "receipt_path"))
    if sha256_file(path) != _required(pointer, "receipt_sha256"):
        raise GateError(f"runtime receipt digest mismatch: {path}")
    receipt = load_json(path)
    if (receipt.get("schema") != SCHEMA_RUNTIME
            or receipt.get("candidate_id") != expected_id
            or receipt.get("candidate_checkpoint_sha256") != expected_checkpoint
            or receipt.get("route_id") != expected_route):
        raise GateError(f"runtime receipt has wrong artifact/route: {path}")
    if expected_action_registry is not None and receipt.get(
            "action_registry_sha256") != expected_action_registry:
        raise GateError(f"runtime receipt has wrong action registry: {path}")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        raise GateError(f"runtime receipt lacks provenance: {path}")
    required_strings = (
        "image_digest", "vllm_revision", "build_revision", "gpu_uuid", "gpu_name",
        "driver_version", "attention_backend", "kv_cache_dtype", "cudagraph_mode",
        "profile")
    if any(not isinstance(provenance.get(field), str) or not provenance[field]
           for field in required_strings):
        raise GateError(f"runtime receipt has absent provenance fields: {path}")
    if not provenance["image_digest"].startswith("sha256:"):
        raise GateError(f"runtime image is not digest-pinned: {path}")
    exact = {
        "candidate_checkpoint_sha256": expected_checkpoint,
        "route_id": expected_route,
        "head_sha256": expected_head_sha256,
        "test_open_state_sha256": open_state_sha256,
    }
    for field, expected in exact.items():
        if provenance.get(field) != expected:
            raise GateError(f"runtime provenance is stale/mismatched at {field}: {path}")
    if not isinstance(provenance.get("mtp_tokens"), int):
        raise GateError(f"runtime provenance lacks exact MTP token count: {path}")
    environment = receipt.get("environment")
    if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()):
        raise GateError(f"runtime receipt lacks exact effective environment: {path}")
    environment_sha = hashlib.sha256(canonical_json(environment)).hexdigest()
    if provenance.get("effective_environment_sha256") != environment_sha:
        raise GateError(f"runtime effective-environment digest mismatch: {path}")
    for evidence_name in ("effective_environment_receipt", "startup_receipt",
                          "runtime_receipt", "raw_evidence"):
        pointer_value = receipt.get(evidence_name)
        if not isinstance(pointer_value, dict):
            raise GateError(f"runtime receipt lacks {evidence_name}: {path}")
        evidence_path = Path(_required(pointer_value, "path"))
        if sha256_file(evidence_path) != _required(pointer_value, "sha256"):
            raise GateError(f"runtime {evidence_name} digest mismatch: {evidence_path}")
        if evidence_name == "effective_environment_receipt":
            environment_receipt = load_json(evidence_path)
            if environment_receipt.get("environment") != environment:
                raise GateError(f"effective environment receipt disagrees: {evidence_path}")
    return receipt


def _runtime_stack_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    provenance = receipt["provenance"]
    fields = (
        "image_digest", "vllm_revision", "build_revision", "gpu_uuid", "gpu_name",
        "driver_version", "attention_backend", "kv_cache_dtype", "mtp_tokens",
        "cudagraph_mode", "profile", "route_id")
    return {field: provenance[field] for field in fields}

def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _validated_byte_manifest(pointer: dict[str, Any], checkpoint_sha256: str,
                             action_sha256: str | None = None,
                             expected_total: int | None = None) -> dict[str, Any]:
    manifest, _ = _load_typed_receipt(
        pointer, SCHEMA_BYTE_MANIFEST, "checkpoint byte manifest")
    if (manifest.get("checkpoint_identity_sha256") != checkpoint_sha256
            or manifest.get("action_sha256") != action_sha256):
        raise GateError("checkpoint byte manifest belongs to another checkpoint/action")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise GateError("checkpoint byte manifest has no files")
    canonical_paths = [row.get("relative_path") for row in files
                       if isinstance(row, dict)]
    if (len(canonical_paths) != len(files)
            or canonical_paths != sorted(canonical_paths)
            or len(set(canonical_paths)) != len(canonical_paths)):
        raise GateError("checkpoint byte manifest paths are duplicate/noncanonical")
    categories = {
        "trellis_payload": "trellis_payload_bytes",
        "sidecar": "sidecar_bytes",
        "header_alignment": "header_alignment_bytes",
        "other_checkpoint": "other_checkpoint_bytes",
    }
    breakdown = {value: 0 for value in categories.values()}
    total = 0
    for row in files:
        artifact = Path(_required(row, "path"))
        size = artifact.stat().st_size
        if (row.get("exact_bytes") != size
                or row.get("sha256") != sha256_file(artifact)
                or row.get("category") not in categories):
            raise GateError(f"checkpoint byte manifest entry is stale: {artifact}")
        total += size
        breakdown[categories[row["category"]]] += size
    if (manifest.get("exact_serialized_bytes") != total
            or manifest.get("serialized_byte_breakdown") != breakdown
            or (expected_total is not None and total != expected_total)):
        raise GateError("checkpoint byte manifest totals were not recomputed from files")
    return manifest


def _legal_action_menu(registry: dict[str, Any]) -> tuple[
        dict[str, dict[str, dict[str, Any]]], str]:
    available: dict[str, dict[str, dict[str, Any]]] = {}
    projection = []
    units = registry.get("units")
    if not isinstance(units, list) or not units:
        raise GateError("R30 action registry has no legal units")
    for unit in units:
        unit_id = unit.get("unit_id")
        actions = unit.get("actions")
        if (not isinstance(unit_id, str) or unit_id in available
                or not isinstance(actions, list) or not actions):
            raise GateError("R30 action registry has duplicate/empty units")
        menu: dict[str, dict[str, Any]] = {}
        for action in actions:
            action_id = action.get("action_id")
            size = action.get("exact_serialized_bytes")
            if (not isinstance(action_id, str) or action_id in menu
                    or action.get("schema") != "wave5/exl3-action/1"
                    or not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise GateError(f"unit {unit_id} has an invalid complete action")
            menu[action_id] = action
        available[unit_id] = menu
        projection.append({
            "unit_id": unit_id,
            "actions": [{"action_id": action_id,
                         "exact_serialized_bytes": menu[action_id]["exact_serialized_bytes"]}
                        for action_id in sorted(menu)],
        })
    projection.sort(key=lambda row: row["unit_id"])
    return available, hashlib.sha256(canonical_json(projection)).hexdigest()


def _assignment_details(assignment: Any, available: dict[
        str, dict[str, dict[str, Any]]]) -> tuple[int, str]:
    if not isinstance(assignment, list) or len(assignment) != len(available):
        raise GateError("assignment does not select exactly one action per legal unit")
    canonical = sorted(assignment, key=lambda row: row.get("unit_id", ""))
    if assignment != canonical:
        raise GateError("assignment must use canonical unit-id order")
    selected: set[str] = set()
    total = 0
    for row in assignment:
        if not isinstance(row, dict) or set(row) != {"unit_id", "action_id"}:
            raise GateError("assignment rows must contain only unit_id/action_id")
        unit_id, action_id = row["unit_id"], row["action_id"]
        if unit_id in selected or unit_id not in available or action_id not in available[unit_id]:
            raise GateError("assignment contains a duplicate or illegal action")
        selected.add(unit_id)
        total += available[unit_id][action_id]["exact_serialized_bytes"]
    return total, hashlib.sha256(canonical_json(assignment)).hexdigest()


def _metric_tuple(point: dict[str, Any]) -> tuple[float, ...]:
    names = ("mean_kld", "p99_kld", "cvar1_kld", "ear", "top1_agreement",
             "runtime_seconds", "startup_seconds")
    raw = [point.get(name) for name in names]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value)) for value in raw):
        raise GateError("frontier point has invalid lexicographic metrics")
    mean, p99, cvar, ear, top1, runtime, startup = map(float, raw)
    return mean, p99, cvar, -ear, -top1, runtime, startup


def exhaustive_frontier_oracle(units: list[dict[str, Any]],
                               measurements: dict[tuple[str, ...], dict[str, float]]
                               ) -> list[dict[str, Any]]:
    """Small-instance exact oracle used to test solver budget/tie semantics."""
    ordered = sorted(units, key=lambda row: row["unit_id"])
    best: dict[int, tuple[tuple[float, ...], tuple[str, ...], dict[str, Any]]] = {}
    for actions in itertools.product(*[
            sorted(unit["actions"], key=lambda row: row["action_id"]) for unit in ordered]):
        ids = tuple(action["action_id"] for action in actions)
        if ids not in measurements:
            raise GateError("exhaustive oracle lacks a validation measurement")
        point = {**measurements[ids],
                 "exact_serialized_bytes": sum(action["exact_serialized_bytes"]
                                               for action in actions),
                 "action_ids": list(ids)}
        key = _metric_tuple(point)
        current = best.get(point["exact_serialized_bytes"])
        if current is None or (key, ids) < (current[0], current[1]):
            best[point["exact_serialized_bytes"]] = (key, ids, point)
    return [best[budget][2] for budget in sorted(best)]


def _thresholds_sha256(contract: dict[str, Any]) -> str:
    frozen = {
        "absolute": contract["absolute_fidelity_constraints"],
        "promotion": contract["promotion"],
        "runtime_gates": contract["deployment_report_schema"]["gates"],
    }
    return hashlib.sha256(canonical_json(frozen)).hexdigest()


def _validate_action_assignment(deployment: dict[str, Any], open_state: dict[str, Any],
                                candidate_bytes: int, contract: dict[str, Any]) -> None:
    registry_path = Path(_required(deployment, "action_registry.path"))
    if sha256_file(registry_path) != open_state["action_registry_sha256"]:
        raise GateError("R30 action registry digest differs from the freeze")
    registry = load_json(registry_path)
    if registry.get("schema") != "wave5/exl3-action-registry/1":
        raise GateError("unexpected R30 action registry schema")
    available, legal_action_set_sha = _legal_action_menu(registry)
    assignment = deployment.get("action_assignment")
    total, candidate_assignment_sha = _assignment_details(assignment, available)
    if total != candidate_bytes:
        raise GateError("selected complete-action bytes do not equal checkpoint bytes")
    if candidate_assignment_sha != open_state["action_assignment_sha256"]:
        raise GateError("deployment assignment differs from the frozen candidate")
    frontier_path = Path(_required(deployment, "dual_frontier.path"))
    if sha256_file(frontier_path) != open_state["dual_frontier_sha256"]:
        raise GateError("dual-frontier digest differs from the freeze")
    frontier = load_json(frontier_path)
    required_semantics = contract["slq_dual_frontier"]["solver_semantics"]
    if (frontier.get("schema") != SCHEMA_FRONTIER
            or frontier.get("action_registry_sha256") != open_state["action_registry_sha256"]
            or frontier.get("legal_action_set_sha256") != legal_action_set_sha
            or frontier.get("solver_semantics") != required_semantics
            or frontier.get("thresholds_sha256") != _thresholds_sha256(contract)
            or frontier.get("selection_split") != "validation"):
        raise GateError("dual frontier is not bound to the legal menu/frozen semantics")
    reachable = frontier.get("reachable_budget_set")
    if (not isinstance(reachable, list) or not reachable
            or reachable != sorted(set(reachable))
            or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                       for value in reachable)
            or hashlib.sha256(canonical_json(reachable)).hexdigest()
            != frontier.get("reachable_budget_set_sha256")):
        raise GateError("dual frontier reachable-budget set/digest is invalid")
    points = frontier.get("points")
    if not isinstance(points, list) or [point.get("exact_serialized_bytes")
                                       for point in points] != reachable:
        raise GateError("dual frontier does not cover every reachable measured budget")
    for point in points:
        point_total, point_assignment_sha = _assignment_details(
            point.get("assignment"), available)
        if (point_total != point["exact_serialized_bytes"]
                or point_assignment_sha != point.get("assignment_sha256")):
            raise GateError("frontier point assignment/bytes are not legal")
        equivalents = point.get("equivalent_assignment_sha256")
        if (not isinstance(equivalents, list) or equivalents != sorted(set(equivalents))
                or point_assignment_sha not in equivalents
                or point_assignment_sha != equivalents[0]):
            raise GateError("frontier tie/equivalence class is noncanonical")
        measurement_path = Path(_required(point, "validation_measurement.path"))
        if sha256_file(measurement_path) != _required(
                point, "validation_measurement.sha256"):
            raise GateError("frontier validation measurement digest mismatch")
        measurement = load_json(measurement_path)
        if (measurement.get("schema") != "qwen38-wave5-frontier-measurement/1"
                or measurement.get("split") != "validation"
                or measurement.get("split_selection_sha256") != _registry_split(
                    load_json(CANONICAL_PREREG), "validation")[
                        "primary_v5_selection_sha256"]
                or measurement.get("assignment_sha256") != point_assignment_sha
                or measurement.get("fidelity_axis") != "body-only-shared-head"
                or measurement.get("metrics") != {
                    name: point[name] for name in
                    ("mean_kld", "p99_kld", "cvar1_kld", "ear", "top1_agreement")}):
            raise GateError("frontier point is not bound to validation-only evidence")
        _metric_tuple(point)
    candidate_points = [point for point in points
                        if point["exact_serialized_bytes"] == candidate_bytes]
    if (len(candidate_points) != 1
            or candidate_points[0]["assignment_sha256"] != candidate_assignment_sha
            or candidate_points[0].get("candidate_checkpoint_sha256")
            != open_state["complete_checkpoint_sha256"]):
        raise GateError("frozen candidate is absent from its exact frontier point")
    for point in points:
        point_metrics = _metric_tuple(point)[:5]
        for cheaper in points:
            if cheaper["exact_serialized_bytes"] >= point["exact_serialized_bytes"]:
                continue
            cheaper_metrics = _metric_tuple(cheaper)[:5]
            if (all(left <= right for left, right in zip(
                    cheaper_metrics, point_metrics))
                    and any(left < right for left, right in zip(
                    cheaper_metrics, point_metrics))):
                raise GateError("dual frontier contains a non-adjacent dominated point")


def evaluate_command(args: argparse.Namespace) -> None:
    """Apply the canonical frozen fidelity, action, byte, and deployment gates."""
    prereg, contract = trusted_contract()
    open_state_path = canonical_state_path(prereg, "test_open_state")
    open_state = _validated_open_state(prereg, contract)
    candidate, candidate_rows = _validated_report(
        args.candidate_report, "candidate", prereg, contract, open_state)
    served, served_rows = _validated_report(
        args.candidate_own_head_report, "candidate-own-head", prereg, contract, open_state)
    fresh, fresh_rows = _validated_report(
        args.f0_fresh_report, "F0-fresh", prereg, contract, open_state)
    shipped, shipped_rows = _validated_report(
        args.f0_report, "F0", prereg, contract, open_state)
    reports = ((candidate, "candidate"), (served, "candidate-own-head"),
               (fresh, "F0-fresh"), (shipped, "F0"))
    canonical_refs = None
    for report, label in reports:
        references = [(row["shard"], row["suite_manifest_sha256"],
                       row["suite_token_sha256"],
                       row["reference_capture_manifest_sha256"],
                       row["reference_capture_contract_sha256"],
                       row["candidate_capture_execution_sha256"])
                      for row in report["captures"]]
        if canonical_refs is None:
            canonical_refs = references
        elif references != canonical_refs:
            raise GateError(f"{label} does not share the identical suite/reference registry")
    if [row["candidate_capture_manifest_sha256"] for row in candidate["captures"]] != [
            row["candidate_capture_manifest_sha256"] for row in served["captures"]]:
        raise GateError("body-only and served-head reports do not use the same candidate capture")
    samples = contract["statistical_contract"]["bootstrap"]["samples"]
    seed = contract["statistical_contract"]["bootstrap"]["seed"]
    candidate_delta = paired_bootstrap(candidate_rows, fresh_rows, samples, seed)
    served_delta = paired_bootstrap(served_rows, fresh_rows, samples, seed)
    fresh_delta = paired_bootstrap(fresh_rows, shipped_rows, samples, seed)
    deployment = load_json(args.deployment_report)
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, operator: str, threshold: Any, passed: bool) -> None:
        checks.append({"name": name, "observed": observed, "operator": operator,
                       "threshold": threshold, "pass": bool(passed)})

    absolute = contract["absolute_fidelity_constraints"]
    noninferiority = contract["promotion"]["paired_noninferiority_candidate_minus_F0_fresh"]

    def fidelity_checks(axis: str, report: dict[str, Any],
                        delta: dict[str, Any]) -> None:
        metrics = report["metrics"]
        mean_high = report["cluster_bootstrap"]["ci95"]["mean_kld"][1]
        check(f"{axis}.absolute.mean_kld_ci95_high", mean_high, "<=",
              absolute["mean_kld_ci95_high_max"],
              mean_high <= absolute["mean_kld_ci95_high_max"])
        for metric, threshold_name, operator in (
                ("p99_kld", "p99_kld_max", "<="),
                ("cvar1_kld", "cvar1_kld_max", "<="),
                ("ear", "ear_min", ">="),
                ("top1_agreement", "top1_agreement_min", ">=")):
            threshold = absolute[threshold_name]
            passed = (metrics[metric] <= threshold if operator == "<="
                      else metrics[metric] >= threshold)
            check(f"{axis}.absolute.{metric}", metrics[metric], operator, threshold, passed)
        for metric, bound_name, endpoint, operator in (
                ("mean_kld", "mean_kld_ci95_high_max", 1, "<="),
                ("p99_kld", "p99_kld_ci95_high_max", 1, "<="),
                ("cvar1_kld", "cvar1_kld_ci95_high_max", 1, "<="),
                ("ear", "ear_ci95_low_min", 0, ">="),
                ("top1_agreement", "top1_agreement_ci95_low_min", 0, ">=")):
            observed = delta["ci95_delta"][metric][endpoint]
            threshold = noninferiority[bound_name]
            passed = observed <= threshold if operator == "<=" else observed >= threshold
            check(f"{axis}.noninferiority.{metric}", observed, operator, threshold, passed)

    fidelity_checks("body_only_shared_head", candidate, candidate_delta)
    fidelity_checks("served_candidate_head", served, served_delta)
    equivalence = contract["immutable_controls"]["practical_equivalence_F0_fresh_minus_F0"][
        "paired_ci95_must_lie_within"]
    for metric, bounds in equivalence.items():
        observed = fresh_delta["ci95_delta"][metric]
        check(f"F0-fresh_equivalence.{metric}", observed, "within", bounds,
              observed[0] >= bounds[0] and observed[1] <= bounds[1])
    if deployment.get("candidate_id") != open_state["candidate_id"]:
        raise GateError("deployment report names another candidate")
    if deployment.get("candidate_checkpoint_sha256") != open_state[
            "complete_checkpoint_sha256"]:
        raise GateError("deployment checkpoint differs from the freeze")
    candidate_bytes = deployment.get("exact_serialized_bytes")
    if candidate_bytes != open_state["exact_serialized_bytes"]:
        raise GateError("deployment bytes differ from the freeze")
    parts = deployment.get("serialized_byte_breakdown", {})
    part_names = ("trellis_payload_bytes", "sidecar_bytes", "header_alignment_bytes",
                  "other_checkpoint_bytes")
    part_values = [parts.get(name) for name in part_names]
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
               for value in part_values) or sum(part_values) != candidate_bytes:
        raise GateError("serialized byte breakdown is not exact")
    candidate_byte_manifest = _validated_byte_manifest(
        _required(deployment, "checkpoint_byte_manifest"),
        open_state["complete_checkpoint_sha256"],
        open_state["action_assignment_sha256"], candidate_bytes)
    if candidate_byte_manifest["serialized_byte_breakdown"] != parts:
        raise GateError("candidate byte summary differs from its hashed artifact manifest")
    _validate_action_assignment(deployment, open_state, candidate_bytes, contract)
    codec_baseline = load_json(args.codec_baseline)
    production_baseline = load_json(args.production_baseline)
    if (codec_baseline.get("candidate_id") != "F0-fresh"
            or codec_baseline.get("candidate_checkpoint_sha256")
            != open_state["F0_fresh_checkpoint_identity_sha256"]
            or codec_baseline.get("stock_action_sha256")
            != open_state["F0_fresh_stock_action_sha256"]):
        raise GateError("codec-exact baseline is not frozen F0-fresh")
    if (production_baseline.get("candidate_id") != "F0"
            or production_baseline.get("candidate_checkpoint_sha256")
            != open_state["F0_checkpoint_identity_sha256"]):
        raise GateError("production baseline is not frozen shipped F0")
    control_specs = (
        ("codec-exact", codec_baseline,
         open_state["F0_fresh_checkpoint_identity_sha256"],
         open_state["F0_fresh_stock_action_sha256"]),
        ("production", production_baseline,
         open_state["F0_checkpoint_identity_sha256"], None),
    )
    for label, baseline_receipt, checkpoint_sha, action_sha in control_specs:
        total = baseline_receipt.get("exact_serialized_bytes")
        breakdown = baseline_receipt.get("serialized_byte_breakdown", {})
        values = [breakdown.get(name) for name in part_names]
        if (not isinstance(total, int) or isinstance(total, bool) or total < 0
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           and value >= 0 for value in values)
                or sum(values) != total):
            raise GateError(f"{label} control byte receipt is not exact")
        byte_manifest = _validated_byte_manifest(
            _required(baseline_receipt, "checkpoint_byte_manifest"),
            checkpoint_sha, action_sha, total)
        if byte_manifest["serialized_byte_breakdown"] != breakdown:
            raise GateError(
                f"{label} byte summary differs from its hashed artifact manifest")
    baseline_bytes = codec_baseline["exact_serialized_bytes"]
    saving = baseline_bytes - candidate_bytes
    minimum_saving = max(1_048_576, math.ceil(baseline_bytes * 0.0001))
    equal_byte = (saving == 0 and candidate_delta["ci95_delta"]["mean_kld"][1] < 0
                  and candidate_delta["ci95_delta"]["ear"][0] > 0)
    check("promotion.byte_or_equal_fidelity", {"saving": saving, "equal_byte": equal_byte},
          "saving>=minimum or equal-byte superiority", minimum_saving,
          saving >= minimum_saving or equal_byte)
    head_evidence = deployment.get("served_head_evidence", {})
    if (head_evidence.get("body_only_report_sha256") != sha256_file(
            args.candidate_report)
            or head_evidence.get("served_head_report_sha256") != sha256_file(
                args.candidate_own_head_report)
            or head_evidence.get("candidate_head_sha256")
            != open_state["candidate_head_sha256"]
            or head_evidence.get("candidate_head_dtype")
            != open_state["candidate_head_dtype"]
            or head_evidence.get("candidate_head_shape")
            != open_state["candidate_head_shape"]):
        raise GateError("deployment conflates or does not bind both fidelity axes")
    deployment_contract = contract["deployment_report_schema"]
    gates = deployment_contract["gates"]
    baselines = {"codec_exact": codec_baseline, "production": production_baseline}
    runtime_receipt_hashes = {}
    open_state_sha = sha256_file(open_state_path)
    for profile_name, profile_spec in deployment_contract["profiles"].items():
        profile_pointer = deployment["profiles"][profile_name]
        profile = _load_profile_receipt(
            profile_pointer, profile_spec["route_id"], open_state["candidate_id"],
            open_state["complete_checkpoint_sha256"], open_state["action_registry_sha256"],
            open_state["candidate_head_sha256"], open_state_sha)
        baseline_id = "F0-fresh" if profile_name == "codec_exact" else "F0"
        baseline_checkpoint = (open_state["F0_fresh_checkpoint_identity_sha256"]
                               if profile_name == "codec_exact"
                               else open_state["F0_checkpoint_identity_sha256"])
        baseline_pointer = baselines[profile_name]["profile"]
        baseline_head_sha = baselines[profile_name].get("head_sha256")
        if not _is_sha256(baseline_head_sha):
            raise GateError(f"{profile_name} baseline head is not content-pinned")
        baseline = _load_profile_receipt(
            baseline_pointer, profile_spec["route_id"], baseline_id,
            baseline_checkpoint, None, baseline_head_sha, open_state_sha)
        runtime_receipt_hashes[profile_name] = {
            "candidate": profile_pointer["receipt_sha256"],
            "baseline": baseline_pointer["receipt_sha256"],
        }
        if _runtime_stack_identity(profile) != _runtime_stack_identity(baseline):
            raise GateError(f"{profile_name} candidate/control runtime stacks differ")
        if (profile_name == "codec_exact"
                and baseline.get("stock_action_sha256")
                != open_state["F0_fresh_stock_action_sha256"]):
            raise GateError("codec-exact baseline is not the frozen R30 stock action")
        for name, expected in profile_spec["required_environment"].items():
            if _required(profile, f"environment.{name}") != expected:
                raise GateError(f"{profile_name} environment differs at {name}")
        for name, expected in profile_spec["required_provenance"].items():
            if _required(profile, f"provenance.{name}") != expected:
                raise GateError(f"{profile_name} provenance differs at {name}")
        for field in deployment_contract["profile_numeric_nonnegative"]:
            if not _finite_nonnegative(_required(profile, field)):
                raise GateError(f"{profile_name} has invalid {field}")
        if profile.get("fallback_count") != 0 or profile.get("fallback_reasons") != []:
            raise GateError(f"{profile_name} used a fallback")
        graph = profile.get("graph_capture", {})
        if (graph.get("success") is not True or not _finite_nonnegative(graph.get("seconds"))
                or not graph.get("modes") or not graph.get("captured_shapes")):
            raise GateError(f"{profile_name} graph capture evidence is incomplete")
        probe = profile.get("decode_graph_probe", {})
        candidate_probe, bf16_probe = probe.get("candidate", {}), probe.get("bf16_control", {})
        if (candidate_probe.get("eager_self_repeat") != 32
                or candidate_probe.get("graph_self_repeat") != 32
                or bf16_probe.get("eager_self_repeat") != 32
                or bf16_probe.get("graph_self_repeat") != 32):
            raise GateError(f"{profile_name} decode graph probe lacks deterministic repeats")
        envelope = (float(candidate_probe["mean_abs_delta_logprob"])
                    <= 1.1 * float(bf16_probe["mean_abs_delta_logprob"])
                    and int(candidate_probe["exact_sequences"])
                    >= int(bf16_probe["exact_sequences"]) - 2)
        check(f"{profile_name}.decode_graph_envelope", candidate_probe,
              "within BF16 control envelope", bf16_probe, envelope)
        context_tokens = int(profile["max_context_tokens"])
        check(f"{profile_name}.context", context_tokens, ">=", gates["context_tokens_min"],
              context_tokens >= gates["context_tokens_min"])
        cold = float(profile["cold_start_seconds"])
        cold_limit = min(gates["cold_start_seconds_absolute_max"],
                         float(baseline["cold_start_seconds"])
                         * gates["cold_start_ratio_to_profile_baseline_max"])
        check(f"{profile_name}.cold_start", cold, "<=", cold_limit, cold <= cold_limit)
        for row in (deployment_contract["runtime_rows"]["prefill"]
                    + deployment_contract["runtime_rows"]["decode"]):
            row_id = row["row_id"]
            observed_row = _required(profile, f"performance_rows.{row_id}")
            baseline_row = _required(baseline, f"performance_rows.{row_id}")
            if row in deployment_contract["runtime_rows"]["decode"]:
                acceptance = observed_row.get("mtp_acceptance")
                if not isinstance(acceptance, (int, float)) or not 0 <= acceptance <= 1:
                    raise GateError(f"{profile_name}.{row_id} lacks valid MTP acceptance")
            observed = float(observed_row["tok_s"])
            floor = float(baseline_row["tok_s"]) * gates[
                "every_pp_and_tg_row_ratio_to_profile_baseline_min"]
            if profile_name == "production":
                floor = max(floor, gates["production_absolute"].get(
                    f"{row_id}_tok_s_min", -math.inf))
            check(f"{profile_name}.{row_id}", observed, ">=", floor, observed >= floor)
    result = {
        "schema": "qwen38-wave5-promotion-decision/2",
        "candidate_id": candidate["candidate_id"],
        "candidate_checkpoint_sha256": open_state["complete_checkpoint_sha256"],
        "candidate_head_sha256": open_state["candidate_head_sha256"],
        "contract_sha256": sha256_file(CANONICAL_CONTRACT),
        "prereg_sha256": sha256_file(CANONICAL_PREREG),
        "gate_sha256": sha256_file(Path(__file__)),
        "test_open_state_sha256": open_state_sha,
        "evidence_sha256": {
            "body_only_report": sha256_file(args.candidate_report),
            "served_head_report": sha256_file(args.candidate_own_head_report),
            "F0_fresh_report": sha256_file(args.f0_fresh_report),
            "F0_report": sha256_file(args.f0_report),
            "deployment_report": sha256_file(args.deployment_report),
            "codec_baseline": sha256_file(args.codec_baseline),
            "production_baseline": sha256_file(args.production_baseline),
            "runtime_profiles": runtime_receipt_hashes,
        },
        "thresholds_sha256": _thresholds_sha256(contract),
        "pass": all(row["pass"] for row in checks),
        "checks": checks,
    }
    output = canonical_state_path(prereg, "promotion_decision")
    write_json_exclusive(output, result)
    if not result["pass"]:
        raise GateError(f"promotion failed {sum(not row['pass'] for row in checks)} checks")

def _metric_test() -> dict[str, Any]:
    p = np.asarray([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]], dtype=np.float64)
    same = probability_metrics(p, p)
    assert np.array_equal(same["kld"], np.zeros(2))
    assert np.array_equal(same["ear"], np.ones(2))
    assert np.array_equal(same["jsd_bits"], np.zeros(2))
    q = np.asarray([[0.4, 0.4, 0.2], [0.8, 0.1, 0.1]], dtype=np.float64)
    changed = probability_metrics(p, q)
    assert np.all(changed["kld"] > 0)
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    assert np.allclose(changed["ear"], 1.0 - tv, rtol=0, atol=2e-15)
    forward = probability_metrics(np.asarray([[0.9, 0.1]]),
                                  np.asarray([[0.5, 0.5]]))["kld"][0]
    reverse = probability_metrics(np.asarray([[0.5, 0.5]]),
                                  np.asarray([[0.9, 0.1]]))["kld"][0]
    expected = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
    assert forward != reverse and math.isclose(forward, expected, abs_tol=1e-15)
    values = np.arange(1, 101, dtype=np.float64)
    assert np.quantile(values, 0.99, method="linear") == 99.01
    assert cvar_upper(values) == 100.0
    assert math.isclose(cvar_upper(np.asarray([3.0, 2.0, 1.0]), 0.5), 8 / 3)
    clusters = np.asarray(["d0", "d0", "d0", "d0", "d1", "d1"])
    contexts = np.asarray(["c0", "c0", "c1", "c1", "c0", "c0"])
    draws_a = hierarchical_cluster_draws(clusters, contexts, 8, 1)
    draws_b = hierarchical_cluster_draws(clusters, contexts, 8, 1)
    assert all(np.array_equal(a, b) for a, b in zip(draws_a, draws_b))
    assert np.array_equal(draws_a[0], np.asarray([0, 1, 2, 3, 0, 1, 2, 3]))
    for draw in draws_a:
        counts = {index: int(np.count_nonzero(draw == index)) for index in range(6)}
        assert len({counts[index] for index in (0, 1, 2, 3)}) == 1
        assert len({counts[index] for index in (4, 5)}) == 1
    try:
        probability_metrics(p, q[:, :2], expected_vocab=3)
    except GateError:
        pass
    else:
        raise AssertionError("truncated-vocabulary input did not fail closed")
    return {
        "kl_zero_positive_and_direction": True, "jsd_zero": True,
        "ear_equals_one_minus_tv": True, "p99_and_fractional_cvar": True,
        "method_seed_one_cluster_draw": True, "paired_cluster_grouping": True,
        "truncated_vocabulary_fail_closed": True,
    }


def _split_test(tmp: Path) -> dict[str, Any]:
    import contextlib
    import io

    tmp.mkdir(parents=True, exist_ok=True)
    h = "a" * 64
    calibration, validation, test = (tmp / name for name in
                                     ("calibration.json", "validation.json", "test.json"))
    for path, split in ((calibration, "calibration"), (validation, "validation"),
                        (test, "untouched_test")):
        path.write_bytes(canonical_json({"documents": [{
            "split": split, "document_id": f"{split}-document",
            "document_sha256": h, "contexts": []}]}))
    document_order, block_order, candidate_head = (
        tmp / "documents.json", tmp / "blocks.json", tmp / "head.safetensors")
    document_order.write_bytes(canonical_json(["d0"]))
    block_order.write_bytes(canonical_json(["b0"]))
    candidate_head.write_bytes(b"synthetic-head")
    contract_value = {
        "served_head_contract": {"dtype": "torch.bfloat16", "shape": [3, 2]},
        "absolute_fidelity_constraints": {"mean_kld_ci95_high_max": 1},
        "promotion": {"paired_noninferiority_candidate_minus_F0_fresh": {
            "mean_kld_ci95_high_max": 1}},
        "deployment_report_schema": {
            "gates": {"context_tokens_min": 1},
            "profiles": {
                "codec_exact": {"route_id": "codec-exact/all-trellis-stock-exl3"},
                "production": {
                    "route_id": "production/throughput-fp4-fp6-materialized"}}},
    }
    contract = tmp / "contract.json"
    contract.write_bytes(canonical_json(contract_value))
    action_registry, frontier = tmp / "actions.json", tmp / "frontier.json"
    action_registry.write_bytes(canonical_json({
        "schema": "wave5/exl3-action-registry/1", "units": [{
            "unit_id": "u0", "actions": [{
                "schema": "wave5/exl3-action/1", "action_id": "a0",
                "exact_serialized_bytes": 100}]}]}))
    assignment = [{"unit_id": "u0", "action_id": "a0"}]
    assignment_sha = hashlib.sha256(canonical_json(assignment)).hexdigest()
    frontier.write_bytes(canonical_json({"schema": SCHEMA_FRONTIER}))
    validation_selection = selection_sha256(validation, "validation")
    seeds = [1]
    budget = {
        "legal_encoder_evaluations": 1,
        "seed_order_sha256": hashlib.sha256(canonical_json(seeds)).hexdigest(),
        "document_order_sha256": sha256_file(document_order),
        "block_order_sha256": sha256_file(block_order)}
    attempts = tmp / "attempts.jsonl"
    for arm in ("F0-fresh", "R32-scale-path"):
        append_hash_log(attempts, {
            "operation": "validation-attempt", "arm_id": arm, "seed": 1,
            "attempt_ordinal": 0, "split": "validation",
            "split_selection_sha256": validation_selection,
            "seed_order_sha256": budget["seed_order_sha256"],
            "document_order_sha256": budget["document_order_sha256"],
            "block_order_sha256": budget["block_order_sha256"],
            "setting_sha256": h, "infrastructure_replacement": False})
    decision = tmp / "decision.json"
    decision.write_bytes(canonical_json({
        "schema": SCHEMA_SELECTION, "selected_arm_id": "R32-scale-path",
        "selection_split": "validation",
        "validation_selection_sha256": validation_selection,
        "validation_attempt_ledger_sha256": sha256_file(attempts),
        "dual_frontier_sha256": sha256_file(frontier),
        "action_assignment_sha256": assignment_sha,
        "candidate_checkpoint_sha256": h,
        "thresholds_sha256": _thresholds_sha256(contract_value)}))
    required = [
        "candidate_id", "selection_split", "untouched_test_accessed",
        "validation_manifest_sha256", "validation_selection_sha256",
        "complete_checkpoint_sha256", "candidate_head_path", "candidate_head_sha256",
        "candidate_head_dtype", "candidate_head_shape", "action_registry_path",
        "action_registry_sha256", "action_assignment_sha256",
        "dual_frontier_path", "dual_frontier_sha256", "exact_serialized_bytes",
        "validation_attempt_ledger_path", "validation_attempt_ledger_sha256",
        "selection_decision_path", "selection_decision_sha256", "ordered_seed_list",
        "matched_search_budget", "codec_exact_route_id", "production_route_id",
        "F0_checkpoint_identity_sha256", "F0_fresh_checkpoint_identity_sha256",
        "F0_fresh_stock_action_sha256", "selected_arm_id", "document_order_path",
        "block_order_path"]
    hash_fields = [
        "validation_manifest_sha256", "validation_selection_sha256",
        "complete_checkpoint_sha256", "candidate_head_sha256",
        "action_registry_sha256", "action_assignment_sha256", "dual_frontier_sha256",
        "validation_attempt_ledger_sha256", "selection_decision_sha256",
        "F0_checkpoint_identity_sha256", "F0_fresh_checkpoint_identity_sha256",
        "F0_fresh_stock_action_sha256"]
    carry = [
        "candidate_id", "complete_checkpoint_sha256", "candidate_head_sha256",
        "candidate_head_dtype", "candidate_head_shape", "action_registry_sha256",
        "action_assignment_sha256", "dual_frontier_sha256", "exact_serialized_bytes",
        "validation_attempt_ledger_sha256", "selection_decision_sha256",
        "ordered_seed_list", "matched_search_budget", "codec_exact_route_id",
        "production_route_id", "F0_checkpoint_identity_sha256",
        "F0_fresh_checkpoint_identity_sha256", "F0_fresh_stock_action_sha256"]
    prereg = tmp / "prereg.json"
    prereg.write_bytes(canonical_json({
        "split_registry": {
            "calibration": {
                "manifest_path": str(calibration), "manifest_sha256": sha256_file(calibration),
                "selection_sha256": selection_sha256(calibration, "calibration"),
                "access": [{"actors": ["researcher"], "phases": ["calibration"],
                            "purposes": ["fit"]}]},
            "validation": {
                "manifest_path": str(validation), "manifest_sha256": sha256_file(validation),
                "selection_sha256": validation_selection,
                "primary_v5_selection_sha256": validation_selection, "access": []},
            "untouched_test": {
                "manifest_path": str(test), "manifest_sha256": sha256_file(test),
                "selection_sha256": selection_sha256(test, "untouched_test"),
                "primary_v5_selection_sha256": selection_sha256(test, "untouched_test"),
                "access": []}},
        "arm_registry": [
            {"arm_id": "R32-scale-path", "test_eligible": True},
            {"arm_id": "F0-fresh"}],
        "candidate_freeze": {
            "required_manifest_fields": required, "sha256_fields": hash_fields,
            "carry_into_freeze": carry,
            "carry_into_open_state": carry + ["prereg_sha256", "contract_sha256"]}}))
    log, freeze, opened = tmp / "access.jsonl", tmp / "freeze.json", tmp / "open.json"
    allowed = argparse.Namespace(
        _test_mode=True, prereg=prereg, contract=contract, split="calibration",
        manifest=calibration, actor="researcher", phase="calibration", purpose="fit",
        access_log=log, event_nonce="allowed")
    with contextlib.redirect_stdout(io.StringIO()):
        authorize_command(allowed)
    denied = argparse.Namespace(
        _test_mode=True, prereg=prereg, contract=contract, split="untouched_test",
        manifest=test, actor="researcher", phase="validation", purpose="select",
        access_log=log, event_nonce="denied")
    try:
        authorize_command(denied)
    except GateError:
        pass
    else:
        raise AssertionError("direct untouched-test access did not fail closed")
    candidate = tmp / "candidate.json"
    candidate.write_bytes(canonical_json({
        "candidate_id": "synthetic", "selection_split": "validation",
        "untouched_test_accessed": False,
        "validation_manifest_sha256": sha256_file(validation),
        "validation_selection_sha256": validation_selection,
        "selected_arm_id": "R32-scale-path",
        "document_order_path": str(document_order), "block_order_path": str(block_order),
        "complete_checkpoint_sha256": h, "candidate_head_path": str(candidate_head),
        "candidate_head_sha256": sha256_file(candidate_head),
        "candidate_head_dtype": "torch.bfloat16", "candidate_head_shape": [3, 2],
        "action_registry_path": str(action_registry),
        "action_registry_sha256": sha256_file(action_registry),
        "action_assignment_sha256": assignment_sha,
        "dual_frontier_path": str(frontier), "dual_frontier_sha256": sha256_file(frontier),
        "exact_serialized_bytes": 100,
        "validation_attempt_ledger_path": str(attempts),
        "validation_attempt_ledger_sha256": sha256_file(attempts),
        "selection_decision_path": str(decision),
        "selection_decision_sha256": sha256_file(decision),
        "ordered_seed_list": seeds,
        "matched_search_budget": {"candidate": budget, "F0-fresh": budget},
        "codec_exact_route_id": "codec-exact/all-trellis-stock-exl3",
        "production_route_id": "production/throughput-fp4-fp6-materialized",
        "F0_checkpoint_identity_sha256": h, "F0_fresh_checkpoint_identity_sha256": h,
        "F0_fresh_stock_action_sha256": h}))
    freeze_args = argparse.Namespace(
        _test_mode=True, prereg=prereg, contract=contract,
        candidate_manifest=candidate, candidate_freeze=freeze)
    freeze_command(freeze_args)
    invalid = load_json(candidate)
    invalid["matched_search_budget"]["F0-fresh"]["legal_encoder_evaluations"] = 3
    try:
        _validate_candidate_manifest(
            invalid, load_json(prereg), contract_value, test_mode=True)
    except GateError:
        pass
    else:
        raise AssertionError("unequal control budget did not fail closed")
    open_args = argparse.Namespace(
        _test_mode=True, prereg=prereg, contract=contract, candidate_freeze=freeze,
        manifest=test, test_open_state=opened, access_log=log, event_nonce="open-once")
    with contextlib.redirect_stdout(io.StringIO()):
        open_test_command(open_args)
    open_args.event_nonce = "open-twice"
    try:
        open_test_command(open_args)
    except GateError:
        pass
    else:
        raise AssertionError("second untouched-test opening did not fail closed")
    rows = _read_hash_log(log)
    assert [row["outcome"] for row in rows] == ["granted", "denied", "granted", "denied"]
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            build_parser().parse_args(
                ["open-test", "--event-nonce", "x", "--open-state", str(tmp / "other")])
        except SystemExit:
            pass
        else:
            raise AssertionError("public open-test accepted an alternate state path")
    tampered = tmp / "tampered.jsonl"
    tampered.write_text(log.read_text(encoding="utf-8").replace(
        "\"outcome\":\"granted\"", "\"outcome\":\"denied\"", 1), encoding="utf-8")
    try:
        _read_hash_log(tampered)
    except GateError:
        pass
    else:
        raise AssertionError("tampered access hash chain was accepted")
    return {
        "canonical_split_hashes": True, "strict_candidate_and_head_freeze": True,
        "all_arm_equal_budget_enforced": True, "direct_test_access_denied": True,
        "owner_capability_opened_once": True, "alternate_state_path_rejected": True,
        "reused_capability_fail_closed": True, "tampered_hash_log_rejected": True}


def _adversarial_integration_test(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    h, candidate_head_sha = "a" * 64, "b" * 64
    prereg = {"split_registry": {"untouched_test": {
        "manifest_sha256": h, "selection_sha256": h,
        "primary_v5_selection_sha256": h, "primary_v5_contexts": 2}}}
    contract = {
        "vocabulary": {"size": 3},
        "benchmark_registry": {"primary": {"scored_positions_per_context": 2}},
        "reference_semantics": {
            "shared_head_sha256": h, "projector_source_sha256": h},
        "metric_contract": {"kld": {"vocab_chunk": 3}},
        "statistical_contract": {"bootstrap": {"samples": 4, "seed": 1}},
        "absolute_fidelity_constraints": {"mean_kld_ci95_high_max": 1},
        "promotion": {"paired_noninferiority_candidate_minus_F0_fresh": {
            "mean_kld_ci95_high_max": 1}},
        "deployment_report_schema": {"gates": {"context_tokens_min": 1}},
    }
    open_state = {
        "candidate_id": "candidate", "candidate_head_sha256": candidate_head_sha,
        "candidate_head_dtype": "torch.bfloat16", "candidate_head_shape": [3, 2]}
    base = {
        "schema": SCHEMA_REPORT, "candidate_id": "candidate", "control_id": "candidate",
        "split": "untouched_test", "split_manifest_sha256": h,
        "split_selection_sha256": h, "vocab_size": 3, "contexts": 2,
        "scored_positions": 4,
        "strata": ["code", "encyclopedic", "literary", "multilingual", "scientific"],
        "head_inclusive": False, "fidelity_axis": "body-only-shared-head",
        "candidate_head": {"sha256": None, "dtype": None, "shape": None},
        "reference": {
            "body_only": True, "full_vocabulary": True, "reference_head_sha256": h,
            "candidate_projection_head_sha256": h},
        "comparator": {
            "direction": "KL(BF16_reference || candidate)", "units": "natural-log nats",
            "projector_source_sha256": h, "vocab_chunk": 3, "accumulation": "float64",
            "two_pass": True, "capture_point": "post-final-norm",
            "scored_positions": "rows 0..2046"},
        "cluster_bootstrap": {
            "samples": 4, "seed": 1,
            "tail_interval_estimand": "global pooled per-position p99/CVaR1%"}}
    _validate_report_header(base, "candidate", prereg, contract, open_state, "positive")
    served = json.loads(json.dumps(base))
    served.update({
        "control_id": "candidate-own-head", "head_inclusive": True,
        "fidelity_axis": "served-candidate-head",
        "candidate_head": {"sha256": candidate_head_sha, "dtype": "torch.bfloat16",
                           "shape": [3, 2]}})
    served["reference"].update({
        "body_only": False, "candidate_projection_head_sha256": candidate_head_sha})
    _validate_report_header(
        served, "candidate-own-head", prereg, contract, open_state, "positive-served")
    mutations = {
        "wrong_kl_direction": ("comparator", "direction", "KL(candidate || BF16_reference)"),
        "top_k": ("reference", "full_vocabulary", False),
        "unlike_head": ("reference", "reference_head_sha256", candidate_head_sha),
        "invalid_tail_aggregation": (
            "cluster_bootstrap", "tail_interval_estimand", "average per-shard quantiles"),
    }
    for name, (section, field, value) in mutations.items():
        invalid = json.loads(json.dumps(base))
        invalid[section][field] = value
        try:
            _validate_report_header(
                invalid, "candidate", prereg, contract, open_state, name)
        except GateError:
            pass
        else:
            raise AssertionError(f"{name} report passed")
    for name, field, value in (
            ("unlike_suite", "split_manifest_sha256", candidate_head_sha),
            ("zero_split_hash", "split_selection_sha256", "0" * 64),
            ("unlike_context_set", "contexts", 1),
            ("unlike_window", "scored_positions", 3)):
        invalid = json.loads(json.dumps(base))
        invalid[field] = value
        try:
            _validate_report_header(
                invalid, "candidate", prereg, contract, open_state, name)
        except GateError:
            pass
        else:
            raise AssertionError(f"{name} report passed")
    conflated = json.loads(json.dumps(base))
    conflated["fidelity_axis"] = "served-candidate-head"
    try:
        _validate_report_header(
            conflated, "candidate", prereg, contract, open_state, "conflated")
    except GateError:
        pass
    else:
        raise AssertionError("body/served-head conflation passed")
    clusters = np.asarray(["a", "a", "b", "b"])
    contexts = np.asarray(["a0", "a0", "b0", "b0"])
    kld = np.asarray([0.0, 0.0, 0.0, 100.0])
    ear = np.asarray([1.0, 1.0, 1.0, 0.0])
    top1 = np.asarray([1.0, 1.0, 1.0, 0.0])
    global_bootstrap = bootstrap_summary(kld, ear, top1, clusters, contexts, 8, 1)
    assert set(global_bootstrap["ci95"]) == set(METRIC_NAMES)
    context_table = _context_metric_table(kld, ear, top1, clusters, contexts)[0]
    assert not math.isclose(
        summarize_metrics(kld, ear, top1)["p99_kld"],
        float(context_table[:, 1].mean()), rel_tol=0, abs_tol=1e-12)
    units = [
        {"unit_id": "u0", "actions": [
            {"action_id": "a", "exact_serialized_bytes": 1},
            {"action_id": "b", "exact_serialized_bytes": 2}]},
        {"unit_id": "u1", "actions": [
            {"action_id": "c", "exact_serialized_bytes": 2},
            {"action_id": "d", "exact_serialized_bytes": 1}]}]
    common = {"mean_kld": 1.0, "p99_kld": 2.0, "cvar1_kld": 3.0,
              "ear": 0.9, "top1_agreement": 0.9,
              "runtime_seconds": 1.0, "startup_seconds": 1.0}
    measurements = {
        ("a", "c"): {**common, "mean_kld": 0.8},
        ("a", "d"): common,
        ("b", "c"): {**common, "mean_kld": 1.2},
        ("b", "d"): common}
    oracle = exhaustive_frontier_oracle(units, measurements)
    assert [(row["exact_serialized_bytes"], row["action_ids"]) for row in oracle] == [
        (2, ["a", "d"]), (3, ["a", "c"]), (4, ["b", "c"])]
    byte_artifact = tmp / "checkpoint.bin"
    byte_artifact.write_bytes(b"abc")
    byte_manifest = {
        "schema": SCHEMA_BYTE_MANIFEST,
        "checkpoint_identity_sha256": h,
        "action_sha256": candidate_head_sha,
        "files": [{
            "relative_path": "checkpoint.bin", "path": str(byte_artifact),
            "sha256": sha256_file(byte_artifact), "exact_bytes": 3,
            "category": "trellis_payload"}],
        "exact_serialized_bytes": 3,
        "serialized_byte_breakdown": {
            "trellis_payload_bytes": 3, "sidecar_bytes": 0,
            "header_alignment_bytes": 0, "other_checkpoint_bytes": 0},
    }
    byte_manifest["content_sha256"] = content_sha256(byte_manifest)
    byte_manifest_path = tmp / "byte-manifest.json"
    byte_manifest_path.write_bytes(canonical_json(byte_manifest))
    byte_pointer = {"path": str(byte_manifest_path),
                    "sha256": sha256_file(byte_manifest_path)}
    _validated_byte_manifest(byte_pointer, h, candidate_head_sha, 3)
    byte_artifact.write_bytes(b"abd")
    try:
        _validated_byte_manifest(byte_pointer, h, candidate_head_sha, 3)
    except GateError:
        pass
    else:
        raise AssertionError("tampered checkpoint bytes passed manifest validation")
    changed_contract = json.loads(json.dumps(contract))
    before = _thresholds_sha256(contract)
    changed_contract["absolute_fidelity_constraints"]["mean_kld_ci95_high_max"] = 2
    assert _thresholds_sha256(changed_contract) != before
    evidence = tmp / "evidence.json"
    evidence.write_bytes(canonical_json({"ok": True}))
    environment = {"PROFILE": "throughput"}
    environment_receipt = tmp / "environment.json"
    environment_receipt.write_bytes(canonical_json({"environment": environment}))
    runtime = {
        "schema": SCHEMA_RUNTIME, "candidate_id": "candidate",
        "candidate_checkpoint_sha256": h, "route_id": "route",
        "action_registry_sha256": h, "environment": environment,
        "provenance": {
            "image_digest": f"sha256:{h}", "vllm_revision": "v", "build_revision": "b",
            "gpu_uuid": "gpu", "gpu_name": "name", "driver_version": "driver",
            "attention_backend": "TRITON_ATTN", "kv_cache_dtype": "fp8_e4m3",
            "mtp_tokens": 6, "cudagraph_mode": "FULL_DECODE_ONLY",
            "profile": "throughput", "candidate_checkpoint_sha256": h,
            "route_id": "route", "head_sha256": candidate_head_sha,
            "test_open_state_sha256": h,
            "effective_environment_sha256": hashlib.sha256(
                canonical_json(environment)).hexdigest()},
        "effective_environment_receipt": {
            "path": str(environment_receipt), "sha256": sha256_file(environment_receipt)},
        "startup_receipt": {"path": str(evidence), "sha256": sha256_file(evidence)},
        "runtime_receipt": {"path": str(evidence), "sha256": sha256_file(evidence)},
        "raw_evidence": {"path": str(evidence), "sha256": sha256_file(evidence)}}
    runtime_path = tmp / "runtime.json"
    runtime_path.write_bytes(canonical_json(runtime))
    pointer = {"receipt_path": str(runtime_path), "receipt_sha256": sha256_file(runtime_path)}
    _load_profile_receipt(pointer, "route", "candidate", h, h, candidate_head_sha, h)
    stale = json.loads(json.dumps(runtime))
    stale["environment"]["PROFILE"] = "changed-after-run"
    runtime_path.write_bytes(canonical_json(stale))
    pointer["receipt_sha256"] = sha256_file(runtime_path)
    try:
        _load_profile_receipt(pointer, "route", "candidate", h, h, candidate_head_sha, h)
    except GateError:
        pass
    else:
        raise AssertionError("stale runtime/environment provenance passed")
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            build_parser().parse_args([
                "evaluate", "--candidate-report", "a", "--f0-fresh-report", "b",
                "--f0-report", "c", "--deployment-report", "d", "--codec-baseline", "e",
                "--production-baseline", "f"])
        except SystemExit:
            pass
        else:
            raise AssertionError("evaluate accepted no served-head report")
    return {
        "positive_body_and_served_headers": True,
        "wrong_kl_direction_rejected": True, "top_k_rejected": True,
        "unlike_suite_head_window_context_rejected": True,
        "zero_split_hash_rejected": True, "invalid_tail_aggregation_rejected": True,
        "body_served_head_conflation_rejected": True,
        "global_position_tail_bootstrap": True,
        "exhaustive_frontier_oracle_matches_enumeration": True,
        "actual_checkpoint_byte_manifest_rehashed": True,
        "stale_runtime_environment_rejected": True,
        "results_dependent_threshold_edit_detected": True,
        "served_head_evidence_required_by_evaluate": True}


def self_test_command(args: argparse.Namespace) -> None:
    import tempfile
    metrics = _metric_test()
    with tempfile.TemporaryDirectory(prefix="wave5-fidelity-test-") as directory:
        test_root = Path(directory)
        access = _split_test(test_root / "access")
        adversarial = _adversarial_integration_test(test_root / "adversarial")
    print(json.dumps({"status": "pass", "metrics": metrics, "split_access": access,
                      "adversarial_integration": adversarial}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "synthetic-metrics", help="non-promotable normalized-vector metric exercise")
    synthetic.add_argument("--input", type=Path, action="append", required=True)
    synthetic.add_argument("--candidate-id", required=True)
    synthetic.add_argument("--rows-out", type=Path, required=True)
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--bootstrap-samples", type=int, default=100)
    synthetic.add_argument("--bootstrap-seed", type=int, default=1)
    synthetic.add_argument("--worst-contexts", type=int, default=20)
    synthetic.set_defaults(func=compute_command)

    replay = subparsers.add_parser(
        "replay", help="capture-bound full-vocabulary body-only or served-head replay")
    replay.add_argument("--reference-root", type=Path, required=True)
    replay.add_argument("--candidate-root", type=Path, required=True)
    replay.add_argument("--suite-root", type=Path, required=True)
    replay.add_argument("--head", type=Path, required=True)
    replay.add_argument(
        "--role", choices=("candidate", "candidate-own-head", "F0", "F0-fresh"),
        required=True)
    replay.add_argument("--candidate-head", type=Path)
    replay.add_argument("--split", choices=("validation", "untouched_test"), required=True)
    replay.add_argument("--candidate-id")
    replay.add_argument("--expected-checkpoint-identity-sha256")
    replay.add_argument("--expected-candidate-head-sha256")
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--rows-out", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(func=replay_command)

    compare = subparsers.add_parser(
        "compare", help="paired clustered comparison of promotable row payloads")
    compare.add_argument("--candidate-rows", type=Path, required=True)
    compare.add_argument("--control-rows", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=compare_command)

    authorize = subparsers.add_parser(
        "authorize-split", help="hash-check and canonically log non-test split access")
    authorize.add_argument("--split", choices=("calibration", "validation"), required=True)
    authorize.add_argument("--actor", required=True)
    authorize.add_argument("--phase", required=True)
    authorize.add_argument("--purpose", required=True)
    authorize.add_argument("--event-nonce", required=True)
    authorize.set_defaults(func=authorize_command)

    freeze = subparsers.add_parser(
        "freeze-candidate", help="write the one canonical immutable candidate freeze")
    freeze.add_argument("--candidate-manifest", type=Path, required=True)
    freeze.set_defaults(func=freeze_command)

    open_test = subparsers.add_parser(
        "open-test", help="capability-open the canonical untouched test exactly once")
    open_test.add_argument("--event-nonce", required=True)
    open_test.set_defaults(func=open_test_command)

    evaluate = subparsers.add_parser(
        "evaluate", help="apply the canonical fidelity/action/deployment promotion gate")
    evaluate.add_argument("--candidate-report", type=Path, required=True)
    evaluate.add_argument("--candidate-own-head-report", type=Path, required=True)
    evaluate.add_argument("--f0-fresh-report", type=Path, required=True)
    evaluate.add_argument("--f0-report", type=Path, required=True)
    evaluate.add_argument("--deployment-report", type=Path, required=True)
    evaluate.add_argument("--codec-baseline", type=Path, required=True)
    evaluate.add_argument("--production-baseline", type=Path, required=True)
    evaluate.set_defaults(func=evaluate_command)

    self_test = subparsers.add_parser("self-test", help="run deterministic metric/access tests")
    self_test.set_defaults(func=self_test_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (GateError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
