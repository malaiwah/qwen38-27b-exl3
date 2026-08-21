#!/usr/bin/env python3
"""R37 exact-byte frontier over measured complete stock-EXL3 actions.

The multiple-choice DP is exact only over the declared screened action menu.
Single-group full-vocabulary validation marginals screen assignments. A candidate
is selected only when its whole checkpoint was freshly rebuilt sequentially and
then directly measured on validation, because downstream activations are
path-dependent. Untouched-test evidence is rejected in Phase B.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn

REGISTRY_SCHEMA = "wave5/r37-measured-action-registry/1"
OUTPUT_SCHEMA = "wave5/r37-screening-frontier/1"
ACTION_SCHEMA = "wave5/exl3-action/1"
ASSIGNMENT_SCHEMA = "qwen38-wave5-primal-assignment/1"
METRIC_PROTOCOL = "qwen38-kld-method-v5/body-only-shared-bf16-head"
VALIDATION_SELECTION_SHA256 = "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de"
SUITE_MANIFEST_SHA256 = "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019"
SUITE_TOKEN_SHA256 = "510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88"
SHARED_BF16_HEAD_SHA256 = "25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff"
FOUNDATION = {
    "data_manifest_file_sha256": "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37",
    "data_manifest_content_sha256": "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2",
    "split_manifest_file_sha256": "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc",
    "split_manifest_content_sha256": "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e",
    "fisher_manifest_file_sha256": "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a",
    "fisher_manifest_content_sha256": "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237",
    "exl3_action_harness_sha256": "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d",
    "exl3_action_schema_sha256": "275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8",
    "exl3_extension_binary_sha256": "e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e",
    "stock_control_receipt_sha256": "894a8992c84aef5dd2091e71ea8f73b405fe5803c0a360de5ec552d5b325cb43",
    "fidelity_gate_sha256": "f4fc059c03331905dca6ad7b0ad4ba0e6af515897e2fc90dfd82f1ce0e8e8482",
    "fidelity_contract_sha256": "e8e1d47694038bbec4aa6f4a4554c4b53e549d2082d87e07353e5d8d16a66783",
    "fidelity_prereg_sha256": "75a81665c75761767a7c71d58f4d59c446a13d3d7b164c5a8b9da9070388a784",
}
REQUIRED_CONTROLS = {
    "stock_hydrated_current_recipe",
    "local_eda_known_negative",
    "uniform_k6",
    "best_module_k5_k6_mixing",
}
METRICS = (
    "context_macro_mean_kld", "mean_kld_ci95_high", "p99_kld", "cvar1_kld",
    "ear", "top1_agreement", "runtime_seconds", "startup_seconds",
)
DEFAULT_THRESHOLDS = {
    "mean_kld_ci95_high_max": 0.012,
    "p99_kld_max": 0.12,
    "cvar1_kld_max": 0.25,
    "ear_min": 0.97,
    "top1_agreement_min": 0.97,
}


class FrontierError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise FrontierError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {path}: {exc}")
    return obj(value, str(path))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def obj(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def seq(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be an array")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty string")
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{label} must be finite")
    return float(value)


def sha256(value: Any, label: str) -> str:
    value = text(value, label)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or value == "0" * 64:
        fail(f"{label} must be a nonzero lowercase SHA256")
    return value


def exact_keys(value: dict[str, Any], required: set[str], optional: set[str], label: str) -> None:
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        fail(f"{label}: missing={sorted(missing)} unknown={sorted(unknown)}")


def validate_metrics(value: Any, label: str) -> dict[str, Any]:
    row = obj(value, label)
    exact_keys(row, set(METRICS) | {"worst_contexts"}, set(), label)
    out: dict[str, Any] = {key: number(row[key], f"{label}.{key}") for key in METRICS}
    out["worst_contexts"] = seq(row["worst_contexts"], f"{label}.worst_contexts")
    for key in (
        "context_macro_mean_kld", "mean_kld_ci95_high", "p99_kld",
        "cvar1_kld", "runtime_seconds", "startup_seconds",
    ):
        if out[key] < 0:
            fail(f"{label}.{key} must be nonnegative")
    for key in ("ear", "top1_agreement"):
        if not 0 <= out[key] <= 1:
            fail(f"{label}.{key} must be in [0,1]")
    if out["mean_kld_ci95_high"] < out["context_macro_mean_kld"]:
        fail(f"{label}.mean_kld_ci95_high is below the mean estimate")
    return out


def validate_measurement(value: Any, label: str) -> dict[str, Any]:
    row = obj(value, label)
    required = {
        "protocol_id", "selection_split", "selection_sha256", "full_vocabulary",
        "report_sha256", "metrics", "candidate_checkpoint_sha256",
        "candidate_payload_sha256", "candidate_capture_sha256",
        "reference_checkpoint_sha256", "reference_capture_sha256",
        "suite_manifest_sha256", "suite_token_sha256",
        "shared_bf16_head_sha256", "direction",
    }
    exact_keys(row, required, {"notes", "report_path", "candidate_capture_path", "reference_capture_path"}, label)
    if row["protocol_id"] != METRIC_PROTOCOL or row["selection_split"] != "validation" or row["full_vocabulary"] is not True:
        fail(f"{label} must be validation-only full-vocabulary method-of-record evidence")
    if row["direction"] != "KL(BF16 reference || candidate)":
        fail(f"{label}.direction is not the method-of-record KL direction")
    if row["suite_manifest_sha256"] != SUITE_MANIFEST_SHA256:
        fail(f"{label}.suite_manifest_sha256 mismatch")
    if row["suite_token_sha256"] != SUITE_TOKEN_SHA256:
        fail(f"{label}.suite_token_sha256 mismatch")
    if row["shared_bf16_head_sha256"] != SHARED_BF16_HEAD_SHA256:
        fail(f"{label}.shared_bf16_head_sha256 mismatch")
    for hash_name in (
        "selection_sha256", "report_sha256", "candidate_checkpoint_sha256",
        "candidate_payload_sha256", "candidate_capture_sha256",
        "reference_checkpoint_sha256", "reference_capture_sha256",
    ):
        sha256(row[hash_name], f"{label}.{hash_name}")
    return {**row, "metrics": validate_metrics(row["metrics"], f"{label}.metrics")}


def verify_file(path_value: Any, expected_sha256: Any, label: str) -> Path:
    path = Path(text(path_value, f"{label}.path"))
    if not path.is_file():
        fail(f"{label} file does not exist: {path}")
    if file_sha256(path) != sha256(expected_sha256, f"{label}.sha256"):
        fail(f"{label} file SHA256 mismatch")
    return path
def verify_measurement_files(measurement: dict[str, Any], label: str) -> None:
    required = {
        "report_path": "report_sha256",
        "candidate_capture_path": "candidate_capture_sha256",
        "reference_capture_path": "reference_capture_sha256",
    }
    for path_name, hash_name in required.items():
        verify_file(measurement.get(path_name), measurement.get(hash_name), f"{label}.{path_name}")




def verify_byte_manifest(path_value: Any, expected_sha256: Any, expected_total: int,
                         checkpoint_sha256: str, label: str) -> None:
    manifest = load_json(verify_file(path_value, expected_sha256, label))
    if manifest.get("schema") != "qwen38-wave5-checkpoint-byte-manifest/1":
        fail(f"{label} has wrong schema")
    if manifest.get("checkpoint_identity_sha256") != checkpoint_sha256:
        fail(f"{label} belongs to another checkpoint")
    files = seq(manifest.get("files"), f"{label}.files")
    relative_paths = [obj(row, f"{label}.files[{i}]").get("relative_path") for i, row in enumerate(files)]
    if not files or relative_paths != sorted(set(relative_paths)):
        fail(f"{label} file order is not canonical")
    total = 0
    for i, row_value in enumerate(files):
        row = obj(row_value, f"{label}.files[{i}]")
        artifact = verify_file(row.get("path"), row.get("sha256"), f"{label}.files[{i}]")
        if row.get("exact_bytes") != artifact.stat().st_size:
            fail(f"{label}.files[{i}] exact bytes are stale")
        total += artifact.stat().st_size
    if manifest.get("exact_serialized_bytes") != total or total != expected_total:
        fail(f"{label} total does not equal stat-sized checkpoint bytes")


def validate_action(value: Any, label: str, skip_deep: bool = False) -> dict[str, Any]:
    action = obj(value, label)
    minimal = {
        "schema", "action_id", "unit", "K", "codebook", "sign_scale_transform",
        "target", "curvature_correction", "viterbi_refinement", "serialized",
        "runtime", "evidence",
    }
    missing_minimal = minimal - action.keys()
    if action.get("schema") != ACTION_SCHEMA or missing_minimal:
        fail(f"{label} is not a complete action shell; missing={sorted(missing_minimal)}")
    if action["K"] not in (4, 5, 6):
        fail(f"{label}.K must be one of the R37 stock K4/K5/K6 controls")
    serialized = obj(action["serialized"], f"{label}.serialized")
    total = integer(serialized.get("buffer_bytes"), f"{label}.serialized.buffer_bytes")
    buffers = obj(serialized.get("buffers"), f"{label}.serialized.buffers")
    if not {"suh", "svh", "trellis"}.issubset(buffers):
        fail(f"{label} lacks stock serialized buffers")
    summed = 0
    for name, buffer_value in buffers.items():
        if name not in {"suh", "svh", "trellis", "mcg", "mul1"}:
            fail(f"{label} uses forbidden new-format buffer {name}")
        summed += integer(obj(buffer_value, f"{label}.buffers.{name}").get("bytes"), f"{label}.buffers.{name}.bytes")
    if summed != total:
        fail(f"{label} buffer sum {summed} != buffer_bytes {total}")
    runtime = obj(action["runtime"], f"{label}.runtime")
    integer(runtime.get("sidecar_bytes"), f"{label}.runtime.sidecar_bytes")
    if skip_deep:
        return action

    complete_fields = {
        "schema", "action_id", "unit", "K", "codebook",
        "sign_scale_transform", "target", "curvature_correction", "viterbi_refinement",
        "curvature", "callback", "runtime", "seed",
        "split_manifest_sha256", "split_manifest_content_sha256",
        "split_selections", "split_disjointness",
        "source_tensor_sha256", "source_revision", "source_layout",
        "encoder_repo", "encoder_commit", "encoder_tree_sha1", "encoder_version",
        "serialized", "hashes", "qualifications", "metric_contract", "evidence",
    }
    exact_keys(action, complete_fields, set(), label)
    if action["codebook"] not in {"mcg", "mul1", "3inst"}:
        fail(f"{label}.codebook is not a stock marker")
    if action["source_layout"] not in {"out_in", "in_out"}:
        fail(f"{label}.source_layout is invalid")
    if (
        action["encoder_repo"] != "turboderp-org/exllamav3"
        or action["encoder_commit"] != "5f3c537ca9d89893d771256f5c43c93656553fbb"
        or action["encoder_tree_sha1"] != "ffc0a1d31c25d4174b96adffef3727f12a7056c7"
        or action["encoder_version"] != "1.4.2"
    ):
        fail(f"{label} is not pinned to the frozen stock encoder")
    if action["split_manifest_sha256"] != FOUNDATION["split_manifest_file_sha256"]:
        fail(f"{label}.split_manifest_sha256 is not the frozen file hash")
    if action["split_manifest_content_sha256"] != FOUNDATION["split_manifest_content_sha256"]:
        fail(f"{label}.split_manifest_content_sha256 is not the frozen content hash")
    validation = obj(obj(action["split_selections"], f"{label}.split_selections").get("validation"), f"{label}.split_selections.validation")
    if validation.get("selection_sha256") != VALIDATION_SELECTION_SHA256:
        fail(f"{label}.split_selections.validation mismatch")
    if obj(action["metric_contract"], f"{label}.metric_contract").get("protocol_id") != METRIC_PROTOCOL:
        fail(f"{label}.metric_contract mismatch")
    sha256(action["source_tensor_sha256"], f"{label}.source_tensor_sha256")
    hashes = obj(action["hashes"], f"{label}.hashes")
    for hash_name in ("action_identity_sha256", "payload_sha256", "source_basis_reconstruction_sha256"):
        sha256(hashes.get(hash_name), f"{label}.hashes.{hash_name}")

    unit = obj(action["unit"], f"{label}.unit")
    if unit.get("granularity") not in {"topology_group", "module", "tensor", "shard"}:
        fail(f"{label}.unit.granularity is invalid")
    if unit.get("topology") not in {"mlp", "gdn", "full_attention", "lm_head", "mtp"}:
        fail(f"{label}.unit.topology is invalid")
    if unit.get("role") not in {
        "gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj",
        "qkv_proj", "z_proj", "out_proj", "in_proj_a", "in_proj_b",
        "lm_head", "mtp_dense", "other_dense",
    }:
        fail(f"{label}.unit.role is invalid")
    keys = seq(unit.get("tensor_keys"), f"{label}.unit.tensor_keys")
    if not keys or len(keys) != len(set(keys)) or not all(isinstance(x, str) and x for x in keys):
        fail(f"{label}.unit.tensor_keys must be nonempty and unique")
    for recipe_name in ("sign_scale_transform", "target", "curvature_correction", "viterbi_refinement"):
        recipe = obj(action[recipe_name], f"{label}.{recipe_name}")
        required_recipe = {"recipe_id", "kind", "parameters", "implementation", "strength"}
        if set(recipe) != required_recipe or recipe["implementation"] not in {"stock", "stock-with-encode-callback"}:
            fail(f"{label}.{recipe_name} is not a complete legal recipe")
    for name, buffer_value in buffers.items():
        buffer_row = obj(buffer_value, f"{label}.buffers.{name}")
        if set(buffer_row) != {"dtype", "shape", "bytes", "sha256"}:
            fail(f"{label}.buffers.{name} is incomplete")
        if buffer_row["dtype"] not in {"float16", "int16", "int32"}:
            fail(f"{label}.buffers.{name}.dtype is invalid")
        seq(buffer_row["shape"], f"{label}.buffers.{name}.shape")
        sha256(buffer_row["sha256"], f"{label}.buffers.{name}.sha256")
    dense_law = obj(serialized.get("dense_byte_law"), f"{label}.serialized.dense_byte_law")
    if dense_law.get("passes") is not True or dense_law.get("expected_bytes") != total:
        fail(f"{label} fails the stock dense byte law")
    if runtime.get("route_id") != "codec-exact/all-trellis-stock-exl3" or runtime.get("graph_capturable") is not True:
        fail(f"{label} is not a graph-capturable codec-exact Phase-B action")
    promoted = obj(action["evidence"], f"{label}.evidence").get("promoted_kld")
    if promoted is not None:
        promoted = obj(promoted, f"{label}.evidence.promoted_kld")
        split = obj(promoted.get("promoted_split"), f"{label}.evidence.promoted_kld.promoted_split")
        if promoted.get("protocol_id") != METRIC_PROTOCOL or promoted.get("full_vocabulary") is not True or split.get("name") != "validation":
            fail(f"{label} carries non-validation KLD evidence")
    return action


@dataclass(frozen=True)
class Choice:
    unit_id: str
    action_id: str
    exact_bytes: int
    delta: tuple[float, ...]
    action: dict[str, Any]


@dataclass(frozen=True)
class Label:
    exact_bytes: int
    metrics: tuple[float, ...]
    assignment: tuple[tuple[str, str], ...]
    equivalent_assignment_sha256s: tuple[str, ...] = ()


def metric_tuple(metrics: dict[str, float]) -> tuple[float, ...]:
    return tuple(metrics[key] for key in METRICS)


def tuple_metrics(values: Iterable[float]) -> dict[str, float]:
    return dict(zip(METRICS, values, strict=True))


def add_tuple(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def metric_lex_key(metrics: tuple[float, ...]) -> tuple[float, ...]:
    m = tuple_metrics(metrics)
    return (
        m["context_macro_mean_kld"],
        m["p99_kld"],
        m["cvar1_kld"],
        -m["ear"],
        -m["top1_agreement"],
        m["runtime_seconds"],
        m["startup_seconds"],
    )


def lex_key(metrics: tuple[float, ...], assignment: tuple[tuple[str, str], ...]) -> tuple[Any, ...]:
    return metric_lex_key(metrics) + (digest(dict(assignment)),)


def validate_assignment(value: dict[str, Any], units: dict[str, list[Choice]], label: str) -> dict[str, str]:
    if set(value) != set(units):
        fail(f"{label} must select every unit exactly once")
    out: dict[str, str] = {}
    for unit_id, action_value in sorted(value.items()):
        action_id = text(action_value, f"{label}.{unit_id}")
        if action_id not in {c.action_id for c in units[unit_id]}:
            fail(f"{label}.{unit_id} selects unknown action {action_id}")
        out[unit_id] = action_id
    return out


def action_bytes(assignment: dict[str, str], units: dict[str, list[Choice]]) -> int:
    return sum(next(c.exact_bytes for c in units[u] if c.action_id == a) for u, a in assignment.items())


def normalize_registry(registry: dict[str, Any], skip_action_schema: bool = False) -> dict[str, Any]:
    required = {
        "schema", "foundation", "selection_split", "selection_sha256", "baseline",
        "fixed_checkpoint_bytes", "fixed_byte_components", "action_rows", "controls",
        "measured_assignments", "thresholds", "route_gates", "interaction_evidence",
        "qmm_diagnostic",
    }
    exact_keys(registry, required, {"notes"}, "registry")
    if registry["schema"] != REGISTRY_SCHEMA:
        fail(f"registry.schema must be {REGISTRY_SCHEMA}")
    foundation = obj(registry["foundation"], "registry.foundation")
    for key, expected in FOUNDATION.items():
        if foundation.get(key) != expected:
            fail(f"registry.foundation.{key} does not match the frozen pin")
    if registry["selection_split"] != "validation":
        fail("registry selection_split must be validation")
    selection_sha = sha256(registry["selection_sha256"], "registry.selection_sha256")
    if selection_sha != VALIDATION_SELECTION_SHA256:
        fail("registry.selection_sha256 is not the frozen R29/R31 validation projection")
    baseline = validate_measurement(registry["baseline"], "registry.baseline")
    if baseline["selection_sha256"] != selection_sha:
        fail("baseline selection hash differs from registry")
    if not skip_action_schema:
        verify_measurement_files(baseline, "registry.baseline")
    fixed = integer(registry["fixed_checkpoint_bytes"], "registry.fixed_checkpoint_bytes")
    components = obj(registry["fixed_byte_components"], "registry.fixed_byte_components")
    if sum(integer(v, f"fixed component {k}") for k, v in components.items()) != fixed:
        fail("fixed byte components do not sum exactly")

    units: dict[str, list[Choice]] = {}
    action_ids: set[str] = set()
    memberships: dict[str, tuple[str, ...]] = {}
    for i, value in enumerate(seq(registry["action_rows"], "registry.action_rows")):
        label = f"registry.action_rows[{i}]"
        row = obj(value, label)
        exact_keys(row, {"action", "exact_serialized_bytes", "matched_stock_measurement", "candidate_measurement", "search_evaluations", "source_lane"}, {"shapley_evidence_ids", "notes", "replacement_proof"}, label)
        action = validate_action(row["action"], f"{label}.action", skip_action_schema)
        action_id = text(action["action_id"], f"{label}.action.action_id")
        if action_id in action_ids:
            fail(f"duplicate action_id {action_id}")
        action_ids.add(action_id)
        unit = action["unit"]
        unit_id = text(unit["unit_id"], f"{label}.action.unit.unit_id")
        member = tuple(sorted(unit["tensor_keys"]))
        if memberships.setdefault(unit_id, member) != member:
            fail(f"unit {unit_id} changes tensor membership")
        exact = integer(row["exact_serialized_bytes"], f"{label}.exact_serialized_bytes")
        expected = action["serialized"]["buffer_bytes"] + action["runtime"]["sidecar_bytes"]
        if exact != expected:
            fail(f"{label} exact bytes {exact} != R30 buffers+sidecars {expected}")
        stock = validate_measurement(row["matched_stock_measurement"], f"{label}.matched_stock_measurement")
        candidate = validate_measurement(row["candidate_measurement"], f"{label}.candidate_measurement")
        if stock["selection_sha256"] != selection_sha or candidate["selection_sha256"] != selection_sha:
            fail(f"{label} selection hash mismatch")
        integer(row["search_evaluations"], f"{label}.search_evaluations", 1)
        if not skip_action_schema and candidate["candidate_payload_sha256"] != action["hashes"]["payload_sha256"]:
            fail(f"{label} candidate measurement is not bound to the action payload")
        if not skip_action_schema:
            verify_measurement_files(stock, f"{label}.matched_stock_measurement")
            verify_measurement_files(candidate, f"{label}.candidate_measurement")
            for lineage_name in (
                "candidate_checkpoint_sha256", "candidate_payload_sha256",
                "reference_checkpoint_sha256", "reference_capture_sha256",
            ):
                if stock[lineage_name] != baseline[lineage_name]:
                    fail(f"{label} matched stock does not share baseline {lineage_name}")
            proof = obj(row.get("replacement_proof"), f"{label}.replacement_proof")
            exact_keys(
                proof,
                {"unit_id", "non_target_files_equal", "changed_buffer_names", "report_path", "report_sha256"},
                set(),
                f"{label}.replacement_proof",
            )
            if proof["unit_id"] != unit_id or proof["non_target_files_equal"] is not True:
                fail(f"{label} is not proven to be a single-group replacement")
            verify_file(proof["report_path"], proof["report_sha256"], f"{label}.replacement_proof")
        delta = tuple(candidate["metrics"][k] - stock["metrics"][k] for k in METRICS)
        units.setdefault(unit_id, []).append(Choice(unit_id, action_id, exact, delta, action))
    if not units:
        fail("registry has no action rows")
    tensor_owners: dict[str, str] = {}
    for unit_id, members in memberships.items():
        for member in members:
            if member in tensor_owners:
                fail(f"tensor {member} belongs to two units; couple it into one legal unit")
            tensor_owners[member] = unit_id
    for unit_id, choices in units.items():
        if not any(all(abs(x) <= 1e-15 for x in choice.delta) for choice in choices):
            fail(f"unit {unit_id} lacks a remeasured zero-delta stock action")
        if not skip_action_schema and {choice.action["K"] for choice in choices} != {4, 5, 6}:
            fail(f"unit {unit_id} lacks a complete actual-stock K4/K5/K6 curve")
        units[unit_id] = sorted(choices, key=lambda c: c.action_id)
    units = dict(sorted(units.items()))

    controls_in = obj(registry["controls"], "registry.controls")
    if REQUIRED_CONTROLS - controls_in.keys():
        fail(f"registry controls missing {sorted(REQUIRED_CONTROLS - controls_in.keys())}")
    controls = {name: validate_assignment(obj(value, f"control {name}"), units, f"control {name}") for name, value in controls_in.items()}
    if not skip_action_schema:
        if any(
            next(choice.action["K"] for choice in units[unit_id] if choice.action_id == action_id) != 6
            for unit_id, action_id in controls["uniform_k6"].items()
        ):
            fail("uniform_k6 control contains a non-K6 action")
        control_hashes = {digest(assignment) for assignment in controls.values()}
        if len(control_hashes) != len(REQUIRED_CONTROLS):
            fail("mandatory controls are not four distinct frozen assignments")
    thresholds = obj(registry["thresholds"], "registry.thresholds")
    for key, expected in DEFAULT_THRESHOLDS.items():
        if number(thresholds.get(key), f"threshold {key}") != expected:
            fail(f"threshold {key} differs from frozen {expected}")
    gates = obj(registry["route_gates"], "registry.route_gates")
    exact_keys(gates, {"required_route_id", "graph_capturable", "runtime_ratio_to_baseline_min", "max_startup_seconds", "min_context_tokens", "no_fallback"}, set(), "registry.route_gates")
    if gates["required_route_id"] != "codec-exact/all-trellis-stock-exl3" or gates["graph_capturable"] is not True or gates["no_fallback"] is not True:
        fail("Phase-B route must be graph-capturable, no-fallback codec-exact stock EXL3")
    if number(gates["runtime_ratio_to_baseline_min"], "runtime_ratio_to_baseline_min") != 0.95:
        fail("runtime ratio gate differs from frozen 0.95")
    if number(gates["max_startup_seconds"], "max_startup_seconds") != 360:
        fail("startup gate differs from frozen 360 seconds")
    if integer(gates["min_context_tokens"], "min_context_tokens", 1) != 238400:
        fail("context gate differs from frozen 238400 tokens")

    interactions = seq(registry["interaction_evidence"], "registry.interaction_evidence")
    seen_interactions: set[str] = set()
    for i, value in enumerate(interactions):
        label = f"interaction[{i}]"
        row = obj(value, label)
        exact_keys(row, {"interaction_id", "unit_id", "action_ids", "method", "report_sha256", "selection_sha256", "high_sensitivity_or_fused", "result"}, set(), label)
        iid = text(row["interaction_id"], f"{label}.interaction_id")
        if iid in seen_interactions:
            fail(f"duplicate interaction {iid}")
        seen_interactions.add(iid)
        if row["unit_id"] not in units or row["high_sensitivity_or_fused"] is not True:
            fail(f"{label} is not a targeted legal unit interaction")
        if row["method"] not in {"direct-factorial", "targeted-shapley"}:
            fail(f"{label} method unsupported")
        sha256(row["report_sha256"], f"{label}.report_sha256")
        if row["selection_sha256"] != selection_sha:
            fail(f"{label} selection hash mismatch")
        for aid in seq(row["action_ids"], f"{label}.action_ids"):
            if aid not in action_ids:
                fail(f"{label} references unknown action {aid}")

    measured = []
    for i, value in enumerate(seq(registry["measured_assignments"], "registry.measured_assignments")):
        label = f"measured_assignments[{i}]"
        row = obj(value, label)
        exact_keys(row, {"assignment_id", "assignment", "exact_checkpoint_bytes", "checkpoint_byte_manifest_path", "checkpoint_byte_manifest_sha256", "checkpoint_sha256", "sequential_rebuild", "route_validation", "validation_measurement"}, {"category", "notes"}, label)
        assignment = validate_assignment(obj(row["assignment"], f"{label}.assignment"), units, f"{label}.assignment")
        total = integer(row["exact_checkpoint_bytes"], f"{label}.exact_checkpoint_bytes")
        if total != fixed + action_bytes(assignment, units):
            fail(f"{label} exact checkpoint byte sum mismatch")
        sha256(row["checkpoint_byte_manifest_sha256"], f"{label}.byte_manifest_sha256")
        sha256(row["checkpoint_sha256"], f"{label}.checkpoint_sha256")
        if not skip_action_schema:
            verify_byte_manifest(
                row["checkpoint_byte_manifest_path"],
                row["checkpoint_byte_manifest_sha256"],
                total,
                row["checkpoint_sha256"],
                f"{label}.checkpoint_byte_manifest",
            )
        if row["sequential_rebuild"] is not True:
            fail(f"{label} is not a fresh sequential rebuild")
        route = obj(row["route_validation"], f"{label}.route_validation")
        exact_keys(route, {"route_id", "graph_capturable", "no_fallback", "context_tokens", "runtime_ratio_to_baseline"}, set(), f"{label}.route_validation")
        measurement = validate_measurement(row["validation_measurement"], f"{label}.validation_measurement")
        if measurement["selection_sha256"] != selection_sha:
            fail(f"{label} selection hash mismatch")
        if measurement["candidate_checkpoint_sha256"] != row["checkpoint_sha256"]:
            fail(f"{label} validation measurement is not bound to the rebuilt checkpoint")
        if not skip_action_schema:
            verify_measurement_files(measurement, f"{label}.validation_measurement")
        measured.append({**row, "assignment": assignment, "validation_measurement": measurement})
    if measured and not skip_action_schema:
        measured_assignment_hashes = {digest(row["assignment"]) for row in measured}
        unmeasured_controls = [
            name for name in REQUIRED_CONTROLS
            if digest(controls[name]) not in measured_assignment_hashes
        ]
        if unmeasured_controls:
            fail(f"measured registry lacks direct whole-checkpoint controls {sorted(unmeasured_controls)}")

    qmm = obj(registry["qmm_diagnostic"], "registry.qmm_diagnostic")
    exact_keys(
        qmm,
        {"effective_bits_gap", "report_sha256", "action_id", "scope"},
        set(),
        "registry.qmm_diagnostic",
    )
    qmm_gap = number(qmm["effective_bits_gap"], "registry.qmm_diagnostic.effective_bits_gap")
    if qmm_gap < 0:
        fail("registry.qmm_diagnostic.effective_bits_gap must be nonnegative")
    sha256(qmm["report_sha256"], "registry.qmm_diagnostic.report_sha256")
    qmm_action = text(qmm["action_id"], "registry.qmm_diagnostic.action_id")
    if not skip_action_schema and qmm_action not in action_ids:
        fail("registry.qmm_diagnostic.action_id is outside the legal menu")
    if not skip_action_schema and qmm["scope"] != "whole-module-k5-k6-residual":
        fail("registry.qmm_diagnostic.scope is not eligible for R38")

    return {
        "foundation": foundation,
        "selection_sha256": selection_sha,
        "baseline": baseline,
        "fixed_checkpoint_bytes": fixed,
        "fixed_byte_components": components,
        "units": units,
        "controls": controls,
        "thresholds": thresholds,
        "route_gates": gates,
        "interactions": interactions,
        "measured": measured,
        "qmm_diagnostic": qmm,
        "registry_sha256": digest(registry),
        "legal_action_set_sha256": digest(
            sorted((c.unit_id, c.action_id, c.exact_bytes) for choices in units.values() for c in choices)
        ),
    }


def solve_dp(normalized: dict[str, Any]) -> list[Label]:
    initial = Label(
        normalized["fixed_checkpoint_bytes"],
        metric_tuple(normalized["baseline"]["metrics"]),
        (),
    )
    states: dict[int, list[Label]] = {initial.exact_bytes: [initial]}
    for unit_id, choices in normalized["units"].items():
        nxt: dict[int, list[Label]] = {}
        for tied_labels in states.values():
            for prior in tied_labels:
                for choice in choices:
                    candidate = Label(
                        prior.exact_bytes + choice.exact_bytes,
                        add_tuple(prior.metrics, choice.delta),
                        prior.assignment + ((unit_id, choice.action_id),),
                    )
                    tied = nxt.get(candidate.exact_bytes)
                    if tied is None:
                        nxt[candidate.exact_bytes] = [candidate]
                        continue
                    candidate_key = metric_lex_key(candidate.metrics)
                    incumbent_key = metric_lex_key(tied[0].metrics)
                    if candidate_key < incumbent_key:
                        nxt[candidate.exact_bytes] = [candidate]
                    elif candidate_key == incumbent_key:
                        tied.append(candidate)
        states = nxt
    output: list[Label] = []
    for exact_bytes in sorted(states):
        tied = states[exact_bytes]
        hashes = tuple(sorted({digest(dict(label.assignment)) for label in tied}))
        chosen = min(tied, key=lambda label: digest(dict(label.assignment)))
        output.append(Label(chosen.exact_bytes, chosen.metrics, chosen.assignment, hashes))
    return output


def brute_oracle(normalized: dict[str, Any]) -> list[Label]:
    units = list(normalized["units"].items())
    baseline = metric_tuple(normalized["baseline"]["metrics"])
    states: dict[int, list[Label]] = {}
    for choices in itertools.product(*(menu for _, menu in units)):
        metrics = baseline
        assignment = ()
        for (unit_id, _), choice in zip(units, choices, strict=True):
            metrics = add_tuple(metrics, choice.delta)
            assignment += ((unit_id, choice.action_id),)
        candidate = Label(normalized["fixed_checkpoint_bytes"] + sum(c.exact_bytes for c in choices), metrics, assignment)
        tied = states.get(candidate.exact_bytes)
        if tied is None or metric_lex_key(candidate.metrics) < metric_lex_key(tied[0].metrics):
            states[candidate.exact_bytes] = [candidate]
        elif metric_lex_key(candidate.metrics) == metric_lex_key(tied[0].metrics):
            tied.append(candidate)
    output: list[Label] = []
    for exact_bytes in sorted(states):
        tied = states[exact_bytes]
        hashes = tuple(sorted({digest(dict(label.assignment)) for label in tied}))
        chosen = min(tied, key=lambda label: digest(dict(label.assignment)))
        output.append(Label(chosen.exact_bytes, chosen.metrics, chosen.assignment, hashes))
    return output


def passes(metrics: dict[str, Any], route: dict[str, Any], thresholds: dict[str, Any], gates: dict[str, Any]) -> bool:
    return (
        metrics["mean_kld_ci95_high"] <= thresholds["mean_kld_ci95_high_max"]
        and metrics["p99_kld"] <= thresholds["p99_kld_max"]
        and metrics["cvar1_kld"] <= thresholds["cvar1_kld_max"]
        and metrics["ear"] >= thresholds["ear_min"]
        and metrics["top1_agreement"] >= thresholds["top1_agreement_min"]
        and route.get("runtime_ratio_to_baseline", 0) >= gates["runtime_ratio_to_baseline_min"]
        and metrics["startup_seconds"] <= gates["max_startup_seconds"]
        and route.get("route_id") == gates["required_route_id"]
        and route.get("graph_capturable") is gates["graph_capturable"]
        and route.get("no_fallback") is gates["no_fallback"]
        and route.get("context_tokens", 0) >= gates["min_context_tokens"]
    )


def frequencies(assignment: dict[str, str], normalized: dict[str, Any]) -> dict[str, Any]:
    by_k: dict[str, int] = {}
    by_role: dict[str, dict[str, int]] = {}
    by_depth: dict[str, dict[str, int]] = {}
    for unit_id, action_id in assignment.items():
        action = next(c.action for c in normalized["units"][unit_id] if c.action_id == action_id)
        k, role = str(action["K"]), str(action["unit"]["role"])
        depth = "none" if action["unit"].get("layer_index") is None else str(action["unit"]["layer_index"])
        by_k[k] = by_k.get(k, 0) + 1
        by_role.setdefault(role, {})[k] = by_role.setdefault(role, {}).get(k, 0) + 1
        by_depth.setdefault(depth, {})[k] = by_depth.setdefault(depth, {}).get(k, 0) + 1
    return {"by_K": dict(sorted(by_k.items())), "by_role_and_K": dict(sorted(by_role.items())), "by_depth_and_K": dict(sorted(by_depth.items(), key=lambda x: (-1 if x[0] == "none" else int(x[0]))))}


def build_output(registry: dict[str, Any], n: dict[str, Any]) -> dict[str, Any]:
    screen = solve_dp(n)
    measured_all: list[dict[str, Any]] = []
    for row in n["measured"]:
        key = lex_key(metric_tuple(row["validation_measurement"]["metrics"]), tuple(sorted(row["assignment"].items())))
        measured_all.append({"row": row, "key": key, "assignment_sha256": digest(row["assignment"])})
    measured_by_bytes: dict[int, dict[str, Any]] = {}
    for item in measured_all:
        row = item["row"]
        incumbent = measured_by_bytes.get(row["exact_checkpoint_bytes"])
        if incumbent is None or item["key"] < incumbent["key"]:
            measured_by_bytes[row["exact_checkpoint_bytes"]] = item
    measured = [measured_by_bytes[key] for key in sorted(measured_by_bytes)]
    feasible = [x for x in measured_all if passes(x["row"]["validation_measurement"]["metrics"], x["row"]["route_validation"], n["thresholds"], n["route_gates"])]
    selected = min(feasible, key=lambda x: (x["row"]["exact_checkpoint_bytes"], x["key"])) if feasible else None

    screen_points = []
    for label in screen:
        m, assignment = tuple_metrics(label.metrics), dict(label.assignment)
        ah = digest(assignment)
        screen_points.append({
            "exact_serialized_bytes": label.exact_bytes, "mean_kld": m["context_macro_mean_kld"],
            "mean_kld_ci95_high": m["mean_kld_ci95_high"], "p99_kld": m["p99_kld"],
            "cvar1_kld": m["cvar1_kld"], "ear": m["ear"],
            "top1_agreement": m["top1_agreement"], "runtime_seconds": m["runtime_seconds"],
            "startup_seconds": m["startup_seconds"], "assignment": assignment,
            "assignment_sha256": ah, "equivalent_assignment_sha256": list(label.equivalent_assignment_sha256s),
            "validation_measurement": {"kind": "additive-screen-from-direct-single-group-full-vocabulary-validation-marginals", "promotable": False, "selection_split": "validation"},
        })
    points = []
    for item in measured:
        row, m = item["row"], item["row"]["validation_measurement"]["metrics"]
        points.append({
            "assignment_id": row["assignment_id"], "exact_serialized_bytes": row["exact_checkpoint_bytes"],
            "mean_kld": m["context_macro_mean_kld"], "mean_kld_ci95_high": m["mean_kld_ci95_high"],
            "p99_kld": m["p99_kld"], "cvar1_kld": m["cvar1_kld"], "ear": m["ear"], "top1_agreement": m["top1_agreement"],
            "runtime_seconds": m["runtime_seconds"], "startup_seconds": m["startup_seconds"],
            "assignment": row["assignment"], "assignment_sha256": item["assignment_sha256"],
            "equivalent_assignment_sha256": [item["assignment_sha256"]], "validation_measurement": row["validation_measurement"],
            "checkpoint_sha256": row["checkpoint_sha256"], "checkpoint_byte_manifest_sha256": row["checkpoint_byte_manifest_sha256"],
            "passes_frozen_gates": item in feasible,
        })
    selected_out = None
    if selected:
        row = selected["row"]
        selected_out = {
            "schema": ASSIGNMENT_SCHEMA, "claim": "best measured registered validation candidate; not a global optimum",
            "assignment_id": row["assignment_id"], "assignment": row["assignment"],
            "assignment_sha256": selected["assignment_sha256"], "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_byte_manifest_sha256": row["checkpoint_byte_manifest_sha256"],
            "exact_serialized_bytes": row["exact_checkpoint_bytes"], "validation_measurement": row["validation_measurement"],
            "route_validation": row["route_validation"], "action_frequencies": frequencies(row["assignment"], n),
        }
    controls = {}
    measured_map = {digest(x["row"]["assignment"]): x for x in measured_all}
    for name, assignment in n["controls"].items():
        ah = digest(assignment)
        controls[name] = {"assignment": assignment, "assignment_sha256": ah, "exact_action_bytes": action_bytes(assignment, n["units"]), "exact_checkpoint_bytes": n["fixed_checkpoint_bytes"] + action_bytes(assignment, n["units"]), "measured_assignment_id": measured_map.get(ah, {}).get("row", {}).get("assignment_id")}
    output = {
        "schema": OUTPUT_SCHEMA, "status": "selected" if selected_out else "no-measured-assignment-passes-frozen-gates",
        "claim_scope": "best measured registered validation candidate; DP exact only over the complete declared screened stock-format menu",
        "foundation": FOUNDATION, "action_registry_sha256": n["registry_sha256"],
        "legal_action_set_sha256": n["legal_action_set_sha256"],
        "reachable_budget_set_sha256": digest([x["exact_serialized_bytes"] for x in screen_points]),
        "thresholds_sha256": digest(n["thresholds"]), "selection_split": "validation",
        "selection_sha256": n["selection_sha256"],
        "solver_semantics": {
            "algorithm": "exact multiple-choice DP keyed by exact integer checkpoint bytes",
            "legal_set": "one complete stock-format action per declared legal topology/fused unit",
            "metric_model": "baseline plus direct single-group validation full-vocabulary marginals; screen only",
            "ties": "lexicographic metrics then canonical assignment SHA256",
            "interactions": "enter allocation only as directly remeasured complete grouped actions",
            "selection": "minimum bytes among fresh sequential whole-checkpoint validation measurements passing every frozen gate",
            "global_optimum_claim": False,
        },
        "fixed_checkpoint_bytes": n["fixed_checkpoint_bytes"], "fixed_byte_components": n["fixed_byte_components"],
        "points": screen_points,
        "screened_exact_byte_frontier": screen_points,
        "fresh_sequential_measured_points": points,
        "controls": controls,
        "interaction_evidence": n["interactions"], "selected_assignment": selected_out,
        "residual_gate": {
            "proceed_r38": False,
            "reason": "A non-promotable R37 screening artifact cannot open R38; a separate frozen measured whole-module gate is required.",
            "exact_target_bytes": None,
            "qmm_innovation_gap_effective_bits": n["qmm_diagnostic"]["effective_bits_gap"],
            "qmm_report_sha256": n["qmm_diagnostic"]["report_sha256"],
        },
        "registry_counts": {"units": len(n["units"]), "actions": sum(len(x) for x in n["units"].values()), "reachable_exact_budgets": len(screen_points), "fresh_sequential_measured_assignments": len(n["measured"]), "passing_measured_assignments": len(feasible)},
        "limitations": [
            "Additive direct-marginal DP cannot represent downstream path dependence.",
            "Only fresh sequential whole-checkpoint validation rows can be selected.",
            "No per-tile, selector-map, entropy-coded, or new-format action is admitted.",
            "The untouched test is not read or used in Phase B.",
        ],
    }
    output["output_content_sha256"] = digest(output)
    return output


def fake_measurement(metrics: dict[str, float], tag: str) -> dict[str, Any]:
    tag_hash = hashlib.sha256(tag.encode()).hexdigest()
    return {
        "protocol_id": METRIC_PROTOCOL,
        "selection_split": "validation",
        "selection_sha256": VALIDATION_SELECTION_SHA256,
        "full_vocabulary": True,
        "report_sha256": tag_hash,
        "metrics": metrics,
        "candidate_checkpoint_sha256": tag_hash,
        "candidate_payload_sha256": tag_hash,
        "candidate_capture_sha256": hashlib.sha256((tag + "-capture").encode()).hexdigest(),
        "reference_checkpoint_sha256": hashlib.sha256(b"reference-checkpoint").hexdigest(),
        "reference_capture_sha256": hashlib.sha256(b"reference-capture").hexdigest(),
        "suite_manifest_sha256": SUITE_MANIFEST_SHA256,
        "suite_token_sha256": SUITE_TOKEN_SHA256,
        "shared_bf16_head_sha256": SHARED_BF16_HEAD_SHA256,
        "direction": "KL(BF16 reference || candidate)",
    }


def fake_action(unit: str, action_id: str, k: int, exact: int) -> dict[str, Any]:
    return {
        "schema": ACTION_SCHEMA, "action_id": action_id,
        "unit": {"unit_id": unit, "granularity": "tensor", "topology": "mlp", "role": "down_proj", "tensor_keys": [f"{unit}.weight"], "layer_index": 0, "shard_id": None, "fused_group": None, "output_splits": []},
        "K": k, "codebook": "mcg",
        "sign_scale_transform": {"recipe_id": "stock"}, "target": {"recipe_id": "stock"},
        "curvature_correction": {"recipe_id": "stock"}, "viterbi_refinement": {"recipe_id": "stock"},
        "serialized": {"buffer_bytes": exact, "buffers": {"suh": {"bytes": 0}, "svh": {"bytes": 0}, "trellis": {"bytes": exact - 4}, "mcg": {"bytes": 4}}},
        "runtime": {"route_id": "codec-exact/all-trellis-stock-exl3", "graph_capturable": True, "sidecar_bytes": 0},
        "evidence": {"local_metrics": {}, "promoted_kld": None},
    }


def self_test() -> dict[str, Any]:
    base = {"context_macro_mean_kld": 0.001, "mean_kld_ci95_high": 0.0011, "p99_kld": 0.01, "cvar1_kld": 0.02, "ear": 0.99, "top1_agreement": 0.99, "runtime_seconds": 1.0, "startup_seconds": 2.0, "worst_contexts": []}
    choices = [("u0", "u0.k5", 5, 100, 0.001), ("u0", "u0.k5.tie", 5, 100, 0.001), ("u0", "u0.k6", 6, 120, 0.0008), ("u1", "u1.k5", 5, 80, 0.001), ("u1", "u1.k6", 6, 100, 0.0007)]
    rows = []
    for unit, aid, k, cost, mean in choices:
        candidate = {**base, "context_macro_mean_kld": mean}
        rows.append({"action": fake_action(unit, aid, k, cost), "exact_serialized_bytes": cost, "matched_stock_measurement": fake_measurement(base, aid + "-stock"), "candidate_measurement": fake_measurement(candidate, aid + "-candidate"), "search_evaluations": 1, "source_lane": "self-test"})
    selector = rows[0]["matched_stock_measurement"]["selection_sha256"]
    registry = {
        "schema": REGISTRY_SCHEMA, "foundation": FOUNDATION, "selection_split": "validation", "selection_sha256": selector,
        "baseline": fake_measurement(base, "baseline"), "fixed_checkpoint_bytes": 20, "fixed_byte_components": {"header": 20}, "action_rows": rows,
        "controls": {"stock_hydrated_current_recipe": {"u0": "u0.k5", "u1": "u1.k5"}, "local_eda_known_negative": {"u0": "u0.k5", "u1": "u1.k6"}, "uniform_k6": {"u0": "u0.k6", "u1": "u1.k6"}, "best_module_k5_k6_mixing": {"u0": "u0.k6", "u1": "u1.k5"}},
        "measured_assignments": [], "thresholds": DEFAULT_THRESHOLDS,
        "route_gates": {"required_route_id": "codec-exact/all-trellis-stock-exl3", "graph_capturable": True, "runtime_ratio_to_baseline_min": 0.95, "max_startup_seconds": 360, "min_context_tokens": 238400, "no_fallback": True},
        "interaction_evidence": [],
        "qmm_diagnostic": {
            "effective_bits_gap": 0.05,
            "report_sha256": hashlib.sha256(b"qmm").hexdigest(),
            "action_id": "self-test-qmm",
            "scope": "self-test",
        },
    }
    n = normalize_registry(registry, skip_action_schema=True)
    dp, oracle = solve_dp(n), brute_oracle(n)
    if dp != oracle:
        fail("DP differs from independent Cartesian-product oracle")
    if [x.exact_bytes for x in dp] != [200, 220, 240]:
        fail("wrong reachable exact-byte set")
    if len(dp[0].equivalent_assignment_sha256s) != 2:
        fail("equal-metric assignment equivalence class was collapsed")
    if build_output(registry, n)["selected_assignment"] is not None:
        fail("screen-only assignment was promoted")
    measured_registry = json.loads(json.dumps(registry))
    whole_measurement = fake_measurement(base, "whole-checkpoint")
    measured_registry["measured_assignments"] = [{
        "assignment_id": "oracle-stock",
        "assignment": {"u0": "u0.k5", "u1": "u1.k5"},
        "exact_checkpoint_bytes": 200,
        "checkpoint_byte_manifest_path": "self-test-byte-manifest.json",
        "checkpoint_byte_manifest_sha256": hashlib.sha256(b"byte-manifest").hexdigest(),
        "checkpoint_sha256": whole_measurement["candidate_checkpoint_sha256"],
        "sequential_rebuild": True,
        "route_validation": {
            "route_id": "codec-exact/all-trellis-stock-exl3",
            "graph_capturable": True,
            "no_fallback": True,
            "context_tokens": 238400,
            "runtime_ratio_to_baseline": 1.0,
        },
        "validation_measurement": whole_measurement,
    }]
    measured_normalized = normalize_registry(measured_registry, skip_action_schema=True)
    measured_output = build_output(measured_registry, measured_normalized)
    if measured_output["selected_assignment"]["assignment_id"] != "oracle-stock":
        fail("fresh sequential measured assignment was not selected")
    if measured_output["residual_gate"]["proceed_r38"] is not False:
        fail("non-promotable screening output opened R38")
    try:
        validate_action(fake_action("illegal", "illegal", 4, 4), "strict-illegal", skip_deep=False)
    except FrontierError:
        pass
    else:
        fail("incomplete action passed strict production normalization")
    inconsistent_metrics = {**base, "context_macro_mean_kld": 0.5, "mean_kld_ci95_high": 0.001}
    try:
        validate_metrics(inconsistent_metrics, "inconsistent-ci")
    except FrontierError:
        pass
    else:
        fail("inconsistent mean/CI passed")
    def expect_reject(mutator: Any) -> None:
        broken = json.loads(json.dumps(registry))
        mutator(broken)
        try:
            normalize_registry(broken, skip_action_schema=True)
        except FrontierError:
            return
        fail("negative fixture was accepted")

    expect_reject(lambda value: value["action_rows"][0].__setitem__("exact_serialized_bytes", 101))
    expect_reject(lambda value: value["action_rows"][0]["action"]["serialized"]["buffers"].__setitem__("selector", {"bytes": 0}))
    expect_reject(lambda value: value["action_rows"][0]["candidate_measurement"].__setitem__("selection_split", "untouched_test"))
    return {
        "status": "pass",
        "oracle_assignments": 6,
        "reachable_exact_budgets": [200, 220, 240],
        "dp_equals_cartesian_oracle": True,
        "screen_only_not_promoted": True,
        "exact_byte_sum_checked": True,
        "new_format_rejected": True,
        "untouched_test_rejected": True,
        "fresh_sequential_selection_checked": True,
        "tie_equivalence_checked": True,
        "strict_action_gate_checked": True,
        "ci_consistency_checked": True,
        "r38_screening_stop_checked": True,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    solve = sub.add_parser("solve")
    solve.add_argument("--registry", type=Path, required=True)
    solve.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(self_test(), indent=2, sort_keys=True))
        else:
            registry = load_json(args.registry)
            output = build_output(registry, normalize_registry(registry))
            write_json(args.out, output)
            print(json.dumps({"status": output["status"], "output": str(args.out), "output_sha256": file_sha256(args.out), "counts": output["registry_counts"]}, indent=2, sort_keys=True))
        return 0
    except FrontierError as exc:
        print(f"r37_slq_frontier: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
