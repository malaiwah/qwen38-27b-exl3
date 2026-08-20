#!/usr/bin/env python3
"""Clean-room ResComp numerical oracle and fail-closed experiment records."""

from __future__ import annotations

import argparse
import copy
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json, sha256_file

ORACLE_INPUT_SCHEMA = "qwen38-frontier-rescomp-oracle-input/1"
ORACLE_OUTPUT_SCHEMA = "qwen38-frontier-rescomp-oracle-output/1"
MANIFEST_INPUT_SCHEMA = "qwen38-frontier-rescomp-experiment-input/1"
MANIFEST_SCHEMA = "qwen38-frontier-rescomp-experiment-manifest/1"
PROMOTION_INPUT_SCHEMA = "qwen38-frontier-rescomp-promotion-input/1"
PROMOTION_SCHEMA = "qwen38-frontier-rescomp-promotion/1"
SELFTEST_INPUT_SCHEMA = "qwen38-frontier-rescomp-selftest-input/1"
SELFTEST_SCHEMA = "qwen38-frontier-rescomp-selftest/1"
PINNED_EXLLAMAV3_REVISION = "5f3c537ca9d89893d771256f5c43c93656553fbb"
PINNED_GDN_SHA256 = "062f9822b0c473963d06ebc7022976b9db16b8dd1436691511b8152801ee9216"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OID_RE = re.compile(r"[0-9a-f]{40,64}\Z")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
GDN_MODULES = frozenset(
    {
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
)
FULL_ATTENTION_MODULES = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
)


class ResCompError(ValueError):
    """A closed validation or numerical failure."""


def fail(message: str) -> NoReturn:
    raise ResCompError(message)


def exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    missing = keys - value.keys()
    unknown = value.keys() - keys
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if unknown:
            parts.append(f"unknown {sorted(unknown)}")
        fail(f"{label} has " + " and ".join(parts))
    return value


def array(value: object, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        fail(f"{label} must be a {qualifier}JSON array")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a nonempty string")
    return value.strip()


def identifier(value: object, label: str) -> str:
    result = text(value, label)
    if ID_RE.fullmatch(result) is None:
        fail(f"{label} must be a safe identifier")
    return result


def sha256(value: object, label: str) -> str:
    result = text(value, label)
    if SHA256_RE.fullmatch(result) is None:
        fail(f"{label} must be a lowercase SHA256")
    return result


def git_oid(value: object, label: str) -> str:
    result = text(value, label)
    if GIT_OID_RE.fullmatch(result) is None:
        fail(f"{label} must be a lowercase 40- or 64-character source object id")
    return result


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < 0 or (positive and value == 0):
        fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError:
        fail(f"{label} must be finite")
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        fail(f"{label} must be finite{' and nonnegative' if nonnegative else ''}")
    return result


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be boolean")
    return value


def validate_identity(value: object, label: str) -> dict[str, str]:
    identity = exact_object(value, {"kind", "name", "revision", "sha256"}, label)
    result = {
        "kind": identifier(identity["kind"], f"{label}.kind"),
        "name": text(identity["name"], f"{label}.name"),
        "revision": git_oid(identity["revision"], f"{label}.revision"),
        "sha256": sha256(identity["sha256"], f"{label}.sha256"),
    }
    if result["name"].startswith(("/", "file:", "ssh:")) or "\\" in result["name"]:
        fail(f"{label}.name must be an immutable source name, not a host path")
    return result


def validate_source_identities(value: object, label: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or not value:
        fail(f"{label} must be a nonempty identity map")
    result: dict[str, dict[str, str]] = {}
    for raw_name, raw_identity in value.items():
        name = identifier(raw_name, f"{label} key")
        result[name] = validate_identity(raw_identity, f"{label}.{name}")
    return result


def matrix(value: object, label: str) -> list[list[float]]:
    rows = array(value, label, nonempty=True)
    width: int | None = None
    result: list[list[float]] = []
    for row_index, raw_row in enumerate(rows):
        row = array(raw_row, f"{label}[{row_index}]", nonempty=True)
        if width is None:
            width = len(row)
        elif len(row) != width:
            fail(f"{label} is ragged at row {row_index}")
        result.append(
            [number(item, f"{label}[{row_index}][{column_index}]") for column_index, item in enumerate(row)]
        )
    return result


def matrix_shape(value: list[list[float]]) -> tuple[int, int]:
    return len(value), len(value[0])


def matmul_right_transpose(inputs: list[list[float]], weights: list[list[float]]) -> list[list[float]]:
    _, features = matrix_shape(inputs)
    outputs, weight_features = matrix_shape(weights)
    if features != weight_features:
        fail(f"matrix multiplication shape mismatch: input width {features}, weight width {weight_features}")
    return [[sum(row[k] * weights[o][k] for k in range(features)) for o in range(outputs)] for row in inputs]


def squared_error(actual: list[list[float]], expected: list[list[float]]) -> float:
    if matrix_shape(actual) != matrix_shape(expected):
        fail("metric matrices have different shapes")
    return math.fsum((a - b) ** 2 for actual_row, expected_row in zip(actual, expected) for a, b in zip(actual_row, expected_row))


def solve_linear(system: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(system)
    if size == 0 or any(len(row) != size for row in system) or len(rhs) != size:
        fail("linear solve requires a square nonempty system")
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(system)]
    scale = max(max(abs(item) for item in row) for row in system)
    tolerance = max(1.0, scale) * 1.0e-13
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            fail("running-input normal matrix is singular; provide positive damping or more independent rows")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def continuous_rescomp_target(
    reference_inputs: list[list[float]],
    running_inputs: list[list[float]],
    weight: list[list[float]],
    damping: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Solve min_C ||X_running C^T - X_reference W^T||_F without post-quant correction."""
    reference_shape = matrix_shape(reference_inputs)
    running_shape = matrix_shape(running_inputs)
    outputs, features = matrix_shape(weight)
    if reference_shape != running_shape:
        fail(f"reference/running input shape drift: {reference_shape} != {running_shape}")
    if reference_shape[1] != features:
        fail("input feature count does not match current-layer weight")
    target_outputs = matmul_right_transpose(reference_inputs, weight)
    rows = len(running_inputs)
    normal = [
        [math.fsum(running_inputs[row][i] * running_inputs[row][j] for row in range(rows)) for j in range(features)]
        for i in range(features)
    ]
    for diagonal in range(features):
        normal[diagonal][diagonal] += damping
    compensated: list[list[float]] = []
    for output in range(outputs):
        rhs = [
            math.fsum(running_inputs[row][feature] * target_outputs[row][output] for row in range(rows))
            for feature in range(features)
        ]
        compensated.append(solve_linear(normal, rhs))
    return compensated, target_outputs


def bf16_round(value: float) -> float:
    """Round through IEEE binary32 to BF16, ties to even, and return a Python float."""
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    exponent = bits & 0x7F800000
    if exponent == 0x7F800000:
        return struct.unpack(">f", struct.pack(">I", bits & 0xFFFF0000))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack(">f", struct.pack(">I", rounded & 0xFFFF0000))[0]


def bf16_matrix(value: list[list[float]]) -> list[list[float]]:
    return [[bf16_round(item) for item in row] for row in value]


def validate_budget(value: object, tile_count: int, label: str) -> dict[str, int]:
    budget = exact_object(
        value,
        {"k", "payload_bytes_per_tile", "total_payload_bytes"},
        label,
    )
    result = {
        "k": integer(budget["k"], f"{label}.k", positive=True),
        "payload_bytes_per_tile": integer(
            budget["payload_bytes_per_tile"], f"{label}.payload_bytes_per_tile", positive=True
        ),
        "total_payload_bytes": integer(budget["total_payload_bytes"], f"{label}.total_payload_bytes", positive=True),
    }
    expected = tile_count * result["payload_bytes_per_tile"]
    if result["total_payload_bytes"] != expected:
        fail(f"{label}.total_payload_bytes drift: expected {expected}, got {result['total_payload_bytes']}")
    return result


def tile_name(row_offset: int, column_offset: int) -> str:
    return f"r{row_offset}c{column_offset}"


def validate_candidate_bank(
    value: object,
    label: str,
    weight_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    budget: dict[str, int],
) -> dict[str, dict[str, Any]]:
    output_rows, input_columns = weight_shape
    tile_rows, tile_columns = tile_shape
    expected = {
        tile_name(row, column): (row, column)
        for row in range(0, output_rows, tile_rows)
        for column in range(0, input_columns, tile_columns)
    }
    result: dict[str, dict[str, Any]] = {}
    for tile_index, raw_tile in enumerate(array(value, label, nonempty=True)):
        tile = exact_object(
            raw_tile,
            {"tile_id", "row_offset", "column_offset", "candidates"},
            f"{label}[{tile_index}]",
        )
        tile_id = identifier(tile["tile_id"], f"{label}[{tile_index}].tile_id")
        row_offset = integer(tile["row_offset"], f"{label}[{tile_index}].row_offset")
        column_offset = integer(tile["column_offset"], f"{label}[{tile_index}].column_offset")
        if tile_id in result:
            fail(f"{label} repeats tile_id {tile_id}")
        if expected.get(tile_id) != (row_offset, column_offset):
            fail(f"{label} tile {tile_id} has shape/grid drift")
        candidates: list[dict[str, Any]] = []
        seen_candidates: set[str] = set()
        for candidate_index, raw_candidate in enumerate(
            array(tile["candidates"], f"{label}.{tile_id}.candidates", nonempty=True)
        ):
            candidate = exact_object(
                raw_candidate,
                {"candidate_id", "k", "payload_bytes", "values"},
                f"{label}.{tile_id}.candidates[{candidate_index}]",
            )
            candidate_id = identifier(
                candidate["candidate_id"], f"{label}.{tile_id}.candidates[{candidate_index}].candidate_id"
            )
            if candidate_id in seen_candidates:
                fail(f"{label}.{tile_id} repeats candidate_id {candidate_id}")
            seen_candidates.add(candidate_id)
            candidate_k = integer(candidate["k"], f"{label}.{tile_id}.{candidate_id}.k", positive=True)
            candidate_bytes = integer(
                candidate["payload_bytes"], f"{label}.{tile_id}.{candidate_id}.payload_bytes", positive=True
            )
            if candidate_k != budget["k"]:
                fail(f"{label}.{tile_id}.{candidate_id} K drift: expected {budget['k']}, got {candidate_k}")
            if candidate_bytes != budget["payload_bytes_per_tile"]:
                fail(
                    f"{label}.{tile_id}.{candidate_id} byte drift: expected "
                    f"{budget['payload_bytes_per_tile']}, got {candidate_bytes}"
                )
            values = matrix(candidate["values"], f"{label}.{tile_id}.{candidate_id}.values")
            if matrix_shape(values) != tile_shape:
                fail(
                    f"{label}.{tile_id}.{candidate_id} shape drift: expected {tile_shape}, "
                    f"got {matrix_shape(values)}"
                )
            candidates.append({"candidate_id": candidate_id, "values": values})
        result[tile_id] = {
            "row_offset": row_offset,
            "column_offset": column_offset,
            "candidates": sorted(candidates, key=lambda item: item["candidate_id"]),
        }
    if set(result) != set(expected):
        fail(f"{label} tile coverage drift: expected {sorted(expected)}, got {sorted(result)}")
    return result


def choose_tiles(
    base_weight: list[list[float]],
    scoring_inputs: list[list[float]],
    target_outputs: list[list[float]],
    bank: dict[str, dict[str, Any]],
    tile_order: list[str],
) -> tuple[list[list[float]], list[dict[str, Any]], float]:
    reconstructed = [list(row) for row in base_weight]
    choices: list[dict[str, Any]] = []
    for tile_id in tile_order:
        tile = bank[tile_id]
        row_offset = tile["row_offset"]
        column_offset = tile["column_offset"]
        candidates = tile["candidates"]
        tile_rows, tile_columns = matrix_shape(candidates[0]["values"])
        best: tuple[float, str, list[list[float]]] | None = None
        for candidate in candidates:
            trial = [list(row) for row in reconstructed]
            values = candidate["values"]
            for tile_row in range(tile_rows):
                for tile_column in range(tile_columns):
                    trial[row_offset + tile_row][column_offset + tile_column] = values[tile_row][tile_column]
            error = squared_error(matmul_right_transpose(scoring_inputs, trial), target_outputs)
            key = (error, candidate["candidate_id"], values)
            if best is None or key[:2] < best[:2]:
                best = key
        if best is None:
            raise AssertionError("validated tile has no candidates")
        error, candidate_id, values = best
        for tile_row in range(tile_rows):
            for tile_column in range(tile_columns):
                reconstructed[row_offset + tile_row][column_offset + tile_column] = values[tile_row][tile_column]
        choices.append({"tile_id": tile_id, "candidate_id": candidate_id, "running_sse": error})
    final_error = squared_error(matmul_right_transpose(scoring_inputs, reconstructed), target_outputs)
    return reconstructed, choices, final_error


def validate_oracle_input(value: object) -> dict[str, Any]:
    document = exact_object(
        value,
        {
            "schema",
            "profile",
            "source_identities",
            "calibration_sha256",
            "predecessor_state_sha256",
            "reference_inputs",
            "running_inputs",
            "weight",
            "damping",
            "tile_shape",
            "budget",
            "tile_order",
            "candidate_banks",
        },
        "oracle input",
    )
    if document["schema"] != ORACLE_INPUT_SCHEMA:
        fail(f"oracle input schema must be {ORACLE_INPUT_SCHEMA}")
    profile = document["profile"]
    if profile not in {"toy", "exl3_16x16"}:
        fail("oracle input profile must be toy or exl3_16x16")
    source_identities = validate_source_identities(document["source_identities"], "source_identities")
    calibration_digest = sha256(document["calibration_sha256"], "calibration_sha256")
    predecessor_digest = sha256(document["predecessor_state_sha256"], "predecessor_state_sha256")
    reference_inputs = matrix(document["reference_inputs"], "reference_inputs")
    running_inputs = matrix(document["running_inputs"], "running_inputs")
    weight = matrix(document["weight"], "weight")
    if matrix_shape(reference_inputs) != matrix_shape(running_inputs):
        fail("reference/running input shape drift")
    if matrix_shape(reference_inputs)[1] != matrix_shape(weight)[1]:
        fail("input/weight feature shape drift")
    damping = number(document["damping"], "damping", nonnegative=True)
    raw_tile_shape = array(document["tile_shape"], "tile_shape")
    if len(raw_tile_shape) != 2:
        fail("tile_shape must have exactly two dimensions")
    tile_shape = (
        integer(raw_tile_shape[0], "tile_shape[0]", positive=True),
        integer(raw_tile_shape[1], "tile_shape[1]", positive=True),
    )
    if profile == "exl3_16x16" and tile_shape != (16, 16):
        fail("EXL3 execution requires exact 16x16 trellis tiles")
    weight_rows, weight_columns = matrix_shape(weight)
    if weight_rows % tile_shape[0] or weight_columns % tile_shape[1]:
        fail("weight shape must be exactly divisible by tile_shape")
    tile_count = (weight_rows // tile_shape[0]) * (weight_columns // tile_shape[1])
    budget = validate_budget(document["budget"], tile_count, "budget")
    candidate_banks = exact_object(document["candidate_banks"], {"ldlq", "rescomp"}, "candidate_banks")
    banks = {
        method: validate_candidate_bank(
            candidate_banks[method], f"candidate_banks.{method}", matrix_shape(weight), tile_shape, budget
        )
        for method in ("ldlq", "rescomp")
    }
    raw_order = array(document["tile_order"], "tile_order", nonempty=True)
    tile_order = [identifier(item, f"tile_order[{index}]") for index, item in enumerate(raw_order)]
    if len(tile_order) != len(set(tile_order)) or set(tile_order) != set(banks["ldlq"]):
        fail("tile_order must contain every tile_id exactly once")
    return {
        "profile": profile,
        "source_identities": source_identities,
        "calibration_sha256": calibration_digest,
        "predecessor_state_sha256": predecessor_digest,
        "reference_inputs": reference_inputs,
        "running_inputs": running_inputs,
        "weight": weight,
        "damping": damping,
        "tile_shape": tile_shape,
        "budget": budget,
        "tile_order": tile_order,
        "banks": banks,
    }


def run_oracle(value: object) -> dict[str, Any]:
    parsed = validate_oracle_input(value)
    compensated, target_outputs = continuous_rescomp_target(
        parsed["reference_inputs"], parsed["running_inputs"], parsed["weight"], parsed["damping"]
    )
    ldlq_weight, ldlq_choices, ldlq_error = choose_tiles(
        parsed["weight"],
        parsed["reference_inputs"],
        target_outputs,
        parsed["banks"]["ldlq"],
        parsed["tile_order"],
    )
    rescomp_weight, rescomp_choices, rescomp_error = choose_tiles(
        compensated,
        parsed["running_inputs"],
        target_outputs,
        parsed["banks"]["rescomp"],
        parsed["tile_order"],
    )
    target_energy = math.fsum(item * item for row in target_outputs for item in row)
    denominator = max(target_energy, sys.float_info.min)
    comparison_identity = {
        "source_identities": parsed["source_identities"],
        "calibration_sha256": parsed["calibration_sha256"],
        "weight_sha256": canonical_sha256(parsed["weight"]),
        "reference_inputs_sha256": canonical_sha256(parsed["reference_inputs"]),
        "tile_shape": list(parsed["tile_shape"]),
        "budget": parsed["budget"],
    }
    path_identity = {
        "comparison_input_sha256": canonical_sha256(comparison_identity),
        "predecessor_state_sha256": parsed["predecessor_state_sha256"],
        "running_inputs_sha256": canonical_sha256(parsed["running_inputs"]),
        "tile_order": parsed["tile_order"],
        "arithmetic": "float64-normal-equations-and-sequential-tile-sse",
    }
    return {
        "schema": ORACLE_OUTPUT_SCHEMA,
        "input_sha256": canonical_sha256(value),
        "comparison_input_sha256": canonical_sha256(comparison_identity),
        "path_sha256": canonical_sha256(path_identity),
        "method_policy": {
            "active": "rescomp",
            "comparator": "ldlq",
            "mutually_exclusive": ["yaqa"],
            "post_quant_correction": "none",
        },
        "continuous_target": {
            "weight": compensated,
            "output_sse": squared_error(
                matmul_right_transpose(parsed["running_inputs"], compensated), target_outputs
            ),
            "bf16_output_sse": squared_error(
                matmul_right_transpose(parsed["running_inputs"], bf16_matrix(compensated)), target_outputs
            ),
        },
        "fixed_budget": parsed["budget"],
        "arms": {
            "ldlq": {
                "choices": ldlq_choices,
                "weight": ldlq_weight,
                "output_sse": ldlq_error,
                "relative_output_sse": ldlq_error / denominator,
            },
            "rescomp": {
                "choices": rescomp_choices,
                "weight": rescomp_weight,
                "output_sse": rescomp_error,
                "relative_output_sse": rescomp_error / denominator,
            },
        },
    }


def validate_method_policy(value: object, label: str) -> dict[str, Any]:
    policy = exact_object(
        value,
        {"active", "comparator", "mutually_exclusive", "post_quant_correction"},
        label,
    )
    if policy["active"] != "rescomp" or policy["comparator"] != "ldlq":
        fail(f"{label} must compare rescomp against ldlq")
    if policy["mutually_exclusive"] != ["yaqa"]:
        fail(f"{label} must make rescomp mutually exclusive with YAQA")
    if policy["post_quant_correction"] != "none":
        fail(f"{label} forbids scalar GPTQ or any other correction after trellis selection")
    return copy.deepcopy(policy)


def validate_arm(value: object, label: str) -> dict[str, Any]:
    arm = exact_object(
        value,
        {"method", "k", "payload_bytes_per_tile", "total_payload_bytes", "comparison_input_sha256"},
        label,
    )
    if arm["method"] not in {"ldlq", "rescomp"}:
        fail(f"{label}.method must be ldlq or rescomp")
    return {
        "method": arm["method"],
        "k": integer(arm["k"], f"{label}.k", positive=True),
        "payload_bytes_per_tile": integer(
            arm["payload_bytes_per_tile"], f"{label}.payload_bytes_per_tile", positive=True
        ),
        "total_payload_bytes": integer(arm["total_payload_bytes"], f"{label}.total_payload_bytes", positive=True),
        "comparison_input_sha256": sha256(arm["comparison_input_sha256"], f"{label}.comparison_input_sha256"),
    }


def expected_modules(kind: str) -> frozenset[str]:
    if kind == "gdn":
        return GDN_MODULES
    if kind == "full_attention":
        return FULL_ATTENTION_MODULES
    fail("suffix block_kind must be gdn or full_attention")
    raise AssertionError("unreachable")


def validate_suffix(value: object, label: str, mode: str) -> dict[str, Any]:
    suffix = exact_object(
        value,
        {
            "suffix_id",
            "block_kind",
            "start_block",
            "end_block",
            "block_indices",
            "whole_block_modules",
            "running_inputs_sha256",
            "predecessor_state_sha256",
            "propagate_to_end_logits",
        },
        label,
    )
    suffix_id = identifier(suffix["suffix_id"], f"{label}.suffix_id")
    kind = suffix["block_kind"]
    modules_expected = expected_modules(kind)
    start = integer(suffix["start_block"], f"{label}.start_block")
    end = integer(suffix["end_block"], f"{label}.end_block")
    if end < start:
        fail(f"{label} has reversed block suffix")
    indices = [integer(item, f"{label}.block_indices[{index}]") for index, item in enumerate(array(suffix["block_indices"], f"{label}.block_indices", nonempty=True))]
    if indices != list(range(start, end + 1)):
        fail(f"{label}.block_indices must be the complete consecutive suffix")
    if mode == "qwen" and (end != 63 or start >= 64):
        fail(f"{label} Qwen suffix must propagate through block 63")
    modules = [text(item, f"{label}.whole_block_modules[{index}]") for index, item in enumerate(array(suffix["whole_block_modules"], f"{label}.whole_block_modules", nonempty=True))]
    if len(modules) != len(set(modules)) or set(modules) != modules_expected:
        fail(f"{label}.whole_block_modules must equal {sorted(modules_expected)}")
    if boolean(suffix["propagate_to_end_logits"], f"{label}.propagate_to_end_logits") is not True:
        fail(f"{label} must propagate the complete suffix to end logits")
    return {
        "suffix_id": suffix_id,
        "block_kind": kind,
        "start_block": start,
        "end_block": end,
        "block_indices": indices,
        "whole_block_modules": modules,
        "running_inputs_sha256": sha256(suffix["running_inputs_sha256"], f"{label}.running_inputs_sha256"),
        "predecessor_state_sha256": sha256(
            suffix["predecessor_state_sha256"], f"{label}.predecessor_state_sha256"
        ),
        "propagate_to_end_logits": True,
    }


def validate_converter_identity(value: object, label: str) -> dict[str, Any]:
    identity = exact_object(value, {"repo", "revision", "tree", "patch_sha256"}, label)
    repo = text(identity["repo"], f"{label}.repo")
    if repo.startswith(("/", "file:", "ssh:")) or "\\" in repo:
        fail(f"{label}.repo must be an immutable repository name or network URL, not a host path")
    return {
        "repo": repo,
        "revision": git_oid(identity["revision"], f"{label}.revision"),
        "tree": git_oid(identity["tree"], f"{label}.tree"),
        "patch_sha256": sha256(identity["patch_sha256"], f"{label}.patch_sha256"),
    }


def validate_comparison_inputs(value: object, label: str) -> dict[str, Any]:
    inputs = exact_object(
        value,
        {
            "weight_sha256",
            "hessian_sha256",
            "calibration_sha256",
            "candidate_order_sha256",
            "quantizer_seed",
            "out_scale",
        },
        label,
    )
    out_scale = number(inputs["out_scale"], f"{label}.out_scale")
    if out_scale <= 0.0:
        fail(f"{label}.out_scale must be positive")
    return {
        "weight_sha256": sha256(inputs["weight_sha256"], f"{label}.weight_sha256"),
        "hessian_sha256": sha256(inputs["hessian_sha256"], f"{label}.hessian_sha256"),
        "calibration_sha256": sha256(inputs["calibration_sha256"], f"{label}.calibration_sha256"),
        "candidate_order_sha256": sha256(
            inputs["candidate_order_sha256"], f"{label}.candidate_order_sha256"
        ),
        "quantizer_seed": integer(inputs["quantizer_seed"], f"{label}.quantizer_seed"),
        "out_scale": out_scale,
    }


def validate_manifest_input(value: object) -> dict[str, Any]:
    document = exact_object(
        value,
        {
            "schema",
            "experiment_id",
            "mode",
            "source_identities",
            "converter_identity",
            "method_policy",
            "arithmetic",
            "trellis",
            "comparison_inputs",
            "arms",
            "suffix_experiments",
        },
        "experiment input",
    )
    if document["schema"] != MANIFEST_INPUT_SCHEMA:
        fail(f"experiment input schema must be {MANIFEST_INPUT_SCHEMA}")
    experiment_id = identifier(document["experiment_id"], "experiment_id")
    mode = document["mode"]
    if mode not in {"toy", "qwen"}:
        fail("mode must be toy or qwen")
    identities = validate_source_identities(document["source_identities"], "source_identities")
    converter = validate_converter_identity(document["converter_identity"], "converter_identity")
    policy = validate_method_policy(document["method_policy"], "method_policy")
    arithmetic = exact_object(
        document["arithmetic"],
        {"solve_dtype", "accumulation_dtype", "weight_limit_dtype", "damping"},
        "arithmetic",
    )
    if arithmetic["solve_dtype"] != "float64" or arithmetic["accumulation_dtype"] != "float32":
        fail("arithmetic must pin float64 compensation solves and float32 trellis accumulation")
    if arithmetic["weight_limit_dtype"] != "bf16":
        fail("arithmetic.weight_limit_dtype must be bf16")
    parsed_arithmetic = dict(arithmetic)
    parsed_arithmetic["damping"] = number(arithmetic["damping"], "arithmetic.damping", nonnegative=True)
    trellis = exact_object(
        document["trellis"],
        {"tile_shape", "codebook_id", "scale_mode", "tile_count"},
        "trellis",
    )
    if trellis["tile_shape"] != [16, 16]:
        fail("trellis.tile_shape must be exactly [16, 16]")
    parsed_trellis = {
        "tile_shape": [16, 16],
        "codebook_id": identifier(trellis["codebook_id"], "trellis.codebook_id"),
        "scale_mode": identifier(trellis["scale_mode"], "trellis.scale_mode"),
        "tile_count": integer(trellis["tile_count"], "trellis.tile_count", positive=True),
    }
    comparison_inputs = validate_comparison_inputs(document["comparison_inputs"], "comparison_inputs")
    comparison_digest = canonical_sha256(comparison_inputs)
    raw_arms = array(document["arms"], "arms", nonempty=True)
    arms = [validate_arm(item, f"arms[{index}]") for index, item in enumerate(raw_arms)]
    if [arm["method"] for arm in arms] != ["ldlq", "rescomp"]:
        fail("arms must contain the LDLQ control followed by ResComp")
    equality_fields = ("k", "payload_bytes_per_tile", "total_payload_bytes", "comparison_input_sha256")
    for field in equality_fields:
        if arms[0][field] != arms[1][field]:
            fail(f"fixed-budget LDLQ comparison has {field} drift")
    if arms[0]["comparison_input_sha256"] != comparison_digest:
        fail(
            "arms.comparison_input_sha256 must bind the canonical fixed W/H/calibration/"
            "candidate-order/seed/out-scale inputs"
        )
    expected_total = parsed_trellis["tile_count"] * arms[0]["payload_bytes_per_tile"]
    if arms[0]["total_payload_bytes"] != expected_total:
        fail(f"arm total byte drift: expected {expected_total}, got {arms[0]['total_payload_bytes']}")
    suffixes = [
        validate_suffix(item, f"suffix_experiments[{index}]", mode)
        for index, item in enumerate(array(document["suffix_experiments"], "suffix_experiments", nonempty=True))
    ]
    if len(suffixes) != 2 or {suffix["block_kind"] for suffix in suffixes} != {"gdn", "full_attention"}:
        fail("suffix_experiments must contain exactly one whole GDN and one whole full-attention suffix")
    if len({suffix["suffix_id"] for suffix in suffixes}) != len(suffixes):
        fail("suffix_experiments repeats a suffix_id")
    return {
        "experiment_id": experiment_id,
        "mode": mode,
        "source_identities": identities,
        "converter_identity": converter,
        "method_policy": policy,
        "arithmetic": parsed_arithmetic,
        "trellis": parsed_trellis,
        "comparison_inputs": comparison_inputs,
        "arms": arms,
        "suffix_experiments": suffixes,
    }


def git_value(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        fail("Qwen mode requires git to verify the pinned external exllamav3 checkout")
        raise AssertionError("unreachable") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        fail(f"cannot verify external exllamav3 checkout: {detail.strip()}")
    return completed.stdout.strip().lower()


def verify_qwen_checkout(checkout: Path | None, identity: dict[str, Any]) -> dict[str, str]:
    if checkout is None:
        fail("Qwen mode requires --exllamav3-checkout pinned to the declared converter identity")
    if not checkout.is_dir():
        fail("Qwen mode external exllamav3 checkout is unavailable or not a directory")
    revision = git_value(checkout, "rev-parse", "HEAD")
    tree = git_value(checkout, "rev-parse", "HEAD^{tree}")
    if revision != identity["revision"]:
        fail(f"external exllamav3 revision mismatch: expected {identity['revision']}, got {revision}")
    if tree != identity["tree"]:
        fail(f"external exllamav3 tree mismatch: expected {identity['tree']}, got {tree}")
    if revision != PINNED_EXLLAMAV3_REVISION:
        fail(f"Qwen mode requires pinned exllamav3 revision {PINNED_EXLLAMAV3_REVISION}")
    if identity["patch_sha256"] != PINNED_GDN_SHA256:
        fail("Qwen mode converter identity does not pin the qualified GDN optimizer-target content")
    required = checkout / "exllamav3" / "modules" / "gated_delta_net.py"
    if not required.is_file():
        fail("external exllamav3 checkout lacks exllamav3/modules/gated_delta_net.py")
    actual_gdn = sha256_file(required)
    if actual_gdn != identity["patch_sha256"]:
        fail(
            "external exllamav3 checkout does not contain the pinned qualified GDN optimizer-target source: "
            f"expected {identity['patch_sha256']}, got {actual_gdn}"
        )
    return {"revision": revision, "tree": tree, "gdn_source_sha256": actual_gdn}


def normalized_manifest_input(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_INPUT_SCHEMA,
        "experiment_id": parsed["experiment_id"],
        "mode": parsed["mode"],
        "source_identities": parsed["source_identities"],
        "converter_identity": parsed["converter_identity"],
        "method_policy": parsed["method_policy"],
        "arithmetic": parsed["arithmetic"],
        "trellis": parsed["trellis"],
        "comparison_inputs": parsed["comparison_inputs"],
        "arms": parsed["arms"],
        "suffix_experiments": parsed["suffix_experiments"],
    }


def build_manifest(value: object, checkout: Path | None) -> dict[str, Any]:
    parsed = validate_manifest_input(value)
    checkout_evidence: dict[str, str] | None = None
    if parsed["mode"] == "qwen":
        checkout_evidence = verify_qwen_checkout(checkout, parsed["converter_identity"])
    return {
        "schema": MANIFEST_SCHEMA,
        "input_sha256": canonical_sha256(normalized_manifest_input(parsed)),
        "experiment_id": parsed["experiment_id"],
        "mode": parsed["mode"],
        "source_identities": parsed["source_identities"],
        "converter_identity": parsed["converter_identity"],
        "checkout_evidence": checkout_evidence,
        "method_policy": parsed["method_policy"],
        "arithmetic": parsed["arithmetic"],
        "trellis": parsed["trellis"],
        "comparison_inputs": parsed["comparison_inputs"],
        "arms": parsed["arms"],
        "suffix_experiments": parsed["suffix_experiments"],
        "path_dependence": {
            suffix["suffix_id"]: canonical_sha256(
                {
                    "running_inputs_sha256": suffix["running_inputs_sha256"],
                    "predecessor_state_sha256": suffix["predecessor_state_sha256"],
                    "comparison_input_sha256": parsed["arms"][0]["comparison_input_sha256"],
                    "block_indices": suffix["block_indices"],
                }
            )
            for suffix in parsed["suffix_experiments"]
        },
    }


def validate_manifest(value: object) -> dict[str, Any]:
    manifest = exact_object(
        value,
        {
            "schema",
            "input_sha256",
            "experiment_id",
            "mode",
            "source_identities",
            "converter_identity",
            "checkout_evidence",
            "method_policy",
            "arithmetic",
            "trellis",
            "comparison_inputs",
            "arms",
            "suffix_experiments",
            "path_dependence",
        },
        "manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        fail(f"manifest schema must be {MANIFEST_SCHEMA}")
    sha256(manifest["input_sha256"], "manifest.input_sha256")
    reconstructed_input = {
        "schema": MANIFEST_INPUT_SCHEMA,
        "experiment_id": manifest["experiment_id"],
        "mode": manifest["mode"],
        "source_identities": manifest["source_identities"],
        "converter_identity": manifest["converter_identity"],
        "method_policy": manifest["method_policy"],
        "arithmetic": manifest["arithmetic"],
        "trellis": manifest["trellis"],
        "comparison_inputs": manifest["comparison_inputs"],
        "arms": manifest["arms"],
        "suffix_experiments": manifest["suffix_experiments"],
    }
    parsed = validate_manifest_input(reconstructed_input)
    if canonical_sha256(reconstructed_input) != manifest["input_sha256"]:
        fail("manifest input_sha256 does not bind its canonical experiment input")
    checkout_evidence = manifest["checkout_evidence"]
    if parsed["mode"] == "qwen":
        evidence = exact_object(
            checkout_evidence,
            {"revision", "tree", "gdn_source_sha256"},
            "manifest.checkout_evidence",
        )
        if evidence != {
            "revision": parsed["converter_identity"]["revision"],
            "tree": parsed["converter_identity"]["tree"],
            "gdn_source_sha256": parsed["converter_identity"]["patch_sha256"],
        }:
            fail("manifest.checkout_evidence does not match converter identity")
    elif checkout_evidence is not None:
        fail("toy manifest must not claim external checkout evidence")
    dependence = exact_object(
        manifest["path_dependence"],
        {suffix["suffix_id"] for suffix in parsed["suffix_experiments"]},
        "manifest.path_dependence",
    )
    for suffix in parsed["suffix_experiments"]:
        expected = canonical_sha256(
            {
                "running_inputs_sha256": suffix["running_inputs_sha256"],
                "predecessor_state_sha256": suffix["predecessor_state_sha256"],
                "comparison_input_sha256": parsed["arms"][0]["comparison_input_sha256"],
                "block_indices": suffix["block_indices"],
            }
        )
        if sha256(dependence[suffix["suffix_id"]], f"path_dependence.{suffix['suffix_id']}") != expected:
            fail(f"manifest path dependence drift for {suffix['suffix_id']}")
    return manifest


def validate_end_logit(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    evidence = exact_object(
        value,
        {
            "reference_logits_sha256",
            "rescomp_logits_sha256",
            "ldlq_logits_sha256",
            "sample_count",
            "token_count",
            "rescomp_mean_kld",
            "ldlq_mean_kld",
            "all_finite",
        },
        label,
    )
    return {
        "reference_logits_sha256": sha256(evidence["reference_logits_sha256"], f"{label}.reference_logits_sha256"),
        "rescomp_logits_sha256": sha256(evidence["rescomp_logits_sha256"], f"{label}.rescomp_logits_sha256"),
        "ldlq_logits_sha256": sha256(evidence["ldlq_logits_sha256"], f"{label}.ldlq_logits_sha256"),
        "sample_count": integer(evidence["sample_count"], f"{label}.sample_count", positive=True),
        "token_count": integer(evidence["token_count"], f"{label}.token_count", positive=True),
        "rescomp_mean_kld": number(evidence["rescomp_mean_kld"], f"{label}.rescomp_mean_kld", nonnegative=True),
        "ldlq_mean_kld": number(evidence["ldlq_mean_kld"], f"{label}.ldlq_mean_kld", nonnegative=True),
        "all_finite": boolean(evidence["all_finite"], f"{label}.all_finite"),
    }


def validate_result(value: object, label: str) -> dict[str, Any]:
    result = exact_object(
        value,
        {
            "suffix_id",
            "status",
            "method",
            "comparator",
            "k",
            "payload_bytes_per_tile",
            "total_payload_bytes",
            "comparison_input_sha256",
            "running_inputs_sha256",
            "predecessor_state_sha256",
            "whole_block_modules",
            "running_quantized_inputs",
            "propagated_block_indices",
            "end_logit_evidence",
        },
        label,
    )
    if result["status"] not in {"complete", "failed", "unsupported"}:
        fail(f"{label}.status must be complete, failed, or unsupported")
    if result["method"] != "rescomp" or result["comparator"] != "ldlq":
        fail(f"{label} must report rescomp against ldlq")
    return {
        "suffix_id": identifier(result["suffix_id"], f"{label}.suffix_id"),
        "status": result["status"],
        "method": "rescomp",
        "comparator": "ldlq",
        "k": integer(result["k"], f"{label}.k", positive=True),
        "payload_bytes_per_tile": integer(
            result["payload_bytes_per_tile"], f"{label}.payload_bytes_per_tile", positive=True
        ),
        "total_payload_bytes": integer(result["total_payload_bytes"], f"{label}.total_payload_bytes", positive=True),
        "comparison_input_sha256": sha256(
            result["comparison_input_sha256"], f"{label}.comparison_input_sha256"
        ),
        "running_inputs_sha256": sha256(result["running_inputs_sha256"], f"{label}.running_inputs_sha256"),
        "predecessor_state_sha256": sha256(
            result["predecessor_state_sha256"], f"{label}.predecessor_state_sha256"
        ),
        "whole_block_modules": [
            text(item, f"{label}.whole_block_modules[{index}]")
            for index, item in enumerate(array(result["whole_block_modules"], f"{label}.whole_block_modules"))
        ],
        "running_quantized_inputs": boolean(result["running_quantized_inputs"], f"{label}.running_quantized_inputs"),
        "propagated_block_indices": [
            integer(item, f"{label}.propagated_block_indices[{index}]")
            for index, item in enumerate(array(result["propagated_block_indices"], f"{label}.propagated_block_indices"))
        ],
        "end_logit_evidence": validate_end_logit(result["end_logit_evidence"], f"{label}.end_logit_evidence"),
    }


def promotion(value: object) -> dict[str, Any]:
    document = exact_object(value, {"schema", "manifest", "results"}, "promotion input")
    if document["schema"] != PROMOTION_INPUT_SCHEMA:
        fail(f"promotion input schema must be {PROMOTION_INPUT_SCHEMA}")
    manifest = validate_manifest(document["manifest"])
    results = [
        validate_result(item, f"results[{index}]")
        for index, item in enumerate(array(document["results"], "results"))
    ]
    if len({result["suffix_id"] for result in results}) != len(results):
        fail("results repeats a suffix_id")
    by_id = {result["suffix_id"]: result for result in results}
    reasons: list[str] = []
    arm = manifest["arms"][0]
    for suffix in manifest["suffix_experiments"]:
        suffix_id = suffix["suffix_id"]
        result = by_id.get(suffix_id)
        if result is None:
            reasons.append(f"{suffix_id}: missing whole-block suffix result")
            continue
        if result["status"] != "complete":
            reasons.append(f"{suffix_id}: result status is {result['status']}")
        for field in ("k", "payload_bytes_per_tile", "total_payload_bytes", "comparison_input_sha256"):
            if result[field] != arm[field]:
                reasons.append(f"{suffix_id}: fixed comparison {field} drift")
        for field in ("running_inputs_sha256", "predecessor_state_sha256"):
            if result[field] != suffix[field]:
                reasons.append(f"{suffix_id}: path-dependent {field} drift")
        if len(result["whole_block_modules"]) != len(set(result["whole_block_modules"])) or set(
            result["whole_block_modules"]
        ) != set(suffix["whole_block_modules"]):
            reasons.append(f"{suffix_id}: incomplete whole-{suffix['block_kind']} module evidence")
        if result["running_quantized_inputs"] is not True:
            reasons.append(f"{suffix_id}: inputs were not produced by the running quantized predecessor")
        if result["propagated_block_indices"] != suffix["block_indices"]:
            reasons.append(f"{suffix_id}: incomplete propagated block suffix")
        evidence = result["end_logit_evidence"]
        if evidence is None:
            reasons.append(f"{suffix_id}: missing propagated end-logit evidence")
        elif evidence["all_finite"] is not True:
            reasons.append(f"{suffix_id}: end-logit evidence is non-finite")
        elif evidence["rescomp_mean_kld"] >= evidence["ldlq_mean_kld"]:
            reasons.append(f"{suffix_id}: no direct end-logit KLD win over fixed-budget LDLQ")
    unknown_results = set(by_id) - {suffix["suffix_id"] for suffix in manifest["suffix_experiments"]}
    if unknown_results:
        fail(f"results name unknown suffix experiments: {sorted(unknown_results)}")
    whole_suffix_ok: dict[str, bool] = {}
    end_logits_ok = True
    for suffix in manifest["suffix_experiments"]:
        result = by_id.get(suffix["suffix_id"])
        whole_suffix_ok[suffix["block_kind"]] = bool(
            result is not None
            and result["status"] == "complete"
            and result["running_quantized_inputs"] is True
            and len(result["whole_block_modules"]) == len(set(result["whole_block_modules"]))
            and set(result["whole_block_modules"]) == set(suffix["whole_block_modules"])
        )
        end_logits_ok = bool(
            end_logits_ok
            and result is not None
            and result["propagated_block_indices"] == suffix["block_indices"]
            and result["end_logit_evidence"] is not None
            and result["end_logit_evidence"]["all_finite"] is True
        )
    return {
        "schema": PROMOTION_SCHEMA,
        "input_sha256": canonical_sha256(value),
        "manifest_sha256": canonical_sha256(manifest),
        "experiment_id": manifest["experiment_id"],
        "disposition": "PASS" if not reasons else "NOT_PROMOTED",
        "requirements": {
            "whole_gdn_suffix": whole_suffix_ok.get("gdn", False),
            "whole_full_attention_suffix": whole_suffix_ok.get("full_attention", False),
            "propagated_end_logits": end_logits_ok,
        },
        "reasons": reasons,
    }


def toy_identity() -> dict[str, dict[str, str]]:
    return {
        "oracle": {
            "kind": "clean-room-oracle",
            "name": "frontier_rescomp",
            "revision": "1" * 40,
            "sha256": "2" * 64,
        }
    }


def toy_bank(values_by_tile: dict[str, list[list[list[float]]]], k: int, payload_bytes: int) -> list[dict[str, Any]]:
    result = []
    for tile_id in sorted(values_by_tile):
        match = re.fullmatch(r"r(\d+)c(\d+)", tile_id)
        if match is None:
            raise AssertionError("bad toy tile")
        candidates = [
            {
                "candidate_id": f"q{index}",
                "k": k,
                "payload_bytes": payload_bytes,
                "values": values,
            }
            for index, values in enumerate(values_by_tile[tile_id])
        ]
        result.append(
            {
                "tile_id": tile_id,
                "row_offset": int(match.group(1)),
                "column_offset": int(match.group(2)),
                "candidates": candidates,
            }
        )
    return result


def selftest(value: object) -> dict[str, Any]:
    document = exact_object(value, {"schema", "seed"}, "self-test input")
    if document["schema"] != SELFTEST_INPUT_SCHEMA:
        fail(f"self-test input schema must be {SELFTEST_INPUT_SCHEMA}")
    if integer(document["seed"], "self-test seed") != 0:
        fail("self-test seed is fixed at 0")
    checks: list[str] = []
    digest = "a" * 64
    exact_values = {
        "r0c0": [[[1.0]], [[0.5]]],
        "r0c1": [[[0.0]], [[1.0]]],
        "r1c0": [[[0.0]], [[1.0]]],
        "r1c1": [[[1.0]], [[2.0]]],
    }
    compensated_values = {
        "r0c0": [[[1.0]], [[0.5]]],
        "r0c1": [[[0.0]], [[1.0]]],
        "r1c0": [[[0.0]], [[1.0]]],
        "r1c1": [[[1.0]], [[2.0]]],
    }
    oracle_input = {
        "schema": ORACLE_INPUT_SCHEMA,
        "profile": "toy",
        "source_identities": toy_identity(),
        "calibration_sha256": digest,
        "predecessor_state_sha256": "b" * 64,
        "reference_inputs": [[1.0, 0.0], [0.0, 1.0]],
        "running_inputs": [[2.0, 0.0], [0.0, 0.5]],
        "weight": [[1.0, 0.0], [0.0, 1.0]],
        "damping": 0.0,
        "tile_shape": [1, 1],
        "budget": {"k": 4, "payload_bytes_per_tile": 1, "total_payload_bytes": 4},
        "tile_order": ["r0c0", "r0c1", "r1c0", "r1c1"],
        "candidate_banks": {
            "ldlq": toy_bank(exact_values, 4, 1),
            "rescomp": toy_bank(compensated_values, 4, 1),
        },
    }
    result = run_oracle(oracle_input)
    if result["continuous_target"]["output_sse"] > 1.0e-24:
        fail("self-test continuous no-quant limit failed")
    checks.append("continuous_no_quant_limit")
    if result["continuous_target"]["bf16_output_sse"] > 1.0e-24:
        fail("self-test BF16 exactly-representable limit failed")
    checks.append("bf16_exact_limit")
    if result["arms"]["ldlq"]["output_sse"] > 1.0e-24 or result["arms"]["rescomp"]["output_sse"] > 1.0e-24:
        fail("self-test fixed-budget arm invariant failed")
    checks.append("deterministic_fixed_k_fixed_byte_ldlq_comparison")
    changed_path = copy.deepcopy(oracle_input)
    changed_path["predecessor_state_sha256"] = "c" * 64
    if run_oracle(changed_path)["path_sha256"] == result["path_sha256"]:
        fail("self-test predecessor path dependence failed")
    checks.append("predecessor_path_dependence")
    for field, mutation, expected_fragment in (
        (
            "shape",
            lambda item: item["candidate_banks"]["rescomp"][0]["candidates"][0].update(values=[[1.0, 2.0]]),
            "shape drift",
        ),
        (
            "K",
            lambda item: item["candidate_banks"]["rescomp"][0]["candidates"][0].update(k=3),
            "K drift",
        ),
        (
            "byte",
            lambda item: item["candidate_banks"]["rescomp"][0]["candidates"][0].update(payload_bytes=2),
            "byte drift",
        ),
    ):
        drifted = copy.deepcopy(oracle_input)
        mutation(drifted)
        try:
            run_oracle(drifted)
        except ResCompError as exc:
            if expected_fragment not in str(exc):
                fail(f"self-test {field} drift raised the wrong failure: {exc}")
        else:
            fail(f"self-test failed to reject {field} drift")
        checks.append(f"reject_{field.lower()}_drift")
    toy_comparison_inputs = {
        "weight_sha256": "a" * 64,
        "hessian_sha256": "b" * 64,
        "calibration_sha256": "c" * 64,
        "candidate_order_sha256": "d" * 64,
        "quantizer_seed": 0,
        "out_scale": 1.0,
    }


    toy_manifest_input = {
        "schema": MANIFEST_INPUT_SCHEMA,
        "experiment_id": "selftest-rescomp",
        "mode": "toy",
        "source_identities": toy_identity(),
        "converter_identity": {
            "repo": "example/exllamav3",
            "revision": "3" * 40,
            "tree": "4" * 40,
            "patch_sha256": "5" * 64,
        },
        "method_policy": {
            "active": "rescomp",
            "comparator": "ldlq",
            "mutually_exclusive": ["yaqa"],
            "post_quant_correction": "none",
        },
        "arithmetic": {
            "solve_dtype": "float64",
            "accumulation_dtype": "float32",
            "weight_limit_dtype": "bf16",
            "damping": 0.0,
        },
        "trellis": {"tile_shape": [16, 16], "codebook_id": "mcg", "scale_mode": "mul1", "tile_count": 2},
        "comparison_inputs": toy_comparison_inputs,
        "arms": [
            {
                "method": method,
                "k": 4,
                "payload_bytes_per_tile": 64,
                "total_payload_bytes": 128,
                "comparison_input_sha256": canonical_sha256(toy_comparison_inputs),
            }
            for method in ("ldlq", "rescomp")
        ],
        "suffix_experiments": [
            {
                "suffix_id": "toy-gdn",
                "block_kind": "gdn",
                "start_block": 0,
                "end_block": 1,
                "block_indices": [0, 1],
                "whole_block_modules": sorted(GDN_MODULES),
                "running_inputs_sha256": "7" * 64,
                "predecessor_state_sha256": "8" * 64,
                "propagate_to_end_logits": True,
            },
            {
                "suffix_id": "toy-attention",
                "block_kind": "full_attention",
                "start_block": 1,
                "end_block": 1,
                "block_indices": [1],
                "whole_block_modules": sorted(FULL_ATTENTION_MODULES),
                "running_inputs_sha256": "9" * 64,
                "predecessor_state_sha256": "0" * 64,
                "propagate_to_end_logits": True,
            },
        ],
    }
    manifest = build_manifest(toy_manifest_input, None)
    incomplete_promotion = promotion(
        {"schema": PROMOTION_INPUT_SCHEMA, "manifest": manifest, "results": []}
    )
    if incomplete_promotion["disposition"] == "PASS":
        fail("self-test promotion accepted missing whole-block/end-logit evidence")
    checks.append("promotion_requires_whole_block_and_end_logits")
    incompatible = copy.deepcopy(toy_manifest_input)
    incompatible["method_policy"]["active"] = "yaqa"
    try:
        build_manifest(incompatible, None)
    except ResCompError:
        pass
    else:
        fail("self-test did not enforce YAQA mutual exclusion")
    checks.append("yaqa_mutual_exclusion")
    post_quant = copy.deepcopy(toy_manifest_input)
    post_quant["method_policy"]["post_quant_correction"] = "scalar_gptq"
    try:
        build_manifest(post_quant, None)
    except ResCompError:
        pass
    else:
        fail("self-test did not forbid post-quant scalar GPTQ")
    checks.append("no_post_quant_scalar_gptq")
    return {
        "schema": SELFTEST_SCHEMA,
        "input_sha256": canonical_sha256(value),
        "status": "PASS",
        "checks": checks,
        "oracle_output_sha256": canonical_sha256(result),
    }


def load_input(path: Path) -> object:
    if not path.is_file():
        fail(f"input artifact is missing or not a regular file: {path}")
    try:
        if path.stat().st_size == 0:
            fail(f"input artifact is empty: {path}")
        return load_strict_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ResCompError):
            raise
        fail(f"cannot load strict input artifact {path}: {exc}")
    raise AssertionError("unreachable")


def validate_output_path(path: Path) -> None:
    if path.exists():
        if not path.is_file():
            fail(f"output path exists and is not a regular file: {path}")
        try:
            if path.stat().st_size != 0:
                fail(f"refusing to overwrite nonempty output artifact: {path}")
        except OSError as exc:
            fail(f"cannot inspect output artifact {path}: {exc}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("oracle", "manifest", "promote", "self-test"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True, help="strict JSON input path")
        command.add_argument("--output", type=Path, required=True, help="new canonical JSON output path")
        if name == "manifest":
            command.add_argument(
                "--exllamav3-checkout",
                type=Path,
                help="external pinned checkout required by mode=qwen; never recorded as a host path",
            )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        value = load_input(args.input)
        validate_output_path(args.output)
        if args.command == "oracle":
            output = run_oracle(value)
        elif args.command == "manifest":
            output = build_manifest(value, args.exllamav3_checkout)
        elif args.command == "promote":
            output = promotion(value)
        elif args.command == "self-test":
            output = selftest(value)
        else:
            raise AssertionError("argparse admitted an unknown command")
        atomic_write_json(args.output, output)
    except (OSError, UnicodeError, ResCompError, ValueError) as exc:
        print(f"frontier_rescomp: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
