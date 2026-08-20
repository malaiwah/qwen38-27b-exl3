#!/usr/bin/env python3
"""Enumerate the exact nondominated frontier of a sparse allocation domain.

The allocator consumes an immutable BF16 census, its measured compatibility
registry, and integer direct marginals.  It uses deterministic single-worker
OR-Tools CP-SAT from an explicitly pinned solver-environment manifest.  It never
forms a Cartesian option table and never uses a weighted or otherwise retunable
scalar objective: each candidate is polished by exact lexicographic
single-coordinate solves, then excluded with a Pareto dominance cut.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from frontier_common import (
    atomic_write_json,
    canonical_sha256,
    load_strict_json,
)


CENSUS_SCHEMA = "qwen38-frontier-bf16-census/1"
REGISTRY_SCHEMA = "qwen38-frontier-compatibility-registry/1"
MARGINALS_SCHEMA = "qwen38-frontier-direct-marginals/1"
SOLVER_ENV_SCHEMA = "qwen38-frontier-solver-environment/1"
OUTPUT_SCHEMA = "qwen38-frontier-allocation/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QKV_RE = re.compile(r"^(.*\.self_attn)\.(q_proj|k_proj|v_proj)$")
RESOURCE_NAMES = (
    "disk_bytes",
    "resident_bytes",
    "scratch_bytes",
    "graph_bytes",
    "mtp_bytes",
    "kv_bytes",
    "startup_us",
)
RESOURCE_AGGREGATION = {
    "disk_bytes": "sum",
    "resident_bytes": "sum",
    "scratch_bytes": "max",
    "graph_bytes": "sum",
    "mtp_bytes": "sum",
    "kv_bytes": "sum",
    "startup_us": "sum",
}
RESOURCE_UNITS = {
    "disk_bytes": "bytes",
    "resident_bytes": "bytes",
    "scratch_bytes": "bytes",
    "graph_bytes": "bytes",
    "mtp_bytes": "bytes",
    "kv_bytes": "bytes",
    "startup_us": "microseconds",
}
INT64_SAFE = (1 << 62) - 1


class AllocationError(ValueError):
    """A closed validation or exact-solver failure."""


def fail(message: str) -> NoReturn:
    raise AllocationError(message)

class CpSolverProtocol(Protocol):
    parameters: Any

    def Solve(self, model: Any) -> int:
        ...

    def StatusName(self, status: int) -> str:
        ...

    def Value(self, expression: Any) -> int:
        ...


class CpModelModuleProtocol(Protocol):
    CHOOSE_FIRST: int
    SELECT_MIN_VALUE: int
    FIXED_SEARCH: int
    INFEASIBLE: int
    OPTIMAL: int

    def CpModel(self) -> Any:
        ...

    def CpSolver(self) -> CpSolverProtocol:
        ...


def object_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def list_value(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a JSON array")
    return cast(list[Any], value)


def exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - value.keys())
    unknown = sorted(value.keys() - fields)
    if missing or unknown:
        fail(f"{label} has missing={missing} unknown={unknown}")


def known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a known, non-empty string")
    return value.strip()


def sha256(value: object, label: str) -> str:
    text = known_string(value, label).lower()
    if not SHA256_RE.fullmatch(text):
        fail(f"{label} must be a lowercase SHA256")
    return text


def integer(value: object, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if nonnegative and value < 0:
        fail(f"{label} must be nonnegative")
    if abs(value) > INT64_SAFE:
        fail(f"{label} exceeds the exact CP-SAT integer range")
    return value


def finite_number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    if not math.isfinite(value):
        fail(f"{label} must be finite")
    return value


def reject_expansion_or_topology(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if "cartesian" in normalized or normalized in {
                "option_bank",
                "qkv_widths",
                "mixed_width_qkv",
                "fused_qkv",
            }:
                prohibition_flags = {
                    "cartesian_bank",
                    "cartesian_expansion",
                    "mixed_width_qkv",
                    "fused_qkv",
                }
                if normalized not in prohibition_flags or child is not False:
                    fail(f"{label} contains forbidden field/value {key!r}")
            if normalized == "topology" and isinstance(child, str):
                topology = child.lower().replace("-", "_")
                if topology != "split_qkv":
                    fail(f"{label} declares unsupported topology {child!r}")
            reject_expansion_or_topology(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_expansion_or_topology(child, f"{label}[{index}]")


def load_input(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"{label} is missing or not a file")
    if path.stat().st_size == 0:
        fail(f"{label} is empty")
    return object_value(load_strict_json(path), label)


def validate_output_path(path: Path) -> None:
    if path.exists():
        if not path.is_file():
            fail("output path exists and is not a file")
        if path.stat().st_size:
            fail("output path is nonempty; refusing to replace an allocation artifact")


def validate_census(census: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if census.get("schema") != CENSUS_SCHEMA:
        fail(f"unsupported census schema {census.get('schema')!r}")
    reject_expansion_or_topology(census, "census")
    source = object_value(census.get("source"), "census source")
    revision = known_string(source.get("revision"), "census source revision")
    tensors = object_value(census.get("tensors"), "census tensors")
    if not tensors:
        fail("census tensor inventory is empty")
    exl3: dict[str, Any] = {}
    for tensor_name, raw_tensor in tensors.items():
        tensor = object_value(raw_tensor, f"census tensor {tensor_name}")
        target = object_value(tensor.get("target"), f"census tensor {tensor_name} target")
        if target.get("format") != "exl3":
            continue
        if not tensor_name.endswith(".weight"):
            fail(f"EXL3 census tensor {tensor_name!r} is not a weight tensor")
        module = tensor_name.removesuffix(".weight")
        shape = list_value(tensor.get("shape"), f"census tensor {tensor_name} shape")
        if len(shape) != 2 or any(integer(v, f"census tensor {tensor_name} shape", nonnegative=True) == 0 for v in shape):
            fail(f"EXL3 census tensor {tensor_name!r} must have a positive 2-D shape")
        exl3[module] = {"tensor": tensor_name, "shape": shape, "target": target}
    if not exl3:
        fail("census contains no EXL3 allocation modules")
    return exl3, {"revision": revision, "canonical_sha256": canonical_sha256(census)}


def option_identity(module: str, option: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = {
        "module": module,
        "K": option["K"],
        "codebook": option["codebook"],
        "scale_mode": option["scale_mode"],
        "topology": option["topology"],
        "registry_entries": option["registry_entries"],
    }
    return canonical_sha256(identity), identity


def validate_registry(
    registry: dict[str, Any],
    census_modules: dict[str, Any],
    census_identity: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        fail(f"unsupported compatibility registry schema {registry.get('schema')!r}")
    reject_expansion_or_topology(registry, "compatibility registry")
    source = object_value(registry.get("source"), "registry source")
    if sha256(source.get("census_sha256"), "registry census_sha256") != census_identity["canonical_sha256"]:
        fail("registry source does not bind the supplied canonical census")
    if source.get("revision") != census_identity["revision"]:
        fail("registry and census revisions disagree")
    policy = object_value(registry.get("policy"), "registry policy")
    if policy.get("topology") != "split_qkv":
        fail("registry policy topology must be split_qkv")
    if policy.get("cartesian_bank") is not False or policy.get("mixed_width_qkv") is not False:
        fail("registry must explicitly prohibit Cartesian and mixed-width QKV banks")
    if policy.get("measured_routes_only") is not True:
        fail("registry does not require measured routes")

    entries = object_value(registry.get("entries"), "registry entries")
    modules = object_value(registry.get("modules"), "registry modules")
    if set(modules) != set(census_modules):
        missing = sorted(set(census_modules) - set(modules))
        unknown = sorted(set(modules) - set(census_modules))
        fail(f"registry module inventory is incomplete: missing={missing} unknown={unknown}")

    normalized: dict[str, list[dict[str, Any]]] = {}
    used_entries: set[str] = set()
    for module in sorted(modules):
        record = object_value(modules[module], f"registry module {module}")
        if record.get("source_tensor") != census_modules[module]["tensor"]:
            fail(f"registry module {module} source tensor disagrees with census")
        if record.get("shape") != census_modules[module]["shape"]:
            fail(f"registry module {module} shape disagrees with census")
        raw_options = list_value(record.get("options"), f"registry module {module} options")
        if not raw_options:
            fail(f"registry module {module} has no choices")
        option_ids: set[str] = set()
        signatures: set[tuple[Any, ...]] = set()
        normalized[module] = []
        for index, raw_option in enumerate(raw_options):
            option = object_value(raw_option, f"registry option {module}[{index}]")
            required = {
                "K",
                "codebook",
                "scale_mode",
                "topology",
                "selection_basis",
                "proxy_error",
                "registry_entries",
            }
            exact_fields(option, required, f"registry option {module}[{index}]")
            k = integer(option["K"], f"registry option {module}[{index}] K", nonnegative=True)
            if k not in {3, 4, 5, 6, 7}:
                fail(f"registry option {module}[{index}] K must be in 3..7")
            codebook = known_string(option["codebook"], f"registry option {module}[{index}] codebook")
            scale_mode = known_string(option["scale_mode"], f"registry option {module}[{index}] scale_mode")
            topology = known_string(option["topology"], f"registry option {module}[{index}] topology").lower().replace("-", "_")
            if topology != "split_qkv":
                fail(f"registry option {module}[{index}] uses unsupported fused topology")
            known_string(option["selection_basis"], f"registry option {module}[{index}] selection_basis")
            finite_number(option["proxy_error"], f"registry option {module}[{index}] proxy_error")
            entry_ids = sorted(
                known_string(value, f"registry option {module}[{index}] route")
                for value in list_value(option["registry_entries"], f"registry option {module}[{index}] routes")
            )
            if not entry_ids or len(entry_ids) != len(set(entry_ids)):
                fail(f"registry option {module}[{index}] has missing or duplicate routes")
            normalized_option = {
                **option,
                "K": k,
                "codebook": codebook,
                "scale_mode": scale_mode,
                "topology": topology,
                "registry_entries": entry_ids,
            }
            option_id, identity = option_identity(module, normalized_option)
            signature = (k, codebook, scale_mode, topology)
            if option_id in option_ids or signature in signatures:
                fail(f"registry module {module} contains duplicate module choices")
            option_ids.add(option_id)
            signatures.add(signature)
            route_modes: dict[str, set[str]] = {}
            for entry_id in entry_ids:
                if entry_id not in entries:
                    fail(f"registry option {module}[{index}] names missing route {entry_id}")
                used_entries.add(entry_id)
                entry = object_value(entries[entry_id], f"registry entry {entry_id}")
                exact_fields(entry, {"key", "observation"}, f"registry entry {entry_id}")
                if canonical_sha256(object_value(entry.get("key"), f"registry entry {entry_id} key")) != entry_id:
                    fail(f"registry entry {entry_id} is not its canonical key digest")
                key = entry["key"]
                exact_fields(
                    key,
                    {
                        "runtime_sha",
                        "sm",
                        "K",
                        "codebook_markers",
                        "shape",
                        "alignment",
                        "N",
                        "row_class",
                        "graph_mode",
                    },
                    f"registry entry {entry_id} key",
                )
                observation = object_value(entry.get("observation"), f"registry entry {entry_id} observation")
                exact_fields(
                    observation,
                    {
                        "route",
                        "scale_mode",
                        "topology",
                        "resources",
                        "fallback",
                        "measurement_kind",
                    },
                    f"registry entry {entry_id} observation",
                )
                if key.get("K") != k or key.get("shape") != record.get("shape"):
                    fail(f"registry route {entry_id} does not match option {module}[{index}]")
                markers = object_value(key.get("codebook_markers"), f"registry entry {entry_id} codebook markers")
                if markers.get("codebook") != codebook:
                    fail(f"registry route {entry_id} codebook disagrees with option")
                if observation.get("scale_mode") != scale_mode or observation.get("topology") != "split_qkv":
                    fail(f"registry route {entry_id} scale/topology disagrees with option")
                if observation.get("measurement_kind") != "measured":
                    fail(f"registry route {entry_id} is modeled or incomplete")
                known_string(observation.get("route"), f"registry entry {entry_id} route")
                fallback = object_value(observation.get("fallback"), f"registry entry {entry_id} fallback")
                exact_fields(
                    fallback,
                    {"observed", "route", "measured"},
                    f"registry entry {entry_id} fallback",
                )
                if fallback.get("measured") is not True or not isinstance(fallback.get("observed"), bool):
                    fail(f"registry route {entry_id} has a modeled fallback")
                if fallback["observed"]:
                    known_string(fallback.get("route"), f"registry entry {entry_id} fallback route")
                elif fallback.get("route") is not None:
                    fail(f"registry route {entry_id} has a fallback route without an observation")
                resources = object_value(observation.get("resources"), f"registry entry {entry_id} resources")
                exact_fields(
                    resources,
                    {"latency_us", "scratch_bytes", "jit_ms", "startup_ms"},
                    f"registry entry {entry_id} resources",
                )
                for field in ("latency_us", "scratch_bytes", "jit_ms", "startup_ms"):
                    if field == "scratch_bytes":
                        integer(
                            resources.get(field),
                            f"registry entry {entry_id} resource {field}",
                            nonnegative=True,
                        )
                    else:
                        measured = finite_number(
                            resources.get(field), f"registry entry {entry_id} resource {field}"
                        )
                        if measured < 0:
                            fail(f"registry entry {entry_id} resource {field} is negative")
                graph_mode = known_string(key.get("graph_mode"), f"registry entry {entry_id} graph_mode")
                row_class = known_string(key.get("row_class"), f"registry entry {entry_id} row_class").lower()
                if row_class not in {"decode", "prefill"}:
                    fail(f"registry route {entry_id} has unknown row class")
                route_modes.setdefault(graph_mode, set()).add(row_class)
            normalized_option["option_id"] = option_id
            normalized_option["identity"] = identity
            normalized_option["route_modes"] = route_modes
            normalized[module].append(normalized_option)
    if used_entries != set(entries):
        fail("registry contains unused Cartesian or unbound route entries")
    return normalized, {
        "canonical_sha256": canonical_sha256(registry),
        "revision": source["revision"],
    }


def validate_resource_model(value: object) -> tuple[dict[str, dict[str, Any]], list[str]]:
    model = object_value(value, "direct marginal resource_model")
    exact_fields(model, {"dimensions", "legal_graph_modes", "legal_topologies"}, "direct marginal resource_model")
    topologies = list_value(model["legal_topologies"], "legal_topologies")
    if topologies != ["split_qkv"]:
        fail("legal_topologies must contain only split_qkv")
    graph_modes = [known_string(mode, "legal graph mode") for mode in list_value(model["legal_graph_modes"], "legal_graph_modes")]
    if not graph_modes or len(graph_modes) != len(set(graph_modes)):
        fail("legal_graph_modes must be nonempty and duplicate-free")
    dimensions = object_value(model["dimensions"], "resource dimensions")
    if set(dimensions) != set(RESOURCE_NAMES):
        fail(f"resource dimensions must be exactly {list(RESOURCE_NAMES)}")
    normalized: dict[str, dict[str, Any]] = {}
    for name in RESOURCE_NAMES:
        dimension = object_value(dimensions[name], f"resource dimension {name}")
        exact_fields(dimension, {"aggregation", "unit", "fixed", "limit"}, f"resource dimension {name}")
        if dimension["aggregation"] != RESOURCE_AGGREGATION[name]:
            fail(f"resource dimension {name} must aggregate as {RESOURCE_AGGREGATION[name]}")
        if dimension["unit"] != RESOURCE_UNITS[name]:
            fail(f"resource dimension {name} must use unit {RESOURCE_UNITS[name]}")
        fixed = integer(dimension["fixed"], f"resource dimension {name} fixed", nonnegative=True)
        limit = integer(dimension["limit"], f"resource dimension {name} limit", nonnegative=True)
        if fixed > limit:
            fail(f"fixed {name} already exceeds its exact limit")
        normalized[name] = {**dimension, "fixed": fixed, "limit": limit}
    return normalized, graph_modes


def validate_marginals(
    marginals: dict[str, Any],
    registry_options: dict[str, list[dict[str, Any]]],
    census_identity: dict[str, Any],
    registry_identity: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]], list[str]]:
    if marginals.get("schema") != MARGINALS_SCHEMA:
        fail(f"unsupported direct-marginals schema {marginals.get('schema')!r}")
    reject_expansion_or_topology(marginals, "direct marginals")
    exact_fields(marginals, {"schema", "source", "objectives", "resource_model", "modules"}, "direct marginals")
    source = object_value(marginals["source"], "direct marginals source")
    exact_fields(source, {"census_sha256", "registry_sha256"}, "direct marginals source")
    if sha256(source["census_sha256"], "direct marginal census_sha256") != census_identity["canonical_sha256"]:
        fail("direct marginals do not bind the supplied census")
    if sha256(source["registry_sha256"], "direct marginal registry_sha256") != registry_identity["canonical_sha256"]:
        fail("direct marginals do not bind the supplied registry")

    objectives: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(list_value(marginals["objectives"], "direct marginal objectives")):
        objective = object_value(raw, f"objective {index}")
        exact_fields(objective, {"name", "sense", "unit"}, f"objective {index}")
        name = known_string(objective["name"], f"objective {index} name")
        if name in seen_names:
            fail(f"duplicate objective {name!r}")
        seen_names.add(name)
        sense = known_string(objective["sense"], f"objective {name} sense").lower()
        if sense not in {"minimize", "maximize"}:
            fail(f"objective {name} sense must be minimize or maximize")
        objectives.append({"name": name, "sense": sense, "unit": known_string(objective["unit"], f"objective {name} unit")})
    if not objectives:
        fail("at least one direct-marginal objective is required")
    objectives.sort(key=lambda item: item["name"])

    dimensions, graph_modes = validate_resource_model(marginals["resource_model"])
    modules = object_value(marginals["modules"], "direct marginal modules")
    if set(modules) != set(registry_options):
        missing = sorted(set(registry_options) - set(modules))
        unknown = sorted(set(modules) - set(registry_options))
        fail(f"direct marginals are incomplete: missing={missing} unknown={unknown}")
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for module in sorted(modules):
        rows = list_value(modules[module], f"direct marginal module {module}")
        expected = {option["option_id"] for option in registry_options[module]}
        normalized[module] = {}
        for index, raw in enumerate(rows):
            row = object_value(raw, f"direct marginal {module}[{index}]")
            exact_fields(row, {"option_id", "resources", "direct_marginals"}, f"direct marginal {module}[{index}]")
            option_id = sha256(row["option_id"], f"direct marginal {module}[{index}] option_id")
            if option_id in normalized[module]:
                fail(f"duplicate direct marginal module choice {module} {option_id}")
            resources = object_value(row["resources"], f"direct marginal {module}[{index}] resources")
            if set(resources) != set(RESOURCE_NAMES):
                fail(f"direct marginal {module}[{index}] has incomplete resources")
            normalized_resources = {
                name: integer(resources[name], f"direct marginal {module}[{index}] resource {name}", nonnegative=True)
                for name in RESOURCE_NAMES
            }
            values = object_value(row["direct_marginals"], f"direct marginal {module}[{index}] values")
            if set(values) != seen_names:
                fail(f"direct marginal {module}[{index}] has incomplete objective values")
            normalized_values = {
                name: integer(values[name], f"direct marginal {module}[{index}] objective {name}")
                for name in seen_names
            }
            normalized[module][option_id] = {
                "resources": normalized_resources,
                "direct_marginals": normalized_values,
            }
        if set(normalized[module]) != expected:
            missing = sorted(expected - set(normalized[module]))
            unknown = sorted(set(normalized[module]) - expected)
            fail(f"direct marginal options for {module} mismatch registry: missing={missing} unknown={unknown}")
        for option in registry_options[module]:
            if set(option["route_modes"]) != set(graph_modes):
                fail(f"option {module} {option['option_id']} has unsupported or missing graph modes")
            for graph_mode in graph_modes:
                if option["route_modes"].get(graph_mode) != {"decode", "prefill"}:
                    fail(f"option {module} {option['option_id']} lacks complete {graph_mode} decode/prefill routes")

    for name, dimension in dimensions.items():
        if dimension["aggregation"] == "sum":
            maximum = dimension["fixed"] + sum(
                max(normalized[module][option["option_id"]]["resources"][name] for option in registry_options[module])
                for module in registry_options
            )
        else:
            maximum = max(
                [dimension["fixed"]]
                + [
                    normalized[module][option["option_id"]]["resources"][name]
                    for module in registry_options
                    for option in registry_options[module]
                ]
            )
        if maximum > INT64_SAFE:
            fail(f"resource expression {name} can overflow exact CP-SAT integers")
    for objective in objectives:
        name = objective["name"]
        bound = sum(
            max(abs(normalized[module][option["option_id"]]["direct_marginals"][name]) for option in registry_options[module])
            for module in registry_options
        )
        if bound > INT64_SAFE:
            fail(f"objective expression {name} can overflow exact CP-SAT integers")
    return objectives, normalized, dimensions, graph_modes


def validate_solver_environment(environment: dict[str, Any]) -> CpModelModuleProtocol:
    if environment.get("schema") != SOLVER_ENV_SCHEMA:
        fail(f"unsupported solver environment schema {environment.get('schema')!r}")
    exact_fields(environment, {"schema", "solver", "lock_sha256"}, "solver environment")
    sha256(environment["lock_sha256"], "solver environment lock_sha256")
    solver_spec = object_value(environment["solver"], "solver environment solver")
    exact_fields(solver_spec, {"name", "package", "version", "parameters"}, "solver environment solver")
    if solver_spec["name"] != "ortools-cp-sat" or solver_spec["package"] != "ortools":
        fail("solver environment must pin ortools-cp-sat from package ortools")
    version = known_string(solver_spec["version"], "solver version")
    parameters = object_value(solver_spec["parameters"], "solver parameters")
    exact_fields(parameters, {"num_search_workers", "random_seed", "search_branching"}, "solver parameters")
    if parameters != {"num_search_workers": 1, "random_seed": 0, "search_branching": "FIXED_SEARCH"}:
        fail("solver environment must pin deterministic single-worker FIXED_SEARCH")
    installed: str | None = None
    try:
        installed = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError:
        fail("pinned exact solver is absent; activate the declared campaign environment")
    if installed is None:
        raise AssertionError("unreachable solver package state")
    if installed != version:
        fail(f"installed ortools {installed!r} does not match pinned version {version!r}")
    cp_model: CpModelModuleProtocol | None = None
    try:
        cp_model = cast(
            CpModelModuleProtocol,
            import_module("ortools.sat.python.cp_model"),
        )
    except (ImportError, OSError) as exc:
        fail(f"pinned OR-Tools CP-SAT solver is unavailable: {exc}")
    if cp_model is None:
        raise AssertionError("unreachable CP-SAT module state")
    return cp_model


def qkv_groups(modules: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    groups: dict[str, dict[str, str]] = {}
    for module in modules:
        match = QKV_RE.fullmatch(module)
        if match:
            groups.setdefault(match.group(1), {})[match.group(2)] = module
    result: list[list[str]] = []
    for prefix, projections in sorted(groups.items()):
        if set(projections) != {"q_proj", "k_proj", "v_proj"}:
            fail(f"incomplete fusion-aware QKV group {prefix}")
        result.append([projections[name] for name in ("q_proj", "k_proj", "v_proj")])
    return result


def solve_frontier(
    cp_model: CpModelModuleProtocol,
    registry_options: dict[str, list[dict[str, Any]]],
    marginal_rows: dict[str, dict[str, dict[str, Any]]],
    objectives: list[dict[str, str]],
    dimensions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    modules = sorted(registry_options)
    groups = qkv_groups(registry_options)
    objective_names = [objective["name"] for objective in objectives]
    signs = {objective["name"]: (1 if objective["sense"] == "minimize" else -1) for objective in objectives}
    dominance_cuts: list[dict[str, int]] = []

    def build_model(
        upper_bounds: dict[str, int] | None = None,
        equalities: dict[str, int] | None = None,
    ) -> tuple[Any, dict[tuple[str, str], Any], dict[str, Any], dict[str, Any]]:
        model = cp_model.CpModel()
        variables: dict[tuple[str, str], Any] = {}
        ordered_vars = []
        for module in modules:
            module_vars = []
            for option in sorted(registry_options[module], key=lambda row: row["option_id"]):
                variable = model.NewBoolVar(f"choose_{module}_{option['option_id']}")
                variables[(module, option["option_id"])] = variable
                module_vars.append(variable)
                ordered_vars.append(variable)
            model.AddExactlyOne(module_vars)
        model.AddDecisionStrategy(ordered_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

        signatures = sorted({
            (option["K"], option["codebook"], option["scale_mode"], option["topology"])
            for options in registry_options.values()
            for option in options
        })
        signature_index = {signature: index for index, signature in enumerate(signatures)}
        for group in groups:
            expressions = []
            for module in group:
                expressions.append(sum(
                    signature_index[(option["K"], option["codebook"], option["scale_mode"], option["topology"])]
                    * variables[(module, option["option_id"])]
                    for option in registry_options[module]
                ))
            for expression in expressions[1:]:
                model.Add(expression == expressions[0])

        resource_expressions: dict[str, Any] = {}
        for name, dimension in dimensions.items():
            if dimension["aggregation"] == "sum":
                expression = dimension["fixed"] + sum(
                    marginal_rows[module][option["option_id"]]["resources"][name]
                    * variables[(module, option["option_id"])]
                    for module in modules
                    for option in registry_options[module]
                )
                model.Add(expression <= dimension["limit"])
                resource_expressions[name] = expression
            else:
                if dimension["fixed"] > dimension["limit"]:
                    fail(f"fixed {name} exceeds limit")
                for module in modules:
                    for option in registry_options[module]:
                        value = marginal_rows[module][option["option_id"]]["resources"][name]
                        if value > dimension["limit"]:
                            model.Add(variables[(module, option["option_id"])] == 0)
                resource_expressions[name] = None

        objective_expressions = {
            name: signs[name] * sum(
                marginal_rows[module][option["option_id"]]["direct_marginals"][name]
                * variables[(module, option["option_id"])]
                for module in modules
                for option in registry_options[module]
            )
            for name in objective_names
        }
        for cut_index, point in enumerate(dominance_cuts):
            better = []
            for name in objective_names:
                literal = model.NewBoolVar(f"cut_{cut_index}_{name}")
                model.Add(objective_expressions[name] <= point[name] - 1).OnlyEnforceIf(literal)
                better.append(literal)
            model.AddBoolOr(better)
        if upper_bounds:
            for name, value in upper_bounds.items():
                model.Add(objective_expressions[name] <= value)
        if equalities:
            for name, value in equalities.items():
                model.Add(objective_expressions[name] == value)
        return model, variables, objective_expressions, resource_expressions

    def new_solver() -> CpSolverProtocol:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        return solver

    frontier: list[dict[str, Any]] = []
    while True:
        model, _, expressions, _ = build_model()
        solver = new_solver()
        status = solver.Solve(model)
        if status == cp_model.INFEASIBLE:
            break
        if status != cp_model.OPTIMAL:
            fail(f"exact feasibility solve ended with non-final status {solver.StatusName(status)}")
        candidate = {name: solver.Value(expressions[name]) for name in objective_names}

        fixed: dict[str, int] = {}
        polished_solver: CpSolverProtocol | None = None
        polished_variables: dict[tuple[str, str], Any] | None = None
        for name in objective_names:
            polished_model, polished_variables, polished_expressions, _ = build_model(
                upper_bounds=candidate,
                equalities=fixed,
            )
            polished_model.Minimize(polished_expressions[name])
            polished_solver = new_solver()
            polished_status = polished_solver.Solve(polished_model)
            if polished_status != cp_model.OPTIMAL:
                fail(f"exact Pareto coordinate solve ended with {polished_solver.StatusName(polished_status)}")
            fixed[name] = polished_solver.Value(polished_expressions[name])
        if polished_solver is None:
            fail("no objectives were available for exact Pareto polishing")
        if polished_variables is None:
            fail("no objectives were available for exact Pareto polishing")

        choices: dict[str, dict[str, Any]] = {}
        totals: dict[str, int] = {}
        for module in modules:
            selected = [
                option
                for option in registry_options[module]
                if polished_solver.Value(polished_variables[(module, option["option_id"])]) == 1
            ]
            if len(selected) != 1:
                fail(f"exact solver returned {len(selected)} choices for module {module}")
            option = selected[0]
            choices[module] = {
                "option_id": option["option_id"],
                "K": option["K"],
                "codebook": option["codebook"],
                "scale_mode": option["scale_mode"],
                "topology": option["topology"],
                "registry_entries": option["registry_entries"],
            }
        for name, dimension in dimensions.items():
            values = [
                marginal_rows[module][choices[module]["option_id"]]["resources"][name]
                for module in modules
            ]
            if dimension["aggregation"] == "sum":
                totals[name] = dimension["fixed"] + sum(values)
            else:
                totals[name] = max([dimension["fixed"]] + values)
            if totals[name] > dimension["limit"]:
                fail(f"solver output violates exact {name} constraint")
        objective_values = {name: signs[name] * fixed[name] for name in objective_names}
        solution_body = {
            "objectives": objective_values,
            "resources": totals,
            "choices": choices,
        }
        frontier.append({
            "solution_id": canonical_sha256(solution_body),
            **solution_body,
        })
        dominance_cuts.append(dict(fixed))

    frontier.sort(key=lambda solution: tuple(
        signs[name] * solution["objectives"][name] for name in objective_names
    ))
    return frontier


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Enumerate exact nondominated sparse frontier allocations with pinned OR-Tools CP-SAT."
    )
    result.add_argument("--census", required=True, type=Path, help="immutable BF16 census JSON")
    result.add_argument("--registry", required=True, type=Path, help="measured route/resource compatibility registry JSON")
    result.add_argument("--direct-marginals", required=True, type=Path, help="integer per-option resource and objective marginals JSON")
    result.add_argument("--solver-environment", required=True, type=Path, help="separately pinned campaign solver environment JSON")
    result.add_argument("--out", required=True, type=Path, help="new allocation frontier JSON (must be absent or empty)")
    return result


def run(args: argparse.Namespace) -> None:
    validate_output_path(args.out)
    census = load_input(args.census, "census")
    registry = load_input(args.registry, "compatibility registry")
    marginals = load_input(args.direct_marginals, "direct marginals")
    environment = load_input(args.solver_environment, "solver environment")
    census_modules, census_identity = validate_census(census)
    registry_options, registry_identity = validate_registry(
        registry, census_modules, census_identity
    )
    objectives, marginal_rows, dimensions, graph_modes = validate_marginals(
        marginals,
        registry_options,
        census_identity,
        registry_identity,
    )
    cp_model = validate_solver_environment(environment)
    frontier = solve_frontier(
        cp_model,
        registry_options,
        marginal_rows,
        objectives,
        dimensions,
    )
    if not frontier:
        fail("exact constraints admit no legal allocation")
    output = {
        "schema": OUTPUT_SCHEMA,
        "source": {
            "revision": census_identity["revision"],
            "census_sha256": census_identity["canonical_sha256"],
            "registry_sha256": registry_identity["canonical_sha256"],
            "direct_marginals_sha256": canonical_sha256(marginals),
            "solver_environment_sha256": canonical_sha256(environment),
        },
        "method": {
            "solver": "ortools-cp-sat",
            "exact_integer_model": True,
            "num_search_workers": 1,
            "random_seed": 0,
            "search_branching": "FIXED_SEARCH",
            "scalarized_objective": False,
            "modeled_fallback": False,
            "cartesian_expansion": False,
            "topology": "split_qkv",
            "legal_graph_modes": graph_modes,
            "pareto_equivalence": "one deterministic representative per distinct nondominated objective vector",
        },
        "objectives": objectives,
        "resource_constraints": dimensions,
        "summary": {
            "module_count": len(registry_options),
            "option_count": sum(len(options) for options in registry_options.values()),
            "pareto_solution_count": len(frontier),
        },
        "solutions": frontier,
    }
    atomic_write_json(args.out, output)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except (AllocationError, OSError, ValueError) as exc:
        print(f"frontier_allocate: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
