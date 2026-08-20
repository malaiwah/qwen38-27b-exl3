#!/usr/bin/env python3
"""Build exact integer direct marginals for the sparse frontier allocator.

The packed-weight estimate is deliberately conservative.  For every measured
route, input/output dimensions are rounded up to that route's declared K/N
alignment before applying the EXL3 byte law::

    ceil(aligned_in * aligned_out * K / 8)
      + 2 * aligned_in + 2 * aligned_out + 4

The two 2-byte terms are the FP16 ``suh``/``svh`` sign vectors and the final
four bytes are the int32 codebook marker.  Taking the maximum across the exact
measured prefill/decode routes accounts for differing legal alignments without
inventing an option.  Proxy error is scaled by 10**12, latency microseconds by
10**3, and all positive decimal-to-integer conversions round upward so the
artifact never understates a measured quantity.
"""
from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_allocate import (
    AllocationError,
    MARGINALS_SCHEMA,
    RESOURCE_AGGREGATION,
    RESOURCE_NAMES,
    RESOURCE_UNITS,
    exact_fields,
    integer,
    known_string,
    object_value,
    option_identity,
    validate_census,
    validate_registry,
)
from frontier_common import atomic_write_json, load_strict_json, sha256_file


LADDER_SCHEMA = "qwen38-proxy-error-ladder/1"
PROXY_ERROR_SCALE = 10**12
NANOSECONDS_PER_MICROSECOND = 10**3
MICROSECONDS_PER_MILLISECOND = 10**3
GIB = 1 << 30

# Campaign-safe one-card defaults.  The 30.24 GiB engine cap, 0.60 GiB shared
# reconstruct arena, 0.46 GiB decode graph pool, 0.27 GiB MTP payload envelope,
# 9.31 GiB KV pool, and 100.2 s measured startup bound are the conservative
# RTX 5090 campaign envelope.  Disk has no GPU analogue, so it is capped at one
# 32 GiB board image.  Every value remains explicitly overrideable on the CLI.
DEFAULT_LIMITS = {
    "disk_bytes": 32 * GIB,
    "resident_bytes": int(Decimal("30.24") * GIB),
    "scratch_bytes": int(Decimal("0.60") * GIB),
    "graph_bytes": int(Decimal("0.46") * GIB),
    "mtp_bytes": int(Decimal("0.27") * GIB),
    "kv_bytes": int(Decimal("9.31") * GIB),
    "startup_us": 100_200_000,
}


class MarginalError(ValueError):
    """A direct-marginal input is incomplete, inconsistent, or ambiguous."""


def fail(message: str) -> NoReturn:
    raise MarginalError(message)


def list_value(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a JSON array")
    return cast(list[Any], value)


def positive_integer(value: object, label: str) -> int:
    result = integer(value, label, nonnegative=True)
    if result == 0:
        fail(f"{label} must be positive")
    return result


def finite_nonnegative_decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        fail(f"{label} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"{label} must be finite")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise MarginalError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or result < 0:
        fail(f"{label} must be finite and nonnegative")
    return result


def ceil_scaled(value: object, scale: int, label: str) -> int:
    scaled = finite_nonnegative_decimal(value, label) * scale
    result = int(scaled.to_integral_value(rounding=ROUND_CEILING))
    return integer(result, f"scaled {label}", nonnegative=True)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def cli_nonnegative_integer(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a base-10 integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def load_document(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"{label} is missing or not a file")
    if path.stat().st_size == 0:
        fail(f"{label} is empty")
    return object_value(load_strict_json(path), label)


def validate_paths(census: Path, registry: Path, ladder: Path, out: Path) -> None:
    inputs = {census.resolve(), registry.resolve(), ladder.resolve()}
    if len(inputs) != 3:
        fail("census, registry, and ladder must be three distinct files")
    if out.resolve() in inputs:
        fail("output path must not replace an input artifact")
    if out.exists() and not out.is_file():
        fail("output path exists and is not a file")


def validate_tensor_inventory(
    census: dict[str, Any], census_modules: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], int]:
    tensors = object_value(census.get("tensors"), "census tensors")
    if not tensors:
        fail("census tensor inventory is empty")
    normalized: dict[str, dict[str, Any]] = {}
    fixed_bytes = 0
    exl3_modules: set[str] = set()
    for tensor_name in sorted(tensors):
        tensor = object_value(tensors[tensor_name], f"census tensor {tensor_name}")
        shape = [
            positive_integer(value, f"census tensor {tensor_name} shape[{index}]")
            for index, value in enumerate(
                list_value(tensor.get("shape"), f"census tensor {tensor_name} shape")
            )
        ]
        if not shape:
            fail(f"census tensor {tensor_name} has an empty shape")
        numel = positive_integer(tensor.get("numel"), f"census tensor {tensor_name} numel")
        expected_numel = math.prod(shape)
        if numel != expected_numel:
            fail(f"census tensor {tensor_name} numel disagrees with its shape")
        if tensor.get("dtype") != "BF16":
            fail(f"census tensor {tensor_name} is not from the immutable BF16 census")
        byte_count = positive_integer(tensor.get("bytes"), f"census tensor {tensor_name} bytes")
        if byte_count != 2 * numel:
            fail(f"census tensor {tensor_name} BF16 byte count disagrees with its numel")
        target = object_value(tensor.get("target"), f"census tensor {tensor_name} target")
        target_format = known_string(target.get("format"), f"census tensor {tensor_name} target format").lower()
        if target_format == "exl3":
            if not tensor_name.endswith(".weight"):
                fail(f"EXL3 census tensor {tensor_name} is not a weight")
            exl3_modules.add(tensor_name.removesuffix(".weight"))
        elif target_format == "bf16":
            fixed_bytes += byte_count
        else:
            fail(f"census tensor {tensor_name} has unsupported target format {target_format!r}")
        normalized[tensor_name] = {
            "shape": shape,
            "numel": numel,
            "bytes": byte_count,
            "role": tensor.get("role"),
            "target": target,
        }
    if exl3_modules != set(census_modules):
        fail("census EXL3 inventory changed during full tensor validation")
    return normalized, fixed_bytes


def validate_ladder_link(
    ladder_path: Path,
    ladder: dict[str, Any],
    registry: dict[str, Any],
    census: dict[str, Any],
    census_tensors: dict[str, dict[str, Any]],
    registry_options: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Decimal]]:
    if ladder.get("schema") != LADDER_SCHEMA:
        fail(f"unsupported proxy-error ladder schema {ladder.get('schema')!r}")
    registry_source = object_value(registry.get("source"), "registry source")
    census_source = object_value(census.get("source"), "census source")
    if registry_source.get("repo") != census_source.get("repo"):
        fail("registry and census source repositories disagree")
    if registry_source.get("ladder_sha256") != sha256_file(ladder_path):
        fail("registry does not bind the supplied immutable ladder bytes")

    ladder_modules = object_value(ladder.get("modules"), "ladder modules")
    registry_modules = object_value(registry.get("modules"), "registry modules")
    if set(ladder_modules) != set(registry_options):
        missing = sorted(set(registry_options) - set(ladder_modules))
        unknown = sorted(set(ladder_modules) - set(registry_options))
        fail(f"ladder module inventory disagrees with registry: missing={missing} unknown={unknown}")

    proxy_values: dict[str, dict[str, Decimal]] = {}
    for module in sorted(registry_options):
        tensor_name = f"{module}.weight"
        tensor = census_tensors[tensor_name]
        registry_module = object_value(registry_modules[module], f"registry module {module}")
        ladder_module = object_value(ladder_modules[module], f"ladder module {module}")
        in_features = positive_integer(
            ladder_module.get("in_features"), f"ladder module {module} in_features"
        )
        out_features = positive_integer(
            ladder_module.get("out_features"), f"ladder module {module} out_features"
        )
        numel = positive_integer(ladder_module.get("numel"), f"ladder module {module} numel")
        if [out_features, in_features] != tensor["shape"] or numel != tensor["numel"]:
            fail(f"ladder module {module} geometry disagrees with census")
        if registry_module.get("shape") != tensor["shape"] or registry_module.get("numel") != numel:
            fail(f"registry module {module} geometry disagrees with census and ladder")
        if registry_module.get("source_tensor") != tensor_name:
            fail(f"registry module {module} source tensor is not canonical")
        if registry_module.get("role") != tensor.get("role"):
            fail(f"registry module {module} role disagrees with census")
        if registry_module.get("qmap") != ladder_module.get("qmap"):
            fail(f"registry module {module} qmap disagrees with ladder")
        recipe_k = positive_integer(
            ladder_module.get("recipe_bits"), f"ladder module {module} recipe_bits"
        )
        if registry_module.get("incumbent_K") != recipe_k:
            fail(f"registry module {module} incumbent K disagrees with ladder")
        target = tensor["target"]
        if target.get("K") != recipe_k or registry_module.get("policy") != target.get("policy"):
            fail(f"census target for {module} disagrees with ladder/registry incumbent")

        raw_rungs = object_value(ladder_module.get("ladder"), f"ladder module {module} rungs")
        if not raw_rungs:
            fail(f"ladder module {module} has no measured rungs")
        rungs: dict[str, Decimal] = {}
        for raw_k, raw_value in raw_rungs.items():
            try:
                k = int(raw_k, 10)
            except ValueError:
                fail(f"ladder module {module} has non-integer rung {raw_k!r}")
            if str(k) != raw_k or k not in {3, 4, 5, 6, 7, 8}:
                fail(f"ladder module {module} has unsupported rung {raw_k!r}")
            rungs[raw_k] = finite_nonnegative_decimal(
                raw_value, f"ladder module {module} rung K{k}"
            )

        option_ids: set[str] = set()
        for index, option in enumerate(registry_options[module]):
            option_id, identity = option_identity(module, option)
            if option.get("option_id") != option_id or option.get("identity") != identity:
                fail(f"registry option {module}[{index}] canonical identity is inconsistent")
            if option_id in option_ids:
                fail(f"registry module {module} repeats option identity {option_id}")
            option_ids.add(option_id)
            rung = str(option["K"])
            if rung not in rungs:
                fail(f"registry option {module}[{index}] has no measured ladder rung K{rung}")
            option_proxy = finite_nonnegative_decimal(
                option.get("proxy_error"), f"registry option {module}[{index}] proxy_error"
            )
            if option_proxy != rungs[rung]:
                fail(f"registry option {module}[{index}] proxy error disagrees with ladder")
        proxy_values[module] = rungs
    return proxy_values


def route_alignment(key: dict[str, Any], label: str) -> tuple[int, int]:
    alignment = object_value(key.get("alignment"), f"{label} alignment")
    exact_fields(alignment, {"K", "N"}, f"{label} alignment")
    return (
        positive_integer(alignment["K"], f"{label} K alignment"),
        positive_integer(alignment["N"], f"{label} N alignment"),
    )


def packed_bytes(
    module: str,
    shape: list[int],
    k: int,
    entries: list[tuple[str, dict[str, Any]]],
) -> int:
    out_features, in_features = shape
    estimates: list[int] = []
    for entry_id, entry in entries:
        label = f"registry entry {entry_id}"
        key = object_value(entry.get("key"), f"{label} key")
        if key.get("shape") != shape or key.get("K") != k:
            fail(f"{label} geometry/width disagrees with option {module}")
        if positive_integer(key.get("N"), f"{label} N") != out_features:
            fail(f"{label} N disagrees with option {module} output features")
        k_alignment, n_alignment = route_alignment(key, label)
        aligned_in = align_up(in_features, k_alignment)
        aligned_out = align_up(out_features, n_alignment)
        payload_bytes = (aligned_in * aligned_out * k + 7) // 8
        sign_bytes = 2 * aligned_in + 2 * aligned_out
        marker_bytes = 4
        estimates.append(payload_bytes + sign_bytes + marker_bytes)
    if not estimates:
        fail(f"registry option {module} has no measured routes for packing")
    return integer(max(estimates), f"packed bytes for {module}", nonnegative=True)


def measured_routes(
    module: str,
    option: dict[str, Any],
    entries: dict[str, Any],
    graph_modes: list[str],
) -> dict[str, dict[str, tuple[str, dict[str, Any]]]]:
    by_mode: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {
        mode: {} for mode in graph_modes
    }
    for entry_id in option["registry_entries"]:
        entry = object_value(entries.get(entry_id), f"registry entry {entry_id}")
        key = object_value(entry.get("key"), f"registry entry {entry_id} key")
        observation = object_value(
            entry.get("observation"), f"registry entry {entry_id} observation"
        )
        fallback = object_value(
            observation.get("fallback"), f"registry entry {entry_id} fallback"
        )
        if observation.get("measurement_kind") != "measured":
            fail(f"registry entry {entry_id} is not directly measured")
        if fallback.get("measured") is not True or fallback.get("observed") is not False:
            fail(f"registry entry {entry_id} used a fallback")
        if fallback.get("route") is not None:
            fail(f"registry entry {entry_id} names a fallback route")
        mode = known_string(key.get("graph_mode"), f"registry entry {entry_id} graph mode")
        row_class = known_string(key.get("row_class"), f"registry entry {entry_id} row class").lower()
        if mode not in by_mode or row_class not in {"decode", "prefill"}:
            fail(f"registry entry {entry_id} has unsupported route coordinates")
        if row_class in by_mode[mode]:
            fail(f"option {module} has ambiguous duplicate {mode} {row_class} measurements")
        resources = object_value(
            observation.get("resources"), f"registry entry {entry_id} resources"
        )
        for field in ("latency_us", "jit_ms", "startup_ms"):
            finite_nonnegative_decimal(resources.get(field), f"registry entry {entry_id} {field}")
        integer(
            resources.get("scratch_bytes"),
            f"registry entry {entry_id} scratch_bytes",
            nonnegative=True,
        )
        by_mode[mode][row_class] = (entry_id, entry)
    for mode in graph_modes:
        if set(by_mode[mode]) != {"decode", "prefill"}:
            fail(f"option {module} lacks exact measured decode/prefill entries for {mode}")
    return by_mode


def resource_model(args: argparse.Namespace, fixed_payload_bytes: int, graph_modes: list[str]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for name in RESOURCE_NAMES:
        fixed = fixed_payload_bytes if name in {"disk_bytes", "resident_bytes"} else 0
        limit = getattr(args, f"{name}_limit")
        if fixed > limit:
            fail(f"fixed {name} {fixed} exceeds CLI limit {limit}")
        dimensions[name] = {
            "aggregation": RESOURCE_AGGREGATION[name],
            "unit": RESOURCE_UNITS[name],
            "fixed": fixed,
            "limit": limit,
        }
    return {
        "dimensions": dimensions,
        "legal_graph_modes": graph_modes,
        "legal_topologies": ["split_qkv"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    validate_paths(args.census, args.registry, args.ladder, args.out)
    census = load_document(args.census, "census")
    registry = load_document(args.registry, "compatibility registry")
    ladder = load_document(args.ladder, "proxy-error ladder")

    census_modules, census_identity = validate_census(census)
    registry_options, registry_identity = validate_registry(
        registry, census_modules, census_identity
    )
    census_tensors, fixed_payload_bytes = validate_tensor_inventory(census, census_modules)
    proxy_values = validate_ladder_link(
        args.ladder,
        ladder,
        registry,
        census,
        census_tensors,
        registry_options,
    )

    entries = object_value(registry.get("entries"), "registry entries")
    graph_modes = sorted(
        {
            known_string(
                object_value(entry, f"registry entry {entry_id}").get("key", {}).get("graph_mode"),
                f"registry entry {entry_id} graph mode",
            )
            for entry_id, entry in entries.items()
        }
    )
    if not graph_modes:
        fail("registry contains no legal graph modes")

    raw_registry_modules = object_value(registry.get("modules"), "registry modules")
    modules: dict[str, list[dict[str, Any]]] = {}
    for module in sorted(registry_options):
        registry_module = object_value(raw_registry_modules[module], f"registry module {module}")
        shape = [
            positive_integer(value, f"registry module {module} shape[{index}]")
            for index, value in enumerate(
                list_value(registry_module.get("shape"), f"registry module {module} shape")
            )
        ]
        if len(shape) != 2:
            fail(f"registry module {module} must have a two-dimensional shape")
        role = known_string(registry_module.get("role"), f"registry module {module} role")
        rows: list[dict[str, Any]] = []
        for option in sorted(registry_options[module], key=lambda value: value["option_id"]):
            routes = measured_routes(module, option, entries, graph_modes)
            flat_routes = [route for mode in graph_modes for route in routes[mode].values()]
            packed = packed_bytes(module, shape, option["K"], flat_routes)
            scratch = max(
                integer(
                    object_value(entry["observation"], f"registry entry {entry_id} observation")["resources"]["scratch_bytes"],
                    f"registry entry {entry_id} scratch_bytes",
                    nonnegative=True,
                )
                for entry_id, entry in flat_routes
            )
            prefill_latency_ns = max(
                ceil_scaled(
                    object_value(
                        routes[mode]["prefill"][1]["observation"],
                        f"registry entry {routes[mode]['prefill'][0]} observation",
                    )["resources"]["latency_us"],
                    NANOSECONDS_PER_MICROSECOND,
                    f"registry entry {routes[mode]['prefill'][0]} prefill latency_us",
                )
                for mode in graph_modes
            )
            startup_us = max(
                sum(
                    ceil_scaled(
                        object_value(
                            routes[mode][row_class][1]["observation"],
                            f"registry entry {routes[mode][row_class][0]} observation",
                        )["resources"]["startup_ms"],
                        MICROSECONDS_PER_MILLISECOND,
                        f"registry entry {routes[mode][row_class][0]} startup_ms",
                    )
                    for row_class in ("decode", "prefill")
                )
                for mode in graph_modes
            )
            resources = {
                "disk_bytes": packed,
                "resident_bytes": packed,
                "scratch_bytes": scratch,
                "graph_bytes": 0,
                "mtp_bytes": packed if role == "mtp_draft" else 0,
                "kv_bytes": 0,
                "startup_us": startup_us,
            }
            rows.append(
                {
                    "option_id": option["option_id"],
                    "resources": resources,
                    "direct_marginals": {
                        "proxy_error_scaled": ceil_scaled(
                            proxy_values[module][str(option["K"])],
                            PROXY_ERROR_SCALE,
                            f"ladder module {module} rung K{option['K']}",
                        ),
                        "prefill_latency_ns": prefill_latency_ns,
                        "resident_bytes": packed,
                    },
                }
            )
        modules[module] = rows

    return {
        "schema": MARGINALS_SCHEMA,
        "source": {
            "census_sha256": census_identity["canonical_sha256"],
            "registry_sha256": registry_identity["canonical_sha256"],
        },
        "objectives": [
            {"name": "proxy_error_scaled", "sense": "minimize", "unit": "proxy_error_x_1e12"},
            {"name": "prefill_latency_ns", "sense": "minimize", "unit": "nanoseconds"},
            {"name": "resident_bytes", "sense": "minimize", "unit": "bytes"},
        ],
        "resource_model": resource_model(args, fixed_payload_bytes, graph_modes),
        "modules": modules,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build exact integer direct marginals from a BF16 census, measured registry, and proxy-error ladder."
    )
    result.add_argument("--census", required=True, type=Path, help="immutable BF16 census JSON")
    result.add_argument("--registry", required=True, type=Path, help="measured compatibility registry JSON")
    result.add_argument("--ladder", required=True, type=Path, help="immutable proxy-error ladder JSON")
    result.add_argument("--out", required=True, type=Path, help="atomic direct-marginals JSON output")
    for name in RESOURCE_NAMES:
        flag = "--" + name.replace("_", "-") + "-limit"
        result.add_argument(
            flag,
            dest=f"{name}_limit",
            type=cli_nonnegative_integer,
            default=DEFAULT_LIMITS[name],
            help=f"exact {RESOURCE_UNITS[name]} limit (default: {DEFAULT_LIMITS[name]})",
        )
    return result


def run(args: argparse.Namespace) -> None:
    atomic_write_json(args.out, build(args))


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except (AllocationError, MarginalError, OSError, ValueError) as exc:
        print(f"frontier_marginals: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
