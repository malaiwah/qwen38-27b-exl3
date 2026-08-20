#!/usr/bin/env python3
"""Build and verify explicit per-layer EXL3 QKV topology metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "exl3_qkv_topology/1"
REGISTRY_SCHEMA = "exl3_qkv_registry/1"
COMPONENTS = ["q_proj", "k_proj", "v_proj"]
QWEN38_OUTPUT_SPLITS = [12288, 1024, 1024]
SCALE_POLICIES = {"always", "never", "auto"}
VARIANTS = {"split", "fused_uniform"}


class TopologyError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise TopologyError(message)


def object_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def list_value(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a list")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        fail(f"{label} has missing={missing} unknown={unknown}")


def known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a non-empty string")
    return value


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{label} must be a positive integer")
    return value


def projection(value: object, expected_name: str, label: str) -> dict[str, Any]:
    item = object_value(value, label)
    exact_keys(item, {"name", "K", "codebook", "scale"}, label)
    name = known_string(item["name"], f"{label}.name")
    if name != expected_name:
        fail(f"{label}.name must be {expected_name!r}, got {name!r}")
    width = positive_int(item["K"], f"{label}.K")
    if width not in range(3, 9):
        fail(f"{label}.K must be in 3..8")
    codebook = known_string(item["codebook"], f"{label}.codebook")
    if codebook not in {"mcg", "mul1"}:
        fail(f"{label}.codebook must be 'mcg' or 'mul1'")
    scale = known_string(item["scale"], f"{label}.scale")
    if scale not in SCALE_POLICIES:
        fail(f"{label}.scale must be one of {sorted(SCALE_POLICIES)}")
    return {"name": name, "K": width, "codebook": codebook, "scale": scale}


def storage_codebook(entry: dict[str, Any], label: str) -> str:
    stored = object_value(entry.get("stored_tensors"), f"{label}.stored_tensors")
    suffixes = {
        known_string(name, f"{label} tensor name").rsplit(".", 1)[-1]
        for name in stored
    }
    markers = [marker for marker in ("mcg", "mul1") if marker in suffixes]
    if len(markers) != 1:
        fail(f"{label} must carry exactly one mcg/mul1 codebook marker")
    declared = known_string(entry.get("codebook"), f"{label}.codebook")
    if declared != markers[0]:
        fail(f"{label}.codebook disagrees with its marker tensor")
    return declared


def storage_scale(entry: dict[str, Any], label: str) -> str:
    value = known_string(entry.get("scale"), f"{label}.scale")
    if value not in SCALE_POLICIES:
        fail(f"{label}.scale must be one of {sorted(SCALE_POLICIES)}")
    return value


def validate_payload(
    tensor_storage: dict[str, Any], layer: str, spec: dict[str, Any]
) -> None:
    key = f"{layer}.{spec['name']}"
    entry = object_value(tensor_storage.get(key), f"tensor_storage[{key!r}]")
    if entry.get("quant_format") != "exl3":
        fail(f"tensor_storage[{key!r}].quant_format must be 'exl3'")
    if entry.get("bits_per_weight") != spec["K"]:
        fail(f"tensor_storage[{key!r}] K disagrees with topology metadata")
    if storage_codebook(entry, f"tensor_storage[{key!r}]") != spec["codebook"]:
        fail(f"tensor_storage[{key!r}] codebook disagrees with topology metadata")
    if storage_scale(entry, f"tensor_storage[{key!r}]") != spec["scale"]:
        fail(f"tensor_storage[{key!r}] scale policy disagrees with topology metadata")


def normalize_row(value: object, index: int, tensor_storage: dict[str, Any]) -> dict[str, Any]:
    label = f"layers[{index}]"
    row = object_value(value, label)
    common = {"layer", "variant", "components", "output_splits"}
    variant = known_string(row.get("variant"), f"{label}.variant")
    if variant not in VARIANTS:
        fail(
            f"{label}.variant {variant!r} is unsupported; mixed-width one-launch "
            "topology remains unsupported"
        )
    expected = common | ({"projections"} if variant == "split" else {"projection"})
    exact_keys(row, expected, label)

    layer = known_string(row["layer"], f"{label}.layer")
    if not layer.endswith(".self_attn") or ".layers." not in layer:
        fail(f"{label}.layer must name a full-attention self_attn block")
    components = list_value(row["components"], f"{label}.components")
    if components != COMPONENTS:
        fail(f"{label}.components must be {COMPONENTS!r} in output order")
    output_splits = list_value(row["output_splits"], f"{label}.output_splits")
    if output_splits != QWEN38_OUTPUT_SPLITS:
        fail(
            f"{label}.output_splits must be {QWEN38_OUTPUT_SPLITS!r} for Qwen3.8"
        )

    split_keys = [f"{layer}.{name}" for name in COMPONENTS]
    fused_key = f"{layer}.qkv_proj"
    if variant == "split":
        specs = list_value(row["projections"], f"{label}.projections")
        if len(specs) != 3:
            fail(f"{label}.projections must contain q_proj, k_proj, v_proj")
        normalized_specs = [
            projection(item, name, f"{label}.projections[{position}]")
            for position, (item, name) in enumerate(zip(specs, COMPONENTS))
        ]
        if fused_key in tensor_storage:
            fail(f"{label} contains duplicate split and fused payloads")
        missing = [key for key in split_keys if key not in tensor_storage]
        if missing:
            fail(f"{label} split payload is missing {missing}")
        for spec in normalized_specs:
            validate_payload(tensor_storage, layer, spec)
        variant_fields: dict[str, Any] = {"projections": normalized_specs}
    else:
        spec = projection(row["projection"], "qkv_proj", f"{label}.projection")
        duplicates = [key for key in split_keys if key in tensor_storage]
        if duplicates:
            fail(f"{label} contains duplicate split and fused payloads: {duplicates}")
        if fused_key not in tensor_storage:
            fail(f"{label} fused_uniform payload is missing {fused_key!r}")
        validate_payload(tensor_storage, layer, spec)
        variant_fields = {"projection": spec}

    return {
        "layer": layer,
        "variant": variant,
        "components": list(COMPONENTS),
        "output_splits": list(QWEN38_OUTPUT_SPLITS),
        **variant_fields,
    }


def discover_payload_layers(tensor_storage: dict[str, Any]) -> set[str]:
    suffixes = tuple(f".{name}" for name in (*COMPONENTS, "qkv_proj"))
    return {
        key[: -len(suffix)]
        for key in tensor_storage
        for suffix in suffixes
        if key.endswith(suffix) and key[: -len(suffix)].endswith(".self_attn")
    }


def validate_document(document: object, tensor_storage: object) -> dict[str, Any]:
    root = object_value(document, "topology")
    exact_keys(root, {"schema", "layers"}, "topology")
    if root["schema"] != SCHEMA:
        fail(f"topology.schema must be {SCHEMA!r}")
    storage = object_value(tensor_storage, "tensor_storage")
    raw_layers = list_value(root["layers"], "topology.layers")
    if not raw_layers:
        fail("topology.layers must not be empty")
    layers = [normalize_row(row, index, storage) for index, row in enumerate(raw_layers)]
    names = [row["layer"] for row in layers]
    if names != sorted(names):
        fail("topology.layers must be sorted by layer")
    if len(names) != len(set(names)):
        fail("topology.layers contains duplicate/mixed metadata for a block")
    discovered = discover_payload_layers(storage)
    declared = set(names)
    if discovered != declared:
        fail(
            "full-attention topology coverage mismatch: "
            f"missing={sorted(discovered - declared)} unknown={sorted(declared - discovered)}"
        )
    return {"schema": SCHEMA, "layers": layers}


def registry_document(document: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in document["layers"]:
        specs = row["projections"] if row["variant"] == "split" else [row["projection"]]
        rows.append(
            {
                "layer": row["layer"],
                "variant": row["variant"],
                "topology": "split_qkv" if row["variant"] == "split" else "fused_uniform_qkv",
                "launches": 3 if row["variant"] == "split" else 1,
                "components": list(row["components"]),
                "output_splits": list(row["output_splits"]),
                "projections": specs,
            }
        )
    return {"schema": REGISTRY_SCHEMA, "rows": rows}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON from {path}: {exc}")


def write_json(path: Path | None, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=False) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def demo_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    def entry(layer: str, name: str, width: int, scale: str) -> tuple[str, dict[str, Any]]:
        key = f"{layer}.{name}"
        return key, {
            "quant_format": "exl3",
            "bits_per_weight": width,
            "codebook": "mcg",
            "scale": scale,
            "stored_tensors": {
                f"{key}.trellis": {},
                f"{key}.suh": {},
                f"{key}.svh": {},
                f"{key}.mcg": {},
            },
        }

    split_layer = "model.language_model.layers.1.self_attn"
    fused_layer = "model.language_model.layers.4.self_attn"
    storage = dict(
        [entry(split_layer, "q_proj", 6, "always")]
        + [entry(split_layer, "k_proj", 5, "never")]
        + [entry(split_layer, "v_proj", 4, "auto")]
        + [entry(fused_layer, "qkv_proj", 6, "always")]
    )
    topology = {
        "schema": SCHEMA,
        "layers": [
            {
                "layer": split_layer,
                "variant": "split",
                "components": list(COMPONENTS),
                "output_splits": list(QWEN38_OUTPUT_SPLITS),
                "projections": [
                    {"name": "q_proj", "K": 6, "codebook": "mcg", "scale": "always"},
                    {"name": "k_proj", "K": 5, "codebook": "mcg", "scale": "never"},
                    {"name": "v_proj", "K": 4, "codebook": "mcg", "scale": "auto"},
                ],
            },
            {
                "layer": fused_layer,
                "variant": "fused_uniform",
                "components": list(COMPONENTS),
                "output_splits": list(QWEN38_OUTPUT_SPLITS),
                "projection": {
                    "name": "qkv_proj",
                    "K": 6,
                    "codebook": "mcg",
                    "scale": "always",
                },
            },
        ],
    }
    return topology, storage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--topology", type=Path, required=True)
        subparser.add_argument("--tensor-storage", type=Path, required=True)
        subparser.add_argument("--output", type=Path)
        subparser.add_argument("--registry-output", type=Path)
    demo = subparsers.add_parser("demo")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--registry-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "demo":
            raw_topology, tensor_storage = demo_checkpoint()
        else:
            raw_topology = read_json(args.topology)
            tensor_storage = read_json(args.tensor_storage)
            if "tensor_storage" in tensor_storage:
                tensor_storage = tensor_storage["tensor_storage"]
        document = validate_document(raw_topology, tensor_storage)
        write_json(args.output, document)
        if args.registry_output is not None:
            write_json(args.registry_output, registry_document(document))
        return 0
    except TopologyError as exc:
        print(f"frontier_qkv_topology: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
