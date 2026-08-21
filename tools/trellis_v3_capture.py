#!/usr/bin/env python3
"""Capture real BF16/quant-flow EXL3 fixtures from the pinned converter."""

from __future__ import annotations

import hashlib
import numbers
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from frontier_common import (
    atomic_write_json,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)

PLAN_SCHEMA = "qwen38-trellis-v3-capture-plan/1"
OUTPUT_SCHEMA = "qwen38-trellis-v3-capture/1"
SEED = 1376380199
OWN_FLAGS = {"--v3-plan": "plan", "--v3-out": "out"}


class CaptureError(ValueError):
    """A fail-closed capture or provenance error."""


class CaptureComplete(Exception):
    """Internal non-error used to stop conversion after the target capture."""


def _extract_flags(argv: list[str]) -> tuple[dict[str, Path], list[str]]:
    values: dict[str, Path] = {}
    remaining = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        match = None
        for flag, name in OWN_FLAGS.items():
            if argument == flag:
                if index + 1 >= len(argv):
                    raise CaptureError(f"{flag} requires a path")
                match = (name, argv[index + 1])
                index += 1
                break
            if argument.startswith(flag + "="):
                match = (name, argument[len(flag) + 1 :])
                break
        if match is None:
            remaining.append(argument)
        else:
            name, raw = match
            if name in values or not raw:
                raise CaptureError(f"invalid or duplicate {name} path")
            values[name] = Path(raw)
        index += 1
    missing = sorted({"plan", "out"} - values.keys())
    if missing:
        raise CaptureError(f"missing v3 capture paths: {missing}")
    return values, remaining


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CaptureError(f"{label} must be an object with string keys")
    return value


def _known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureError(f"{label} must be a nonempty string")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CaptureError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    text = _known_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CaptureError(f"{label} must be a lowercase SHA256")
    return text


def _validate_plan(path: Path) -> dict[str, Any]:
    plan = _object(load_strict_json(path), "capture plan")
    expected = {
        "schema",
        "capture_id",
        "flow",
        "target",
        "sample_rows",
        "slices",
        "source",
        "converter",
    }
    if set(plan) != expected:
        raise CaptureError(
            f"capture plan keys differ: missing={sorted(expected - plan.keys())}, "
            f"extra={sorted(plan.keys() - expected)}"
        )
    if plan["schema"] != PLAN_SCHEMA:
        raise CaptureError("unsupported capture plan schema")
    _known_string(plan["capture_id"], "capture_id")
    if plan["flow"] not in {"bf16", "quant"}:
        raise CaptureError("flow must be bf16 or quant")
    target = _object(plan["target"], "target")
    if set(target) != {"linear_key", "tensor_name"}:
        raise CaptureError("target requires exact linear_key and tensor_name")
    _known_string(target["linear_key"], "target.linear_key")
    _known_string(target["tensor_name"], "target.tensor_name")
    _positive_int(plan["sample_rows"], "sample_rows")
    slices = plan["slices"]
    if not isinstance(slices, list) or not slices:
        raise CaptureError("slices must be a nonempty array")
    identifiers: set[str] = set()
    for index, item in enumerate(slices):
        row = _object(item, f"slices[{index}]")
        if set(row) != {"id", "input_start", "output_start", "size"}:
            raise CaptureError(f"slices[{index}] has wrong keys")
        identifier = _known_string(row["id"], f"slices[{index}].id")
        if identifier in identifiers:
            raise CaptureError(f"duplicate slice id {identifier}")
        identifiers.add(identifier)
        size = _positive_int(row["size"], f"slices[{index}].size")
        for field in ("input_start", "output_start"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CaptureError(f"slices[{index}].{field} must be nonnegative")
            if value % 16:
                raise CaptureError(f"slices[{index}].{field} must be 16-aligned")
        if size % 128:
            raise CaptureError(f"slices[{index}].size must be 128-aligned")
    source = _object(plan["source"], "source")
    if set(source) != {"model_root", "model_revision", "index_sha256"}:
        raise CaptureError("source keys differ")
    _known_string(source["model_root"], "source.model_root")
    _known_string(source["model_revision"], "source.model_revision")
    _sha256(source["index_sha256"], "source.index_sha256")
    converter = _object(plan["converter"], "converter")
    if set(converter) != {"commit", "tree", "source_sha256"}:
        raise CaptureError("converter keys differ")
    _known_string(converter["commit"], "converter.commit")
    _known_string(converter["tree"], "converter.tree")
    _sha256(converter["source_sha256"], "converter.source_sha256")
    return plan


_PATHS, sys.argv = _extract_flags(sys.argv)
_PLAN_PATH = _PATHS["plan"].resolve(strict=True)
_OUT_PATH = _PATHS["out"].resolve(strict=False)
if _OUT_PATH.exists():
    raise CaptureError(f"refusing to overwrite output: {_OUT_PATH}")
_PLAN = _validate_plan(_PLAN_PATH)

# The converter source is mounted explicitly by the runbook.
sys.path.insert(0, os.environ.get("TRELLIS_V3_EXLLAMAV3_SOURCE", "/work/exllamav3"))

import torch
from exllamav3.conversion import (  # pyright: ignore[reportMissingImports]
    convert_model as cm,
)
from exllamav3.modules.linear import Linear  # pyright: ignore[reportMissingImports]
from safetensors import safe_open
from safetensors.torch import save_file

_TARGET_KEY = _PLAN["target"]["linear_key"]
_SAMPLE_LIMIT = _PLAN["sample_rows"]
_SAMPLES: list[torch.Tensor] = []
_CAPTURED = False
_ORIGINAL_CAPTURE_H = Linear.capture_H


def _capture_h_with_samples(
    self: Linear, x: torch.Tensor, params: dict[str, Any]
) -> None:
    _ORIGINAL_CAPTURE_H(self, x, params)
    if (
        self.key != _TARGET_KEY
        or sum(value.shape[0] for value in _SAMPLES) >= _SAMPLE_LIMIT
    ):
        return
    flat = x.detach().reshape(-1, x.shape[-1])
    finite = torch.isfinite(flat).all(dim=1)
    flat = flat[finite]
    remaining = _SAMPLE_LIMIT - sum(value.shape[0] for value in _SAMPLES)
    if remaining > 0 and flat.shape[0]:
        _SAMPLES.append(flat[:remaining].float().cpu().contiguous())


Linear.capture_H = _capture_h_with_samples


def _source_slice(
    tensor_name: str, item: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    model_root = Path(_PLAN["source"]["model_root"]).resolve(strict=True)
    index_path = model_root / "model.safetensors.index.json"
    if sha256_file(index_path) != _PLAN["source"]["index_sha256"]:
        raise CaptureError("model index hash differs from plan")
    index = _object(load_strict_json(index_path), "model index")
    weight_map = _object(index.get("weight_map"), "model index weight_map")
    shard_name = weight_map.get(tensor_name)
    if not isinstance(shard_name, str):
        raise CaptureError(f"target tensor is absent from model index: {tensor_name}")
    shard_path = (model_root / shard_name).resolve(strict=True)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        if tensor_name not in set(handle.keys()):
            raise CaptureError("target tensor is absent from mapped shard")
        view = handle.get_slice(tensor_name)
        shape = view.get_shape()
        output_start = item["output_start"]
        input_start = item["input_start"]
        size = item["size"]
        source = view[
            output_start : output_start + size,
            input_start : input_start + size,
        ]
    if source.dtype != torch.bfloat16:
        raise CaptureError(f"source tensor is not BF16: {source.dtype}")
    return source.T.float().contiguous(), {
        "shard": shard_name,
        "shard_sha256": sha256_file(shard_path),
        "source_shape": shape,
        "source_dtype": "BF16",
    }


def _atomic_save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_fixture(linear: Any, capture_h: dict[str, Any]) -> None:
    global _CAPTURED
    if _CAPTURED:
        raise CaptureError("target was encountered more than once")
    if not _SAMPLES:
        raise CaptureError("target captured no finite activation rows")
    weight = linear.inner.get_weight_tensor().detach().cpu()
    expected_shape = (linear.in_features, linear.out_features)
    if tuple(weight.shape) != expected_shape:
        raise CaptureError(
            f"converter weight orientation changed: {tuple(weight.shape)} != {expected_shape}"
        )
    h_data = capture_h.get(linear.qmap)
    if not isinstance(h_data, dict):
        raise CaptureError("target has no Hessian capture")
    hessian = h_data.get("H")
    raw_count = h_data.get("count")
    if (
        not isinstance(hessian, torch.Tensor)
        or isinstance(raw_count, bool)
        or not isinstance(raw_count, numbers.Integral)
        or raw_count <= 0
    ):
        raise CaptureError("target Hessian capture is incomplete")
    count = int(raw_count)
    hessian = hessian.detach().float().cpu()
    activations = torch.cat(_SAMPLES, dim=0)
    tensors: dict[str, torch.Tensor] = {}
    slice_rows: list[dict[str, Any]] = []
    shard_identity: dict[str, Any] | None = None
    tensor_name = _PLAN["target"]["tensor_name"]
    for item in _PLAN["slices"]:
        identifier = item["id"]
        input_start = item["input_start"]
        output_start = item["output_start"]
        size = item["size"]
        input_stop = input_start + size
        output_stop = output_start + size
        if input_stop > linear.in_features or output_stop > linear.out_features:
            raise CaptureError(f"slice {identifier} exceeds target shape")
        key = identifier.replace("-", "_")
        weight_slice = (
            weight[input_start:input_stop, output_start:output_stop]
            .float()
            .contiguous()
        )
        source_slice, source_identity = _source_slice(tensor_name, item)
        if not torch.equal(weight_slice, source_slice):
            max_diff = float((weight_slice - source_slice).abs().max().item())
            raise CaptureError(
                f"converter/source BF16 decode mismatch for {identifier}: {max_diff}"
            )
        if shard_identity is None:
            shard_identity = source_identity
        elif shard_identity != source_identity:
            raise CaptureError(
                "one target unexpectedly spans multiple shard identities"
            )
        tensors[f"weight.{key}"] = weight_slice
        tensors[f"hessian.{key}"] = hessian[
            input_start:input_stop, input_start:input_stop
        ].contiguous()
        tensors[f"activations.{key}"] = activations[
            :, input_start:input_stop
        ].contiguous()
        slice_rows.append(
            {
                **item,
                "weight_sha256": hashlib.sha256(
                    weight_slice.numpy().tobytes(order="C")
                ).hexdigest(),
                "hessian_sha256": hashlib.sha256(
                    tensors[f"hessian.{key}"].numpy().tobytes(order="C")
                ).hexdigest(),
                "activations_sha256": hashlib.sha256(
                    tensors[f"activations.{key}"].numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    tensor_path = _OUT_PATH.with_suffix(".safetensors")
    _atomic_save_tensors(tensor_path, tensors)
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass",
        "capture_id": _PLAN["capture_id"],
        "flow": _PLAN["flow"],
        "plan": {
            "path": str(_PLAN_PATH),
            "sha256": sha256_file(_PLAN_PATH),
            "canonical_sha256": canonical_sha256(_PLAN),
        },
        "target": {
            **_PLAN["target"],
            "in_features": linear.in_features,
            "out_features": linear.out_features,
            "qmap": linear.qmap,
            "converter_weight_dtype": str(weight.dtype),
        },
        "capture": {
            "hessian_count": count,
            "activation_sample_rows": activations.shape[0],
            "activation_width": activations.shape[1],
            "slices": slice_rows,
        },
        "source": {
            **_PLAN["source"],
            **(shard_identity or {}),
        },
        "converter": _PLAN["converter"],
        "tensors": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "keys": sorted(tensors),
            "bytes": tensor_path.stat().st_size,
        },
    }
    atomic_write_json(_OUT_PATH, result)
    _CAPTURED = True


def _wrap_quantizer(original: Any) -> Any:
    def wrapper(
        args: Any,
        linears: list[Any],
        config: Any,
        strategy: Any,
        idx: int,
        devices: Any,
        device_ratios: Any,
        capture_h: dict[str, Any] | None,
        state: Any,
    ) -> None:
        targets = [linear for linear in linears if linear.key == _TARGET_KEY]
        if targets:
            if len(targets) != 1 or capture_h is None:
                raise CaptureError("target linear/capture cardinality changed")
            _publish_fixture(targets[0], capture_h)
            raise CaptureComplete
        if _PLAN["flow"] == "quant":
            original(
                args,
                linears,
                config,
                strategy,
                idx,
                devices,
                device_ratios,
                capture_h,
                state,
            )

    return wrapper


cm.quantize_linears_single = _wrap_quantizer(cm.quantize_linears_single)
cm.quantize_linears_parallel = _wrap_quantizer(cm.quantize_linears_parallel)


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    args = cm.parser.parse_args()
    in_args, job_state, ok, error = cm.prepare(args)
    if not ok:
        raise CaptureError(f"converter prepare failed: {error}")
    try:
        cm.main(in_args, job_state)
    except CaptureComplete:
        pass
    if not _CAPTURED or not _OUT_PATH.exists():
        raise CaptureError("converter completed without the target capture")
    print(f"PASS: captured {_PLAN['flow']} fixture {_PLAN['capture_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
