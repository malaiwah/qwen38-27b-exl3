#!/usr/bin/env python3
"""Measure exllamav3 per-module proxy-error ladders.

With no frontier flags this is the historical five-rung ladder pass and retains
its historical output schema and propagation behavior.  Frontier sparse mode is
an explicit, ordered option path.  It neither constructs nor accepts a
Cartesian option bank.  Each measured option is cached independently under a
key that binds the calibration tokens, propagated state, converter identity,
module shape, and complete quantization identity.

Sparse-only flags are removed before exllamav3 parses its normal conversion
arguments:

  --frontier-plan PATH
  --frontier-cache-dir PATH
  --frontier-calibration-int64 PATH
  --frontier-ladder-out PATH
  [--frontier-packed-dir PATH]

The calibration input is raw little-endian signed int64 tokens.  Packed
quantizer outputs are durably written only when ``--frontier-packed-dir`` is
present.  Hessians, finalized factors, and saved source weights are never
serialized.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from frontier_common import (
    atomic_write_json,
    canonical_bytes,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)


SPARSE_PLAN_SCHEMA = "qwen38-frontier-sparse-ladder-plan/1"
SPARSE_OUTPUT_SCHEMA = "qwen38-frontier-sparse-ladder/1"
CACHE_SCHEMA = "qwen38-frontier-ladder-cache-entry/1"
OPTION_KEY_SCHEMA = "qwen38-frontier-ladder-option-key/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OWN_FLAGS = {
    "--frontier-plan": "plan",
    "--frontier-cache-dir": "cache_dir",
    "--frontier-calibration-int64": "calibration",
    "--frontier-ladder-out": "ladder_out",
    "--frontier-packed-dir": "packed_dir",
}


class SparseLadderError(ValueError):
    """A closed sparse-ladder validation failure."""


def _extract_frontier_flags(argv: list[str]) -> tuple[dict[str, Path] | None, list[str]]:
    values: dict[str, Path] = {}
    remaining = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        matched: tuple[str, str] | None = None
        for flag, name in OWN_FLAGS.items():
            if argument == flag:
                if index + 1 >= len(argv):
                    raise SparseLadderError(f"{flag} requires a path")
                matched = (name, argv[index + 1])
                index += 1
                break
            if argument.startswith(flag + "="):
                matched = (name, argument[len(flag) + 1 :])
                break
        if matched is None:
            remaining.append(argument)
        else:
            name, raw_path = matched
            if name in values:
                raise SparseLadderError(f"duplicate frontier flag for {name}")
            if not raw_path:
                raise SparseLadderError(f"frontier path for {name} must not be empty")
            values[name] = Path(raw_path)
        index += 1
    if not values:
        return None, remaining
    required = {"plan", "cache_dir", "calibration", "ladder_out"}
    missing = sorted(required - values.keys())
    if missing:
        raise SparseLadderError(f"sparse mode is missing explicit paths: {missing}")
    return values, remaining


_frontier_paths, sys.argv = _extract_frontier_flags(sys.argv)

sys.path.insert(0, "/work/exllamav3")

import torch
from exllamav3.conversion import convert_model as cm  # pyright: ignore[reportMissingImports]
from exllamav3.modules.quant.exl3_lib.quantize import (  # pyright: ignore[reportMissingImports]
    quantize_exl3_batch,
)


LADDER_OUT = os.environ.get("LADDER_OUT", "/work/kld6/ladder.json")
BIG_NUMEL = 52_000_000
CAND_BIG = (3, 4, 5, 6, 7)
CAND_SMALL = (4, 5, 6, 7, 8)
SKIP_SUBSTR = ("lm_head", "mtp.", "visual")

records: dict[str, dict[str, Any]] = {}
_t_start = time.time()
_sparse: dict[str, Any] | None = None

def _require_sparse() -> dict[str, Any]:
    if _sparse is None:
        raise SparseLadderError("sparse state is unavailable outside frontier mode")
    return _sparse


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SparseLadderError(f"{label} must be a JSON object with string keys")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SparseLadderError(f"{label} must be a JSON array")
    return value


def _known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SparseLadderError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    text = _known_string(value, label).lower()
    if not SHA256_RE.fullmatch(text):
        raise SparseLadderError(f"{label} must be a lowercase SHA256")
    return text


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SparseLadderError(f"{label} must be a positive integer")
    return value


def _json_identity(
    value: object, label: str, *, allow_null: bool = False
) -> object:
    try:
        canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise SparseLadderError(f"{label} must be finite canonical JSON") from exc
    if value in ("", {}, []) or (value is None and not allow_null):
        raise SparseLadderError(f"{label} must carry an explicit identity")
    return value


def _reject_cartesian(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if "cartesian" in normalized or normalized in {
                "candidate_widths",
                "option_bank",
                "qkv_widths",
                "widths",
                "k_values",
            }:
                raise SparseLadderError(f"{label} contains forbidden expansion field {key!r}")
            _reject_cartesian(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_cartesian(child, f"{label}[{index}]")


def _tensor_digest(tensor: torch.Tensor) -> str:
    detached = tensor.detach().contiguous().cpu()
    raw = detached.view(torch.uint8).numpy().tobytes()
    body = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }
    del detached, raw
    return canonical_sha256(body)


def _state_identity(value: object, label: str = "state") -> object:
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": _tensor_digest(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SparseLadderError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _state_identity(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise SparseLadderError(f"{label} has a non-string mapping key")
        return {
            key: _state_identity(value[key], f"{label}.{key}")
            for key in sorted(value)
        }
    raise SparseLadderError(
        f"{label} contains unsupported state object {type(value).__name__}"
    )


def _state_sha256(state: object) -> str:
    return canonical_sha256(_state_identity(state))


def _validate_sparse_plan(
    plan: dict[str, Any], calibration_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan.get("schema") != SPARSE_PLAN_SCHEMA:
        raise SparseLadderError(f"unsupported sparse plan schema {plan.get('schema')!r}")
    _reject_cartesian(plan, "sparse plan")
    calibration = _object(plan.get("calibration"), "plan calibration")
    if calibration.get("encoding") != "int64-le":
        raise SparseLadderError("plan calibration encoding must be int64-le")
    token_count = _positive_int(calibration.get("token_count"), "calibration token_count")
    expected_calibration_sha = _sha256(
        calibration.get("token_sha256"), "calibration token_sha256"
    )
    try:
        size = calibration_path.stat().st_size
    except OSError as exc:
        raise SparseLadderError(f"cannot stat calibration token input: {exc}") from exc
    if size != token_count * 8:
        raise SparseLadderError(
            f"calibration int64 byte count {size} does not equal token_count * 8"
        )
    actual_calibration_sha = sha256_file(calibration_path)
    if actual_calibration_sha != expected_calibration_sha:
        raise SparseLadderError("calibration int64 token digest disagrees with plan")

    converter = _object(plan.get("converter"), "plan converter")
    converter_identity = {
        "tree_sha256": _sha256(converter.get("tree_sha256"), "converter tree_sha256"),
        "diff_sha256": _sha256(converter.get("diff_sha256"), "converter diff_sha256"),
    }
    topology = _known_string(plan.get("topology"), "plan topology").lower().replace("-", "_")
    if topology != "split_qkv":
        raise SparseLadderError("sparse mode supports only split_qkv topology")
    initial_predecessor = _sha256(
        plan.get("initial_predecessor_sha256"), "initial predecessor"
    )

    normalized: list[dict[str, Any]] = []
    previous = initial_predecessor
    closed_modules: set[str] = set()
    current_module: str | None = None
    seen_module_widths: set[tuple[str, int]] = set()
    for index, raw in enumerate(_list(plan.get("options"), "plan options")):
        option = _object(raw, f"plan option {index}")
        allowed = {
            "module",
            "shape",
            "predecessor_sha256",
            "state_sha256",
            "rounding",
            "transform",
            "K",
            "codebook",
            "scale",
            "topology",
        }
        unknown = sorted(set(option) - allowed)
        missing = sorted(allowed - set(option))
        if unknown or missing:
            raise SparseLadderError(
                f"plan option {index} has missing={missing} unknown={unknown}"
            )
        module = _known_string(option["module"], f"plan option {index} module")
        if module != current_module:
            if current_module is not None:
                closed_modules.add(current_module)
            if module in closed_modules:
                raise SparseLadderError(
                    f"plan options for module {module!r} are not contiguous"
                )
            current_module = module
        shape_raw = _list(option["shape"], f"plan option {index} shape")
        if len(shape_raw) != 2:
            raise SparseLadderError(f"plan option {index} shape must have two dimensions")
        shape = [
            _positive_int(value, f"plan option {index} shape")
            for value in shape_raw
        ]
        predecessor = _sha256(
            option["predecessor_sha256"], f"plan option {index} predecessor"
        )
        if predecessor != previous:
            raise SparseLadderError(
                f"plan option {index} does not follow the preceding option key"
            )
        state_sha = _sha256(option["state_sha256"], f"plan option {index} state")
        k = _positive_int(option["K"], f"plan option {index} K")
        if k not in {3, 4, 5, 6, 7, 8}:
            raise SparseLadderError(f"plan option {index} K must be in 3..8")
        codebook = _known_string(option["codebook"], f"plan option {index} codebook")
        rounding = _json_identity(option["rounding"], f"plan option {index} rounding")
        transform = _json_identity(option["transform"], f"plan option {index} transform")
        scale = _json_identity(
            option["scale"], f"plan option {index} scale", allow_null=True
        )
        option_topology = (
            _known_string(option["topology"], f"plan option {index} topology")
            .lower()
            .replace("-", "_")
        )
        if option_topology != topology:
            raise SparseLadderError(
                f"plan option {index} changes the legal split_qkv topology"
            )
        module_width = (module, k)
        if module_width in seen_module_widths:
            raise SparseLadderError(
                f"plan repeats K{k} for module {module!r}; multi-axis/Cartesian expansion is forbidden"
            )
        seen_module_widths.add(module_width)
        key = {
            "schema": OPTION_KEY_SCHEMA,
            "calibration": {
                "encoding": "int64-le",
                "token_count": token_count,
                "token_sha256": actual_calibration_sha,
            },
            "predecessor_sha256": predecessor,
            "state_sha256": state_sha,
            "converter": converter_identity,
            "module": {"name": module, "shape": shape},
            "rounding": rounding,
            "transform": transform,
            "K": k,
            "codebook": codebook,
            "scale": scale,
            "topology": topology,
        }
        key_sha = canonical_sha256(key)
        normalized.append({"key": key, "key_sha256": key_sha})
        previous = key_sha
    if not normalized:
        raise SparseLadderError("sparse plan options must not be empty")
    source = {
        "plan_sha256": canonical_sha256(plan),
        "calibration_token_sha256": actual_calibration_sha,
        "converter": converter_identity,
        "topology": topology,
    }
    return normalized, source


def _validate_cache_directory(path: Path, suffix: str) -> None:
    if path.exists() and not path.is_dir():
        raise SparseLadderError(f"cache path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if not child.is_file() or child.suffix != suffix or not SHA256_RE.fullmatch(child.stem):
            raise SparseLadderError(f"unknown artifact in sparse cache: {child.name}")


def _load_sparse_output(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": SPARSE_OUTPUT_SCHEMA,
            "source": source,
            "options": {},
        }
    if path.stat().st_size == 0:
        raise SparseLadderError(f"sparse ladder output exists but is empty: {path}")
    output = _object(load_strict_json(path), "sparse ladder output")
    if set(output) != {"schema", "source", "options"}:
        raise SparseLadderError("existing sparse ladder output has unknown or missing fields")
    if output.get("schema") != SPARSE_OUTPUT_SCHEMA or output.get("source") != source:
        raise SparseLadderError("existing sparse ladder output belongs to another source path")
    _object(output.get("options"), "sparse ladder output options")
    return output


def _initialize_sparse(paths: dict[str, Path]) -> dict[str, Any]:
    plan = _object(load_strict_json(paths["plan"]), "sparse ladder plan")
    options, source = _validate_sparse_plan(plan, paths["calibration"])
    cache_dir = paths["cache_dir"]
    _validate_cache_directory(cache_dir, ".json")
    packed_dir = paths.get("packed_dir")
    if packed_dir is not None:
        _validate_cache_directory(packed_dir, ".pt")
    output = _load_sparse_output(paths["ladder_out"], source)
    by_key = {option["key_sha256"]: option for option in options}
    planned_keys = set(by_key)
    cache_keys = {path.stem for path in cache_dir.iterdir()}
    unknown_cache = sorted(cache_keys - planned_keys)
    unknown_output = sorted(set(output["options"]) - planned_keys)
    if unknown_cache or unknown_output:
        raise SparseLadderError(
            "sparse resume contains unplanned artifacts: "
            f"cache={unknown_cache} output={unknown_output}"
        )
    ordered_keys = [option["key_sha256"] for option in options]
    expected_cache_prefix = set(ordered_keys[: len(cache_keys)])
    if cache_keys != expected_cache_prefix:
        raise SparseLadderError("sparse cache is not a verified prefix of the option path")
    if set(output["options"]) - cache_keys:
        raise SparseLadderError("existing sparse output names a missing cache entry")
    output_keys = set(output["options"])
    expected_output_prefix = set(ordered_keys[: len(output_keys)])
    if output_keys != expected_output_prefix:
        raise SparseLadderError("sparse output is not a verified prefix of the option path")
    if packed_dir is not None:
        packed_keys = {path.stem for path in packed_dir.iterdir()}
        if packed_keys - planned_keys:
            raise SparseLadderError("packed directory contains an unplanned output")
        expected_packed_prefix = set(ordered_keys[: len(packed_keys)])
        if packed_keys != expected_packed_prefix or len(packed_keys) > len(cache_keys) + 1:
            raise SparseLadderError("packed outputs are not a recoverable prefix of the option path")
    for key_sha in sorted(cache_keys):
        entry = _validate_cache_entry(
            cache_dir / f"{key_sha}.json", by_key[key_sha], packed_dir
        )
        if key_sha in output["options"]:
            expected_output = {
                "module": by_key[key_sha]["key"]["module"]["name"],
                "K": by_key[key_sha]["key"]["K"],
                "proxy_error": entry["proxy_error"],
                "g_scale": entry["g_scale"],
                "q_fallback": entry["q_fallback"],
                "packed": entry["packed"],
            }
            if output["options"][key_sha] != expected_output:
                last_output_key = ordered_keys[len(output_keys) - 1]
                if key_sha != last_output_key:
                    raise SparseLadderError(
                        "sparse output disagrees with its verified cache entry"
                    )
    by_module: dict[str, list[dict[str, Any]]] = {}
    for option in options:
        module = option["key"]["module"]["name"]
        by_module.setdefault(module, []).append(option)
    return {
        "paths": paths,
        "options": options,
        "by_module": by_module,
        "source": source,
        "output": output,
        "encountered_modules": [],
        "next_module": 0,
    }




def candidates(numel: int) -> tuple[int, ...]:
    return CAND_BIG if numel >= BIG_NUMEL else CAND_SMALL


def dump() -> None:
    if _sparse is not None:
        atomic_write_json(_sparse["paths"]["ladder_out"], _sparse["output"])
        return
    tmp = LADDER_OUT + ".tmp"
    with open(tmp, "w") as f:
        import json

        json.dump(
            {
                "schema": "qwen38-proxy-error-ladder/1",
                "metric": "proxy_err = tr(E^T H E) / tr(W^T H W), exllamav3's own per-module "
                "Hessian-weighted relative quantization error",
                "out_energy": "tr(W^T H W) / count on the raw accumulated Hessian: the module's "
                "mean per-calibration-row output energy, so out_energy * proxy_err "
                "is the module's absolute output error energy",
                "candidate_widths": {
                    "big": list(CAND_BIG),
                    "small": list(CAND_SMALL),
                    "big_numel_threshold": BIG_NUMEL,
                },
                "propagation_recipe": "hydrated (attention K6, mlp gate/up K5, down K6, head K6/mcg, "
                "MTP as -mb 4 plus the fixed/override regexes, BF16 embed+vision)",
                "elapsed_sec": round(time.time() - _t_start, 1),
                "modules": records,
            },
            f,
            indent=1,
        )
    os.replace(tmp, LADDER_OUT)


def out_energy(linear, H_data) -> float | None:
    """Per-row calibration output energy, from raw H, before finalization."""
    if H_data is None or H_data.get("finalized") or H_data["H"].is_meta:
        return None
    count = int(H_data["count"])
    if count == 0:
        return None
    dev = torch.device(H_data["device"])
    w = linear.inner.get_weight_tensor()
    W = w.to(dev, torch.float32)
    H = H_data["H"].to(dev, torch.float32)
    energy = float((W * (H @ W)).sum().item()) / count
    del W, H
    return energy


_orig_print = cm.print_quantized_linear


def print_hook(config, linear, quant_args, proxy_err, time_str=""):
    _orig_print(config, linear, quant_args, proxy_err, time_str)
    rec = records.setdefault(linear.key, {})
    rec.setdefault("ladder", {})[str(int(quant_args["K"]))] = proxy_err
    rec["recipe_bits"] = int(quant_args["K"])
    rec["recipe_proxy_err"] = proxy_err
    rec["q_fallback"] = bool(quant_args.get("q_fallback"))
    rec.setdefault("numel", linear.weights_numel())
    rec.setdefault("in_features", linear.in_features)
    rec.setdefault("out_features", linear.out_features)


cm.print_quantized_linear = print_hook


def _quant_identity_matches(
    args: dict[str, Any], qa: dict[str, Any], option: dict[str, Any]
) -> None:
    key = option["key"]
    if qa.get("K") != key["K"]:
        raise SparseLadderError("converter K disagrees with sparse option key")
    actual_codebook = (
        "mcg"
        if qa.get("mcg") is True
        else "mul1"
        if qa.get("mul1") is True
        else args.get("codebook")
    )
    if actual_codebook != key["codebook"]:
        raise SparseLadderError("converter codebook disagrees with sparse option key")
    if "apply_out_scales" not in qa:
        raise SparseLadderError("converter quant args omit scale-mode identity")
    try:
        actual_scale = canonical_bytes(qa["apply_out_scales"])
    except (TypeError, ValueError) as exc:
        raise SparseLadderError("converter scale mode is not canonical JSON") from exc
    if actual_scale != canonical_bytes(key["scale"]):
        raise SparseLadderError("converter scale mode disagrees with sparse option key")


def _validate_cache_entry(
    path: Path, option: dict[str, Any], packed_dir: Path | None
) -> dict[str, Any]:
    entry = _object(load_strict_json(path), f"cache entry {path.name}")
    required = {
        "schema",
        "key",
        "key_sha256",
        "proxy_error",
        "g_scale",
        "q_fallback",
        "packed",
    }
    if set(entry) != required:
        raise SparseLadderError(f"cache entry {path.name} has unknown or missing fields")
    if (
        entry["schema"] != CACHE_SCHEMA
        or entry["key"] != option["key"]
        or entry["key_sha256"] != option["key_sha256"]
        or canonical_sha256(entry["key"]) != entry["key_sha256"]
    ):
        raise SparseLadderError(f"cache entry {path.name} key verification failed")
    cached_scale = entry["g_scale"]
    if (
        isinstance(cached_scale, bool)
        or not isinstance(cached_scale, (int, float))
        or not math.isfinite(cached_scale)
    ):
        raise SparseLadderError(f"cache entry {path.name} g_scale is not finite")
    proxy = entry["proxy_error"]
    if isinstance(proxy, bool) or not isinstance(proxy, (int, float)) or not math.isfinite(proxy):
        raise SparseLadderError(f"cache entry {path.name} proxy_error is not finite")
    if not isinstance(entry["q_fallback"], bool):
        raise SparseLadderError(f"cache entry {path.name} q_fallback is not boolean")
    packed = entry["packed"]
    if packed is not None:
        packed_obj = _object(packed, f"cache entry {path.name} packed")
        if set(packed_obj) != {"sha256", "bytes"}:
            raise SparseLadderError(f"cache entry {path.name} packed identity is malformed")
        packed_sha = _sha256(packed_obj["sha256"], f"cache entry {path.name} packed sha")
        packed_bytes = _positive_int(
            packed_obj["bytes"], f"cache entry {path.name} packed bytes"
        )
        if packed_dir is None:
            return entry
        packed_path = packed_dir / f"{option['key_sha256']}.pt"
        if not packed_path.is_file():
            raise SparseLadderError(f"cache entry {path.name} names missing packed output")
        if packed_path.stat().st_size != packed_bytes or sha256_file(packed_path) != packed_sha:
            raise SparseLadderError(f"packed output for {path.name} failed verification")
    return entry


def _atomic_torch_save(path: Path, value: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _record_sparse_result(
    option: dict[str, Any],
    proxy_error: float,
    qa: dict[str, Any],
    packed: dict[str, Any] | None,
) -> dict[str, Any]:
    sparse = _require_sparse()
    g_scale = qa.get("g_scale")
    if (
        isinstance(g_scale, bool)
        or not isinstance(g_scale, (int, float))
        or not math.isfinite(g_scale)
    ):
        raise SparseLadderError("quantizer did not return a finite g_scale")
    entry = {
        "schema": CACHE_SCHEMA,
        "key": option["key"],
        "key_sha256": option["key_sha256"],
        "proxy_error": proxy_error,
        "g_scale": g_scale,
        "q_fallback": bool(qa.get("q_fallback")),
        "packed": packed,
    }
    cache_path = sparse["paths"]["cache_dir"] / f"{option['key_sha256']}.json"
    atomic_write_json(cache_path, entry)
    return entry


def _sparse_module(
    args,
    linear,
    config,
    idx,
    devices,
    device_ratios,
    capture_H,
    state_sha,
    saved_weight,
) -> None:
    sparse = _require_sparse()
    del config
    options = sparse["by_module"][linear.key]
    for option in options:
        if option["key"]["state_sha256"] != state_sha:
            raise SparseLadderError(
                f"propagated state digest disagrees for {linear.key}"
            )
        if option["key"]["module"]["shape"] != [
            linear.out_features,
            linear.in_features,
        ]:
            raise SparseLadderError(f"module shape disagrees for {linear.key}")
        qa = cm.make_quant_args(
            args,
            idx,
            option["key"]["K"],
            *cm._tile_split_devices(
                linear.weights_numel(), devices, device_ratios
            ),
        )
        _quant_identity_matches(args, qa, option)
        cache_path = sparse["paths"]["cache_dir"] / f"{option['key_sha256']}.json"
        packed_dir = sparse["paths"].get("packed_dir")
        entry: dict[str, Any] | None = None
        if cache_path.exists():
            if cache_path.stat().st_size == 0:
                raise SparseLadderError(f"cache entry is empty: {cache_path.name}")
            entry = _validate_cache_entry(cache_path, option, packed_dir)
            if packed_dir is not None and entry["packed"] is None:
                entry = None
        if entry is None:
            result = quantize_exl3_batch(
                [saved_weight.float()],
                [capture_H[linear.qmap]],
                [qa],
                None,
            )
            if len(result) != 1:
                raise SparseLadderError("quantizer returned an unexpected batch size")
            proxy_error, out_tensors = result[0]
            proxy_error = float(proxy_error)
            if not math.isfinite(proxy_error):
                raise SparseLadderError("quantizer returned a non-finite proxy error")
            packed_identity = None
            if packed_dir is not None:
                packed_path = packed_dir / f"{option['key_sha256']}.pt"
                packed_identity = _atomic_torch_save(packed_path, out_tensors)
            entry = _record_sparse_result(
                option, proxy_error, qa, packed_identity
            )
            del out_tensors, result
        sparse["output"]["options"][option["key_sha256"]] = {
            "module": linear.key,
            "K": option["key"]["K"],
            "proxy_error": entry["proxy_error"],
            "g_scale": entry["g_scale"],
            "q_fallback": entry["q_fallback"],
            "packed": entry["packed"],
        }
        dump()


def _wrap_historical(orig):
    def fn(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state):
        hessians = capture_H if capture_H is not None else {}
        todo = [
            linear
            for linear in linears
            if strategy.get(linear.key, 16) != 16
            and linear.qmap in hessians
            and not any(skip in linear.key for skip in SKIP_SUBSTR)
        ]
        saved = {}
        for linear in todo:
            saved[linear.key] = linear.inner.get_weight_tensor().clone()
            rec = records.setdefault(linear.key, {})
            rec["numel"] = linear.weights_numel()
            rec["in_features"] = linear.in_features
            rec["out_features"] = linear.out_features
            rec["qmap"] = linear.qmap
            rec["cal_rows"] = int(hessians[linear.qmap]["count"])
            if rec.get("out_energy") is None:
                rec["out_energy"] = out_energy(linear, hessians[linear.qmap])

        orig(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state)

        all_k = sorted(
            {
                k
                for linear in todo
                for k in candidates(records[linear.key]["numel"])
            }
        )
        for K in all_k:
            targets = [
                linear
                for linear in todo
                if K in candidates(records[linear.key]["numel"])
                and K != strategy[linear.key]
            ]
            if not targets:
                continue
            ustrat = {linear.key: K for linear in targets}
            for group in cm.group_quant_linears(targets, ustrat, hessians):
                qal = [
                    cm.make_quant_args(
                        args,
                        idx,
                        K,
                        *cm._tile_split_devices(
                            linear.weights_numel(), devices, device_ratios
                        ),
                    )
                    for linear in group
                ]
                weights = [saved[linear.key].float() for linear in group]
                t0 = time.time()
                res = quantize_exl3_batch(
                    weights, [hessians[linear.qmap] for linear in group], qal, None
                )
                dt = (time.time() - t0) / len(group)
                for linear, qa, (proxy_err, out_tensors) in zip(group, qal, res):
                    rec = records[linear.key]
                    rec.setdefault("ladder", {})[str(K)] = proxy_err
                    rec.setdefault("ladder_sec", {})[str(K)] = round(dt, 2)
                    rec.setdefault("g_scale", {})[str(K)] = qa.get("g_scale")
                    if qa.get("q_fallback"):
                        rec["q_fallback"] = True
                    print(
                        f" ~~ Ladder: {linear.key:70}  K={K}  proxy_err: {proxy_err:10.8f}"
                        f"  g_sc: {qa.get('g_scale', float('nan')):.6f}  [{dt:5.2f} s]",
                        flush=True,
                    )
                    del out_tensors
                del res, weights
        saved.clear()
        torch.cuda.empty_cache()
        dump()

    return fn


def _wrap_sparse(orig):
    sparse = _require_sparse()
    def fn(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state):
        targeted = [
            linear for linear in linears if linear.key in sparse["by_module"]
        ]
        if targeted and capture_H is None:
            raise SparseLadderError("targeted sparse module has no captured Hessian")
        for linear in targeted:
            if linear.qmap not in capture_H:
                raise SparseLadderError(
                    f"targeted sparse module {linear.key} has no Hessian map"
                )
            expected_modules = list(sparse["by_module"])
            next_index = sparse["next_module"]
            if next_index >= len(expected_modules) or expected_modules[next_index] != linear.key:
                raise SparseLadderError(
                    f"converter encountered sparse module {linear.key!r} out of plan order"
                )
            sparse["next_module"] += 1
            sparse["encountered_modules"].append(linear.key)
        saved = {
            linear.key: linear.inner.get_weight_tensor().clone()
            for linear in targeted
        }
        state_sha = _state_sha256(state) if targeted else None
        if state_sha is not None:
            for linear in targeted:
                for option in sparse["by_module"][linear.key]:
                    if option["key"]["state_sha256"] != state_sha:
                        raise SparseLadderError(
                            f"propagated state digest disagrees for {linear.key}"
                        )

        orig(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state)

        for linear in targeted:
            _sparse_module(
                args,
                linear,
                config,
                idx,
                devices,
                device_ratios,
                capture_H,
                state_sha,
                saved[linear.key],
            )
        saved.clear()
        torch.cuda.empty_cache()

    return fn


if _frontier_paths is not None:
    _sparse = _initialize_sparse(_frontier_paths)


if _sparse is None:
    cm.quantize_linears_single = _wrap_historical(cm.quantize_linears_single)
    cm.quantize_linears_parallel = _wrap_historical(cm.quantize_linears_parallel)
else:
    cm.quantize_linears_single = _wrap_sparse(cm.quantize_linears_single)
    cm.quantize_linears_parallel = _wrap_sparse(cm.quantize_linears_parallel)


if __name__ == "__main__":
    _args = cm.parser.parse_args()
    _in_args, _job_state, _ok, _err = cm.prepare(_args)
    if not _ok:
        print(f" !! Error: {_err}")
        raise SystemExit(1)
    try:
        cm.main(_in_args, _job_state)
        if _sparse is not None and _sparse["next_module"] != len(_sparse["by_module"]):
            missing = list(_sparse["by_module"])[_sparse["next_module"] :]
            raise SparseLadderError(
                f"converter did not encounter planned sparse modules: {missing}"
            )
    finally:
        dump()
        output_path = (
            str(_sparse["paths"]["ladder_out"]) if _sparse is not None else LADDER_OUT
        )
        count = (
            len(_sparse["output"]["options"]) if _sparse is not None else len(records)
        )
        noun = "options" if _sparse is not None else "modules"
        print(f" ~~ Ladder written: {output_path} ({count} {noun})", flush=True)
