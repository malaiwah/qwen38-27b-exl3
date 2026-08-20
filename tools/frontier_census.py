#!/usr/bin/env python3
"""Build the immutable BF16 tensor census and measured sparse route registry."""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_common import (
    atomic_write_json,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)

CENSUS_SCHEMA = "qwen38-frontier-bf16-census/1"
REGISTRY_SCHEMA = "qwen38-frontier-compatibility-registry/1"
EXPECTED_EXL3_MODULES = 409
EXPECTED_SHARDS = 18
EXPECTED_LOGICAL_TENSORS = 1_199
EXPECTED_LOGICAL_PARAMETERS = 27_781_427_952
EXPECTED_BF16_PAYLOAD_BYTES = 55_562_855_904
UNKNOWN_STRINGS = {"", "unknown", "unmeasured", "n/a", "na", "none", "null", "tbd"}
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
SHA_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
FULL_ATTN_RE = re.compile(r"^(.*\.self_attn)\.(q_proj|k_proj|v_proj)\.weight$")


class CensusError(ValueError):
    """A closed validation failure with an operator-readable message."""


def fail(message: str) -> NoReturn:
    raise CensusError(message)


def as_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def as_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a JSON array")
    return cast(list[Any], value)


def known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() in UNKNOWN_STRINGS:
        fail(f"{label} must be a known, non-empty string")
    return value.strip()


def sha(value: object, label: str) -> str:
    text = known_string(value, label).lower()
    if not SHA_RE.fullmatch(text):
        fail(f"{label} must be a 40- or 64-character hexadecimal SHA")
    return text


def nonnegative_number(value: object, label: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "finite and nonnegative"
        fail(f"{label} must be {qualifier}")
    return value


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < 0 or (positive and value <= 0):
        fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def integral_number(value: object, label: str, *, positive: bool = False) -> int:
    numeric = nonnegative_number(value, label, positive=positive)
    if int(numeric) != numeric:
        fail(f"{label} must be integer-valued")
    return int(numeric)




def product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def extract_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = receipt
    for key in ("bf16_identity", "reference_identity", "identity"):
        if isinstance(receipt.get(key), dict):
            identity = receipt[key]
            break

    revision_value = identity.get("revision", identity.get("model_revision"))
    if revision_value is None and isinstance(identity.get("source"), dict):
        revision_value = identity["source"].get("revision")
    revision = sha(revision_value, "identity revision")

    repo_value = identity.get("repo", identity.get("model_repo"))
    if repo_value is None and isinstance(identity.get("source"), dict):
        repo_value = identity["source"].get("repo")
    repo = known_string(repo_value, "identity repo")

    index_value = identity.get("index_sha256")
    if index_value is None and isinstance(identity.get("index"), dict):
        index_value = identity["index"].get("sha256")
    config_value = identity.get("config_sha256")
    if config_value is None and isinstance(identity.get("config"), dict):
        config_value = identity["config"].get("sha256")

    shard_value = identity.get("shard_sha256", identity.get("shards"))
    shards_obj = as_object(shard_value, "identity shard hashes")
    shards: dict[str, str] = {}
    for name, digest_value in shards_obj.items():
        if isinstance(digest_value, dict):
            digest_value = digest_value.get("sha256")
        shards[name] = sha(digest_value, f"identity shard {name} sha256")

    return {
        "repo": repo,
        "revision": revision,
        "index_sha256": sha(index_value, "identity index sha256"),
        "config_sha256": sha(config_value, "identity config sha256"),
        "shard_sha256": shards,
    }


def validate_shard_names(shards: dict[str, str]) -> list[str]:
    if len(shards) != EXPECTED_SHARDS:
        fail(f"identity must name exactly {EXPECTED_SHARDS} shards, found {len(shards)}")
    ordered: list[tuple[int, str]] = []
    for name in shards:
        match = SHARD_RE.fullmatch(name)
        if match is None:
            fail(f"identity contains invalid 18-shard filename: {name!r}")
        if int(match.group(2)) != EXPECTED_SHARDS:
            fail(f"identity contains invalid 18-shard filename: {name!r}")
        ordered.append((int(match.group(1)), name))
    ordered.sort()
    if [number for number, _ in ordered] != list(range(1, EXPECTED_SHARDS + 1)):
        fail("identity shard sequence is incomplete or duplicated")
    return [name for _, name in ordered]


def strict_json_bytes(data: bytes, label: str) -> object:
    def reject_constant(token: str) -> None:
        fail(f"{label} contains forbidden JSON constant {token}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except UnicodeDecodeError as exc:
        fail(f"{label} is not UTF-8: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"{label} is invalid JSON: {exc}")
    raise AssertionError("unreachable")


def read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    file_size: int | None = None
    header_length: int | None = None
    header_bytes: bytes | None = None
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                fail(f"{path.name} is too short for a safetensors header")
            header_length = struct.unpack("<Q", raw_length)[0]
            if file_size is None or header_length is None:
                fail(f"{path.name} header state was not initialized")
            if header_length < 2 or header_length > file_size - 8:
                fail(f"{path.name} has impossible header length {header_length}")
            if header_length > 128 * 1024 * 1024:
                fail(f"{path.name} header exceeds the 128 MiB safety bound")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                fail(f"{path.name} has a truncated safetensors header")
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    if file_size is None or header_length is None or header_bytes is None:
        raise AssertionError("unreachable safetensors header state")
    header = as_object(strict_json_bytes(header_bytes, f"{path.name} header"), f"{path.name} header")
    return header, header_length, file_size


def parse_headers(
    model: Path,
    ordered_shards: list[str],
    weight_map: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    tensors: dict[str, dict[str, Any]] = {}
    shard_summaries: dict[str, dict[str, int]] = {}
    indexed_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            fail("index weight_map must map string tensor names to string shard names")
        if shard_name not in ordered_shards:
            fail(f"index routes {tensor_name!r} to unexpected shard {shard_name!r}")
        indexed_by_shard[shard_name].add(tensor_name)

    for shard_name in ordered_shards:
        header, header_length, file_size = read_header(model / shard_name)
        header.pop("__metadata__", None)
        names = set(header)
        if names != indexed_by_shard[shard_name]:
            missing = sorted(indexed_by_shard[shard_name] - names)
            extra = sorted(names - indexed_by_shard[shard_name])
            fail(f"{shard_name} header/index mismatch; missing={missing[:3]}, extra={extra[:3]}")

        intervals: list[tuple[int, int, str]] = []
        payload_bytes = 0
        for name, raw_entry in header.items():
            entry = as_object(raw_entry, f"header entry {name}")
            dtype = known_string(entry.get("dtype"), f"{name}.dtype").upper()
            if dtype not in DTYPE_BYTES:
                fail(f"{name} uses unsupported safetensors dtype {dtype!r}")
            raw_shape = as_list(entry.get("shape"), f"{name}.shape")
            shape = [integer(dimension, f"{name}.shape[{index}]") for index, dimension in enumerate(raw_shape)]
            offsets = as_list(entry.get("data_offsets"), f"{name}.data_offsets")
            if len(offsets) != 2:
                fail(f"{name}.data_offsets must contain exactly two integers")
            start = integer(offsets[0], f"{name}.data_offsets[0]")
            end = integer(offsets[1], f"{name}.data_offsets[1]")
            if end < start:
                fail(f"{name} has reversed data offsets")
            expected_bytes = product(shape) * DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                fail(f"{name} shape/dtype requires {expected_bytes} bytes but offsets span {end - start}")
            if name in tensors:
                fail(f"tensor {name!r} occurs in more than one shard header")
            tensors[name] = {
                "dtype": dtype,
                "shape": shape,
                "numel": product(shape),
                "bytes": expected_bytes,
                "shard": shard_name,
                "data_offsets": [start, end],
            }
            intervals.append((start, end, name))
            payload_bytes += expected_bytes

        cursor = 0
        for start, end, name in sorted(intervals):
            if start != cursor:
                fail(f"{shard_name} payload is not contiguous before {name!r}: expected {cursor}, got {start}")
            cursor = end
        # Header JSON may contain whitespace; the declared length locates the payload.
        actual_payload = file_size - 8 - header_length
        if cursor != actual_payload:
            fail(f"{shard_name} payload size mismatch: offsets end at {cursor}, file has {actual_payload}")
        shard_summaries[shard_name] = {
            "bytes": file_size,
            "payload_bytes": payload_bytes,
            "tensor_count": len(header),
        }
    if set(tensors) != set(weight_map):
        fail("combined shard headers do not equal the index weight_map")
    return tensors, shard_summaries


def source_revision(document: dict[str, Any], label: str) -> str:
    value: object = document.get("source_revision")
    if value is None and isinstance(document.get("source"), dict):
        value = document["source"].get("revision")
    return sha(value, f"{label} source revision")


def policy_for(name: str, module_names: set[str]) -> str:
    module = name[:-7] if name.endswith(".weight") else None
    if module in module_names:
        if module == "lm_head":
            return "head"
        if module.startswith("mtp."):
            return "mtp"
        return "language_body"
    lowered = name.lower()
    if "embed_tokens" in lowered:
        return "embedding"
    if lowered.startswith("model.visual.") or ".visual." in lowered or lowered.startswith("visual."):
        return "vision"
    return "norm_small"

def tensor_role(name: str) -> str:
    if ".visual." in name or name.startswith("visual."):
        return "vision_tower"
    if name.startswith("mtp"):
        return "mtp_draft"
    if name == "lm_head" or name.startswith("lm_head."):
        return "lm_head"
    if "embed_tokens" in name:
        return "embed_tokens"
    if "linear_attn" in name:
        return "linear_attention"
    if "self_attn" in name:
        return "full_attention"
    if ".mlp." in name:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if name.endswith(projection) or name.endswith(f"{projection}.weight"):
                return f"mlp_{projection}"
        return "mlp_other"
    return "norms_and_small"


def tensor_layer(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def expected_module_inventory() -> dict[str, tuple[int, int]]:
    expected: dict[str, tuple[int, int]] = {}
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}"
        if layer % 4 == 3:
            attention = (
                ("q_proj", 5120, 12288),
                ("k_proj", 5120, 1024),
                ("v_proj", 5120, 1024),
                ("o_proj", 6144, 5120),
            )
            for name, in_features, out_features in attention:
                expected[f"{prefix}.self_attn.{name}"] = (in_features, out_features)
        else:
            attention = (
                ("in_proj_qkv", 5120, 10240),
                ("in_proj_z", 5120, 6144),
                ("out_proj", 6144, 5120),
            )
            for name, in_features, out_features in attention:
                expected[f"{prefix}.linear_attn.{name}"] = (in_features, out_features)
        for name, in_features, out_features in (
            ("gate_proj", 5120, 17408),
            ("up_proj", 5120, 17408),
            ("down_proj", 17408, 5120),
        ):
            expected[f"{prefix}.mlp.{name}"] = (in_features, out_features)
    for name, in_features, out_features in (
        ("self_attn.q_proj", 5120, 12288),
        ("self_attn.k_proj", 5120, 1024),
        ("self_attn.v_proj", 5120, 1024),
        ("self_attn.o_proj", 6144, 5120),
        ("mlp.gate_proj", 5120, 17408),
        ("mlp.up_proj", 5120, 17408),
        ("mlp.down_proj", 17408, 5120),
    ):
        expected[f"mtp.layers.0.{name}"] = (in_features, out_features)
    expected["mtp.fc"] = (10240, 5120)
    expected["lm_head"] = (5120, 248320)
    return expected


def normalize_disposition(raw: object, rule_id: str) -> dict[str, Any]:
    if isinstance(raw, str):
        disposition: dict[str, Any] = {"format": raw}
    else:
        disposition = dict(as_object(raw, f"recipe rule {rule_id} disposition"))
    fmt = known_string(disposition.get("format", disposition.get("kind")), f"recipe rule {rule_id} format").lower()
    if fmt in {"bfloat16", "fixed_bf16"}:
        fmt = "bf16"
    if fmt not in {"bf16", "exl3"}:
        fail(f"recipe rule {rule_id} has unsupported format {fmt!r}")
    normalized: dict[str, Any] = {"format": fmt}
    normalized["policy"] = known_string(disposition.get("policy"), f"recipe rule {rule_id} policy")
    if fmt == "exl3":
        k_value = disposition.get("K", disposition.get("k", disposition.get("bits")))
        k = integer(k_value, f"recipe rule {rule_id} K", positive=True)
        if k not in {3, 4, 5, 6, 7}:
            fail(f"recipe rule {rule_id} K must be in 3..7")
        topology = known_string(disposition.get("topology"), f"recipe rule {rule_id} topology").lower().replace("-", "_")
        if topology != "split_qkv":
            fail(f"recipe rule {rule_id} requests unsupported topology {topology!r}; only split_qkv is eligible")
        normalized.update(
            {
                "K": k,
                "codebook": known_string(disposition.get("codebook"), f"recipe rule {rule_id} codebook"),
                "scale_mode": known_string(disposition.get("scale_mode"), f"recipe rule {rule_id} scale_mode"),
                "topology": topology,
            }
        )
    else:
        forbidden = {"K", "k", "bits", "codebook", "scale_mode", "topology"} & disposition.keys()
        if forbidden:
            fail(f"BF16 recipe rule {rule_id} carries quantization fields: {sorted(forbidden)}")
    return normalized

def reject_cartesian_or_mixed_qkv(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower().replace("-", "_")
            if "cartesian" in normalized_key or normalized_key in {"option_bank", "qkv_widths"}:
                fail(f"{label} contains forbidden field {key!r}")
            if normalized_key in {"bank_kind", "option_layout"} and isinstance(child, str) and "cartesian" in child.lower():
                fail(f"{label} declares a Cartesian option bank")
            if normalized_key == "topology" and isinstance(child, str):
                topology = child.lower().replace("-", "_")
                if topology != "split_qkv":
                    fail(f"{label} declares unsupported topology {child!r}")
            reject_cartesian_or_mixed_qkv(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_cartesian_or_mixed_qkv(child, f"{label}[{index}]")


def resolve_recipe(
    recipe: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
    module_names: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    schema = known_string(recipe.get("schema"), "recipe schema")
    if schema not in {"qwen38-frontier-recipe/1", "qwen38-frontier-recipe-spec/1"}:
        fail(f"unsupported recipe schema {schema!r}")
    reject_cartesian_or_mixed_qkv(recipe, "recipe")
    topology = known_string(recipe.get("topology"), "recipe topology").lower().replace("-", "_")
    if topology != "split_qkv":
        fail("recipe topology must be split_qkv; fused and mixed-width QKV are unsupported")
    resolution = known_string(recipe.get("resolution"), "recipe resolution").lower().replace("-", "_")
    if resolution not in {"first_match", "unique"}:
        fail("recipe resolution must be explicitly first_match or unique")

    compiled: list[tuple[str, re.Pattern[str], str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for index, raw_rule in enumerate(as_list(recipe.get("rules"), "recipe rules")):
        rule = as_object(raw_rule, f"recipe rule {index}")
        rule_id = known_string(rule.get("id"), f"recipe rule {index} id")
        if rule_id in seen_ids:
            fail(f"duplicate recipe rule id {rule_id!r}")
        seen_ids.add(rule_id)
        target = known_string(rule.get("target", "tensor"), f"recipe rule {rule_id} target").lower()
        if target not in {"tensor", "module"}:
            fail(f"recipe rule {rule_id} target must be tensor or module")
        pattern_text = known_string(rule.get("pattern"), f"recipe rule {rule_id} pattern")
        pattern: re.Pattern[str] | None = None
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            fail(f"recipe rule {rule_id} has invalid regex: {exc}")
        if pattern is None:
            raise AssertionError("unreachable compiled regex state")
        disposition_source = rule.get("disposition")
        if disposition_source is None:
            disposition_source = {
                key: rule[key]
                for key in ("format", "kind", "policy", "K", "k", "bits", "codebook", "scale_mode", "topology")
                if key in rule
            }
        compiled.append((rule_id, pattern, target, normalize_disposition(disposition_source, rule_id)))
    if not compiled:
        fail("recipe rules must not be empty")

    matches_per_rule: Counter[str] = Counter()
    resolved: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        module = name[:-7] if name.endswith(".weight") and name[:-7] in module_names else None
        matches: list[tuple[str, dict[str, Any]]] = []
        for rule_id, pattern, target, disposition in compiled:
            candidate = module if target == "module" else name
            if candidate is not None and pattern.fullmatch(candidate):
                matches.append((rule_id, disposition))
        if not matches:
            fail(f"recipe leaves tensor {name!r} without a disposition")
        if resolution == "unique" and len(matches) != 1:
            fail(f"recipe resolves tensor {name!r} through {len(matches)} rules under unique resolution")
        rule_id, disposition = matches[0]
        matches_per_rule[rule_id] += 1
        actual_policy = policy_for(name, module_names)
        if disposition["policy"] != actual_policy:
            fail(f"recipe rule {rule_id} assigns {name!r} policy {disposition['policy']!r}, expected {actual_policy!r}")
        expected_format = "exl3" if module in module_names else "bf16"
        if disposition["format"] != expected_format:
            fail(f"recipe assigns {name!r} {disposition['format']}, expected {expected_format}")
        resolved[name] = {
            "rule": rule_id,
            "matching_rules": [matched_rule for matched_rule, _ in matches],
            **disposition,
        }

    unused = [rule_id for rule_id, _, _, _ in compiled if matches_per_rule[rule_id] == 0]
    if unused:
        fail(f"recipe contains rules that resolve no tensors: {unused}")
    if len(resolved) != len(tensors):
        fail("recipe did not produce exactly one disposition per logical tensor")
    return resolved, dict(sorted(matches_per_rule.items()))


def validate_ladder(
    ladder: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
    dispositions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if ladder.get("schema") != "qwen38-proxy-error-ladder/1":
        fail(f"unsupported measured ladder schema {ladder.get('schema')!r}")
    modules = as_object(ladder.get("modules"), "ladder modules")
    if len(modules) != EXPECTED_EXL3_MODULES:
        fail(f"ladder must contain exactly {EXPECTED_EXL3_MODULES} EXL3 modules, found {len(modules)}")
    expected = expected_module_inventory()
    if set(modules) != set(expected):
        missing = sorted(set(expected) - set(modules))
        extra = sorted(set(modules) - set(expected))
        fail(f"ladder is not the exact Qwen3.8 409-module set; missing={missing[:3]}, extra={extra[:3]}")
    roles = Counter(tensor_role(module) for module in modules)
    expected_roles = {
        "full_attention": 64,
        "linear_attention": 144,
        "lm_head": 1,
        "mlp_down_proj": 64,
        "mlp_gate_proj": 64,
        "mlp_up_proj": 64,
        "mtp_draft": 8,
    }
    if dict(roles) != expected_roles:
        fail(f"ladder module role counts are {dict(roles)}, expected {expected_roles}")
    validated: dict[str, dict[str, Any]] = {}
    for module, raw_record in modules.items():
        record = as_object(raw_record, f"ladder module {module}")
        tensor_name = f"{module}.weight"
        if tensor_name not in tensors:
            fail(f"ladder module {module!r} has no BF16 weight tensor")
        tensor = tensors[tensor_name]
        in_features = integer(record.get("in_features"), f"{module}.in_features", positive=True)
        out_features = integer(record.get("out_features"), f"{module}.out_features", positive=True)
        numel = integer(record.get("numel"), f"{module}.numel", positive=True)
        if (in_features, out_features) != expected[module]:
            fail(
                f"ladder geometry for {module!r} is {(in_features, out_features)}, "
                f"expected {expected[module]}"
            )
        if tensor["shape"] != [out_features, in_features] or tensor["numel"] != numel:
            fail(f"ladder geometry for {module!r} disagrees with BF16 header {tensor['shape']}")
        recipe_k = integer(record.get("recipe_bits"), f"{module}.recipe_bits", positive=True)
        disposition = dispositions[tensor_name]
        if disposition["format"] != "exl3" or disposition["K"] != recipe_k:
            fail(f"recipe K for {module!r} disagrees with measured ladder incumbent K{recipe_k}")
        raw_rungs = as_object(record.get("ladder"), f"{module}.ladder")
        rungs: dict[str, float] = {}
        for raw_k, raw_metric in raw_rungs.items():
            k: int | None = None
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                fail(f"{module}.ladder has non-integer K key {raw_k!r}")
            if k is None:
                raise AssertionError("unreachable ladder K state")
            if str(k) != raw_k or k not in {3, 4, 5, 6, 7, 8}:
                fail(f"{module}.ladder has unsupported K key {raw_k!r}")
            metric = nonnegative_number(raw_metric, f"{module}.ladder[{raw_k}]")
            rungs[raw_k] = float(metric)
        if str(recipe_k) not in rungs:
            fail(f"ladder for {module!r} omits incumbent K{recipe_k}")
        validated[module] = {
            "shape": tensor["shape"],
            "numel": numel,
            "in_features": in_features,
            "out_features": out_features,
            "recipe_K": recipe_k,
            "rungs": rungs,
            "qmap": record.get("qmap"),
            "role": tensor_role(module),
            "layer": tensor_layer(module),
        }
    return validated


def validate_qkv(tensors: dict[str, dict[str, Any]], dispositions: dict[str, dict[str, Any]]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for name in tensors:
        match = FULL_ATTN_RE.fullmatch(name)
        if match:
            groups[match.group(1)].add(match.group(2))
    for prefix, projections in groups.items():
        if projections != {"q_proj", "k_proj", "v_proj"}:
            fail(f"full-attention group {prefix!r} is not split into complete q/k/v tensors")
        for projection in projections:
            disposition = dispositions[f"{prefix}.{projection}.weight"]
            if disposition.get("topology") != "split_qkv":
                fail(f"{prefix}.{projection} is not assigned split_qkv topology")
    if not groups:
        fail("BF16 snapshot contains no split full-attention q/k/v groups")


def edge_justifications(recipe: dict[str, Any], modules: dict[str, dict[str, Any]]) -> dict[tuple[str, int], str]:
    raw = recipe.get("edge_justifications", {})
    obj = as_object(raw, "recipe edge_justifications")
    result: dict[tuple[str, int], str] = {}
    for module, raw_edges in obj.items():
        if module not in modules:
            fail(f"edge justification names unknown module {module!r}")
        if modules[module]["role"] in {"lm_head", "mtp_draft"}:
            fail(f"head/MTP edge K policy must remain separate; remove justification for {module!r}")
        edges = as_object(raw_edges, f"edge justifications for {module}")
        for raw_k, reason in edges.items():
            if raw_k not in {"3", "7"}:
                fail(f"edge justification for {module!r} may name only K3 or K7")
            k = int(raw_k)
            if raw_k not in modules[module]["rungs"]:
                fail(f"edge justification for {module!r} K{k} has no measured ladder rung")
            result[(module, k)] = known_string(reason, f"edge justification for {module} K{k}")
    return result


def selected_ks(
    module: str,
    record: dict[str, Any],
    policy: str,
    justifications: dict[tuple[str, int], str],
) -> list[int]:
    incumbent = record["recipe_K"]
    if policy in {"head", "mtp"}:
        return [incumbent]
    selected = {incumbent}
    for candidate in (incumbent - 1, incumbent + 1):
        if candidate in {4, 5, 6}:
            if str(candidate) not in record["rungs"]:
                fail(f"measured ladder for {module!r} omits required adjacent K{candidate}")
            selected.add(candidate)
    for edge in (3, 7):
        if (module, edge) in justifications:
            selected.add(edge)
    return sorted(selected)


def normalize_alignment(value: object, label: str) -> dict[str, int]:
    obj = as_object(value, label)
    if not obj:
        fail(f"{label} must not be empty")
    result: dict[str, int] = {}
    for key, raw in obj.items():
        result[key] = integer(raw, f"{label}.{key}", positive=True)
    return dict(sorted(result.items()))


def normalize_markers(value: object, label: str, codebook: str) -> dict[str, str | int | bool]:
    obj = as_object(value, label)
    if not obj:
        fail(f"{label} must not be empty")
    result: dict[str, str | int | bool] = {}
    for key, raw in obj.items():
        if not isinstance(raw, (str, int, bool)) or isinstance(raw, float):
            fail(f"{label}.{key} must be a string, integer, or boolean")
        if isinstance(raw, str):
            known_string(raw, f"{label}.{key}")
        result[key] = raw
    marker_codebook = result.get("codebook")
    if marker_codebook != codebook:
        fail(f"{label}.codebook must equal recipe codebook {codebook!r}")
    return dict(sorted(result.items()))


def normalize_fallback(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        if value.lower() != "none":
            fail(f"{label} string form must be 'none'")
        return {"observed": False, "route": None, "measured": True}
    obj = as_object(value, label)
    observed = obj.get("observed")
    measured = obj.get("measured")
    if not isinstance(observed, bool) or not isinstance(measured, bool):
        fail(f"{label} must declare boolean observed and measured")
    route = obj.get("route")
    if observed:
        route = known_string(route, f"{label}.route")
        if not measured:
            fail(f"{label} observed fallback is not measured")
    elif route is not None:
        fail(f"{label}.route must be null when no fallback was observed")
    if not measured:
        fail(f"{label} cannot use a modeled/unmeasured fallback statement")
    return {"observed": observed, "route": route, "measured": measured}


def route_rows(routes: dict[str, Any]) -> list[dict[str, Any]]:
    if routes.get("schema") not in {"qwen38-frontier-runtime-routes/1", "qwen38-frontier-route-manifest/1"}:
        fail(f"unsupported runtime-route schema {routes.get('schema')!r}")
    reject_cartesian_or_mixed_qkv(routes, "runtime-route manifest")
    raw_rows = routes.get("routes", routes.get("measurements"))
    return [as_object(row, f"runtime route {index}") for index, row in enumerate(as_list(raw_rows, "runtime routes"))]


def build_registry(
    routes: dict[str, Any],
    modules: dict[str, dict[str, Any]],
    dispositions: dict[str, dict[str, Any]],
    recipe: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    global_runtime = routes.get("runtime_sha")
    global_sm = routes.get("sm")
    justifications = edge_justifications(recipe, modules)
    normalized_rows: list[dict[str, Any]] = []
    registry_entries: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(route_rows(routes)):
        label = f"runtime route {index}"
        module_value = row.get("module")
        modules_value = row.get("modules")
        if module_value is not None and modules_value is not None:
            fail(f"{label} cannot declare both module and modules")
        if module_value is not None:
            applies = [known_string(module_value, f"{label}.module")]
        elif modules_value is not None:
            applies = [known_string(item, f"{label}.modules") for item in as_list(modules_value, f"{label}.modules")]
            if not applies:
                fail(f"{label}.modules must not be empty")
        else:
            applies = []
        for module in applies:
            if module not in modules:
                fail(f"{label} names unknown module {module!r}")

        runtime_sha = sha(row.get("runtime_sha", global_runtime), f"{label}.runtime_sha")
        sm_raw = row.get("sm", global_sm)
        if isinstance(sm_raw, str):
            sm_text = known_string(sm_raw, f"{label}.sm").lower().removeprefix("sm_").removeprefix("sm")
            if not sm_text.isdigit():
                fail(f"{label}.sm must identify a numeric SM")
            sm = int(sm_text)
        else:
            sm = integer(sm_raw, f"{label}.sm", positive=True)
        k = integer(row.get("K", row.get("k")), f"{label}.K", positive=True)
        if k not in {3, 4, 5, 6, 7}:
            fail(f"{label}.K must be in 3..7")
        codebook = known_string(row.get("codebook"), f"{label}.codebook")
        markers = normalize_markers(row.get("codebook_markers"), f"{label}.codebook_markers", codebook)
        scale_mode = known_string(row.get("scale_mode"), f"{label}.scale_mode")
        topology = known_string(row.get("topology"), f"{label}.topology").lower().replace("-", "_")
        if topology != "split_qkv":
            fail(f"{label} uses unsupported topology {topology!r}")
        shape = [integer(value, f"{label}.shape", positive=True) for value in as_list(row.get("shape"), f"{label}.shape")]
        if len(shape) != 2:
            fail(f"{label}.shape must have two dimensions")
        alignment = normalize_alignment(row.get("alignment"), f"{label}.alignment")
        n_value = integer(row.get("N", row.get("n")), f"{label}.N", positive=True)
        row_class = known_string(row.get("row_class"), f"{label}.row_class").lower()
        if row_class not in {"decode", "prefill"}:
            fail(f"{label}.row_class must be decode or prefill")
        graph_mode = known_string(row.get("graph_mode"), f"{label}.graph_mode")
        route = known_string(row.get("route"), f"{label}.route")
        kind = row.get("measurement_kind", row.get("status"))
        measured_flag = row.get("measured")
        if kind != "measured" and measured_flag is not True:
            fail(f"{label} is not an observed measurement")
        if isinstance(kind, str) and kind.lower() == "modeled":
            fail(f"{label} is modeled, not measured")

        resources_source = row.get("resources", row)
        resources_obj = as_object(resources_source, f"{label}.resources")
        resources = {
            "latency_us": nonnegative_number(resources_obj.get("latency_us"), f"{label}.latency_us", positive=True),
            "scratch_bytes": integer(resources_obj.get("scratch_bytes"), f"{label}.scratch_bytes"),
            "jit_ms": nonnegative_number(resources_obj.get("jit_ms"), f"{label}.jit_ms"),
            "startup_ms": nonnegative_number(resources_obj.get("startup_ms"), f"{label}.startup_ms"),
        }
        fallback = normalize_fallback(row.get("fallback"), f"{label}.fallback")
        key = {
            "runtime_sha": runtime_sha,
            "sm": sm,
            "K": k,
            "codebook_markers": markers,
            "shape": shape,
            "alignment": alignment,
            "N": n_value,
            "row_class": row_class,
            "graph_mode": graph_mode,
        }
        entry_id = canonical_sha256(key)
        observation = {
            "route": route,
            "scale_mode": scale_mode,
            "topology": topology,
            "resources": resources,
            "fallback": fallback,
            "measurement_kind": "measured",
        }
        existing = registry_entries.get(entry_id)
        entry = {"key": key, "observation": observation}
        if existing is not None and existing != entry:
            fail(f"conflicting runtime observations for compatibility key {entry_id}")
        registry_entries[entry_id] = entry
        normalized_rows.append({"entry_id": entry_id, "modules": applies, "K": k, "codebook": codebook, "shape": shape, "scale_mode": scale_mode, "row_class": row_class})

    if not normalized_rows:
        fail("runtime-route manifest contains no measurements")

    module_registry: dict[str, Any] = {}
    option_count = 0
    for module, record in sorted(modules.items()):
        disposition = dispositions[f"{module}.weight"]
        policy = disposition["policy"]
        options: list[dict[str, Any]] = []
        selected = selected_ks(module, record, policy, justifications)
        for k in selected:
            matching = [
                row
                for row in normalized_rows
                if row["K"] == k
                and row["shape"] == record["shape"]
                and row["codebook"] == disposition["codebook"]
                and row["scale_mode"] == disposition["scale_mode"]
                and (not row["modules"] or module in row["modules"])
            ]
            classes = {row["row_class"] for row in matching}
            if classes != {"decode", "prefill"}:
                missing = sorted({"decode", "prefill"} - classes)
                fail(f"selected option {module} K{k} lacks measured runtime route classes {missing}")
            entry_ids = sorted({row["entry_id"] for row in matching})
            justification: str
            if k == record["recipe_K"]:
                justification = "incumbent"
            elif k in {3, 7}:
                justification = justifications[(module, k)]
            else:
                justification = "adjacent"
            options.append(
                {
                    "K": k,
                    "codebook": disposition["codebook"],
                    "scale_mode": disposition["scale_mode"],
                    "topology": "split_qkv",
                    "selection_basis": justification,
                    "proxy_error": record["rungs"][str(k)],
                    "registry_entries": entry_ids,
                }
            )
        option_count += len(options)
        module_registry[module] = {
            "source_tensor": f"{module}.weight",
            "shape": record["shape"],
            "numel": record["numel"],
            "policy": policy,
            "incumbent_K": record["recipe_K"],
            "role": record["role"],
            "layer": record["layer"],
            "qmap": record["qmap"],
            "options": options,
        }

    used_entries = {entry_id for module in module_registry.values() for option in module["options"] for entry_id in option["registry_entries"]}
    unused_entries = sorted(set(registry_entries) - used_entries)
    if unused_entries:
        fail(f"runtime-route manifest contains {len(unused_entries)} unselected/Cartesian entries")
    return {
        "entries": {entry_id: registry_entries[entry_id] for entry_id in sorted(used_entries)},
        "modules": module_registry,
    }, option_count


def validate_hydrated_manifest(
    manifest: dict[str, Any],
    revision: str,
    tensors: dict[str, dict[str, Any]],
) -> None:
    if manifest.get("schema") != "qwen38-quantization-manifest/1":
        fail(f"unsupported hydrated manifest schema {manifest.get('schema')!r}")
    if source_revision(manifest, "hydrated manifest") != revision:
        fail("hydrated manifest source revision disagrees with identity receipt")
    if integer(manifest.get("exl3_modules"), "hydrated manifest exl3_modules") != EXPECTED_EXL3_MODULES:
        fail("hydrated manifest does not declare the 409-module EXL3 set")
    logical_count = integer(manifest.get("logical_tensor_count"), "hydrated manifest logical_tensor_count")
    parameter_count = integer(manifest.get("logical_parameter_count"), "hydrated manifest logical_parameter_count")
    if logical_count != len(tensors):
        fail(f"hydrated logical tensor count {logical_count} disagrees with BF16 headers {len(tensors)}")
    if parameter_count != sum(tensor["numel"] for tensor in tensors.values()):
        fail("hydrated logical parameter count disagrees with BF16 headers")
    roles = as_object(manifest.get("roles"), "hydrated manifest roles")
    for role in ("embed_tokens", "vision_tower", "norms_and_small"):
        role_obj = as_object(roles.get(role), f"hydrated manifest role {role}")
        formats = as_object(role_obj.get("formats"), f"hydrated manifest role {role} formats")
        if set(formats) != {"BF16"}:
            fail(f"hydrated manifest role {role} is not fixed BF16")
    if integer(as_object(roles.get("lm_head"), "hydrated lm_head").get("modules"), "hydrated lm_head modules") != 1:
        fail("hydrated manifest must contain a separate one-module head policy")
    if integer(as_object(roles.get("mtp_draft"), "hydrated mtp_draft").get("modules"), "hydrated mtp modules") <= 0:
        fail("hydrated manifest must contain a separate MTP policy")
    role_module_sum = sum(integer(as_object(role, "hydrated role").get("modules"), "hydrated role modules") for role in roles.values())
    if role_module_sum != EXPECTED_EXL3_MODULES:
        fail(f"hydrated role module counts sum to {role_module_sum}, expected {EXPECTED_EXL3_MODULES}")


def verify_revision(model: Path, config: dict[str, Any], revision: str) -> None:
    commit = config.get("_commit_hash")
    if commit is not None and sha(commit, "config _commit_hash") != revision:
        fail("config _commit_hash disagrees with identity receipt")
    parts = model.resolve().parts
    if "snapshots" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("snapshots")
        if index + 1 >= len(parts) or parts[index + 1] != revision:
            fail("Hugging Face snapshot directory does not match identity revision")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate a Qwen3.8 BF16 snapshot and emit its target census and sparse measured route registry.",
        epilog="metadata-only reads config, index, and safetensors headers but neither tensor payloads nor Torch.",
    )
    result.add_argument("--model", required=True, type=Path, help="immutable BF16 snapshot directory")
    result.add_argument("--identity-receipt", required=True, type=Path, help="expected repo/revision/config/index/18-shard hashes")
    result.add_argument("--hydrated-manifest", required=True, type=Path, help="qualified hydrated quantization manifest")
    result.add_argument("--ladder", required=True, type=Path, help="measured 409-module K ladder")
    result.add_argument("--recipe-spec", required=True, type=Path, help="ordered or unique regex recipe specification")
    result.add_argument("--routes", required=True, type=Path, help="measured sparse runtime-route manifest")
    result.add_argument("--census-out", required=True, type=Path, help="BF16 census output JSON")
    result.add_argument("--registry-out", required=True, type=Path, help="compatibility registry output JSON")
    result.add_argument("--metadata-only", action="store_true", help="validate headers/index without reading shard payloads for SHA256")
    return result


def run(args: argparse.Namespace) -> None:
    model: Path = args.model
    if not model.is_dir():
        fail(f"model is not a directory: {model}")
    output_paths = {args.census_out.resolve(), args.registry_out.resolve()}
    if len(output_paths) != 2:
        fail("--census-out and --registry-out must be separate paths")
    if any(output.is_relative_to(model.resolve()) for output in output_paths):
        fail("outputs must not be written inside the immutable model snapshot")
    inputs = {
        "identity_receipt": (args.identity_receipt, as_object(load_strict_json(args.identity_receipt), "identity receipt")),
        "hydrated_manifest": (args.hydrated_manifest, as_object(load_strict_json(args.hydrated_manifest), "hydrated manifest")),
        "ladder": (args.ladder, as_object(load_strict_json(args.ladder), "ladder")),
        "recipe_spec": (args.recipe_spec, as_object(load_strict_json(args.recipe_spec), "recipe spec")),
        "routes": (args.routes, as_object(load_strict_json(args.routes), "runtime routes")),
    }
    input_paths = {path.resolve() for path, _ in inputs.values()}
    if output_paths & input_paths:
        fail("an output path aliases an immutable input manifest")
    identity = extract_identity(inputs["identity_receipt"][1])
    ordered_shards = validate_shard_names(identity["shard_sha256"])
    index_path = model / "model.safetensors.index.json"
    config_path = model / "config.json"
    index = as_object(load_strict_json(index_path), "safetensors index")
    config = as_object(load_strict_json(config_path), "model config")
    if sha256_file(index_path) != identity["index_sha256"]:
        fail("model.safetensors.index.json SHA256 disagrees with identity receipt")
    if sha256_file(config_path) != identity["config_sha256"]:
        fail("config.json SHA256 disagrees with identity receipt")
    verify_revision(model, config, identity["revision"])
    weight_map = as_object(index.get("weight_map"), "safetensors index weight_map")
    tensors, shard_summaries = parse_headers(model, ordered_shards, weight_map)
    if any(tensor["dtype"] != "BF16" for tensor in tensors.values()):
        counts = Counter(tensor["dtype"] for tensor in tensors.values())
        fail(f"immutable source is not an all-BF16 snapshot: {dict(counts)}")
    if len(tensors) != EXPECTED_LOGICAL_TENSORS:
        fail(f"BF16 snapshot has {len(tensors)} logical tensors, expected {EXPECTED_LOGICAL_TENSORS}")
    parameter_count = sum(tensor["numel"] for tensor in tensors.values())
    if parameter_count != EXPECTED_LOGICAL_PARAMETERS:
        fail(
            f"BF16 snapshot has {parameter_count} logical parameters, "
            f"expected {EXPECTED_LOGICAL_PARAMETERS}"
        )
    metadata = as_object(index.get("metadata"), "safetensors index metadata")
    total_size = integral_number(metadata.get("total_size"), "safetensors index metadata.total_size")
    header_payload_size = sum(tensor["bytes"] for tensor in tensors.values())
    if total_size != header_payload_size:
        fail(f"index total_size {total_size} disagrees with header payload {header_payload_size}")
    if header_payload_size != EXPECTED_BF16_PAYLOAD_BYTES:
        fail(
            f"BF16 payload is {header_payload_size} bytes, "
            f"expected {EXPECTED_BF16_PAYLOAD_BYTES}"
        )

    manifest = inputs["hydrated_manifest"][1]
    ladder_doc = inputs["ladder"][1]
    recipe = inputs["recipe_spec"][1]
    routes = inputs["routes"][1]
    validate_hydrated_manifest(manifest, identity["revision"], tensors)
    if source_revision(recipe, "recipe") != identity["revision"]:
        fail("recipe source revision disagrees with identity receipt")
    module_names = set(as_object(ladder_doc.get("modules"), "ladder modules"))
    dispositions, rule_counts = resolve_recipe(recipe, tensors, module_names)
    modules = validate_ladder(ladder_doc, tensors, dispositions)
    validate_qkv(tensors, dispositions)
    registry_body, option_count = build_registry(routes, modules, dispositions, recipe)

    actual_shards: dict[str, dict[str, Any]] = {}
    for shard_name in ordered_shards:
        summary: dict[str, Any] = dict(shard_summaries[shard_name])
        summary["expected_sha256"] = identity["shard_sha256"][shard_name]
        if args.metadata_only:
            summary["sha256_status"] = "not_read_metadata_only"
        else:
            digest = sha256_file(model / shard_name)
            if digest != identity["shard_sha256"][shard_name]:
                fail(f"{shard_name} SHA256 disagrees with identity receipt")
            summary["sha256"] = digest
            summary["sha256_status"] = "verified"
        actual_shards[shard_name] = summary

    input_hashes = {label: sha256_file(path) for label, (path, _) in inputs.items()}
    census_tensors = {
        name: {
            **tensors[name],
            "role": tensor_role(name),
            "layer": tensor_layer(name),
            "target": dispositions[name],
        }
        for name in sorted(tensors)
    }
    census = {
        "schema": CENSUS_SCHEMA,
        "validation_mode": "metadata_only" if args.metadata_only else "full_hash",
        "source": {
            "repo": identity["repo"],
            "revision": identity["revision"],
            "config_sha256": identity["config_sha256"],
            "index_sha256": identity["index_sha256"],
            "shards": actual_shards,
        },
        "inputs": input_hashes,
        "summary": {
            "shard_count": len(ordered_shards),
            "logical_tensor_count": len(tensors),
            "logical_parameter_count": sum(tensor["numel"] for tensor in tensors.values()),
            "payload_bytes": header_payload_size,
            "exl3_module_count": len(modules),
            "bf16_target_tensor_count": sum(disposition["format"] == "bf16" for disposition in dispositions.values()),
            "target_policy_counts": dict(sorted(Counter(disposition["policy"] for disposition in dispositions.values()).items())),
        },
        "recipe": {
            "schema": recipe["schema"],
            "topology": "split_qkv",
            "resolution": recipe["resolution"],
            "rule_match_counts": rule_counts,
        },
        "tensors": census_tensors,
    }
    registry = {
        "schema": REGISTRY_SCHEMA,
        "source": {
            "repo": identity["repo"],
            "revision": identity["revision"],
            "census_sha256": canonical_sha256(census),
            "ladder_sha256": input_hashes["ladder"],
            "recipe_spec_sha256": input_hashes["recipe_spec"],
            "routes_sha256": input_hashes["routes"],
        },
        "policy": {
            "topology": "split_qkv",
            "option_domain": "sparse_incumbent_adjacent_with_justified_edges",
            "measured_routes_only": True,
            "cartesian_bank": False,
            "mixed_width_qkv": False,
        },
        "summary": {
            "module_count": len(modules),
            "option_count": option_count,
            "registry_entry_count": len(registry_body["entries"]),
        },
        **registry_body,
    }
    atomic_write_json(args.census_out, census)
    atomic_write_json(args.registry_out, registry)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except (CensusError, OSError, ValueError) as exc:
        print(f"frontier_census: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
