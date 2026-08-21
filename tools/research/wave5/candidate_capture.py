#!/usr/bin/env python3
"""One-tensor R30 EXL3 checkpoint materializer and R31 capture launcher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

CHECKPOINT_SCHEMA = "qwen38-wave5-one-tensor-checkpoint/1"
BINDING_SCHEMA = "qwen38-wave5-capture-binding/2"
EXECUTION_SCHEMA = "qwen38-wave5-capture-execution/1"
RESULT_SCHEMA = "qwen38-wave5-candidate-capture-result/1"
BYTE_MANIFEST_SCHEMA = "qwen38-wave5-checkpoint-byte-manifest/1"
ROUTE_ID = "codec-exact/all-trellis-stock-exl3"
PHYSICAL_SUFFIXES = ("suh", "svh", "trellis", "mcg", "mul1")
SHA_CHUNK = 8 * 1024 * 1024
MIN_FREE_BYTES = 60 * 1024**3
CAPTURE_BYTES = 512 * 2047 * 5120 * 2
LEGACY_ROOT = Path("/tmp/kld-data")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HARNESS = Path(__file__).with_name("exl3_action.py")
QUALIFIED_OPERATIONAL_HARNESS_SHA256 = "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d"
QUALIFIED_OPERATIONAL_EXTENSION_SHA256 = "e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e"
CANONICAL_V5_REPO = "malaiwah/qwen38-27b-fidelity-suite-v5"
CANONICAL_V5_REVISION = "7797fcce3ffed62b99871348887f4626dc9b2b3b"
CANONICAL_V5_REFERENCE_PATH = "reference/hidden-bf16"
CANONICAL_V5_REFERENCE_MANIFEST_SHA256 = "01a2f676edcf5d1f958f0e0ffb9dfa1dd8cf5671e3d8c69b822cb806e13200ec"
CANONICAL_V3_REPO = "malaiwah/qwen38-27b-fidelity-suite-v3"
CANONICAL_V3_REVISION = "73252e77e96bbf5596484e29dd8041f9f38c95f1"
CANONICAL_V3_HEAD_PATH = "lm-head/weight.safetensors"


class CaptureError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(SHA_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def reject_legacy(label: str, path: Path) -> None:
    lexical = Path(os.path.abspath(path))
    resolved = path.resolve()
    if (lexical == LEGACY_ROOT or LEGACY_ROOT in lexical.parents
            or resolved == LEGACY_ROOT or LEGACY_ROOT in resolved.parents):
        raise CaptureError(f"{label} may not use legacy {LEGACY_ROOT}: {path}")


def require_file(label: str, path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise CaptureError(f"missing {label}: {path}")
    return path


def require_dir(label: str, path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise CaptureError(f"missing {label}: {path}")
    return path


def run_checked(command: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(list(command), check=True, text=True,
                              capture_output=capture_output)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = f" stdout={exc.stdout!r} stderr={exc.stderr!r}"
        raise CaptureError(f"command failed: {list(command)!r}{detail}") from exc


# Minimal streaming safetensors reader/writer. Only the changed payload is held in RAM.
DTYPE_NAMES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16",
               "U16": "uint16", "I32": "int32", "U32": "uint32", "I64": "int64",
               "U64": "uint64", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
               "F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def parse_safetensors(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise CaptureError(f"truncated safetensors prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 0 < header_length <= path.stat().st_size - 8:
            raise CaptureError(f"invalid safetensors header length: {path}")
        raw = handle.read(header_length)
    try:
        header = json.loads(raw.rstrip(b" \t\r\n\x00"))
    except json.JSONDecodeError as exc:
        raise CaptureError(f"invalid safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise CaptureError(f"safetensors header is not an object: {path}")
    data_size = path.stat().st_size - 8 - header_length
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise CaptureError(f"invalid tensor entry {name!r} in {path}")
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in shape):
            raise CaptureError(f"invalid dtype/shape for {name!r} in {path}")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(
                isinstance(v, int) and not isinstance(v, bool) for v in offsets)):
            raise CaptureError(f"invalid offsets for {name!r} in {path}")
        start, end = offsets
        if (start < 0 or end < start or end - start != product(shape) * DTYPE_BYTES[dtype]
                or end > data_size):
            raise CaptureError(f"invalid byte range for {name!r} in {path}")
        ranges.append((start, end, name))
    ordered = sorted(ranges)
    if ordered and ([x[0] for x in ordered] != [0] + [x[1] for x in ordered[:-1]]
                    or ordered[-1][1] != data_size):
        raise CaptureError(f"non-contiguous safetensors data ranges: {path}")
    return header, 8 + header_length


def tensor_names(header: dict[str, Any]) -> list[str]:
    return [name for name in header if name != "__metadata__"]


def hash_range(source: BinaryIO, offset: int, length: int) -> str:
    source.seek(offset)
    digest, remaining = hashlib.sha256(), length
    while remaining:
        block = source.read(min(SHA_CHUNK, remaining))
        if not block:
            raise CaptureError("unexpected EOF while hashing tensor buffer")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def copy_range(source: BinaryIO, target: BinaryIO, offset: int,
               length: int) -> tuple[str, str]:
    """Prefer filesystem range cloning, then hash the source bytes independently."""
    target_offset = target.tell()
    copied = 0
    method = "range-clone"
    if hasattr(os, "copy_file_range"):
        try:
            while copied < length:
                count = min(1024**3, length - copied)
                done = os.copy_file_range(
                    source.fileno(), target.fileno(), count,
                    offset_src=offset + copied, offset_dst=target_offset + copied)
                if done == 0:
                    raise OSError("copy_file_range made no progress")
                copied += done
        except OSError:
            method = "range-clone+copy" if copied else "copy"
    else:
        method = "copy"
    if copied < length:
        source.seek(offset + copied)
        target.seek(target_offset + copied)
        remaining = length - copied
        while remaining:
            block = source.read(min(SHA_CHUNK, remaining))
            if not block:
                raise CaptureError("unexpected EOF while copying tensor buffer")
            target.write(block)
            remaining -= len(block)
    target.seek(target_offset + length)
    return hash_range(source, offset, length), method


def payload_manifest(payload: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    header, data_start = parse_safetensors(payload)
    manifest: dict[str, Any] = {}
    buffers: dict[str, bytes] = {}
    with payload.open("rb") as handle:
        for name in tensor_names(header):
            entry = header[name]
            start, end = entry["data_offsets"]
            handle.seek(data_start + start)
            raw = handle.read(end - start)
            if len(raw) != end - start:
                raise CaptureError(f"truncated payload tensor {name}")
            buffers[name] = raw
            manifest[name] = {"dtype": DTYPE_NAMES[entry["dtype"]], "shape": entry["shape"],
                              "bytes": len(raw), "sha256": sha256_bytes(raw),
                              "safetensors_dtype": entry["dtype"]}
    return manifest, buffers


def action_payload_digest(manifest: dict[str, Any], buffers: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(manifest):
        row = manifest[name]
        descriptor = json.dumps({
            "name": name,
            "dtype": row["dtype"],
            "shape": row["shape"],
            "bytes": len(buffers[name]),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(buffers[name])
    return digest.hexdigest()


def encoded_header(header: dict[str, Any]) -> bytes:
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    return raw + b" " * ((-len(raw)) % 8)


def rewrite_safetensors(source: Path, destination: Path,
                        replacements: dict[str, tuple[dict[str, Any], bytes]]) -> dict[str, Any]:
    source_header, source_data = parse_safetensors(source)
    names = sorted(tensor_names(source_header),
                   key=lambda name: source_header[name]["data_offsets"][0])
    if not replacements or not set(replacements) <= set(names):
        raise CaptureError("replacement names are empty or absent from source shard")
    for name, (row, raw) in replacements.items():
        source_entry = source_header[name]
        start, end = source_entry["data_offsets"]
        if (row["safetensors_dtype"] != source_entry["dtype"]
                or row["shape"] != source_entry["shape"]
                or len(raw) != end - start):
            raise CaptureError(
                f"replacement descriptor/byte length differs from source buffer {name}")
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    unchanged: dict[str, str] = {}
    replaced: dict[str, str] = {}
    range_methods: dict[str, int] = {}
    with source.open("rb") as src, temporary.open("xb") as dst:
        source_prefix_and_header = src.read(source_data)
        if len(source_prefix_and_header) != source_data:
            raise CaptureError("source safetensors header was truncated")
        dst.write(source_prefix_and_header)
        for name in names:
            if name in replacements:
                raw = replacements[name][1]
                dst.write(raw)
                replaced[name] = sha256_bytes(raw)
            else:
                start, end = source_header[name]["data_offsets"]
                digest, method = copy_range(src, dst, source_data + start, end - start)
                unchanged[name] = digest
                range_methods[method] = range_methods.get(method, 0) + 1
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    candidate_header, candidate_data = parse_safetensors(destination)
    if candidate_data != source_data or candidate_header != source_header:
        raise CaptureError("candidate safetensors header differs from source")
    with destination.open("rb") as candidate_file:
        candidate_prefix_and_header = candidate_file.read(candidate_data)
    return {
        "source_header_sha256": sha256_bytes(source_prefix_and_header),
        "candidate_header_sha256": sha256_bytes(candidate_prefix_and_header),
        "unchanged_tensor_buffers": len(unchanged),
        "unchanged_tensor_buffer_manifest_sha256": sha256_bytes(canonical_json(unchanged)),
        "range_copy_methods": range_methods,
        "replacement_tensor_buffers": replaced,
        "candidate_shard_sha256": sha256_file(destination),
        "candidate_shard_bytes": destination.stat().st_size,
    }


def load_action_module(path: Path) -> Any:
    path = require_file("R30 action harness", path)
    spec = importlib.util.spec_from_file_location("wave5_exl3_action", path)
    if spec is None or spec.loader is None:
        raise CaptureError(f"cannot import R30 action harness {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_action(action_path: Path, payload: Path, harness_path: Path,
                  expected_harness_sha256: str,
                  operational_extension_sha256: str) -> tuple[
                      dict[str, Any], str, dict[str, Any], dict[str, bytes]]:
    action_path = require_file("action", action_path); payload = require_file("payload", payload)
    if (expected_harness_sha256 != QUALIFIED_OPERATIONAL_HARNESS_SHA256
            or operational_extension_sha256 != QUALIFIED_OPERATIONAL_EXTENSION_SHA256):
        raise CaptureError("campaign harness/extension pins are not overridable")
    harness_sha = sha256_file(require_file("R30 action harness", harness_path))
    if harness_sha != QUALIFIED_OPERATIONAL_HARNESS_SHA256:
        raise CaptureError("R30 action harness hash differs from the fixed campaign pin")
    module = load_action_module(harness_path)
    if (getattr(module, "STOCK_EXTENSION_BINARY_SHA256", None)
            != QUALIFIED_OPERATIONAL_EXTENSION_SHA256):
        raise CaptureError("R30 harness extension binary hash differs from the fixed campaign pin")
    try:
        action_object = module.load_action(action_path)
        action, identity = action_object.to_dict(), action_object.identity_sha256()
    except Exception as exc:
        raise CaptureError(f"R30 EXL3Action validation failed: {exc}") from exc
    keys = action.get("unit", {}).get("tensor_keys")
    if (action.get("schema") != "wave5/exl3-action/1"
            or action.get("unit", {}).get("granularity") != "tensor"
            or not isinstance(keys, list) or len(keys) != 1):
        raise CaptureError("launcher accepts exactly one full logical EXL3 tensor action")
    if action.get("runtime", {}).get("route_id") != ROUTE_ID:
        raise CaptureError(f"capture requires codec-exact route {ROUTE_ID}")
    if action.get("hashes", {}).get("action_identity_sha256") != identity:
        raise CaptureError("action identity hash is stale")
    manifest, buffers = payload_manifest(payload)
    expected = {"suh", "svh", "trellis"}
    marker = {"mcg": "mcg", "mul1": "mul1", "3inst": None}.get(action.get("codebook"))
    if marker: expected.add(marker)
    if set(manifest) != expected:
        raise CaptureError(f"payload keys {sorted(manifest)} differ from action {sorted(expected)}")
    stripped = {name: {k: v for k, v in row.items() if k != "safetensors_dtype"}
                for name, row in manifest.items()}
    serialized = action.get("serialized")
    if (not isinstance(serialized, dict) or serialized.get("buffers") != stripped
            or serialized.get("safetensors_sha256") != sha256_file(payload)):
        raise CaptureError("payload file/buffer manifest differs from EXL3Action.serialized")
    if action.get("hashes", {}).get("payload_sha256") != action_payload_digest(manifest, buffers):
        raise CaptureError("payload digest differs from EXL3Action")
    return action, identity, manifest, buffers


def checkpoint_files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*")
                   if (path.is_file() or path.is_symlink())
                   and ".git" not in path.relative_to(root).parts),
                  key=lambda path: path.relative_to(root).as_posix())


def resolved_regular(path: Path) -> Path:
    target = path.resolve() if path.is_symlink() else path
    if not target.is_file(): raise CaptureError(f"not a regular checkpoint file: {path}")
    return target


def component_manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": visible.relative_to(root).as_posix(),
             "bytes": resolved_regular(visible).stat().st_size,
             "sha256": sha256_file(resolved_regular(visible))}
            for visible in checkpoint_files(root)]


def link_or_clone(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True); source = resolved_regular(source)
    try:
        os.link(source, destination); return "hardlink"
    except OSError:
        if sys.platform == "darwin":
            try:
                subprocess.run(["cp", "-c", str(source), str(destination)], check=True)
                return "reflink"
            except (OSError, subprocess.CalledProcessError): pass
        shutil.copyfile(source, destination); return "copy"


def source_revision(root: Path) -> str | None:
    path = root / "revision.txt"
    return path.read_text().split("\n", 1)[0].strip() or None if path.is_file() else None


def checkpoint_identity(root: Path, shards: dict[str, str], index_sha: str,
                        config_sha: str) -> tuple[dict[str, Any], str]:
    identity = {"model_revision": source_revision(root), "index_sha256": index_sha,
                "config_sha256": config_sha, "shard_sha256": dict(sorted(shards.items()))}
    return identity, sha256_bytes(canonical_json(identity))


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def ensure_capacity(path: Path, needed: int) -> None:
    free = free_bytes(path)
    if free - needed < MIN_FREE_BYTES:
        raise CaptureError(f"disk gate failed: free={free}, new={needed}, floor={MIN_FREE_BYTES}")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    base = require_dir("source checkpoint", args.base_checkpoint)
    candidate, evidence = args.candidate_checkpoint.resolve(), args.evidence_dir.resolve()
    if (candidate == base or inside(base, candidate) or inside(candidate, base)
            or evidence == candidate or inside(evidence, candidate)
            or evidence == base or inside(evidence, base)):
        raise CaptureError(
            "source checkpoint, candidate checkpoint, and evidence must be disjoint")
    if candidate.exists() or (evidence / "checkpoint-manifest.json").exists():
        raise CaptureError("candidate or checkpoint evidence already exists")
    index_path = require_file("checkpoint index", base / "model.safetensors.index.json")
    config_path = require_file("checkpoint config", base / "config.json")
    index_raw = index_path.read_bytes()
    try: weight_map = json.loads(index_raw)["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CaptureError(f"invalid checkpoint index: {exc}") from exc
    action, action_identity, payload_rows, payload_raw = verify_action(
        args.action, args.payload, args.action_harness,
        args.expected_action_harness_sha256, args.operational_extension_sha256)
    logical = action["unit"]["tensor_keys"][0]
    if not logical.endswith(".weight"): raise CaptureError("logical tensor key must end in .weight")
    prefix = logical.removesuffix(".weight")
    physical = {name: f"{prefix}.{name}" for name in payload_rows}
    base_physical = {key.rsplit(".", 1)[1]: key for key in weight_map
                     if key.rsplit(".", 1)[0] == prefix and key.rsplit(".", 1)[1] in PHYSICAL_SUFFIXES}
    if base_physical != physical:
        raise CaptureError(f"non-unique/wrong replacement: base={base_physical}, payload={physical}")
    shards = {weight_map[key] for key in physical.values()}
    if len(shards) != 1: raise CaptureError("one logical tensor spans multiple source shards")
    changed_shard = shards.pop(); source_shard = require_file("source shard", base / changed_shard)
    source_header, _ = parse_safetensors(source_shard)
    if set(tensor_names(source_header)) != {key for key, shard in weight_map.items() if shard == changed_shard}:
        raise CaptureError("source shard header/index disagree")
    estimated_materialization = source_shard.stat().st_size
    disk_before = free_bytes(candidate.parent)
    ensure_capacity(candidate.parent, estimated_materialization + CAPTURE_BYTES)
    candidate.mkdir(parents=True); evidence.mkdir(parents=True, exist_ok=True)
    methods: dict[str, int] = {}
    try:
        for visible in checkpoint_files(base):
            relative = visible.relative_to(base)
            if relative.as_posix() == changed_shard: continue
            method = link_or_clone(visible, candidate / relative)
            methods[method] = methods.get(method, 0) + 1
        rewrite = rewrite_safetensors(source_shard, candidate / changed_shard,
            {physical[name]: (payload_rows[name], payload_raw[name]) for name in payload_rows})
        if (candidate / "model.safetensors.index.json").read_bytes() != index_raw:
            raise CaptureError("candidate index drift")
        if sha256_file(candidate / "config.json") != sha256_file(config_path):
            raise CaptureError("candidate config drift")
        base_rows, cand_rows = component_manifest(base), component_manifest(candidate)
        base_by = {x["path"]: x for x in base_rows}; cand_by = {x["path"]: x for x in cand_rows}
        if set(base_by) != set(cand_by): raise CaptureError("checkpoint component set drift")
        drift = [path for path in base_by if path != changed_shard and base_by[path] != cand_by[path]]
        if drift: raise CaptureError(f"unchanged component drift: {drift[:8]}")
        shard_names = sorted(set(weight_map.values()))
        base_shards = {name: base_by[name]["sha256"] for name in shard_names}
        cand_shards = {name: cand_by[name]["sha256"] for name in shard_names}
        unchanged = {name: cand_shards[name] for name in shard_names if name != changed_shard}
        if unchanged != {name: base_shards[name] for name in unchanged}:
            raise CaptureError("unchanged shard drift")
        index_sha, config_sha = sha256_bytes(index_raw), sha256_file(config_path)
        source_id, source_id_sha = checkpoint_identity(base, base_shards, index_sha, config_sha)
        cand_id, cand_id_sha = checkpoint_identity(candidate, cand_shards, index_sha, config_sha)
        action_copy, payload_copy = evidence / "action.json", evidence / "changed-payload.safetensors"
        shutil.copyfile(args.action, action_copy); shutil.copyfile(args.payload, payload_copy)
        preserved_shard = evidence / "changed-shard" / changed_shard
        preserve_method = link_or_clone(candidate / changed_shard, preserved_shard)
        disk_after = free_bytes(candidate)
        if disk_after - CAPTURE_BYTES < MIN_FREE_BYTES:
            raise CaptureError(
                "materialized checkpoint cannot leave 60 GiB free after candidate capture")
        exact_bytes = sum(row["bytes"] for row in cand_rows)
        byte_manifest = {"schema": BYTE_MANIFEST_SCHEMA,
                         "checkpoint_identity_sha256": cand_id_sha,
                         "checkpoint_tree_sha256": sha256_bytes(canonical_json(cand_rows)),
                         "total_bytes": exact_bytes, "components": cand_rows}
        byte_path = evidence / "checkpoint-bytes.json"; write_json(byte_path, byte_manifest)
        result = {
            "schema": CHECKPOINT_SCHEMA, "source_checkpoint": str(base),
            "candidate_checkpoint": str(candidate), "source_checkpoint_identity": source_id,
            "source_checkpoint_identity_sha256": source_id_sha,
            "candidate_checkpoint_identity": cand_id,
            "candidate_checkpoint_identity_sha256": cand_id_sha,
            "source_index_sha256": index_sha,
            "candidate_index_sha256": sha256_file(
                candidate / "model.safetensors.index.json"),
            "source_config_sha256": config_sha,
            "candidate_config_sha256": sha256_file(candidate / "config.json"),
            "logical_tensor_key": logical, "physical_tensor_keys": physical,
            "action_identity_sha256": action_identity, "action_file_sha256": sha256_file(args.action),
            "action_harness_sha256": sha256_file(args.action_harness),
            "operational_extension_sha256": args.operational_extension_sha256,
            "action_payload_sha256": action["hashes"]["payload_sha256"],
            "action_source_tensor_sha256": action["source_tensor_sha256"],
            "action_source_revision": action["source_revision"],
            "action_split_manifest_sha256": action["split_manifest_sha256"],
            "action_split_manifest_content_sha256":
                action["split_manifest_content_sha256"],
            "action_split_selections": action["split_selections"],
            "action_split_disjointness_sha256":
                action["split_disjointness"]["artifact_sha256"],
            "codec_route_id": action["runtime"]["route_id"], "changed_shard": changed_shard,
            "source_changed_shard_sha256": base_shards[changed_shard],
            "candidate_changed_shard_sha256": cand_shards[changed_shard],
            "unchanged_shard_sha256": unchanged, "changed_shard_rewrite": rewrite,
            "unchanged_component_count": len(cand_rows) - 1,
            "unchanged_component_manifest_sha256": sha256_bytes(canonical_json(
                [x for x in cand_rows if x["path"] != changed_shard])),
            "exact_checkpoint_bytes": exact_bytes,
            "checkpoint_byte_manifest": {"path": str(byte_path), "sha256": sha256_file(byte_path)},
            "preserved_action": {"path": str(action_copy), "sha256": sha256_file(action_copy)},
            "preserved_payload": {"path": str(payload_copy), "sha256": sha256_file(payload_copy)},
            "preserved_changed_shard": {"path": str(preserved_shard),
                "sha256": sha256_file(preserved_shard), "method": preserve_method},
            "clone_methods": methods,
            "disk": {
                "free_before_bytes": disk_before,
                "free_after_materialize_bytes": disk_after,
                "reserved_capture_bytes": CAPTURE_BYTES,
                "minimum_free_bytes": MIN_FREE_BYTES,
            },
            "complete": True}
        write_json(evidence / "checkpoint-manifest.json", result); return result
    except Exception:
        shutil.rmtree(candidate, ignore_errors=True); raise


def selection_sha256(manifest: dict[str, Any], split: str) -> str:
    documents = manifest.get("documents")
    if not isinstance(documents, list): raise CaptureError("R29 split manifest lacks documents")
    selected = [x for x in documents if isinstance(x, dict) and x.get("split") == split]
    projection = {"schema": "qwen38-wave5-split-projection/1",
                  "selector": {"field": "split", "op": "eq", "value": split},
                  "document_sha256": sorted(x["document_sha256"] for x in selected),
                  "context_token_sha256": sorted(c["token_sha256"] for x in selected
                                                 for c in (x.get("contexts") or []))}
    digest = sha256_bytes(canonical_json(projection))
    if manifest.get("split_contracts", {}).get(split, {}).get("projection_sha256", digest) != digest:
        raise CaptureError("R29 embedded split projection hash is stale")
    return digest


def validation_inputs(args: argparse.Namespace) -> dict[str, Any]:
    r31 = require_dir("R31 repository", args.r31_root)
    paths = {"prereg": require_file("R31 prereg", r31 / "receipts/wave5/fidelity-prereg.json"),
             "contract": require_file("R31 contract", r31 / "tools/research/wave5/fidelity_contract.json"),
             "gate": require_file("R31 gate", r31 / "tools/research/wave5/fidelity_gate.py"),
             "projector": require_file("R31 projector", r31 / "tools/fidelity.py")}
    prereg, contract = load_json(paths["prereg"]), load_json(paths["contract"])
    if prereg.get("contract_sha256") != sha256_file(paths["contract"]): raise CaptureError("R31 contract drift")
    if prereg.get("gate_sha256") != sha256_file(paths["gate"]): raise CaptureError("R31 gate drift")
    if prereg.get("projector_sha256") != sha256_file(paths["projector"]): raise CaptureError("R31 projector drift")
    if sha256_file(REPO_ROOT / "tools/fidelity.py") != prereg["projector_sha256"]:
        raise CaptureError("repository-head fidelity.py differs from R31 projector")
    split_path = require_file("R29 split manifest", args.r29_split_manifest)
    data_path = require_file("R29 data manifest", args.r29_data_manifest)
    binding = prereg["r29_bindings"]
    if (sha256_file(split_path) != binding["split_manifest_file_sha256"]
            or sha256_file(data_path) != binding["data_manifest_file_sha256"]):
        raise CaptureError("R29 file pin mismatch")
    split_manifest, data_manifest = load_json(split_path), load_json(data_path)
    if (split_manifest.get("content_sha256") != binding["split_manifest_content_sha256"]
            or data_manifest.get("content_sha256") != binding["data_manifest_content_sha256"]):
        raise CaptureError("R29 content pin mismatch")
    validation = prereg["split_registry"]["validation"]
    if selection_sha256(split_manifest, "validation") != validation["selection_sha256"]:
        raise CaptureError("R29 validation selector mismatch")
    reject_legacy("suite", args.suite_root)
    suite_root = require_dir("pinned v5 suite root", args.suite_root)
    suite_dir = suite_root / "shard-0000"; suite_path = require_file("v5 shard0 manifest", suite_dir / "suite-manifest.json")
    suite = load_json(suite_path)
    if (sha256_file(suite_path) != validation["retained_suite_manifest_sha256"]
            or suite.get("suite_token_sha256") != validation["retained_suite_token_sha256"]):
        raise CaptureError("v5 shard0 identity mismatch")
    contexts = {int(x["index"]): x for x in suite.get("context_index", [])}; selected: dict[int, str] = {}
    for document in split_manifest["documents"]:
        if document.get("split") != "validation": continue
        for context in document.get("contexts") or []:
            if int(context["shard"]) != 0: raise CaptureError("validation/test selector mixup")
            index = int(context["index"])
            if index in selected: raise CaptureError("duplicate validation context")
            selected[index] = context["token_sha256"]
    if (len(selected) != validation["primary_v5_contexts"] or set(selected) != set(contexts)
            or any(contexts[i]["token_sha256"] != value for i, value in selected.items())):
        raise CaptureError("R29 validation set differs from v5 shard0 tokens")
    reject_legacy("head", args.shared_head)
    head = require_file("pinned v3 shared BF16 head", args.shared_head)
    if sha256_file(head) != contract["reference_semantics"]["shared_head_sha256"]:
        raise CaptureError("shared head identity mismatch")
    reject_legacy("reference", args.reference_root)
    reference = require_dir("pinned v5 BF16 reference", args.reference_root)
    reference_shard = reference / "shard-0000" if (reference / "shard-0000").is_dir() else reference
    reference_manifest_path = require_file(
        "reference manifest", reference_shard / "capture-manifest.json")
    reference_manifest = load_json(reference_manifest_path)
    reference_is_mirror = inside(reference, LEGACY_ROOT)
    if sha256_file(reference_manifest_path) != CANONICAL_V5_REFERENCE_MANIFEST_SHA256:
        raise CaptureError("reference differs from the pinned public manifest")
    if reference_manifest.get("complete") is not True or reference_manifest.get("suite_token_sha256") != suite["suite_token_sha256"]:
        raise CaptureError("retained reference is incomplete or belongs to another suite")
    records = reference_manifest.get("captures")
    if not isinstance(records, list) or {row.get("index") for row in records} != set(contexts):
        raise CaptureError("retained reference does not contain the exact shard0 context set")
    for row in records:
        hidden = require_file(
            "retained BF16 hidden state",
            reference_shard / f"hidden_{row['index']:04d}.safetensors")
        if sha256_file(hidden) != row.get("sha256"):
            raise CaptureError(f"retained reference digest mismatch: {hidden}")
    canonical_artifacts = {
        "v5": {
            "repo": CANONICAL_V5_REPO,
            "revision": CANONICAL_V5_REVISION,
            "suite_path": "suite/shard-0000",
            "suite_manifest_sha256": sha256_file(suite_path),
            "suite_token_sha256": suite["suite_token_sha256"],
            "reference_path": CANONICAL_V5_REFERENCE_PATH,
            "reference_manifest_sha256": sha256_file(reference_manifest_path),
            "local_resolved_reference": str(reference),
            "verified_local_mirror": reference_is_mirror,
        },
        "v3": {
            "repo": CANONICAL_V3_REPO,
            "revision": CANONICAL_V3_REVISION,
            "head_path": CANONICAL_V3_HEAD_PATH,
            "head_sha256": sha256_file(head),
            "local_resolved_head": str(head),
            "verified_local_mirror": inside(head, LEGACY_ROOT),
        },
    }
    return {"r31": r31, "paths": paths, "prereg": prereg, "contract": contract,
            "split_path": split_path, "data_path": data_path, "validation": validation,
            "suite_root": suite_root, "suite_dir": suite_dir, "suite": suite,
            "head": head, "reference": reference,
            "canonical_artifacts": canonical_artifacts}

def validate_action_lineage(checkpoint: dict[str, Any],
                            inputs: dict[str, Any]) -> None:
    split_contract = inputs["contract"]["split_identity_contract"]
    expected_selections = {
        name: {
            "selection_sha256": row["selection_sha256"],
            "selector": row["selector"],
        }
        for name, row in split_contract["labels"].items()
    }
    if (checkpoint.get("action_split_manifest_sha256")
            != split_contract["manifest_file_sha256"]
            or checkpoint.get("action_split_manifest_content_sha256")
            != split_contract["manifest_content_sha256"]
            or checkpoint.get("action_split_selections") != expected_selections
            or checkpoint.get("action_split_disjointness_sha256")
            != inputs["prereg"]["r29_bindings"]["leakage_audit_sha256"]):
        raise CaptureError("EXL3Action data/split lineage differs from R29/R31")



def link_r31_runtime(inputs: dict[str, Any], destination: Path) -> Path:
    if destination.exists(): shutil.rmtree(destination)
    files = {"tools/research/wave5/fidelity_gate.py": inputs["paths"]["gate"],
             "tools/research/wave5/fidelity_contract.json": inputs["paths"]["contract"],
             "tools/fidelity.py": inputs["paths"]["projector"],
             "receipts/wave5/fidelity-prereg.json": inputs["paths"]["prereg"],
             "receipts/wave5/split-manifest.json": inputs["split_path"],
             "receipts/wave5/data-manifest.json": inputs["data_path"]}
    for relative, source in files.items(): link_or_clone(source, destination / relative)
    return destination


def inspect_image(image: str) -> dict[str, Any]:
    try: row = json.loads(run_checked(["podman", "image", "inspect", image], capture_output=True).stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc: raise CaptureError(f"bad image identity: {exc}") from exc
    return {"requested": image, "id": row.get("Id"), "digest": row.get("Digest"), "repo_digests": row.get("RepoDigests")}


def gpu_identity() -> dict[str, Any]:
    output = run_checked(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"], capture_output=True).stdout
    rows = [x.strip() for x in output.splitlines() if x.strip()]
    if len(rows) != 1: raise CaptureError(f"expected one visible GPU, got {rows}")
    name, uuid, driver = [x.strip() for x in rows[0].split(",", 2)]
    return {"name": name, "uuid": uuid, "driver_version": driver, "container_device": "nvidia.com/gpu=all"}


def mounts(paths: Iterable[tuple[Path, bool]]) -> list[str]:
    result: list[str] = []; seen: set[Path] = set()
    for path, writable in paths:
        path = path.resolve()
        if path in seen: continue
        seen.add(path); result += ["--volume", f"{path}:{path}:{'rw' if writable else 'ro'}"]
    return result


def prepare_capture_suite_view(inputs: dict[str, Any], destination: Path) -> Path:
    """Adapt the published ``shard-0000`` + parent ``tokens`` layout for fidelity.py."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    link_or_clone(inputs["suite_dir"] / "suite-manifest.json",
                  destination / "suite-manifest.json")
    for context in inputs["suite"]["context_index"]:
        relative = Path(context["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CaptureError(f"unsafe v5 token path: {relative}")
        source = inputs["suite_dir"] / relative
        if not source.is_file():
            source = inputs["suite_root"] / relative
        source = require_file("v5 token IDs", source)
        link_or_clone(source, destination / relative)
    return destination


def capture_candidate(args: argparse.Namespace, checkpoint: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    validate_action_lineage(checkpoint, inputs)
    candidate = require_dir("candidate checkpoint", Path(checkpoint["candidate_checkpoint"]))
    reject_legacy("capture", args.capture_dir)
    capture = args.capture_dir.resolve()
    if capture.exists() and any(capture.iterdir()): raise CaptureError("capture directory is not empty")
    capture.mkdir(parents=True, exist_ok=True)
    disk_before_capture = free_bytes(capture)
    ensure_capacity(capture, CAPTURE_BYTES)
    suite_view = prepare_capture_suite_view(
        inputs, args.evidence_dir.resolve() / "capture-suite-view")
    image, gpu = inspect_image(args.container_image), gpu_identity(); fidelity = REPO_ROOT / "tools/fidelity.py"
    command = ["podman", "run", "--rm", "--name", f"wave5-capture-{os.getpid()}",
               "--device", "nvidia.com/gpu=all", "--ipc=host", "--network", "none",
               "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
               "--env", "VLLM_EXL3_MULTIPRECISION=0",
               "--env", "VLLM_EXL3_EMBED_ONLINE_BITS=6",
               "--env", "VLLM_USE_V2_MODEL_RUNNER=1",
               "--env", "VLLM_ALLOW_INSECURE_SERIALIZATION=1"]
    command += mounts([(REPO_ROOT, False), (candidate, False),
                       (suite_view, False), (capture, True)])
    for environment in args.container_env:
        if "=" not in environment:
            raise CaptureError("additional container environment must be NAME=VALUE")
        command += ["--env", environment]
    for volume in args.container_volume:
        if ((not volume.endswith(":ro") and not volume.endswith(":rw"))
                or volume.count(":") < 2):
            raise CaptureError("additional container volumes must be SOURCE:DEST:ro|rw")
        source = Path(volume.split(":", 1)[0])
        if not source.exists():
            raise CaptureError(f"missing container runtime mount source: {source}")
        command += ["--volume", volume]
    command += ["--entrypoint", "python3", args.container_image, str(fidelity), "capture",
                "--model", str(candidate), "--suite", str(suite_view), "--out", str(capture),
                "--quantization", "auto", "--kv-cache-dtype", "bfloat16",
                "--attention-backend", "TRITON_ATTN", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--filter", "all", "--max-batched-tokens", "2048", "--hash-shards"]
    started = time.time(); status, error = "fail", None
    try: run_checked(command); status = "pass"
    except Exception as exc: error = str(exc); raise
    finally:
        base_receipt = {"schema": EXECUTION_SCHEMA, "status": status, "error": error,
            "command": command, "elapsed_sec": time.time() - started, "container_image": image,
            "gpu": gpu, "python_host": sys.executable, "host_python_version": platform.python_version(),
            "platform": platform.platform(), "repository_head_fidelity_path": str(fidelity),
            "repository_head_fidelity_sha256": sha256_file(fidelity),
            "checkpoint_identity_sha256": checkpoint["candidate_checkpoint_identity_sha256"],
            "disk_free_before_capture_bytes": disk_before_capture}
        if status == "fail": write_json(capture / "capture-execution-failed.json", base_receipt)
    disk_after_capture = free_bytes(capture)
    if disk_after_capture < MIN_FREE_BYTES:
        raise CaptureError("candidate capture crossed the 60 GiB free-space floor")
    manifest_path = require_file("capture manifest", capture / "capture-manifest.json"); manifest = load_json(manifest_path)
    if (manifest.get("complete") is not True or manifest.get("suite_token_sha256") != inputs["suite"]["suite_token_sha256"]
            or set(manifest.get("expected_indices", [])) != set(range(512)) or manifest.get("filter") != "all"):
        raise CaptureError("capture suite/selector/completeness mismatch")
    identity = manifest.get("candidate_identity", {}); expected = checkpoint["candidate_checkpoint_identity"]
    for key in ("index_sha256", "config_sha256", "shard_sha256"):
        if identity.get(key) != expected[key]: raise CaptureError(f"capture checkpoint mismatch: {key}")
    projection = {key: identity.get(key) for key in ("model_revision", "index_sha256", "config_sha256", "shard_sha256")}
    identity_sha = sha256_bytes(canonical_json(projection))
    if identity_sha != checkpoint["candidate_checkpoint_identity_sha256"]: raise CaptureError("capture identity digest mismatch")
    execution = {**base_receipt, "status": "pass", "error": None,
        "capture_execution": inputs["contract"]["capture_execution_contract"],
        "capture_manifest_sha256": sha256_file(manifest_path),
        "suite_manifest_sha256": sha256_file(inputs["suite_dir"] / "suite-manifest.json"),
        "suite_token_sha256": inputs["suite"]["suite_token_sha256"],
        "r29_validation_selector": inputs["validation"]["selector"],
        "r29_validation_selection_sha256": inputs["validation"]["selection_sha256"],
        "shared_head_sha256": inputs["contract"]["reference_semantics"]["shared_head_sha256"],
        "action_identity_sha256": checkpoint["action_identity_sha256"],
        "action_payload_sha256": checkpoint["action_payload_sha256"],
        "codec_route_id": checkpoint["codec_route_id"],
        "canonical_artifacts": inputs["canonical_artifacts"],
        "disk_free_after_capture_bytes": disk_after_capture}
    execution_path = capture / "capture-execution.json"; write_json(execution_path, execution)
    binding = {"schema": BINDING_SCHEMA, "role": "candidate", "checkpoint_identity_sha256": identity_sha,
        "capture_manifest_sha256": sha256_file(manifest_path),
        "capture_execution": inputs["contract"]["capture_execution_contract"],
        "capture_execution_receipt": {"path": str(execution_path), "sha256": sha256_file(execution_path)},
        "action_identity_sha256": checkpoint["action_identity_sha256"],
        "action_file_sha256": checkpoint["action_file_sha256"],
        "action_payload_sha256": checkpoint["action_payload_sha256"],
        "logical_tensor_key": checkpoint["logical_tensor_key"],
        "changed_shard_sha256": checkpoint["candidate_changed_shard_sha256"],
        "r29_validation_selection_sha256": inputs["validation"]["selection_sha256"],
        "suite_manifest_sha256": sha256_file(inputs["suite_dir"] / "suite-manifest.json"),
        "suite_token_sha256": inputs["suite"]["suite_token_sha256"],
        "shared_head_sha256": inputs["contract"]["reference_semantics"]["shared_head_sha256"],
        "codec_route_id": checkpoint["codec_route_id"],
        "canonical_artifacts": inputs["canonical_artifacts"]}
    binding_path = capture / "wave5-capture-binding.json"; write_json(binding_path, binding)
    return {"capture_dir": str(capture), "capture_manifest_sha256": sha256_file(manifest_path),
            "capture_binding_sha256": sha256_file(binding_path),
            "capture_execution_sha256": sha256_file(execution_path), "checkpoint_identity_sha256": identity_sha}


def prepare_reference_view(reference: Path, destination: Path) -> Path:
    """Hardlink resolved Hub snapshot blobs so container-visible links cannot escape."""
    if destination.exists():
        shutil.rmtree(destination)
    for visible in checkpoint_files(reference):
        link_or_clone(visible, destination / visible.relative_to(reference))
    manifest = (destination / "shard-0000/capture-manifest.json"
                if (destination / "shard-0000").is_dir()
                else destination / "capture-manifest.json")
    require_file("reference view manifest", manifest)
    return destination

def prepare_replay_suite_view(inputs: dict[str, Any], destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    link_or_clone(inputs["suite_dir"] / "suite-manifest.json",
                  destination / "shard-0000/suite-manifest.json")
    return destination



def replay_candidate(args: argparse.Namespace, checkpoint: dict[str, Any], inputs: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    validate_action_lineage(checkpoint, inputs)
    evidence = args.evidence_dir.resolve()
    runtime = link_r31_runtime(inputs, evidence / "r31-runtime")
    suite_view = prepare_replay_suite_view(
        inputs, evidence / "r31-suite-view")
    reference_view = prepare_reference_view(
        inputs["reference"], evidence / "reference-view")
    rows, report = evidence / "r31-validation-rows.npz", evidence / "r31-validation-report.json"
    command = ["podman", "run", "--rm", "--name", f"wave5-replay-{os.getpid()}",
               "--device", "nvidia.com/gpu=all", "--ipc=host", "--network", "none"]
    command += mounts([(runtime, False), (reference_view, False), (Path(capture["capture_dir"]), False),
                       (suite_view, False), (inputs["head"], False), (evidence, True)])
    gate = runtime / "tools/research/wave5/fidelity_gate.py"
    command += ["--entrypoint", "python3", args.container_image, str(gate), "replay",
        "--reference-root", str(reference_view), "--candidate-root", capture["capture_dir"],
        "--suite-root", str(suite_view), "--head", str(inputs["head"]),
        "--role", "candidate", "--split", "validation", "--candidate-id", args.candidate_id,
        "--expected-checkpoint-identity-sha256", checkpoint["candidate_checkpoint_identity_sha256"],
        "--rows-out", str(rows), "--output", str(report), "--device", "cuda"]
    started = time.time(); run_checked(command); elapsed = time.time() - started; value = load_json(report)
    if (value.get("candidate_checkpoint_sha256") != checkpoint["candidate_checkpoint_identity_sha256"]
            or value.get("split") != "validation" or value.get("contexts") != 512
            or value.get("scored_positions") != 512 * 2047): raise CaptureError("R31 output geometry/identity mismatch")
    receipt = {"schema": "qwen38-wave5-r31-replay-execution/1", "status": "pass",
        "command": command, "elapsed_sec": elapsed, "r31_gate_sha256": sha256_file(inputs["paths"]["gate"]),
        "r31_contract_sha256": sha256_file(inputs["paths"]["contract"]),
        "r31_prereg_sha256": sha256_file(inputs["paths"]["prereg"]),
        "canonical_reference_root": str(inputs["reference"]),
        "reference_view": str(reference_view),
        "canonical_suite_root": str(inputs["suite_root"]),
        "suite_view": str(suite_view),
        "report": {"path": str(report), "sha256": sha256_file(report)},
        "rows": {"path": str(rows), "sha256": sha256_file(rows)}}
    receipt_path = evidence / "r31-replay-execution.json"; write_json(receipt_path, receipt)
    return {"report": str(report), "report_sha256": sha256_file(report), "rows": str(rows),
            "rows_sha256": sha256_file(rows), "execution_receipt": str(receipt_path),
            "execution_receipt_sha256": sha256_file(receipt_path), "metrics": value.get("metrics")}


def stop_service() -> None:
    run_checked(["systemctl", "--user", "stop", "qwen38-27b.service"])


def restore_service(timeout: int = 1800) -> dict[str, Any]:
    run_checked(["systemctl", "--user", "unset-environment", "PROFILE"])
    started = time.time(); run_checked(["systemctl", "--user", "start", "qwen38-27b.service"])
    last_error = ""
    while time.time() < started + timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                active = run_checked(["systemctl", "--user", "is-active", "qwen38-27b.service"], capture_output=True).stdout.strip()
                if 200 <= response.status < 300 and active == "active":
                    return {"profile": "throughput (systemd default; PROFILE unset)",
                            "health_http_status": response.status, "systemd_active": True,
                            "start_to_health_seconds": time.time() - started}
        except Exception as exc: last_error = str(exc)
        time.sleep(2)
    raise CaptureError(f"throughput service unhealthy: {last_error}")


def cleanup_checkpoint(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(require_file("checkpoint manifest", manifest_path))
    candidate = Path(manifest.get("candidate_checkpoint", "")).resolve()
    source = Path(manifest.get("source_checkpoint", "")).resolve()
    preserved = Path(manifest.get("preserved_changed_shard", {}).get("path", "")).resolve()
    if (manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("complete") is not True
            or candidate == source or not candidate.is_dir() or not preserved.is_file()
            or sha256_file(preserved) != manifest["candidate_changed_shard_sha256"]):
        raise CaptureError("unsafe cleanup or missing changed-shard evidence")
    shutil.rmtree(candidate)
    result = {"candidate_checkpoint_removed": str(candidate), "preserved_changed_shard": str(preserved),
              "preserved_changed_shard_sha256": sha256_file(preserved), "free_bytes": shutil.disk_usage(preserved).free}
    write_json(manifest_path.parent / "cleanup-receipt.json", result)
    if result["free_bytes"] < MIN_FREE_BYTES: raise CaptureError("less than 60 GiB free after cleanup")
    return result


def load_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    value = load_json(require_file("checkpoint manifest", args.evidence_dir.resolve() / "checkpoint-manifest.json"))
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("complete") is not True: raise CaptureError("incomplete checkpoint evidence")
    return value


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = validation_inputs(args)
    checkpoint = materialize(args)
    try:
        validate_action_lineage(checkpoint, inputs)
    except Exception:
        cleanup_checkpoint(args.evidence_dir.resolve() / "checkpoint-manifest.json")
        raise
    stopped = False
    try:
        if args.manage_service:
            stop_service()
            stopped = True
        capture = capture_candidate(args, checkpoint, inputs)
        replay = replay_candidate(args, checkpoint, inputs, capture)
        result = {"schema": RESULT_SCHEMA, "status": "pass", "candidate_id": args.candidate_id,
                  "checkpoint": checkpoint, "capture": capture, "r31_validation_replay": replay,
                  "strength_zero_claim": "requires paired R31 stock-control rows; never inferred from strength"}
        write_json(args.evidence_dir.resolve() / "result.json", result)
        if args.cleanup_checkpoint:
            result["cleanup"] = cleanup_checkpoint(args.evidence_dir.resolve() / "checkpoint-manifest.json")
            write_json(args.evidence_dir.resolve() / "result.json", result)
        return result
    finally:
        if args.manage_service and stopped:
            write_json(args.evidence_dir.resolve() / "service-restoration.json", restore_service())


def cmd_self_test(_: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="candidate-capture-") as tmp:
        root = Path(tmp); source, candidate = root / "source.safetensors", root / "candidate.safetensors"
        header = {
            "module.trellis": {
                "dtype": "I16", "shape": [1, 1, 4], "data_offsets": [12, 20]},
            "module.suh": {
                "dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
            "module.svh": {
                "dtype": "F16", "shape": [2], "data_offsets": [8, 12]},
        }
        raw_header = encoded_header(header); source.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + bytes(range(20)))
        replacement = bytes(reversed(range(8))); row = {"safetensors_dtype": "I16", "shape": [1, 1, 4]}
        rewrite_safetensors(source, candidate, {"module.trellis": (row, replacement)})
        left_header, left_data = parse_safetensors(source); right_header, right_data = parse_safetensors(candidate)
        with source.open("rb") as left, candidate.open("rb") as right:
            for name in ("module.suh", "module.svh"):
                a, b = left_header[name]["data_offsets"]; c, d = right_header[name]["data_offsets"]
                left.seek(left_data + a); right.seek(right_data + c)
                if left.read(b-a) != right.read(d-c): raise CaptureError("untouched buffer changed")
        try: reject_legacy("self-test", LEGACY_ROOT / "captures")
        except CaptureError: pass
        else: raise CaptureError("legacy path accepted")
    return {"status": "pass", "one_replacement": True, "untouched_buffers_identical": True, "legacy_rejected": True}


def add_materialize(parser: argparse.ArgumentParser) -> None:
    for name in ("base-checkpoint", "action", "payload", "candidate-checkpoint", "evidence-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--action-harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--expected-action-harness-sha256",
                        default=QUALIFIED_OPERATIONAL_HARNESS_SHA256)
    parser.add_argument("--operational-extension-sha256",
                        default=QUALIFIED_OPERATIONAL_EXTENSION_SHA256)


def add_protocol(parser: argparse.ArgumentParser) -> None:
    for name in ("suite-root", "reference-root", "shared-head", "r29-split-manifest", "r29-data-manifest", "r31-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-volume", action="append", default=[],
                        help="additional read-only runtime mount, SOURCE:DEST:ro")
    parser.add_argument("--container-env", action="append", default=[],
                        help="additional recorded runtime environment NAME=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("materialize"); add_materialize(p); p.set_defaults(func=materialize)
    p = sub.add_parser("run"); add_materialize(p); add_protocol(p)
    p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--candidate-id", required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--manage-service", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cleanup-checkpoint", action="store_true"); p.set_defaults(func=cmd_run)
    p = sub.add_parser("capture"); p.add_argument("--evidence-dir", type=Path, required=True); add_protocol(p)
    p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.set_defaults(func=lambda args: capture_candidate(args, load_checkpoint(args), validation_inputs(args)))
    p = sub.add_parser("replay"); p.add_argument("--evidence-dir", type=Path, required=True); add_protocol(p)
    p.add_argument("--capture-dir", type=Path, required=True); p.add_argument("--candidate-id", required=True)
    p.set_defaults(func=lambda args: replay_candidate(args, load_checkpoint(args), validation_inputs(args),
                                                      {"capture_dir": str(require_dir("capture", args.capture_dir))}))
    p = sub.add_parser("cleanup"); p.add_argument("--evidence-dir", type=Path, required=True)
    p.set_defaults(func=lambda args: cleanup_checkpoint(args.evidence_dir.resolve() / "checkpoint-manifest.json"))
    p = sub.add_parser("self-test"); p.set_defaults(func=cmd_self_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try: result = args.func(args)
    except CaptureError as exc:
        print(f"candidate_capture: FAIL: {exc}", file=sys.stderr); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
