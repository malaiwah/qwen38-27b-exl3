#!/usr/bin/env python3
"""Plan, execute, and verify private HF Bucket preservation without deletion."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json

SCHEMA = "qwen38-frontier-volatile-preservation/1"
DESTINATION_RE = re.compile(
    r"^hf://buckets/(?P<namespace>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<bucket>[A-Za-z0-9][A-Za-z0-9_.-]*)(?:/(?P<prefix>[^?#]*))?$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATES = {"planned", "preserved"}
RECEIPT_KEYS = {
    "schema", "state", "source", "local_copy_evidence", "destination", "transfer",
    "verification", "preservation", "prior_receipt_sha256",
}


class PreservationError(ValueError):
    """A fail-closed preservation error."""


def fail(message: str) -> NoReturn:
    raise PreservationError(message)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def obj(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a JSON array")
    return cast(list[Any], value)


def exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        fail(f"{label} is missing fields: {missing}")
    if unknown:
        fail(f"{label} has unknown fields: {unknown}")


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown", "unmeasured", "n/a", "none", "null", "tbd"
    }:
        fail(f"{label} must be a known non-empty string")
    return value.strip()


def integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be boolean")
    return value


def strict_json_bytes(data: bytes, label: str) -> object:
    def duplicate_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=duplicate_guard,
            parse_constant=lambda token: fail(f"{label} contains non-finite number {token}"),
        )
    except UnicodeDecodeError as exc:
        fail(f"{label} is not UTF-8: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"{label} is not strict JSON: {exc}")
    raise AssertionError("unreachable")


def parse_destination(value: str) -> dict[str, str]:
    destination = text(value, "destination")
    match = DESTINATION_RE.fullmatch(destination)
    if match is None or "@" in destination:
        fail("destination must be a credential-free hf://buckets/<namespace>/<bucket>[/prefix] URI")
    prefix = (match.group("prefix") or "").strip("/")
    parts = prefix.split("/") if prefix else []
    if prefix:
        safe_relative_name(prefix, "destination prefix")
    if any(part in {"", ".", ".."} for part in parts) or "//" in prefix:
        fail("destination prefix must be a normalized nonempty-component relative path")
    namespace = match.group("namespace")
    bucket = match.group("bucket")
    if namespace is None or bucket is None:
        fail("destination must contain a namespace and bucket")
    bucket_id = f"{namespace}/{bucket}"
    uri = f"hf://buckets/{bucket_id}" + (f"/{prefix}" if prefix else "")
    if destination != uri:
        fail("destination URI must already be canonical (no leading or trailing prefix slash)")
    return {"uri": uri, "bucket_id": bucket_id, "prefix": prefix}


def safe_symlink_target(target: str, link_path: str, label: str) -> str:
    try:
        target.encode("utf-8", "strict")
    except UnicodeError as exc:
        fail(f"{label} is not canonical UTF-8: {exc}")
    if (
        not target
        or target.startswith("/")
        or "\\" in target
        or any(ord(char) < 32 for char in target)
        or any(part == "" for part in target.split("/"))
    ):
        fail(f"{label} must be a relative UTF-8 target without empty components")
    stack = link_path.split("/")[:-1]
    for part in target.split("/"):
        if part == ".":
            continue
        if part == "..":
            if not stack:
                fail(f"{label} escapes the preservation source tree")
            stack.pop()
        else:
            stack.append(part)
    return target


def safe_relative_name(name: str, label: str) -> str:
    try:
        encoded = name.encode("utf-8", "strict")
        if encoded.decode("utf-8") != name:
            fail(f"{label} is not canonical UTF-8")
    except UnicodeError as exc:
        fail(f"{label} is not canonical UTF-8: {exc}")
    if not name or name.startswith("/") or "\\" in name or any(ord(char) < 32 for char in name):
        fail(f"{label} is not a portable relative POSIX name")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        fail(f"{label} is not normalized")
    return name


def is_tmp_source(path: Path) -> bool:
    resolved = path.resolve(strict=True)
    temporary_roots = (Path("/tmp"), Path("/var/tmp"), Path("/private/tmp"), Path("/private/var/tmp"))
    return any(resolved == root or root in resolved.parents for root in temporary_roots)


def readonly_filesystem(path: Path) -> bool:
    return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))


def checked_file_hash(path: Path, before: os.stat_result, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"cannot securely open {label}: {exc}")
    if descriptor is None:
        fail(f"cannot securely open {label}")
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            fail(f"{label} changed identity while it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        ):
            fail(f"{label} changed while it was hashed")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def inventory_tree(root: Path) -> tuple[dict[str, Any], int]:
    root_stat: os.stat_result | None = None
    try:
        root = root.resolve(strict=True)
        root_stat = root.lstat()
    except OSError as exc:
        fail(f"cannot resolve source directory: {exc}")
    if root_stat is None or not stat.S_ISDIR(root_stat.st_mode):
        fail("source must be a directory, not a file or symlink")
    filesystem_readonly = readonly_filesystem(root)
    device = root_stat.st_dev
    entries: list[dict[str, Any]] = []
    directories_before: dict[Path, tuple[int, int, int, int]] = {}

    def walk(directory: Path, relative: str) -> None:
        children: list[os.DirEntry[str]] = []
        try:
            before = directory.lstat()
            if before.st_dev != device:
                fail("a preservation source may not span filesystems; use one independently inventoried source per filesystem")
            if not filesystem_readonly and before.st_mode & 0o222:
                fail("source is mutable: every containing directory must be read-only")
            directories_before[directory] = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            children = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            fail(f"cannot inventory source directory: {exc}")
        seen_names: set[str] = set()
        for child in children:
            name = safe_relative_name(child.name, "source entry name")
            if name in seen_names:
                fail(f"duplicate source entry name {name!r}")
            seen_names.add(name)
            rel = f"{relative}/{name}" if relative else name
            safe_relative_name(rel, "source relative path")
            path = directory / child.name
            before_child: os.stat_result | None = None
            try:
                before_child = path.lstat()
            except OSError as exc:
                fail(f"cannot stat source entry {rel!r}: {exc}")
            if before_child is None:
                fail(f"cannot stat source entry {rel!r}")
            if before_child.st_dev != device:
                fail("a preservation source may not span filesystems")
            if stat.S_ISDIR(before_child.st_mode):
                walk(path, rel)
            elif stat.S_ISREG(before_child.st_mode):
                if not filesystem_readonly and before_child.st_mode & 0o222:
                    fail(f"source is mutable: regular file {rel!r} has a write bit")
                digest = checked_file_hash(path, before_child, f"source entry {rel!r}")
                entries.append({
                    "path": rel,
                    "type": "file",
                    "size_bytes": before_child.st_size,
                    "sha256": digest,
                    "mode": stat.S_IMODE(before_child.st_mode),
                })
            elif stat.S_ISLNK(before_child.st_mode):
                target: str | None = None
                try:
                    target = os.readlink(path)
                except OSError as exc:
                    fail(f"cannot read symlink {rel!r}: {exc}")
                if target is None:
                    fail(f"cannot read symlink {rel!r}")
                target = safe_symlink_target(target, rel, f"symlink target for {rel!r}")
                target_bytes = target.encode("utf-8")
                entries.append({
                    "path": rel,
                    "type": "symlink",
                    "size_bytes": len(target_bytes),
                    "sha256": hashlib.sha256(target_bytes).hexdigest(),
                    "link_target": target,
                })
            else:
                fail(f"source contains unsupported special entry {rel!r}")

    walk(root, "")
    if not entries:
        fail("source artifact is empty; nothing can be preserved")
    entries.sort(key=lambda item: item["path"].encode("utf-8"))
    for directory, signature in directories_before.items():
        after = directory.lstat()
        if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature:
            fail("source directory changed while it was inventoried")
    source = {
        "tree_sha256": canonical_sha256(entries),
        "entry_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "entries": entries,
        "immutable": True,
        "immutability_basis": "read-only filesystem or no write bits on every directory and regular file; stable double inventory",
    }
    return source, device


def stable_inventory(root: Path) -> tuple[dict[str, Any], int]:
    first, first_device = inventory_tree(root)
    second, second_device = inventory_tree(root)
    if first != second or first_device != second_device:
        fail("source changed between independent inventories")
    return first, first_device


def validate_source_record(value: object, label: str = "source") -> dict[str, Any]:
    source = obj(value, label)
    exact_keys(source, {"tree_sha256", "entry_count", "total_bytes", "entries", "immutable", "immutability_basis"}, label)
    digest = text(source["tree_sha256"], f"{label}.tree_sha256")
    if not SHA256_RE.fullmatch(digest):
        fail(f"{label}.tree_sha256 must be lowercase SHA256")
    if boolean(source["immutable"], f"{label}.immutable") is not True:
        fail(f"{label} must be immutable")
    entries = array(source["entries"], f"{label}.entries")
    if integer(source["entry_count"], f"{label}.entry_count", 1) != len(entries):
        fail(f"{label}.entry_count does not match entries")
    total = 0
    prior: bytes | None = None
    for index, raw_entry in enumerate(entries):
        entry_label = f"{label}.entries[{index}]"
        entry = obj(raw_entry, entry_label)
        entry_type = text(entry.get("type"), f"{entry_label}.type")
        required = {"path", "type", "size_bytes", "sha256", "mode"} if entry_type == "file" else {"path", "type", "size_bytes", "sha256", "link_target"}
        if entry_type not in {"file", "symlink"}:
            fail(f"{entry_label}.type is unsupported")
        exact_keys(entry, required, entry_label)
        path = safe_relative_name(text(entry["path"], f"{entry_label}.path"), f"{entry_label}.path")
        encoded = path.encode("utf-8")
        if prior is not None and encoded <= prior:
            fail(f"{label}.entries must be uniquely sorted by UTF-8 path bytes")
        prior = encoded
        size = integer(entry["size_bytes"], f"{entry_label}.size_bytes")
        digest_value = text(entry["sha256"], f"{entry_label}.sha256")
        if not SHA256_RE.fullmatch(digest_value):
            fail(f"{entry_label}.sha256 must be lowercase SHA256")
        if entry_type == "file":
            integer(entry["mode"], f"{entry_label}.mode")
        else:
            target = safe_symlink_target(
                text(entry["link_target"], f"{entry_label}.link_target"),
                path,
                f"{entry_label}.link_target",
            )
            target_bytes = target.encode("utf-8")
            if size != len(target_bytes) or digest_value != hashlib.sha256(target_bytes).hexdigest():
                fail(f"{entry_label} symlink byte identity is inconsistent")
        total += size
    if integer(source["total_bytes"], f"{label}.total_bytes") != total:
        fail(f"{label}.total_bytes does not match entries")
    if canonical_sha256(entries) != digest:
        fail(f"{label}.tree_sha256 does not match canonical entries")
    text(source["immutability_basis"], f"{label}.immutability_basis")
    return source


def validate_timestamp(value: object, label: str) -> str:
    timestamp = text(value, label)
    if not timestamp.endswith("Z"):
        fail(f"{label} must be an explicit UTC timestamp ending in Z")
    try:
        dt.datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        fail(f"{label} is not an ISO-8601 timestamp: {exc}")
    return timestamp


def validate_transfer_files(
    value: object,
    source: dict[str, Any],
    label: str,
    *,
    readback: bool,
) -> None:
    rows = array(value, label)
    entries = source["entries"]
    if len(rows) != len(entries):
        fail(f"{label} does not cover every source entry")
    for index, (raw_row, entry) in enumerate(zip(rows, entries)):
        row_label = f"{label}[{index}]"
        row = obj(raw_row, row_label)
        required_keys = {
            "path", "size_bytes", "sha256", "started_utc", "finished_utc", "elapsed_ns"
        }
        if readback:
            required_keys.add("method")
        exact_keys(row, required_keys, row_label)
        for key in ("path", "size_bytes", "sha256"):
            if row[key] != entry[key]:
                fail(f"{row_label}.{key} does not match source inventory")
        validate_timestamp(row["started_utc"], f"{row_label}.started_utc")
        validate_timestamp(row["finished_utc"], f"{row_label}.finished_utc")
        integer(row["elapsed_ns"], f"{row_label}.elapsed_ns")
        if readback:
            text(row["method"], f"{row_label}.method")


def validate_receipt(raw: object) -> dict[str, Any]:
    receipt = obj(raw, "receipt")
    exact_keys(receipt, RECEIPT_KEYS, "receipt")
    if receipt["schema"] != SCHEMA:
        fail(f"receipt.schema must be {SCHEMA!r}")
    state = text(receipt["state"], "receipt.state")
    if state not in STATES:
        fail(f"receipt.state must be one of {sorted(STATES)}")
    source = validate_source_record(receipt["source"], "receipt.source")

    local = obj(receipt["local_copy_evidence"], "receipt.local_copy_evidence")
    exact_keys(
        local,
        {
            "required_local_copy_count", "verified_local_copy_count",
            "distinct_filesystem_count", "temporary_source",
        },
        "receipt.local_copy_evidence",
    )
    required = integer(
        local["required_local_copy_count"],
        "receipt.local_copy_evidence.required_local_copy_count",
        1,
    )
    verified = integer(
        local["verified_local_copy_count"],
        "receipt.local_copy_evidence.verified_local_copy_count",
        1,
    )
    domains = integer(
        local["distinct_filesystem_count"],
        "receipt.local_copy_evidence.distinct_filesystem_count",
        1,
    )
    boolean(local["temporary_source"], "receipt.local_copy_evidence.temporary_source")
    if verified < required or domains != verified:
        fail("receipt local copy evidence is incomplete or conflates filesystems")

    destination = obj(receipt["destination"], "receipt.destination")
    exact_keys(destination, {"uri", "bucket_id", "prefix", "bucket_identity"}, "receipt.destination")
    parsed_destination = parse_destination(text(destination["uri"], "receipt.destination.uri"))
    if {key: destination[key] for key in ("uri", "bucket_id", "prefix")} != parsed_destination:
        fail("receipt destination fields are not canonical")

    preservation = obj(receipt["preservation"], "receipt.preservation")
    exact_keys(
        preservation,
        {
            "source_tree_sha256", "bucket_tree_sha256", "source_and_bucket_match",
            "independent_verified_copy_count", "preserved",
        },
        "receipt.preservation",
    )
    source_hash = text(preservation["source_tree_sha256"], "receipt.preservation.source_tree_sha256")
    if source_hash != source["tree_sha256"]:
        fail("receipt preservation source hash does not match source inventory")
    preserved = boolean(preservation["preserved"], "receipt.preservation.preserved")
    matches = boolean(
        preservation["source_and_bucket_match"],
        "receipt.preservation.source_and_bucket_match",
    )
    copy_count = integer(
        preservation["independent_verified_copy_count"],
        "receipt.preservation.independent_verified_copy_count",
        1,
    )

    if state == "planned":
        if (
            preserved
            or matches
            or preservation["bucket_tree_sha256"] is not None
            or copy_count != verified
            or destination["bucket_identity"] is not None
            or receipt["transfer"] is not None
            or receipt["verification"] is not None
            or receipt["prior_receipt_sha256"] is not None
        ):
            fail("planned receipt contains preservation, transfer, bucket, or readback claims")
        return receipt

    if not preserved or not matches or copy_count != verified + 1:
        fail("preserved receipt lacks the independently verified source and bucket copy count")
    bucket_hash = text(
        preservation["bucket_tree_sha256"],
        "receipt.preservation.bucket_tree_sha256",
    )
    if bucket_hash != source_hash:
        fail("preserved receipt source and bucket tree hashes differ")
    bucket = obj(destination["bucket_identity"], "receipt.destination.bucket_identity")
    exact_keys(bucket, {"id", "private", "created_at"}, "receipt.destination.bucket_identity")
    if (
        text(bucket["id"], "receipt.destination.bucket_identity.id") != destination["bucket_id"]
        or boolean(bucket["private"], "receipt.destination.bucket_identity.private") is not True
    ):
        fail("preserved receipt does not identify the exact private destination bucket")
    validate_timestamp(bucket["created_at"], "receipt.destination.bucket_identity.created_at")

    transfer = obj(receipt["transfer"], "receipt.transfer")
    if "files" in transfer:
        exact_keys(
            transfer,
            {"started_utc", "finished_utc", "elapsed_ns", "bytes", "files", "method"},
            "receipt.transfer",
        )
        validate_timestamp(transfer["started_utc"], "receipt.transfer.started_utc")
        validate_timestamp(transfer["finished_utc"], "receipt.transfer.finished_utc")
        integer(transfer["elapsed_ns"], "receipt.transfer.elapsed_ns")
        if integer(transfer["bytes"], "receipt.transfer.bytes") != source["total_bytes"]:
            fail("receipt.transfer.bytes does not match source inventory")
        text(transfer["method"], "receipt.transfer.method")
        validate_transfer_files(transfer["files"], source, "receipt.transfer.files", readback=False)
    else:
        exact_keys(
            transfer,
            {"prior_receipt_sha256", "reused_verified_upload"},
            "receipt.transfer",
        )
        transfer_prior = text(
            transfer["prior_receipt_sha256"],
            "receipt.transfer.prior_receipt_sha256",
        )
        if not SHA256_RE.fullmatch(transfer_prior) or transfer["reused_verified_upload"] is not True:
            fail("receipt.transfer does not identify a verified prior upload")

    verification = obj(receipt["verification"], "receipt.verification")
    exact_keys(
        verification,
        {
            "started_utc", "finished_utc", "elapsed_ns", "bytes", "files",
            "object_set_exact", "bucket_identity_stable",
        },
        "receipt.verification",
    )
    validate_timestamp(verification["started_utc"], "receipt.verification.started_utc")
    validate_timestamp(verification["finished_utc"], "receipt.verification.finished_utc")
    integer(verification["elapsed_ns"], "receipt.verification.elapsed_ns")
    if integer(verification["bytes"], "receipt.verification.bytes") != source["total_bytes"]:
        fail("receipt.verification.bytes does not match source inventory")
    if verification["object_set_exact"] is not True or verification["bucket_identity_stable"] is not True:
        fail("receipt verification does not establish exact objects and stable bucket identity")
    validate_transfer_files(
        verification["files"],
        source,
        "receipt.verification.files",
        readback=True,
    )

    prior = text(receipt["prior_receipt_sha256"], "receipt.prior_receipt_sha256")
    if not SHA256_RE.fullmatch(prior):
        fail("receipt.prior_receipt_sha256 must be lowercase SHA256")
    return receipt


def output_path(value: str, source: Path | None = None) -> Path:
    path = Path(value)
    if path.name != "volatile-preservation.json":
        fail("--output basename must be volatile-preservation.json")
    if path.exists() and path.stat().st_size:
        fail(f"refusing to overwrite nonempty output artifact {path}")
    if path.is_dir():
        fail(f"output path is a directory: {path}")
    if source is not None:
        try:
            path.resolve().relative_to(source.resolve(strict=True))
        except ValueError:
            pass
        else:
            fail("output must not be inside the immutable source tree")
    return path


def load_receipt(path: Path) -> tuple[dict[str, Any], object]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"input artifact is missing, non-file, or empty: {path}")
    raw = load_strict_json(path)
    return validate_receipt(raw), raw


def local_evidence(source: Path, second_source: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory, source_device = stable_inventory(source)
    temporary = is_tmp_source(source)
    verified = 1
    domains = {source_device}
    if second_source is not None:
        second_inventory, second_device = stable_inventory(second_source)
        if second_inventory != inventory:
            fail("second source inventory does not exactly match the primary source identity and hashes")
        if second_device in domains:
            fail("primary and second source are on the same filesystem and count as one copy (including /home and /tmp)")
        domains.add(second_device)
        verified += 1
    evidence = {
        "required_local_copy_count": 1,
        "verified_local_copy_count": verified,
        "distinct_filesystem_count": len(domains),
        "temporary_source": temporary,
    }
    return inventory, evidence


def assert_expected_source(source_path: Path, second_path: Path | None, receipt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory, evidence = local_evidence(source_path, second_path)
    if inventory != receipt["source"]:
        fail("live source identity or hashes do not match the input receipt")
    expected = receipt["local_copy_evidence"]
    if evidence["verified_local_copy_count"] < expected["required_local_copy_count"]:
        fail("required independent local source copy is missing")
    if evidence["temporary_source"] != expected["temporary_source"]:
        fail("source volatility class changed from the plan")
    return inventory, evidence


def hf_executable() -> str:
    executable = shutil.which("hf")
    if executable is None:
        fail("installed 'hf' CLI is required for execute and verify")
    return executable


def run_captured(argv: list[str], label: str) -> bytes:
    completed: subprocess.CompletedProcess[bytes] | None = None
    try:
        completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        fail(f"cannot run {label}: {exc}")
    if completed is None:
        fail(f"cannot run {label}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else "no diagnostic"
        fail(f"{label} failed with exit {completed.returncode}: {tail[:500]}")
    return completed.stdout


def bucket_identity(hf: str, bucket_id: str) -> dict[str, Any]:
    raw = strict_json_bytes(run_captured([hf, "buckets", "info", bucket_id], "hf buckets info"), "hf buckets info output")
    info = obj(raw, "hf buckets info output")
    returned_id = text(info.get("id"), "bucket info id")
    if returned_id != bucket_id:
        fail(f"bucket identity mismatch: requested {bucket_id!r}, server returned {returned_id!r}")
    if info.get("private") is not True:
        fail("destination bucket must exist and be private")
    created_at = text(info.get("created_at"), "bucket info created_at")
    parsed_created_at: dt.datetime | None = None
    try:
        parsed_created_at = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        fail(f"bucket info created_at is not ISO-8601: {exc}")
    if parsed_created_at is None or parsed_created_at.tzinfo is None:
        fail("bucket info created_at lacks an explicit timezone")
    canonical_created_at = (
        parsed_created_at.astimezone(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    return {"id": returned_id, "private": True, "created_at": canonical_created_at}


def remote_uri(destination: dict[str, Any], relative: str) -> str:
    safe_relative_name(relative, "remote relative path")
    return f"{destination['uri']}/{relative}"


def expected_remote_paths(destination: dict[str, Any], source: dict[str, Any]) -> list[str]:
    prefix = destination["prefix"]
    return [f"{prefix}/{entry['path']}" if prefix else entry["path"] for entry in source["entries"]]


def list_remote(hf: str, destination: dict[str, Any]) -> list[str]:
    stdout = run_captured([hf, "buckets", "list", destination["uri"], "-R", "-q"], "hf buckets list")
    lines: list[str] = []
    try:
        lines = stdout.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError as exc:
        fail(f"hf buckets list output is not UTF-8: {exc}")
    result: list[str] = []
    for index, raw in enumerate(lines):
        name = raw.strip()
        if not name or name.endswith("/"):
            continue
        result.append(safe_relative_name(name, f"remote listing line {index + 1}"))
    if len(result) != len(set(result)):
        fail("remote listing contains duplicate paths")
    return sorted(result, key=lambda value: value.encode("utf-8"))


def timed_upload(hf: str, source_root: Path, destination: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    target = remote_uri(destination, entry["path"])
    started_utc = now_utc()
    started = time.monotonic_ns()
    completed: subprocess.CompletedProcess[Any] | None = None
    try:
        if entry["type"] == "file":
            completed = subprocess.run([hf, "buckets", "cp", str(source_root / entry["path"]), target], check=False)
        else:
            completed = subprocess.run(
                [hf, "buckets", "cp", "-", target],
                input=entry["link_target"].encode("utf-8"),
                check=False,
            )
    except OSError as exc:
        fail(f"cannot upload {entry['path']!r}: {exc}")
    elapsed = time.monotonic_ns() - started
    finished_utc = now_utc()
    if completed is None:
        fail(f"cannot upload {entry['path']!r}")
    if completed.returncode != 0:
        fail(f"upload of {entry['path']!r} failed with exit {completed.returncode}")
    return {
        "path": entry["path"],
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_ns": elapsed,
    }


def timed_readback(hf: str, destination: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    target = remote_uri(destination, entry["path"])
    started_utc = now_utc()
    started = time.monotonic_ns()
    digest = hashlib.sha256()
    byte_count = 0
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen([hf, "buckets", "cp", target, "-"], stdout=subprocess.PIPE)
    except OSError as exc:
        fail(f"cannot start readback for {entry['path']!r}: {exc}")
    if process is None or process.stdout is None:
        fail(f"cannot capture readback for {entry['path']!r}")
    stream = process.stdout
    while chunk := stream.read(8 << 20):
        digest.update(chunk)
        byte_count += len(chunk)
    return_code = process.wait()
    elapsed = time.monotonic_ns() - started
    finished_utc = now_utc()
    if return_code != 0:
        fail(f"independent readback of {entry['path']!r} failed with exit {return_code}")
    result_hash = digest.hexdigest()
    if byte_count != entry["size_bytes"] or result_hash != entry["sha256"]:
        fail(f"independent bucket readback identity mismatch for {entry['path']!r}")
    return {
        "path": entry["path"],
        "size_bytes": byte_count,
        "sha256": result_hash,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_ns": elapsed,
        "method": "hf buckets cp URI -; SHA256 computed independently over stdout bytes",
    }


def verify_remote_set(hf: str, destination: dict[str, Any], source: dict[str, Any]) -> None:
    expected = sorted(expected_remote_paths(destination, source), key=lambda value: value.encode("utf-8"))
    actual = list_remote(hf, destination)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        fail(f"bucket prefix object set mismatch; missing={missing[:5]}, extra={extra[:5]}")


def preservation_summary(
    source: dict[str, Any],
    local_verified_copy_count: int,
    preserved: bool,
) -> dict[str, Any]:
    return {
        "source_tree_sha256": source["tree_sha256"],
        "bucket_tree_sha256": source["tree_sha256"] if preserved else None,
        "source_and_bucket_match": preserved,
        "independent_verified_copy_count": local_verified_copy_count + (1 if preserved else 0),
        "preserved": preserved,
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source)
    second = Path(args.second_source) if args.second_source else None
    source, evidence = local_evidence(source_path, second)
    destination: dict[str, Any] = parse_destination(args.destination)
    destination["bucket_identity"] = None
    return {
        "schema": SCHEMA,
        "state": "planned",
        "source": source,
        "local_copy_evidence": evidence,
        "destination": destination,
        "transfer": None,
        "verification": None,
        "preservation": preservation_summary(
            source,
            evidence["verified_local_copy_count"],
            False,
        ),
        "prior_receipt_sha256": None,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    receipt, raw = load_receipt(Path(args.input))
    if receipt["state"] != "planned":
        fail("execute requires a planned receipt")
    destination_arg = parse_destination(args.destination)
    if destination_arg != {key: receipt["destination"][key] for key in ("uri", "bucket_id", "prefix")}:
        fail("--destination does not exactly match the planned destination identity")
    source_path = Path(args.source)
    second = Path(args.second_source) if args.second_source else None
    source, evidence = assert_expected_source(source_path, second, receipt)
    hf = hf_executable()
    identity_before = bucket_identity(hf, destination_arg["bucket_id"])
    if list_remote(hf, destination_arg):
        fail("destination prefix is nonempty; refusing to overwrite or merge preservation artifacts")

    upload_start_utc = now_utc()
    upload_start = time.monotonic_ns()
    uploads = [timed_upload(hf, source_path.resolve(strict=True), destination_arg, entry) for entry in source["entries"]]
    upload_elapsed = time.monotonic_ns() - upload_start
    upload_finish_utc = now_utc()
    source_after, _ = stable_inventory(source_path)
    if source_after != source:
        fail("source changed during upload")
    verify_remote_set(hf, destination_arg, source)

    readback_start_utc = now_utc()
    readback_start = time.monotonic_ns()
    readbacks = [timed_readback(hf, destination_arg, entry) for entry in source["entries"]]
    readback_elapsed = time.monotonic_ns() - readback_start
    readback_finish_utc = now_utc()
    identity_after = bucket_identity(hf, destination_arg["bucket_id"])
    if identity_after != identity_before:
        fail("bucket identity changed during preservation")
    source_final, _ = stable_inventory(source_path)
    if source_final != source:
        fail("source changed during independent readback")

    destination: dict[str, Any] = dict(destination_arg)
    destination["bucket_identity"] = identity_after
    return {
        "schema": SCHEMA,
        "state": "preserved",
        "source": source,
        "local_copy_evidence": evidence,
        "destination": destination,
        "transfer": {
            "started_utc": upload_start_utc,
            "finished_utc": upload_finish_utc,
            "elapsed_ns": upload_elapsed,
            "bytes": source["total_bytes"],
            "files": uploads,
            "method": "installed hf buckets cp; no delete operation",
        },
        "verification": {
            "started_utc": readback_start_utc,
            "finished_utc": readback_finish_utc,
            "elapsed_ns": readback_elapsed,
            "bytes": source["total_bytes"],
            "files": readbacks,
            "object_set_exact": True,
            "bucket_identity_stable": True,
        },
        "preservation": preservation_summary(
            source,
            evidence["verified_local_copy_count"],
            True,
        ),
        "prior_receipt_sha256": canonical_sha256(raw),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    receipt, raw = load_receipt(Path(args.input))
    if receipt["state"] != "preserved" or not receipt["preservation"]["preserved"]:
        fail("verify requires a receipt that already has complete preservation evidence")
    destination_arg = parse_destination(args.destination)
    if destination_arg != {key: receipt["destination"][key] for key in ("uri", "bucket_id", "prefix")}:
        fail("--destination does not exactly match the preserved destination identity")
    source_path = Path(args.source)
    second = Path(args.second_source) if args.second_source else None
    source, evidence = assert_expected_source(source_path, second, receipt)
    hf = hf_executable()
    identity = bucket_identity(hf, destination_arg["bucket_id"])
    if identity != receipt["destination"]["bucket_identity"]:
        fail("live bucket identity does not match the preserved bucket identity")
    verify_remote_set(hf, destination_arg, source)
    started_utc = now_utc()
    started = time.monotonic_ns()
    readbacks = [timed_readback(hf, destination_arg, entry) for entry in source["entries"]]
    elapsed = time.monotonic_ns() - started
    finished_utc = now_utc()
    source_after, _ = stable_inventory(source_path)
    if source_after != source:
        fail("source changed during verification")
    identity_after = bucket_identity(hf, destination_arg["bucket_id"])
    if identity_after != identity:
        fail("bucket identity changed during verification")

    destination: dict[str, Any] = dict(destination_arg)
    destination["bucket_identity"] = identity
    return {
        "schema": SCHEMA,
        "state": "preserved",
        "source": source,
        "local_copy_evidence": evidence,
        "destination": destination,
        "transfer": {"prior_receipt_sha256": canonical_sha256(raw), "reused_verified_upload": True},
        "verification": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_ns": elapsed,
            "bytes": source["total_bytes"],
            "files": readbacks,
            "object_set_exact": True,
            "bucket_identity_stable": True,
        },
        "preservation": preservation_summary(
            source,
            evidence["verified_local_copy_count"],
            True,
        ),
        "prior_receipt_sha256": canonical_sha256(raw),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="inventory immutable local bytes without network transfer")
    plan_parser.add_argument("--source", required=True, help="immutable source directory (path is never recorded)")
    plan_parser.add_argument(
        "--second-source",
        help="optional matching copy on a distinct filesystem; same-filesystem copies count once",
    )
    plan_parser.add_argument("--destination", required=True, help="private hf://buckets/... URI")
    plan_parser.add_argument("--output", required=True, help="new volatile-preservation.json path")

    for name in ("execute", "verify"):
        child = commands.add_parser(name, help=f"{name} a preservation receipt")
        child.add_argument("--input", required=True, help=f"strict {SCHEMA} receipt path")
        child.add_argument("--source", required=True, help="immutable source directory (path is never recorded)")
        child.add_argument(
            "--second-source",
            help="optional matching copy on a distinct filesystem; same-filesystem copies count once",
        )
        child.add_argument("--destination", required=True, help="exact private hf://buckets/... URI from the receipt")
        child.add_argument("--output", required=True, help="new volatile-preservation.json path")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        source_path = Path(args.source)
        result = plan(args) if args.command == "plan" else execute(args) if args.command == "execute" else verify(args)
        validate_receipt(result)
        atomic_write_json(output_path(args.output, source_path), result)
    except (OSError, ValueError) as exc:
        print(f"frontier_preserve: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
