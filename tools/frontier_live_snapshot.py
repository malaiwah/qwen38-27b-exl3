#!/usr/bin/env python3
"""Capture and verify the exact live AIBoss service restoration contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json, sha256_file

SNAPSHOT_SCHEMA = "qwen38-frontier-live-snapshot/1"
LOG_SCHEMA = "qwen38-frontier-snapshot-log-hashes/1"
METRICS_SCHEMA = "qwen38-frontier-metrics-capture/1"
VERIFY_SCHEMA = "qwen38-frontier-snapshot-verification/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SnapshotError(ValueError):
    """A fail-closed snapshot or verification error."""


def fail(message: str) -> NoReturn:
    raise SnapshotError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expect_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        fail(f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}")


def known_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a non-empty string")
    return value.strip()


def command(argv: list[str], label: str, *, allow: tuple[int, ...] = (0,)) -> bytes:
    result: subprocess.CompletedProcess[bytes] | None = None
    try:
        result = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        fail(f"cannot execute {label}: {exc}")
    if result is None:
        fail(f"cannot execute {label}")
    if result.returncode not in allow:
        diagnostic = result.stderr.decode("utf-8", "replace").strip()[-500:]
        fail(f"{label} exited {result.returncode}: {diagnostic}")
    return result.stdout

def command_combined(argv: list[str], label: str) -> bytes:
    result: subprocess.CompletedProcess[bytes] | None = None
    try:
        result = subprocess.run(
            argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except OSError as exc:
        fail(f"cannot execute {label}: {exc}")
    if result is None:
        fail(f"cannot execute {label}")
    if result.returncode != 0:
        diagnostic = result.stdout.decode("utf-8", "replace").strip()[-500:]
        fail(f"{label} exited {result.returncode}: {diagnostic}")
    return result.stdout


def utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{label} is not UTF-8: {exc}")
    raise AssertionError("unreachable")


def command_json(argv: list[str], label: str) -> object:
    data = command(argv, label)
    try:
        return json.loads(utf8(data, label))
    except json.JSONDecodeError as exc:
        fail(f"{label} did not return JSON: {exc}")
    raise AssertionError("unreachable")


def http_get(url: str, label: str, *, json_response: bool) -> tuple[int, bytes, object | None]:
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    status: int | None = None
    body: bytes | None = None
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read(64 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail(f"{label} request failed: {exc}")
    if status is None or body is None:
        fail(f"{label} request returned no response")
    if status != 200:
        fail(f"{label} returned HTTP {status}")
    parsed: object | None = None
    if json_response:
        try:
            parsed = json.loads(utf8(body, label))
        except json.JSONDecodeError as exc:
            fail(f"{label} returned invalid JSON: {exc}")
    return status, body, parsed


def reject_existing_outputs(paths: list[Path]) -> None:
    resolved: set[Path] = set()
    for path in paths:
        absolute = path.resolve(strict=False)
        if absolute in resolved:
            fail(f"output paths alias: {path}")
        resolved.add(absolute)
        if path.exists() or path.is_symlink():
            fail(f"output already exists: {path}")
        if not path.parent.exists() or not path.parent.is_dir():
            fail(f"output parent directory is missing: {path.parent}")


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "integrity": {"canonical_sha256": canonical_sha256(body)}}


def validate_seal(value: dict[str, Any], label: str) -> str:
    integrity = expect_object(value.get("integrity"), f"{label}.integrity")
    exact_keys(integrity, {"canonical_sha256"}, f"{label}.integrity")
    digest = known_string(integrity["canonical_sha256"], f"{label}.integrity.canonical_sha256")
    if not SHA256_RE.fullmatch(digest):
        fail(f"{label} canonical digest is malformed")
    body = {key: item for key, item in value.items() if key != "integrity"}
    if canonical_sha256(body) != digest:
        fail(f"{label} canonical digest mismatch")
    return digest


def env_identity(entries: object) -> list[dict[str, str | int]]:
    if not isinstance(entries, list):
        fail("podman inspect Config.Env is malformed")
    values: dict[str, list[str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, str) or "=" not in raw_entry:
            fail("podman inspect Config.Env is malformed")
        name, raw = raw_entry.split("=", 1)
        if not name:
            fail("podman inspect contains an empty environment name")
        values.setdefault(name, []).append(hashlib.sha256(raw.encode()).hexdigest())
    return [
        {
            "name": name,
            "value_sha256": values[name][-1],
            "occurrences": len(values[name]),
            "value_multiset_sha256": canonical_sha256(sorted(values[name])),
        }
        for name in sorted(values)
    ]


def label_identity(labels: object) -> dict[str, str]:
    if labels is None:
        return {}
    if not isinstance(labels, dict):
        fail("podman inspect Config.Labels is malformed")
    result: dict[str, str] = {}
    for key, value in labels.items():
        if not isinstance(key, str) or not isinstance(value, str):
            fail("podman inspect Config.Labels is malformed")
        result[key] = hashlib.sha256(value.encode()).hexdigest()
    return dict(sorted(result.items()))


def source_binding(source: str) -> str:
    return hashlib.sha256(os.fsencode(os.path.realpath(source))).hexdigest()


def normalize_mount(mount: object) -> dict[str, Any]:
    item = expect_object(mount, "podman mount")
    source = known_string(item.get("Source"), "podman mount source")
    destination = known_string(item.get("Destination"), "podman mount destination")
    options_raw = item.get("Options") or []
    if not isinstance(options_raw, list):
        fail("podman mount Options must be a string array")
    options: list[str] = []
    for option in options_raw:
        if not isinstance(option, str):
            fail("podman mount Options must be a string array")
        options.append(option)
    return {
        "type": known_string(item.get("Type"), "podman mount type"),
        "destination": destination,
        "source_binding_sha256": source_binding(source),
        "source_kind": "file" if os.path.isfile(source) else "directory" if os.path.isdir(source) else "other",
        "rw": item.get("RW") is True,
        "options": sorted(options),
        "propagation": item.get("Propagation") or "",
        "name": item.get("Name") or "",
    }


def normalize_devices(devices: object) -> list[dict[str, str]]:
    if devices is None:
        return []
    if not isinstance(devices, list):
        fail("podman inspect HostConfig.Devices must be an array")
    result: list[dict[str, str]] = []
    for raw in devices:
        device = expect_object(raw, "podman device")
        result.append(
            {
                "path_on_host_sha256": source_binding(
                    known_string(device.get("PathOnHost"), "device PathOnHost")
                ),
                "path_in_container": device.get("PathInContainer") or "",
                "cgroup_permissions": device.get("CgroupPermissions") or "",
            }
        )
    return result


def inspect_identity(inspect: dict[str, Any]) -> dict[str, Any]:
    config = expect_object(inspect.get("Config"), "podman inspect Config")
    host = expect_object(inspect.get("HostConfig"), "podman inspect HostConfig")
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        fail("podman inspect Mounts must be an array")
    fields = {
        "image": config.get("Image"),
        "container_image_id": inspect.get("Image"),
        "create_command_sha256": canonical_sha256(config.get("CreateCommand") or []),
        "command": config.get("Cmd") or [],
        "entrypoint": config.get("Entrypoint") or "",
        "working_dir": config.get("WorkingDir") or "",
        "user": config.get("User") or "",
        "environment": env_identity(config.get("Env") or []),
        "labels": label_identity(config.get("Labels")),
        "healthcheck": config.get("Healthcheck"),
        "stop_signal": config.get("StopSignal"),
        "network_mode": host.get("NetworkMode"),
        "ipc_mode": host.get("IpcMode"),
        "shm_size": host.get("ShmSize"),
        "pid_mode": host.get("PidMode") or "",
        "uts_mode": host.get("UTSMode") or "",
        "cgroupns_mode": host.get("CgroupnsMode") or "",
        "read_only_rootfs": host.get("ReadonlyRootfs") is True,
        "runtime": host.get("Runtime") or "",
        "cap_add": host.get("CapAdd") or [],
        "cap_drop": host.get("CapDrop") or [],
        "security_opt": host.get("SecurityOpt") or [],
        "port_bindings": host.get("PortBindings") or {},
        "devices": normalize_devices(host.get("Devices")),
        "device_requests": host.get("DeviceRequests") or [],
        "restart_policy": host.get("RestartPolicy"),
        "privileged": host.get("Privileged") is True,
        "ulimits": host.get("Ulimits") or [],
        "tmpfs": host.get("Tmpfs") or {},
        "mounts": sorted((normalize_mount(item) for item in mounts), key=lambda item: item["destination"]),
    }
    return {"fields": fields, "canonical_sha256": canonical_sha256(fields)}


def inspect_one(container: str) -> dict[str, Any]:
    raw = command_json(["podman", "inspect", container], f"podman inspect {container}")
    if not isinstance(raw, list) or len(raw) != 1:
        fail(f"podman inspect {container} did not return exactly one object")
    return expect_object(raw[0], f"podman inspect {container}[0]")


def image_identity(inspect: dict[str, Any], expected_image: str) -> dict[str, str]:
    if not IMAGE_RE.fullmatch(expected_image):
        fail("--expected-image must be an immutable name@sha256:<64 hex> reference")
    config = expect_object(inspect.get("Config"), "podman inspect Config")
    configured = known_string(config.get("Image"), "podman inspect Config.Image")
    image_raw = command_json(["podman", "image", "inspect", expected_image], "podman image inspect")
    if not isinstance(image_raw, list) or len(image_raw) != 1:
        fail("podman image inspect did not return exactly one image")
    image = expect_object(image_raw[0], "podman image inspect[0]")
    digests = image.get("RepoDigests") or []
    if expected_image not in digests and configured != expected_image:
        fail("running container does not resolve to --expected-image")
    digest = expected_image.rsplit("@", 1)[1]
    return {
        "reference": expected_image,
        "digest": digest,
        "image_id": known_string(image.get("Id"), "podman image Id"),
    }


def overlay_identities(inspect: dict[str, Any]) -> list[dict[str, Any]]:
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        fail("podman inspect Mounts must be an array")
    overlays: list[dict[str, Any]] = []
    for raw in mounts:
        mount = expect_object(raw, "podman mount")
        source = known_string(mount.get("Source"), "podman mount source")
        if mount.get("RW") is False and Path(source).is_file():
            overlays.append({
                "container_path": known_string(mount.get("Destination"), "overlay destination"),
                "sha256": sha256_file(Path(source)),
                "bytes": Path(source).stat().st_size,
            })
    return sorted(overlays, key=lambda item: item["container_path"])


def clean_artifact(relative: str) -> PurePosixPath:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        fail(f"model artifact is not a clean relative path: {relative!r}")
    return pure


def model_artifacts(model_root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    if not model_root.is_dir() or model_root.is_symlink():
        fail(f"model root must be a real directory: {model_root}")
    if not names:
        fail("at least one --model-artifact is required")
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        pure = clean_artifact(name)
        if str(pure) in result:
            fail(f"duplicate model artifact: {name}")
        path = model_root.joinpath(*pure.parts)
        try:
            path.resolve(strict=True).relative_to(model_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            fail(f"model artifact is missing or escapes model root: {name}: {exc}")
        if not path.is_file():
            fail(f"model artifact does not resolve to a regular file: {name}")
        result[str(pure)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return dict(sorted(result.items()))


def systemd_show(unit: str) -> dict[str, str]:
    properties = (
        "Id,LoadState,ActiveState,SubState,UnitFileState,FragmentPath,ExecStart,ExecStop,"
        "ExecStopPost,Restart,RestartUSec,TimeoutStartUSec,MainPID"
    )
    text = utf8(command(["systemctl", "--user", "show", unit, f"--property={properties}"], f"systemctl show {unit}"), "systemctl show")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            fail("systemctl show returned a malformed line")
        key, value = line.split("=", 1)
        if key in result:
            fail(f"systemctl show duplicated property {key}")
        result[key] = value
    expected = set(properties.split(","))
    if result.keys() != expected:
        fail(f"systemctl show properties differ: {sorted(result.keys())}")
    return result


def public_systemd_show(show: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in show.items():
        if key == "FragmentPath":
            continue
        if key in {"ExecStart", "ExecStop", "ExecStopPost"}:
            result[f"{key}_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        elif key == "MainPID":
            try:
                result["MainPID"] = int(value)
            except ValueError:
                fail("systemctl show MainPID is not an integer")
        else:
            result[key] = value
    return result



def gpu_state(gpu_uuid: str) -> dict[str, Any]:
    fields = [
        "uuid", "name", "driver_version", "memory.total", "memory.used", "memory.free",
        "compute_mode", "persistence_mode", "power.draw", "power.limit",
        "clocks.current.sm", "clocks.current.memory", "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    raw = utf8(command([
        "nvidia-smi", f"--id={gpu_uuid}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"
    ], "nvidia-smi GPU query"), "nvidia-smi GPU query").strip()
    rows = [row for row in raw.splitlines() if row.strip()]
    if len(rows) != 1:
        fail(f"nvidia-smi returned {len(rows)} GPU rows, expected one")
    values = [part.strip() for part in rows[0].split(",")]
    if len(values) != len(fields):
        fail("nvidia-smi GPU row has an unexpected column count")
    data = dict(zip(fields, values))
    if data["uuid"] != gpu_uuid:
        fail("nvidia-smi GPU UUID disagrees with --gpu-uuid")
    numeric: dict[str, int | float | None] = {}
    for field in (
        "memory.total",
        "memory.used",
        "memory.free",
        "clocks.current.sm",
        "clocks.current.memory",
    ):
        try:
            numeric[field] = int(float(data[field]))
        except ValueError:
            fail(f"nvidia-smi {field} is not numeric: {data[field]!r}")
    for field in ("clocks.applications.graphics", "clocks.applications.memory"):
        try:
            numeric[field] = int(float(data[field]))
        except ValueError:
            numeric[field] = None
    for field in ("power.draw", "power.limit"):
        try:
            numeric[field] = float(data[field])
        except ValueError:
            fail(f"nvidia-smi {field} is not numeric: {data[field]!r}")
    apps_raw = utf8(command([
        "nvidia-smi", f"--id={gpu_uuid}", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"
    ], "nvidia-smi compute apps"), "nvidia-smi compute apps")
    apps: list[dict[str, Any]] = []
    for row in apps_raw.splitlines():
        if not row.strip():
            continue
        parts = [part.strip() for part in row.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[2].isdigit():
            fail(f"malformed nvidia-smi compute-app row: {row!r}")
        apps.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": int(parts[2])})
    return {
        "uuid": data["uuid"], "name": data["name"], "driver_version": data["driver_version"],
        "memory_total_mib": numeric["memory.total"], "memory_used_mib": numeric["memory.used"],
        "memory_free_mib": numeric["memory.free"], "compute_mode": data["compute_mode"],
        "persistence_mode": data["persistence_mode"], "power_draw_w": numeric["power.draw"],
        "power_limit_w": numeric["power.limit"], "sm_clock_mhz": numeric["clocks.current.sm"],
        "memory_clock_mhz": numeric["clocks.current.memory"],
        "application_clocks_supported": all(
            numeric[field] is not None
            for field in ("clocks.applications.graphics", "clocks.applications.memory")
        ),
        "application_sm_clock_mhz": numeric["clocks.applications.graphics"],
        "application_memory_clock_mhz": numeric["clocks.applications.memory"], "compute_apps": apps,
    }


def capacity_entries(values: list[str]) -> list[dict[str, Any]]:
    if not values:
        fail("at least one --capacity LABEL=PATH is required")
    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            fail(f"capacity must be LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        if not PROFILE_RE.fullmatch(label) or label in labels:
            fail(f"capacity label is invalid or duplicated: {label!r}")
        labels.add(label)
        path = Path(raw_path)
        if not path.exists() or path.is_symlink():
            fail(f"capacity path is missing or a symlink: {path}")
        stat = os.statvfs(path)
        result.append({"label": label, "free_bytes": stat.f_bavail * stat.f_frsize, "total_bytes": stat.f_blocks * stat.f_frsize})
    return sorted(result, key=lambda item: item["label"])


def model_api(models: object, revision: str, max_model_len: int) -> dict[str, Any]:
    root = expect_object(models, "/v1/models response")
    data = root.get("data")
    if not isinstance(data, list) or not data:
        fail("/v1/models data must be a non-empty array")
    matches: list[dict[str, Any]] = []
    for item in data:
        model = expect_object(item, "/v1/models entry")
        model_revision = model.get("revision") or model.get("model_revision") or model.get("root")
        observed_length = model.get("max_model_len") or model.get("max_model_length")
        if isinstance(model_revision, str) and revision in model_revision and observed_length == max_model_len:
            matches.append(model)
    if len(matches) != 1:
        fail("/v1/models does not identify exactly one expected revision and max-model-len")
    selected = matches[0]
    return {
        "id": known_string(selected.get("id"), "/v1/models selected id"),
        "revision": revision,
        "max_model_len": max_model_len,
        "response_canonical_sha256": canonical_sha256(root),
    }


def validate_snapshot(value: object) -> tuple[dict[str, Any], str]:
    root = expect_object(value, "snapshot")
    exact_keys(
        root,
        {
            "schema", "captured_utc", "host", "service", "container", "image",
            "model", "profile", "mounts", "storage", "endpoints", "gpu",
            "artifacts", "integrity",
        },
        "snapshot",
    )
    if root["schema"] != SNAPSHOT_SCHEMA:
        fail(f"unsupported snapshot schema: {root['schema']!r}")
    nested_keys = {
        "host": {"hostname", "uid", "container_engine"},
        "service": {
            "scope", "unit", "show", "unit_file_sha256", "unit_cat_sha256",
            "launcher_sha256", "overlays",
        },
        "container": {
            "name", "id", "created", "state", "health", "argv", "environment",
            "inspect_identity",
        },
        "image": {"reference", "digest", "image_id"},
        "model": {"repo", "revision", "max_model_len", "served", "artifacts"},
        "profile": {"name", "proof"},
        "endpoints": {"api_base", "health", "models", "metrics"},
        "gpu": {
            "uuid", "name", "driver_version", "memory_total_mib",
            "memory_used_mib", "memory_free_mib", "compute_mode",
            "persistence_mode", "power_draw_w", "power_limit_w",
            "sm_clock_mhz", "memory_clock_mhz", "application_clocks_supported",
            "application_sm_clock_mhz", "application_memory_clock_mhz", "compute_apps",
        },
        "artifacts": {"logs", "metrics"},
    }
    objects: dict[str, dict[str, Any]] = {}
    for key, keys in nested_keys.items():
        objects[key] = expect_object(root[key], f"snapshot.{key}")
        exact_keys(objects[key], keys, f"snapshot.{key}")
    if root["service"]["scope"] != "user":
        fail("snapshot service scope is not user")
    if root["container"]["state"] != "running" or root["container"]["health"] != "healthy":
        fail("snapshot does not describe a running healthy container")
    if not isinstance(root["mounts"], list) or not root["mounts"]:
        fail("snapshot mounts must be a non-empty array")
    if not isinstance(root["storage"], list) or not root["storage"]:
        fail("snapshot storage must be a non-empty array")
    if not isinstance(root["service"]["overlays"], list):
        fail("snapshot service overlays must be an array")
    if not isinstance(root["container"]["argv"], list) or not isinstance(root["container"]["environment"], list):
        fail("snapshot container argv/environment must be arrays")
    if not IMAGE_RE.fullmatch(known_string(root["image"]["reference"], "snapshot.image.reference")):
        fail("snapshot image reference is not digest-pinned")
    image_digest = known_string(root["image"]["digest"], "snapshot.image.digest")
    if not image_digest.startswith("sha256:"):
        fail("snapshot image digest lacks sha256 prefix")
    for location in (
        root["service"]["unit_file_sha256"],
        root["service"]["unit_cat_sha256"],
        root["service"]["launcher_sha256"],
        image_digest.removeprefix("sha256:"),
    ):
        if not isinstance(location, str) or not SHA256_RE.fullmatch(location):
            fail("snapshot contains a malformed source SHA256")
    digest = validate_seal(root, "snapshot")
    inspect_doc = expect_object(root["container"]["inspect_identity"], "snapshot.container.inspect_identity")
    exact_keys(inspect_doc, {"fields", "canonical_sha256"}, "snapshot.container.inspect_identity")
    if not isinstance(inspect_doc["canonical_sha256"], str) or not SHA256_RE.fullmatch(inspect_doc["canonical_sha256"]):
        fail("snapshot inspect identity SHA256 is malformed")
    if canonical_sha256(inspect_doc["fields"]) != inspect_doc["canonical_sha256"]:
        fail("snapshot inspect identity digest mismatch")
    artifacts = expect_object(root["model"]["artifacts"], "snapshot.model.artifacts")
    if not artifacts:
        fail("snapshot model artifacts are empty")
    for name, artifact in artifacts.items():
        clean_artifact(name)
        entry = expect_object(artifact, f"snapshot.model.artifacts.{name}")
        exact_keys(entry, {"sha256", "bytes"}, f"snapshot.model.artifacts.{name}")
        if not isinstance(entry["sha256"], str) or not SHA256_RE.fullmatch(entry["sha256"]):
            fail(f"snapshot model artifact {name} SHA256 is malformed")
        if isinstance(entry["bytes"], bool) or not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
            fail(f"snapshot model artifact {name} byte count is invalid")
    return root, digest


def common_source_identity(
    unit: str,
    unit_file: Path,
    launcher: Path,
    container: str,
    model_root: Path,
    artifact_names: list[str],
) -> dict[str, Any]:
    if not unit_file.is_file() or unit_file.is_symlink():
        fail(f"unit file must be a regular non-symlink file: {unit_file}")
    if not launcher.is_file() or launcher.is_symlink():
        fail(f"launcher must be a regular non-symlink file: {launcher}")
    inspect = inspect_one(container)
    return {
        "unit_file_sha256": sha256_file(unit_file),
        "unit_cat_sha256": hashlib.sha256(
            command(["systemctl", "--user", "cat", unit], "systemctl cat")
        ).hexdigest(),
        "launcher_sha256": sha256_file(launcher),
        "overlays": overlay_identities(inspect),
        "model_artifacts": model_artifacts(model_root, artifact_names),
        "inspect_identity": inspect_identity(inspect),
    }


def capture(args: argparse.Namespace) -> None:
    if not UNIT_RE.fullmatch(args.unit):
        fail("--unit must be a valid .service unit name")
    if not CONTAINER_RE.fullmatch(args.container):
        fail("--container is invalid")
    if not PROFILE_RE.fullmatch(args.profile):
        fail("--profile is invalid")
    if not REVISION_RE.fullmatch(args.model_revision):
        fail("--model-revision must be a 40- or 64-character lowercase hex revision")
    if args.max_model_len <= 0:
        fail("--max-model-len must be positive")
    outputs = [args.output, args.logs_output, args.metrics_output]
    reject_existing_outputs(outputs)
    show = systemd_show(args.unit)
    if show["LoadState"] != "loaded" or show["ActiveState"] != "active" or show["SubState"] != "running":
        fail(f"owner unit is not loaded/active/running: {show['LoadState']}/{show['ActiveState']}/{show['SubState']}")
    fragment: Path | None = None
    try:
        fragment = Path(show["FragmentPath"]).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        fail(f"cannot resolve systemd FragmentPath: {exc}")
    if fragment is None or fragment != args.unit_file.resolve(strict=True):
        fail("systemd FragmentPath does not equal the explicit unit file")
    launcher_resolved = str(args.launcher.resolve(strict=True))
    if launcher_resolved not in show["ExecStart"]:
        fail("systemd ExecStart does not name the explicit launcher")
    sources = common_source_identity(
        args.unit, args.unit_file, args.launcher, args.container, args.model_root, args.model_artifact
    )
    inspect = inspect_one(args.container)
    state = expect_object(inspect.get("State"), "podman inspect State")
    health = expect_object(state.get("Health"), "podman inspect State.Health")
    if state.get("Running") is not True or health.get("Status") != "healthy":
        fail("live container is not running and healthy")
    labels = expect_object(expect_object(inspect.get("Config"), "podman inspect Config").get("Labels"), "podman inspect Config.Labels")
    if labels.get("PODMAN_SYSTEMD_UNIT") != args.unit:
        fail("container is not labelled as owned by the explicit user unit")
    image = image_identity(inspect, args.expected_image)
    command_args = expect_object(inspect.get("Config"), "podman inspect Config").get("Cmd") or []
    command_text = "\n".join(str(item) for item in command_args)
    if args.model_revision not in command_text:
        fail("container argv does not contain the expected model revision")
    max_tokens = (
        f"--max-model-len {args.max_model_len}",
        f"--max-model-len={args.max_model_len}",
        f"--max-model-len '{args.max_model_len}'",
        f'--max-model-len "{args.max_model_len}"',
    )
    if not any(token in command_text for token in max_tokens):
        fail("container argv does not contain the expected max-model-len")
    health_status, health_body, _ = http_get(f"{args.api_base.rstrip('/')}/health", "/health", json_response=False)
    _, _, models = http_get(f"{args.api_base.rstrip('/')}/v1/models", "/v1/models", json_response=True)
    model_endpoint = model_api(models, args.model_revision, args.max_model_len)
    metrics_status, metrics_body, _ = http_get(f"{args.api_base.rstrip('/')}/metrics", "/metrics", json_response=False)
    journal = command(["journalctl", "--user", "--unit", args.unit, "--no-pager", "--output=short-iso", f"--lines={args.journal_lines}"], "journalctl")
    container_log = command_combined(["podman", "logs", f"--tail={args.container_log_lines}", args.container], "podman logs")
    metrics_text = utf8(metrics_body, "/metrics")
    metric_families: set[str] = set()
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[{\\s]", line, maxsplit=1)[0]
        if not re.fullmatch(r"[A-Za-z_:][A-Za-z0-9_:]*", name):
            fail(f"/metrics contains a malformed metric name: {name!r}")
        metric_families.add(name)
    if not metric_families:
        fail("/metrics contains no metric families")
    metrics_artifact = seal({
        "schema": METRICS_SCHEMA,
        "captured_utc": utc_now(),
        "http_status": metrics_status,
        "body_sha256": hashlib.sha256(metrics_body).hexdigest(),
        "bytes": len(metrics_body),
        "lines": len(metrics_body.splitlines()),
        "metric_families": sorted(metric_families),
    })
    logs_artifact = seal({
        "schema": LOG_SCHEMA,
        "captured_utc": utc_now(),
        "unit": args.unit,
        "container": args.container,
        "journal": {"sha256": hashlib.sha256(journal).hexdigest(), "bytes": len(journal), "lines": len(journal.splitlines())},
        "container_log": {"sha256": hashlib.sha256(container_log).hexdigest(), "bytes": len(container_log), "lines": len(container_log.splitlines())},
    })
    mounts = sources["inspect_identity"]["fields"]["mounts"]
    snapshot_body = {
        "schema": SNAPSHOT_SCHEMA,
        "captured_utc": utc_now(),
        "host": {"hostname": socket.gethostname(), "uid": os.getuid(), "container_engine": utf8(command(["podman", "--version"], "podman --version"), "podman version").strip()},
        "service": {
            "scope": "user", "unit": args.unit, "show": public_systemd_show(show),
            "unit_file_sha256": sources["unit_file_sha256"], "unit_cat_sha256": sources["unit_cat_sha256"],
            "launcher_sha256": sources["launcher_sha256"], "overlays": sources["overlays"],
        },
        "container": {
            "name": args.container, "id": known_string(inspect.get("Id"), "podman inspect Id"),
            "created": known_string(inspect.get("Created"), "podman inspect Created"), "state": "running", "health": "healthy",
            "argv": command_args, "environment": sources["inspect_identity"]["fields"]["environment"],
            "inspect_identity": sources["inspect_identity"],
        },
        "image": image,
        "model": {"repo": args.model_repo, "revision": args.model_revision, "max_model_len": args.max_model_len, "served": model_endpoint, "artifacts": sources["model_artifacts"]},
        "profile": {"name": args.profile, "proof": "operator-declared profile bound to exact argv/environment/inspect identity"},
        "mounts": mounts,
        "storage": capacity_entries(args.capacity),
        "endpoints": {
            "api_base": args.api_base, "health": {"http_status": health_status, "body_sha256": hashlib.sha256(health_body).hexdigest()},
            "models": model_endpoint, "metrics": {"artifact_schema": METRICS_SCHEMA, "artifact_canonical_sha256": metrics_artifact["integrity"]["canonical_sha256"], "body_sha256": hashlib.sha256(metrics_body).hexdigest()},
        },
        "gpu": gpu_state(args.gpu_uuid),
        "artifacts": {
            "logs": {"schema": LOG_SCHEMA, "canonical_sha256": logs_artifact["integrity"]["canonical_sha256"]},
            "metrics": {"schema": METRICS_SCHEMA, "canonical_sha256": metrics_artifact["integrity"]["canonical_sha256"]},
        },
    }
    atomic_write_json(args.logs_output, logs_artifact)
    atomic_write_json(args.metrics_output, metrics_artifact)
    atomic_write_json(args.output, seal(snapshot_body))


def verify(args: argparse.Namespace) -> None:
    reject_existing_outputs([args.output])
    _, digest = validate_snapshot(load_strict_json(args.input))
    result = seal({"schema": VERIFY_SCHEMA, "verified_utc": utc_now(), "snapshot_canonical_sha256": digest, "status": "pass", "checks": ["strict_json", "exact_root_keys", "schema", "canonical_digest", "inspect_digest", "model_artifacts_nonempty"]})
    atomic_write_json(args.output, result)


def verify_live(args: argparse.Namespace) -> None:
    reject_existing_outputs([args.output])
    snapshot, digest = validate_snapshot(load_strict_json(args.input))
    service = expect_object(snapshot["service"], "snapshot.service")
    container_doc = expect_object(snapshot["container"], "snapshot.container")
    model = expect_object(snapshot["model"], "snapshot.model")
    if service.get("unit") != args.unit or container_doc.get("name") != args.container:
        fail("requested unit/container disagree with verified snapshot")
    artifact_doc = expect_object(model.get("artifacts"), "snapshot.model.artifacts")
    sources = common_source_identity(
        args.unit, args.unit_file, args.launcher, args.container, args.model_root, list(artifact_doc)
    )
    comparisons = {
        "unit_file_sha256": (sources["unit_file_sha256"], service.get("unit_file_sha256")),
        "unit_cat_sha256": (sources["unit_cat_sha256"], service.get("unit_cat_sha256")),
        "launcher_sha256": (sources["launcher_sha256"], service.get("launcher_sha256")),
        "overlays": (sources["overlays"], service.get("overlays")),
        "model_artifacts": (sources["model_artifacts"], artifact_doc),
        "inspect_identity": (sources["inspect_identity"], container_doc.get("inspect_identity")),
    }
    mismatches = [name for name, (actual, expected) in comparisons.items() if actual != expected]
    if mismatches:
        fail(f"live source identity differs from snapshot: {mismatches}")
    atomic_write_json(args.output, seal({
        "schema": VERIFY_SCHEMA, "verified_utc": utc_now(), "snapshot_canonical_sha256": digest,
        "status": "pass", "checks": sorted(comparisons),
    }))


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="verified snapshot JSON input")
    parser.add_argument("--output", required=True, type=Path, help="new verification JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture or verify a fail-closed current-service restoration snapshot.")
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture", help="capture a fresh running-service snapshot")
    capture_parser.add_argument("--unit", required=True)
    capture_parser.add_argument("--unit-file", required=True, type=Path)
    capture_parser.add_argument("--launcher", required=True, type=Path)
    capture_parser.add_argument("--container", required=True)
    capture_parser.add_argument("--expected-image", required=True)
    capture_parser.add_argument("--model-root", required=True, type=Path)
    capture_parser.add_argument("--model-repo", required=True)
    capture_parser.add_argument("--model-revision", required=True)
    capture_parser.add_argument("--model-artifact", required=True, action="append")
    capture_parser.add_argument("--profile", required=True)
    capture_parser.add_argument("--max-model-len", required=True, type=int)
    capture_parser.add_argument("--gpu-uuid", required=True)
    capture_parser.add_argument("--api-base", required=True)
    capture_parser.add_argument("--capacity", required=True, action="append", metavar="LABEL=PATH")
    capture_parser.add_argument("--journal-lines", type=int, default=1000)
    capture_parser.add_argument("--container-log-lines", type=int, default=1000)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--logs-output", required=True, type=Path)
    capture_parser.add_argument("--metrics-output", required=True, type=Path)
    capture_parser.set_defaults(func=capture)
    verify_parser = sub.add_parser("verify", help="strictly verify a snapshot without reading live state")
    add_source_arguments(verify_parser)
    verify_parser.set_defaults(func=verify)
    live_parser = sub.add_parser("verify-live", help="compare live source and inspect identities with a snapshot")
    add_source_arguments(live_parser)
    live_parser.add_argument("--unit", required=True)
    live_parser.add_argument("--unit-file", required=True, type=Path)
    live_parser.add_argument("--launcher", required=True, type=Path)
    live_parser.add_argument("--container", required=True)
    live_parser.add_argument("--model-root", required=True, type=Path)
    live_parser.set_defaults(func=verify_live)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if hasattr(args, "journal_lines") and (args.journal_lines <= 0 or args.container_log_lines <= 0):
            fail("log line limits must be positive")
        args.func(args)
    except (SnapshotError, OSError, ValueError) as exc:
        print(f"frontier_live_snapshot: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
