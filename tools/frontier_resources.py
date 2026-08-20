#!/usr/bin/env python3
"""Build a fail-closed prospective storage live-set ledger for Frontier G0/G1."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json

INPUT_SCHEMA = "qwen38-frontier-resource-plan/1"
OUTPUT_SCHEMA = "qwen38-frontier-resource-ledger/1"
AIBOSS_RESERVE_BYTES = 60 * 1024**3
SIZE_BASES = {"exact", "upper_bound"}
LOCATION_KINDS = {"filesystem", "object_storage"}
RETENTIONS = {"ephemeral", "campaign", "permanent"}
COMPONENTS = ("atomic_temp", "checksum_staging", "upload_staging", "retry_reserve")
BYTE_CATEGORIES = ("live_artifacts", "retained_copies") + COMPONENTS


class ResourceError(ValueError):
    """A closed planning or validation failure."""


def fail(message: str) -> NoReturn:
    raise ResourceError(message)


def obj(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{label} must be a JSON array")
    return cast(list[Any], value)


def keys(value: dict[str, Any], required: set[str], optional: set[str], label: str) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
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


def reject_host_paths(value: object, label: str) -> None:
    """Keep canonical identities independent of a particular machine."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"path", "local_path", "host_path", "source_path"}:
                fail(f"{label} contains forbidden host-path field {key!r}")
            reject_host_paths(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_host_paths(child, f"{label}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("/", "file://")):
            fail(f"{label} contains a host path rather than an immutable identity")


def phase_index(value: object, label: str, phases: dict[str, int], *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    name = text(value, label)
    if name not in phases:
        fail(f"{label} names unknown phase {name!r}")
    return phases[name]


def interval(
    component: dict[str, Any],
    label: str,
    phases: dict[str, int],
    *,
    zero: bool,
) -> tuple[int | None, int | None]:
    birth = phase_index(component["birth_phase"], f"{label}.birth_phase", phases, nullable=zero)
    release = phase_index(component["release_phase"], f"{label}.release_phase", phases, nullable=True)
    if zero:
        if birth is not None or release is not None:
            fail(f"{label} with zero allocation must use null birth_phase and release_phase")
        return None, None
    assert birth is not None
    if release is not None and release <= birth:
        fail(f"{label}.release_phase must be after birth_phase")
    return birth, release


def active(phase: int, birth: int | None, release: int | None) -> bool:
    return birth is not None and phase >= birth and (release is None or phase < release)


def byte_pair(size_bytes: int, basis: str, factor: int) -> tuple[int, int]:
    amount = size_bytes * factor
    return (amount, amount) if basis == "exact" else (0, amount)


def parse_document(raw: object) -> dict[str, Any]:
    document = obj(raw, "input")
    keys(
        document,
        {"schema", "campaign_identity", "phase_order", "branches", "locations", "upload", "artifacts"},
        set(),
        "input",
    )
    if document["schema"] != INPUT_SCHEMA:
        fail(f"input.schema must be {INPUT_SCHEMA!r}")

    identity = obj(document["campaign_identity"], "campaign_identity")
    if not identity:
        fail("campaign_identity must not be empty")
    reject_host_paths(identity, "campaign_identity")

    phase_names = [text(item, f"phase_order[{index}]") for index, item in enumerate(array(document["phase_order"], "phase_order"))]
    if not phase_names or len(set(phase_names)) != len(phase_names):
        fail("phase_order must be non-empty and unique")
    phase_map = {name: index for index, name in enumerate(phase_names)}

    branch_names = [text(item, f"branches[{index}]") for index, item in enumerate(array(document["branches"], "branches"))]
    if not branch_names or len(set(branch_names)) != len(branch_names):
        fail("branches must be non-empty and unique")
    branch_set = set(branch_names)

    locations: dict[str, dict[str, Any]] = {}
    for index, raw_location in enumerate(array(document["locations"], "locations")):
        location = obj(raw_location, f"locations[{index}]")
        keys(
            location,
            {"id", "kind", "copy_domain", "capacity_bytes", "available_bytes", "role"},
            set(),
            f"locations[{index}]",
        )
        name = text(location["id"], f"locations[{index}].id")
        if name in locations:
            fail(f"duplicate location id {name!r}")
        kind = text(location["kind"], f"locations[{index}].kind")
        if kind not in LOCATION_KINDS:
            fail(f"locations[{index}].kind must be one of {sorted(LOCATION_KINDS)}")
        capacity = integer(location["capacity_bytes"], f"locations[{index}].capacity_bytes", 1)
        available = integer(location["available_bytes"], f"locations[{index}].available_bytes")
        if available > capacity:
            fail(f"locations[{index}].available_bytes exceeds capacity_bytes")
        role = text(location["role"], f"locations[{index}].role")
        copy_domain = text(location["copy_domain"], f"locations[{index}].copy_domain")
        reject_host_paths(copy_domain, f"locations[{index}].copy_domain")
        locations[name] = {
            "id": name,
            "kind": kind,
            "copy_domain": copy_domain,
            "capacity_bytes": capacity,
            "available_bytes": available,
            "role": role,
        }
    domains = [location["copy_domain"] for location in locations.values()]
    if len(domains) != len(set(domains)):
        fail(
            "locations sharing a physical copy_domain must be combined into one capacity ledger "
            "(for example, same-filesystem /home and /tmp)"
        )
    aiboss = [item for item in locations.values() if item["role"] == "aiboss"]
    if len(aiboss) != 1 or aiboss[0]["kind"] != "filesystem":
        fail("locations must contain exactly one filesystem with role 'aiboss'")
    if aiboss[0]["available_bytes"] < AIBOSS_RESERVE_BYTES:
        fail("AIBoss measured available bytes are already below the mandatory 60-GiB reserve")

    upload = obj(document["upload"], "upload")
    keys(upload, {"source_location", "destination_location", "measured_bandwidth_bytes_per_second", "window_seconds"}, set(), "upload")
    upload_source = text(upload["source_location"], "upload.source_location")
    upload_destination = text(upload["destination_location"], "upload.destination_location")
    if upload_source not in locations or upload_destination not in locations:
        fail("upload source_location and destination_location must name declared locations")
    if locations[upload_destination]["kind"] != "object_storage":
        fail("upload.destination_location must be object_storage")
    upload_parsed = {
        "source_location": upload_source,
        "destination_location": upload_destination,
        "measured_bandwidth_bytes_per_second": integer(upload["measured_bandwidth_bytes_per_second"], "upload.measured_bandwidth_bytes_per_second", 1),
        "window_seconds": integer(upload["window_seconds"], "upload.window_seconds", 1),
    }

    artifacts: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    for index, raw_artifact in enumerate(array(document["artifacts"], "artifacts")):
        label = f"artifacts[{index}]"
        artifact = obj(raw_artifact, label)
        keys(
            artifact,
            {
                "id", "source_identity", "size", "multiplicity", "location", "branches",
                "birth_phase", "release_phase", "present_at_start", "retention",
                "atomic_temp", "checksum_staging", "upload_staging", "retry_reserve", "preservation",
            },
            set(),
            label,
        )
        artifact_id = text(artifact["id"], f"{label}.id")
        if artifact_id in artifact_ids:
            fail(f"duplicate artifact id {artifact_id!r}")
        artifact_ids.add(artifact_id)
        source_identity = obj(artifact["source_identity"], f"{label}.source_identity")
        if not source_identity:
            fail(f"{label}.source_identity must not be empty")
        reject_host_paths(source_identity, f"{label}.source_identity")
        size = obj(artifact["size"], f"{label}.size")
        keys(size, {"basis", "bytes"}, set(), f"{label}.size")
        basis = text(size["basis"], f"{label}.size.basis")
        if basis not in SIZE_BASES:
            fail(f"{label}.size.basis must be exact or upper_bound")
        size_bytes = integer(size["bytes"], f"{label}.size.bytes", 1)
        multiplicity = integer(artifact["multiplicity"], f"{label}.multiplicity", 1)
        location_name = text(artifact["location"], f"{label}.location")
        if location_name not in locations:
            fail(f"{label}.location names an unknown location")
        artifact_branches = [text(item, f"{label}.branches[{i}]") for i, item in enumerate(array(artifact["branches"], f"{label}.branches"))]
        if not artifact_branches or len(set(artifact_branches)) != len(artifact_branches) or not set(artifact_branches) <= branch_set:
            fail(f"{label}.branches must be a non-empty unique subset of branches")
        birth = phase_index(artifact["birth_phase"], f"{label}.birth_phase", phase_map)
        release = phase_index(artifact["release_phase"], f"{label}.release_phase", phase_map, nullable=True)
        assert birth is not None
        if release is not None and release <= birth:
            fail(f"{label}.release_phase must be after birth_phase")
        present = boolean(artifact["present_at_start"], f"{label}.present_at_start")
        if present and birth != 0:
            fail(f"{label}.present_at_start requires birth at the first phase")
        retention = text(artifact["retention"], f"{label}.retention")
        if retention not in RETENTIONS:
            fail(f"{label}.retention must be one of {sorted(RETENTIONS)}")
        if retention == "permanent" and release is not None:
            fail(f"{label} has permanent retention but a release_phase")

        components: dict[str, dict[str, Any]] = {}
        for component_name in COMPONENTS:
            raw_component = obj(artifact[component_name], f"{label}.{component_name}")
            amount_key = "multiplier" if component_name == "atomic_temp" else "copies"
            keys(raw_component, {"location", amount_key, "birth_phase", "release_phase"}, set(), f"{label}.{component_name}")
            component_location = text(raw_component["location"], f"{label}.{component_name}.location")
            if component_location not in locations:
                fail(f"{label}.{component_name}.location names an unknown location")
            amount = integer(raw_component[amount_key], f"{label}.{component_name}.{amount_key}", 1 if component_name == "atomic_temp" else 0)
            factor = amount - 1 if component_name == "atomic_temp" else amount
            component_birth, component_release = interval(raw_component, f"{label}.{component_name}", phase_map, zero=factor == 0)
            components[component_name] = {
                "location": component_location,
                "factor": factor,
                "birth": component_birth,
                "release": component_release,
                amount_key: amount,
            }

        preservation = obj(artifact["preservation"], f"{label}.preservation")
        keys(preservation, {"upload", "copies"}, set(), f"{label}.preservation")
        needs_upload = boolean(preservation["upload"], f"{label}.preservation.upload")
        copies: list[dict[str, Any]] = []
        copy_ids: set[str] = set()
        for copy_index, raw_copy in enumerate(array(preservation["copies"], f"{label}.preservation.copies")):
            copy_label = f"{label}.preservation.copies[{copy_index}]"
            copy = obj(raw_copy, copy_label)
            keys(
                copy,
                {
                    "id", "location", "verified_phase", "verification",
                    "content_binding", "storage_accounting",
                },
                set(),
                copy_label,
            )
            copy_id = text(copy["id"], f"{copy_label}.id")
            if copy_id in copy_ids:
                fail(f"{label} has duplicate copy id {copy_id!r}")
            copy_ids.add(copy_id)
            copy_location = text(copy["location"], f"{copy_label}.location")
            if copy_location not in locations:
                fail(f"{copy_label}.location names an unknown location")
            verification = text(copy["verification"], f"{copy_label}.verification")
            if verification != "independent_sha256":
                fail(f"{copy_label}.verification must be 'independent_sha256'")
            binding = text(copy["content_binding"], f"{copy_label}.content_binding")
            if binding != "artifact_sha256":
                fail(f"{copy_label}.content_binding must be 'artifact_sha256'")
            verified = phase_index(copy["verified_phase"], f"{copy_label}.verified_phase", phase_map)
            assert verified is not None
            if verified < birth:
                fail(f"{copy_label}.verified_phase precedes artifact birth")
            storage_accounting = text(copy["storage_accounting"], f"{copy_label}.storage_accounting")
            if storage_accounting not in {"base_artifact", "additional_copy"}:
                fail(f"{copy_label}.storage_accounting must be base_artifact or additional_copy")
            if storage_accounting == "base_artifact" and copy_location != location_name:
                fail(f"{copy_label} base_artifact accounting must use the artifact location")
            copies.append({
                "id": copy_id,
                "location": copy_location,
                "copy_domain": locations[copy_location]["copy_domain"],
                "verified": verified,
                "verification": verification,
                "content_binding": binding,
                "storage_accounting": storage_accounting,
            })
        if release is not None and retention != "ephemeral":
            verified_before = {copy["copy_domain"] for copy in copies if copy["verified"] is not None and copy["verified"] < release}
            if len(verified_before) < 2:
                fail(f"{label} releases before two independently verified physical-copy domains")
        if retention != "ephemeral" and len({copy["copy_domain"] for copy in copies}) < 2:
            fail(f"{label} retention requires two independently verified physical-copy domains")
        if needs_upload and not any(
            copy["location"] == upload_destination and copy["storage_accounting"] == "additional_copy"
            for copy in copies
        ):
            fail(f"{label} requests upload but has no additionally accounted copy at upload.destination_location")

        artifacts.append({
            "id": artifact_id,
            "source_identity_sha256": canonical_sha256(source_identity),
            "size": {"basis": basis, "bytes": size_bytes},
            "multiplicity": multiplicity,
            "location": location_name,
            "branches": artifact_branches,
            "birth": birth,
            "release": release,
            "present_at_start": present,
            "retention": retention,
            "components": components,
            "preservation": {"upload": needs_upload, "copies": copies},
        })
    if not artifacts:
        fail("artifacts must not be empty")

    return {
        "campaign_identity": identity,
        "phase_names": phase_names,
        "branches": branch_names,
        "locations": locations,
        "upload": upload_parsed,
        "artifacts": artifacts,
    }


def build_ledger(raw: object) -> dict[str, Any]:
    parsed = parse_document(raw)
    phases: list[str] = parsed["phase_names"]
    locations: dict[str, dict[str, Any]] = parsed["locations"]
    artifacts: list[dict[str, Any]] = parsed["artifacts"]
    upload = parsed["upload"]
    branch_rows: list[dict[str, Any]] = []
    global_peak_by_location: dict[str, dict[str, Any]] = {}
    global_incremental_peak_by_location: dict[str, dict[str, Any]] = {}

    for branch in parsed["branches"]:
        phase_rows: list[dict[str, Any]] = []
        branch_upload_bytes = 0
        for artifact in artifacts:
            if branch in artifact["branches"] and artifact["preservation"]["upload"]:
                branch_upload_bytes += artifact["size"]["bytes"] * artifact["multiplicity"]
        upload_capacity = upload["measured_bandwidth_bytes_per_second"] * upload["window_seconds"]
        if branch_upload_bytes > upload_capacity:
            fail(
                f"branch {branch!r} requires {branch_upload_bytes} upload bytes but measured bandwidth "
                f"can move only {upload_capacity} bytes in the window"
            )

        branch_peaks: dict[str, dict[str, Any]] = {}
        branch_incremental_peaks: dict[str, dict[str, Any]] = {}
        for phase_number, phase_name in enumerate(phases):
            totals: dict[str, dict[str, Any]] = {
                name: {
                    "exact_bytes": 0,
                    "upper_bound_bytes": 0,
                    "by_category": {category: {"exact_bytes": 0, "upper_bound_bytes": 0} for category in BYTE_CATEGORIES},
                    "incremental_upper_bound_bytes": 0,
                }
                for name in locations
            }
            live_artifact_ids: list[str] = []
            for artifact in artifacts:
                if branch not in artifact["branches"]:
                    continue
                basis = artifact["size"]["basis"]
                size_bytes = artifact["size"]["bytes"]
                multiplicity = artifact["multiplicity"]
                base_factor = multiplicity
                is_live = active(phase_number, artifact["birth"], artifact["release"])
                if is_live:
                    live_artifact_ids.append(artifact["id"])
                    exact, upper = byte_pair(size_bytes, basis, base_factor)
                    target = totals[artifact["location"]]
                    target["exact_bytes"] += exact
                    target["upper_bound_bytes"] += upper
                    target["by_category"]["live_artifacts"]["exact_bytes"] += exact
                    target["by_category"]["live_artifacts"]["upper_bound_bytes"] += upper
                if artifact["present_at_start"]:
                    baseline_upper = size_bytes * multiplicity
                    if not is_live:
                        totals[artifact["location"]]["incremental_upper_bound_bytes"] -= baseline_upper
                elif is_live:
                    totals[artifact["location"]]["incremental_upper_bound_bytes"] += size_bytes * multiplicity

                for component_name, component in artifact["components"].items():
                    if not active(phase_number, component["birth"], component["release"]):
                        continue
                    factor = multiplicity * component["factor"]
                    exact, upper = byte_pair(size_bytes, basis, factor)
                    target = totals[component["location"]]
                    target["exact_bytes"] += exact
                    target["upper_bound_bytes"] += upper
                    target["by_category"][component_name]["exact_bytes"] += exact
                    target["by_category"][component_name]["upper_bound_bytes"] += upper
                    target["incremental_upper_bound_bytes"] += upper

                for copy in artifact["preservation"]["copies"]:
                    if copy["storage_accounting"] != "additional_copy" or phase_number < copy["verified"]:
                        continue
                    exact, upper = byte_pair(size_bytes, basis, multiplicity)
                    target = totals[copy["location"]]
                    target["exact_bytes"] += exact
                    target["upper_bound_bytes"] += upper
                    target["by_category"]["retained_copies"]["exact_bytes"] += exact
                    target["by_category"]["retained_copies"]["upper_bound_bytes"] += upper
                    target["incremental_upper_bound_bytes"] += upper

            location_rows: list[dict[str, Any]] = []
            for name, total in totals.items():
                location = locations[name]
                reserve = AIBOSS_RESERVE_BYTES if location["role"] == "aiboss" else 0
                projected_available = location["available_bytes"] - total["incremental_upper_bound_bytes"]
                compliant = projected_available >= reserve
                if not compliant:
                    fail(
                        f"branch {branch!r} phase {phase_name!r} location {name!r} leaves "
                        f"{projected_available} bytes, below required reserve {reserve}"
                    )
                row = {
                    "location": name,
                    "exact_bytes": total["exact_bytes"],
                    "upper_bound_bytes": total["upper_bound_bytes"],
                    "by_category": total["by_category"],
                    "incremental_upper_bound_bytes": total["incremental_upper_bound_bytes"],
                    "projected_available_bytes": projected_available,
                    "required_reserve_bytes": reserve,
                    "reserve_compliant": compliant,
                }
                location_rows.append(row)
                prior = branch_peaks.get(name)
                if prior is None or row["upper_bound_bytes"] > prior["upper_bound_bytes"]:
                    branch_peaks[name] = {"phase": phase_name, **row}
                incremental_prior = branch_incremental_peaks.get(name)
                if (
                    incremental_prior is None
                    or row["incremental_upper_bound_bytes"] > incremental_prior["incremental_upper_bound_bytes"]
                ):
                    branch_incremental_peaks[name] = {"phase": phase_name, **row}
                global_prior = global_peak_by_location.get(name)
                if global_prior is None or row["upper_bound_bytes"] > global_prior["upper_bound_bytes"]:
                    global_peak_by_location[name] = {"branch": branch, "phase": phase_name, **row}
                global_incremental_prior = global_incremental_peak_by_location.get(name)
                if (
                    global_incremental_prior is None
                    or row["incremental_upper_bound_bytes"]
                    > global_incremental_prior["incremental_upper_bound_bytes"]
                ):
                    global_incremental_peak_by_location[name] = {
                        "branch": branch,
                        "phase": phase_name,
                        **row,
                    }
            phase_rows.append({
                "phase": phase_name,
                "live_artifact_ids": sorted(live_artifact_ids),
                "locations": location_rows,
                "aggregate_upper_bound_bytes": sum(item["upper_bound_bytes"] for item in location_rows),
            })
        branch_rows.append({
            "branch": branch,
            "upload": {
                "required_bytes": branch_upload_bytes,
                "capacity_bytes": upload_capacity,
                "minimum_seconds": math.ceil(branch_upload_bytes / upload["measured_bandwidth_bytes_per_second"]),
                "bandwidth_sufficient": True,
            },
            "phases": phase_rows,
            "peaks_by_location": [branch_peaks[name] for name in locations],
            "incremental_peaks_by_location": [branch_incremental_peaks[name] for name in locations],
            "aggregate_peak": max(
                ({"phase": row["phase"], "upper_bound_bytes": row["aggregate_upper_bound_bytes"]} for row in phase_rows),
                key=lambda item: item["upper_bound_bytes"],
            ),
        })

    artifact_rows = []
    for artifact in artifacts:
        artifact_rows.append({
            "id": artifact["id"],
            "source_identity_sha256": artifact["source_identity_sha256"],
            "size": artifact["size"],
            "multiplicity": artifact["multiplicity"],
            "location": artifact["location"],
            "branches": artifact["branches"],
            "birth_phase": phases[artifact["birth"]],
            "release_phase": None if artifact["release"] is None else phases[artifact["release"]],
            "present_at_start": artifact["present_at_start"],
            "retention": artifact["retention"],
            "components": {
                name: {
                    "location": component["location"],
                    ("multiplier" if name == "atomic_temp" else "copies"): component.get("multiplier", component.get("copies")),
                    "birth_phase": None if component["birth"] is None else phases[component["birth"]],
                    "release_phase": None if component["release"] is None else phases[component["release"]],
                }
                for name, component in artifact["components"].items()
            },
            "preservation": {
                "upload": artifact["preservation"]["upload"],
                "copies": [
                    {
                        "id": copy["id"],
                        "location": copy["location"],
                        "copy_domain": copy["copy_domain"],
                        "verified_phase": phases[copy["verified"]],
                        "verification": copy["verification"],
                        "content_binding": copy["content_binding"],
                        "storage_accounting": copy["storage_accounting"],
                    }
                    for copy in artifact["preservation"]["copies"]
                ],
            },
        })

    return {
        "schema": OUTPUT_SCHEMA,
        "input_schema": INPUT_SCHEMA,
        "input_sha256": canonical_sha256(raw),
        "campaign_identity_sha256": canonical_sha256(parsed["campaign_identity"]),
        "accounting": {
            "birth_inclusive_release_exclusive": True,
            "size_bases": ["exact", "upper_bound"],
            "aiboss_reserve_bytes": AIBOSS_RESERVE_BYTES,
            "atomic_temp_multiplier_includes_live_output": True,
            "copy_rule": "release requires two distinct copy_domain values verified by independent_sha256 before release",
        },
        "locations": list(locations.values()),
        "upload": upload,
        "artifacts": artifact_rows,
        "branches": branch_rows,
        "global_peaks_by_location": [global_peak_by_location[name] for name in locations],
        "global_incremental_peaks_by_location": [
            global_incremental_peak_by_location[name] for name in locations
        ],
        "global_aggregate_peak": max(
            (
                {
                    "branch": branch["branch"],
                    "phase": phase["phase"],
                    "upper_bound_bytes": phase["aggregate_upper_bound_bytes"],
                }
                for branch in branch_rows
                for phase in branch["phases"]
            ),
            key=lambda item: item["upper_bound_bytes"],
        ),
        "all_reserves_and_bandwidth_satisfied": True,
    }


def output_path(value: str) -> Path:
    path = Path(value)
    if path.exists() and path.stat().st_size:
        fail(f"refusing to overwrite nonempty output artifact {path}")
    if path.is_dir():
        fail(f"output path is a directory: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help=f"strict {INPUT_SCHEMA} JSON plan")
    parser.add_argument("--output", required=True, help="new ledger JSON path")
    args = parser.parse_args(argv)
    try:
        source = Path(args.input)
        if not source.is_file() or source.stat().st_size == 0:
            fail(f"input artifact is missing, non-file, or empty: {source}")
        raw = load_strict_json(source)
        result = build_ledger(raw)
        atomic_write_json(output_path(args.output), result)
    except (OSError, ValueError) as exc:
        print(f"frontier_resources: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
