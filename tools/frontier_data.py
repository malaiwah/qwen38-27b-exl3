#!/usr/bin/env python3
"""Build and verify the pre-tokenization Final Frontier master data ledger."""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlparse

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json

SPEC_SCHEMA = "qwen38-frontier-data-spec/1"
LEDGER_SCHEMA = "qwen38-frontier-data-ledger/1"
ROLES = (
    "c1_text",
    "c1_mm",
    "s_wave",
    "d_v6",
    "e_final_v7",
    "capability_panel",
)
EDGE_KINDS = (
    "exact_bytes",
    "normalized_12word_minhash",
    "code_clone",
    "translation_task_family",
    "template_aware_tool",
)
UNKNOWN = {"", "unknown", "unset", "unresolved", "n/a", "na", "none", "null", "tbd"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
MINHASH_RE = re.compile(r"^[0-9a-f]{16}$")
TRANSFORMS = {
    "annotate",
    "crop",
    "deduplicate",
    "filter",
    "format_convert",
    "normalize_unicode",
    "redact_pii",
    "resize",
    "tokenize",
    "transcode_lossless",
    "transcode_lossy",
    "translate",
    "whitespace_normalize",
}
REDISTRIBUTION_POLICIES = {"allowed", "allowed_with_attribution", "prohibited", "restricted"}
DERIVATIVE_POLICIES = {"allowed", "allowed_with_conditions", "prohibited", "restricted"}
ACCESS_CLASSES = {"public", "authenticated", "gated", "restricted", "sealed"}
RAW_STATES = {"available_immutable", "outside_git_unopened"}


class DataError(ValueError):
    """Closed validation failure for a data-ledger input or artifact."""


def fail(message: str) -> NoReturn:
    raise DataError(message)


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
    extra = sorted(value.keys() - required)
    if missing or extra:
        fail(f"{label} keys differ from schema; missing={missing}, unknown={extra}")


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() in UNKNOWN:
        fail(f"{label} must be a known non-empty string")
    return value.strip()


def enum(value: object, choices: set[str] | tuple[str, ...], label: str) -> str:
    parsed = text(value, label)
    if parsed not in choices:
        fail(f"{label} must be one of {sorted(choices)}, got {parsed!r}")
    return parsed


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < 0 or (positive and value == 0):
        fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{label} must be a finite number")
    return float(value)


def sha256(value: object, label: str) -> str:
    parsed = text(value, label).lower()
    if SHA256_RE.fullmatch(parsed) is None:
        fail(f"{label} must be a lowercase 64-character SHA256")
    return parsed


def revision(value: object, label: str) -> str:
    parsed = text(value, label).lower()
    if REVISION_RE.fullmatch(parsed) is None:
        fail(f"{label} must be an immutable 40- or 64-character hexadecimal revision")
    return parsed


def nullable_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return text(value, label)


def identity(value: object, label: str) -> dict[str, str]:
    document = obj(value, label)
    exact_keys(document, {"repo", "revision", "config_sha256"}, label)
    return {
        "repo": text(document["repo"], f"{label}.repo"),
        "revision": revision(document["revision"], f"{label}.revision"),
        "config_sha256": sha256(document["config_sha256"], f"{label}.config_sha256"),
    }


def validate_url(value: object, label: str) -> str:
    parsed = text(value, label)
    url = urlparse(parsed)
    if url.scheme not in {"https", "http"} or not url.netloc or url.username or url.password:
        fail(f"{label} must be an absolute HTTP(S) URL without embedded credentials")
    return parsed


def validate_source(value: object, label: str) -> dict[str, Any]:
    source = obj(value, label)
    exact_keys(
        source,
        {
            "dataset",
            "revision",
            "config",
            "split",
            "row_identity",
            "media_identity",
            "url",
            "raw_sha256",
            "file_sha256",
            "raw_bytes_state",
        },
        label,
    )
    return {
        "dataset": text(source["dataset"], f"{label}.dataset"),
        "revision": revision(source["revision"], f"{label}.revision"),
        "config": text(source["config"], f"{label}.config"),
        "split": text(source["split"], f"{label}.split"),
        "row_identity": text(source["row_identity"], f"{label}.row_identity"),
        "media_identity": sha256(source["media_identity"], f"{label}.media_identity"),
        "url": validate_url(source["url"], f"{label}.url"),
        "raw_sha256": sha256(source["raw_sha256"], f"{label}.raw_sha256"),
        "file_sha256": sha256(source["file_sha256"], f"{label}.file_sha256"),
        "raw_bytes_state": enum(source["raw_bytes_state"], RAW_STATES, f"{label}.raw_bytes_state"),
    }


def validate_rights(value: object, label: str) -> dict[str, Any]:
    rights = obj(value, label)
    exact_keys(
        rights,
        {
            "license",
            "attribution",
            "permitted_transforms",
            "redistribution_policy",
            "derivative_policy",
            "access_class",
            "verification_sha256",
        },
        label,
    )
    license_id = text(rights["license"], f"{label}.license")
    attribution = text(rights["attribution"], f"{label}.attribution")
    transforms_raw = array(rights["permitted_transforms"], f"{label}.permitted_transforms")
    if not transforms_raw:
        fail(f"{label}.permitted_transforms must not be empty")
    transforms = [enum(entry, TRANSFORMS, f"{label}.permitted_transforms[{index}]") for index, entry in enumerate(transforms_raw)]
    if len(transforms) != len(set(transforms)):
        fail(f"{label}.permitted_transforms contains duplicates")
    return {
        "license": license_id,
        "attribution": attribution,
        "permitted_transforms": sorted(transforms),
        "redistribution_policy": enum(rights["redistribution_policy"], REDISTRIBUTION_POLICIES, f"{label}.redistribution_policy"),
        "derivative_policy": enum(rights["derivative_policy"], DERIVATIVE_POLICIES, f"{label}.derivative_policy"),
        "access_class": enum(rights["access_class"], ACCESS_CLASSES, f"{label}.access_class"),
        "verification_sha256": sha256(rights["verification_sha256"], f"{label}.verification_sha256"),
    }


def validate_dedup(value: object, permutations: int, label: str) -> dict[str, Any]:
    dedup = obj(value, label)
    exact_keys(
        dedup,
        {
            "exact_bytes_sha256",
            "normalized_12word_minhash",
            "code_clone_hashes",
            "translation_family",
            "task_family",
            "tool",
            "evidence_sha256",
        },
        label,
    )
    signature_raw = array(dedup["normalized_12word_minhash"], f"{label}.normalized_12word_minhash")
    if len(signature_raw) != permutations:
        fail(f"{label}.normalized_12word_minhash must contain exactly {permutations} values")
    signature: list[str] = []
    for index, raw in enumerate(signature_raw):
        value = text(raw, f"{label}.normalized_12word_minhash[{index}]").lower()
        if MINHASH_RE.fullmatch(value) is None:
            fail(f"{label}.normalized_12word_minhash[{index}] must be 16 lowercase hexadecimal digits")
        signature.append(value)
    clones = [sha256(raw, f"{label}.code_clone_hashes[{index}]") for index, raw in enumerate(array(dedup["code_clone_hashes"], f"{label}.code_clone_hashes"))]
    if len(clones) != len(set(clones)):
        fail(f"{label}.code_clone_hashes contains duplicates")
    tool_raw = dedup["tool"]
    tool: dict[str, str] | None
    if tool_raw is None:
        tool = None
    else:
        tool_obj = obj(tool_raw, f"{label}.tool")
        exact_keys(tool_obj, {"template_sha256", "signature_sha256"}, f"{label}.tool")
        tool = {
            "template_sha256": sha256(tool_obj["template_sha256"], f"{label}.tool.template_sha256"),
            "signature_sha256": sha256(tool_obj["signature_sha256"], f"{label}.tool.signature_sha256"),
        }
    return {
        "exact_bytes_sha256": sha256(dedup["exact_bytes_sha256"], f"{label}.exact_bytes_sha256"),
        "normalized_12word_minhash": signature,
        "code_clone_hashes": sorted(clones),
        "translation_family": nullable_text(dedup["translation_family"], f"{label}.translation_family"),
        "task_family": nullable_text(dedup["task_family"], f"{label}.task_family"),
        "tool": tool,
        "evidence_sha256": sha256(dedup["evidence_sha256"], f"{label}.evidence_sha256"),
    }


def validate_item(value: object, permutations: int, waves: set[str], label: str) -> dict[str, Any]:
    item = obj(value, label)
    exact_keys(item, {"id", "collection", "role", "wave", "source", "rights", "processing", "source_family", "dedup"}, label)
    item_id = text(item["id"], f"{label}.id")
    collection = enum(item["collection"], ROLES, f"{label}.collection")
    role = enum(item["role"], ROLES, f"{label}.role")
    if collection != role:
        fail(f"{label} collection and pre-tokenization role must be identical")
    wave = nullable_text(item["wave"], f"{label}.wave")
    if collection == "s_wave":
        if wave not in waves:
            fail(f"{label}.wave must name one of the finite declared S waves")
    elif wave is not None:
        fail(f"{label}.wave is only permitted for s_wave items")
    source = validate_source(item["source"], f"{label}.source")
    rights = validate_rights(item["rights"], f"{label}.rights")
    if collection == "e_final_v7":
        if source["raw_bytes_state"] != "outside_git_unopened" or rights["access_class"] != "sealed":
            fail(f"{label} E-final-v7 bytes must be sealed and outside_git_unopened")
    elif source["raw_bytes_state"] != "available_immutable":
        fail(f"{label} non-E-final raw bytes must be available_immutable")
    dedup = validate_dedup(item["dedup"], permutations, f"{label}.dedup")
    if dedup["exact_bytes_sha256"] != source["raw_sha256"]:
        fail(f"{label}.dedup.exact_bytes_sha256 must equal source.raw_sha256")
    processing = obj(item["processing"], f"{label}.processing")
    exact_keys(processing, {"processor", "template", "tokenizer"}, f"{label}.processing")
    return {
        "id": item_id,
        "collection": collection,
        "role": role,
        "wave": wave,
        "source": source,
        "rights": rights,
        "processing": {
            "processor": identity(processing["processor"], f"{label}.processing.processor"),
            "template": identity(processing["template"], f"{label}.processing.template"),
            "tokenizer": identity(processing["tokenizer"], f"{label}.processing.tokenizer"),
        },
        "source_family": text(item["source_family"], f"{label}.source_family"),
        "dedup": dedup,
    }


def pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def jaccard(left: list[str], right: list[str]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def compare_items(left: dict[str, Any], right: dict[str, Any], minhash_threshold: float, clone_threshold: float) -> list[dict[str, Any]]:
    pair = [left["id"], right["id"]]
    evidence = canonical_sha256({"left": left["dedup"], "right": right["dedup"]})
    results: list[dict[str, Any]] = []

    exact_score = 1.0 if left["dedup"]["exact_bytes_sha256"] == right["dedup"]["exact_bytes_sha256"] else 0.0
    results.append({"items": pair, "kind": "exact_bytes", "score": exact_score, "threshold": 1.0, "evidence_sha256": evidence})

    lhs = left["dedup"]["normalized_12word_minhash"]
    rhs = right["dedup"]["normalized_12word_minhash"]
    minhash_score = sum(a == b for a, b in zip(lhs, rhs)) / len(lhs)
    results.append({"items": pair, "kind": "normalized_12word_minhash", "score": minhash_score, "threshold": minhash_threshold, "evidence_sha256": evidence})

    clone_score = jaccard(left["dedup"]["code_clone_hashes"], right["dedup"]["code_clone_hashes"])
    results.append({"items": pair, "kind": "code_clone", "score": clone_score, "threshold": clone_threshold, "evidence_sha256": evidence})

    left_translation = left["dedup"]["translation_family"]
    right_translation = right["dedup"]["translation_family"]
    left_task = left["dedup"]["task_family"]
    right_task = right["dedup"]["task_family"]
    translation_match = left_translation is not None and left_translation == right_translation
    task_match = left_task is not None and left_task == right_task
    family_score = 1.0 if translation_match else (0.5 if task_match else 0.0)
    results.append({"items": pair, "kind": "translation_task_family", "score": family_score, "threshold": 0.5, "evidence_sha256": evidence})

    left_tool = left["dedup"]["tool"]
    right_tool = right["dedup"]["tool"]
    if left_tool is None or right_tool is None:
        tool_score = 0.0
    elif left_tool == right_tool:
        tool_score = 1.0
    elif left_tool["signature_sha256"] == right_tool["signature_sha256"]:
        tool_score = 0.5
    else:
        tool_score = 0.0
    results.append({"items": pair, "kind": "template_aware_tool", "score": tool_score, "threshold": 1.0, "evidence_sha256": evidence})
    return results


class UnionFind:
    def __init__(self, members: list[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        while self.parent[member] != member:
            self.parent[member] = self.parent[self.parent[member]]
            member = self.parent[member]
        return member

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def validate_policies(value: object) -> dict[str, Any]:
    policies = obj(value, "policies")
    exact_keys(
        policies,
        {"roles", "s_waves", "minhash_permutations", "minhash_edge_threshold", "code_clone_edge_threshold", "family_caps", "disjoint_role_sets"},
        "policies",
    )
    roles = [enum(raw, ROLES, f"policies.roles[{index}]") for index, raw in enumerate(array(policies["roles"], "policies.roles"))]
    if roles != list(ROLES):
        fail(f"policies.roles must be the canonical complete role list {list(ROLES)}")
    waves = [text(raw, f"policies.s_waves[{index}]") for index, raw in enumerate(array(policies["s_waves"], "policies.s_waves"))]
    if not waves or len(waves) != len(set(waves)):
        fail("policies.s_waves must be a non-empty finite list without duplicates")
    permutations = integer(policies["minhash_permutations"], "policies.minhash_permutations", positive=True)
    if permutations < 16:
        fail("policies.minhash_permutations must be at least 16")
    minhash_threshold = number(policies["minhash_edge_threshold"], "policies.minhash_edge_threshold")
    clone_threshold = number(policies["code_clone_edge_threshold"], "policies.code_clone_edge_threshold")
    if not 0 < minhash_threshold <= 1 or not 0 < clone_threshold <= 1:
        fail("dedup edge thresholds must be in (0, 1]")
    caps_obj = obj(policies["family_caps"], "policies.family_caps")
    if set(caps_obj) != set(ROLES):
        fail("policies.family_caps must define every and only canonical role")
    caps = {role: integer(caps_obj[role], f"policies.family_caps.{role}", positive=True) for role in ROLES}
    disjoint: list[list[str]] = []
    for index, raw_group in enumerate(array(policies["disjoint_role_sets"], "policies.disjoint_role_sets")):
        group = [enum(raw, ROLES, f"policies.disjoint_role_sets[{index}]") for raw in array(raw_group, f"policies.disjoint_role_sets[{index}]")]
        if len(group) < 2 or len(group) != len(set(group)):
            fail(f"policies.disjoint_role_sets[{index}] must contain at least two distinct roles")
        disjoint.append(sorted(group))
    if not disjoint:
        fail("policies.disjoint_role_sets must not be empty")
    return {
        "roles": roles,
        "s_waves": waves,
        "minhash_permutations": permutations,
        "minhash_edge_threshold": minhash_threshold,
        "code_clone_edge_threshold": clone_threshold,
        "family_caps": caps,
        "disjoint_role_sets": sorted(disjoint),
    }


def validate_reviews(value: object, item_ids: set[str]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(array(value, "match_reviews")):
        label = f"match_reviews[{index}]"
        review = obj(raw, label)
        exact_keys(review, {"left", "right", "kind", "status", "score", "evidence_sha256", "reason"}, label)
        left = text(review["left"], f"{label}.left")
        right = text(review["right"], f"{label}.right")
        if left not in item_ids or right not in item_ids or left == right:
            fail(f"{label} must reference two distinct known item ids")
        left, right = pair_key(left, right)
        kind = enum(review["kind"], EDGE_KINDS, f"{label}.kind")
        status = enum(review["status"], {"edge", "excluded", "unresolved"}, f"{label}.status")
        if status == "unresolved":
            fail(f"{label} is unresolved; all candidate matches must be adjudicated before build")
        score = number(review["score"], f"{label}.score")
        if not 0 <= score <= 1:
            fail(f"{label}.score must be in [0, 1]")
        key = (left, right, kind)
        if key in seen:
            fail(f"duplicate match review for {key}")
        seen.add(key)
        reviews.append({
            "items": [left, right],
            "kind": kind,
            "status": status,
            "score": score,
            "evidence_sha256": sha256(review["evidence_sha256"], f"{label}.evidence_sha256"),
            "reason": text(review["reason"], f"{label}.reason"),
        })
    return sorted(reviews, key=lambda entry: (entry["items"], entry["kind"]))


def construct(spec_value: object) -> dict[str, Any]:
    spec = obj(spec_value, "spec")
    exact_keys(spec, {"schema", "seed", "policies", "items", "match_reviews"}, "spec")
    if spec["schema"] != SPEC_SCHEMA:
        fail(f"spec.schema must be {SPEC_SCHEMA!r}")
    seed = integer(spec["seed"], "spec.seed", positive=True)
    policies = validate_policies(spec["policies"])
    items = [validate_item(raw, policies["minhash_permutations"], set(policies["s_waves"]), f"items[{index}]") for index, raw in enumerate(array(spec["items"], "items"))]
    if not items:
        fail("items must not be empty")
    observed_roles = {item["role"] for item in items}
    if observed_roles != set(ROLES):
        fail(f"items must cover every campaign role; missing={sorted(set(ROLES) - observed_roles)}")
    observed_waves = {item["wave"] for item in items if item["role"] == "s_wave"}
    if observed_waves != set(policies["s_waves"]):
        fail("items must cover every and only declared finite S wave")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        fail("item ids must be unique")
    immutable_rows = [
        (item["source"]["dataset"], item["source"]["revision"], item["source"]["config"], item["source"]["split"], item["source"]["row_identity"], item["source"]["media_identity"])
        for item in items
    ]
    if len(immutable_rows) != len(set(immutable_rows)):
        fail("immutable source/config/split/row/media identities must be unique")
    items.sort(key=lambda item: item["id"])
    item_by_id = {item["id"]: item for item in items}
    reviews = validate_reviews(spec["match_reviews"], set(ids))

    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            comparisons.extend(compare_items(left, right, policies["minhash_edge_threshold"], policies["code_clone_edge_threshold"]))
    comparison_by_key = {(entry["items"][0], entry["items"][1], entry["kind"]): entry for entry in comparisons}
    for review in reviews:
        key = (review["items"][0], review["items"][1], review["kind"])
        computed = comparison_by_key[key]
        computed_status = "edge" if computed["score"] >= computed["threshold"] else "excluded"
        if review["status"] != computed_status or abs(review["score"] - computed["score"]) > 1e-12:
            fail(f"match review {key} disagrees with deterministic score/status")

    edges: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    uf = UnionFind(ids)
    for entry in comparisons:
        published = dict(entry)
        if entry["score"] >= entry["threshold"]:
            published["decision"] = "edge"
            edges.append(published)
            uf.union(entry["items"][0], entry["items"][1])
        else:
            published["decision"] = "excluded"
            published["reason"] = "score_below_fixed_threshold"
            exclusions.append(published)

    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        grouped[uf.find(item_id)].append(item_id)
    components: list[dict[str, Any]] = []
    for members in sorted((sorted(group) for group in grouped.values()), key=lambda group: group[0]):
        roles = {item_by_id[member]["role"] for member in members}
        waves = {item_by_id[member]["wave"] for member in members if item_by_id[member]["role"] == "s_wave"}
        if len(roles) != 1 or len(waves) > 1:
            fail(f"connected component {members} crosses a pre-tokenization role or S-wave boundary")
        role = next(iter(roles))
        components.append({
            "component_id": canonical_sha256(members),
            "members": members,
            "role": role,
            "wave": next(iter(waves)) if waves else None,
            "source_families": sorted({item_by_id[member]["source_family"] for member in members}),
        })

    cap_counts: Counter[tuple[str, str | None, str]] = Counter()
    family_roles: dict[str, set[str]] = defaultdict(set)
    for item in items:
        cap_counts[(item["role"], item["wave"], item["source_family"])] += 1
        family_roles[item["source_family"]].add(item["role"])
    for (role, wave, family), count in sorted(cap_counts.items()):
        if count > policies["family_caps"][role]:
            suffix = f" wave={wave}" if wave is not None else ""
            fail(f"source family {family!r} exceeds {role}{suffix} cap: {count} > {policies['family_caps'][role]}")
    for family, assigned_roles in sorted(family_roles.items()):
        for group in policies["disjoint_role_sets"]:
            overlap = assigned_roles & set(group)
            if len(overlap) > 1:
                fail(f"source family {family!r} overlaps disjoint roles {sorted(overlap)}")

    item_records = []
    for item in items:
        record = dict(item)
        record["component_id"] = canonical_sha256(sorted(grouped[uf.find(item["id"])]))
        item_records.append(record)
    edges.sort(key=lambda entry: (entry["items"], entry["kind"]))
    exclusions.sort(key=lambda entry: (entry["items"], entry["kind"]))
    return {
        "schema": LEDGER_SCHEMA,
        "source": {"spec_schema": SPEC_SCHEMA, "spec_sha256": canonical_sha256(spec), "seed": seed},
        "policy": policies,
        "summary": {
            "items": len(items),
            "components": len(components),
            "edges": len(edges),
            "exclusions": len(exclusions),
            "unresolved_matches": 0,
            "role_counts": dict(sorted(Counter(item["role"] for item in items).items())),
            "source_family_counts": dict(sorted(Counter(item["source_family"] for item in items).items())),
        },
        "items": item_records,
        "components": components,
        "edges": edges,
        "exclusions": exclusions,
        "reviewed_matches": reviews,
        "unresolved_matches": [],
    }


def load_nonempty(path: Path, label: str) -> object:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"{label} must be an existing non-empty file: {path}")
        return load_strict_json(path)
    except OSError as exc:
        fail(f"cannot inspect {label} {path}: {exc}")
    raise AssertionError("unreachable")


def require_empty_output(path: Path) -> None:
    try:
        if path.exists() and (not path.is_file() or path.stat().st_size != 0):
            fail(f"refusing to replace non-empty output artifact: {path}")
    except OSError as exc:
        fail(f"cannot inspect output {path}: {exc}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build or verify the Final Frontier pre-tokenization master ledger.")
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="validate a locked metadata spec and write a new ledger")
    build.add_argument("--spec", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="reconstruct and byte-semantically verify an existing ledger")
    verify.add_argument("--spec", required=True, type=Path)
    verify.add_argument("--ledger", required=True, type=Path)
    return result


def run(args: argparse.Namespace) -> None:
    expected = construct(load_nonempty(args.spec, "spec"))
    if args.command == "build":
        require_empty_output(args.out)
        atomic_write_json(args.out, expected)
        return
    actual = load_nonempty(args.ledger, "ledger")
    if not isinstance(actual, dict) or actual.get("schema") != LEDGER_SCHEMA:
        fail(f"ledger.schema must be {LEDGER_SCHEMA!r}")
    if actual != expected:
        fail("ledger does not exactly reproduce from the locked spec")
    if canonical_sha256(actual) != canonical_sha256(expected):
        fail("ledger canonical SHA256 verification failed")


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except (DataError, OSError, ValueError) as exc:
        print(f"frontier_data: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
