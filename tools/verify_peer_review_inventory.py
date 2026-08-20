#!/usr/bin/env python3
"""Verify the retrospective peer-review ledger against the current repository.

The checker is intentionally CPU-only and dependency-free. It re-enumerates every
repository file, verifies byte digests, checks external-card identity attestations,
and requires every finding to appear in the final review with a terminal review
status. The ledger itself is the sole self-exclusion because hashing a file inside
itself has no finite fixed point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "qwen38-retrospective-peer-review-ledger/1"
TERMINAL_FINDING_STATUSES = {
    "fixed",
    "later-resolved",
    "dismissed",
    "blocked",
    "open",
}
IGNORED_PARTS = {".git", ".venv", "__pycache__"}
MODEL_HINTS = (
    "qwen",
    "glm",
    "gemma",
    "dflash",
    "dspark",
    "speculator",
    "nvfp",
    "gguf",
    "exl",
    "awq",
    "gptq",
    "paro",
    "warpquant",
)
NON_MODEL_IDS = {
    "malaiwah/qwen38-27b-exl3",
    "malaiwah/qwen38-27b-fidelity-suite-v3",
    "malaiwah/qwen38-27b-fidelity-suite-v5",
    "malaiwah/qwen38-27b-terminal-bench-2.1",
    "z-lab/dflash",
}
HF_URL_RE = re.compile(
    r"https?://(?:www\.)?huggingface\.co/"
    r"(?!datasets/|collections/|spaces/|api/)"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
MODEL_ID_RE = re.compile(
    r"(?<![\w/])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@[A-Fa-f0-9.-]+)?"
)
FINDING_RE = re.compile(r"\bF(\d{3})\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files(root: Path, ledger_rel: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {
        "documents": [],
        "receipts": [],
        "supporting_files": [],
        "publication_assets": [],
        "other_files": [],
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        if rel == ledger_rel:
            continue
        if rel.startswith("receipts/"):
            groups["receipts"].append(path)
        elif path.suffix.lower() == ".md":
            groups["documents"].append(path)
        elif rel.startswith(("assets/", "charts/")):
            groups["publication_assets"].append(path)
        elif rel.startswith(("tools/", "patches/", "docker/", "upstream/", ".omp/")) or rel == "requirements.txt":
            groups["supporting_files"].append(path)
        else:
            groups["other_files"].append(path)
    return groups


def load_records(ledger: dict[str, Any], key: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    raw = ledger.get("repository", {}).get(key)
    if not isinstance(raw, list):
        errors.append(f"repository.{key} must be a list")
        return {}
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append(f"repository.{key}[{index}] has no string path")
            continue
        path = record["path"]
        if path in records:
            errors.append(f"repository.{key} duplicates {path}")
        records[path] = record
    return records


def verify_repository(root: Path, ledger: dict[str, Any], ledger_rel: str, errors: list[str]) -> dict[str, int]:
    actual = repository_files(root, ledger_rel)
    counts: dict[str, int] = {}
    for key, paths in actual.items():
        records = load_records(ledger, key, errors)
        actual_rel = {path.relative_to(root).as_posix() for path in paths}
        recorded_rel = set(records)
        missing = sorted(actual_rel - recorded_rel)
        stale = sorted(recorded_rel - actual_rel)
        if missing:
            errors.append(f"repository.{key} missing {len(missing)} paths: {missing[:8]}")
        if stale:
            errors.append(f"repository.{key} has {len(stale)} stale paths: {stale[:8]}")
        for path in paths:
            rel = path.relative_to(root).as_posix()
            record = records.get(rel)
            if record is None:
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            if record.get("bytes") != size:
                errors.append(f"{rel}: bytes ledger={record.get('bytes')} current={size}")
            if record.get("sha256") != digest:
                errors.append(f"{rel}: stale sha256")
            if record.get("reviewed") is not True:
                errors.append(f"{rel}: reviewed must be true")
            evidence = record.get("review_evidence")
            if not isinstance(evidence, str) or len(evidence.strip()) < 8:
                errors.append(f"{rel}: substantive review_evidence is required")
            if key == "receipts":
                parse = record.get("parse_status")
                if parse not in {
                    "json",
                    "jsonl",
                    "utf8",
                    "gzip",
                    "binary",
                    "invalid-json-documented",
                    "empty-documented",
                }:
                    errors.append(f"{rel}: unsupported receipt parse_status {parse!r}")
        counts[key] = len(paths)
    return counts


def discover_model_ids(root: Path, owner_allowlist: set[str]) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.md"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        found.update(HF_URL_RE.findall(text))
        for candidate in MODEL_ID_RE.findall(text):
            owner, repo = candidate.split("/", 1)
            if candidate.endswith("-") or candidate in NON_MODEL_IDS:
                continue
            if owner in owner_allowlist and any(hint in repo.lower() for hint in MODEL_HINTS):
                found.add(candidate)
    landscape = root / "receipts" / "quant-landscape-scan.json"
    if landscape.exists():
        payload = json.loads(landscape.read_text(encoding="utf-8"))
        for candidate in payload.get("curated_classification", {}):
            if "*" not in candidate:
                found.add(candidate)
    # docs/58 compresses these nine card ids with brace notation in prose.
    found.update(
        f"UnstableLlama/Qwen3.6-27B-exl3-{bpw}bpw"
        for bpw in ("2.06", "6.00", "8.00")
    )
    found.update(
        f"UnstableLlama/Qwopus3.6-27B-v2-exl3-{bpw}bpw"
        for bpw in ("2.50", "2.90", "3.08", "4.15", "6.00", "8.00")
    )
    return found


def verify_external_cards(root: Path, ledger: dict[str, Any], errors: list[str]) -> int:
    cards = ledger.get("external_model_cards")
    if not isinstance(cards, list):
        errors.append("external_model_cards must be a list")
        return 0
    by_id: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(cards):
        if not isinstance(card, dict) or not isinstance(card.get("id"), str):
            errors.append(f"external_model_cards[{index}] has no id")
            continue
        model_id = card["id"]
        if model_id in by_id:
            errors.append(f"external_model_cards duplicates {model_id}")
        by_id[model_id] = card
        if card.get("reviewed") is not True:
            errors.append(f"external card {model_id}: reviewed must be true")
        if not isinstance(card.get("review_evidence"), str) or len(card["review_evidence"].strip()) < 8:
            errors.append(f"external card {model_id}: review_evidence required")
        status = card.get("status")
        if status == "accessible":
            if not re.fullmatch(r"[0-9a-f]{40}", str(card.get("revision", ""))):
                errors.append(f"external card {model_id}: accessible card needs 40-hex revision")
            if not re.fullmatch(r"[0-9a-f]{64}", str(card.get("readme_sha256", ""))):
                errors.append(f"external card {model_id}: accessible card needs README sha256")
        elif status not in {"no-readme", "unavailable", "not-a-model-card"}:
            errors.append(f"external card {model_id}: invalid status {status!r}")
    owners = ledger.get("scope", {}).get("external_model_owner_allowlist")
    if not isinstance(owners, list) or not all(isinstance(x, str) for x in owners):
        errors.append("scope.external_model_owner_allowlist must be a string list")
        owners = []
    discovered = discover_model_ids(root, set(owners))
    missing = sorted(discovered - set(by_id))
    if missing:
        errors.append(f"external_model_cards misses {len(missing)} discovered ids: {missing[:12]}")
    return len(cards)


def verify_findings(root: Path, ledger: dict[str, Any], review_rel: str, errors: list[str]) -> int:
    findings = ledger.get("findings")
    if not isinstance(findings, list) or not findings:
        errors.append("findings must be a non-empty list")
        return 0
    ids: list[str] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            errors.append(f"findings[{index}] is not an object")
            continue
        finding_id = finding.get("id")
        if not isinstance(finding_id, str) or not re.fullmatch(r"F\d{3}", finding_id):
            errors.append(f"findings[{index}] has invalid id {finding_id!r}")
            continue
        ids.append(finding_id)
        if finding.get("status") not in TERMINAL_FINDING_STATUSES:
            errors.append(f"{finding_id}: invalid terminal status {finding.get('status')!r}")
        if finding.get("severity") not in {"critical", "high", "medium", "low", "note"}:
            errors.append(f"{finding_id}: invalid severity")
        if not isinstance(finding.get("resolution"), str) or len(finding["resolution"].strip()) < 8:
            errors.append(f"{finding_id}: resolution required")
    if len(ids) != len(set(ids)):
        errors.append("findings contain duplicate ids")
    numbers = sorted(int(item[1:]) for item in set(ids))
    if numbers and numbers != list(range(1, max(numbers) + 1)):
        errors.append("finding ids must be contiguous from F001 through the maximum id")
    review_path = root / review_rel
    if not review_path.is_file():
        errors.append(f"final review missing: {review_rel}")
        return len(ids)
    review_text = review_path.read_text(encoding="utf-8")
    mentioned = {f"F{int(n):03d}" for n in FINDING_RE.findall(review_text)}
    absent = sorted(set(ids) - mentioned)
    if absent:
        errors.append(f"final review omits {len(absent)} findings: {absent[:12]}")
    return len(ids)


def verify_upstream_items(ledger: dict[str, Any], errors: list[str]) -> int:
    items = ledger.get("upstream_items")
    if not isinstance(items, list):
        errors.append("upstream_items must be a list")
        return 0
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            errors.append(f"upstream_items[{index}] has no URL")
            continue
        url = item["url"]
        if url in seen:
            errors.append(f"upstream_items duplicates {url}")
        seen.add(url)
        if item.get("reviewed") is not True or not isinstance(item.get("verdict"), str):
            errors.append(f"upstream item {url}: reviewed/verdict required")
    return len(items)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--ledger", default="docs/60-peer-review-ledger.json")
    parser.add_argument("--review", default="docs/60-retrospective-peer-review.md")
    args = parser.parse_args()
    root = args.root.resolve()
    ledger_path = root / args.ledger
    errors: list[str] = []
    if not ledger_path.is_file():
        print(f"ERROR: ledger missing: {ledger_path}", file=sys.stderr)
        return 1
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot parse ledger: {exc}", file=sys.stderr)
        return 1
    if ledger.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if ledger.get("self_exclusion") != {
        "path": args.ledger,
        "reason": "self-digest has no finite fixed point",
    }:
        errors.append("ledger self_exclusion is missing or changed")
    counts = verify_repository(root, ledger, args.ledger, errors)
    card_count = verify_external_cards(root, ledger, errors)
    finding_count = verify_findings(root, ledger, args.review, errors)
    upstream_count = verify_upstream_items(ledger, errors)
    verification = ledger.get("verification")
    if not isinstance(verification, list) or not verification:
        errors.append("verification evidence list is required")
    else:
        for index, row in enumerate(verification):
            if not isinstance(row, dict) or row.get("passed") is not True or not row.get("evidence"):
                errors.append(f"verification[{index}] must be passed with evidence")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "PASS",
        "schema": SCHEMA,
        "repository": counts,
        "external_model_cards": card_count,
        "findings": finding_count,
        "upstream_items": upstream_count,
        "self_excluded": args.ledger,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
