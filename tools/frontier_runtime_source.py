#!/usr/bin/env python3
"""Assemble and byte-verify the Final Frontier runtime from public source locks."""

from __future__ import annotations

import argparse
import base64
import csv
import configparser
import hashlib
import os
import importlib.metadata
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

from frontier_common import (
    atomic_write_json,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)

LOCK_SCHEMA = "qwen38-frontier-runtime-source-lock/1"
RESOLVED_SCHEMA = "qwen38-frontier-runtime-resolved-lock/1"
RECEIPT_SCHEMA = "qwen38-frontier-runtime-source-receipt/1"
SBOM_SCHEMA = "qwen38-frontier-runtime-source-sbom/1"
VERIFY_SCHEMA = "qwen38-frontier-runtime-source-verification/1"
IMAGE_SCHEMA = "qwen38-frontier-runtime-image-receipt/1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HTTPS_GITHUB_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$"
)


class LockError(ValueError):
    """A fail-closed source-lock violation."""


def _fail(message: str, code: int = 2) -> NoReturn:
    print(f"frontier_runtime_source.py: {message}", file=sys.stderr)
    raise SystemExit(code)


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockError(f"{where} must be an object")
    return value


def _array(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise LockError(f"{where} must be an array")
    return value


def _string(value: object, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise LockError(f"{where} must be a nonempty string")
    return value


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise LockError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    if missing:
        raise LockError(f"{where} is missing keys: {', '.join(sorted(missing))}")


def _sha(value: object, where: str) -> str:
    text = _string(value, where)
    if not SHA_RE.fullmatch(text):
        raise LockError(f"{where} must be a full lowercase 40-hex Git identity")
    return text


def _safe_component(value: object, where: str) -> str:
    text = _string(value, where)
    if text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise LockError(f"{where} is not one safe path component")
    return text


def _require_new_file(path: Path, where: str) -> None:
    if path.exists() or path.is_symlink():
        raise LockError(f"{where} already exists: {path}")
    if not path.parent.is_dir():
        raise LockError(f"{where} parent directory does not exist: {path.parent}")


def _require_new_directory(path: Path, where: str) -> None:
    if path.exists() or path.is_symlink():
        raise LockError(f"{where} already exists: {path}")
    if not path.parent.is_dir():
        raise LockError(f"{where} parent directory does not exist: {path.parent}")


def _validate_source(item: object, index: int) -> dict[str, Any]:
    source = _object(item, f"sources[{index}]")
    fields = {"name", "role", "repository", "commit", "tree", "destination"}
    _keys(source, fields, fields, f"sources[{index}]")
    _safe_component(source["name"], f"sources[{index}].name")
    _string(source["role"], f"sources[{index}].role")
    repository = _string(source["repository"], f"sources[{index}].repository")
    if not HTTPS_GITHUB_RE.fullmatch(repository):
        raise LockError(
            f"sources[{index}].repository must be a public GitHub HTTPS .git URL"
        )
    _sha(source["commit"], f"sources[{index}].commit")
    _sha(source["tree"], f"sources[{index}].tree")
    _safe_component(source["destination"], f"sources[{index}].destination")
    return source


def _validate_lock_document(value: object) -> dict[str, Any]:
    lock = _object(value, "lock")
    fields = {
        "schema",
        "lock_id",
        "status",
        "source_policy",
        "sources",
        "patches",
        "submodule_bindings",
        "public_references",
        "requirements",
        "toolchain",
        "container",
        "exclusions",
        "blockers",
    }
    _keys(lock, fields, fields, "lock")
    if lock["schema"] != LOCK_SCHEMA:
        raise LockError(f"lock.schema must be {LOCK_SCHEMA!r}")
    _safe_component(lock["lock_id"], "lock.lock_id")
    if lock["status"] not in {"ready", "blocked", "qualification-required"}:
        raise LockError(
            "lock.status must be 'ready', 'blocked', or 'qualification-required'"
        )

    policy = _object(lock["source_policy"], "lock.source_policy")
    policy_fields = {
        "public_https_only",
        "immutable_full_shas",
        "reject_unlocked_submodules",
        "installed_bytes_manifest_required",
    }
    _keys(policy, policy_fields, policy_fields, "lock.source_policy")
    if any(policy[key] is not True for key in policy_fields):
        raise LockError("every source_policy gate must be true")

    sources = [_validate_source(item, idx) for idx, item in enumerate(_array(lock["sources"], "lock.sources"))]
    if not sources:
        raise LockError("lock.sources must not be empty")
    names = [item["name"] for item in sources]
    destinations = [item["destination"] for item in sources]
    if len(names) != len(set(names)):
        raise LockError("lock.sources contains duplicate names")
    if len(destinations) != len(set(destinations)):
        raise LockError("lock.sources contains duplicate destinations")

    patches = _array(lock["patches"], "lock.patches")
    patch_fields = {
        "id",
        "target_source",
        "path",
        "sha256",
        "pre_tree",
        "post_tree",
        "public_ancestor",
    }
    known_names = set(names)
    for idx, item in enumerate(patches):
        patch = _object(item, f"patches[{idx}]")
        _keys(patch, patch_fields, patch_fields, f"patches[{idx}]")
        _safe_component(patch["id"], f"patches[{idx}].id")
        target = _string(patch["target_source"], f"patches[{idx}].target_source")
        if target not in known_names:
            raise LockError(f"patches[{idx}] targets unknown source {target!r}")
        patch_path = Path(_string(patch["path"], f"patches[{idx}].path"))
        if patch_path.is_absolute() or ".." in patch_path.parts:
            raise LockError(f"patches[{idx}].path must be repository-relative")
        digest = _string(patch["sha256"], f"patches[{idx}].sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LockError(f"patches[{idx}].sha256 must be lowercase hex")
        _sha(patch["pre_tree"], f"patches[{idx}].pre_tree")
        _sha(patch["post_tree"], f"patches[{idx}].post_tree")
        _string(patch["public_ancestor"], f"patches[{idx}].public_ancestor")
    binding_fields = {"parent", "path", "source"}
    bound_paths: set[tuple[str, str]] = set()
    bound_sources: set[str] = set()
    for idx, item in enumerate(
        _array(lock["submodule_bindings"], "lock.submodule_bindings")
    ):
        binding = _object(item, f"submodule_bindings[{idx}]")
        _keys(
            binding,
            binding_fields,
            binding_fields,
            f"submodule_bindings[{idx}]",
        )
        parent = _string(binding["parent"], f"submodule_bindings[{idx}].parent")
        source = _string(binding["source"], f"submodule_bindings[{idx}].source")
        if parent not in known_names or source not in known_names or parent == source:
            raise LockError(f"submodule_bindings[{idx}] names an invalid source")
        relative_text = _string(
            binding["path"], f"submodule_bindings[{idx}].path"
        )
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
        ):
            raise LockError(f"submodule_bindings[{idx}].path is not relative")
        key = (parent, relative_text)
        if key in bound_paths or source in bound_sources:
            raise LockError("submodule bindings contain duplicate paths or sources")
        bound_paths.add(key)
        bound_sources.add(source)

    reference_fields = {"id", "repository", "commit", "tree", "disposition"}
    for idx, item in enumerate(
        _array(lock["public_references"], "lock.public_references")
    ):
        reference = _object(item, f"public_references[{idx}]")
        _keys(
            reference,
            reference_fields,
            reference_fields,
            f"public_references[{idx}]",
        )
        _string(reference["id"], f"public_references[{idx}].id")
        repository = _string(
            reference["repository"], f"public_references[{idx}].repository"
        )
        if not HTTPS_GITHUB_RE.fullmatch(repository):
            raise LockError(
                f"public_references[{idx}].repository must be a public GitHub URL"
            )
        _sha(reference["commit"], f"public_references[{idx}].commit")
        _sha(reference["tree"], f"public_references[{idx}].tree")
        if reference["disposition"] not in {
            "base",
            "included-by-ancestry",
            "included-by-public-port",
            "reference-only-blocked-port",
        }:
            raise LockError(f"public_references[{idx}].disposition is unknown")

    requirement_fields = {"id", "behavior", "status", "source"}
    for idx, item in enumerate(_array(lock["requirements"], "lock.requirements")):
        requirement = _object(item, f"requirements[{idx}]")
        _keys(
            requirement,
            requirement_fields,
            requirement_fields,
            f"requirements[{idx}]",
        )
        for field in ("id", "behavior", "source"):
            _string(requirement[field], f"requirements[{idx}].{field}")
        if requirement["status"] not in {
            "provided",
            "provided-by-vllm-base",
            "qualification-required",
            "blocked",
        }:
            raise LockError(f"requirements[{idx}].status is unknown")

    toolchain = _object(lock["toolchain"], "lock.toolchain")
    toolchain_fields = {
        "architecture",
        "base_image",
        "cuda",
        "nvcc_wheel",
        "torch",
        "torchvision",
        "vllm_build_version",
        "flashinfer",
        "exllamav3_runtime",
        "b12x_source_version",
        "compiler_flags",
    }
    _keys(toolchain, toolchain_fields, toolchain_fields, "lock.toolchain")
    for field in toolchain_fields - {"compiler_flags"}:
        _string(toolchain[field], f"lock.toolchain.{field}")
    compiler_flags = _array(toolchain["compiler_flags"], "lock.toolchain.compiler_flags")
    if not compiler_flags:
        raise LockError("lock.toolchain.compiler_flags must not be empty")
    for idx, flag in enumerate(compiler_flags):
        _string(flag, f"lock.toolchain.compiler_flags[{idx}]")

    exclusion_fields = {"id", "reason"}
    exclusions = _array(lock["exclusions"], "lock.exclusions")
    for idx, item in enumerate(exclusions):
        exclusion = _object(item, f"exclusions[{idx}]")
        _keys(exclusion, exclusion_fields, exclusion_fields, f"exclusions[{idx}]")
        _string(exclusion["id"], f"exclusions[{idx}].id")
        _string(exclusion["reason"], f"exclusions[{idx}].reason")
    excluded_ids = {item["id"] for item in exclusions}
    required_exclusions = {
        "vllm-project/vllm#52530",
        "local-inference-lab/vllm#436",
    }
    if excluded_ids != required_exclusions:
        raise LockError(
            "lock.exclusions must contain exactly #52530 and local-inference-lab/vllm#436"
        )

    blocker_fields = {
        "id",
        "component",
        "destination",
        "missing",
        "required_behavior",
        "resolution",
    }
    blockers = _array(lock["blockers"], "lock.blockers")
    for idx, item in enumerate(blockers):
        blocker = _object(item, f"blockers[{idx}]")
        _keys(blocker, blocker_fields, blocker_fields, f"blockers[{idx}]")
        for field in blocker_fields:
            _string(blocker[field], f"blockers[{idx}].{field}")

    required_destinations = {
        "vllm",
        "b12x",
        "flashinfer",
        "flashinfer-cccl",
        "flashinfer-cutlass",
        "flashinfer-nixl",
        "flashinfer-spdlog",
        "exllamav3-runtime",
        "exllamav3-converter",
    }
    if lock["status"] == "ready" and any(
        item["status"] == "blocked" for item in lock["requirements"]
    ):
        raise LockError("a ready lock cannot contain blocked requirements")
    if lock["status"] == "ready" and blockers:
        raise LockError("a ready lock cannot contain blockers")
    if lock["status"] == "blocked" and not blockers:
        raise LockError("a blocked lock must identify at least one blocker")
    if lock["status"] == "qualification-required":
        if blockers:
            raise LockError("qualification-required lock cannot contain source blockers")
        if not any(
            item["status"] == "qualification-required"
            for item in lock["requirements"]
        ):
            raise LockError(
                "qualification-required lock must identify a qualification gate"
            )

    container = _object(lock["container"], "lock.container")
    if container.get("status") not in {"ready", "blocked"}:
        raise LockError("lock.container.status must be 'ready' or 'blocked'")
    if container["status"] == "ready":
        allowed_container = {"status", "base_reference", "platform"}
        _keys(container, allowed_container, allowed_container, "lock.container")
        reference = _string(
            container["base_reference"], "lock.container.base_reference"
        )
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", reference):
            raise LockError(
                "lock.container.base_reference must contain an immutable sha256 digest"
            )
    else:
        blocked_container = {"status", "required_identity", "forensic_tag_only"}
        _keys(container, blocked_container, blocked_container, "lock.container")
        _string(container["required_identity"], "lock.container.required_identity")
        _string(container["forensic_tag_only"], "lock.container.forensic_tag_only")
    if lock["status"] == "ready" and container["status"] != "ready":
        raise LockError("a ready lock requires a ready container identity")
    if container["status"] == "ready":
        if toolchain["base_image"] != container["base_reference"]:
            raise LockError("toolchain and container base-image identities differ")
        if toolchain["architecture"] != container["platform"]:
            raise LockError("toolchain and container platforms differ")
    return lock


def _load_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LockError(f"lock is not a regular file: {path}")
    return _validate_lock_document(load_strict_json(path))


def _run(argv: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise LockError(f"command failed ({' '.join(argv)}): {detail}")
    return result.stdout.strip()


def _cache_checkout(source: dict[str, Any], cache: Path, destination: Path) -> None:
    mirror = cache / f"{source['name']}.git"
    if mirror.exists():
        if not mirror.is_dir():
            raise LockError(f"cache entry is not a directory: {mirror}")
        origin = _run(["git", "--git-dir", str(mirror), "remote", "get-url", "origin"])
        if origin != source["repository"]:
            raise LockError(f"cache origin mismatch for {source['name']}: {origin}")
    else:
        _run(["git", "init", "--bare", str(mirror)])
        _run(
            ["git", "--git-dir", str(mirror), "remote", "add", "origin", source["repository"]]
        )
    _run(
        [
            "git",
            "--git-dir",
            str(mirror),
            "fetch",
            "--force",
            "--no-tags",
            "--depth=1",
            "origin",
            source["commit"],
        ]
    )
    fetched = _run(["git", "--git-dir", str(mirror), "rev-parse", "FETCH_HEAD^{commit}"])
    if fetched != source["commit"]:
        raise LockError(f"fetched commit mismatch for {source['name']}")
    tree = _run(["git", "--git-dir", str(mirror), "rev-parse", "FETCH_HEAD^{tree}"])
    if tree != source["tree"]:
        raise LockError(f"fetched tree mismatch for {source['name']}: {tree}")
    _run(
        [
            "git",
            "--git-dir",
            str(mirror),
            "update-ref",
            f"refs/frontier/{source['commit']}",
            "FETCH_HEAD",
        ]
    )
    _run(["git", "clone", "--no-checkout", "--shared", str(mirror), str(destination)])
    _run(
        [
            "git",
            "fetch",
            "--force",
            "--no-tags",
            "--depth=1",
            str(mirror),
            f"refs/frontier/{source['commit']}",
        ],
        cwd=destination,
    )
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    head = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    if head != source["commit"]:
        raise LockError(f"checkout commit mismatch for {source['name']}")
    checkout_tree = _run(["git", "write-tree"], cwd=destination)
    if checkout_tree != source["tree"]:
        raise LockError(f"checkout tree mismatch for {source['name']}: {checkout_tree}")


def _apply_patches(
    lock: dict[str, Any], repo_root: Path, source_dirs: dict[str, Path]
) -> dict[str, str]:
    current_trees = {source["name"]: source["tree"] for source in lock["sources"]}
    for patch in lock["patches"]:
        target = patch["target_source"]
        if current_trees[target] != patch["pre_tree"]:
            raise LockError(
                f"patch {patch['id']} pre_tree {patch['pre_tree']} does not match {current_trees[target]}"
            )
        patch_path = repo_root / patch["path"]
        if not patch_path.is_file() or sha256_file(patch_path) != patch["sha256"]:
            raise LockError(f"patch {patch['id']} is missing or has the wrong SHA256")
        source_dir = source_dirs[target]
        _run(
            ["git", "apply", "--index", "--whitespace=error-all", str(patch_path)],
            cwd=source_dir,
        )
        result_tree = _run(["git", "write-tree"], cwd=source_dir)
        if result_tree != patch["post_tree"]:
            raise LockError(
                f"patch {patch['id']} produced tree {result_tree}, expected {patch['post_tree']}"
            )
        current_trees[target] = result_tree
    return current_trees


def _verify_submodule_bindings(
    lock: dict[str, Any], source_dirs: dict[str, Path]
) -> None:
    sources = {source["name"]: source for source in lock["sources"]}
    expected_by_parent: dict[str, dict[str, str]] = {}
    for binding in lock["submodule_bindings"]:
        expected_by_parent.setdefault(binding["parent"], {})[binding["path"]] = binding[
            "source"
        ]
    for parent, source_dir in source_dirs.items():
        expected = expected_by_parent.get(parent, {})
        gitmodules = source_dir / ".gitmodules"
        actual: dict[str, str] = {}
        if gitmodules.exists():
            parser = configparser.ConfigParser(interpolation=None)
            try:
                parser.read(gitmodules, encoding="utf-8")
            except configparser.Error as exc:
                raise LockError(f"invalid {gitmodules}: {exc}") from exc
            for section in parser.sections():
                if not section.startswith('submodule "') or not section.endswith('"'):
                    raise LockError(f"unknown .gitmodules section {section!r}")
                if set(parser[section]) != {"path", "url"}:
                    raise LockError(f"{gitmodules} section {section!r} has unknown keys")
                path = parser[section]["path"]
                url = parser[section]["url"]
                if path in actual:
                    raise LockError(f"{gitmodules} repeats submodule path {path!r}")
                actual[path] = url
        if set(actual) != set(expected):
            raise LockError(
                f"source {parent} submodule set is not completely locked: "
                f"actual={sorted(actual)} expected={sorted(expected)}"
            )
        for path, source_name in expected.items():
            child = sources[source_name]
            if actual[path] != child["repository"]:
                raise LockError(
                    f"source {parent} submodule {path} URL does not match {source_name}"
                )
            staged = _run(["git", "ls-files", "--stage", "--", path], cwd=source_dir)
            fields = staged.split(maxsplit=3)
            if (
                len(fields) != 4
                or fields[0] != "160000"
                or fields[1] != child["commit"]
            ):
                raise LockError(
                    f"source {parent} submodule {path} Git identity is not locked "
                    f"to {child['commit']}"
                )


def _manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise LockError(f"source root is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISREG(info.st_mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": f"{mode:04o}",
                    "size": info.st_size,
                    "sha256": sha256_file(path),
                }
            )
        elif stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": f"{mode:04o}",
                    "target": target,
                    "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                }
            )
        else:
            raise LockError(f"unsupported source filesystem object: {path}")
    if not entries:
        raise LockError(f"source root is empty: {root}")
    return entries


def _source_manifests(
    lock: dict[str, Any], workspace: Path, final_trees: dict[str, str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in lock["sources"]:
        entries = _manifest(workspace / source["destination"])
        result.append(
            {
                "name": source["name"],
                "destination": source["destination"],
                "repository": source["repository"],
                "commit": source["commit"],
                "base_tree": source["tree"],
                "tree": final_trees[source["name"]],
                "files": entries,
                "files_sha256": canonical_sha256(entries),
            }
        )
    return result


def _assert_ready(lock: dict[str, Any]) -> None:
    if lock["status"] != "ready" or lock["blockers"]:
        unresolved = [item.get("id", "unnamed") for item in lock["blockers"]]
        unresolved.extend(
            item["id"]
            for item in lock["requirements"]
            if item["status"] == "qualification-required"
        )
        raise LockError(f"source lock is not build-ready: {', '.join(unresolved)}")


def command_validate(args: argparse.Namespace) -> None:
    report = Path(args.report)
    _require_new_file(report, "validation report")
    lock_path = Path(args.lock)
    lock = _load_lock(lock_path)
    value = {
        "schema": VERIFY_SCHEMA,
        "operation": "validate-lock",
        "lock_schema": lock["schema"],
        "lock_id": lock["lock_id"],
        "lock_sha256": canonical_sha256(lock),
        "status": lock["status"],
        "blocker_ids": [item.get("id") for item in lock["blockers"]],
        "qualification_ids": [
            item["id"]
            for item in lock["requirements"]
            if item["status"] == "qualification-required"
        ],
        "valid": True,
    }
    atomic_write_json(report, value)
    if lock["status"] != "ready":
        _fail(
            "lock is structurally valid but not build-ready: "
            + ", ".join(value["blocker_ids"] + value["qualification_ids"]),
            4,
        )


def command_assemble(args: argparse.Namespace) -> None:
    lock = _load_lock(Path(args.lock))
    _assert_ready(lock)
    workspace = Path(args.workspace)
    cache = Path(args.cache)
    resolved_path = Path(args.resolved_lock)
    receipt_path = Path(args.source_receipt)
    sbom_path = Path(args.sbom)
    _require_new_directory(workspace, "workspace")
    if not cache.is_dir():
        raise LockError(f"campaign cache must already exist as a directory: {cache}")
    for path, label in (
        (resolved_path, "resolved lock"),
        (receipt_path, "source receipt"),
        (sbom_path, "source SBOM"),
    ):
        _require_new_file(path, label)

    workspace.mkdir(mode=0o700)
    source_dirs: dict[str, Path] = {}
    try:
        for source in lock["sources"]:
            destination = workspace / source["destination"]
            _cache_checkout(source, cache, destination)
            source_dirs[source["name"]] = destination
        final_trees = _apply_patches(lock, Path(args.repo_root), source_dirs)
        _verify_submodule_bindings(lock, source_dirs)
        for source in lock["sources"]:
            source_dir = source_dirs[source["name"]]
            _run(["git", "diff", "--exit-code", "--no-ext-diff"], cwd=source_dir)
            untracked = _run(
                ["git", "ls-files", "--others", "--exclude-standard"], cwd=source_dir
            )
            if untracked:
                raise LockError(
                    f"source {source['name']} contains unlocked files: {untracked}"
                )
        for directory in source_dirs.values():
            shutil.rmtree(directory / ".git")
        manifests = _source_manifests(lock, workspace, final_trees)
        resolved_core = {
            "schema": RESOLVED_SCHEMA,
            "lock_id": lock["lock_id"],
            "lock_sha256": canonical_sha256(lock),
            "sources": manifests,
            "submodule_bindings": lock["submodule_bindings"],
            "requirements": lock["requirements"],
            "toolchain": lock["toolchain"],
            "container": lock["container"],
            "exclusions": lock["exclusions"],
        }
        resolved = resolved_core | {"resolved_sha256": canonical_sha256(resolved_core)}
        atomic_write_json(resolved_path, resolved)
        sbom_core = {
            "schema": SBOM_SCHEMA,
            "lock_id": lock["lock_id"],
            "resolved_sha256": resolved["resolved_sha256"],
            "components": [
                {
                    "name": item["name"],
                    "repository": item["repository"],
                    "commit": item["commit"],
                    "tree": item["tree"],
                    "files_sha256": item["files_sha256"],
                    "file_count": len(item["files"]),
                }
                for item in manifests
            ],
            "toolchain": lock["toolchain"],
        }
        atomic_write_json(sbom_path, sbom_core | {"sbom_sha256": canonical_sha256(sbom_core)})
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "lock_id": lock["lock_id"],
            "lock_sha256": canonical_sha256(lock),
            "resolved_sha256": resolved["resolved_sha256"],
            "source_count": len(manifests),
            "source_files_sha256": canonical_sha256(
                {item["name"]: item["files_sha256"] for item in manifests}
            ),
            "excluded_stacks": [item["id"] for item in lock["exclusions"]],
            "qualification_gates": [
                item["id"]
                for item in lock["requirements"]
                if item["status"] == "qualification-required"
            ],
            "installed_bytes_verified": True,
        }
        atomic_write_json(receipt_path, receipt)
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        for output in (resolved_path, receipt_path, sbom_path):
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        raise


def _load_resolved(path: Path) -> dict[str, Any]:
    value = _object(load_strict_json(path), "resolved lock")
    allowed = {
        "schema",
        "lock_id",
        "lock_sha256",
        "sources",
        "submodule_bindings",
        "requirements",
        "toolchain",
        "container",
        "exclusions",
        "resolved_sha256",
    }
    _keys(value, allowed, allowed, "resolved lock")
    if value["schema"] != RESOLVED_SCHEMA:
        raise LockError(f"resolved lock schema must be {RESOLVED_SCHEMA}")
    _safe_component(value["lock_id"], "resolved lock.lock_id")
    for field in ("lock_sha256", "resolved_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", _string(value[field], field)):
            raise LockError(f"resolved lock.{field} must be a lowercase SHA256")
    source_fields = {
        "name",
        "destination",
        "repository",
        "commit",
        "base_tree",
        "tree",
        "files",
        "files_sha256",
    }
    names: set[str] = set()
    destinations: set[str] = set()
    for source_index, item in enumerate(
        _array(value["sources"], "resolved lock.sources")
    ):
        source = _object(item, f"resolved lock.sources[{source_index}]")
        _keys(
            source,
            source_fields,
            source_fields,
            f"resolved lock.sources[{source_index}]",
        )
        name = _safe_component(
            source["name"], f"resolved lock.sources[{source_index}].name"
        )
        destination = _safe_component(
            source["destination"],
            f"resolved lock.sources[{source_index}].destination",
        )
        if name in names or destination in destinations:
            raise LockError("resolved lock contains duplicate source identities")
        names.add(name)
        destinations.add(destination)
        repository = _string(
            source["repository"],
            f"resolved lock.sources[{source_index}].repository",
        )
        if not HTTPS_GITHUB_RE.fullmatch(repository):
            raise LockError("resolved lock contains a non-public source repository")
        for field in ("commit", "base_tree", "tree"):
            _sha(source[field], f"resolved lock.sources[{source_index}].{field}")
        files = _array(
            source["files"], f"resolved lock.sources[{source_index}].files"
        )
        seen_paths: set[str] = set()
        for file_index, file_item in enumerate(files):
            entry = _object(
                file_item,
                f"resolved lock.sources[{source_index}].files[{file_index}]",
            )
            entry_type = entry.get("type")
            entry_fields = (
                {"path", "type", "mode", "size", "sha256"}
                if entry_type == "file"
                else {"path", "type", "mode", "target", "sha256"}
            )
            _keys(
                entry,
                entry_fields,
                entry_fields,
                f"resolved lock.sources[{source_index}].files[{file_index}]",
            )
            relative_text = _string(entry["path"], "manifest path")
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != relative_text
                or relative_text in seen_paths
            ):
                raise LockError(f"invalid or duplicate manifest path {relative_text!r}")
            seen_paths.add(relative_text)
            if entry_type not in {"file", "symlink"}:
                raise LockError(f"unknown manifest entry type {entry_type!r}")
            if not re.fullmatch(r"[0-7]{4}", _string(entry["mode"], "manifest mode")):
                raise LockError("manifest mode must be four octal digits")
            if not re.fullmatch(
                r"[0-9a-f]{64}", _string(entry["sha256"], "manifest SHA256")
            ):
                raise LockError("manifest SHA256 must be lowercase hex")
            if entry_type == "file":
                if (
                    not isinstance(entry["size"], int)
                    or isinstance(entry["size"], bool)
                    or entry["size"] < 0
                ):
                    raise LockError("manifest file size must be a nonnegative integer")
            else:
                _string(entry["target"], "manifest symlink target", nonempty=False)
        if not files:
            raise LockError(f"resolved source {name} has an empty manifest")
        expected_files_digest = _string(
            source["files_sha256"],
            f"resolved lock.sources[{source_index}].files_sha256",
        )
        if canonical_sha256(files) != expected_files_digest:
            raise LockError(f"resolved source {name} file-manifest digest mismatch")
    resolved_binding_fields = {"parent", "path", "source"}
    for idx, item in enumerate(
        _array(value["submodule_bindings"], "resolved lock.submodule_bindings")
    ):
        binding = _object(item, f"resolved lock.submodule_bindings[{idx}]")
        _keys(
            binding,
            resolved_binding_fields,
            resolved_binding_fields,
            f"resolved lock.submodule_bindings[{idx}]",
        )
        if binding["parent"] not in names or binding["source"] not in names:
            raise LockError("resolved lock submodule binding names an unknown source")
        path = Path(_string(binding["path"], "resolved submodule path"))
        if path.is_absolute() or ".." in path.parts:
            raise LockError("resolved lock submodule binding path is not relative")
    resolved_requirement_fields = {"id", "behavior", "status", "source"}
    for idx, item in enumerate(
        _array(value["requirements"], "resolved lock.requirements")
    ):
        requirement = _object(item, f"resolved lock.requirements[{idx}]")
        _keys(
            requirement,
            resolved_requirement_fields,
            resolved_requirement_fields,
            f"resolved lock.requirements[{idx}]",
        )
        for field in resolved_requirement_fields:
            _string(requirement[field], f"resolved requirement.{field}")
    _object(value["toolchain"], "resolved lock.toolchain")
    _object(value["container"], "resolved lock.container")
    _array(value["exclusions"], "resolved lock.exclusions")
    core = {key: item for key, item in value.items() if key != "resolved_sha256"}
    if canonical_sha256(core) != value["resolved_sha256"]:
        raise LockError("resolved lock canonical digest mismatch")
    return value

def _verify_sources(
    resolved: dict[str, Any], workspace: Path
) -> tuple[dict[str, str], str]:
    expected_destinations = {item["destination"] for item in resolved["sources"]}
    actual_destinations = {
        path.name
        for path in workspace.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if actual_destinations != expected_destinations:
        raise LockError(
            f"workspace source set mismatch: actual={sorted(actual_destinations)} "
            f"expected={sorted(expected_destinations)}"
        )
    verified: dict[str, str] = {}
    for source in resolved["sources"]:
        actual = _manifest(workspace / source["destination"])
        digest = canonical_sha256(actual)
        if digest != source["files_sha256"] or actual != source["files"]:
            raise LockError(f"installed source byte mismatch for {source['name']}")
        verified[source["name"]] = digest
    return verified, canonical_sha256(verified)


def command_verify(args: argparse.Namespace) -> None:
    report_path = Path(args.report)
    _require_new_file(report_path, "verification report")
    resolved = _load_resolved(Path(args.resolved_lock))
    verified, source_digest = _verify_sources(resolved, Path(args.workspace))
    report = {
        "schema": VERIFY_SCHEMA,
        "operation": "verify-tree",
        "lock_id": resolved["lock_id"],
        "resolved_sha256": resolved["resolved_sha256"],
        "source_files_sha256": source_digest,
        "verified_sources": verified,
        "every_installed_source_byte_verified": True,
    }
    atomic_write_json(report_path, report)


def _verify_record(project: str, expected_version: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(project)
    if distribution.version != expected_version:
        raise LockError(
            f"{project} version {distribution.version!r} != {expected_version!r}"
        )
    record_text = distribution.read_text("RECORD")
    if not record_text:
        raise LockError(f"{project} has no installed wheel RECORD")
    base = Path(sys.prefix).resolve()
    checked = 0
    unhashed: list[str] = []
    for row in csv.reader(record_text.splitlines()):
        if len(row) != 3:
            raise LockError(f"{project} has a malformed RECORD row")
        relative_text, hash_spec, size_text = row
        installed = Path(distribution.locate_file(relative_text))
        resolved_installed = installed.resolve()
        if not resolved_installed.is_relative_to(base):
            raise LockError(f"{project} RECORD escapes the environment: {relative_text}")
        if not installed.is_file():
            raise LockError(f"{project} RECORD file is missing: {relative_text}")
        if not hash_spec:
            if not (
                relative_text.endswith(".pyc")
                or relative_text.endswith(".dist-info/RECORD")
            ):
                unhashed.append(relative_text)
            continue
        algorithm, separator, encoded = hash_spec.partition("=")
        if separator != "=" or algorithm != "sha256":
            raise LockError(f"{project} RECORD uses a non-SHA256 hash")
        expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
        actual = sha256_file(installed)
        if actual != expected:
            raise LockError(f"{project} RECORD hash mismatch: {relative_text}")
        if size_text and installed.stat().st_size != int(size_text):
            raise LockError(f"{project} RECORD size mismatch: {relative_text}")
        checked += 1
    if unhashed or checked == 0:
        raise LockError(f"{project} RECORD has unhashed payloads: {unhashed}")
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url or "/opt/frontier/wheels/" not in direct_url:
        raise LockError(f"{project} was not installed from the source-built wheel set")
    return {
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record_text.encode("utf-8")).hexdigest(),
        "hashed_files": checked,
        "direct_url_sha256": hashlib.sha256(direct_url.encode("utf-8")).hexdigest(),
    }


def command_verify_runtime(args: argparse.Namespace) -> None:
    report_path = Path(args.report)
    _require_new_file(report_path, "runtime verification report")
    resolved = _load_resolved(Path(args.resolved_lock))
    verified, source_digest = _verify_sources(resolved, Path(args.workspace))
    toolchain = resolved["toolchain"]
    expected = {
        "vllm": toolchain["vllm_build_version"],
        "b12x": toolchain["b12x_source_version"],
        "flashinfer-python": toolchain["flashinfer"],
        "exllamav3": toolchain["exllamav3_runtime"],
    }
    audits = {
        project: _verify_record(project, version)
        for project, version in expected.items()
    }
    extension_root = Path(sys.prefix) / "lib/python3.12/site-packages/vllm"
    extension_files = sorted(extension_root.rglob("*.so"))
    if not extension_files:
        raise LockError("digest-pinned base supplied no compiled vLLM extensions")
    base_extensions = [
        {
            "path": path.relative_to(extension_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in extension_files
    ]
    torch_version = importlib.metadata.version("torch")
    torchvision_version = importlib.metadata.version("torchvision")
    torch_module = __import__("torch")
    torch_cuda_version = torch_module.version.cuda
    nvcc_project, separator, nvcc_version = toolchain["nvcc_wheel"].partition("==")
    if separator != "==" or not nvcc_project or not nvcc_version:
        raise LockError("resolved nvcc_wheel is not an exact package pin")
    installed_nvcc = importlib.metadata.version(nvcc_project)
    if (
        torch_version != toolchain["torch"]
        or torchvision_version != toolchain["torchvision"]
        or torch_cuda_version != toolchain["cuda"]
        or installed_nvcc != nvcc_version
    ):
        raise LockError("immutable base toolchain versions do not match the lock")
    for legacy in (Path("/opt/exllamav3-python"), Path("/opt/exllamav3")):
        if legacy.exists() or legacy.is_symlink():
            raise LockError(f"legacy runtime source remains installed: {legacy}")
    report = {
        "schema": VERIFY_SCHEMA,
        "operation": "verify-runtime",
        "lock_id": resolved["lock_id"],
        "resolved_sha256": resolved["resolved_sha256"],
        "source_files_sha256": source_digest,
        "verified_sources": verified,
        "every_installed_source_byte_verified": True,
        "distributions": audits,
        "base_compiled_extensions": base_extensions,
        "toolchain_versions": {
            "torch": torch_version,
            "torchvision": torchvision_version,
            "torch_cuda": torch_cuda_version,
            "nvidia-cuda-nvcc": installed_nvcc,
            "base_image": toolchain["base_image"],
        },
        "old_packages_removed": True,
        "qualification_gates": [
            item["id"]
            for item in resolved["requirements"]
            if item["status"] == "qualification-required"
        ],
    }
    atomic_write_json(report_path, report)


def command_base_reference(args: argparse.Namespace) -> None:
    output = Path(args.output)
    _require_new_file(output, "base-reference output")
    lock = _load_lock(Path(args.lock))
    _assert_ready(lock)
    atomic_write_json(
        output,
        {
            "schema": "qwen38-frontier-runtime-base-reference/1",
            "lock_id": lock["lock_id"],
            "base_reference": lock["container"]["base_reference"],
        },
    )


def command_finalize(args: argparse.Namespace) -> None:
    output = Path(args.output)
    _require_new_file(output, "image receipt")
    source_receipt = _object(
        load_strict_json(Path(args.source_receipt)), "source receipt"
    )
    verification = _object(
        load_strict_json(Path(args.verification_report)), "verification report"
    )
    sbom = _object(load_strict_json(Path(args.sbom)), "source SBOM")
    source_receipt_fields = {
        "schema",
        "lock_id",
        "lock_sha256",
        "resolved_sha256",
        "source_count",
        "source_files_sha256",
        "excluded_stacks",
        "qualification_gates",
        "installed_bytes_verified",
    }
    _keys(
        source_receipt,
        source_receipt_fields,
        source_receipt_fields,
        "source receipt",
    )
    if source_receipt["schema"] != RECEIPT_SCHEMA:
        raise LockError("source receipt schema mismatch")
    if source_receipt["installed_bytes_verified"] is not True:
        raise LockError("source receipt does not attest installed-byte verification")
    qualification_gates = _array(
        source_receipt["qualification_gates"], "source receipt.qualification_gates"
    )
    for idx, gate in enumerate(qualification_gates):
        _string(gate, f"source receipt.qualification_gates[{idx}]")
    for field in ("lock_sha256", "resolved_sha256", "source_files_sha256"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", _string(source_receipt[field], f"source receipt.{field}")
        ):
            raise LockError(f"source receipt.{field} must be a lowercase SHA256")
    verification_fields = {
        "schema",
        "operation",
        "lock_id",
        "resolved_sha256",
        "source_files_sha256",
        "verified_sources",
        "every_installed_source_byte_verified",
        "distributions",
        "base_compiled_extensions",
        "toolchain_versions",
        "old_packages_removed",
        "qualification_gates",
    }
    _keys(
        verification,
        verification_fields,
        verification_fields,
        "verification report",
    )
    if (
        verification["schema"] != VERIFY_SCHEMA
        or verification["operation"] != "verify-runtime"
        or verification["every_installed_source_byte_verified"] is not True
        or verification["old_packages_removed"] is not True
    ):
        raise LockError(
            "candidate verification did not cover sources, RECORDs, and old-package removal"
        )
    sbom_fields = {
        "schema",
        "lock_id",
        "resolved_sha256",
        "components",
        "toolchain",
        "sbom_sha256",
    }
    _keys(sbom, sbom_fields, sbom_fields, "source SBOM")
    if sbom["schema"] != SBOM_SCHEMA:
        raise LockError("source SBOM schema mismatch")
    sbom_core = {key: value for key, value in sbom.items() if key != "sbom_sha256"}
    if canonical_sha256(sbom_core) != sbom["sbom_sha256"]:
        raise LockError("source SBOM canonical digest mismatch")
    resolved = source_receipt["resolved_sha256"]
    if (
        verification["resolved_sha256"] != resolved
        or sbom["resolved_sha256"] != resolved
        or verification["lock_id"] != source_receipt["lock_id"]
        or sbom["lock_id"] != source_receipt["lock_id"]
        or verification["source_files_sha256"]
        != source_receipt["source_files_sha256"]
        or verification["qualification_gates"] != qualification_gates
    ):
        raise LockError("receipt, verification, and SBOM resolve different source locks")
    digest = _string(args.image_digest, "image digest")
    if not DIGEST_RE.fullmatch(digest):
        raise LockError("image digest must be sha256:<64 lowercase hex>")
    result = {
        "schema": IMAGE_SCHEMA,
        "lock_id": source_receipt["lock_id"],
        "resolved_sha256": resolved,
        "image_reference": _string(args.image_reference, "image reference"),
        "image_digest": digest,
        "source_receipt_sha256": canonical_sha256(source_receipt),
        "source_sbom_sha256": canonical_sha256(sbom),
        "verification_sha256": canonical_sha256(verification),
        "buildable": True,
        "every_installed_source_byte_verified_before_promotion": True,
        "qualification_gates": source_receipt["qualification_gates"],
        "qualified": not source_receipt["qualification_gates"],
        "qualification_status": (
            "complete" if not source_receipt["qualification_gates"] else "pending"
        ),
    }
    atomic_write_json(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-lock")
    validate.add_argument("--lock", required=True)
    validate.add_argument("--report", required=True)
    validate.set_defaults(func=command_validate)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--lock", required=True)
    assemble.add_argument("--repo-root", required=True)
    assemble.add_argument("--cache", required=True)
    assemble.add_argument("--workspace", required=True)
    assemble.add_argument("--resolved-lock", required=True)
    assemble.add_argument("--source-receipt", required=True)
    assemble.add_argument("--sbom", required=True)
    assemble.set_defaults(func=command_assemble)

    verify = subparsers.add_parser("verify-tree")
    verify.add_argument("--resolved-lock", required=True)
    verify.add_argument("--workspace", required=True)
    verify.add_argument("--report", required=True)
    verify.set_defaults(func=command_verify)
    runtime = subparsers.add_parser("verify-runtime")
    runtime.add_argument("--resolved-lock", required=True)
    runtime.add_argument("--workspace", required=True)
    runtime.add_argument("--report", required=True)
    runtime.set_defaults(func=command_verify_runtime)


    base = subparsers.add_parser("base-reference")
    base.add_argument("--lock", required=True)
    base.add_argument("--output", required=True)
    base.set_defaults(func=command_base_reference)

    finalize = subparsers.add_parser("finalize-image")
    finalize.add_argument("--source-receipt", required=True)
    finalize.add_argument("--verification-report", required=True)
    finalize.add_argument("--sbom", required=True)
    finalize.add_argument("--image-reference", required=True)
    finalize.add_argument("--image-digest", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        _fail(str(exc))


if __name__ == "__main__":
    main()
