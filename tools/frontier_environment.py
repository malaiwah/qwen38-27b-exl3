#!/usr/bin/env python3
"""Capture and verify path-independent Python and host environment locks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import ipaddress
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Sequence, Tuple

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json, sha256_file


SCHEMA = "qwen38-frontier-environment-lock/1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_ARGUMENT = re.compile(r"^[A-Za-z0-9_.+,=:@%-]+$")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_WORD = re.compile(r"(?:token|secret|password|passwd|credential|api[_-]?key|authorization)", re.I)
MAX_COMMAND_OUTPUT = 1 << 20


class LockError(ValueError):
    """A fail-closed environment-lock violation."""


def fail(message: str, code: int = 2) -> NoReturn:
    print("frontier_environment.py: {}".format(message), file=sys.stderr)
    raise SystemExit(code)


def require_id(value: str, where: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise LockError("{} must be a stable logical ID".format(where))
    return value


def parse_assignments(values: Sequence[str], where: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        require_id(key, where)
        if separator != "=" or not value:
            raise LockError("{} entries must be ID=VALUE".format(where))
        if key in result:
            raise LockError("{} repeats ID {!r}".format(where, key))
        result[key] = value
    return result


def parse_binary_arguments(values: Sequence[str], binary_ids: Sequence[str]) -> Dict[str, List[str]]:
    result = {key: [] for key in binary_ids}
    for item in values:
        key, separator, argument = item.partition("=")
        require_id(key, "--binary-arg")
        if separator != "=" or not argument:
            raise LockError("--binary-arg entries must be ID=ARG")
        if key not in result:
            raise LockError("--binary-arg names undeclared binary {!r}".format(key))
        if not SAFE_ARGUMENT.fullmatch(argument):
            raise LockError("binary version arguments must be path-free option tokens")
        if argument.startswith("/") or WINDOWS_ABSOLUTE.match(argument) or SECRET_WORD.search(argument):
            raise LockError("binary version arguments may not contain paths or credential-like names")
        result[key].append(argument)
    for key, arguments in result.items():
        if not arguments:
            raise LockError("binary {!r} has no --binary-arg version command".format(key))
    return result


def parse_source_bindings(values: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    result: Dict[str, Tuple[str, str]] = {}
    for item in values:
        project, separator, binding = item.partition("=")
        kind, colon, source_id = binding.partition(":")
        normalized = normalize_project(project)
        if not normalized or separator != "=" or colon != ":":
            raise LockError("--source-distribution entries must be PROJECT=KIND:SOURCE_ID")
        if kind not in {"source", "editable"}:
            raise LockError("source distribution KIND must be 'source' or 'editable'")
        require_id(source_id, "source distribution source ID")
        if normalized in result:
            raise LockError("source distribution {!r} is repeated".format(project))
        result[normalized] = (kind, source_id)
    return result


def normalize_project(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def normalize_distribution_selection(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values:
        name = normalize_project(value)
        if not name or not SAFE_ID.fullmatch(name):
            raise LockError("--distribution must name a Python project")
        if name in seen:
            raise LockError("--distribution repeats {!r}".format(value))
        seen.add(name)
        normalized.append(name)
    return sorted(normalized)




def resolve_executable(value: str, where: str) -> Path:
    if os.sep in value or (os.altsep is not None and os.altsep in value):
        candidate = Path(value).expanduser()
    else:
        found = shutil.which(value)
        if found is None:
            raise LockError("{} executable is not on PATH: {!r}".format(where, value))
        candidate = Path(found)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LockError("cannot resolve {} executable {!r}: {}".format(where, value, exc)) from exc
    if not resolved.is_file():
        raise LockError("{} is not a file: {}".format(where, resolved))
    if not os.access(str(resolved), os.X_OK):
        raise LockError("{} is not executable: {}".format(where, resolved))
    return resolved


def resolve_launcher(value: str, where: str) -> Path:
    """Resolve a command name without dereferencing a virtualenv launcher."""
    if os.sep in value or (os.altsep is not None and os.altsep in value):
        candidate = Path(value).expanduser()
    else:
        found = shutil.which(value)
        if found is None:
            raise LockError("{} executable is not on PATH: {!r}".format(where, value))
        candidate = Path(found)
    absolute = Path(os.path.abspath(str(candidate)))
    if not absolute.is_file():
        raise LockError("{} is not a file: {}".format(where, absolute))
    if not os.access(str(absolute), os.X_OK):
        raise LockError("{} is not executable: {}".format(where, absolute))
    return absolute




def run_bytes(
    argv: Sequence[str],
    where: str,
    cwd: Optional[Path] = None,
    allowed: Sequence[int] = (0,),
    max_output: Optional[int] = MAX_COMMAND_OUTPUT,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            list(argv), cwd=str(cwd) if cwd is not None else None, env=env,
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise LockError("cannot run {}: {}".format(where, exc)) from exc
    if completed.returncode not in allowed:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise LockError("{} failed with exit {}: {}".format(where, completed.returncode, stderr))
    if max_output is not None and (
        len(completed.stdout) > max_output or len(completed.stderr) > max_output
    ):
        raise LockError("{} produced more than {} bytes".format(where, max_output))
    return completed


def command_provenance(completed: subprocess.CompletedProcess) -> Dict[str, Any]:
    return {
        "returncode": completed.returncode,
        "stdout_utf8": completed.stdout.decode("utf-8", errors="replace"),
        "stderr_utf8": completed.stderr.decode("utf-8", errors="replace"),
        "stdout_base64": base64.b64encode(completed.stdout).decode("ascii"),
        "stderr_base64": base64.b64encode(completed.stderr).decode("ascii"),
    }


PYTHON_PROBE = r'''
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import sysconfig
from pathlib import Path


def die(message):
    print(message, file=sys.stderr)
    raise SystemExit(2)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def under(path, root):
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            die("direct_url.json contains duplicate key %r" % key)
        result[key] = value
    return result


prefix = Path(sys.prefix).resolve()
distributions = []
seen = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name:
        die("installed distribution has no metadata Name")
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if SELECTED_DISTRIBUTIONS and normalized not in SELECTED_DISTRIBUTIONS:
        continue
    if normalized in seen:
        die("duplicate installed distribution identity %r" % normalized)
    seen.add(normalized)
    version = distribution.version
    if not isinstance(version, str) or not version:
        die("installed distribution %r has no version" % name)
    metadata_path = Path(getattr(distribution, "_path", "")).resolve()
    if not metadata_path.exists():
        die("installed distribution %r metadata path is missing" % name)

    direct_path = metadata_path / "direct_url.json"
    if direct_path.is_file():
        direct_raw = direct_path.read_bytes()
        try:
            direct_value = json.loads(
                direct_raw.decode("utf-8"), object_pairs_hook=unique_object,
                parse_constant=lambda value: die("non-finite direct URL JSON"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            die("installed distribution %r has invalid direct_url.json: %s" % (name, exc))
        if not isinstance(direct_value, dict):
            die("installed distribution %r direct_url.json is not an object" % name)
        direct_url = {
            "present": True,
            "sha256": hashlib.sha256(direct_raw).hexdigest(),
            "value": direct_value,
        }
    else:
        direct_url = {"present": False, "sha256": None, "value": None}

    record_path = metadata_path / "RECORD"
    record = None
    if record_path.is_file():
        if record_path.is_symlink():
            die("installed distribution %r RECORD is a symlink" % name)
        record_raw = record_path.read_bytes()
        try:
            rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"), newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            die("installed distribution %r has invalid RECORD: %s" % (name, exc))
        hashed = 0
        unhashed = 0
        entries = 0
        record_resolved = record_path.resolve()
        for row in rows:
            entries += 1
            if len(row) != 3 or not row[0]:
                die("installed distribution %r has malformed RECORD row" % name)
            relative, hash_spec, size_text = row
            if relative.endswith(".pyc"):
                # Bytecode caches are generated and may live under a
                # platform-specific pycache prefix outside the environment.
                unhashed += 1
                continue
            installed = Path(distribution.locate_file(relative))
            try:
                resolved = installed.resolve(strict=True)
            except OSError:
                die("installed distribution %r RECORD payload is missing: %s" % (name, relative))
            if not under(resolved, prefix):
                die("installed distribution %r RECORD escapes sys.prefix: %s" % (name, relative))
            if not resolved.is_file() or installed.is_symlink():
                die("installed distribution %r RECORD payload is not a regular file: %s" % (name, relative))
            if not hash_spec:
                if resolved != record_resolved and not relative.endswith(".pyc"):
                    die("installed distribution %r has unhashed RECORD payload: %s" % (name, relative))
                if size_text:
                    die("installed distribution %r unhashed RECORD row has a size" % name)
                unhashed += 1
                continue
            algorithm, separator, encoded = hash_spec.partition("=")
            if separator != "=" or algorithm != "sha256" or not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
                die("installed distribution %r RECORD uses an invalid non-SHA256 hash" % name)
            try:
                expected = base64.urlsafe_b64decode(encoded + "=").hex()
            except Exception:
                die("installed distribution %r RECORD has invalid base64" % name)
            if sha_file(resolved) != expected:
                die("installed distribution %r RECORD hash mismatch: %s" % (name, relative))
            if not size_text.isdigit() or resolved.stat().st_size != int(size_text):
                die("installed distribution %r RECORD size mismatch: %s" % (name, relative))
            hashed += 1
        if entries == 0 or hashed == 0:
            die("installed distribution %r RECORD has no hashed payloads" % name)
        record = {
            "sha256": hashlib.sha256(record_raw).hexdigest(),
            "entries": entries,
            "hashed_payloads": hashed,
            "unhashed_generated_or_record": unhashed,
        }
    elif record_path.exists():
        die("installed distribution %r RECORD is not a regular file" % name)

    distributions.append({
        "name": name,
        "normalized_name": normalized,
        "version": version,
        "metadata_path": str(metadata_path),
        "direct_url": direct_url,
        "record": record,
    })
missing_selected = sorted(set(SELECTED_DISTRIBUTIONS) - seen)
if missing_selected:
    die("selected distributions are not installed: %s" % ", ".join(missing_selected))


implementation_version = sys.implementation.version
result = {
    "reported_executable": sys.executable,
    "executable_sha256": sha_file(Path(sys.executable).resolve()),
    "prefix": str(prefix),
    "python": {
        "version": platform.python_version(),
        "full_version": sys.version,
        "version_info": list(sys.version_info),
        "implementation": sys.implementation.name,
        "implementation_version": list(implementation_version),
        "cache_tag": sys.implementation.cache_tag,
        "abi_flags": getattr(sys, "abiflags", ""),
        "sysconfig_platform": sysconfig.get_platform(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "libc": list(platform.libc_ver()),
        },
    },
    "distributions": distributions,
}
print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


def probe_python(
    python_request: str, distribution_names: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    launcher = resolve_launcher(python_request, "Python")
    selection = normalize_distribution_selection(distribution_names)
    probe = "SELECTED_DISTRIBUTIONS = {!r}\n".format(selection) + PYTHON_PROBE
    completed = run_bytes([str(launcher), "-I", "-c", probe], "target Python metadata probe")
    if completed.stderr:
        raise LockError("target Python metadata probe wrote to stderr: {}".format(
            completed.stderr.decode("utf-8", errors="replace")[-2000:]
        ))
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockError("target Python metadata probe returned invalid JSON: {}".format(exc)) from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("python"), dict)
        or not isinstance(value.get("distributions"), list)
        or not isinstance(value.get("executable_sha256"), str)
        or not HEX_SHA256.fullmatch(value["executable_sha256"])
    ):
        raise LockError("target Python metadata probe returned the wrong shape")
    python_identity = dict(value["python"])
    full_version = python_identity.pop("full_version", None)
    if not isinstance(full_version, str) or not full_version:
        raise LockError("target Python omitted its full version")
    python_identity["full_version_sha256"] = hashlib.sha256(
        full_version.encode("utf-8")
    ).hexdigest()
    python_identity["executable_sha256"] = value["executable_sha256"]
    provenance = {
        "requested_executable": python_request,
        "launcher_path": str(launcher),
        "reported_executable": str(value.get("reported_executable", "")),
        "prefix": str(value.get("prefix", "")),
        "full_version": full_version,
    }
    return {"python": python_identity, "raw_distributions": value["distributions"]}, provenance


def path_within(candidate: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(candidate), str(root)]) == str(root)
    except ValueError:
        return False


def direct_local_path(direct_value: Any) -> Optional[Path]:
    if not isinstance(direct_value, dict):
        return None
    url = direct_value.get("url")
    if not isinstance(url, str):
        return None
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "file" or parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    if parsed.netloc not in {"", "localhost"}:
        return None
    return Path(urllib.parse.unquote(parsed.path)).resolve()

def canonical_direct_url(
    direct_value: Any, binding: Optional[Tuple[str, str]], name: str
) -> Tuple[Dict[str, Any], str]:
    if not isinstance(direct_value, dict) or set(direct_value) - {
        "url", "archive_info", "dir_info", "vcs_info"
    } or "url" not in direct_value:
        raise LockError("distribution {!r} has unsupported direct_url.json fields".format(name))
    url = direct_value["url"]
    if not isinstance(url, str) or not url:
        raise LockError("distribution {!r} has invalid direct URL".format(name))
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "file":
        if parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.netloc not in {"", "localhost"}:
            raise LockError("distribution {!r} has an unsafe local direct URL".format(name))
        if binding is not None:
            normalized_url: Any = {"kind": "logical-source", "source_id": binding[1]}
            origin_kind = "logical-source"
        else:
            filename = Path(urllib.parse.unquote(parsed.path)).name
            if not filename:
                raise LockError("distribution {!r} local direct URL has no artifact name".format(name))
            normalized_url = {"kind": "local-artifact", "filename": filename}
            origin_kind = "local-artifact"
    elif parsed.scheme == "https":
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in {None, 443}
        ):
            raise LockError("distribution {!r} has a credentialed or mutable direct URL".format(name))
        normalized_url = urllib.parse.urlunsplit(
            ("https", parsed.hostname.lower(), parsed.path, "", "")
        )
        origin_kind = "public-https"
    else:
        raise LockError("distribution {!r} direct URL must be local file or public HTTPS".format(name))

    normalized: Dict[str, Any] = {"url": normalized_url}
    archive_info = direct_value.get("archive_info")
    if archive_info is not None:
        if not isinstance(archive_info, dict) or set(archive_info) - {"hash", "hashes"}:
            raise LockError("distribution {!r} has unsupported archive_info".format(name))
        archive_normalized: Dict[str, Any] = {}
        legacy_hash = archive_info.get("hash")
        if legacy_hash is not None:
            if not isinstance(legacy_hash, str) or not legacy_hash:
                raise LockError("distribution {!r} has invalid archive hash".format(name))
            archive_normalized["hash"] = legacy_hash
        hashes = archive_info.get("hashes")
        if hashes is not None:
            if (
                not isinstance(hashes, dict)
                or not hashes
                or not all(
                    isinstance(algorithm, str)
                    and isinstance(digest, str)
                    and re.fullmatch(r"[A-Za-z0-9]+", algorithm)
                    and re.fullmatch(r"[0-9A-Fa-f]+", digest)
                    for algorithm, digest in hashes.items()
                )
            ):
                raise LockError("distribution {!r} has invalid archive hashes".format(name))
            archive_normalized["hashes"] = dict(hashes)
        normalized["archive_info"] = archive_normalized
    dir_info = direct_value.get("dir_info")
    if dir_info is not None:
        if (
            not isinstance(dir_info, dict)
            or set(dir_info) - {"editable"}
            or (
                "editable" in dir_info
                and not isinstance(dir_info["editable"], bool)
            )
        ):
            raise LockError("distribution {!r} has invalid dir_info".format(name))
        normalized["dir_info"] = {"editable": dir_info.get("editable", False)}
    vcs_info = direct_value.get("vcs_info")
    if vcs_info is not None:
        if not isinstance(vcs_info, dict) or set(vcs_info) - {
            "vcs", "commit_id", "requested_revision"
        }:
            raise LockError("distribution {!r} has unsupported vcs_info".format(name))
        if vcs_info.get("vcs") != "git" or not isinstance(vcs_info.get("commit_id"), str) or not GIT_OID.fullmatch(vcs_info["commit_id"]):
            raise LockError("distribution {!r} VCS direct URL lacks a full Git commit".format(name))
        normalized["vcs_info"] = {
            "vcs": "git",
            "commit_id": vcs_info["commit_id"],
        }
    if sum(key in normalized for key in ("archive_info", "dir_info", "vcs_info")) != 1:
        raise LockError("distribution {!r} direct URL must have exactly one origin kind".format(name))
    return {
        "present": True,
        "sha256": canonical_sha256(normalized),
        "kind": origin_kind,
    }, origin_kind


def classify_distributions(raw: Sequence[Any], sources: Mapping[str, Path], bindings: Mapping[str, Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    identities: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    seen = set()
    used_bindings = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LockError("Python distribution {} is not an object".format(index))
        name = item.get("name")
        normalized = item.get("normalized_name")
        version = item.get("version")
        direct = item.get("direct_url")
        record = item.get("record")
        metadata_path_text = item.get("metadata_path")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(normalized, str)
            or not normalized
            or not isinstance(version, str)
            or not version
            or not isinstance(metadata_path_text, str)
            or not metadata_path_text
        ):
            raise LockError("Python distribution {} has incomplete metadata".format(index))
        if normalized != normalize_project(name) or normalized in seen:
            raise LockError("Python distribution names are not unique and normalized")
        seen.add(normalized)
        if not isinstance(direct, dict) or set(direct) != {"present", "sha256", "value"}:
            raise LockError("distribution {!r} has invalid direct URL metadata".format(name))
        present = direct["present"]
        direct_sha = direct["sha256"]
        if not isinstance(present, bool) or (present and (not isinstance(direct_sha, str) or not HEX_SHA256.fullmatch(direct_sha))) or (not present and direct_sha is not None):
            raise LockError("distribution {!r} has invalid direct URL identity".format(name))
        direct_value = direct["value"]
        editable = isinstance(direct_value, dict) and isinstance(direct_value.get("dir_info"), dict) and direct_value["dir_info"].get("editable") is True
        binding = bindings.get(normalized)
        if present:
            direct_identity, direct_origin_kind = canonical_direct_url(
                direct_value, binding, name
            )
        else:
            direct_identity = {"present": False, "sha256": None, "kind": "absent"}
            direct_origin_kind = "absent"

        if record is not None:
            if not isinstance(record, dict) or set(record) != {"sha256", "entries", "hashed_payloads", "unhashed_generated_or_record"}:
                raise LockError("distribution {!r} has invalid RECORD identity".format(name))
            if not isinstance(record["sha256"], str) or not HEX_SHA256.fullmatch(record["sha256"]):
                raise LockError("distribution {!r} has invalid RECORD SHA256".format(name))
            for field in ("entries", "hashed_payloads", "unhashed_generated_or_record"):
                if not isinstance(record[field], int) or isinstance(record[field], bool) or record[field] < 0:
                    raise LockError("distribution {!r} has invalid RECORD count".format(name))
            if record["entries"] <= 0 or record["hashed_payloads"] <= 0:
                raise LockError("distribution {!r} RECORD is empty".format(name))
        if record is None and binding is None:
            raise LockError("distribution {!r} has no RECORD and no explicit source classification".format(name))
        if editable and binding is None:
            raise LockError("editable distribution {!r} is not explicitly git-bound".format(name))
        if binding is not None:
            kind, source_id = binding
            if source_id not in sources:
                raise LockError("distribution {!r} binds unknown source {!r}".format(name, source_id))
            root = sources[source_id]
            local_direct = direct_local_path(direct_value)
            metadata_path = Path(metadata_path_text).resolve()
            if kind == "editable":
                if not editable or local_direct is None or not path_within(local_direct, root):
                    raise LockError("distribution {!r} is not an editable install from source {!r}".format(name, source_id))
            else:
                if record is not None:
                    raise LockError("non-editable source distribution {!r} unexpectedly has RECORD".format(name))
                if local_direct is not None and not path_within(local_direct, root):
                    raise LockError("distribution {!r} direct source does not match {!r}".format(name, source_id))
                if not path_within(metadata_path, root):
                    raise LockError("distribution {!r} metadata is not inside source {!r}".format(name, source_id))
            installation = {"kind": kind, "source_id": source_id}
            used_bindings.add(normalized)
        else:
            installation = {"kind": "wheel"}

        identities.append({
            "name": name,
            "normalized_name": normalized,
            "version": version,
            "direct_url": direct_identity,
            "record": record,
            "installation": installation,
        })
        provenance.append({
            "name": name,
            "metadata_path": metadata_path_text,
            "direct_url_kind": "editable" if editable else direct_origin_kind,
            "direct_url_raw_sha256": direct_sha,
            "direct_url": direct_value if present else None,
        })
    unused = set(bindings) - used_bindings
    if unused:
        raise LockError("source classifications name absent or inapplicable distributions: {}".format(", ".join(sorted(unused))))
    identities.sort(key=lambda item: (item["normalized_name"], item["name"]))
    provenance.sort(key=lambda item: normalize_project(item["name"]))
    return identities, provenance

def normalize_repository(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LockError("source origin must be a credential-free public HTTPS repository URL")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise LockError("source origin hostname is not public")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise LockError("source origin address is not public")
    if parsed.port not in {None, 443} or not parsed.path or parsed.path == "/" or "%" in parsed.path:
        raise LockError("source origin is not a stable public repository URL")
    path = parsed.path.rstrip("/")
    if not path.endswith(".git"):
        path += ".git"
    return urllib.parse.urlunsplit(("https", hostname, path, "", ""))


def git(source: Path, arguments: Sequence[str], where: str, allowed: Sequence[int] = (0,)) -> subprocess.CompletedProcess:
    clean_environment = dict(os.environ)
    for key in list(clean_environment):
        if key.startswith("GIT_"):
            del clean_environment[key]
    clean_environment["LC_ALL"] = "C"
    return run_bytes(
        ["git", "-c", "core.quotepath=false", "-C", str(source)] + list(arguments),
        where,
        allowed=allowed,
        max_output=None,
        env=clean_environment,
    )


def dirty_patch(source: Path) -> Tuple[bool, bytes]:
    status = git(source, ["status", "--porcelain=v1", "-z", "--untracked-files=all"], "git status")
    if not status.stdout:
        return False, b""
    fields = status.stdout.split(b"\0")
    untracked: List[str] = []
    index = 0
    while index < len(fields) and fields[index]:
        field = fields[index]
        if len(field) < 4:
            raise LockError("git status returned malformed porcelain output")
        code = field[:2]
        if any(97 <= byte <= 122 for byte in code):
            raise LockError("dirty submodule worktrees cannot be preserved by a source patch")
        path = os.fsdecode(field[3:])
        if code == b"??":
            untracked.append(path)
        index += 2 if b"R" in code or b"C" in code else 1
    tracked = git(
        source,
        ["diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
        "git dirty diff",
    ).stdout
    pieces = [tracked]
    for relative in sorted(untracked, key=lambda value: os.fsencode(value)):
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or (
                not (source / relative_path).is_file()
                and not (source / relative_path).is_symlink()
            )
        ):
            raise LockError("untracked source entry cannot be preserved: {!r}".format(relative))
        patch = git(
            source,
            ["diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-index", "--", "/dev/null", relative],
            "git untracked diff",
            allowed=(0, 1),
        ).stdout
        if not patch:
            raise LockError("untracked source file produced no preservable patch: {!r}".format(relative))
        pieces.append(patch)
    patch_bytes = b"".join(pieces)
    if not patch_bytes:
        raise LockError("dirty source produced no complete patch")
    return True, patch_bytes


def capture_source(source_id: str, path_text: str, allow_dirty: bool) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    require_id(source_id, "source ID")
    try:
        source = Path(path_text).expanduser().resolve(strict=True)
    except OSError as exc:
        raise LockError("cannot resolve source {!r}: {}".format(source_id, exc)) from exc
    if not source.is_dir():
        raise LockError("source {!r} is not a directory".format(source_id))
    repository = normalize_repository(git(source, ["config", "--get", "remote.origin.url"], "git origin").stdout.decode("utf-8", errors="strict").strip())
    commit = git(source, ["rev-parse", "--verify", "HEAD"], "git commit").stdout.decode("ascii").strip()
    tree = git(source, ["rev-parse", "--verify", "HEAD^{tree}"], "git tree").stdout.decode("ascii").strip()
    if not GIT_OID.fullmatch(commit) or not GIT_OID.fullmatch(tree):
        raise LockError("source {!r} has unsupported Git object identities".format(source_id))
    dirty, patch = dirty_patch(source)
    if dirty and not allow_dirty:
        raise LockError("source {!r} has uncommitted changes; use --allow-dirty with --patch-output".format(source_id))
    identity = {
        "id": source_id,
        "repository": repository,
        "commit": commit,
        "tree": tree,
        "dirty": dirty,
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest() if dirty else None,
    }
    provenance = {"id": source_id, "requested_path": path_text, "resolved_path": str(source)}
    return identity, provenance, patch


def write_new_bytes(path: Path, content: bytes) -> None:
    if not path.parent.is_dir():
        raise LockError("patch-output parent directory does not exist: {}".format(path.parent))
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise LockError("patch-output already exists: {}".format(path)) from exc


def capture_binary(binary_id: str, path_text: str, arguments: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require_id(binary_id, "binary ID")
    executable = resolve_executable(path_text, "binary {!r}".format(binary_id))
    completed = run_bytes([str(executable)] + list(arguments), "binary {!r} version command".format(binary_id))
    identity = {
        "id": binary_id,
        "executable_sha256": sha256_file(executable),
        "version_argv": list(arguments),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }
    provenance = {
        "id": binary_id,
        "requested_path": path_text,
        "resolved_path": str(executable),
        "version_command": [str(executable)] + list(arguments),
        "output": command_provenance(completed),
    }
    return identity, provenance


def capture_cuda(path_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    executable = resolve_executable(path_text, "nvidia-smi")
    fields = ["index", "name", "uuid", "pci.bus_id", "compute_cap", "memory.total", "driver_version", "vbios_version"]
    query_arguments = ["--query-gpu={}".format(",".join(fields)), "--format=csv,noheader,nounits"]
    query = run_bytes([str(executable)] + query_arguments, "nvidia-smi GPU query")
    version = run_bytes([str(executable), "--version"], "nvidia-smi version query")
    try:
        lines = query.stdout.decode("utf-8", errors="strict").splitlines()
        devices = []
        for line in lines:
            columns = [column.strip() for column in line.split(",")]
            if len(columns) != len(fields):
                raise LockError("nvidia-smi GPU query returned an unexpected column count")
            devices.append(dict(zip(fields, columns)))
    except UnicodeDecodeError as exc:
        raise LockError("nvidia-smi GPU query was not UTF-8") from exc
    if not devices:
        raise LockError("nvidia-smi reported no GPUs")
    devices.sort(key=lambda item: int(item["index"]))
    version_text = version.stdout.decode("utf-8", errors="strict")
    raw_facts: Dict[str, str] = {}
    for line in version_text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            raw_facts[key.strip().lower()] = value.strip()
    aliases = {
        "NVIDIA-SMI version": ("nvidia-smi version",),
        "NVML version": ("nvml version",),
        "DRIVER version": ("kmd version", "driver version"),
        "CUDA Version": ("cuda umd version", "cuda version"),
    }
    facts: Dict[str, str] = {}
    for canonical_name, candidates in aliases.items():
        value = next(
            (
                raw_facts[candidate]
                for candidate in candidates
                if candidate in raw_facts
                and not raw_facts[candidate].lower().startswith("deprecated")
            ),
            None,
        )
        if value is None:
            raise LockError(
                "nvidia-smi --version omitted usable CUDA, driver, NVML, or "
                "NVIDIA-SMI version"
            )
        facts[canonical_name] = value
    identity = {
        "nvidia_smi_sha256": sha256_file(executable),
        "query_argv": query_arguments,
        "query_stdout_sha256": hashlib.sha256(query.stdout).hexdigest(),
        "version_stdout_sha256": hashlib.sha256(version.stdout).hexdigest(),
        "versions": facts,
        "devices": devices,
    }
    provenance = {
        "requested_path": path_text,
        "resolved_path": str(executable),
        "query_output": command_provenance(query),
        "version_output": command_provenance(version),
    }
    return identity, provenance


def assert_canonical_safe(value: Any, where: str = "environment") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if "\x00" in value or "\n" in value or "\r" in value:
            raise LockError("{} contains control characters".format(where))
        if value.startswith("/") or WINDOWS_ABSOLUTE.match(value) or value.startswith("file:"):
            raise LockError("{} contains a host absolute path".format(where))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_canonical_safe(item, "{}[{}]".format(where, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LockError("{} has a non-string key".format(where))
            if SECRET_WORD.search(key):
                raise LockError("{} contains a credential-like field".format(where))
            assert_canonical_safe(item, "{}.{}".format(where, key))
        return
    raise LockError("{} contains a non-JSON value".format(where))


def capture_environment(
    python_path: str,
    binary_paths: Mapping[str, str],
    binary_arguments: Mapping[str, Sequence[str]],
    source_paths: Mapping[str, str],
    source_bindings: Mapping[str, Tuple[str, str]],
    distribution_names: Sequence[str],
    allow_dirty: bool,
    cuda_path: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, bytes]]:
    source_identities = []
    source_provenance = []
    source_roots: Dict[str, Path] = {}
    patches: Dict[str, bytes] = {}
    for source_id in sorted(source_paths):
        identity, provenance, patch = capture_source(source_id, source_paths[source_id], allow_dirty)
        source_identities.append(identity)
        source_provenance.append(provenance)
        source_roots[source_id] = Path(provenance["resolved_path"])
        if identity["dirty"]:
            patches[source_id] = patch

    distribution_selection = normalize_distribution_selection(distribution_names)
    probed, python_provenance = probe_python(python_path, distribution_selection)
    distributions, distribution_provenance = classify_distributions(
        probed["raw_distributions"], source_roots, source_bindings
    )

    binary_identities = []
    binary_provenance = []
    if set(binary_paths) != set(binary_arguments):
        raise LockError("binary paths and version commands name different IDs")
    for binary_id in sorted(binary_paths):
        identity, provenance = capture_binary(binary_id, binary_paths[binary_id], binary_arguments[binary_id])
        binary_identities.append(identity)
        binary_provenance.append(provenance)

    accelerator_identity = None
    accelerator_provenance = None
    if cuda_path is not None:
        accelerator_identity, accelerator_provenance = capture_cuda(cuda_path)

    environment = {
        "python": probed["python"],
        "distributions": distributions,
        "distribution_scope": {
            "mode": "selected" if distribution_selection else "all",
            "names": distribution_selection,
        },
        "binaries": binary_identities,
        "accelerator": accelerator_identity,
        "sources": source_identities,
    }
    validate_environment(environment)
    assert_canonical_safe(environment)
    provenance = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "invoking_python": sys.executable,
        "python": python_provenance,
        "distributions": distribution_provenance,
        "binaries": binary_provenance,
        "accelerator": accelerator_provenance,
        "sources": source_provenance,
    }
    return environment, provenance, patches


def identity_for(environment: object) -> str:
    return canonical_sha256({"schema": SCHEMA, "environment": environment})


def exact_object(value: Any, fields: set, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LockError("{} must contain exactly {}".format(where, ", ".join(sorted(fields))))
    return value


def digest_value(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise LockError("{} must be a lowercase SHA256".format(where))
    return value


def nonnegative_integer(value: Any, where: str, positive: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < (1 if positive else 0)
    ):
        raise LockError("{} must be a {}integer".format(where, "positive " if positive else "nonnegative "))
    return value


def validate_environment(environment: Dict[str, Any]) -> None:
    exact_object(
        environment,
        {
            "python", "distributions", "distribution_scope", "binaries",
            "accelerator", "sources",
        },
        "lock environment",
    )
    python = exact_object(
        environment["python"],
        {
            "version", "full_version_sha256", "version_info", "implementation",
            "implementation_version", "cache_tag", "abi_flags",
            "sysconfig_platform", "platform", "executable_sha256",
        },
        "lock Python",
    )
    for field in ("version", "implementation", "cache_tag", "abi_flags", "sysconfig_platform"):
        if not isinstance(python[field], str) or (field != "abi_flags" and not python[field]):
            raise LockError("lock Python {} must be a string".format(field))
    digest_value(python["full_version_sha256"], "lock Python full version")
    for field in ("version_info", "implementation_version"):
        parts = python[field]
        if (
            not isinstance(parts, list)
            or len(parts) != 5
            or not all(
                isinstance(parts[index], int) and not isinstance(parts[index], bool)
                for index in (0, 1, 2, 4)
            )
            or not isinstance(parts[3], str)
        ):
            raise LockError("lock Python {} is invalid".format(field))
    digest_value(python["executable_sha256"], "lock Python executable")
    platform_value = exact_object(
        python["platform"],
        {"system", "release", "version", "machine", "processor", "platform", "libc"},
        "lock Python platform",
    )
    for field in ("system", "release", "version", "machine", "processor", "platform"):
        if not isinstance(platform_value[field], str):
            raise LockError("lock Python platform.{} must be a string".format(field))
    if (
        not isinstance(platform_value["libc"], list)
        or len(platform_value["libc"]) != 2
        or not all(isinstance(part, str) for part in platform_value["libc"])
    ):
        raise LockError("lock Python platform.libc is invalid")

    sources = environment["sources"]
    if not isinstance(sources, list):
        raise LockError("lock sources must be an array")
    source_ids = set()
    source_order = []
    for index, item in enumerate(sources):
        source = exact_object(
            item,
            {"id", "repository", "commit", "tree", "dirty", "dirty_patch_sha256"},
            "lock source {}".format(index),
        )
        source_id = require_id(source["id"], "lock source ID")
        if source_id in source_ids:
            raise LockError("lock repeats source ID {!r}".format(source_id))
        source_ids.add(source_id)
        source_order.append(source_id)
        if normalize_repository(source["repository"]) != source["repository"]:
            raise LockError("lock source repository is not canonical")
        if not isinstance(source["commit"], str) or not GIT_OID.fullmatch(source["commit"]):
            raise LockError("lock source commit is invalid")
        if not isinstance(source["tree"], str) or not GIT_OID.fullmatch(source["tree"]):
            raise LockError("lock source tree is invalid")
        if len(source["commit"]) != len(source["tree"]):
            raise LockError("lock source Git object formats differ")
        if not isinstance(source["dirty"], bool):
            raise LockError("lock source dirty must be boolean")
        if source["dirty"]:
            digest_value(source["dirty_patch_sha256"], "lock dirty patch")
        elif source["dirty_patch_sha256"] is not None:
            raise LockError("clean lock source has a dirty patch SHA")
    if source_order != sorted(source_order):
        raise LockError("lock sources are not sorted by logical ID")

    distributions = environment["distributions"]
    if not isinstance(distributions, list):
        raise LockError("lock distributions must be an array")
    distribution_keys = []
    normalized_names = set()
    for index, item in enumerate(distributions):
        distribution = exact_object(
            item,
            {"name", "normalized_name", "version", "direct_url", "record", "installation"},
            "lock distribution {}".format(index),
        )
        name = distribution["name"]
        normalized = distribution["normalized_name"]
        version = distribution["version"]
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise LockError("lock distribution name and version must be nonempty strings")
        if not isinstance(normalized, str) or normalized != normalize_project(name) or normalized in normalized_names:
            raise LockError("lock distribution normalized names are invalid or repeated")
        normalized_names.add(normalized)
        distribution_keys.append((normalized, name))
        direct = exact_object(
            distribution["direct_url"], {"present", "sha256", "kind"},
            "lock distribution direct_url",
        )
        if not isinstance(direct["present"], bool) or direct["kind"] not in {
            "absent", "logical-source", "local-artifact", "public-https"
        }:
            raise LockError("lock distribution direct URL classification is invalid")
        if direct["present"]:
            digest_value(direct["sha256"], "lock distribution direct URL")
            if direct["kind"] == "absent":
                raise LockError("present direct URL is classified absent")
        elif direct["sha256"] is not None or direct["kind"] != "absent":
            raise LockError("absent direct URL has an identity")
        record = distribution["record"]
        if record is not None:
            record = exact_object(
                record,
                {"sha256", "entries", "hashed_payloads", "unhashed_generated_or_record"},
                "lock distribution RECORD",
            )
            digest_value(record["sha256"], "lock distribution RECORD")
            entries = nonnegative_integer(record["entries"], "RECORD entries", positive=True)
            hashed = nonnegative_integer(record["hashed_payloads"], "RECORD hashed payloads", positive=True)
            unhashed = nonnegative_integer(
                record["unhashed_generated_or_record"], "RECORD unhashed rows"
            )
            if entries != hashed + unhashed:
                raise LockError("lock distribution RECORD counts do not cover every row")
        installation = distribution["installation"]
        if not isinstance(installation, dict):
            raise LockError("lock distribution installation must be an object")
        kind = installation.get("kind")
        if kind == "wheel":
            if set(installation) != {"kind"} or record is None:
                raise LockError("wheel distribution must have exactly a hashed RECORD")
        elif kind in {"source", "editable"}:
            if set(installation) != {"kind", "source_id"} or installation["source_id"] not in source_ids:
                raise LockError("source distribution has an invalid Git binding")
            if kind == "source" and record is not None:
                raise LockError("source-classified distribution unexpectedly has RECORD")
            if kind == "editable" and direct["kind"] != "logical-source":
                raise LockError("editable distribution direct URL is not logically source-bound")
        else:
            raise LockError("lock distribution installation kind is invalid")
    if distribution_keys != sorted(distribution_keys):
        raise LockError("lock distributions are not canonically sorted")
    scope = exact_object(
        environment["distribution_scope"],
        {"mode", "names"},
        "lock distribution_scope",
    )
    if scope["mode"] not in {"all", "selected"}:
        raise LockError("lock distribution scope mode is invalid")
    names = scope["names"]
    if (
        not isinstance(names, list)
        or names != sorted(set(names))
        or not all(
            isinstance(name, str)
            and name == normalize_project(name)
            and SAFE_ID.fullmatch(name)
            for name in names
        )
    ):
        raise LockError("lock distribution scope names are invalid")
    if scope["mode"] == "all":
        if names:
            raise LockError("all-distribution scope must not list names")
    elif not names or set(names) != normalized_names:
        raise LockError(
            "selected-distribution scope must exactly name captured distributions"
        )

    binaries = environment["binaries"]
    if not isinstance(binaries, list):
        raise LockError("lock binaries must be an array")
    binary_ids = []
    for index, item in enumerate(binaries):
        binary = exact_object(
            item,
            {
                "id", "executable_sha256", "version_argv", "returncode",
                "stdout_sha256", "stderr_sha256",
            },
            "lock binary {}".format(index),
        )
        binary_id = require_id(binary["id"], "lock binary ID")
        if binary_id in binary_ids:
            raise LockError("lock repeats binary ID {!r}".format(binary_id))
        binary_ids.append(binary_id)
        for field in ("executable_sha256", "stdout_sha256", "stderr_sha256"):
            digest_value(binary[field], "lock binary {}".format(field))
        if binary["returncode"] != 0:
            raise LockError("lock binary version command did not succeed")
        arguments = binary["version_argv"]
        if (
            not isinstance(arguments, list)
            or not arguments
            or not all(
                isinstance(argument, str)
                and SAFE_ARGUMENT.fullmatch(argument)
                and not argument.startswith("/")
                and not WINDOWS_ABSOLUTE.match(argument)
                and not SECRET_WORD.search(argument)
                for argument in arguments
            )
        ):
            raise LockError("lock binary version command is unsafe")
    if binary_ids != sorted(binary_ids):
        raise LockError("lock binaries are not sorted by logical ID")

    accelerator = environment["accelerator"]
    if accelerator is not None:
        accelerator = exact_object(
            accelerator,
            {
                "nvidia_smi_sha256", "query_argv", "query_stdout_sha256",
                "version_stdout_sha256", "versions", "devices",
            },
            "lock accelerator",
        )
        for field in ("nvidia_smi_sha256", "query_stdout_sha256", "version_stdout_sha256"):
            digest_value(accelerator[field], "lock accelerator {}".format(field))
        if not isinstance(accelerator["query_argv"], list) or accelerator["query_argv"] != [
            "--query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,vbios_version",
            "--format=csv,noheader,nounits",
        ]:
            raise LockError("lock accelerator query command is not the fixed query")
        versions = exact_object(
            accelerator["versions"],
            {"NVIDIA-SMI version", "NVML version", "DRIVER version", "CUDA Version"},
            "lock accelerator versions",
        )
        if not all(isinstance(item, str) and item for item in versions.values()):
            raise LockError("lock accelerator versions must be nonempty strings")
        devices = accelerator["devices"]
        device_fields = {
            "index", "name", "uuid", "pci.bus_id", "compute_cap",
            "memory.total", "driver_version", "vbios_version",
        }
        if not isinstance(devices, list) or not devices:
            raise LockError("lock accelerator must contain at least one GPU")
        device_indices = []
        for index, item in enumerate(devices):
            device = exact_object(item, device_fields, "lock GPU {}".format(index))
            if not all(isinstance(part, str) and part for part in device.values()):
                raise LockError("lock GPU facts must be nonempty strings")
            try:
                device_indices.append(int(device["index"]))
            except ValueError as exc:
                raise LockError("lock GPU index must be an integer") from exc
        if device_indices != sorted(device_indices) or len(device_indices) != len(set(device_indices)):
            raise LockError("lock GPU indices are repeated or unsorted")


def validate_lock(value: object) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "identity_sha256", "environment", "provenance"}:
        raise LockError("lock must contain exactly schema, identity_sha256, environment, and provenance")
    if value["schema"] != SCHEMA:
        raise LockError("lock schema must be {!r}".format(SCHEMA))
    identity = value["identity_sha256"]
    if not isinstance(identity, str) or not HEX_SHA256.fullmatch(identity):
        raise LockError("lock identity_sha256 is invalid")
    environment = value["environment"]
    if not isinstance(environment, dict):
        raise LockError("lock environment must be an object")
    validate_environment(environment)
    assert_canonical_safe(environment)
    actual = identity_for(environment)
    if actual != identity:
        raise LockError("lock canonical identity does not match its environment")
    if not isinstance(value["provenance"], dict):
        raise LockError("lock provenance must be an object")
    return value


def bindings_from_environment(environment: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
    distributions = environment.get("distributions")
    if not isinstance(distributions, list):
        raise LockError("lock distributions must be an array")
    result: Dict[str, Tuple[str, str]] = {}
    for item in distributions:
        if not isinstance(item, dict):
            raise LockError("lock distribution is not an object")
        normalized = item.get("normalized_name")
        installation = item.get("installation")
        if not isinstance(normalized, str) or not isinstance(installation, dict):
            raise LockError("lock distribution has invalid installation metadata")
        kind = installation.get("kind")
        if kind in {"source", "editable"}:
            source_id = installation.get("source_id")
            if not isinstance(source_id, str):
                raise LockError("lock source distribution omits source_id")
            result[normalized] = (kind, source_id)
        elif kind != "wheel" or set(installation) != {"kind"}:
            raise LockError("lock distribution installation kind is invalid")
    return result


def expected_binary_arguments(environment: Mapping[str, Any]) -> Dict[str, List[str]]:
    binaries = environment.get("binaries")
    if not isinstance(binaries, list):
        raise LockError("lock binaries must be an array")
    result: Dict[str, List[str]] = {}
    for item in binaries:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("version_argv"), list):
            raise LockError("lock binary has invalid version command")
        binary_id = item["id"]
        require_id(binary_id, "lock binary ID")
        arguments = item["version_argv"]
        if binary_id in result or not arguments or not all(isinstance(argument, str) and SAFE_ARGUMENT.fullmatch(argument) for argument in arguments):
            raise LockError("lock binary version command is invalid")
        result[binary_id] = list(arguments)
    return result


def expected_ids(environment: Mapping[str, Any], field: str) -> set:
    values = environment.get(field)
    if not isinstance(values, list):
        raise LockError("lock {} must be an array".format(field))
    result = set()
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise LockError("lock {} item has no ID".format(field))
        require_id(item["id"], "lock {} ID".format(field))
        if item["id"] in result:
            raise LockError("lock {} repeats ID {!r}".format(field, item["id"]))
        result.add(item["id"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    capture = subparsers.add_parser("capture", help="capture a new environment lock")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--python", required=True, help="target Python executable")
    capture.add_argument(
        "--distribution",
        action="append",
        default=[],
        metavar="PROJECT",
        help="capture only this installed project (repeatable; default: all)",
    )
    capture.add_argument("--binary", action="append", default=[], metavar="ID=PATH")
    capture.add_argument("--binary-arg", action="append", default=[], metavar="ID=ARG")
    capture.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    capture.add_argument("--source-distribution", action="append", default=[], metavar="PROJECT=KIND:SOURCE_ID")
    capture.add_argument("--allow-dirty", action="store_true")
    capture.add_argument("--patch-output", action="append", default=[], metavar="ID=PATH")
    capture.add_argument("--cuda", action="store_true", help="capture NVIDIA CUDA, driver, and GPU facts")
    capture.add_argument("--nvidia-smi", default="nvidia-smi")

    verify = subparsers.add_parser("verify", help="rehash an environment against a lock")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--python", required=True, help="target Python executable")
    verify.add_argument("--binary", action="append", default=[], metavar="ID=PATH")
    verify.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    verify.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.mode == "capture":
            if args.output.exists() or args.output.is_symlink():
                raise LockError("output already exists: {}".format(args.output))
            binaries = parse_assignments(args.binary, "--binary")
            binary_arguments = parse_binary_arguments(args.binary_arg, list(binaries))
            sources = parse_assignments(args.source, "--source")
            bindings = parse_source_bindings(args.source_distribution)
            patch_outputs = parse_assignments(args.patch_output, "--patch-output")
            if set(patch_outputs) - set(sources):
                raise LockError("--patch-output names an undeclared source")
            environment, provenance, patches = capture_environment(
                args.python, binaries, binary_arguments, sources, bindings,
                args.distribution, args.allow_dirty,
                args.nvidia_smi if args.cuda else None,
            )
            if set(patches) != set(patch_outputs):
                missing = set(patches) - set(patch_outputs)
                extra = set(patch_outputs) - set(patches)
                if missing:
                    raise LockError("dirty sources need --patch-output: {}".format(", ".join(sorted(missing))))
                raise LockError("clean sources must not have --patch-output: {}".format(", ".join(sorted(extra))))
            for source_id in sorted(patches):
                output_path = Path(patch_outputs[source_id]).expanduser()
                write_new_bytes(output_path, patches[source_id])
                for source_provenance in provenance["sources"]:
                    if source_provenance["id"] == source_id:
                        source_provenance["patch_output"] = str(output_path.resolve())
            lock = {
                "schema": SCHEMA,
                "identity_sha256": identity_for(environment),
                "environment": environment,
                "provenance": provenance,
            }
            atomic_write_json(args.output, lock)
            print(lock["identity_sha256"])
            return

        lock = validate_lock(load_strict_json(args.lock))
        expected = lock["environment"]
        binaries = parse_assignments(args.binary, "--binary")
        sources = parse_assignments(args.source, "--source")
        expected_binary_ids = expected_ids(expected, "binaries")
        expected_source_ids = expected_ids(expected, "sources")
        if set(binaries) != expected_binary_ids:
            raise LockError("--binary IDs do not exactly match lock: expected {}".format(", ".join(sorted(expected_binary_ids))))
        if set(sources) != expected_source_ids:
            raise LockError("--source IDs do not exactly match lock: expected {}".format(", ".join(sorted(expected_source_ids))))
        binary_arguments = expected_binary_arguments(expected)
        bindings = bindings_from_environment(expected)
        cuda_expected = expected.get("accelerator") is not None
        scope = exact_object(
            expected["distribution_scope"],
            {"mode", "names"},
            "lock distribution_scope",
        )
        distribution_names = scope["names"] if scope["mode"] == "selected" else []
        current, _provenance, patches = capture_environment(
            args.python, binaries, binary_arguments, sources, bindings,
            distribution_names=distribution_names,
            allow_dirty=any(isinstance(item, dict) and item.get("dirty") is True for item in expected["sources"]),
            cuda_path=args.nvidia_smi if cuda_expected else None,
        )
        if patches and not any(isinstance(item, dict) and item.get("dirty") is True for item in expected["sources"]):
            raise LockError("current sources are unexpectedly dirty")
        current_identity = identity_for(current)
        if current_identity != lock["identity_sha256"] or current != expected:
            raise LockError("current environment does not match lock (current identity {})".format(current_identity))
        print("verified {}".format(current_identity))
    except (LockError, OSError, UnicodeError, ValueError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
