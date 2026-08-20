#!/usr/bin/env python3
"""Fail-closed verifier for the versioned Final Frontier campaign record."""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from frontier_common import (
    atomic_write_json,
    canonical_bytes,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)

CAMPAIGN_SCHEMA = "frontier-g01-campaign-v1"
PREREG_SCHEMA = "frontier-g01-prereg-v1"
RESULT_SCHEMA = "frontier-g01-result-v1"
CORRECTION_SCHEMA = "frontier-g01-correction-v1"
DISPOSITION_SCHEMA = "frontier-g01-disposition-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

REQUIRED_G0_PATHS = {
    "volatile-preservation": "g0/volatile-preservation.json",
    "bf16-census": "g0/bf16-census.json",
    "compatibility-registry": "g0/compatibility-registry.json",
    "converter-environment": "g0/converter-environment.json",
    "runtime-environment": "g0/runtime-environment.json",
    "resource-ledger": "g0/resource-ledger.json",
    "data-manifest": "g0/data-manifest.json",
    "power-plan": "g0/power-plan.json",
    "autoround-audit": "g0/autoround-audit.json",
    "runtime-source": "g0/runtime-source.json",
    "incumbent-reproduction": "g0/incumbent-reproduction.json",
    "maintenance-transaction": "g0/maintenance-transaction.json",
    "campaign-contract": "campaign-contract.json",
    "upstream-record": "g0/upstream-record.json",
}
REQUIRED_EXPERIMENT_KINDS = {
    "baseline-pilot",
    "converter-qualification",
    "direct-marginals",
    "sparse-options",
    "autoround-bridge",
    "rescomp",
    "qkv-pilot",
    "candidate",
}
NON_CANDIDATE_KINDS = REQUIRED_EXPERIMENT_KINDS - {"candidate"}
TERMINAL_DISPOSITIONS = {"gate_a_pass", "three_candidate_no_go"}
RESULT_DISPOSITIONS = {"pass", "no_go", "unsupported"}


class IncompleteCampaign(Exception):
    """The campaign has not yet assembled every required piece."""


class MalformedEvidence(Exception):
    """Present evidence violates the versioned campaign contract."""


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedEvidence(f"{context} must be a JSON object")
    return value


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedEvidence(f"{context} must be a JSON array")
    return value


def _exact_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys - value.keys()
    extra = value.keys() - keys
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise MalformedEvidence(f"{context} has " + " and ".join(details))


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise MalformedEvidence(f"{context} must be a safe nonempty identifier")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedEvidence(f"{context} must be a nonempty string")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MalformedEvidence(f"{context} must be a lowercase SHA256")
    return value


def _strict_load(path: Path, context: str) -> object:
    try:
        return load_strict_json(path)
    except FileNotFoundError as exc:
        raise IncompleteCampaign(f"missing {context}: {path}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise MalformedEvidence(f"cannot load {context} {path}: {exc}") from exc


class CampaignVerifier:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.references: dict[str, tuple[str, str | None]] = {}

    def _relative_path(self, value: object, context: str) -> tuple[str, Path]:
        text = _nonempty_string(value, f"{context}.path")
        if "\\" in text:
            raise MalformedEvidence(f"{context}.path must use POSIX separators")
        pure = PurePosixPath(text)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/")):
            raise MalformedEvidence(f"{context}.path is not a clean campaign-relative path")
        path = self.root.joinpath(*pure.parts)
        cursor = self.root
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise MalformedEvidence(f"{context}.path may not traverse a symlink: {text}")
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise MalformedEvidence(f"{context}.path escapes the campaign root") from exc
        return text, path

    def _file(self, reference: object, context: str, *, canonical: bool) -> tuple[Path, object | None]:
        ref = _object(reference, context)
        expected_keys = {"path", "sha256"}
        if canonical:
            expected_keys.add("canonical_sha256")
        _exact_keys(ref, expected_keys, context)
        relative, path = self._relative_path(ref["path"], context)
        expected_file_hash = _sha256(ref["sha256"], f"{context}.sha256")
        expected_canonical_hash = (
            _sha256(ref["canonical_sha256"], f"{context}.canonical_sha256")
            if canonical
            else None
        )
        previous = self.references.get(relative)
        identity = (expected_file_hash, expected_canonical_hash)
        if previous is not None and previous != identity:
            raise MalformedEvidence(f"conflicting digests declared for {relative}")
        self.references[relative] = identity
        if not path.exists():
            raise IncompleteCampaign(f"missing referenced artifact: {relative}")
        if not path.is_file():
            raise MalformedEvidence(f"referenced artifact is not a regular file: {relative}")
        try:
            if path.stat().st_size == 0:
                raise MalformedEvidence(f"referenced artifact is empty: {relative}")
            actual_file_hash = sha256_file(path)
        except OSError as exc:
            raise MalformedEvidence(f"cannot read referenced artifact {relative}: {exc}") from exc
        if actual_file_hash != expected_file_hash:
            raise MalformedEvidence(
                f"whole-file SHA256 mismatch for {relative}: expected {expected_file_hash}, got {actual_file_hash}"
            )
        parsed: object | None = None
        if canonical:
            parsed = _strict_load(path, context)
            if parsed == {} or parsed == []:
                raise MalformedEvidence(f"referenced JSON artifact is semantically empty: {relative}")
            actual_canonical_hash = canonical_sha256(parsed)
            if actual_canonical_hash != expected_canonical_hash:
                raise MalformedEvidence(
                    f"canonical SHA256 mismatch for {relative}: expected {expected_canonical_hash}, got {actual_canonical_hash}"
                )
        return path, parsed

    def _json_reference(
        self,
        reference: object,
        context: str,
        *,
        expected_path: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        path, parsed = self._file(reference, context, canonical=True)
        ref = _object(reference, context)
        if expected_path is not None and ref["path"] != expected_path:
            raise MalformedEvidence(
                f"{context}.path must be {expected_path!r}, got {ref['path']!r}"
            )
        return path, _object(parsed, f"{context} content")

    def _artifact_reference(self, reference: object, context: str) -> None:
        ref = _object(reference, context)
        path_text = ref.get("path")
        canonical = isinstance(path_text, str) and path_text.endswith(".json")
        self._file(reference, context, canonical=canonical)

    def _validate_g0(self, value: object) -> None:
        g0 = _object(value, "manifest.g0")
        _exact_keys(g0, {"status", "artifacts"}, "manifest.g0")
        if g0["status"] != "pass":
            raise IncompleteCampaign("manifest.g0.status is not pass")
        artifacts = _object(g0["artifacts"], "manifest.g0.artifacts")
        missing = REQUIRED_G0_PATHS.keys() - artifacts.keys()
        if missing:
            raise IncompleteCampaign(f"manifest lacks required G0 artifacts: {sorted(missing)}")
        for artifact_id, reference in artifacts.items():
            _identifier(artifact_id, f"G0 artifact id {artifact_id!r}")
            entry = _object(reference, f"G0 artifact {artifact_id}")
            _exact_keys(
                entry,
                {"path", "sha256", "canonical_sha256", "validated"},
                f"G0 artifact {artifact_id}",
            )
            if entry["validated"] is not True:
                raise IncompleteCampaign(f"G0 artifact {artifact_id} is not declared validated")
            expected = REQUIRED_G0_PATHS.get(artifact_id)
            path, parsed = self._json_reference(
                {key: entry[key] for key in ("path", "sha256", "canonical_sha256")},
                f"G0 artifact {artifact_id}",
                expected_path=expected,
            )
            if expected is None and not str(path.relative_to(self.root)).startswith("g0/"):
                raise MalformedEvidence(f"additional G0 artifact {artifact_id} must be under g0/")
            _ = parsed

    def _validate_prereg(
        self,
        value: dict[str, Any],
        experiment: dict[str, Any],
        all_experiment_ids: set[str],
    ) -> set[str]:
        context = f"preregistration {experiment['experiment_id']}"
        keys = {
            "schema_version",
            "campaign_id",
            "experiment_id",
            "parent_experiment_id",
            "kind",
            "candidate_id",
            "hypothesis",
            "falsifier",
            "changed_variables",
            "held_variables",
            "identities",
            "analysis_code_sha256",
            "candidate_options",
            "engineering_margins",
            "decision_rules",
            "maximum_spend",
            "planned_output_paths",
        }
        _exact_keys(value, keys, context)
        if value["schema_version"] != PREREG_SCHEMA:
            raise MalformedEvidence(f"{context} has unsupported schema_version")
        for name in ("campaign_id", "experiment_id", "kind", "candidate_id"):
            if value[name] != experiment[name]:
                raise MalformedEvidence(f"{context}.{name} does not match the manifest")
        parent = value["parent_experiment_id"]
        if parent is not None:
            parent = _identifier(parent, f"{context}.parent_experiment_id")
            if parent == experiment["experiment_id"] or parent not in all_experiment_ids:
                raise MalformedEvidence(f"{context}.parent_experiment_id is not another declared experiment")
        _nonempty_string(value["hypothesis"], f"{context}.hypothesis")
        _nonempty_string(value["falsifier"], f"{context}.falsifier")
        _object(value["changed_variables"], f"{context}.changed_variables")
        _object(value["held_variables"], f"{context}.held_variables")
        if not _object(value["identities"], f"{context}.identities"):
            raise MalformedEvidence(f"{context}.identities may not be empty")
        _sha256(value["analysis_code_sha256"], f"{context}.analysis_code_sha256")
        if not _array(value["candidate_options"], f"{context}.candidate_options"):
            raise MalformedEvidence(f"{context}.candidate_options may not be empty")
        _object(value["engineering_margins"], f"{context}.engineering_margins")
        rules = _object(value["decision_rules"], f"{context}.decision_rules")
        _exact_keys(rules, {"accept", "reject", "futility", "abort"}, f"{context}.decision_rules")
        for name, rule in rules.items():
            _nonempty_string(rule, f"{context}.decision_rules.{name}")
        spend = _object(value["maximum_spend"], f"{context}.maximum_spend")
        _exact_keys(spend, {"gpu_hours", "usd"}, f"{context}.maximum_spend")
        for name, amount in spend.items():
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise MalformedEvidence(f"{context}.maximum_spend.{name} must be numeric")
            if amount < 0 or (isinstance(amount, float) and not math.isfinite(amount)):
                raise MalformedEvidence(f"{context}.maximum_spend.{name} must be finite and nonnegative")
        outputs = _array(value["planned_output_paths"], f"{context}.planned_output_paths")
        if not outputs:
            raise MalformedEvidence(f"{context}.planned_output_paths may not be empty")
        planned: set[str] = set()
        for index, output in enumerate(outputs):
            relative, _ = self._relative_path(output, f"{context}.planned_output_paths[{index}]")
            experiment_prefix = f"experiments/{experiment['experiment_id']}/"
            if not relative.startswith(experiment_prefix):
                raise MalformedEvidence(
                    f"{context} planned outputs must stay under {experiment_prefix}"
                )
            if relative in planned:
                raise MalformedEvidence(f"{context} repeats planned output {relative}")
            planned.add(relative)
        return planned

    def _validate_result_record(
        self,
        value: dict[str, Any],
        experiment: dict[str, Any],
        prereg_digest: str,
        *,
        correction: bool,
        supersedes_digest: str | None,
    ) -> str:
        context = (
            f"correction for {experiment['experiment_id']}"
            if correction
            else f"result for {experiment['experiment_id']}"
        )
        keys = {
            "schema_version",
            "campaign_id",
            "experiment_id",
            "kind",
            "candidate_id",
            "prereg_canonical_sha256",
            "artifacts",
            "results",
            "uncertainty",
            "failures",
            "cost",
            "disposition",
            "unsupported_reason",
        }
        if correction:
            keys |= {"supersedes_canonical_sha256", "reason"}
        _exact_keys(value, keys, context)
        expected_schema = CORRECTION_SCHEMA if correction else RESULT_SCHEMA
        if value["schema_version"] != expected_schema:
            raise MalformedEvidence(f"{context} has unsupported schema_version")
        for name in ("campaign_id", "experiment_id", "kind", "candidate_id"):
            if value[name] != experiment[name]:
                raise MalformedEvidence(f"{context}.{name} does not match the manifest")
        if value["prereg_canonical_sha256"] != prereg_digest:
            raise MalformedEvidence(f"{context} does not link to its immutable preregistration")
        if correction:
            if value["supersedes_canonical_sha256"] != supersedes_digest:
                raise MalformedEvidence(f"{context} does not supersede the immediately preceding object")
            _nonempty_string(value["reason"], f"{context}.reason")
        artifacts = _object(value["artifacts"], f"{context}.artifacts")
        if "raw-log" not in artifacts:
            raise MalformedEvidence(f"{context}.artifacts lacks required raw-log")
        for role, reference in artifacts.items():
            _identifier(role, f"{context} artifact role {role!r}")
            artifact = _object(reference, f"{context}.artifacts.{role}")
            experiment_prefix = f"experiments/{experiment['experiment_id']}/"
            if not isinstance(artifact.get("path"), str) or not artifact["path"].startswith(
                experiment_prefix
            ):
                raise MalformedEvidence(f"{context} artifacts must stay under {experiment_prefix}")
            self._artifact_reference(artifact, f"{context}.artifacts.{role}")
        results = _object(value["results"], f"{context}.results")
        _object(value["uncertainty"], f"{context}.uncertainty")
        failures = _array(value["failures"], f"{context}.failures")
        _object(value["cost"], f"{context}.cost")
        disposition = value["disposition"]
        if disposition not in RESULT_DISPOSITIONS:
            raise MalformedEvidence(f"{context}.disposition must be one of {sorted(RESULT_DISPOSITIONS)}")
        unsupported_reason = value["unsupported_reason"]
        if disposition == "unsupported":
            if experiment["kind"] != "autoround-bridge":
                raise MalformedEvidence("only autoround-bridge may terminate unsupported")
            if unsupported_reason != "pinned_consumer_cannot_build":
                raise MalformedEvidence(
                    "autoround-bridge unsupported requires pinned_consumer_cannot_build"
                )
            if results.get("pinned_consumer_build") is not False or not failures:
                raise MalformedEvidence(
                    "autoround-bridge unsupported requires a failed pinned-consumer build result"
                )
        elif unsupported_reason is not None:
            raise MalformedEvidence(f"{context}.unsupported_reason must be null unless unsupported")
        return disposition

    def _validate_experiments(
        self, value: object, campaign_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        entries = _array(value, "manifest.experiments")
        if not entries:
            raise IncompleteCampaign("manifest.experiments is empty")
        experiments: list[dict[str, Any]] = []
        experiment_ids: set[str] = set()
        kinds: set[str] = set()
        candidate_ids: set[str] = set()
        candidate_entries: set[str] = set()
        entry_keys = {"experiment_id", "kind", "candidate_id", "prereg", "result", "corrections"}
        for index, item in enumerate(entries):
            entry = _object(item, f"manifest.experiments[{index}]")
            _exact_keys(entry, entry_keys, f"manifest.experiments[{index}]")
            experiment_id = _identifier(entry["experiment_id"], f"experiment {index} id")
            kind = _identifier(entry["kind"], f"experiment {experiment_id} kind")
            if experiment_id in experiment_ids:
                raise MalformedEvidence(f"duplicate experiment_id {experiment_id}")
            experiment_ids.add(experiment_id)
            kinds.add(kind)
            candidate_id = entry["candidate_id"]
            if kind == "candidate":
                candidate_id = _identifier(candidate_id, f"experiment {experiment_id}.candidate_id")
                if candidate_id in candidate_entries:
                    raise MalformedEvidence(f"candidate {candidate_id} has more than one opportunity result")
                candidate_entries.add(candidate_id)
                candidate_ids.add(candidate_id)
            elif candidate_id is not None:
                raise MalformedEvidence(f"non-candidate experiment {experiment_id} must have null candidate_id")
            experiments.append(entry)
        missing_kinds = REQUIRED_EXPERIMENT_KINDS - kinds
        if missing_kinds:
            raise IncompleteCampaign(f"manifest lacks required experiment kinds: {sorted(missing_kinds)}")
        for kind in NON_CANDIDATE_KINDS:
            if sum(entry["kind"] == kind for entry in experiments) != 1:
                raise MalformedEvidence(f"required experiment kind {kind} must occur exactly once")
        if not candidate_ids:
            raise IncompleteCampaign("manifest has no candidate opportunity")
        if len(candidate_ids) > 3:
            raise MalformedEvidence("campaign declares more than three unique candidate IDs")

        effective: dict[str, str] = {}
        for entry in experiments:
            experiment_id = entry["experiment_id"]
            expected_entry = {**entry, "campaign_id": campaign_id}
            prereg_path = f"experiments/{experiment_id}/prereg.json"
            result_path = f"experiments/{experiment_id}/result.json"
            _, prereg = self._json_reference(
                entry["prereg"], f"experiment {experiment_id} prereg", expected_path=prereg_path
            )
            if prereg.get("campaign_id") != campaign_id:
                raise MalformedEvidence(f"preregistration {experiment_id} campaign_id mismatch")
            planned = self._validate_prereg(prereg, expected_entry, experiment_ids)
            if result_path not in planned:
                raise MalformedEvidence(f"preregistration {experiment_id} did not plan its result path")
            prereg_digest = canonical_sha256(prereg)
            _, result = self._json_reference(
                entry["result"], f"experiment {experiment_id} result", expected_path=result_path
            )
            disposition = self._validate_result_record(
                result, expected_entry, prereg_digest, correction=False, supersedes_digest=None
            )
            for role, artifact_ref in _object(result["artifacts"], "result.artifacts").items():
                if _object(artifact_ref, f"result artifact {role}")["path"] not in planned:
                    raise MalformedEvidence(
                        f"result artifact {role} for {experiment_id} was not preregistered"
                    )
            previous_digest = canonical_sha256(result)
            corrections = _array(entry["corrections"], f"experiment {experiment_id}.corrections")
            declared_correction_paths: set[str] = set()
            for correction_index, correction_ref in enumerate(corrections, 1):
                correction_path = f"experiments/{experiment_id}/correction-{correction_index:03d}.json"
                declared_correction_paths.add(correction_path)
                _, correction_value = self._json_reference(
                    correction_ref,
                    f"experiment {experiment_id} correction {correction_index}",
                    expected_path=correction_path,
                )
                disposition = self._validate_result_record(
                    correction_value,
                    expected_entry,
                    prereg_digest,
                    correction=True,
                    supersedes_digest=previous_digest,
                )
                previous_digest = canonical_sha256(correction_value)
            experiment_directory = self.root / "experiments" / experiment_id
            if experiment_directory.exists():
                actual_corrections = {
                    str(path.relative_to(self.root))
                    for path in experiment_directory.glob("correction-*.json")
                    if path.is_file()
                }
                undeclared = actual_corrections - declared_correction_paths
                if undeclared:
                    raise MalformedEvidence(
                        f"experiment {experiment_id} has undeclared corrections: {sorted(undeclared)}"
                    )
            effective[experiment_id] = disposition
        return experiments, effective

    def _validate_terminal(
        self,
        reference: object,
        campaign_id: str,
        experiments: list[dict[str, Any]],
        effective: dict[str, str],
    ) -> str:
        _, disposition = self._json_reference(
            reference, "terminal disposition", expected_path="disposition.json"
        )
        keys = {
            "schema_version",
            "campaign_id",
            "disposition",
            "candidate_ids",
            "selected_candidate_id",
            "experiment_ids",
            "reason",
        }
        _exact_keys(disposition, keys, "terminal disposition")
        if disposition["schema_version"] != DISPOSITION_SCHEMA:
            raise MalformedEvidence("terminal disposition has unsupported schema_version")
        if disposition["campaign_id"] != campaign_id:
            raise MalformedEvidence("terminal disposition campaign_id mismatch")
        terminal = disposition["disposition"]
        if terminal not in TERMINAL_DISPOSITIONS:
            raise MalformedEvidence(
                f"terminal disposition must be exactly one of {sorted(TERMINAL_DISPOSITIONS)}"
            )
        _nonempty_string(disposition["reason"], "terminal disposition.reason")
        expected_experiments = [entry["experiment_id"] for entry in experiments]
        if disposition["experiment_ids"] != expected_experiments:
            raise MalformedEvidence("terminal disposition does not cover the exact experiment sequence")
        candidate_entries = [entry for entry in experiments if entry["kind"] == "candidate"]
        expected_candidates = [entry["candidate_id"] for entry in candidate_entries]
        if disposition["candidate_ids"] != expected_candidates:
            raise MalformedEvidence("terminal disposition does not cover the exact candidate sequence")
        for entry in experiments:
            outcome = effective[entry["experiment_id"]]
            if entry["kind"] != "candidate" and outcome not in {"pass", "no_go", "unsupported"}:
                raise MalformedEvidence(f"required experiment {entry['experiment_id']} is not terminal")
        candidate_outcomes = {
            entry["candidate_id"]: effective[entry["experiment_id"]] for entry in candidate_entries
        }
        if terminal == "gate_a_pass":
            selected = disposition["selected_candidate_id"]
            if selected not in candidate_outcomes:
                raise MalformedEvidence("gate_a_pass must select a declared candidate")
            if candidate_outcomes[selected] != "pass":
                raise MalformedEvidence("gate_a_pass selected candidate does not have a pass result")
            if any(outcome not in {"pass", "no_go"} for outcome in candidate_outcomes.values()):
                raise MalformedEvidence("gate_a_pass has a nonterminal candidate opportunity")
        else:
            if disposition["selected_candidate_id"] is not None:
                raise MalformedEvidence("three_candidate_no_go may not select a candidate")
            if len(candidate_entries) != 3 or len(candidate_outcomes) != 3:
                raise MalformedEvidence("three_candidate_no_go requires exactly three candidate opportunities")
            if any(outcome != "no_go" for outcome in candidate_outcomes.values()):
                raise MalformedEvidence("three_candidate_no_go requires three no_go candidate results")
        return terminal

    def verify(self) -> str:
        manifest_path = self.root / "manifest.json"
        manifest = _object(_strict_load(manifest_path, "campaign manifest"), "campaign manifest")
        keys = {"schema_version", "campaign_id", "g0", "experiments", "terminal_disposition"}
        _exact_keys(manifest, keys, "campaign manifest")
        if manifest["schema_version"] != CAMPAIGN_SCHEMA:
            raise MalformedEvidence("campaign manifest has unsupported schema_version")
        campaign_id = _identifier(manifest["campaign_id"], "manifest.campaign_id")
        self._validate_g0(manifest["g0"])
        experiments, effective = self._validate_experiments(manifest["experiments"], campaign_id)
        return self._validate_terminal(
            manifest["terminal_disposition"], campaign_id, experiments, effective
        )


def _json_ref(path: Path, root: Path) -> dict[str, str]:
    value = load_strict_json(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_sha256(value),
    }


def _expect_exception(exception_type: type[Exception], action: Any, description: str) -> None:
    try:
        action()
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(f"{description} raised {type(exc).__name__}, not {exception_type.__name__}") from exc
    raise AssertionError(f"{description} was accepted")


def _run_self_test() -> None:
    if canonical_bytes({"b": 1, "a": "é"}) != b'{"a":"\xc3\xa9","b":1}':
        raise AssertionError("canonical byte encoding is unstable")
    with tempfile.TemporaryDirectory(prefix="frontier-g01-self-test-") as temporary:
        root = Path(temporary) / "campaign"
        root.mkdir()
        strict_path = root / "strict.json"
        strict_path.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        _expect_exception(ValueError, lambda: load_strict_json(strict_path), "duplicate JSON key")
        strict_path.write_text('{"x":NaN}\n', encoding="utf-8")
        _expect_exception(ValueError, lambda: load_strict_json(strict_path), "NaN")
        strict_path.write_text('{"x":1e9999}\n', encoding="utf-8")
        _expect_exception(ValueError, lambda: load_strict_json(strict_path), "overflowing number")
        strict_path.unlink()

        campaign_id = "self-test"
        g0_artifacts: dict[str, dict[str, Any]] = {}
        for artifact_id, relative in REQUIRED_G0_PATHS.items():
            path = root / relative
            atomic_write_json(
                path,
                {"schema_version": "frontier-g01-g0-artifact-v1", "artifact_id": artifact_id},
            )
            g0_artifacts[artifact_id] = {**_json_ref(path, root), "validated": True}

        experiments: list[dict[str, Any]] = []
        for ordinal, kind in enumerate(sorted(NON_CANDIDATE_KINDS) + ["candidate"], 1):
            experiment_id = f"e{ordinal:02d}-{kind}"
            candidate_id = "candidate-1" if kind == "candidate" else None
            directory = root / "experiments" / experiment_id
            raw_log = directory / "raw.log"
            raw_log.parent.mkdir(parents=True, exist_ok=True)
            raw_log.write_text(f"completed {experiment_id}\n", encoding="utf-8")
            result_path = directory / "result.json"
            prereg = {
                "schema_version": PREREG_SCHEMA,
                "campaign_id": campaign_id,
                "experiment_id": experiment_id,
                "parent_experiment_id": None,
                "kind": kind,
                "candidate_id": candidate_id,
                "hypothesis": "The frozen decision rule is satisfied.",
                "falsifier": "The frozen decision rule is not satisfied.",
                "changed_variables": {},
                "held_variables": {},
                "identities": {"fixture": "self-test"},
                "analysis_code_sha256": "0" * 64,
                "candidate_options": ["frozen"],
                "engineering_margins": {},
                "decision_rules": {
                    "accept": "pass",
                    "reject": "no_go",
                    "futility": "stop",
                    "abort": "invalid evidence",
                },
                "maximum_spend": {"gpu_hours": 0, "usd": 0},
                "planned_output_paths": [
                    result_path.relative_to(root).as_posix(),
                    raw_log.relative_to(root).as_posix(),
                ],
            }
            prereg_path = directory / "prereg.json"
            atomic_write_json(prereg_path, prereg)
            result = {
                "schema_version": RESULT_SCHEMA,
                "campaign_id": campaign_id,
                "experiment_id": experiment_id,
                "kind": kind,
                "candidate_id": candidate_id,
                "prereg_canonical_sha256": canonical_sha256(prereg),
                "artifacts": {
                    "raw-log": {
                        "path": raw_log.relative_to(root).as_posix(),
                        "sha256": sha256_file(raw_log),
                    }
                },
                "results": {"self_test": True},
                "uncertainty": {},
                "failures": [],
                "cost": {"gpu_hours": 0, "usd": 0},
                "disposition": "pass",
                "unsupported_reason": None,
            }
            atomic_write_json(result_path, result)
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "kind": kind,
                    "candidate_id": candidate_id,
                    "prereg": _json_ref(prereg_path, root),
                    "result": _json_ref(result_path, root),
                    "corrections": [],
                }
            )

        disposition_path = root / "disposition.json"
        disposition = {
            "schema_version": DISPOSITION_SCHEMA,
            "campaign_id": campaign_id,
            "disposition": "gate_a_pass",
            "candidate_ids": ["candidate-1"],
            "selected_candidate_id": "candidate-1",
            "experiment_ids": [entry["experiment_id"] for entry in experiments],
            "reason": "The self-test candidate passed its frozen rule.",
        }
        atomic_write_json(disposition_path, disposition)
        manifest = {
            "schema_version": CAMPAIGN_SCHEMA,
            "campaign_id": campaign_id,
            "g0": {"status": "pass", "artifacts": g0_artifacts},
            "experiments": experiments,
            "terminal_disposition": _json_ref(disposition_path, root),
        }
        manifest_path = root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        if CampaignVerifier(root).verify() != "gate_a_pass":
            raise AssertionError("valid fixture did not produce gate_a_pass")

        disposition_path.rename(root / "disposition.saved")
        _expect_exception(
            IncompleteCampaign,
            lambda: CampaignVerifier(root).verify(),
            "missing terminal disposition",
        )
        (root / "disposition.saved").rename(disposition_path)

        candidate_raw = next(root.glob("experiments/*candidate*/raw.log"))
        original_raw = candidate_raw.read_bytes()
        candidate_raw.write_bytes(original_raw + b"tamper\n")
        _expect_exception(
            MalformedEvidence,
            lambda: CampaignVerifier(root).verify(),
            "tampered referenced artifact",
        )
        candidate_raw.write_bytes(original_raw)

        candidate_entry = next(entry for entry in manifest["experiments"] if entry["kind"] == "candidate")
        candidate_result_path = root / candidate_entry["result"]["path"]
        original_result = _object(
            load_strict_json(candidate_result_path), "self-test candidate result"
        )
        broken_result = copy.deepcopy(original_result)
        broken_result["prereg_canonical_sha256"] = "f" * 64
        atomic_write_json(candidate_result_path, broken_result)
        candidate_entry["result"] = _json_ref(candidate_result_path, root)
        atomic_write_json(manifest_path, manifest)
        _expect_exception(
            MalformedEvidence,
            lambda: CampaignVerifier(root).verify(),
            "broken immutable preregistration link",
        )
        atomic_write_json(candidate_result_path, original_result)
        candidate_entry["result"] = _json_ref(candidate_result_path, root)

        bad_disposition = copy.deepcopy(disposition)
        bad_disposition["disposition"] = "three_candidate_no_go"
        bad_disposition["selected_candidate_id"] = None
        atomic_write_json(disposition_path, bad_disposition)
        manifest["terminal_disposition"] = _json_ref(disposition_path, root)
        atomic_write_json(manifest_path, manifest)
        _expect_exception(
            MalformedEvidence,
            lambda: CampaignVerifier(root).verify(),
            "premature three-candidate no-go",
        )
        atomic_write_json(disposition_path, disposition)
        manifest["terminal_disposition"] = _json_ref(disposition_path, root)

        original_experiments = manifest["experiments"]
        clones = []
        for number in range(2, 5):
            clone = copy.deepcopy(candidate_entry)
            clone["experiment_id"] = f"overflow-{number}"
            clone["candidate_id"] = f"candidate-{number}"
            clones.append(clone)
        manifest["experiments"] = original_experiments + clones
        atomic_write_json(manifest_path, manifest)
        _expect_exception(
            MalformedEvidence,
            lambda: CampaignVerifier(root).verify(),
            "four-candidate campaign",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify strict, hashed Final Frontier G0/G1 campaign evidence.",
        epilog="Exit 0: valid terminal campaign; 1: incomplete campaign; 2: malformed evidence.",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        help="campaign root containing manifest.json, g0/, experiments/, and disposition.json",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise canonical/strict/atomic primitives and verifier rejection cases in a temporary directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    campaign: Path | None = arguments.campaign
    if arguments.self_test and campaign is not None:
        parser.error("--campaign and --self-test are mutually exclusive")
    if not arguments.self_test and campaign is None:
        parser.error("one of --campaign or --self-test is required")
    try:
        if arguments.self_test:
            _run_self_test()
            print("PASS: frontier G0/G1 verifier self-test")
            return 0
        assert campaign is not None
        terminal = CampaignVerifier(campaign).verify()
        print(f"PASS: verified terminal disposition {terminal}")
        return 0
    except IncompleteCampaign as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 1
    except (MalformedEvidence, AssertionError, OSError, UnicodeError, ValueError) as exc:
        print(f"MALFORMED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
