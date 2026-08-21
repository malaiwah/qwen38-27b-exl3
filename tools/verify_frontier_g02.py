#!/usr/bin/env python3
"""Verify the method-neutral Frontier G02 pre-candidate campaign."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from frontier_common import canonical_sha256, load_strict_json, sha256_file

MANIFEST_SCHEMA = "frontier-g02-manifest-v1"
CONTRACT_SCHEMA = "frontier-g02-contract-v1"
CORRECTION_SCHEMA = "frontier-g02-contract-correction-v1"
PREREG_SCHEMA = "frontier-g02-prereg-v1"
RESULT_SCHEMA = "frontier-g02-result-v1"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
EXPECTED_EXPERIMENTS = (
    "g1-exact-fidelity-control",
    "g1-exact-fidelity-control-v2",
    "g1-exact-fidelity-control-n3",
    "g1-exact-fidelity-kld",
)
EXPECTED_KINDS = {
    "g1-exact-fidelity-control": "exact-fidelity-control",
    "g1-exact-fidelity-control-v2": "exact-fidelity-control",
    "g1-exact-fidelity-control-n3": "exact-fidelity-control",
    "g1-exact-fidelity-kld": "exact-fidelity-kld-control",
}


class IncompleteCampaign(Exception):
    """Required campaign evidence is not yet present."""


class MalformedEvidence(Exception):
    """Present evidence violates the frozen G02 contract."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedEvidence(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedEvidence(f"{label} must be an array")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - value.keys()
    extra = value.keys() - keys
    if missing or extra:
        raise MalformedEvidence(
            f"{label} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise MalformedEvidence(f"{label} must be a lowercase SHA256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise MalformedEvidence(f"{label} must be a safe identifier")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedEvidence(f"{label} must be a nonempty string")
    return value


class Verifier:
    def __init__(self, campaign: Path):
        self.campaign = campaign.resolve()
        self.repo = self.campaign.parent.parent.resolve()
        self.refs: dict[str, tuple[str, str | None]] = {}

    def _path(self, base: Path, raw: object, label: str) -> tuple[str, Path]:
        text = _nonempty(raw, f"{label}.path")
        if "\\" in text:
            raise MalformedEvidence(f"{label}.path must use POSIX separators")
        pure = PurePosixPath(text)
        if pure.is_absolute() or any(part in {"", "."} for part in text.split("/")):
            raise MalformedEvidence(f"{label}.path is not a clean relative path")
        path = base.joinpath(*pure.parts)
        cursor = base
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise MalformedEvidence(f"{label}.path traverses a symlink")
        try:
            path.resolve(strict=False).relative_to(self.repo)
        except ValueError as exc:
            raise MalformedEvidence(f"{label}.path escapes repository") from exc
        return text, path

    def _ref(
        self,
        value: object,
        label: str,
        *,
        base: Path | None = None,
        canonical: bool = True,
        expected: str | None = None,
    ) -> tuple[Path, object | None]:
        ref = _object(value, label)
        keys = {"path", "sha256"} | ({"canonical_sha256"} if canonical else set())
        _exact(ref, keys, label)
        base = self.campaign if base is None else base
        text, path = self._path(base, ref["path"], label)
        if expected is not None and text != expected:
            raise MalformedEvidence(f"{label}.path must be {expected!r}, got {text!r}")
        file_sha = _sha(ref["sha256"], f"{label}.sha256")
        canonical_sha = (
            _sha(ref["canonical_sha256"], f"{label}.canonical_sha256")
            if canonical
            else None
        )
        key = str(path.resolve(strict=False))
        identity = (file_sha, canonical_sha)
        if key in self.refs and self.refs[key] != identity:
            raise MalformedEvidence(f"conflicting identities declared for {text}")
        self.refs[key] = identity
        if not path.exists():
            raise IncompleteCampaign(f"missing referenced artifact: {text}")
        if not path.is_file() or path.stat().st_size == 0:
            raise MalformedEvidence(
                f"referenced artifact is not a nonempty file: {text}"
            )
        if sha256_file(path) != file_sha:
            raise MalformedEvidence(f"whole-file SHA256 differs for {text}")
        parsed = None
        if canonical:
            parsed = load_strict_json(path)
            if canonical_sha256(parsed) != canonical_sha:
                raise MalformedEvidence(f"canonical SHA256 differs for {text}")
        return path, parsed

    def _validate_contract(self, value: object) -> dict[str, Any]:
        contract = _object(value, "campaign contract")
        expected = {
            "schema_version",
            "campaign_id",
            "status",
            "frozen_utc",
            "objective",
            "primary_control",
            "inherited_evidence",
            "baseline_execution",
            "data_policy",
            "candidate_policy",
            "promotion_gates",
            "rental",
            "planned_outputs",
        }
        _exact(contract, expected, "campaign contract")
        if (
            contract["schema_version"] != CONTRACT_SCHEMA
            or contract["campaign_id"] != "frontier-g02"
        ):
            raise MalformedEvidence("campaign contract identity differs")
        primary = _object(contract["primary_control"], "primary_control")
        if (
            primary.get("profile") != "fidelity"
            or primary.get("embedding_bits") != 6
            or primary.get("max_model_len") != 238400
        ):
            raise MalformedEvidence("primary fidelity control differs")
        control_path, _ = self._ref(
            primary["contract"],
            "primary_control.contract",
            base=self.campaign,
            canonical=False,
        )
        control_obj = _object(load_strict_json(control_path), "fidelity control")
        if (
            control_obj.get("status") != "frozen"
            or _object(
                control_obj.get("candidate_contract"), "control candidate_contract"
            ).get("baseline_must_pass_absolute_gate_a")
            is not False
        ):
            raise MalformedEvidence("frozen fidelity control semantics differ")
        inherited = _object(contract["inherited_evidence"], "inherited_evidence")
        if set(inherited) != {
            "data_manifest",
            "power_plan",
            "resource_ledger",
            "runtime_source",
            "runtime_environment",
            "converter_environment",
            "maintenance_transaction",
        }:
            raise MalformedEvidence("inherited evidence set differs")
        for name, ref in inherited.items():
            self._ref(
                ref, f"inherited_evidence.{name}", base=self.campaign, canonical=False
            )
        baseline = _object(contract["baseline_execution"], "baseline_execution")
        for name in ("profile_gate", "baseline", "callback", "transaction"):
            self._ref(
                baseline[name],
                f"baseline_execution.{name}",
                base=self.campaign,
                canonical=False,
            )
        policy = _object(contract["candidate_policy"], "candidate_policy")
        if (
            policy.get("research_results_observed") is not False
            or policy.get("selection_frozen") is not False
            or policy.get("maximum_fresh_sequential_candidates") != 3
            or policy.get("bank_splicing_allowed") is not False
        ):
            raise MalformedEvidence("method-neutral candidate policy differs")
        slots = _array(
            policy.get("candidate_slots"), "candidate_policy.candidate_slots"
        )
        expected_slots = [
            {"candidate_id": f"candidate-{index:02d}", "status": "unopened"}
            for index in range(1, 4)
        ]
        if slots != expected_slots:
            raise MalformedEvidence("contract candidate slots differ")
        data = _object(contract["data_policy"], "data_policy")
        if data.get("sealed_final_opened") is not False:
            raise MalformedEvidence("sealed final set was opened")
        rental = _object(contract["rental"], "rental")
        if (
            rental.get("authorized") is not False
            or rental.get("diagnostic_exception") is not None
        ):
            raise MalformedEvidence("rental was authorized before a candidate pass")
        return contract

    def _validate_correction(self, value: object) -> dict[str, Any]:
        correction = _object(value, "contract correction")
        expected = {
            "schema_version",
            "campaign_id",
            "correction_id",
            "supersedes",
            "reason",
            "failed_experiment",
            "runtime_extension",
            "effective_baseline_execution",
            "unchanged_invariants",
        }
        _exact(correction, expected, "contract correction")
        if (
            correction["schema_version"] != CORRECTION_SCHEMA
            or correction["campaign_id"] != "frontier-g02"
            or correction["correction_id"] != "campaign-contract-correction-001"
        ):
            raise MalformedEvidence("contract correction identity differs")
        self._ref(
            correction["supersedes"],
            "contract correction supersedes",
            expected="campaign-contract.json",
        )
        runtime = _object(correction["runtime_extension"], "runtime_extension")
        if (
            runtime.get("base_commit") != "b19029d2309b26c4942425e52b74a0e6dd5d141e"
            or runtime.get("integrated_commit")
            != "8e5b5a2c6d955270f30ce9f3c8baaffa2da80710"
            or _object(runtime.get("focused_tests"), "focused_tests").get("passed")
            != 19
        ):
            raise MalformedEvidence("clean INT6 runtime correction differs")
        self._ref(
            runtime["patch"],
            "runtime_extension.patch",
            base=self.campaign,
            canonical=False,
        )
        compatibility = _object(runtime.get("compatibility"), "compatibility")
        if (
            compatibility.get("historical_packed_bytes_equal") is not True
            or compatibility.get("historical_scales_equal") is not True
        ):
            raise MalformedEvidence("historical INT6 compatibility is not proven")
        _, compatibility_path = self._path(
            self.campaign,
            compatibility.get("path"),
            "runtime_extension.compatibility",
        )
        if (
            not compatibility_path.is_file()
            or sha256_file(compatibility_path)
            != _sha(
                compatibility.get("sha256"),
                "runtime_extension.compatibility.sha256",
            )
        ):
            raise MalformedEvidence("historical compatibility receipt hash differs")
        return correction

    def _validate_prereg(self, value: object, experiment_id: str) -> dict[str, Any]:
        prereg = _object(value, f"prereg {experiment_id}")
        if prereg.get("schema_version") != PREREG_SCHEMA:
            raise MalformedEvidence(f"prereg {experiment_id} schema differs")
        if (
            prereg.get("campaign_id") != "frontier-g02"
            or prereg.get("experiment_id") != experiment_id
        ):
            raise MalformedEvidence(f"prereg {experiment_id} identity differs")
        if (
            prereg.get("kind") != EXPECTED_KINDS[experiment_id]
            or prereg.get("candidate_id") is not None
        ):
            raise MalformedEvidence(f"prereg {experiment_id} kind differs")
        outputs = _array(
            prereg.get("planned_output_paths"), f"prereg {experiment_id} outputs"
        )
        for required in (
            f"experiments/{experiment_id}/raw-log.json",
            f"experiments/{experiment_id}/result.json",
        ):
            if required not in outputs:
                raise MalformedEvidence(
                    f"prereg {experiment_id} did not plan {required}"
                )
        _nonempty(prereg.get("hypothesis"), f"prereg {experiment_id}.hypothesis")
        _nonempty(prereg.get("falsifier"), f"prereg {experiment_id}.falsifier")
        return prereg

    def _validate_result(
        self, value: object, experiment_id: str, prereg: dict[str, Any]
    ) -> dict[str, Any]:
        result = _object(value, f"result {experiment_id}")
        expected = {
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
        }
        _exact(result, expected, f"result {experiment_id}")
        if (
            result["schema_version"] != RESULT_SCHEMA
            or result["campaign_id"] != "frontier-g02"
            or result["experiment_id"] != experiment_id
            or result["kind"] != EXPECTED_KINDS[experiment_id]
            or result["candidate_id"] is not None
        ):
            raise MalformedEvidence(f"result {experiment_id} identity differs")
        if result["prereg_canonical_sha256"] != canonical_sha256(prereg):
            raise MalformedEvidence(f"result {experiment_id} prereg link differs")
        artifacts = _object(result["artifacts"], f"result {experiment_id}.artifacts")
        if set(artifacts) != {"raw-log"}:
            raise MalformedEvidence(f"result {experiment_id} artifact roles differ")
        self._ref(
            artifacts["raw-log"],
            f"result {experiment_id}.raw-log",
            expected=f"experiments/{experiment_id}/raw-log.json",
        )
        if result["disposition"] not in {"pass", "no_go"}:
            raise MalformedEvidence(f"result {experiment_id} disposition differs")
        failures = _array(result["failures"], f"result {experiment_id}.failures")
        if (result["disposition"] == "pass") == bool(failures):
            raise MalformedEvidence(
                f"result {experiment_id} failures/disposition disagree"
            )
        cost = _object(result["cost"], f"result {experiment_id}.cost")
        for name, amount in cost.items():
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or amount < 0
            ):
                raise MalformedEvidence(f"result {experiment_id}.cost.{name} differs")
            if isinstance(amount, float) and not math.isfinite(amount):
                raise MalformedEvidence(
                    f"result {experiment_id}.cost.{name} is non-finite"
                )
        return result

    def verify(self) -> str:
        manifest_path = self.campaign / "manifest.json"
        if not manifest_path.exists():
            raise IncompleteCampaign("manifest.json is absent")
        manifest = _object(load_strict_json(manifest_path), "manifest")
        expected = {
            "schema_version",
            "campaign_id",
            "status",
            "contract",
            "contract_corrections",
            "experiments",
            "fidelity_control_experiment_id",
            "baseline_control_experiment_id",
            "candidate_slots",
            "sealed_final",
            "method_infrastructure",
            "rental",
        }
        _exact(manifest, expected, "manifest")
        if (
            manifest["schema_version"] != MANIFEST_SCHEMA
            or manifest["campaign_id"] != "frontier-g02"
        ):
            raise MalformedEvidence("manifest identity differs")
        _, contract_value = self._ref(
            manifest["contract"], "manifest.contract", expected="campaign-contract.json"
        )
        contract = self._validate_contract(contract_value)
        corrections = _array(manifest["contract_corrections"], "contract_corrections")
        if len(corrections) != 1:
            raise MalformedEvidence("exactly one contract correction is required")
        _, correction_value = self._ref(
            corrections[0],
            "manifest.contract_corrections[0]",
            expected="campaign-contract-correction-001.json",
        )
        self._validate_correction(correction_value)
        entries = _array(manifest["experiments"], "manifest.experiments")
        ids = [
            entry.get("experiment_id")
            for entry in map(_object, entries, ["experiment"] * len(entries))
        ]
        if tuple(ids) != EXPECTED_EXPERIMENTS:
            raise MalformedEvidence(f"experiment sequence differs: {ids}")
        outcomes: dict[str, str] = {}
        for raw_entry in entries:
            entry = _object(raw_entry, "experiment entry")
            _exact(entry, {"experiment_id", "prereg", "result"}, "experiment entry")
            experiment_id = _identifier(entry["experiment_id"], "experiment_id")
            _, prereg_value = self._ref(
                entry["prereg"],
                f"experiment {experiment_id}.prereg",
                expected=f"experiments/{experiment_id}/prereg.json",
            )
            prereg = self._validate_prereg(prereg_value, experiment_id)
            _, result_value = self._ref(
                entry["result"],
                f"experiment {experiment_id}.result",
                expected=f"experiments/{experiment_id}/result.json",
            )
            result = self._validate_result(result_value, experiment_id, prereg)
            outcomes[experiment_id] = result["disposition"]
        if (
            outcomes[EXPECTED_EXPERIMENTS[0]] != "no_go"
            or outcomes[EXPECTED_EXPERIMENTS[1]] != "no_go"
        ):
            raise MalformedEvidence("failed predecessor controls were rewritten")
        baseline_id = manifest["baseline_control_experiment_id"]
        if baseline_id != EXPECTED_EXPERIMENTS[2]:
            raise MalformedEvidence("effective baseline control experiment differs")
        fidelity_id = manifest["fidelity_control_experiment_id"]
        if fidelity_id != EXPECTED_EXPERIMENTS[3]:
            raise MalformedEvidence("effective fidelity control experiment differs")
        effective_pass = (
            outcomes[baseline_id] == "pass" and outcomes[fidelity_id] == "pass"
        )
        slots = _array(manifest["candidate_slots"], "manifest.candidate_slots")
        if slots != contract["candidate_policy"]["candidate_slots"]:
            raise MalformedEvidence("manifest candidate slots differ from contract")
        sealed = _object(manifest["sealed_final"], "sealed_final")
        if sealed != {"id": "E-final-v7", "opened": False}:
            raise MalformedEvidence("sealed final state differs")
        method = _object(manifest["method_infrastructure"], "method_infrastructure")
        _exact(
            method,
            {
                "status",
                "capture_prereg",
                "capture_result",
                "run_prereg",
                "run_result",
            },
            "method_infrastructure",
        )
        if method["status"] not in {"pending", "control_ready"}:
            raise MalformedEvidence("method infrastructure status differs")
        if method["status"] == "control_ready":
            _, capture_prereg = self._ref(
                method["capture_prereg"], "method_infrastructure.capture_prereg"
            )
            _, capture_result = self._ref(
                method["capture_result"], "method_infrastructure.capture_result"
            )
            _, run_prereg = self._ref(
                method["run_prereg"], "method_infrastructure.run_prereg"
            )
            _, run_result = self._ref(
                method["run_result"], "method_infrastructure.run_result"
            )
            if (
                _object(capture_prereg, "capture prereg").get("schema_version")
                != "qwen38-trellis-v3-prereg-correction/1"
                or _object(capture_result, "capture result").get("status") != "pass"
                or _object(run_prereg, "run prereg").get("schema_version")
                != "qwen38-trellis-v3-run-prereg/1"
                or _object(run_result, "run result").get("schema")
                != "qwen38-trellis-v3-result/1"
                or _object(run_result, "run result").get("status") != "pass"
                or _object(run_result, "run result").get("method_claims_allowed")
                is not False
            ):
                raise MalformedEvidence("v3 control infrastructure semantics differ")
        elif any(
            method[name] is not None
            for name in ("capture_result", "run_result")
        ):
            raise MalformedEvidence("pending method infrastructure declares results")
        rental = _object(manifest["rental"], "rental")
        if rental.get("authorized") is not False:
            raise MalformedEvidence("pre-candidate manifest authorizes rental")
        expected_status = (
            "pre_candidate_ready"
            if effective_pass and method["status"] == "control_ready"
            else "blocked"
        )
        if manifest["status"] != expected_status:
            raise MalformedEvidence(
                f"manifest status {manifest['status']!r} != {expected_status!r}"
            )
        return expected_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status = Verifier(args.campaign).verify()
        print(f"PASS: verified frontier-g02 status {status}")
        return 0
    except IncompleteCampaign as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 1
    except (MalformedEvidence, OSError, UnicodeError, ValueError) as exc:
        print(f"MALFORMED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
