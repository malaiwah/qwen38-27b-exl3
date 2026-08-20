#!/usr/bin/env python3
"""Build and verify a candidate-blind Final Frontier family-bootstrap power plan."""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_common import atomic_write_json, canonical_sha256, load_strict_json

BASELINES_SCHEMA = "qwen38-frontier-power-baselines/1"
MARGIN_SCHEMA = "qwen38-frontier-power-margin-spec/1"
PLAN_SCHEMA = "qwen38-frontier-power-plan/1"
UNKNOWN = {"", "unknown", "unset", "unresolved", "n/a", "na", "none", "null", "tbd"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOMAINS = {"fidelity", "performance", "capability"}
DIRECTIONS = {"higher_is_better", "lower_is_better"}
TESTS = {"superiority", "noninferiority"}
MULTIPLICITY = {"bonferroni", "sidak"}
EFFECT_SOURCES = {"external_literature", "preregistered_requirement"}


class PowerError(ValueError):
    """Closed validation failure for power inputs or generated plans."""


def fail(message: str) -> NoReturn:
    raise PowerError(message)


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


def enum(value: object, choices: set[str], label: str) -> str:
    parsed = text(value, label)
    if parsed not in choices:
        fail(f"{label} must be one of {sorted(choices)}, got {parsed!r}")
    return parsed


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be boolean")
    return value


def integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < 0 or (positive and value == 0):
        fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{label} must be a finite number")
    parsed = float(value)
    if parsed < 0 or (positive and parsed == 0):
        fail(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return parsed


def sha256(value: object, label: str) -> str:
    parsed = text(value, label).lower()
    if SHA256_RE.fullmatch(parsed) is None:
        fail(f"{label} must be a lowercase 64-character SHA256")
    return parsed


def validate_baselines(value: object) -> dict[str, Any]:
    document = obj(value, "baselines")
    exact_keys(document, {"schema", "measurement_role", "repetitions_per_family", "metrics", "rows"}, "baselines")
    if document["schema"] != BASELINES_SCHEMA:
        fail(f"baselines.schema must be {BASELINES_SCHEMA!r}")
    if document["measurement_role"] != "baseline_only":
        fail("baselines.measurement_role must be exactly 'baseline_only'; candidate or result-derived rows are forbidden")
    repetitions = integer(document["repetitions_per_family"], "baselines.repetitions_per_family", positive=True)
    if repetitions < 2:
        fail("baselines.repetitions_per_family must be at least two")
    metrics = [text(raw, f"baselines.metrics[{index}]") for index, raw in enumerate(array(document["metrics"], "baselines.metrics"))]
    if not metrics or len(metrics) != len(set(metrics)) or metrics != sorted(metrics):
        fail("baselines.metrics must be a non-empty sorted list without duplicates")

    rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, int]] = set()
    seen_sources: set[str] = set()
    for index, raw in enumerate(array(document["rows"], "baselines.rows")):
        label = f"baselines.rows[{index}]"
        row = obj(raw, label)
        exact_keys(row, {"family", "stratum", "repetition", "source", "values"}, label)
        family = text(row["family"], f"{label}.family")
        stratum = text(row["stratum"], f"{label}.stratum")
        repetition = integer(row["repetition"], f"{label}.repetition")
        if repetition >= repetitions:
            fail(f"{label}.repetition must be below repetitions_per_family")
        row_key = (family, repetition)
        if row_key in seen_rows:
            fail(f"duplicate baseline family/repetition row {row_key}")
        seen_rows.add(row_key)
        source = obj(row["source"], f"{label}.source")
        exact_keys(source, {"artifact_sha256", "row_sha256", "measurement_role"}, f"{label}.source")
        if source["measurement_role"] != "baseline":
            fail(f"{label}.source.measurement_role must be exactly 'baseline'")
        normalized_source = {
            "artifact_sha256": sha256(source["artifact_sha256"], f"{label}.source.artifact_sha256"),
            "row_sha256": sha256(source["row_sha256"], f"{label}.source.row_sha256"),
            "measurement_role": "baseline",
        }
        if normalized_source["row_sha256"] in seen_sources:
            fail(f"duplicate immutable baseline row identity in {label}")
        seen_sources.add(normalized_source["row_sha256"])
        values_obj = obj(row["values"], f"{label}.values")
        if set(values_obj) != set(metrics):
            fail(f"{label}.values must contain every and only declared baseline metric")
        values = {metric: number(values_obj[metric], f"{label}.values.{metric}") for metric in metrics}
        rows.append({"family": family, "stratum": stratum, "repetition": repetition, "source": normalized_source, "values": values})
    if not rows:
        fail("baselines.rows must not be empty")

    family_strata: dict[str, str] = {}
    family_repetitions: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        family = row["family"]
        if family in family_strata and family_strata[family] != row["stratum"]:
            fail(f"baseline family {family!r} occurs in more than one stratum")
        family_strata[family] = row["stratum"]
        family_repetitions[family].add(row["repetition"])
    expected_repetitions = set(range(repetitions))
    for family, observed in sorted(family_repetitions.items()):
        if observed != expected_repetitions:
            fail(f"baseline family {family!r} lacks the fixed repetition sequence 0..{repetitions - 1}")
    rows.sort(key=lambda row: (row["family"], row["repetition"]))
    return {
        "schema": BASELINES_SCHEMA,
        "measurement_role": "baseline_only",
        "repetitions_per_family": repetitions,
        "metrics": metrics,
        "rows": rows,
    }


def validate_metric(value: object, label: str) -> dict[str, Any]:
    metric = obj(value, label)
    exact_keys(metric, {"name", "domain", "direction", "test", "null_margin", "assumed_effect", "effect_source"}, label)
    test = enum(metric["test"], TESTS, f"{label}.test")
    null_margin = number(metric["null_margin"], f"{label}.null_margin")
    if test == "superiority" and null_margin != 0:
        fail(f"{label}.null_margin must be zero for superiority")
    effect = number(metric["assumed_effect"], f"{label}.assumed_effect", positive=True)
    source = obj(metric["effect_source"], f"{label}.effect_source")
    exact_keys(source, {"kind", "sha256"}, f"{label}.effect_source")
    return {
        "name": text(metric["name"], f"{label}.name"),
        "domain": enum(metric["domain"], DOMAINS, f"{label}.domain"),
        "direction": enum(metric["direction"], DIRECTIONS, f"{label}.direction"),
        "test": test,
        "null_margin": null_margin,
        "assumed_effect": effect,
        "effect_source": {
            "kind": enum(source["kind"], EFFECT_SOURCES, f"{label}.effect_source.kind"),
            "sha256": sha256(source["sha256"], f"{label}.effect_source.sha256"),
        },
    }


def validate_margin_spec(value: object) -> dict[str, Any]:
    document = obj(value, "margin_spec")
    exact_keys(
        document,
        {
            "schema",
            "candidate_blind",
            "seed",
            "simulation_repetitions",
            "bootstrap_repetitions",
            "sample_sizes",
            "target_power",
            "multiplicity",
            "families",
            "strata",
            "metrics",
            "futility_rules",
        },
        "margin_spec",
    )
    if document["schema"] != MARGIN_SCHEMA:
        fail(f"margin_spec.schema must be {MARGIN_SCHEMA!r}")
    blind = obj(document["candidate_blind"], "margin_spec.candidate_blind")
    exact_keys(blind, {"inputs", "locked_before_candidate_access", "candidate_results_accessed", "declaration_sha256"}, "margin_spec.candidate_blind")
    if blind["inputs"] != "baseline_only" or boolean(blind["locked_before_candidate_access"], "margin_spec.candidate_blind.locked_before_candidate_access") is not True:
        fail("margin spec must be locked from baseline-only inputs before candidate access")
    if boolean(blind["candidate_results_accessed"], "margin_spec.candidate_blind.candidate_results_accessed") is not False:
        fail("result-derived power inputs are forbidden")
    candidate_blind = {
        "inputs": "baseline_only",
        "locked_before_candidate_access": True,
        "candidate_results_accessed": False,
        "declaration_sha256": sha256(blind["declaration_sha256"], "margin_spec.candidate_blind.declaration_sha256"),
    }
    seed = integer(document["seed"], "margin_spec.seed", positive=True)
    simulation_repetitions = integer(document["simulation_repetitions"], "margin_spec.simulation_repetitions", positive=True)
    bootstrap_repetitions = integer(document["bootstrap_repetitions"], "margin_spec.bootstrap_repetitions", positive=True)
    if simulation_repetitions < 100 or bootstrap_repetitions < 100:
        fail("simulation_repetitions and bootstrap_repetitions must each be at least 100")
    sample_sizes = [integer(raw, f"margin_spec.sample_sizes[{index}]", positive=True) for index, raw in enumerate(array(document["sample_sizes"], "margin_spec.sample_sizes"))]
    if not sample_sizes or sample_sizes != sorted(set(sample_sizes)):
        fail("margin_spec.sample_sizes must be a non-empty strictly increasing list")
    target_power = number(document["target_power"], "margin_spec.target_power", positive=True)
    if target_power >= 1:
        fail("margin_spec.target_power must be below one")

    multiplicity_raw = obj(document["multiplicity"], "margin_spec.multiplicity")
    exact_keys(multiplicity_raw, {"method", "familywise_alpha", "tests"}, "margin_spec.multiplicity")
    familywise_alpha = number(multiplicity_raw["familywise_alpha"], "margin_spec.multiplicity.familywise_alpha", positive=True)
    tests = integer(multiplicity_raw["tests"], "margin_spec.multiplicity.tests", positive=True)
    if familywise_alpha >= 1:
        fail("margin_spec.multiplicity.familywise_alpha must be below one")
    multiplicity = {
        "method": enum(multiplicity_raw["method"], MULTIPLICITY, "margin_spec.multiplicity.method"),
        "familywise_alpha": familywise_alpha,
        "tests": tests,
    }

    strata = [text(raw, f"margin_spec.strata[{index}]") for index, raw in enumerate(array(document["strata"], "margin_spec.strata"))]
    if not strata or strata != sorted(set(strata)):
        fail("margin_spec.strata must be a non-empty sorted list without duplicates")
    families: list[dict[str, str]] = []
    seen_families: set[str] = set()
    for index, raw in enumerate(array(document["families"], "margin_spec.families")):
        label = f"margin_spec.families[{index}]"
        family = obj(raw, label)
        exact_keys(family, {"name", "stratum"}, label)
        name = text(family["name"], f"{label}.name")
        stratum = text(family["stratum"], f"{label}.stratum")
        if name in seen_families or stratum not in strata:
            fail(f"{label} duplicates a family or names an undeclared stratum")
        seen_families.add(name)
        families.append({"name": name, "stratum": stratum})
    if not families or families != sorted(families, key=lambda entry: entry["name"]):
        fail("margin_spec.families must be a non-empty list sorted by name")
    if {family["stratum"] for family in families} != set(strata):
        fail("every declared stratum must contain at least one fixed family")

    metrics = [validate_metric(raw, f"margin_spec.metrics[{index}]") for index, raw in enumerate(array(document["metrics"], "margin_spec.metrics"))]
    metric_names = [metric["name"] for metric in metrics]
    if not metrics or metric_names != sorted(set(metric_names)):
        fail("margin_spec.metrics must be a non-empty list sorted by unique name")
    if tests != len(metrics):
        fail("margin_spec.multiplicity.tests must equal the fixed number of metrics")
    if {metric["domain"] for metric in metrics} != DOMAINS:
        fail("margin_spec.metrics must include fidelity, performance, and capability domains")

    futility = obj(document["futility_rules"], "margin_spec.futility_rules")
    exact_keys(futility, {"interim_information_fraction", "conditional_power_below", "action", "nonbinding"}, "margin_spec.futility_rules")
    interim = number(futility["interim_information_fraction"], "margin_spec.futility_rules.interim_information_fraction", positive=True)
    conditional = number(futility["conditional_power_below"], "margin_spec.futility_rules.conditional_power_below", positive=True)
    if interim >= 1 or conditional >= 1:
        fail("futility fractions must be below one")
    if futility["action"] != "stop_for_futility" or boolean(futility["nonbinding"], "margin_spec.futility_rules.nonbinding") is not True:
        fail("futility rule must be a fixed nonbinding stop_for_futility rule")
    return {
        "schema": MARGIN_SCHEMA,
        "candidate_blind": candidate_blind,
        "seed": seed,
        "simulation_repetitions": simulation_repetitions,
        "bootstrap_repetitions": bootstrap_repetitions,
        "sample_sizes": sample_sizes,
        "target_power": target_power,
        "multiplicity": multiplicity,
        "families": families,
        "strata": strata,
        "metrics": metrics,
        "futility_rules": {
            "interim_information_fraction": interim,
            "conditional_power_below": conditional,
            "action": "stop_for_futility",
            "nonbinding": True,
        },
    }


def effective_alpha(multiplicity: dict[str, Any]) -> float:
    alpha = multiplicity["familywise_alpha"]
    tests = multiplicity["tests"]
    if multiplicity["method"] == "bonferroni":
        return alpha / tests
    return 1.0 - (1.0 - alpha) ** (1.0 / tests)


def derived_rng(seed: int, *labels: object) -> random.Random:
    material = canonical_sha256([seed, *labels])
    return random.Random(int(material, 16))


def quantile(values: list[float], probability: float) -> float:
    if not values:
        fail("cannot compute an empirical quantile from no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def hierarchical_statistic(
    rng: random.Random,
    residuals: dict[str, list[float]],
    families_by_stratum: dict[str, list[str]],
    n_per_family: int,
) -> float:
    stratum_means: list[float] = []
    for stratum in sorted(families_by_stratum):
        families = families_by_stratum[stratum]
        sampled_family_means: list[float] = []
        for _ in range(len(families)):
            family = families[rng.randrange(len(families))]
            observations = residuals[family]
            total = 0.0
            for _ in range(n_per_family):
                total += observations[rng.randrange(len(observations))]
            sampled_family_means.append(total / n_per_family)
        stratum_means.append(sum(sampled_family_means) / len(sampled_family_means))
    return sum(stratum_means) / len(stratum_means)


def construct(baselines_value: object, margin_value: object) -> dict[str, Any]:
    baselines = validate_baselines(baselines_value)
    margin = validate_margin_spec(margin_value)
    baseline_metrics = baselines["metrics"]
    metric_names = [metric["name"] for metric in margin["metrics"]]
    if baseline_metrics != metric_names:
        fail("baseline and margin metric identities differ")

    declared_families = {entry["name"]: entry["stratum"] for entry in margin["families"]}
    observed_families: dict[str, str] = {}
    for row in baselines["rows"]:
        if row["family"] in observed_families and observed_families[row["family"]] != row["stratum"]:
            fail(f"baseline family {row['family']!r} has inconsistent strata")
        observed_families[row["family"]] = row["stratum"]
    if observed_families != declared_families:
        fail("margin spec fixed families/strata must exactly equal baseline families/strata")

    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baselines["rows"]:
        rows_by_family[row["family"]].append(row)
    families_by_stratum: dict[str, list[str]] = defaultdict(list)
    for family, stratum in declared_families.items():
        families_by_stratum[stratum].append(family)
    for stratum in families_by_stratum:
        families_by_stratum[stratum].sort()

    residuals_by_metric: dict[str, dict[str, list[float]]] = {}
    baseline_centers: dict[str, dict[str, float]] = {}
    for metric in margin["metrics"]:
        name = metric["name"]
        residuals: dict[str, list[float]] = {}
        centers: dict[str, float] = {}
        for family in sorted(rows_by_family):
            raw_values = [row["values"][name] for row in rows_by_family[family]]
            center = sum(raw_values) / len(raw_values)
            centers[family] = center
            orientation = 1.0 if metric["direction"] == "higher_is_better" else -1.0
            residuals[family] = [orientation * (value - center) for value in raw_values]
        residuals_by_metric[name] = residuals
        baseline_centers[name] = centers

    alpha = effective_alpha(margin["multiplicity"])
    power_table: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for n_per_family in margin["sample_sizes"]:
        metric_rows: list[dict[str, Any]] = []
        for metric in margin["metrics"]:
            name = metric["name"]
            null_rng = derived_rng(margin["seed"], "null_bootstrap", name, n_per_family)
            null_statistics = [
                hierarchical_statistic(null_rng, residuals_by_metric[name], families_by_stratum, n_per_family)
                for _ in range(margin["bootstrap_repetitions"])
            ]
            empirical_tail = quantile(null_statistics, 1.0 - alpha)
            null_boundary = 0.0 if metric["test"] == "superiority" else -metric["null_margin"]
            decision_threshold = null_boundary + empirical_tail
            simulation_rng = derived_rng(margin["seed"], "alternative_simulation", name, n_per_family)
            passes = 0
            for _ in range(margin["simulation_repetitions"]):
                simulated = hierarchical_statistic(simulation_rng, residuals_by_metric[name], families_by_stratum, n_per_family)
                simulated += metric["assumed_effect"]
                if simulated > decision_threshold:
                    passes += 1
            power = passes / margin["simulation_repetitions"]
            metric_rows.append({
                "name": name,
                "domain": metric["domain"],
                "n_per_family": n_per_family,
                "one_sided_alpha_after_multiplicity": alpha,
                "null_boundary": null_boundary,
                "empirical_tail_offset": empirical_tail,
                "decision_threshold_oriented_effect": decision_threshold,
                "simulated_power": power,
                "passes_target": power >= margin["target_power"],
            })
        row = {
            "n_per_family": n_per_family,
            "metrics": metric_rows,
            "all_metrics_pass_target": all(entry["passes_target"] for entry in metric_rows),
        }
        power_table.append(row)
        if selected is None and row["all_metrics_pass_target"]:
            selected = row
    if selected is not None:
        selected_n = selected["n_per_family"]
        selected_metrics = selected["metrics"]
    else:
        fail("no fixed sample size reaches target power for every multiplicity-controlled metric")

    tail_thresholds = {
        entry["name"]: {
            "one_sided_alpha_after_multiplicity": entry["one_sided_alpha_after_multiplicity"],
            "null_boundary": entry["null_boundary"],
            "empirical_tail_offset": entry["empirical_tail_offset"],
            "decision_threshold_oriented_effect": entry["decision_threshold_oriented_effect"],
        }
        for entry in selected_metrics
    }
    source_rows = [
        {
            "family": row["family"],
            "stratum": row["stratum"],
            "repetition": row["repetition"],
            "artifact_sha256": row["source"]["artifact_sha256"],
            "row_sha256": row["source"]["row_sha256"],
        }
        for row in baselines["rows"]
    ]
    return {
        "schema": PLAN_SCHEMA,
        "source": {
            "baselines_schema": BASELINES_SCHEMA,
            "baselines_sha256": canonical_sha256(baselines),
            "margin_spec_schema": MARGIN_SCHEMA,
            "margin_spec_sha256": canonical_sha256(margin),
            "candidate_blind": True,
            "measurement_role": "baseline_only",
            "baseline_rows": source_rows,
        },
        "method": {
            "simulation": "deterministic_stratified_hierarchical_family_bootstrap",
            "family_sampling": "with_replacement_within_fixed_stratum",
            "row_sampling": "with_replacement_within_sampled_family",
            "stratum_aggregation": "equal_weight",
            "family_aggregation": "equal_weight_within_stratum",
            "tail": "one_sided_conservative_empirical_order_statistic",
            "effect_orientation": "positive_is_better",
        },
        "fixed_design": {
            "n_per_family": selected_n,
            "families": margin["families"],
            "strata": margin["strata"],
            "baseline_repetitions_per_family": baselines["repetitions_per_family"],
            "simulation_repetitions": margin["simulation_repetitions"],
            "bootstrap_repetitions": margin["bootstrap_repetitions"],
            "seed": margin["seed"],
            "target_power": margin["target_power"],
            "multiplicity": {**margin["multiplicity"], "one_sided_alpha_per_test": alpha},
            "tail_thresholds": tail_thresholds,
            "futility_rules": margin["futility_rules"],
        },
        "metrics": margin["metrics"],
        "baseline_family_centers": baseline_centers,
        "power_simulations": power_table,
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
    result = argparse.ArgumentParser(description="Build or verify a candidate-blind Final Frontier family-bootstrap power plan.")
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="simulate and write a new fixed power plan")
    build.add_argument("--baselines", required=True, type=Path)
    build.add_argument("--margin-spec", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="reconstruct and verify an existing fixed power plan")
    verify.add_argument("--baselines", required=True, type=Path)
    verify.add_argument("--margin-spec", required=True, type=Path)
    verify.add_argument("--plan", required=True, type=Path)
    return result


def run(args: argparse.Namespace) -> None:
    expected = construct(load_nonempty(args.baselines, "baselines"), load_nonempty(args.margin_spec, "margin spec"))
    if args.command == "build":
        require_empty_output(args.out)
        atomic_write_json(args.out, expected)
        return
    actual = load_nonempty(args.plan, "power plan")
    if not isinstance(actual, dict) or actual.get("schema") != PLAN_SCHEMA:
        fail(f"power plan.schema must be {PLAN_SCHEMA!r}")
    if actual != expected:
        fail("power plan does not exactly reproduce from baseline-only inputs and locked margin spec")
    if canonical_sha256(actual) != canonical_sha256(expected):
        fail("power plan canonical SHA256 verification failed")


def main(argv: list[str] | None = None) -> int:
    try:
        run(parser().parse_args(argv))
    except (PowerError, OSError, ValueError) as exc:
        print(f"frontier_power: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
