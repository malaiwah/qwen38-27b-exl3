#!/usr/bin/env python3
"""Deterministic, machine-checkable downstream retention smoke suite.

This is not a public leaderboard. It asks whether a quantized candidate preserves
BF16 behavior on identical generated tasks. Reports carry exact model/runtime
identities, the generated task-set digest, absolute pass rates, BF16-pass retention,
and exact-output agreement. Model-generated Python is statically restricted and
executed in a resource-limited subprocess.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def model_identity(path: str, revision: str | None, evidence: str | None) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise SystemExit("--model-path must be the local directory served by vLLM")
    index = root / "model.safetensors.index.json"
    config = root / "config.json"
    evidence_path = Path(evidence) if evidence else None
    shard_paths = sorted(root.glob("*.safetensors"))
    if not shard_paths:
        raise SystemExit(f"no safetensors shards found under {root}")
    shard_sha256 = {shard.name: sha256_file(shard) for shard in shard_paths}
    identity = {
        "path": str(root.resolve()),
        "revision": revision,
        "index_sha256": sha256_file(index) if index.is_file() else None,
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "shard_sha256": shard_sha256,
        "release_evidence": str(evidence_path) if evidence_path else None,
        "release_evidence_sha256": (
            sha256_file(evidence_path) if evidence_path and evidence_path.is_file() else None
        ),
    }
    if evidence and not identity["release_evidence_sha256"]:
        raise SystemExit(f"release evidence does not exist: {evidence}")
    if evidence_path:
        artifact = (json.loads(evidence_path.read_text()).get("artifact") or {})
        expected_index = artifact.get("index_sha256")
        expected_shards = artifact.get("shard_sha256")
        if not isinstance(expected_index, str) or not expected_index:
            raise SystemExit("release evidence does not carry artifact.index_sha256")
        if identity["index_sha256"] != expected_index:
            raise SystemExit("served index digest does not match release evidence")
        if not isinstance(expected_shards, dict) or not expected_shards:
            raise SystemExit("release evidence does not carry artifact.shard_sha256")
        if shard_sha256 != expected_shards:
            raise SystemExit("served shard digests do not match release evidence")
    return identity


def chat(url: str, prompt: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"malformed chat response: {str(body)[:300]}") from exc
    if not isinstance(text, str):
        raise RuntimeError(f"chat content is not text: {text!r}")
    return text


def final_answer(text: str) -> str:
    """Discard an explicitly delimited reasoning block, never guessed prose."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


def gen_arithmetic(rng: random.Random, n: int) -> list[dict]:
    tasks = []
    seen = set()
    while len(tasks) < n:
        crates = rng.randrange(7, 40)
        per = rng.randrange(6, 25)
        broken = rng.randrange(3, max(4, min(crates * per // 4, 60)))
        total = crates * per - broken
        sold = rng.randrange(10, max(11, min(total, 90)))
        key = (crates, per, broken, sold)
        if key in seen:
            continue
        seen.add(key)
        tasks.append({
            "kind": "arithmetic",
            "prompt": (
                f"A warehouse receives {crates} crates with {per} jars each. "
                f"{broken} jars arrive broken and are discarded, then {sold} jars are sold. "
                "How many jars remain? Work it out carefully, then output the number only."
            ),
            "answer": str(total - sold),
            "check": "number",
        })
    return tasks


def gen_code(_rng: random.Random, n: int) -> list[dict]:
    specs = [
        ("count_vowels", "return the number of vowels (aeiou, case-insensitive) in a string",
         [(('Hello World',), 3), (('xyz',), 0), (('AEIOU',), 5)]),
        ("second_largest", "return the second largest distinct integer in a list, or None if absent",
         [(([3, 1, 4, 4, 2],), 3), (([5],), None), (([2, 2],), None)]),
        ("run_length", "compress a string into run-length pairs, for example aaab becomes a3b1",
         [(('aaab',), 'a3b1'), (('',), ''), (('xy',), 'x1y1')]),
        ("balanced", "return True when the brackets ()[]{} in a string are balanced",
         [(('([]{})',), True), (('(]',), False), (('',), True)]),
        ("digit_sum_until", "sum the digits of a non-negative integer repeatedly until one digit remains",
         [((0,), 0), ((38,), 2), ((12345,), 6)]),
        ("dedupe_preserve", "remove duplicate list items while preserving first-occurrence order",
         [(([3, 1, 3, 2],), [3, 1, 2]), (([],), []), ((['a', 'a'],), ['a'])]),
        ("rotate_right", "rotate a list right by k positions without modifying the input",
         [(([1, 2, 3, 4], 1), [4, 1, 2, 3]), (([1, 2], 4), [1, 2]), (([], 3), [])]),
        ("flatten_once", "flatten a list of lists by exactly one level",
         [(([[1, 2], [3]],), [1, 2, 3]), (([[], [1]],), [1]), (([],), [])]),
        ("clamp_values", "return a new list with every number clamped inclusively between lo and hi",
         [(([-2, 3, 9], 0, 5), [0, 3, 5]), (([], 1, 2), []), (([4], 4, 4), [4])]),
        ("longest_word", "return the first longest string in a list, or None for an empty list",
         [((['a', 'bbb', 'cc'],), 'bbb'), ((['aa', 'bb'],), 'aa'), (([],), None)]),
    ]
    if n > len(specs):
        raise SystemExit(f"--per-kind may not exceed {len(specs)} for unique code tasks")
    tasks = []
    for name, description, cases in specs[:n]:
        tasks.append({
            "kind": "code",
            "name": name,
            "prompt": (
                f"Write a Python function `{name}` that will {description}. Use Python builtins only; "
                "do not import anything. Output only the function definition in one ```python code "
                "block, with no explanation or tests."
            ),
            "cases": cases,
            "check": "exec",
        })
    return tasks


def gen_format(rng: random.Random, n: int) -> list[dict]:
    topics = ["rivers", "metals", "planets", "instruments", "spices"]
    combinations = [(topic, count) for topic in topics for count in range(3, 7)]
    rng.shuffle(combinations)
    if n > len(combinations):
        raise SystemExit(f"--per-kind may not exceed {len(combinations)} for format tasks")
    return [{
        "kind": "format",
        "prompt": (
            f"List exactly {count} {topic}. Output a JSON array of {count} lowercase strings "
            "and nothing else."
        ),
        "count": count,
        "check": "json_array",
    } for topic, count in combinations[:n]]


def gen_tool(rng: random.Random, n: int) -> list[dict]:
    combinations = [(city, days) for city in ["Oslo", "Lima", "Osaka", "Cairo", "Perth"]
                    for days in range(2, 9)]
    rng.shuffle(combinations)
    if n > len(combinations):
        raise SystemExit(f"--per-kind may not exceed {len(combinations)} for tool tasks")
    tasks = []
    for city, days in combinations[:n]:
        tasks.append({
            "kind": "tool",
            "prompt": (
                "You have one tool:\n"
                '{"name":"get_forecast","parameters":{"city":"string","days":"integer"}}\n'
                f"The user asks for the weather in {city} for the next {days} days. "
                "Output one JSON object with name get_forecast and arguments city and days, "
                "and nothing else."
            ),
            "city": city,
            "days": days,
            "check": "tool_call",
        })
    return tasks


SAFE_CALLS = {
    "all", "any", "bool", "enumerate", "int", "len", "list", "max", "min",
    "range", "reversed", "set", "sorted", "str", "sum", "tuple", "zip",
}
SAFE_METHODS = {
    "add", "append", "clear", "copy", "count", "difference", "discard",
    "extend", "get", "index", "insert", "intersection", "isdigit", "issubset",
    "issuperset", "items", "join", "keys", "lower", "pop", "remove", "replace",
    "reverse", "sort", "split", "strip", "symmetric_difference", "union",
    "update", "upper", "values",
}


def validate_generated_code(body: str, function_name: str) -> None:
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc.msg}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("response must contain exactly one function definition")
    if tree.body[0].name != function_name or isinstance(tree.body[0], ast.AsyncFunctionDef):
        raise ValueError(f"expected synchronous function {function_name}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda,
                             ast.ClassDef, ast.With, ast.AsyncWith, ast.Try)):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name is disallowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in SAFE_METHODS:
                raise ValueError(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_CALLS and node.func.id != function_name:
                    raise ValueError(f"disallowed call: {node.func.id}")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("indirect calls are disallowed")


SANDBOX_WRAPPER = """\
import os
import resource
import sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
os.execv(sys.executable, [sys.executable, "-I", sys.argv[1]])
"""


def score_code(task: dict, text: str) -> tuple[bool, str]:
    match = re.fullmatch(r"\s*```(?:python)?\s*(.*?)```\s*", text, re.S)
    body = match.group(1) if match else text
    try:
        validate_generated_code(body, task["name"])
    except ValueError as exc:
        return False, str(exc)[:100]
    harness = body + "\n\nimport json\n_results=[]\n"
    for case_index, (arguments, expected) in enumerate(task["cases"]):
        harness += f"_args_{case_index} = {arguments!r}\n"
        harness += f"_before_{case_index} = repr(_args_{case_index})\n"
        harness += (
            f"_got_{case_index} = {task['name']}(*_args_{case_index})\n"
            f"_results.append(_got_{case_index} == {expected!r} and "
            f"repr(_args_{case_index}) == _before_{case_index})\n"
        )
    harness += "print(json.dumps(_results))\n"
    with tempfile.TemporaryDirectory(prefix="qwen38-retention-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        script = root / "candidate.py"
        script.write_text(harness)
        script.chmod(0o444)
        wrapper = root / "sandbox.py"
        wrapper.write_text(SANDBOX_WRAPPER)
        wrapper.chmod(0o444)
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(wrapper), str(script)],
                cwd=root,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if process.returncode != 0:
                return False, f"exit {process.returncode}: {process.stderr[-80:].strip()}"
            result = json.loads(process.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(result) and all(result), f"{sum(bool(x) for x in result)}/{len(result)} cases"


def score(task: dict, response: str) -> tuple[bool, str, str]:
    answer = final_answer(response)
    if task["check"] == "number":
        normalized = answer.replace(",", "").strip()
        ok = re.fullmatch(r"-?\d+", normalized) is not None and normalized == task["answer"]
        return ok, normalized[:80], answer
    if task["check"] == "json_array":
        try:
            value = json.loads(answer)
        except Exception:
            return False, answer[:80], answer
        ok = (isinstance(value, list) and len(value) == task["count"]
              and len(set(value)) == len(value)
              and all(isinstance(item, str) and item and item == item.lower()
                      for item in value))
        return ok, json.dumps(value, separators=(",", ":"))[:200], answer
    if task["check"] == "tool_call":
        try:
            value = json.loads(answer)
        except Exception:
            return False, answer[:80], answer
        arguments = value.get("arguments") if isinstance(value, dict) else None
        ok = (
            isinstance(value, dict)
            and set(value) == {"name", "arguments"}
            and value.get("name") == "get_forecast"
            and isinstance(arguments, dict)
            and set(arguments) == {"city", "days"}
            and arguments.get("city") == task["city"]
            and isinstance(arguments.get("days"), int)
            and not isinstance(arguments.get("days"), bool)
            and arguments.get("days") == task["days"]
        )
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, dict) else ""
        return ok, canonical[:200], answer
    if task["check"] == "exec":
        ok, extracted = score_code(task, answer)
        return ok, extracted, answer
    raise ValueError(f"unknown checker: {task['check']}")


def wilson(successes: int, total: int) -> dict:
    if total == 0:
        return {"successes": 0, "n": 0, "rate": None, "ci95_low": None, "ci95_high": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "n": total, "rate": rate,
            "ci95_low": center - half, "ci95_high": center + half}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--release-evidence")
    parser.add_argument("--runtime-json", default="{}")
    parser.add_argument("--per-kind", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--require-all-pass", action="store_true")
    parser.add_argument("--require-no-regressions", action="store_true")
    args = parser.parse_args()
    if args.per_kind <= 0:
        raise SystemExit("--per-kind must be positive")
    if args.concurrency <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--concurrency, --max-tokens and --timeout must be positive")
    if args.require_no_regressions and not args.reference:
        raise SystemExit("--require-no-regressions requires --reference")
    try:
        runtime = json.loads(args.runtime_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--runtime-json is invalid: {exc}") from exc
    if not isinstance(runtime, dict):
        raise SystemExit("--runtime-json must decode to an object")
    if runtime.get("identity_verified") is not True:
        raise SystemExit("--runtime-json must carry an observed identity_verified:true runtime")

    rng = random.Random(args.seed)
    tasks = (gen_arithmetic(rng, args.per_kind) + gen_code(rng, args.per_kind)
             + gen_format(rng, args.per_kind) + gen_tool(rng, args.per_kind))
    for index, task in enumerate(tasks):
        task["index"] = index
    suite_sha256 = sha256_bytes(canonical_bytes(tasks))

    reference = None
    reference_sha256 = None
    ref_by_index: dict[int, dict] = {}
    if args.reference:
        reference_path = Path(args.reference)
        reference = json.loads(reference_path.read_text())
        reference_sha256 = sha256_file(reference_path)
        if reference.get("suite_sha256") != suite_sha256:
            raise SystemExit("reference task suite does not match this generated suite")
        reference_rows = reference.get("results", [])
        if len(reference_rows) != len(tasks):
            raise SystemExit("reference report is incomplete")
        ref_by_index = {row.get("index"): row for row in reference_rows}
        if set(ref_by_index) != set(range(len(tasks))):
            raise SystemExit("reference report has duplicate or missing task indices")
        for task in tasks:
            expected = sha256_bytes(canonical_bytes(task))
            if ref_by_index[task["index"]].get("task_sha256") != expected:
                raise SystemExit(
                    f"reference task {task['index']} is not bound to this generated task"
                )

    def run_one(task: dict) -> dict:
        response = chat(args.url, task["prompt"], args.max_tokens, args.timeout)
        passed, extracted, answer = score(task, response)
        return {
            "index": task["index"],
            "kind": task["kind"],
            "task_sha256": sha256_bytes(canonical_bytes(task)),
            "passed": passed,
            "extracted": extracted,
            "response_sha256": sha256_bytes(response.encode()),
            "response": response,
            "final_answer": answer,
        }

    results_by_index = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        try:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                results_by_index[row["index"]] = row
                print(f"{row['kind']:11s} {row['index']:3d} "
                      f"{'PASS' if row['passed'] else 'fail'} {row['extracted'][:60]}", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    results = [results_by_index[index] for index in range(len(tasks))]
    if reference is not None:

        for row in results:
            ref = ref_by_index[row["index"]]
            row["reference_passed"] = bool(ref["passed"])
            row["pass_outcome_agreement"] = row["passed"] == bool(ref["passed"])
            row["exact_output_agreement"] = row["final_answer"] == ref["final_answer"]

    per_kind = {}
    for kind in ("arithmetic", "code", "format", "tool"):
        group = [row for row in results if row["kind"] == kind]
        block = {"pass": wilson(sum(row["passed"] for row in group), len(group))}
        if reference is not None:
            ref_pass = [row for row in group if row["reference_passed"]]
            block.update({
                "bf16_pass_retention": wilson(sum(row["passed"] for row in ref_pass), len(ref_pass)),
                "pass_outcome_agreement": wilson(sum(row["pass_outcome_agreement"] for row in group), len(group)),
                "exact_output_agreement": wilson(sum(row["exact_output_agreement"] for row in group), len(group)),
            })
        per_kind[kind] = block

    overall = {"pass": wilson(sum(row["passed"] for row in results), len(results))}
    paired = None
    if reference is not None:
        reference_passes = [row for row in results if row["reference_passed"]]
        regressions = sum(row["reference_passed"] and not row["passed"] for row in results)
        improvements = sum(not row["reference_passed"] and row["passed"] for row in results)
        paired = {
            "bf16_pass_retention": wilson(sum(row["passed"] for row in reference_passes), len(reference_passes)),
            "pass_outcome_agreement": wilson(sum(row["pass_outcome_agreement"] for row in results), len(results)),
            "exact_output_agreement": wilson(sum(row["exact_output_agreement"] for row in results), len(results)),
            "regressions": regressions,
            "improvements": improvements,
        }

    report = {
        "schema": "qwen38-task-retention/2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "seed": args.seed,
        "per_kind_requested": args.per_kind,
        "tasks": len(tasks),
        "suite_sha256": suite_sha256,
        "candidate_identity": model_identity(args.model_path, args.model_revision, args.release_evidence),
        "runtime": runtime,
        "reference": ({"path": args.reference, "sha256": reference_sha256,
                       "candidate_identity": reference.get("candidate_identity")}
                      if reference is not None else None),
        "per_kind": per_kind,
        "overall": overall,
        "paired": paired,
        "results": results,
    }
    atomic_write_json(Path(args.out), report)
    print(json.dumps({"overall": overall, "paired": paired}), flush=True)
    if args.require_all_pass and overall["pass"]["successes"] != len(results):
        return 1
    if args.require_no_regressions and paired and paired["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
