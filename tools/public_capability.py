#!/usr/bin/env python3
"""Build and run a deterministic, public MMLU-Pro capability suite."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# task_retention.py is beside this script, both when invoked directly and in the repo.
try:
    from task_retention import model_identity
except ModuleNotFoundError:  # pragma: no cover - supports importlib-based callers
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_retention import model_identity

SUITE_SCHEMA = "qwen38-public-capability-suite/1"
PLAN_SCHEMA = "qwen38-public-capability-plan/1"
REPORT_SCHEMA = "qwen38-public-capability/1"
DATASET = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
TEST_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
VALIDATION_SHA256 = "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078"
SELECTION_NAMESPACE = "qwen38-public-capability-v1"
CHOICES = "ABCDEFGHIJ"
CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
SOURCE = {
    "dataset": DATASET,
    "revision": DATASET_REVISION,
    "test": {
        "filename": "test-00000-of-00001.parquet",
        "sha256": TEST_SHA256,
    },
    "validation": {
        "filename": "validation-00000-of-00001.parquet",
        "sha256": VALIDATION_SHA256,
    },
}
ANSWER_RE = re.compile(r"(?:The answer is\s+|Answer:\s*)\(?([A-J])\)?(?![A-Za-z])")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one complete JSON object using a same-directory atomic rename."""
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def strict_json_text(text: str, description: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{description} is not valid strict JSON: {exc}") from exc


def load_json(path: Path, description: str) -> Any:
    try:
        return strict_json_text(path.read_text(encoding="utf-8"), description)
    except OSError as exc:
        raise ValueError(f"cannot read {description} {path}: {exc}") from exc


def require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA256 digest")
    return value


def without_key(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def self_digest(value: dict[str, Any], key: str) -> str:
    return sha256_bytes(canonical_bytes(without_key(value, key)))


def json_value(value: Any, description: str) -> Any:
    """Normalize Arrow values while rejecting identities JSON cannot preserve."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item, description) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{description} contains a non-string object key")
        return {key: json_value(item, description) for key, item in value.items()}
    raise ValueError(f"{description} contains unsupported value {type(value).__name__}")


def read_parquet(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValueError("build requires pyarrow") from exc
    try:
        rows = parquet.read_table(path).to_pylist()
    except Exception as exc:
        raise ValueError(f"cannot read {description} parquet: {exc}") from exc
    normalized = json_value(rows, description)
    if not isinstance(normalized, list) or not all(isinstance(row, dict) for row in normalized):
        raise ValueError(f"{description} parquet did not decode to rows")
    return normalized


def checked_row(row: dict[str, Any], split: str, row_index: int, need_cot: bool) -> dict[str, Any]:
    required = {"question_id", "question", "options", "answer", "answer_index", "category"}
    if need_cot:
        required.add("cot_content")
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{split} row {row_index} is missing fields: {', '.join(missing)}")
    question_id = row["question_id"]
    if isinstance(question_id, bool) or not isinstance(question_id, (str, int)):
        raise ValueError(f"{split} row {row_index} has invalid question_id")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{split} row {row_index} has invalid question")
    if not isinstance(row["category"], str) or row["category"] not in CATEGORIES:
        raise ValueError(f"{split} row {row_index} has unknown category {row['category']!r}")
    options = row["options"]
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError(f"{split} row {row_index} has invalid options")
    official_options = [option for option in options if option != "N/A"]
    if not 2 <= len(official_options) <= len(CHOICES):
        raise ValueError(f"{split} row {row_index} has an invalid number of options")
    answer_index = row["answer_index"]
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ValueError(f"{split} row {row_index} has invalid answer_index")
    if not 0 <= answer_index < len(official_options):
        raise ValueError(f"{split} row {row_index} answer_index is out of range")
    answer = row["answer"]
    if not isinstance(answer, str) or answer != CHOICES[answer_index]:
        raise ValueError(f"{split} row {row_index} answer and answer_index disagree")
    if need_cot and not isinstance(row["cot_content"], str):
        raise ValueError(f"{split} row {row_index} has invalid cot_content")
    return {
        "question_id": question_id,
        "question": row["question"],
        "options": official_options,
        "answer": answer,
        "answer_index": answer_index,
        "category": row["category"],
        "cot_content": row.get("cot_content"),
        "src": row.get("src"),
        "source_row_sha256": sha256_bytes(canonical_bytes(row)),
    }


def format_example(question: str, options: list[str], cot_content: str = "") -> str:
    """Byte-for-byte prompt formatter from the pinned MMLU-Pro evaluator."""
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    for index, option in enumerate(options):
        example += f"{CHOICES[index]}. {option}\n"
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def category_instruction(category: str) -> str:
    return (
        "The following are multiple choice questions (with answers) about {}. Think step by"
        " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    ).format(category)


def category_prefix(category: str, examples: list[dict[str, Any]]) -> str:
    prompt = category_instruction(category)
    for example in examples:
        prompt += format_example(example["question"], example["options"], example["cot_content"])
    return prompt


def source_identity(row: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    return {
        "split": split,
        "row_index": row_index,
        "question_id": row["question_id"],
        "src": row["src"],
        "source_row_sha256": row["source_row_sha256"],
    }


def build_suite(test_path: Path, validation_path: Path, per_category: int) -> dict[str, Any]:
    if per_category <= 0:
        raise ValueError("--per-category must be positive")
    for path, expected, description in (
        (test_path, TEST_SHA256, "test"),
        (validation_path, VALIDATION_SHA256, "validation"),
    ):
        if not path.is_file():
            raise ValueError(f"{description} parquet does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{description} parquet SHA256 mismatch: expected {expected}, got {actual}")

    test_raw = read_parquet(test_path, "test")
    validation_raw = read_parquet(validation_path, "validation")
    test_rows = [checked_row(row, "test", index, False) for index, row in enumerate(test_raw)]
    validation_rows = [
        checked_row(row, "validation", index, True) for index, row in enumerate(validation_raw)
    ]
    test_ids = [str(row["question_id"]) for row in test_rows]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("test parquet has duplicate question_id values")

    validation_by_category: dict[str, list[tuple[int, dict[str, Any]]]] = {
        category: [] for category in CATEGORIES
    }
    for row_index, row in enumerate(validation_rows):
        validation_by_category[row["category"]].append((row_index, row))
    for category in CATEGORIES:
        if len(validation_by_category[category]) != 5:
            raise ValueError(
                f"validation category {category!r} must contain the official five rows; "
                f"found {len(validation_by_category[category])}"
            )

    selected: list[tuple[str, bytes, str, int, dict[str, Any]]] = []
    counts = {category: 0 for category in CATEGORIES}
    for row_index, row in enumerate(test_rows):
        category = row["category"]
        key_material = (
            f"{SELECTION_NAMESPACE}\0{category}\0{row['question_id']}".encode("utf-8")
        )
        selected.append((category, hashlib.sha256(key_material).digest(), str(row["question_id"]), row_index, row))
        counts[category] += 1
    short = [category for category, count in counts.items() if count < per_category]
    if short:
        raise ValueError(f"test parquet has fewer than {per_category} rows for: {', '.join(short)}")

    tasks: list[dict[str, Any]] = []
    for category in CATEGORIES:
        category_rows = sorted(
            (item for item in selected if item[0] == category),
            key=lambda item: (item[1], item[2], item[3]),
        )[:per_category]
        validation = validation_by_category[category]
        examples = [row for _, row in validation]
        prefix = category_prefix(category, examples)
        example_identities = [
            source_identity(row, "validation", row_index) for row_index, row in validation
        ]
        for _, _, _, row_index, row in category_rows:
            user_prompt = prefix + format_example(row["question"], row["options"])
            task: dict[str, Any] = {
                "index": len(tasks),
                "category": category,
                "question_id": row["question_id"],
                "source_row": source_identity(row, "test", row_index),
                "selection_sha256": sha256_bytes(
                    f"{SELECTION_NAMESPACE}\0{category}\0{row['question_id']}".encode("utf-8")
                ),
                "question": row["question"],
                "options": row["options"],
                "gold_answer": row["answer"],
                "gold_answer_index": row["answer_index"],
                "few_shot_source_rows": example_identities,
                "system_prompt": "",
                "user_prompt": user_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            task["task_sha256"] = self_digest(task, "task_sha256")
            tasks.append(task)

    suite: dict[str, Any] = {
        "schema": SUITE_SCHEMA,
        "source": SOURCE,
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "method": "lowest_sha256_per_category",
            "per_category": per_category,
            "categories": list(CATEGORIES),
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }
    suite["suite_sha256"] = self_digest(suite, "suite_sha256")
    return suite


def validate_task(task: Any, expected_index: int, per_category: int) -> dict[str, Any]:
    task = require_object(task, f"suite task {expected_index}")
    required = {
        "index", "category", "question_id", "source_row", "selection_sha256", "question",
        "options", "gold_answer", "gold_answer_index", "few_shot_source_rows",
        "system_prompt", "user_prompt", "messages", "task_sha256",
    }
    missing = sorted(required - task.keys())
    if missing:
        raise ValueError(f"suite task {expected_index} is missing: {', '.join(missing)}")
    if task["index"] != expected_index or isinstance(task["index"], bool):
        raise ValueError(f"suite task {expected_index} has an invalid index")
    expected_category = CATEGORIES[expected_index // per_category]
    if task["category"] != expected_category:
        raise ValueError(f"suite task {expected_index} is out of stable category order")
    if not isinstance(task["question_id"], (str, int)) or isinstance(task["question_id"], bool):
        raise ValueError(f"suite task {expected_index} has invalid question_id")
    if not isinstance(task["question"], str) or not task["question"]:
        raise ValueError(f"suite task {expected_index} has invalid question")
    options = task["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 10 or not all(
        isinstance(option, str) and option != "N/A" for option in options
    ):
        raise ValueError(f"suite task {expected_index} has invalid options")
    gold_index = task["gold_answer_index"]
    if isinstance(gold_index, bool) or not isinstance(gold_index, int) or not 0 <= gold_index < len(options):
        raise ValueError(f"suite task {expected_index} has invalid gold index")
    if task["gold_answer"] != CHOICES[gold_index]:
        raise ValueError(f"suite task {expected_index} has inconsistent gold answer")
    source_row = require_object(task["source_row"], f"suite task {expected_index} source row")
    if source_row.get("split") != "test" or source_row.get("question_id") != task["question_id"]:
        raise ValueError(f"suite task {expected_index} has inconsistent source identity")
    require_sha256(source_row.get("source_row_sha256"), f"suite task {expected_index} source row digest")
    expected_selection = sha256_bytes(
        f"{SELECTION_NAMESPACE}\0{expected_category}\0{task['question_id']}".encode("utf-8")
    )
    if task["selection_sha256"] != expected_selection:
        raise ValueError(f"suite task {expected_index} has invalid selection digest")
    examples = task["few_shot_source_rows"]
    if not isinstance(examples, list) or len(examples) != 5:
        raise ValueError(f"suite task {expected_index} does not bind five validation examples")
    for example in examples:
        example = require_object(example, f"suite task {expected_index} validation identity")
        if example.get("split") != "validation":
            raise ValueError(f"suite task {expected_index} has invalid validation identity")
        require_sha256(example.get("source_row_sha256"), "validation source row digest")
    if task["system_prompt"] != "":
        raise ValueError(f"suite task {expected_index} changes the official empty system prompt")
    if task["messages"] != [{"role": "user", "content": task["user_prompt"]}]:
        raise ValueError(f"suite task {expected_index} messages and prompt disagree")
    if not isinstance(task["user_prompt"], str) or not task["user_prompt"]:
        raise ValueError(f"suite task {expected_index} has invalid user prompt")
    if not task["user_prompt"].startswith(category_instruction(expected_category)):
        raise ValueError(f"suite task {expected_index} changes the official instruction")
    if not task["user_prompt"].endswith(format_example(task["question"], options)):
        raise ValueError(f"suite task {expected_index} changes the official test question prompt")
    require_sha256(task["task_sha256"], f"suite task {expected_index} digest")
    if task["task_sha256"] != self_digest(task, "task_sha256"):
        raise ValueError(f"suite task {expected_index} digest mismatch")
    return task


def validate_suite(value: Any) -> dict[str, Any]:
    suite = require_object(value, "suite")
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"suite schema must be {SUITE_SCHEMA}")
    if suite.get("source") != SOURCE:
        raise ValueError("suite source revision or file digests do not match the pinned source")
    selection = require_object(suite.get("selection"), "suite selection")
    if selection.get("namespace") != SELECTION_NAMESPACE or selection.get("method") != "lowest_sha256_per_category":
        raise ValueError("suite selection policy is invalid")
    if selection.get("categories") != list(CATEGORIES):
        raise ValueError("suite category set or order is invalid")
    per_category = selection.get("per_category")
    if isinstance(per_category, bool) or not isinstance(per_category, int) or per_category <= 0:
        raise ValueError("suite per-category count is invalid")
    tasks = suite.get("tasks")
    expected_count = len(CATEGORIES) * per_category
    if not isinstance(tasks, list) or len(tasks) != expected_count or suite.get("task_count") != expected_count:
        raise ValueError("suite task list is incomplete")
    for index, task in enumerate(tasks):
        validate_task(task, index, per_category)
    ids = [(task["category"], str(task["question_id"])) for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("suite has duplicate source questions")
    for offset in range(0, len(tasks), per_category):
        block = tasks[offset:offset + per_category]
        expected_examples = block[0]["few_shot_source_rows"]
        if any(task["few_shot_source_rows"] != expected_examples for task in block[1:]):
            raise ValueError("suite category tasks do not share one validation prefix")
        if [task["selection_sha256"] for task in block] != sorted(
            task["selection_sha256"] for task in block
        ):
            raise ValueError("suite tasks are not in selection digest order")
    require_sha256(suite.get("suite_sha256"), "suite digest")
    if suite["suite_sha256"] != self_digest(suite, "suite_sha256"):
        raise ValueError("suite digest mismatch")
    return suite


def validate_plan(value: Any, suite_sha256: str) -> dict[str, Any]:
    plan = require_object(value, "plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA}")
    if plan.get("suite_sha256") != suite_sha256:
        raise ValueError("plan is not bound to this suite")
    if "plan_sha256" in plan:
        require_sha256(plan["plan_sha256"], "plan digest")
        if plan["plan_sha256"] != self_digest(plan, "plan_sha256"):
            raise ValueError("plan digest mismatch")
    return plan


def extract_answer(response: str) -> str | None:
    visible = response.rsplit("</think>", 1)[1] if "</think>" in response else response
    matches = ANSWER_RE.findall(visible)
    return matches[-1] if matches else None


def wilson(successes: int, total: int) -> dict[str, int | float | None]:
    if total == 0:
        return {"successes": 0, "n": 0, "rate": None, "ci95_low": None, "ci95_high": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "n": total,
        "rate": rate,
        "ci95_low": center - half,
        "ci95_high": center + half,
    }


def request_payload(task: dict[str, Any], model_name: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": task["messages"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    }


def chat(url: str, payload: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any], str, str]:
    request_bytes = canonical_bytes(payload)
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        raw_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("chat response is not UTF-8") from exc
    body = strict_json_text(raw_text, "chat response")
    body = require_object(body, "chat response")
    try:
        message = body["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"malformed chat response: {raw_text[:300]}") from exc
    if not isinstance(message, dict) or not isinstance(content, str):
        raise RuntimeError("chat response message content is not text")
    return content, body, raw_text, sha256_bytes(raw)


def validate_runtime(runtime: Any, model_name: str, model_path: str, revision: str | None) -> dict[str, Any]:
    runtime = require_object(runtime, "--runtime-json")
    if runtime.get("identity_verified") is not True:
        raise ValueError("--runtime-json must carry observed identity_verified:true")
    expected_path = str(Path(model_path).resolve())
    comparisons = {
        "model_name": model_name,
        "served_model_name": model_name,
        "model_path": expected_path,
        "model_revision": revision,
    }
    for key, expected in comparisons.items():
        if key not in runtime:
            continue
        actual = runtime[key]
        if key == "model_path" and isinstance(actual, str):
            actual = str(Path(actual).resolve())
        if actual != expected:
            raise ValueError(f"runtime {key} does not match the requested model identity")
    return runtime


def validate_recorded_model_identity(identity: Any, description: str) -> dict[str, Any]:
    identity = require_object(identity, description)
    if not isinstance(identity.get("path"), str) or not identity["path"]:
        raise ValueError(f"{description} has no model path")
    revision = identity.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError(f"{description} has an invalid revision")
    require_sha256(identity.get("config_sha256"), f"{description} config digest")
    require_sha256(identity.get("index_sha256"), f"{description} index digest")
    shards = identity.get("shard_sha256")
    if not isinstance(shards, dict) or not shards:
        raise ValueError(f"{description} has no safetensors shard digests")
    for name, digest in shards.items():
        if (
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or Path(name).name != name
        ):
            raise ValueError(f"{description} has an invalid shard name")
        require_sha256(digest, f"{description} shard {name} digest")
    evidence = identity.get("release_evidence")
    evidence_sha256 = identity.get("release_evidence_sha256")
    if evidence is None:
        if evidence_sha256 is not None:
            raise ValueError(f"{description} has an unpaired release evidence digest")
    elif not isinstance(evidence, str) or not evidence:
        raise ValueError(f"{description} has an invalid release evidence path")
    else:
        require_sha256(evidence_sha256, f"{description} release evidence digest")
    return identity


def checked_model_identity(path: str, revision: str | None, evidence: str | None) -> dict[str, Any]:
    identity = validate_recorded_model_identity(
        model_identity(path, revision, evidence), "model identity"
    )
    root = Path(path)
    config = require_object(load_json(root / "config.json", "model config"), "model config")
    if not config:
        raise ValueError("model config is empty")
    index = require_object(
        load_json(root / "model.safetensors.index.json", "model index"), "model index"
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(name, str) and name
        and isinstance(shard, str) and shard.endswith(".safetensors")
        and Path(shard).name == shard
        for name, shard in weight_map.items()
    ):
        raise ValueError("model index has an invalid weight_map")
    indexed_shards = set(weight_map.values())
    recorded_shards = set(identity["shard_sha256"])
    if indexed_shards != recorded_shards:
        raise ValueError("model index and local safetensors shard set disagree")
    return identity


def generation_binding(model_name: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
        "answer_extraction": "last explicit `The answer is X` or `Answer: X` after last </think>; A-J only",
    }


def validate_reference(
    value: Any,
    suite: dict[str, Any],
    plan: dict[str, Any],
    generation: dict[str, Any],
    *,
    require_unpaired: bool = True,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    report = require_object(value, "reference report")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"reference schema must be {REPORT_SCHEMA}")
    if report.get("status") != "complete":
        raise ValueError("reference report is not complete")
    if not isinstance(report.get("label"), str) or not report["label"].strip():
        raise ValueError("reference report has no label")
    if report.get("suite_sha256") != suite["suite_sha256"]:
        raise ValueError("reference report is not bound to this suite")
    require_sha256(report.get("report_sha256"), "reference report digest")
    if report["report_sha256"] != self_digest(report, "report_sha256"):
        raise ValueError("reference report digest mismatch")
    if require_unpaired and (
        report.get("reference") is not None or report.get("paired") is not None
    ):
        raise ValueError("reference report must be an unpaired BF16 baseline")

    suite_binding = require_object(report.get("suite"), "reference suite binding")
    if (
        suite_binding.get("schema") != SUITE_SCHEMA
        or suite_binding.get("suite_sha256") != suite["suite_sha256"]
        or suite_binding.get("task_count") != suite["task_count"]
    ):
        raise ValueError("reference suite binding is inconsistent")
    require_sha256(suite_binding.get("file_sha256"), "reference suite file digest")
    plan_binding = require_object(report.get("plan"), "reference plan binding")
    if (
        plan_binding.get("schema") != PLAN_SCHEMA
        or plan_binding.get("suite_sha256") != suite["suite_sha256"]
        or plan_binding.get("content_sha256") != sha256_bytes(canonical_bytes(plan))
    ):
        raise ValueError("reference plan binding is inconsistent")
    require_sha256(plan_binding.get("file_sha256"), "reference plan file digest")
    require_sha256(plan_binding.get("content_sha256"), "reference plan content digest")

    reference_generation = require_object(report.get("generation"), "reference generation")
    require_sha256(report.get("generation_sha256"), "reference generation digest")
    if report["generation_sha256"] != sha256_bytes(canonical_bytes(reference_generation)):
        raise ValueError("reference generation digest mismatch")
    comparable_generation_keys = (
        "max_tokens",
        "temperature",
        "stream",
        "chat_template_kwargs",
        "answer_extraction",
    )
    if any(reference_generation.get(key) != generation.get(key) for key in comparable_generation_keys):
        raise ValueError("reference generation contract differs from the candidate")

    reference_identity = validate_recorded_model_identity(
        report.get("candidate_identity"), "reference model identity"
    )
    if not isinstance(report.get("runtime"), dict):
        raise ValueError("reference report lacks runtime identity")
    require_sha256(report.get("candidate_identity_sha256"), "reference model identity digest")
    if report["candidate_identity_sha256"] != sha256_bytes(canonical_bytes(reference_identity)):
        raise ValueError("reference model identity digest mismatch")
    require_sha256(report.get("runtime_sha256"), "reference runtime digest")
    if report["runtime_sha256"] != sha256_bytes(canonical_bytes(report["runtime"])):
        raise ValueError("reference runtime digest mismatch")
    if report["runtime"].get("identity_verified") is not True:
        raise ValueError("reference runtime identity was not verified")

    results = report.get("results")
    tasks = suite["tasks"]
    if (
        not isinstance(results, list)
        or len(results) != len(tasks)
        or report.get("task_count") != len(tasks)
    ):
        raise ValueError("reference report is incomplete")
    by_index: dict[int, dict[str, Any]] = {}
    for position, row_value in enumerate(results):
        row = require_object(row_value, f"reference result {position}")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise ValueError("reference report has duplicate or invalid result indices")
        by_index[index] = row
    if set(by_index) != set(range(len(tasks))):
        raise ValueError("reference report has missing result indices")
    for task in tasks:
        index = task["index"]
        row = by_index[index]
        if (
            row.get("task_sha256") != task["task_sha256"]
            or row.get("category") != task["category"]
            or row.get("question_id") != task["question_id"]
            or row.get("gold_answer") != task["gold_answer"]
        ):
            raise ValueError(f"reference result {index} is not bound to its task")
        request = require_object(row.get("request"), f"reference result {index} request")
        if row.get("request_sha256") != sha256_bytes(canonical_bytes(request)):
            raise ValueError(f"reference result {index} request digest mismatch")
        if (
            request.get("messages") != task["messages"]
            or request.get("max_tokens") != reference_generation["max_tokens"]
            or request.get("temperature") != 0
            or request.get("stream") is not False
            or request.get("chat_template_kwargs")
            != {"enable_thinking": True, "reasoning_effort": "low"}
            or not isinstance(request.get("model"), str)
            or not request["model"]
        ):
            raise ValueError(f"reference result {index} request contract is invalid")

        response = row.get("response")
        if not isinstance(response, str):
            raise ValueError(f"reference result {index} lacks raw response text")
        if row.get("response_sha256") != sha256_bytes(response.encode("utf-8")):
            raise ValueError(f"reference result {index} response digest mismatch")
        raw_api_response = row.get("raw_api_response")
        if not isinstance(raw_api_response, str):
            raise ValueError(f"reference result {index} lacks raw API response")
        if row.get("api_response_sha256") != sha256_bytes(raw_api_response.encode("utf-8")):
            raise ValueError(f"reference result {index} API response digest mismatch")
        parsed_api_response = strict_json_text(
            raw_api_response, f"reference result {index} raw API response"
        )
        if parsed_api_response != row.get("api_response"):
            raise ValueError(f"reference result {index} raw and parsed API responses disagree")
        try:
            api_content = parsed_api_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"reference result {index} API response is malformed") from exc
        if api_content != response:
            raise ValueError(f"reference result {index} response fields disagree")
        extracted = extract_answer(response)
        passed = extracted == task["gold_answer"]
        if (
            row.get("extracted_answer") != extracted
            or type(row.get("passed")) is not bool
            or row["passed"] != passed
        ):
            raise ValueError(f"reference result {index} does not rescore from its raw response")
    expected_overall = wilson(sum(row["passed"] for row in by_index.values()), len(tasks))
    overall = require_object(report.get("overall"), "reference overall metrics")
    if overall.get("pass") != expected_overall:
        raise ValueError("reference overall metrics do not rescore")
    category_metrics = require_object(
        report.get("per_category"), "reference per-category metrics"
    )
    if set(category_metrics) != set(CATEGORIES):
        raise ValueError("reference per-category metrics are incomplete")
    for category in CATEGORIES:
        group = [row for row in by_index.values() if row["category"] == category]
        block = require_object(
            category_metrics[category], f"reference {category} metrics"
        )
        if block.get("pass") != wilson(sum(row["passed"] for row in group), len(group)):
            raise ValueError(f"reference {category} metrics do not rescore")
    return report, by_index


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("--concurrency, --max-tokens and --timeout must be positive")
    if not args.model_name.strip() or not args.label.strip():
        raise ValueError("--model-name and --label must be non-empty")

    suite_path = Path(args.suite)
    plan_path = Path(args.plan)
    suite = validate_suite(load_json(suite_path, "suite"))
    plan = validate_plan(load_json(plan_path, "plan"), suite["suite_sha256"])
    runtime_value = strict_json_text(args.runtime_json, "--runtime-json")
    runtime = validate_runtime(runtime_value, args.model_name, args.model_path, args.model_revision)
    identity = checked_model_identity(args.model_path, args.model_revision, args.release_evidence)
    identity_sha256 = sha256_bytes(canonical_bytes(identity))
    runtime_sha256 = sha256_bytes(canonical_bytes(runtime))
    generation = generation_binding(args.model_name, args.max_tokens, args.timeout)
    generation_sha256 = sha256_bytes(canonical_bytes(generation))

    reference = None
    ref_by_index: dict[int, dict[str, Any]] = {}
    reference_path = Path(args.reference) if args.reference else None
    if reference_path is not None:
        reference, ref_by_index = validate_reference(
            load_json(reference_path, "reference report"), suite, plan, generation
        )

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        payload = request_payload(task, args.model_name, args.max_tokens)
        content, api_response, raw_api_response, api_response_sha256 = chat(
            args.url, payload, args.timeout
        )
        extracted = extract_answer(content)
        return {
            "index": task["index"],
            "category": task["category"],
            "question_id": task["question_id"],
            "task_sha256": task["task_sha256"],
            "gold_answer": task["gold_answer"],
            "request": payload,
            "request_sha256": sha256_bytes(canonical_bytes(payload)),
            "response": content,
            "response_sha256": sha256_bytes(content.encode("utf-8")),
            "extracted_answer": extracted,
            "passed": extracted == task["gold_answer"],
            "api_response": api_response,
            "raw_api_response": raw_api_response,
            "api_response_sha256": api_response_sha256,
        }

    results_by_index: dict[int, dict[str, Any]] = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    try:
        for task in suite["tasks"]:
            futures[pool.submit(run_one, task)] = task
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results_by_index[row["index"]] = row
            print(
                f"{row['category']:16s} {row['index']:3d} "
                f"{'PASS' if row['passed'] else 'fail'} {row['extracted_answer'] or '-'}",
                flush=True,
            )
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    if set(results_by_index) != set(range(len(suite["tasks"]))):
        raise RuntimeError("request execution completed without a full result set")
    results = [results_by_index[index] for index in range(len(suite["tasks"]))]

    if reference is not None:
        for row in results:
            ref = ref_by_index[row["index"]]
            row["reference_passed"] = ref["passed"]
            row["pass_outcome_agreement"] = row["passed"] == ref["passed"]
            row["exact_output_agreement"] = row["response"] == ref["response"]

    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        group = [row for row in results if row["category"] == category]
        block: dict[str, Any] = {
            "pass": wilson(sum(row["passed"] for row in group), len(group))
        }
        if reference is not None:
            reference_passes = [row for row in group if row["reference_passed"]]
            block["bf16_pass_retention"] = wilson(
                sum(row["passed"] for row in reference_passes), len(reference_passes)
            )
            block["regressions"] = sum(
                row["reference_passed"] and not row["passed"] for row in group
            )
            block["improvements"] = sum(
                not row["reference_passed"] and row["passed"] for row in group
            )
        per_category[category] = block

    overall = {"pass": wilson(sum(row["passed"] for row in results), len(results))}
    paired = None
    if reference is not None:
        reference_passes = [row for row in results if row["reference_passed"]]
        paired = {
            "bf16_pass_retention": wilson(
                sum(row["passed"] for row in reference_passes), len(reference_passes)
            ),
            "pass_outcome_agreement": wilson(
                sum(row["pass_outcome_agreement"] for row in results), len(results)
            ),
            "exact_output_agreement": wilson(
                sum(row["exact_output_agreement"] for row in results), len(results)
            ),
            "regressions": sum(
                row["reference_passed"] and not row["passed"] for row in results
            ),
            "improvements": sum(
                not row["reference_passed"] and row["passed"] for row in results
            ),
        }

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "suite": {
            "path": str(suite_path.resolve()),
            "file_sha256": sha256_file(suite_path),
            "schema": suite["schema"],
            "suite_sha256": suite["suite_sha256"],
            "task_count": suite["task_count"],
        },
        "suite_sha256": suite["suite_sha256"],
        "plan": {
            "path": str(plan_path.resolve()),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": sha256_bytes(canonical_bytes(plan)),
            "schema": plan["schema"],
            "suite_sha256": plan["suite_sha256"],
        },
        "candidate_identity": identity,
        "candidate_identity_sha256": identity_sha256,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
        "generation": generation,
        "generation_sha256": generation_sha256,
        "reference": (
            {
                "path": str(reference_path.resolve()),
                "file_sha256": sha256_file(reference_path),
                "report_sha256": reference["report_sha256"],
                "candidate_identity": reference["candidate_identity"],
                "candidate_identity_sha256": reference["candidate_identity_sha256"],
                "label": reference["label"],
            }
            if reference is not None and reference_path is not None
            else None
        ),
        "task_count": len(results),
        "overall": overall,
        "per_category": per_category,
        "paired": paired,
        "results": results,
    }
    report["report_sha256"] = self_digest(report, "report_sha256")
    validate_reference(
        report, suite, plan, generation, require_unpaired=False
    )
    return report


def command_build(args: argparse.Namespace) -> int:
    suite = build_suite(Path(args.test_parquet), Path(args.validation_parquet), args.per_category)
    atomic_write_json(Path(args.out), suite)
    print(json.dumps({"suite_sha256": suite["suite_sha256"], "tasks": suite["task_count"]}))
    return 0


def command_run(args: argparse.Namespace) -> int:
    report = run_suite(args)
    atomic_write_json(Path(args.out), report)
    print(json.dumps({"overall": report["overall"], "paired": report["paired"]}), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build the pinned deterministic MMLU-Pro suite")
    build.add_argument("--test-parquet", required=True)
    build.add_argument("--validation-parquet", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--per-category", type=int, default=5)
    build.set_defaults(handler=command_build)

    run = commands.add_parser("run", help="run and atomically publish a complete suite report")
    run.add_argument("--suite", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--model-name", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-revision")
    run.add_argument("--release-evidence")
    run.add_argument("--runtime-json", required=True)
    run.add_argument("--reference")
    run.add_argument("--out", required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--timeout", type=int, default=600)
    run.set_defaults(handler=command_run)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("interrupted; no report was published", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}; no report was published", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
