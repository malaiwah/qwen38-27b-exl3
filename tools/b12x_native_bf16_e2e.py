#!/usr/bin/env python3
"""Reproducible Qwen3.8 native-BF16 serving bracket measurement.

Drives an already-healthy Docker Compose service. It does not mutate service
state. Every decode request is greedy, non-streaming, and fixed-length via
``ignore_eos``. Distinct arm-neutral prompts avoid cross-request prefix-cache
sharing while remaining byte-identical across A/B/A boots. Prefill requests
use distinct exact-length token-id prompts and fail unless server counters
prove zero cache hits and full local KV computation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "b12x.qwen38.native_bf16.e2e_arm.v2"
BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B"
CONTAINER = "qwen38-27b"
PROMPT_TEMPLATES = (
    "Write a detailed technical history of {topic}, with dates and mechanisms.",
    "Explain step by step how {topic} works internally and where it can fail.",
    "Compare {topic} with its closest alternative using twelve concrete criteria.",
    "Teach an advanced student the open research problems in {topic}.",
)
TOPICS = ("cryptography", "orbital mechanics", "compilers", "acoustics")
PP_PROMPT = (
    "The quick brown fox jumps over the lazy dog. " * 100
    + "In machine learning, gradient descent is an optimization algorithm. " * 50
    + "The mitochondria is the powerhouse of the cell. " * 50
)
RELEVANT_ENV = (
    "PREFIX_CACHING",
    "VLLM_EXL3_B12X_NATIVE_BF16",
    "VLLM_EXL3_B12X_REACHABILITY",
    "VLLM_EXL3_B12X_MIN_M",
    "VLLM_EXL3_B12X_ANY_BITS",
    "VLLM_EXL3_B12X_N_RANGE",
    "VLLM_EXL3_GRAPH_DECODE",
    "VLLM_EXL3_MULTIPRECISION",
)
PREFILL_WARMUPS = 2
PREFILL_METRICS = {
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total", ()),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total", ()),
    "prompt_tokens": ("vllm:prompt_tokens_total", ()),
    "prompt_tokens_local_compute": (
        "vllm:prompt_tokens_by_source_total",
        ('source="local_compute"',),
    ),
    "prompt_tokens_local_cache_hit": (
        "vllm:prompt_tokens_by_source_total",
        ('source="local_cache_hit"',),
    ),
    "prompt_tokens_external_kv": (
        "vllm:prompt_tokens_by_source_total",
        ('source="external_kv_transfer"',),
    ),
    "prefill_kv_computed_tokens": (
        "vllm:request_prefill_kv_computed_tokens_sum",
        (),
    ),
    "prefill_request_count": (
        "vllm:request_prefill_kv_computed_tokens_count",
        (),
    ),
}


def _json_request(path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def _health() -> None:
    with urllib.request.urlopen(BASE_URL + "/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"health returned HTTP {response.status}")


def _metrics_snapshot() -> dict[str, float]:
    with urllib.request.urlopen(BASE_URL + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    result: dict[str, float] = {}
    for key, (metric, required_labels) in PREFILL_METRICS.items():
        values = []
        for line in lines:
            if not (line.startswith(metric + "{") or line.startswith(metric + " ")):
                continue
            if not all(label in line for label in required_labels):
                continue
            values.append(float(line.rsplit(" ", 1)[1]))
        if not values:
            raise RuntimeError(
                f"required prefill metric is missing: {metric} {required_labels}"
            )
        result[key] = sum(values)
    return result


def _metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {key: after[key] - before[key] for key in PREFILL_METRICS}


def _spec_counters() -> tuple[int, int]:
    with urllib.request.urlopen(BASE_URL + "/metrics", timeout=30) as response:
        text = response.read().decode()
    accepted = 0
    drafted = 0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            accepted += int(float(line.split()[-1]))
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted += int(float(line.split()[-1]))
    return accepted, drafted


def _gpu_snapshot() -> dict[str, Any]:
    fields = (
        "uuid",
        "name",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "memory.used",
        "memory.total",
        "clocks_throttle_reasons.active",
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi row: {result.stdout!r}")
    return dict(zip(fields, values, strict=True))


def _container_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", CONTAINER],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)[0]
    env = {}
    for entry in record["Config"].get("Env", []):
        name, _, value = entry.partition("=")
        if name in RELEVANT_ENV:
            env[name] = value
    return {
        "container_id": record["Id"],
        "image_id": record["Image"],
        "command": record["Config"].get("Cmd"),
        "entrypoint": record["Config"].get("Entrypoint"),
        "state": {
            "status": record["State"]["Status"],
            "running": record["State"]["Running"],
            "oom_killed": record["State"]["OOMKilled"],
            "restart_count": record["RestartCount"],
            "started_at": record["State"]["StartedAt"],
        },
        "environment": env,
    }


def _prompt(concurrency: int, repetition: int, stream: int) -> str:
    index = stream % len(PROMPT_TEMPLATES)
    return (
        PROMPT_TEMPLATES[index].format(topic=TOPICS[index])
        + f"\nDeterministic benchmark case C{concurrency}-R{repetition}-S{stream}."
    )


def _decode_request(
    concurrency: int, repetition: int, stream: int, tokens: int
) -> dict[str, Any]:
    prompt = _prompt(concurrency, repetition, stream)
    started = time.perf_counter()
    response = _json_request(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        },
    )
    elapsed = time.perf_counter() - started
    text = str(response["choices"][0]["text"])
    return {
        "completion_tokens": int(response["usage"]["completion_tokens"]),
        "prompt_tokens": int(response["usage"]["prompt_tokens"]),
        "request_seconds": elapsed,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_prefix": text[:120],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def _decode_round(concurrency: int, repetition: int, tokens: int) -> dict[str, Any]:
    accepted_before, drafted_before = _spec_counters()
    gpu_before = _gpu_snapshot()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = list(
            pool.map(
                lambda stream: _decode_request(concurrency, repetition, stream, tokens),
                range(concurrency),
            )
        )
    wall_seconds = time.perf_counter() - started
    gpu_after = _gpu_snapshot()
    accepted_after, drafted_after = _spec_counters()
    completion_tokens = sum(row["completion_tokens"] for row in requests)
    expected_tokens = concurrency * tokens
    if completion_tokens != expected_tokens:
        raise RuntimeError(
            f"fixed decode returned {completion_tokens} tokens, expected {expected_tokens}"
        )
    accepted_delta = accepted_after - accepted_before
    drafted_delta = drafted_after - drafted_before
    return {
        "repetition": repetition,
        "wall_seconds": wall_seconds,
        "completion_tokens": completion_tokens,
        "aggregate_tok_s": completion_tokens / wall_seconds,
        "per_stream_tok_s": completion_tokens / wall_seconds / concurrency,
        "accepted_delta": accepted_delta,
        "drafted_delta": drafted_delta,
        "acceptance": (accepted_delta / drafted_delta if drafted_delta > 0 else None),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "requests": requests,
    }


def _measure_decode_level(
    concurrency: int,
    *,
    tokens: int,
    repetitions: int,
    warmup_rounds: int,
    settle_seconds: float,
) -> dict[str, Any]:
    warmups = [
        _decode_round(concurrency, -1 - index, min(tokens, 128))
        for index in range(warmup_rounds)
    ]
    rounds = []
    for repetition in range(repetitions):
        rounds.append(_decode_round(concurrency, repetition, tokens))
        if settle_seconds:
            time.sleep(settle_seconds)
    values = [row["aggregate_tok_s"] for row in rounds]
    acceptances = [row["acceptance"] for row in rounds if row["acceptance"] is not None]
    return {
        "concurrency": concurrency,
        "tokens_per_request": tokens,
        "warmup_rounds": warmups,
        "rounds": rounds,
        "aggregate_tok_s_median": statistics.median(values),
        "aggregate_tok_s_min": min(values),
        "aggregate_tok_s_max": max(values),
        "acceptance_median": (statistics.median(acceptances) if acceptances else None),
    }


def _cache_proof_prefill_prompts(count: int) -> list[list[int]]:
    tokenized = _json_request(
        "/tokenize",
        {"model": MODEL, "prompt": PP_PROMPT, "add_special_tokens": False},
    )
    base_tokens = [int(token) for token in tokenized["tokens"]]
    if int(tokenized["count"]) != len(base_tokens) or not base_tokens:
        raise RuntimeError("live tokenizer returned an inconsistent prefill prompt")
    distinct_tokens = list(dict.fromkeys(base_tokens))
    salt_width = 4
    if len(distinct_tokens) < count * salt_width:
        raise RuntimeError(
            "prefill prompt does not contain enough distinct salt tokens"
        )
    prompts = []
    for index in range(count):
        prompt = list(base_tokens)
        begin = index * salt_width
        prompt[:salt_width] = distinct_tokens[begin : begin + salt_width]
        prompts.append(prompt)
    if len({tuple(prompt[:16]) for prompt in prompts}) != count:
        raise RuntimeError("cache-proof prefill prefixes are not unique")
    return prompts


def _prefill_request(prompt: list[int], request_id: str) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    before = _metrics_snapshot()
    started = time.perf_counter()
    response = _json_request("/v1/completions", payload)
    wall_seconds = time.perf_counter() - started
    metrics = _metric_delta(before, _metrics_snapshot())
    prompt_tokens = int(response["usage"]["prompt_tokens"])
    expected = float(len(prompt))
    required = {
        "prefix_cache_queries": expected,
        "prefix_cache_hits": 0.0,
        "prompt_tokens": expected,
        "prompt_tokens_local_compute": expected,
        "prompt_tokens_local_cache_hit": 0.0,
        "prompt_tokens_external_kv": 0.0,
        "prefill_kv_computed_tokens": expected,
        "prefill_request_count": 1.0,
    }
    failures = {
        key: {"expected": value, "observed": metrics[key]}
        for key, value in required.items()
        if metrics[key] != value
    }
    if prompt_tokens != len(prompt):
        failures["response_prompt_tokens"] = {
            "expected": len(prompt),
            "observed": prompt_tokens,
        }
    if failures:
        raise RuntimeError(f"{request_id} is not a cache-proof prefill: {failures}")
    prompt_bytes = json.dumps(prompt, separators=(",", ":")).encode()
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_prefix_token_ids": prompt[:16],
        "wall_seconds": wall_seconds,
        "uncached_prompt_tok_s_including_one_decode_step": prompt_tokens / wall_seconds,
        "metrics_delta": metrics,
        "cache_proof_pass": True,
    }


def _measure_prefill(repetitions: int) -> dict[str, Any]:
    prompts = _cache_proof_prefill_prompts(PREFILL_WARMUPS + repetitions)
    warmups = [
        _prefill_request(prompts[index], f"warmup-{index}")
        for index in range(PREFILL_WARMUPS)
    ]
    rounds = []
    for repetition in range(repetitions):
        row = _prefill_request(
            prompts[PREFILL_WARMUPS + repetition],
            f"timed-{repetition}",
        )
        row["repetition"] = repetition
        rounds.append(row)
    values = [row["uncached_prompt_tok_s_including_one_decode_step"] for row in rounds]
    return {
        "method": (
            "Distinct exact-length token-id prompts; headline is logical prompt tokens "
            "divided by client wall time including one decode step. Every warmup and timed "
            "request requires zero cache hits and full local KV computation from server metrics."
        ),
        "warmups": warmups,
        "rounds": rounds,
        "uncached_prompt_tok_s_median_including_one_decode_step": statistics.median(
            values
        ),
        "prompt_tokens": rounds[0]["prompt_tokens"],
        "cache_proof_pass": all(row["cache_proof_pass"] for row in [*warmups, *rounds]),
    }


def _sanity() -> dict[str, Any]:
    response = _json_request(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        },
    )
    text = str(response["choices"][0]["text"])
    return {"pass": "Paris" in text, "text": text, "usage": response["usage"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--prefill-repetitions", type=int, default=5)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    args = parser.parse_args()

    _health()
    script_path = Path(__file__).resolve()
    started = dt.datetime.now(dt.timezone.utc)
    result = {
        "schema": SCHEMA,
        "label": args.label,
        "started_at": started.isoformat(),
        "command": {
            "concurrency": args.concurrency,
            "tokens": args.tokens,
            "repetitions": args.repetitions,
            "warmup_rounds": args.warmup_rounds,
            "prefill_repetitions": args.prefill_repetitions,
            "settle_seconds": args.settle_seconds,
        },
        "tool": {
            "path": str(script_path),
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "container_before": _container_snapshot(),
        "gpu_before": _gpu_snapshot(),
        "models": _json_request("/v1/models"),
        "sanity": _sanity(),
        "decode": [],
    }
    for raw_level in args.concurrency.split(","):
        concurrency = int(raw_level)
        result["decode"].append(
            _measure_decode_level(
                concurrency,
                tokens=args.tokens,
                repetitions=args.repetitions,
                warmup_rounds=args.warmup_rounds,
                settle_seconds=args.settle_seconds,
            )
        )
    result["prefill"] = _measure_prefill(args.prefill_repetitions)
    result["gpu_after"] = _gpu_snapshot()
    result["container_after"] = _container_snapshot()
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if not result["sanity"]["pass"]:
        raise RuntimeError(f"sanity failed: {result['sanity']['text']!r}")
    if result["container_after"]["state"] != result["container_before"]["state"]:
        raise RuntimeError("container state changed during the measurement arm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "decode": {
                    f"C{row['concurrency']}": row["aggregate_tok_s_median"]
                    for row in result["decode"]
                },
                "uncached_prefill_tok_s_including_one_decode_step": result["prefill"][
                    "uncached_prompt_tok_s_median_including_one_decode_step"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
