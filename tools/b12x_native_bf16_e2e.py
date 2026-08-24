#!/usr/bin/env python3
"""Reproducible Qwen3.8 native-BF16 serving bracket measurement.

Drives an already-healthy Docker Compose service. It does not mutate service
state. Every decode request is greedy, non-streaming, and fixed-length via
``ignore_eos``. Distinct arm-neutral prompts avoid cross-request prefix-cache
sharing while remaining byte-identical across A/B/A boots.
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


SCHEMA = "b12x.qwen38.native_bf16.e2e_arm.v1"
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
    "VLLM_EXL3_B12X_NATIVE_BF16",
    "VLLM_EXL3_B12X_REACHABILITY",
    "VLLM_EXL3_B12X_MIN_M",
    "VLLM_EXL3_B12X_ANY_BITS",
    "VLLM_EXL3_B12X_N_RANGE",
    "VLLM_EXL3_GRAPH_DECODE",
    "VLLM_EXL3_MULTIPRECISION",
)


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


def _measure_prefill(repetitions: int) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": PP_PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    for _ in range(2):
        _json_request("/v1/completions", payload)
    rounds = []
    for repetition in range(repetitions):
        started = time.perf_counter()
        response = _json_request("/v1/completions", payload)
        wall_seconds = time.perf_counter() - started
        prompt_tokens = int(response["usage"]["prompt_tokens"])
        rounds.append(
            {
                "repetition": repetition,
                "prompt_tokens": prompt_tokens,
                "wall_seconds": wall_seconds,
                "prompt_tok_s": prompt_tokens / wall_seconds,
            }
        )
    values = [row["prompt_tok_s"] for row in rounds]
    return {
        "rounds": rounds,
        "prompt_tok_s_median": statistics.median(values),
        "prompt_tokens": rounds[0]["prompt_tokens"],
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
                "prefill_tok_s": result["prefill"]["prompt_tok_s_median"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
