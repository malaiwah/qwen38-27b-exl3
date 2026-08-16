#!/usr/bin/env python3
"""Measure one running server: prefill tok/s, decode tok/s, speculative step economics.

Every configuration in the lever sweep is measured by this one client so the rows are
comparable: prompts are frozen token-id arrays (identical bytes for every configuration),
each (temperature, concurrency) point gets one discarded warm repeat followed by N timed
repeats, and the speculative counters are read from /metrics around each timed repeat.

Accepted tokens per step and step time are reported separately, not folded into a single
ratio: a depth change can buy acceptance and pay for it in latency, and the ratio hides
that. Step count comes from vllm:iteration_tokens_total_count (exact engine iterations),
not from drafts/concurrency, so the ramp-up and tail iterations of a concurrent run are
counted correctly.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCALARS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "iterations": "vllm:iteration_tokens_total_count",
    "iteration_tokens": "vllm:iteration_tokens_total_sum",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
}
PER_POS = re.compile(
    r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}'
    r"\s+([0-9.eE+-]+)"
)


def post(url: str, payload: dict, timeout: int = 3600) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def scrape(base: str, timeout: int = 60) -> dict:
    """Read the counters this probe differences. A missing counter leaves its key absent."""
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as response:
        text = response.read().decode()
    out: dict = {}
    per_pos: dict[int, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = PER_POS.match(line)
        if match:
            per_pos[int(match.group(1))] = float(match.group(2))
            continue
        for key, metric in SCALARS.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                out[key] = float(line.rsplit(" ", 1)[1])
    out["per_pos"] = per_pos
    return out


def delta(before: dict, after: dict) -> dict:
    out: dict = {
        key: after[key] - before[key] for key in SCALARS if key in before and key in after
    }
    positions = sorted(set(before["per_pos"]) | set(after["per_pos"]))
    out["accepted_per_pos"] = [
        after["per_pos"].get(p, 0.0) - before["per_pos"].get(p, 0.0) for p in positions
    ]
    return out


def spec_economics(d: dict, elapsed: float, output_tokens: int) -> dict:
    """Turn counter deltas into the decision metric and, separately, its two factors."""
    steps = d.get("iterations", 0.0)
    drafts = d.get("drafts", 0.0)
    accepted = d.get("accepted", 0.0)
    draft_tokens = d.get("draft_tokens", 0.0)
    row: dict = {
        "engine_steps": steps,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted,
        "accepted_per_pos": d.get("accepted_per_pos", []),
        "preemptions": d.get("preemptions", 0.0),
    }
    if drafts:
        # One draft entry per (request, step): the tokens that request emits in that step
        # are its accepted draft tokens plus the always-emitted target token.
        row["accepted_tokens_per_step_per_request"] = round((accepted + drafts) / drafts, 4)
        row["draft_acceptance_rate"] = (
            round(accepted / draft_tokens, 4) if draft_tokens else None
        )
    if steps:
        row["step_time_ms"] = round(elapsed / steps * 1000.0, 4)
        row["tokens_per_step_batch_wide"] = round(output_tokens / steps, 4)
        if drafts:
            # The decision metric: accepted tokens per step over step time.
            row["per_request_tok_s_from_steps"] = round(
                (accepted + drafts) / drafts / (elapsed / steps), 2
            )
    return row


def decode_point(
    base: str,
    model: str,
    prompts: list[list[int]],
    concurrency: int,
    tokens: int,
    temperature: float,
    seed: int | None,
    repeats: int,
) -> dict:
    url = base.rstrip("/") + "/v1/completions"

    def body(index: int) -> dict:
        payload = {
            "model": model,
            "prompt": prompts[index % len(prompts)],
            "max_tokens": tokens,
            "min_tokens": tokens,
            "ignore_eos": True,
            "temperature": temperature,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    def fire() -> tuple[float, int, int]:
        started = time.monotonic()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(lambda i: post(url, body(i)), range(concurrency)))
        elapsed = time.monotonic() - started
        completion = sum(r["usage"]["completion_tokens"] for r in results)
        return elapsed, completion, results[0]["usage"]["prompt_tokens"]

    fire()  # warm this (temperature, concurrency) shape: graph replay, sampler JIT
    reps = []
    for _ in range(repeats):
        before = scrape(base)
        elapsed, completion, prompt_tokens = fire()
        after = scrape(base)
        rep = {
            "elapsed_sec": round(elapsed, 4),
            "output_tokens": completion,
            "prompt_tokens_each": prompt_tokens,
            "aggregate_tok_s": round(completion / elapsed, 2),
            "per_stream_tok_s": round(completion / elapsed / concurrency, 2),
        }
        rep.update(spec_economics(delta(before, after), elapsed, completion))
        reps.append(rep)

    def median_of(key: str):
        values = [r[key] for r in reps if r.get(key) is not None]
        return round(statistics.median(values), 4) if values else None

    rates = [r["aggregate_tok_s"] for r in reps]
    return {
        "concurrency": concurrency,
        "output_tokens_per_request": tokens,
        "temperature": temperature,
        "seed": seed,
        "repeats": reps,
        "median_aggregate_tok_s": round(statistics.median(rates), 2),
        "median_per_stream_tok_s": round(statistics.median(rates) / concurrency, 2),
        "min_aggregate_tok_s": min(rates),
        "max_aggregate_tok_s": max(rates),
        "median_step_time_ms": median_of("step_time_ms"),
        "median_accepted_tokens_per_step": median_of("accepted_tokens_per_step_per_request"),
        "median_draft_acceptance_rate": median_of("draft_acceptance_rate"),
        "median_tokens_per_step_batch_wide": median_of("tokens_per_step_batch_wide"),
    }


def prefill_point(base: str, model: str, prompt: list[int], repeats: int) -> dict:
    url = base.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": False,
    }
    post(url, payload)  # warm this prompt shape
    reps = []
    for _ in range(repeats):
        before = scrape(base)
        started = time.monotonic()
        result = post(url, payload)
        elapsed = time.monotonic() - started
        d = delta(before, scrape(base))
        got = result["usage"]["prompt_tokens"]
        reps.append(
            {
                "prompt_tokens": got,
                "elapsed_sec": round(elapsed, 4),
                "prefill_tok_s_including_one_decode_step": round(got / elapsed, 1),
                "engine_steps": d.get("iterations", 0.0),
                "prefix_cache_queries": d.get("prefix_cache_queries", 0.0),
                "prefix_cache_hits": d.get("prefix_cache_hits", 0.0),
            }
        )
    rates = [r["prefill_tok_s_including_one_decode_step"] for r in reps]
    return {
        "requested_prompt_tokens": len(prompt),
        "measured_prompt_tokens": reps[0]["prompt_tokens"],
        "repeats": reps,
        "median_prefill_tok_s": round(statistics.median(rates), 1),
        "min_prefill_tok_s": min(rates),
        "max_prefill_tok_s": max(rates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8231")
    parser.add_argument("--model", default="m")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[2048, 6144])
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.0, 0.6],
        help="0 is the deterministic arm; a non-zero arm is seeded and carries the "
        "acceptance-sensitive numbers",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    decode_prompts = [
        json.loads(path.read_text()) for path in sorted(prompt_dir.glob("decode-*.json"))
    ]
    if not decode_prompts:
        raise SystemExit(f"no decode-*.json prompts under {prompt_dir}")

    out: dict = {
        "schema": "qwen38-perf-probe-5090/1",
        "label": args.label,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "output_tokens": args.tokens,
            "timed_repeats_per_point": args.repeats,
            "discarded_warm_repeats_per_point": 1,
            "decode_prompt_tokens_each": len(decode_prompts[0]),
            "distinct_decode_prompts": len(decode_prompts),
            "step_count_source": "vllm:iteration_tokens_total_count delta",
            "decision_metric": "accepted tokens per step / step time",
        },
        "baseline_counters": {k: v for k, v in scrape(args.base).items() if k != "per_pos"},
    }

    prefill = []
    for n in args.prefill_lengths:
        path = prompt_dir / f"prefill-{n}.json"
        if not path.is_file():
            raise SystemExit(f"missing frozen prompt {path}")
        point = prefill_point(args.base, args.model, json.loads(path.read_text()), args.repeats)
        prefill.append(point)
        print(
            json.dumps(
                {
                    "prefill_tokens": point["measured_prompt_tokens"],
                    "median_prefill_tok_s": point["median_prefill_tok_s"],
                }
            ),
            flush=True,
        )
    out["prefill"] = prefill

    decode = []
    for temperature in args.temperatures:
        for concurrency in args.concurrency:
            point = decode_point(
                args.base,
                args.model,
                decode_prompts,
                concurrency,
                args.tokens,
                temperature,
                args.seed if temperature > 0 else None,
                args.repeats,
            )
            decode.append(point)
            print(
                json.dumps(
                    {
                        "T": temperature,
                        "C": concurrency,
                        "agg_tok_s": point["median_aggregate_tok_s"],
                        "acc_tok_per_step": point["median_accepted_tokens_per_step"],
                        "step_ms": point["median_step_time_ms"],
                    }
                ),
                flush=True,
            )
    out["decode"] = decode

    final = scrape(args.base)
    out["final_counters"] = {k: v for k, v in final.items() if k != "per_pos"}
    out["final_accepted_per_pos"] = {str(k): v for k, v in sorted(final["per_pos"].items())}
    queries = final.get("prefix_cache_queries", 0.0)
    out["prefix_cache_hit_rate"] = (
        round(final.get("prefix_cache_hits", 0.0) / queries, 6) if queries else 0.0
    )

    temporary = Path(args.out + ".tmp")
    temporary.write_text(json.dumps(out, indent=2) + "\n")
    temporary.replace(Path(args.out))
    print(
        json.dumps({"wrote": args.out, "prefix_cache_hit_rate": out["prefix_cache_hit_rate"]}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
