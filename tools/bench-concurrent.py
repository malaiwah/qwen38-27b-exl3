#!/usr/bin/env python3
"""Concurrent decode throughput: C1 / C4 / C8 aggregate tok/s, GLM-style.

Every TG number this project published so far is single-stream (C1). This tool
measures aggregate decode across N simultaneous streams the way the GLM-5.2 r34
page reports it, so the two stacks' numbers are comparable and the HF cards can
stop implying C1 numbers hold under load.

Method per concurrency level C:
  - C simultaneous /v1/completions requests, distinct prompts (prefix-cache-proof),
    max_tokens=TOKENS, temperature=0, ignore_eos via max_tokens padding prompt.
  - warmup round first (full C), then MEASURE round timed end-to-end.
  - aggregate tok/s = sum(completion_tokens) / wall  (the GLM convention);
    per-user mean = aggregate / C.
  - MTP acceptance differenced across the measure round from /metrics.

Usage: bench-concurrent.py [--levels 1,4,8] [--tokens 200] [--reps 3]
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
MODEL = "Qwen3.8-27B"

PROMPTS = [
    "Write a detailed essay about the history of {}.",
    "Explain, step by step, how {} works internally.",
    "List and describe twelve important facts about {}.",
    "Compare and contrast {} with its closest alternative.",
    "Describe a day in the life of someone who studies {}.",
    "What are the main open problems in the field of {}?",
    "Summarise the last two decades of progress in {}.",
    "Write a beginner's tutorial introducing {}.",
]
TOPICS = ["volcanology", "typography", "beekeeping", "cryptography",
          "orbital mechanics", "fermentation", "cartography", "acoustics"]


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spec_counters():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        text = r.read().decode()
    acc = drf = 0.0
    for ln in text.splitlines():
        if ln.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc += float(ln.split()[-1])
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drf += float(ln.split()[-1])
    return acc, drf


def one_request(i, tokens):
    p = PROMPTS[i % len(PROMPTS)].format(TOPICS[i % len(TOPICS)])
    t0 = time.perf_counter()
    r = post("/v1/completions", {
        "model": MODEL, "prompt": p, "max_tokens": tokens, "temperature": 0.0})
    dt = time.perf_counter() - t0
    return r["usage"]["completion_tokens"], dt


def measure_level(c, tokens, reps):
    # warmup: one full concurrent round, unmeasured
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        list(ex.map(lambda i: one_request(i, tokens), range(c)))
    aggs = []
    acc_rate = None
    for rep in range(reps):
        a0, d0 = spec_counters()
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            results = list(ex.map(lambda i: one_request(i + rep * c, tokens),
                                  range(c)))
        wall = time.perf_counter() - t0
        a1, d1 = spec_counters()
        toks = sum(t for t, _ in results)
        aggs.append(toks / wall)
        if d1 > d0:
            acc_rate = (a1 - a0) / (d1 - d0)
    return {
        "concurrency": c,
        "aggregate_tok_s_median": statistics.median(aggs),
        "aggregate_tok_s_all": [round(a, 1) for a in aggs],
        "per_user_tok_s": statistics.median(aggs) / c,
        "acceptance": acc_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        urllib.request.urlopen(BASE + "/health", timeout=10)
    except Exception as e:
        print(f"server unreachable: {e}", file=sys.stderr)
        return 2

    levels = [int(x) for x in args.levels.split(",")]
    out = []
    for c in levels:
        r = measure_level(c, args.tokens, args.reps)
        out.append(r)
        acc = f"{r['acceptance']:.3f}" if r["acceptance"] is not None else "n/a"
        print(f"  C{c}: aggregate {r['aggregate_tok_s_median']:7.1f} tok/s "
              f"(per-user {r['per_user_tok_s']:6.1f})  [MTP acc {acc}]  "
              f"all={r['aggregate_tok_s_all']}")
    if args.json_out:
        json.dump({"levels": out, "tokens": args.tokens, "reps": args.reps},
                  open(args.json_out, "w"), indent=2)
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
