#!/usr/bin/env python3
"""Minimal decode-throughput probe against an OpenAI-compatible endpoint.

Greedy, fixed output length, ignore_eos, so token counts are exact and the
measurement is decode-bound. Reports aggregate output tok/s per concurrency and
the per-stream rate, plus TTFT from a single-token request.
"""
import argparse, json, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020/v1/completions")
    ap.add_argument("--model", default="m")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--label", default="candidate")
    args = ap.parse_args()

    prompt = "Write a detailed technical explanation of tensor parallelism. " * 4
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.tokens,
            "temperature": 0, "ignore_eos": True, "stream": False}

    # warm one request so any first-call JIT is excluded
    post(args.url, dict(body, max_tokens=8))

    out = {"label": args.label, "output_tokens_per_request": args.tokens, "runs": []}
    for c in args.concurrency:
        started = time.monotonic()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda _: post(args.url, body), range(c)))
        elapsed = time.monotonic() - started
        completion = sum(r["usage"]["completion_tokens"] for r in res)
        out["runs"].append({
            "concurrency": c,
            "elapsed_sec": round(elapsed, 3),
            "output_tokens": completion,
            "aggregate_tok_s": round(completion / elapsed, 2),
            "per_stream_tok_s": round(completion / elapsed / c, 2),
        })
        print(json.dumps(out["runs"][-1]), flush=True)

    # single-token latency as a TTFT proxy
    lat = []
    for _ in range(5):
        t = time.monotonic()
        post(args.url, dict(body, max_tokens=1))
        lat.append(time.monotonic() - t)
    out["ttft_proxy_sec"] = round(statistics.median(lat), 4)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
