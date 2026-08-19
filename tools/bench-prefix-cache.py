#!/usr/bin/env python3
"""Measure what prefix caching buys at depth on the Qwen3.8-27B hybrid.

The prefill law t(D) = a*D + b*D^2 (see receipts/context-curve-2026-08-19.md)
is not improved by caching -- caching SKIPS it for the span already resident.
That makes it the highest-leverage deep-context lever we have, and the only one
that touches the quadratic term at all, because it removes the work rather than
speeding it up.

Method: send the SAME long prompt N+1 times and read the server's own TTFT
histogram for each request.

  * request 0  -> cold, populates the cache
  * requests 1..N -> should hit the cache

TTFT_cold / TTFT_warm is the speedup on the prefill phase. With caching off,
every request is cold and the ratio must be ~1.0; that arm is the control, and
it is what makes a warm-arm speedup attributable to caching rather than to
allocator or clock warmup.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"


def _metric(name: str) -> tuple[float, float]:
    """Return (sum, count) for a Prometheus histogram."""
    s = c = 0.0
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=20) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith(f"{name}_sum"):
                    s = float(line.rsplit(" ", 1)[1])
                elif line.startswith(f"{name}_count"):
                    c = float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return s, c


def _served_model() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def _one(prompt: str, model: str, gen: int) -> tuple[float, float, int]:
    """Issue one completion; return (server TTFT, wall time, prompt_tokens)."""
    s0, c0 = _metric("vllm:time_to_first_token_seconds")
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": gen, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.loads(r.read())
    wall = time.perf_counter() - t0

    # The histogram observation can land just after the response is written;
    # poll briefly rather than reporting a spurious miss.
    ttft = 0.0
    for _ in range(20):
        s1, c1 = _metric("vllm:time_to_first_token_seconds")
        if c1 > c0:
            ttft = (s1 - s0) / (c1 - c0)
            break
        time.sleep(0.5)
    return ttft, wall, payload.get("usage", {}).get("prompt_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=131072)
    ap.add_argument("--repeats", type=int, default=3, help="warm requests after the cold one")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    model = _served_model()
    # Distinct prefix so a previous arm's cache cannot serve this one.
    prompt = f"session-{int(time.time())} " + ("word " * args.depth)

    rows = []
    for i in range(args.repeats + 1):
        ttft, wall, ptok = _one(prompt, model, args.gen)
        kind = "cold" if i == 0 else "warm"
        pp = (ptok / ttft) if ttft > 0 else None
        rows.append({"i": i, "kind": kind, "prompt_tokens": ptok,
                     "ttft_s": ttft, "wall_s": wall, "pp_tok_s": pp})
        print(f"  [{i}] {kind:<4} prompt {ptok:>8,}  TTFT {ttft:>8.3f}s  "
              f"wall {wall:>8.3f}s" + (f"  PP {pp:>8.1f}" if pp else ""))

    cold = rows[0]["ttft_s"]
    warm = [r["ttft_s"] for r in rows[1:] if r["ttft_s"] > 0]
    speedup = (cold / (sum(warm) / len(warm))) if warm and cold > 0 else None
    if speedup:
        print(f"  cold {cold:.3f}s -> warm mean {sum(warm)/len(warm):.3f}s "
              f"= {speedup:.2f}x prefill speedup")

    out = {"depth": args.depth, "gen_tokens": args.gen, "rows": rows,
           "cold_ttft_s": cold,
           "warm_mean_ttft_s": (sum(warm) / len(warm)) if warm else None,
           "prefill_speedup": speedup}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
