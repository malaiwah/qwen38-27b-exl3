#!/usr/bin/env python3
"""Measure PP and TG across the whole served context range, per profile.

Why
---
Every published number for this stack is measured at near-zero context depth: PP
on a fixed 2,051-token prompt, TG on a short prompt. Nothing says how either rate
behaves at 128k or 238k, which is where the long-context claims actually live.
This model is a hybrid (16 full-attention + 48 gated-delta-net layers), so the
degradation curve is not the textbook quadratic one and is worth measuring rather
than assuming.

Method (decomposition, not conflation)
--------------------------------------
For each target depth D:
  1. request with `max_tokens=1`   -> t_prefill(D)     => PP(D) = tokens_in / t1
  2. request with `max_tokens=1+N` -> t_total(D)       => TG(D) = N / (t2 - t1)

Both requests use the same prompt, so the subtraction removes prefill, queueing
and HTTP overhead from the decode figure. Prefix caching is off in all our
profiles, so request 2 genuinely re-prefills - which is what makes the
subtraction valid.

Depth is reported as the **measured** `usage.prompt_tokens`, never the requested
word count, because the two differ by tokenisation.

Acceptance is sampled around the decode call from the engine's own speculative
counters, so MTP behaviour at depth is recorded alongside the rates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402

BASE = "http://localhost:8000"


def _post(payload: dict, timeout: int = 1800) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def measure_at_depth(depth_tokens: int, gen_tokens: int, model: str) -> dict | None:
    # "word " tokenises to roughly one token per word for this vocab; the exact
    # count is read back from usage rather than assumed.
    prompt = "word " * depth_tokens
    try:
        r1, t1 = _post({"model": model, "prompt": prompt, "max_tokens": 1,
                        "temperature": 0})
    except Exception as e:
        print(f"    depth {depth_tokens}: prefill request failed: {type(e).__name__}")
        return None
    actual = r1.get("usage", {}).get("prompt_tokens") or depth_tokens

    a0, d0 = bl.spec_counters()
    try:
        _r2, t2 = _post({"model": model, "prompt": prompt,
                         "max_tokens": 1 + gen_tokens, "temperature": 0})
    except Exception as e:
        print(f"    depth {depth_tokens}: decode request failed: {type(e).__name__}")
        return None
    a1, d1 = bl.spec_counters()

    dt = t2 - t1
    if dt <= 0:
        print(f"    depth {depth_tokens}: non-positive decode window, skipped")
        return None
    acc = ((a1 - a0) / (d1 - d0)) if (d1 - d0) > 0 else None
    return {
        "requested_depth": depth_tokens,
        "prompt_tokens": actual,
        "t_prefill_s": t1,
        "t_total_s": t2,
        "pp_tok_s": actual / t1,
        "tg_tok_s": gen_tokens / dt,
        "acceptance": acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="label for this run")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--depths", default="2048,8192,32768,65536,131072,199104,238400")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx = bl.max_model_len()
    print(f"profile={args.profile}  served max_model_len={ctx:,}")
    bl.warmup()

    rows = []
    for d in [int(x) for x in args.depths.split(",")]:
        # leave room for the generated tokens plus a margin
        if d > ctx - args.gen_tokens - 256:
            print(f"    depth {d:,} exceeds served context, skipped")
            continue
        r = measure_at_depth(d, args.gen_tokens, args.model)
        if r:
            rows.append(r)
            print(f"    depth {r['prompt_tokens']:>7,}  PP {r['pp_tok_s']:>8.1f}  "
                  f"TG {r['tg_tok_s']:>6.1f}"
                  + (f"  acc {r['acceptance']:.3f}" if r["acceptance"] is not None else ""))

    out = {"profile": args.profile, "max_model_len": ctx,
           "gen_tokens": args.gen_tokens, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
