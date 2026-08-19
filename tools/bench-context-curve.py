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

Method: SERVER-SIDE metrics. Two earlier methods were both biased.
------------------------------------------------------------------
Attempt 1 - difference two non-streaming requests (`max_tokens=1` vs `1+N`) -
is invalid past a few thousand tokens: at 32k the prefill is ~13 s while 64
tokens of decode is ~0.3 s, so prefill jitter swamps the decode window. It
reported 1,127.9 tok/s on a profile whose ceiling is ~228.

Attempt 2 - client-side SSE streaming, timestamping chunk arrivals - is invalid
in the other direction: Python SSE parsing costs more per token than decode
does, so it reported 36.4 tok/s where the harness measures 215.6. It also
returned zero text chunks at depths >= 32k.

This version reads the ENGINE's own Prometheus histograms around a plain
non-streaming request, so neither prefill nor the client is in the path:
  * TTFT  = delta(vllm:time_to_first_token_seconds_sum) / delta(count)
            -> PP(D) = prompt_tokens / TTFT
  * ITL   = delta(vllm:request_time_per_output_token_seconds_sum) / delta(count)
            -> TG(D) = 1 / ITL
Both are the server's own measurements of the request we just issued.

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


def _metric_pairs() -> dict[str, tuple[float, float]]:
    """Return {metric: (sum, count)} for the timing histograms we use."""
    want = ("vllm:time_to_first_token_seconds",
            "vllm:request_time_per_output_token_seconds")
    out: dict[str, list[float]] = {w: [0.0, 0.0] for w in want}
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=20) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                for w in want:
                    if line.startswith(f"{w}_sum"):
                        out[w][0] = float(line.rsplit(" ", 1)[1])
                    elif line.startswith(f"{w}_count"):
                        out[w][1] = float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return {k: (v[0], v[1]) for k, v in out.items()}


def measure_at_depth(depth_tokens: int, gen_tokens: int, model: str) -> dict | None:
    # "word " tokenises to roughly one token per word for this vocab; the exact
    # count is read back from usage rather than assumed.
    prompt = "word " * depth_tokens
    m0 = _metric_pairs()
    a0, d0 = bl.spec_counters()
    try:
        r, wall = _post({"model": model, "prompt": prompt,
                         "max_tokens": gen_tokens, "temperature": 0})
    except Exception as e:
        print(f"    depth {depth_tokens}: request failed: {type(e).__name__}: {e}")
        return None
    a1, d1 = bl.spec_counters()
    m1 = _metric_pairs()

    actual = r.get("usage", {}).get("prompt_tokens") or depth_tokens
    completed = r.get("usage", {}).get("completion_tokens") or gen_tokens

    def delta(metric: str) -> float | None:
        s0, c0 = m0.get(metric, (0.0, 0.0))
        s1, c1 = m1.get(metric, (0.0, 0.0))
        return (s1 - s0) / (c1 - c0) if (c1 - c0) > 0 else None

    ttft = delta("vllm:time_to_first_token_seconds")
    itl = delta("vllm:request_time_per_output_token_seconds")
    if not ttft or ttft <= 0:
        print(f"    depth {depth_tokens}: no server TTFT delta, skipped")
        return None
    acc = ((a1 - a0) / (d1 - d0)) if (d1 - d0) > 0 else None
    return {
        "requested_depth": depth_tokens,
        "prompt_tokens": actual,
        "completion_tokens": completed,
        "ttft_s": ttft,
        "itl_s": itl,
        "wall_s": wall,
        "pp_tok_s": actual / ttft,
        "tg_tok_s": (1.0 / itl) if itl and itl > 0 else None,
        "acceptance_len": acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="label for this run")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--depths", default="2048,8192,32768,65536,131072,199104,238400")
    ap.add_argument("--gen-tokens", type=int, default=128)
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
            tg = f"{r['tg_tok_s']:>6.1f}" if r["tg_tok_s"] else "   n/a"
            print(f"    depth {r['prompt_tokens']:>7,}  PP {r['pp_tok_s']:>8.1f}  "
                  f"TG {tg}  TTFT {r['ttft_s']:>7.2f}s"
                  + (f"  acc_len {r['acceptance_len']:.3f}"
                     if r["acceptance_len"] is not None else ""))

    out = {"profile": args.profile, "max_model_len": ctx,
           "gen_tokens": args.gen_tokens, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
