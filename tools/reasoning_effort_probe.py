#!/usr/bin/env python3
"""reasoning_effort_probe - measure what reasoning_effort actually costs.

The card documents which values the upstream template accepts (`xhigh` default,
`medium`, `low`, `none`; `high` raises). What it has never documented is what
they *cost*, because the first attempt capped output at 4,096 tokens and the
model was still reasoning when the cap hit - so every arm returned `length` and
the comparison measured the cap, not the effort.

This probe fixes the two faults in that attempt: a generous output budget (the
model must be allowed to finish), and `finish_reason` recorded per request so a
truncated arm is visible rather than silently averaged in.

Reported per effort: completion tokens, reasoning-field size, whether the answer
terminated naturally, and wall latency. Deterministic sampling (temperature 0)
so repeats measure the engine, not the sampler.

Usage:
  reasoning_effort_probe.py --base-url http://10.0.0.3:8000 --out FILE
                            [--model qwen38] [--max-tokens 32768]
                            [--repeats 2] [--efforts xhigh,medium,low,none]
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import time
import urllib.error
import urllib.request

# One prompt, reused from the inconclusive attempt so the two are comparable.
# It invites real reasoning and has a checkable artefact (the algorithm's name
# and a code block), rather than a single token that hides effort differences.
PROMPT = ("Write a Python function that finds the longest palindromic "
          "substring in O(n) time using Manacher's algorithm. Explain your "
          "reasoning.")


def call(base: str, model: str, effort: str | None, max_tokens: int,
         timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if effort is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
            status = r.status
    except urllib.error.HTTPError as e:
        return {"http_status": e.code, "error": e.read().decode()[:400],
                "latency_s": round(time.monotonic() - t0, 3)}
    except Exception as e:  # noqa: BLE001 - transport shape is not the subject
        return {"http_status": None, "error": f"{type(e).__name__}: {e}",
                "latency_s": round(time.monotonic() - t0, 3)}

    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = out.get("usage") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    return {
        "http_status": status,
        "latency_s": round(time.monotonic() - t0, 3),
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "mentions_manacher": "manacher" in (content + reasoning).lower(),
        "has_code_block": "```" in content,
        "error": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--efforts", default="xhigh,medium,low,none")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--also-invalid", default="high",
                    help="comma list of values expected to be REJECTED; "
                         "recorded as evidence the template still rejects them")
    args = ap.parse_args()

    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    runs: list[dict] = []

    # Unset control first: what the endpoint does when the client says nothing.
    for i in range(args.repeats):
        r = call(args.base_url, args.model, None, args.max_tokens, args.timeout)
        r.update({"effort": "(unset)", "repeat": i})
        runs.append(r)
        print(f"[probe] (unset) #{i}: {r.get('completion_tokens')} tok "
              f"finish={r.get('finish_reason')} {r['latency_s']}s", flush=True)

    for eff in efforts:
        for i in range(args.repeats):
            r = call(args.base_url, args.model, eff, args.max_tokens, args.timeout)
            r.update({"effort": eff, "repeat": i})
            runs.append(r)
            print(f"[probe] {eff} #{i}: {r.get('completion_tokens')} tok "
                  f"finish={r.get('finish_reason')} {r['latency_s']}s", flush=True)

    invalid = []
    for eff in [e.strip() for e in args.also_invalid.split(",") if e.strip()]:
        r = call(args.base_url, args.model, eff, 64, 120.0)
        r["effort"] = eff
        r["rejected_as_expected"] = r.get("http_status") not in (200,)
        invalid.append(r)
        print(f"[probe] invalid {eff}: http={r.get('http_status')} "
              f"rejected={r['rejected_as_expected']}", flush=True)

    def med(rows: list[dict], key: str):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 1) if vals else None

    by_effort = {}
    for eff in ["(unset)"] + efforts:
        rows = [r for r in runs if r["effort"] == eff]
        ok = [r for r in rows if r.get("http_status") == 200]
        by_effort[eff] = {
            "requests": len(rows),
            "http_200": len(ok),
            "median_completion_tokens": med(ok, "completion_tokens"),
            "median_reasoning_chars": med(ok, "reasoning_chars"),
            "median_content_chars": med(ok, "content_chars"),
            "median_latency_s": med(ok, "latency_s"),
            "finish_reasons": sorted({r.get("finish_reason") for r in ok}),
            "truncated_by_cap": sum(1 for r in ok if r.get("finish_reason") == "length"),
            "answered_correct_algorithm": sum(1 for r in ok if r.get("mentions_manacher")),
            "code_block_present": sum(1 for r in ok if r.get("has_code_block")),
        }

    rep = {
        "schema": "qwen38-reasoning-effort-probe/1",
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base_url,
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "repeats": args.repeats,
        "why_the_budget_is_large": (
            "The first attempt used 4,096 output tokens and every arm returned "
            "finish_reason=length, so it measured the cap. Any arm still showing "
            "truncated_by_cap>0 here is reported, not averaged away."
        ),
        "by_effort": by_effort,
        "invalid_values": invalid,
        "runs": runs,
    }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[probe] -> {args.out}")
    for eff, s in by_effort.items():
        print(f"  {eff:9s} completion={s['median_completion_tokens']} "
              f"reasoning_chars={s['median_reasoning_chars']} "
              f"latency={s['median_latency_s']}s "
              f"truncated={s['truncated_by_cap']} "
              f"correct_algo={s['answered_correct_algorithm']}/{s['http_200']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
