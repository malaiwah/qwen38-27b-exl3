#!/usr/bin/env python3
"""Long-context fidelity proxy: multi-needle retrieval at 100k / 195k tokens.

Criteria 3/4 are certified at 2,047 positions; serving advertises 238,400. Our
own measurement shows attention error COMPOUNDS with position (+46 % additivity
failure, receipts/selfattn-fp4-additivity-failure-2026-08-19.md), so short-window
KLD does not certify long-window fidelity. Full long-context KLD is not runnable
(the suite is 2,047-position), so this is the honest proxy: distinct factual
needles planted at controlled depths, exact-match retrieval at temperature 0.

Not a KLD substitute - a needle miss is a coarse failure, a needle hit is not
proof of distributional fidelity. It upgrades "a 200k prompt returns HTTP 200"
to "a 200k prompt is actually read end to end".

Usage: longctx-needles.py [--tokens 100000,195000] [--out receipts/...json]
"""
import argparse
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
MODEL = "Qwen3.8-27B"

NEEDLES = [
    ("the access code for the harbour gate", "7391-KESTREL"),
    ("the name of the lighthouse keeper's cat", "Barnaby"),
    ("the volume of the northern cistern in litres", "48250"),
    ("the colour of the survey team's third flag", "vermilion"),
    ("the registration number of the supply vessel", "VX-2287"),
    ("the number of steps in the western stairwell", "413"),
    ("the password for the archive room", "juniper-elm-9",),
    ("the departure time of the last ferry", "21:47"),
]
DEPTHS = [0.05, 0.18, 0.32, 0.45, 0.58, 0.72, 0.86, 0.97]

FILLER = (
    "The maintenance log for that week records routine inspections of the "
    "pumps, valves and gauges along the seawall, with no anomalies beyond "
    "ordinary wear. Crews rotated on the usual schedule and the weather "
    "remained within seasonal norms for the district. "
)


def post(payload, timeout=1800):
    req = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def approx_tokens(s):
    return len(s) // 4  # conservative for this filler


def build_doc(target_tokens):
    reps = max(1, target_tokens * 4 // len(FILLER))
    body = [FILLER] * reps
    n = len(body)
    placed = []
    for (desc, val), depth in zip(NEEDLES, DEPTHS):
        idx = min(n - 1, int(depth * n))
        body[idx] = (f"For the record, {desc} is {val}. " + FILLER)
        placed.append((desc, val, depth))
    return "".join(body), placed


def run_level(target_tokens):
    doc, placed = build_doc(target_tokens)
    results = []
    for desc, val, depth in placed:
        prompt = (doc + f"\n\nQuestion: What is {desc}? "
                        "Answer with only the exact value.\nAnswer:")
        t0 = time.perf_counter()
        try:
            r = post({"model": MODEL, "prompt": prompt, "max_tokens": 24,
                      "temperature": 0.0})
            txt = r["choices"][0]["text"]
            ptoks = r["usage"]["prompt_tokens"]
            hit = val.lower() in txt.lower()
        except Exception as e:
            txt, ptoks, hit = f"<{type(e).__name__}>", 0, False
        dt = time.perf_counter() - t0
        results.append({"depth": depth, "needle": desc, "expected": val,
                        "got": txt.strip()[:48], "hit": hit,
                        "prompt_tokens": ptoks, "wall_s": round(dt, 1)})
        print(f"    depth {depth:4.0%}  {'HIT ' if hit else 'MISS'}  "
              f"{desc[:38]:40} -> {txt.strip()[:32]!r}  "
              f"({ptoks:,} tok, {dt:5.1f}s)")
    hits = sum(r["hit"] for r in results)
    print(f"  == {target_tokens:,}-token level: {hits}/{len(results)} retrieved ==")
    return {"target_tokens": target_tokens, "hits": hits,
            "total": len(results), "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="100000,195000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        urllib.request.urlopen(BASE + "/health", timeout=10)
    except Exception as e:
        print(f"server unreachable: {e}", file=sys.stderr)
        return 2
    # control: same needle protocol at short context proves the harness itself
    print("  -- control (2k tokens) --")
    levels = [run_level(2000)]
    for t in (int(x) for x in args.tokens.split(",")):
        print(f"  -- {t:,} tokens --")
        levels.append(run_level(t))
    if args.out:
        json.dump({"levels": levels, "model": MODEL,
                   "needles": len(NEEDLES)}, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    ctrl = levels[0]
    if ctrl["hits"] < ctrl["total"]:
        print("  WARNING: control level missed needles - harness suspect, "
              "long-context misses not interpretable")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
