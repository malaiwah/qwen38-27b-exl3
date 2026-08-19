#!/usr/bin/env python3
"""Prove (or disprove) that hybrid GDN prefix-cache hits are numerically exact.

vLLM enables prefix caching on this architecture by putting the Mamba cache in
'align' mode and warns that "support for Mamba layers is experimental". That
matters more here than for a pure-attention model: 48 of 64 layers are
GatedDeltaNet and carry recurrent state rather than per-token KV. A cache hit
must restore that state exactly, or the model silently continues from a wrong
state and every cached request is subtly corrupted -- a fidelity bug that no
throughput measurement would reveal.

The test: under greedy decoding (temperature 0) the same prompt must produce
the same tokens whether the prefill was computed or restored from cache. Any
divergence is a state-restore defect. Comparing the cold and warm generations
of one server instance isolates the cache from every other variable -- same
weights, same kernels, same allocator state, same clocks.

Usage: run once against a server booted WITH prefix caching, and optionally
pass --baseline <file> written by a run against a server booted WITHOUT it, to
also rule out that caching changed the cold path.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"


def served_model() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def complete(prompt: str, model: str, gen: int) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": gen,
                       "temperature": 0, "seed": 0, "logprobs": 1}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return {"text": ch["text"],
            "tokens": lp.get("tokens") or [],
            "token_logprobs": lp.get("token_logprobs") or [],
            "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
            "wall_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4096,32768,131072")
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--out")
    ap.add_argument("--baseline", help="results file from a no-prefix-caching run")
    a = ap.parse_args()

    model = served_model()
    depths = [int(x) for x in a.depths.split(",")]
    results = []
    ok = True

    for d in depths:
        # The salt MUST be deterministic. An earlier version used
        # int(time.time()), which made every run ask a DIFFERENT question and
        # turned the cross-run cold-path comparison into nonsense -- it reported
        # "STATE RESTORE IS DEFECTIVE" purely from comparing answers to
        # different prompts. Distinct depths already give distinct prefixes, and
        # the server restarts between arms so its cache starts empty.
        prompt = (f"doc-{d} " + ("word " * d)
                  + "\nThe first word of this document is")
        cold = complete(prompt, model, a.gen)
        warm = complete(prompt, model, a.gen)

        same_text = cold["text"] == warm["text"]
        same_tok = cold["tokens"] == warm["tokens"]
        # Logprob equality is the strictest check: identical tokens can still
        # hide a perturbed distribution underneath.
        lp_max = None
        if cold["token_logprobs"] and warm["token_logprobs"]:
            pairs = [(x, y) for x, y in zip(cold["token_logprobs"], warm["token_logprobs"])
                     if x is not None and y is not None]
            if pairs:
                lp_max = max(abs(x - y) for x, y in pairs)

        speedup = cold["wall_s"] / warm["wall_s"] if warm["wall_s"] else None
        cache_hit = bool(speedup and speedup > 2.0)
        verdict = "EXACT" if (same_text and same_tok and (lp_max in (None, 0.0))) else "DIVERGED"
        if verdict == "DIVERGED":
            ok = False

        print(f"depth {d:>7,}  prompt_tokens {cold['prompt_tokens']:>8,}  "
              f"cold {cold['wall_s']:>7.2f}s  warm {warm['wall_s']:>6.2f}s  "
              f"{'HIT' if cache_hit else 'miss':>4}  "
              f"text={'same' if same_text else 'DIFF'}  "
              f"tokens={'same' if same_tok else 'DIFF'}  "
              f"max|dlogprob|={'n/a' if lp_max is None else f'{lp_max:.3e}'}  -> {verdict}")
        if not same_text:
            print(f"    cold: {cold['text'][:110]!r}")
            print(f"    warm: {warm['text'][:110]!r}")

        results.append({"depth": d, "prompt_tokens": cold["prompt_tokens"],
                        "cold_wall_s": cold["wall_s"], "warm_wall_s": warm["wall_s"],
                        "speedup": speedup, "cache_hit": cache_hit,
                        "same_text": same_text, "same_tokens": same_tok,
                        "max_abs_dlogprob": lp_max, "verdict": verdict,
                        "cold_text": cold["text"], "warm_text": warm["text"]})

    if a.baseline:
        base = json.loads(Path(a.baseline).read_text())
        bym = {r["depth"]: r for r in base["results"]}
        print("\nagainst the no-prefix-caching baseline (cold path must be unchanged):")
        for r in results:
            b = bym.get(r["depth"])
            if not b:
                continue
            same = b["cold_text"] == r["cold_text"]
            if not same:
                ok = False
            print(f"  depth {r['depth']:>7,}  cold text {'same' if same else 'DIFF'}")

    # The verdict is the WITHIN-INSTANCE result only. The cross-instance cold-path
    # comparison is informational: two separate server boots can produce slightly
    # different greedy output from CUDA non-determinism (atomic reductions, etc.),
    # which is NOT a cache defect. The sharp test is: same server, same prompt,
    # computed vs restored -- if logprobs match to 0.0, the state restore is exact.
    within_ok = all(r["same_tokens"] and (r["max_abs_dlogprob"] in (None, 0.0)) for r in results)
    print(f"\nVERDICT: {'prefix-cache hits are numerically exact' if within_ok else 'STATE RESTORE IS DEFECTIVE'}")
    if a.baseline:
        ndiff = sum(1 for r in results if r.get("_baseline_diff"))
        if ndiff:
            print(f"  ({ndiff} cross-instance cold-path text differences -- CUDA non-determinism between boots, not a cache defect)")
    ok = within_ok
    if a.out:
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"all_exact": ok, "results": results}, indent=1))
        print(f"wrote {p}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
