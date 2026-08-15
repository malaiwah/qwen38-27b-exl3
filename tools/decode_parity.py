#!/usr/bin/env python3
"""Matched eager-vs-graph decode parity, following the RTX 5090 qualification protocol.

The parity check published with PR #314 measured the wrong path: it compared hidden
states from a single **prefill** forward, and `FULL_DECODE_ONLY` captures no prefill
graph, so the captured code was never exercised. This drives real decode steps through
an OpenAI-compatible endpoint and compares chosen tokens and their logprobs.

    decode_parity.py --url-a http://127.0.0.1:8090/v1 --url-b http://127.0.0.1:8091/v1 \\
        --label-a eager --label-b graph --out parity.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

PROMPTS = [
    "Ottawa is a city that", "Explain what a loopback interface is:",
    "The ampere is defined as", "Write a Python function that reverses a string:",
    "The capital of Japan is", "In linear algebra, an eigenvector is",
    "Describe the difference between TCP and UDP:", "A tokenizer converts",
    "The second law of thermodynamics states", "Explain quantization of neural networks:",
    "def fibonacci(n):", "The Rhine flows through",
    "A CUDA graph reduces", "Photosynthesis converts", "In Rust, ownership means",
    "The HTTP status code 429 means", "A B-tree is used for",
    "Summarise the causes of the French Revolution:", "The speed of light is",
    "In statistics, a p-value is", "Explain the difference between latency and throughput:",
    "A hash table resolves collisions by", "The mitochondrion is",
    "Write a SQL query that counts rows per group:", "Kubernetes pods are",
    "The Fourier transform maps", "In chess, the en passant rule",
    "A neural network layer normalises", "The Great Barrier Reef is",
    "Explain what an inode stores:", "Bayes theorem relates", "A semaphore differs from a mutex because",
]


def post(base: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(base: str, tokens: int, seed: int) -> list[dict]:
    out = []
    for p in PROMPTS:
        r = post(base, {"model": "m", "prompt": p, "max_tokens": tokens,
                        "temperature": 0, "seed": seed, "logprobs": 5,
                        "ignore_eos": True})
        c = r["choices"][0]
        lp = c.get("logprobs") or {}
        out.append({"prompt": p, "text": c["text"],
                    "tokens": lp.get("tokens") or [],
                    "token_logprobs": lp.get("token_logprobs") or []})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    # Collect and compare are separate because the two servers cannot co-reside:
    # each wants 0.85 of a 96 GB GPU, so they are run one at a time.
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--tokens", type=int, default=32)
    c.add_argument("--seed", type=int, default=314159)
    c.add_argument("--repeat", action="store_true",
                   help="run the corpus twice to prove within-mode determinism")
    c.add_argument("--out", required=True)
    k = sub.add_parser("compare")
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "collect":
        first = run(args.url, args.tokens, args.seed)
        payload = {"label": args.label, "tokens": args.tokens, "seed": args.seed,
                   "runs": [first]}
        if args.repeat:
            second = run(args.url, args.tokens, args.seed)
            payload["runs"].append(second)
            payload["self_repeat_exact"] = sum(
                1 for x, y in zip(first, second) if x["text"] == y["text"])
        Path(args.out).write_text(json.dumps(payload))
        print("collected " + json.dumps({k2: payload[k2] for k2 in payload
                                         if k2 != "runs"}), flush=True)
        return 0

    pa = json.loads(Path(args.a).read_text())
    pb = json.loads(Path(args.b).read_text())
    a, b = pa["runs"][0], pb["runs"][0]
    args.label_a, args.label_b = pa["label"], pb["label"]
    args.tokens, args.seed = pa["tokens"], pa["seed"]
    exact, diffs, deltas = 0, [], []
    for ia, ib in zip(a, b):
        if ia["text"] == ib["text"]:
            exact += 1
        else:
            first = next((i for i, (x, y) in enumerate(zip(ia["tokens"], ib["tokens"]))
                          if x != y), None)
            diffs.append({"prompt": ia["prompt"], "first_divergence_index": first,
                          "a_token": ia["tokens"][first] if first is not None else None,
                          "b_token": ib["tokens"][first] if first is not None else None,
                          "a_logprob": ia["token_logprobs"][first] if first is not None else None,
                          "b_logprob": ib["token_logprobs"][first] if first is not None else None})
        for x, y, lx, ly in zip(ia["tokens"], ib["tokens"],
                                ia["token_logprobs"], ib["token_logprobs"]):
            if x != y:
                break
            if lx is not None and ly is not None:
                deltas.append(abs(lx - ly))

    report = {
        "schema": "qwen38-decode-parity/1",
        "a": args.label_a, "b": args.label_b,
        "prompts": len(PROMPTS), "tokens_per_prompt": args.tokens, "seed": args.seed,
        "exact_sequence_matches": exact,
        "matched_prefix_tokens": len(deltas),
        "chosen_logprob_abs_delta_mean": statistics.fmean(deltas) if deltas else 0.0,
        "chosen_logprob_abs_delta_max": max(deltas) if deltas else 0.0,
        "divergences": diffs,
    }
    report["self_repeat_exact_a"] = pa.get("self_repeat_exact")
    report["self_repeat_exact_b"] = pb.get("self_repeat_exact")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("parity " + json.dumps({k: report[k] for k in (
        "exact_sequence_matches", "prompts", "matched_prefix_tokens",
        "chosen_logprob_abs_delta_mean", "chosen_logprob_abs_delta_max")}), flush=True)
    for d in report["divergences"][:5]:
        print("  diverge:", json.dumps(d), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
