#!/usr/bin/env python3
"""Bit-exactness probe: fixed greedy completions, text captured and hashed.

Eight deterministic prompts sliced from the frozen literary corpus at fixed
byte offsets, 256 greedy output tokens each (temperature 0, ignore_eos), plus
one 2048-token-ish prefill prompt. The concatenated completion text is
SHA-256 hashed; two serving arms are bit-exact on this probe set iff the
digests match.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

FILES = ["pg1342.txt", "pg2701.txt", "pg84.txt", "pg1661.txt",
         "pg2600.txt", "pg345.txt", "pg98.txt", "pg174.txt"]
OFFSET = 20000
SPAN = 6000


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    results = []
    for name in FILES:
        text = (corpus / name).read_text(errors="replace")[OFFSET:OFFSET + SPAN]
        prompt = ("Continue the passage faithfully in the same style.\n\n"
                  + text)
        body = {"model": "m", "prompt": prompt, "max_tokens": args.tokens,
                "temperature": 0, "ignore_eos": True, "stream": False}
        t = time.monotonic()
        r = post(args.url, body)
        el = time.monotonic() - t
        completion = r["choices"][0]["text"]
        results.append({
            "file": name,
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "completion_tokens": r["usage"]["completion_tokens"],
            "elapsed_sec": round(el, 3),
            "completion_sha256": hashlib.sha256(
                completion.encode()).hexdigest(),
            "completion_head": completion[:120],
        })
        print(json.dumps(results[-1]), flush=True)

    concat = hashlib.sha256()
    for row in results:
        concat.update(row["completion_sha256"].encode())
    out = {
        "schema": "exl3-arena-greedy-probe/1",
        "label": args.label,
        "output_tokens_per_request": args.tokens,
        "probe_files": FILES,
        "offset": OFFSET,
        "span": SPAN,
        "results": results,
        "combined_sha256": concat.hexdigest(),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("combined:", out["combined_sha256"])


if __name__ == "__main__":
    main()
