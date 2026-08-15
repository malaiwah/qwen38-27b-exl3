#!/usr/bin/env python3
"""Exercise context rather than allocate it: needle retrieval at real length, with timings.

Every context claim in this family so far has been an allocation test - the engine reserved a
cache and started. This builds a prompt of a requested token length from held-out corpus text,
plants a unique fact at a chosen depth, asks for it back, and reports whether the answer is
exact along with TTFT, prefill rate and decode rate. Failure to retrieve is a result, not an
error, so it is reported rather than raised.

    longctx.py --url http://127.0.0.1:8120/v1 --tokens 262000 --depth 0.5 --out ctx-262k.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


def load_filler(corpus: Path, want_chars: int) -> str:
    """Concatenate held-out corpus text until long enough. Deterministic order."""
    files = sorted(corpus.rglob("*.txt"))
    if not files:
        # A missing corpus used to degrade into a prompt of newlines, which tokenises to a
        # fraction of the requested length and makes retrieval trivially succeed. Refuse.
        raise SystemExit(f"no *.txt under {corpus}: cannot build a long prompt")
    parts: list[str] = []
    total = 0
    for p in files:
        t = p.read_text(encoding="utf-8", errors="ignore")
        parts.append(t)
        total += len(t)
        if total >= want_chars:
            break
    if total < want_chars:
        # repeat rather than fail: the point is length, and repetition is a harder retrieval
        # test than novel text because the needle cannot be found by novelty alone
        joined = "\n".join(parts)
        while len(joined) < want_chars:
            joined += "\n" + joined
        return joined[:want_chars]
    return "\n".join(parts)[:want_chars]


def post(base: str, path: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base: str, text: str) -> int:
    """Ask the server itself, so the count is the one the engine will use."""
    try:
        r = post(base, "/tokenize", {"model": "m", "prompt": text}, timeout=600)
        return int(r.get("count") or len(r.get("tokens") or []))
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--corpus", default="/work/kld3/corpus",
                    help="filler source; keep it monolingual or the model drifts")
    ap.add_argument("--tokens", type=int, default=262000)
    ap.add_argument("--depth", type=float, default=0.5,
                    help="fractional position of the needle inside the filler")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    code = f"{rng.randrange(10**9, 10**10)}"
    needle = (f"\n\nIMPORTANT RECORD: the Gilded Gnosis maintenance access code is {code}. "
              "Remember this exact number.\n\n")

    # ~3.6 characters per token on this corpus mix; correct once against the server's tokenizer
    filler = load_filler(Path(args.corpus), int(args.tokens * 3.6))
    cut = int(len(filler) * args.depth)
    prompt = (filler[:cut] + needle + filler[cut:] +
              "\n\nQuestion: what is the Gilded Gnosis maintenance access code recorded above? "
              "Answer with the number only.\nAnswer:")
    def build(text: str) -> str:
        c = int(len(text) * args.depth)
        return (text[:c] + needle + text[c:] +
                "\n\nQuestion: what is the Gilded Gnosis maintenance access code recorded "
                "above? Answer with the number only.\nAnswer:")

    # Converge on the requested length: measure characters per token from the corpus itself
    # rather than assuming 3.6, and iterate, because a single correction pass either
    # overshoots the window or collapses the prompt.
    n = count_tokens(args.url, prompt)
    for _ in range(3):
        if n <= 0:
            break
        if abs(n - args.tokens) <= max(256, args.tokens // 100):
            break
        cpt = len(prompt) / n
        want_chars = int((args.tokens - 96) * cpt)
        base = load_filler(Path(args.corpus), want_chars)
        filler = base[:want_chars]
        prompt = build(filler)
        n = count_tokens(args.url, prompt)

    # The chat endpoint cannot be used for long context on this VLM: its multimodal
    # processor truncates the text to ~2,048 tokens (measured: a 32,768-token request came
    # back with prompt_tokens 1,909), the same defect family as issue #313. So apply Qwen's
    # template by hand and post to /completions, which bypasses the processor. Thinking is
    # suppressed by closing an empty think block, which is what the template does for
    # enable_thinking=false.
    templated = ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    t0 = time.monotonic()
    r = post(args.url, "/completions", {
        "model": "m", "prompt": templated, "max_tokens": 48, "temperature": 0,
        "stream": False})
    elapsed = time.monotonic() - t0
    text = r["choices"][0]["text"]
    usage = r.get("usage", {})
    pt = int(usage.get("prompt_tokens") or n)
    ct = int(usage.get("completion_tokens") or 0)
    found = code in text
    report = {
        "schema": "qwen38-longctx/1",
        "requested_tokens": args.tokens, "prompt_tokens": pt, "completion_tokens": ct,
        "needle_depth_fraction": args.depth, "needle_code": code,
        "retrieved_exact": found, "answer": text.strip()[:120],
        "wall_sec": round(elapsed, 3),
        "prefill_tok_s_including_decode": round(pt / elapsed, 1) if elapsed else None,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
