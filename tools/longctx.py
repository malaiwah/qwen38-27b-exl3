#!/usr/bin/env python3
"""Generate at a measured long-context length and test exact needle retrieval.

The server's own tokenizer sizes the fully templated completion prompt. A missing
or incompatible tokenization endpoint is fatal: approximate character counts are
not acceptable evidence for a context-length claim.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


def load_filler(corpus: Path, want_chars: int) -> str:
    """Concatenate deterministic held-out text, repeating only when necessary."""
    files = sorted(corpus.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no *.txt under {corpus}: cannot build a long prompt")
    parts: list[str] = []
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts.append(text)
        total += len(text)
        if total >= want_chars:
            break
    joined = "\n".join(parts)
    if not joined:
        raise SystemExit(f"corpus under {corpus} contains no text")
    while len(joined) < want_chars:
        joined = joined + "\n" + joined
    return joined[:want_chars]


def post(base: str, path: str, payload: dict, timeout: int = 3600) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def api_root(base: str) -> str:
    root = base.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def count_tokens(base: str, text: str) -> int:
    result = post(api_root(base), "/tokenize", {"model": "m", "prompt": text}, timeout=600)
    count = result.get("count")
    if count is None:
        tokens = result.get("tokens")
        count = len(tokens) if isinstance(tokens, list) else None
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"tokenizer returned no positive count: {str(result)[:300]}")
    return count


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--tokens", type=int, default=261800)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-token-error", type=int, default=128)
    parser.add_argument("--require-pass", action="store_true",
                        help="require exact retrieval at the observed requested length")
    args = parser.parse_args()
    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    if not 0 <= args.depth <= 1:
        raise SystemExit("--depth must be between 0 and 1")
    if args.max_token_error < 0:
        raise SystemExit("--max-token-error must be non-negative")

    rng = random.Random(args.seed)
    code = str(rng.randrange(10**9, 10**10))
    needle = (
        "\n\nIMPORTANT RECORD: the Gilded Gnosis maintenance access code is "
        f"{code}. Remember this exact number.\n\n"
    )

    def build_prompt(filler: str) -> str:
        cut = int(len(filler) * args.depth)
        return (
            filler[:cut]
            + needle
            + filler[cut:]
            + "\n\nQuestion: what is the Gilded Gnosis maintenance access code recorded above? "
            "Answer with the number only.\nAnswer:"
        )

    def template(prompt: str) -> str:
        # The chat endpoint truncates long text through this VLM's multimodal processor.
        # Apply the model's text template and use /v1/completions instead.
        return (
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    filler_chars = max(1, int(args.tokens * 3.6))
    measured_tokens = 0
    templated = ""
    for _ in range(8):
        filler = load_filler(Path(args.corpus), filler_chars)
        templated = template(build_prompt(filler))
        measured_tokens = count_tokens(args.url, templated)
        error = measured_tokens - args.tokens
        if abs(error) <= args.max_token_error:
            break
        filler_chars = max(1, int(filler_chars * args.tokens / measured_tokens))
    else:
        raise RuntimeError(
            f"prompt sizing did not converge: requested {args.tokens}, measured {measured_tokens}"
        )

    started = time.monotonic()
    result = post(
        args.url,
        "/completions",
        {
            "model": "m",
            "prompt": templated,
            "max_tokens": 48,
            "temperature": 0,
            "stream": False,
        },
    )
    elapsed = time.monotonic() - started
    text = result["choices"][0]["text"]
    usage = result.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int):
        raise RuntimeError("completion response lacks observed usage.prompt_tokens")
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = int(usage.get("completion_tokens") or 0)
    answer = text.strip()
    retrieved = answer == code
    length_preserved = abs(prompt_tokens - args.tokens) <= args.max_token_error
    passed = retrieved and length_preserved
    report = {
        "schema": "qwen38-longctx/3",
        "requested_tokens": args.tokens,
        "measured_tokens_before_submit": measured_tokens,
        "prompt_tokens": prompt_tokens,
        "token_count_delta": prompt_tokens - args.tokens,
        "length_preserved": length_preserved,
        "completion_tokens": completion_tokens,
        "needle_depth_fraction": args.depth,
        "needle_code": code,
        "retrieved_exact": retrieved,
        "passed": passed,
        "answer": answer[:120],
        "wall_sec": round(elapsed, 3),
        "prefill_tok_s_including_decode": round(prompt_tokens / elapsed, 1) if elapsed else None,
    }
    atomic_write_json(Path(args.out), report)
    print(json.dumps(report), flush=True)
    return 0 if passed or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
