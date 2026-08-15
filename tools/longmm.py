#!/usr/bin/env python3
"""Exercise long-text retrieval and image understanding in one served request."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import random
import time
from pathlib import Path

from longctx import atomic_write_json, count_tokens, load_filler, post


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-tokens", type=int, default=230_000)
    parser.add_argument("--minimum-prompt-tokens", type=int, default=220_000)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.text_tokens <= 0 or args.minimum_prompt_tokens <= 0:
        raise SystemExit("token limits must be positive")
    if not 0 <= args.depth <= 1:
        raise SystemExit("--depth must be between 0 and 1")

    image_path = Path(args.image)
    image_bytes = image_path.read_bytes()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    code = str(random.Random(args.seed).randrange(10**9, 10**10))
    needle = (
        "\n\nIMPORTANT RECORD: the Gilded Gnosis maintenance access code is "
        f"{code}. Remember this exact number.\n\n"
    )

    def build_text(filler: str) -> str:
        cut = int(len(filler) * args.depth)
        return (
            filler[:cut]
            + needle
            + filler[cut:]
            + "\n\nReturn the maintenance code and the two large image colours from left "
            "to right. Answer exactly: CODE | color, color"
        )

    filler_chars = max(1, int(args.text_tokens * 3.6))
    measured_text_tokens = 0
    text = ""
    for _ in range(8):
        text = build_text(load_filler(Path(args.corpus), filler_chars))
        templated = (
            "<|im_start|>user\n"
            + text
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        measured_text_tokens = count_tokens(args.url, templated)
        error = measured_text_tokens - args.text_tokens
        if abs(error) <= 128:
            break
        filler_chars = max(1, int(filler_chars * args.text_tokens / measured_text_tokens))
    else:
        raise RuntimeError(
            f"text prompt sizing did not converge: requested {args.text_tokens}, "
            f"measured {measured_text_tokens}"
        )

    started = time.monotonic()
    result = post(
        args.url,
        "/chat/completions",
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.monotonic() - started
    answer = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    length_ok = prompt_tokens >= args.minimum_prompt_tokens
    parts = answer.split(" | ")
    code_ok = len(parts) == 2 and parts[0] == code
    colours_ok = len(parts) == 2 and parts[1] == "red, blue"
    answer_exact = code_ok and colours_ok
    report = {
        "schema": "qwen38-longmm/1",
        "requested_text_tokens": args.text_tokens,
        "measured_text_tokens_before_submit": measured_text_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "minimum_prompt_tokens": args.minimum_prompt_tokens,
        "length_preserved": length_ok,
        "needle_depth_fraction": args.depth,
        "needle_code": code,
        "retrieved_exact": code_ok,
        "image": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "bytes": len(image_bytes),
        },
        "colours_exact": colours_ok,
        "answer_exact": answer_exact,
        "answer": answer[:240],
        "wall_sec": round(elapsed, 3),
        "prompt_tok_s_including_decode": round(prompt_tokens / elapsed, 1) if elapsed else None,
        "passed": length_ok and answer_exact,
    }
    atomic_write_json(Path(args.out), report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
