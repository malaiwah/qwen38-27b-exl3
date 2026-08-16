#!/usr/bin/env python3
"""One text and one image request against a served recipe, exact-match scored.

This is the per-recipe start/serve gate for the immutable production image
(receipts/production-image.json), not a capability measurement: the fidelity numbers come
from tools/longctx.py, tools/longmm.py, tools/vision_eval.py and tools/bench.py. Keeping it
small means the four model-card recipes can each be proven to start and answer without
spending long-context GPU time four times over.

The image is drawn by the published generator (tools/vision_eval.draw_digits) so the
fixture is byte-identical to the deterministic 30-case suite's digit family, and its
SHA-256 is recorded.

    smoke_client.py --url http://127.0.0.1:8123/v1 --label context --out smoke-context.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from vision_eval import draw_digits  # noqa: E402  published generator, imported unmodified


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def served_models(url: str, timeout: int = 60) -> list[str]:
    with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as response:
        return [entry["id"] for entry in json.loads(response.read()).get("data", [])]


def chat(url: str, content, timeout: int, max_tokens: int = 64) -> tuple[dict, float]:
    payload = {
        "model": "m",
        "max_tokens": max_tokens,
        "temperature": 0,
        # thinking mode would eat the whole token budget before the answer
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    started = time.monotonic()
    result = post(url, payload, timeout)
    return result, round(time.monotonic() - started, 3)


def digits_only(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="OpenAI-compatible base, e.g. .../v1")
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    text_code = str(rng.randrange(10**9, 10**10))
    image_code = "".join(str(rng.randrange(10)) for _ in range(6))
    fixture = draw_digits(image_code)

    text_prompt = (
        "Read the record below and answer with the access code only, digits only, "
        "no other words.\n\n"
        f"BEGIN RECORD\nsite: aiboss\nrole: production runtime smoke\n"
        f"access code: {text_code}\nEND RECORD\n"
    )
    text_result, text_seconds = chat(args.url, text_prompt, args.timeout)
    text_answer = text_result["choices"][0]["message"]["content"].strip()
    text_ok = digits_only(text_answer) == text_code

    image_result, image_seconds = chat(
        args.url,
        [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(fixture).decode()}},
            {"type": "text", "text": "Read the 6-digit number in this image. "
                                     "Answer with the digits only."},
        ],
        args.timeout,
    )
    image_answer = image_result["choices"][0]["message"]["content"].strip()
    image_ok = digits_only(image_answer) == image_code

    report = {
        "schema": "qwen38-image-smoke/1",
        "label": args.label,
        "seed": args.seed,
        "url": args.url,
        "served_model_ids": served_models(args.url),
        "text": {
            "expected": text_code,
            "answer": text_answer[:120],
            "extracted": digits_only(text_answer)[:12],
            "exact": text_ok,
            "prompt_tokens": int(text_result.get("usage", {}).get("prompt_tokens") or 0),
            "completion_tokens": int(
                text_result.get("usage", {}).get("completion_tokens") or 0),
            "wall_sec": text_seconds,
        },
        "image": {
            "expected": image_code,
            "answer": image_answer[:120],
            "extracted": digits_only(image_answer)[:12],
            "exact": image_ok,
            "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
            "fixture_bytes": len(fixture),
            "prompt_tokens": int(image_result.get("usage", {}).get("prompt_tokens") or 0),
            "completion_tokens": int(
                image_result.get("usage", {}).get("completion_tokens") or 0),
            "wall_sec": image_seconds,
        },
    }
    report["passed"] = text_ok and image_ok
    temporary = Path(args.out + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(Path(args.out))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
