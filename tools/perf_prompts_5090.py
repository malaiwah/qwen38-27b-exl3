#!/usr/bin/env python3
"""Freeze the exact token-id prompts the lever sweep replays for every configuration.

Runs inside the serving image (CPU only, no GPU device, no server) so the prompts exist
and are digested before any configuration starts. Token ids rather than text mean the
prompt length is exact instead of estimated, and identical bytes across configurations
mean an A/B difference cannot come from a different prompt.

    perf_prompts_5090.py --model /models/ctx-repo/snapshots/<rev> \
        --corpus /corpus/literary --out-dir /work/prompts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def corpus_text(corpus: Path, want_chars: int) -> str:
    """Deterministic concatenation: sorted files, no sampling, no repetition."""
    files = sorted(corpus.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no *.txt under {corpus}")
    parts: list[str] = []
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts.append(text)
        total += len(text)
        if total >= want_chars:
            break
    joined = "\n".join(parts)
    if len(joined) < want_chars:
        raise SystemExit(f"corpus under {corpus} holds {len(joined)} chars, need {want_chars}")
    return joined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="tokenizer source (the served repo)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[2048, 6144])
    parser.add_argument("--decode-prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-prompts", type=int, default=8, help="one per stream at C8")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # in-image dependency

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # 6 chars/token is a safe floor for English prose; assert the real count afterwards.
    need_tokens = max(args.prefill_lengths) + args.decode_prompts * args.decode_prompt_tokens
    ids = tokenizer(corpus_text(Path(args.corpus), need_tokens * 12), add_special_tokens=False)[
        "input_ids"
    ]
    if len(ids) < need_tokens:
        raise SystemExit(f"tokenized {len(ids)} ids, need {need_tokens}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    def emit(name: str, token_ids: list[int]) -> None:
        payload = json.dumps(token_ids)
        path = out_dir / f"{name}.json"
        path.write_text(payload + "\n")
        manifest[path.name] = {
            "tokens": len(token_ids),
            "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
        }

    for n in args.prefill_lengths:
        emit(f"prefill-{n}", ids[:n])
    # Distinct decode prompts, taken from a region no prefill prompt uses, so a concurrent
    # run cannot share a prefix between streams.
    cursor = max(args.prefill_lengths)
    for i in range(args.decode_prompts):
        emit(f"decode-{i:02d}", ids[cursor : cursor + args.decode_prompt_tokens])
        cursor += args.decode_prompt_tokens

    report = {
        "schema": "qwen38-perf-prompts-5090/1",
        "tokenizer_source": args.model,
        "tokenizer_class": type(tokenizer).__name__,
        "corpus": args.corpus,
        "corpus_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(args.corpus).rglob("*.txt"))
        },
        "prompts": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prompts": len(manifest), "out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
