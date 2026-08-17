#!/usr/bin/env python3
"""Build the deterministic short-context probe set for the static-YaRN penalty measurement.

The probe set is 4 length buckets x N prompts, cut from a corpus assembled from
this repository's own prose and receipts in a fixed, recorded order.  Windows are
disjoint and are laid out round-robin across buckets so that no bucket is
confounded with a region (or genre) of the corpus.

Token counts are exact: each window's text is trimmed until the model's own fast
tokenizer returns exactly the requested number of ids.  The live server's
/tokenize endpoint re-verifies every count at run time (yarn_probe_run.py).

Usage (inside the container, no GPU needed):
    ggrun.sh python /work/yarn/yarn_probe_build.py \
        --repo /work/yarn/repo --model /models/Qwen3.8-27B-EXL3-K5K6-hydrated \
        --out /work/yarn/probes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

BUCKETS = [512, 2048, 8192, 32768]
PER_FILE_BYTE_CAP = 400_000  # keep one huge receipt from dominating the corpus


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_files(repo: Path) -> list[Path]:
    """Fixed, reproducible file order: docs prose, then cards, then receipts."""
    out: list[Path] = []
    out += sorted((repo / "docs").glob("*.md"))
    out += sorted(repo.glob("*.md"))
    out += sorted((repo / "receipts").glob("*.json"))
    return out


def build_corpus(repo: Path, need_chars: int) -> tuple[str, list[dict]]:
    parts: list[str] = []
    prov: list[dict] = []
    total = 0
    for path in corpus_files(repo):
        raw = path.read_bytes()
        chunk = raw[:PER_FILE_BYTE_CAP]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            text = chunk.decode("utf-8", "ignore")
        if not text.strip():
            continue
        parts.append(text)
        prov.append(
            {
                "path": str(path.relative_to(repo)),
                "file_sha256": sha256_hex(raw),
                "file_bytes": len(raw),
                "bytes_taken": len(chunk),
            }
        )
        total += len(text) + 2
        if total >= need_chars:
            break
    return "\n\n".join(parts), prov


def cut_window(tok, corpus: str, offsets, start_tok: int, n_tok: int) -> tuple[str, int]:
    """Return text whose exact tokenization length is n_tok, and the next free token index."""
    end_tok = start_tok + n_tok
    for _ in range(64):
        end_tok = min(end_tok, len(offsets))
        text = corpus[offsets[start_tok][0] : offsets[end_tok - 1][1]]
        got = len(tok(text, add_special_tokens=False)["input_ids"])
        if got == n_tok:
            return text, start_tok + n_tok
        end_tok += n_tok - got
        if end_tok <= start_tok + 1:
            raise RuntimeError("window collapsed")
    raise RuntimeError(f"could not converge on {n_tok} tokens at {start_tok}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--index-out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    need_tokens = args.per_bucket * sum(BUCKETS)
    # ~4.0 chars/token on this material; ask for 1.6x headroom for the trim loop.
    corpus, prov = build_corpus(Path(args.repo), int(need_tokens * 4.0 * 1.6))
    enc = tok(corpus, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    print(f"corpus: {len(corpus)} chars, {len(offsets)} tokens, {len(prov)} files")
    if len(offsets) < need_tokens + 4096:
        raise SystemExit(f"corpus too small: {len(offsets)} tokens < {need_tokens} needed")

    probes = []
    cursor = 0
    for rnd in range(args.per_bucket):
        for n_tok in BUCKETS:
            text, cursor = cut_window(tok, corpus, offsets, cursor, n_tok)
            pid = f"b{n_tok}-{rnd:02d}"
            probes.append(
                {
                    "id": pid,
                    "bucket": n_tok,
                    "round": rnd,
                    "n_tokens_offline": n_tok,
                    "text_sha256": sha256_hex(text.encode()),
                    "text": text,
                }
            )

    payload = {
        "schema": "yarn-short-context-probes-1",
        "buckets": BUCKETS,
        "per_bucket": args.per_bucket,
        "n_prompts": len(probes),
        "tokenizer": {
            "source": args.model,
            "tokenizer_json_sha256": sha256_hex(
                (Path(args.model) / "tokenizer.json").read_bytes()
            ),
            "add_special_tokens": False,
        },
        "corpus": {
            "assembly": "files concatenated with a blank line between them, in the order listed; "
            f"at most {PER_FILE_BYTE_CAP} bytes taken from each file; windows are disjoint and "
            "allocated round-robin over the four buckets so no bucket is confounded with a region "
            "or genre of the corpus",
            "corpus_chars": len(corpus),
            "corpus_tokens": len(offsets),
            "corpus_sha256": sha256_hex(corpus.encode()),
            "files": prov,
        },
        "probes": probes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False)
    out.write_text(blob)
    digest = sha256_hex(blob.encode())
    print(f"wrote {out} ({len(blob)} bytes) sha256={digest}")

    if args.index_out:
        # The full probe set is 1.6 MB of text copied verbatim out of this repository,
        # so committing it would duplicate the repo into itself.  The committed artefact
        # is this index: it pins the full file's digest, every source file's digest and
        # byte budget, and every prompt's own text digest and exact token count, which is
        # everything needed to regenerate the set and prove the regeneration matches.
        index = {k: v for k, v in payload.items() if k != "probes"}
        index["schema"] = "yarn-short-context-probes-index-1"
        index["full_probe_set"] = {
            "filename": out.name,
            "bytes": len(blob),
            "sha256": digest,
            "regenerate_with": "tools/yarn_probe_build.py --repo <repo> --model <model dir> "
            f"--out probes.json --per-bucket {args.per_bucket}",
            "verify": "the regenerated file's sha256 must equal the value above; per-prompt "
            "text_sha256 values below let a reader localise any mismatch",
        }
        index["probes"] = [
            {
                "id": p["id"],
                "bucket": p["bucket"],
                "round": p["round"],
                "n_tokens": p["n_tokens_offline"],
                "text_sha256": p["text_sha256"],
                "text_chars": len(p["text"]),
                "head_64_chars": p["text"][:64],
                "tail_64_chars": p["text"][-64:],
            }
            for p in payload["probes"]
        ]
        Path(args.index_out).write_text(json.dumps(index, ensure_ascii=False, indent=1))
        print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
