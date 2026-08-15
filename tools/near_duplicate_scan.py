#!/usr/bin/env python3
"""Offset-independent lexical overlap audit for held-out evaluation corpora.

The suite builder's historical fixed-stride character shingles can miss copied text
when the two files start at different offsets. This scanner hashes every normalized
lexical-token n-gram, with one token per Han/Kana character. It reports exact 12-gram runs
and local 5-gram containment in sliding windows; the latter catches lightly edited
near-duplicates but is not a semantic paraphrase detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import unicodedata
from pathlib import Path



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_cjk_character(character: str) -> bool:
    value = ord(character)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0x20000 <= value <= 0x323AF
        or 0x3040 <= value <= 0x30FF
        or 0x31F0 <= value <= 0x31FF
        or 0xFF66 <= value <= 0xFF9D
    )


def word_tokens(text: str) -> list[str]:
    """Unicode words, with one token per Han/Kana character.

    Whitespace-free Chinese and Japanese sentences must still produce shingles;
    Python's ``\\w+`` otherwise collapses a whole sentence into one token.
    """
    tokens: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        if _is_cjk_character(character):
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(character)
        elif character.isalnum() or character == "_" or (
            current and unicodedata.category(character).startswith("M")
        ):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def words(path: Path) -> list[str]:
    return word_tokens(path.read_text(encoding="utf-8", errors="ignore"))


def ngram_hashes(tokens: list[str], n: int) -> list[bytes]:
    if len(tokens) < n:
        return []
    sep = "\x1f"
    return [
        hashlib.blake2b(sep.join(tokens[i:i + n]).encode(), digest_size=16).digest()
        for i in range(len(tokens) - n + 1)
    ]


def longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exact-n", type=int, default=12)
    parser.add_argument("--near-n", type=int, default=5)
    parser.add_argument("--window-words", type=int, default=256)
    parser.add_argument("--stride-words", type=int, default=128)
    args = parser.parse_args()

    calibration = sorted(Path(args.calibration).glob("*.utf8"))
    corpus = sorted(Path(args.corpus).glob("**/*.txt"))
    if not calibration or not corpus:
        raise SystemExit("calibration *.utf8 files and corpus *.txt files are required")
    if min(args.exact_n, args.near_n, args.window_words, args.stride_words) < 1:
        raise SystemExit("n-gram and window sizes must be positive")
    if args.window_words < args.near_n:
        raise SystemExit("--window-words must be at least --near-n")

    exact_cal: set[bytes] = set()
    near_cal: set[bytes] = set()
    calibration_words = 0
    for path in calibration:
        tokens = words(path)
        calibration_words += len(tokens)
        exact_cal.update(ngram_hashes(tokens, args.exact_n))
        near_cal.update(ngram_hashes(tokens, args.near_n))
        print(f"calibration {path.name}: {len(tokens)} words", flush=True)

    documents = []
    all_window_containment: list[float] = []
    for path in corpus:
        tokens = words(path)
        exact = [value in exact_cal for value in ngram_hashes(tokens, args.exact_n)]
        near = [value in near_cal for value in ngram_hashes(tokens, args.near_n)]
        windows = []
        if near:
            width = args.window_words - args.near_n + 1
            starts = list(range(0, max(1, len(tokens) - args.window_words + 1),
                                args.stride_words))
            last = max(0, len(tokens) - args.window_words)
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in starts:
                count = sum(near[start:min(len(near), start + width)])
                denominator = min(width, len(near) - start)
                windows.append(count / denominator if denominator else 0.0)
        all_window_containment.extend(windows)
        longest = longest_true_run(exact)
        record = {
            "file": str(path.relative_to(args.corpus)),
            "sha256": sha256_file(path),
            "words": len(tokens),
            "exact_ngram_hits": sum(exact),
            "longest_exact_run_ngrams": longest,
            "longest_exact_run_words": longest + args.exact_n - 1 if longest else 0,
            "max_near_window_containment": max(windows, default=0.0),
        }
        documents.append(record)
        print(
            f"corpus {record['file']}: exact={record['exact_ngram_hits']} "
            f"longest={record['longest_exact_run_words']} words "
            f"near_max={record['max_near_window_containment']:.3f}",
            flush=True,
        )

    top = sorted(
        documents,
        key=lambda row: (row["max_near_window_containment"],
                         row["longest_exact_run_words"]),
        reverse=True,
    )[:20]
    report = {
        "schema": "qwen38-near-duplicate-audit/2",
        "method": {
            "normalization": "Unicode NFKC and casefold; Unicode words plus one token per Han/Kana character",
            "digest": "blake2b-128",
            "exact_ngram_words": args.exact_n,
            "near_ngram_words": args.near_n,
            "near_window_words": args.window_words,
            "near_stride_words": args.stride_words,
            "limitation": "lexical near-duplicate audit; does not detect semantic paraphrases",
        },
        "inputs": {
            "calibration_files": {p.name: sha256_file(p) for p in calibration},
            "corpus_files": len(corpus),
            "calibration_words": calibration_words,
            "exact_calibration_ngrams": len(exact_cal),
            "near_calibration_ngrams": len(near_cal),
        },
        "summary": {
            "documents": len(documents),
            "documents_with_exact_hit": sum(r["exact_ngram_hits"] > 0 for r in documents),
            "documents_with_exact_run_32_words": sum(
                r["longest_exact_run_words"] >= 32 for r in documents
            ),
            "total_exact_ngram_hits": sum(r["exact_ngram_hits"] for r in documents),
            "longest_exact_run_words": max(
                (r["longest_exact_run_words"] for r in documents), default=0
            ),
            "near_window_containment_mean": statistics.fmean(all_window_containment)
            if all_window_containment else 0.0,
            "near_window_containment_p95": quantile(all_window_containment, 0.95),
            "near_window_containment_p99": quantile(all_window_containment, 0.99),
            "near_window_containment_max": max(all_window_containment, default=0.0),
        },
        "top_documents": top,
        "documents": documents,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(target)
    print("summary " + json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
