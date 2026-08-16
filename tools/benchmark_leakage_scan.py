#!/usr/bin/env python3
"""Calibration 12-gram contamination scan of public benchmark item texts.

Same normalized all-position method as the corpus scans: every normalized
lexical-token 12-gram (Unicode words plus one token per Han/Kana character,
NFKC + casefold) of every calibration corpus is hashed, then every benchmark
item text is checked for membership. The algorithm is imported unchanged from
tools/near_duplicate_scan.py (the scanner the v3/v4/v5 corpus audits used);
this file only feeds it benchmark items instead of corpus documents.

Benchmarks scanned:
- the pinned 70-item MMLU-Pro draw (receipts/public-capability-suite-mmlupro-70.json),
  both the bare item text and the full rendered five-shot prompt;
- HumanEval+ at a pinned revision (prompt, canonical solution and test per item);
- GPQA Diamond at a pinned revision (question plus all four answer options);
- the 40-task deterministic retention suite, regenerated from its seed and
  bound to the published suite_sha256 before scanning.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from near_duplicate_scan import longest_true_run, ngram_hashes, sha256_file, word_tokens
from task_retention import canonical_bytes, sha256_bytes

SHINGLE_WORDS = 12
DEFAULT_CAL_DIR = Path("/var/tmp/work/exllamav3/exllamav3/conversion/standard_cal_data")
RETENTION_SEED = 20260815
RETENTION_PER_KIND = 10
RETENTION_SUITE_SHA256 = "e1eccaaf98071d7dc25cb1b10da08459d0de8a3061e76790c490f80072642030"


def cal_shingles(cal_dir: Path) -> tuple[dict[bytes, str], list[dict]]:
    """Every normalized lexical-token 12-gram in every calibration corpus,
    mapped to the corpus file that contributed it (first writer wins)."""
    out: dict[bytes, str] = {}
    sources = []
    paths = sorted(cal_dir.glob("*.utf8"))
    if not paths:
        raise SystemExit(f"no calibration corpora found under {cal_dir}")
    for path in paths:
        raw = path.read_bytes()
        tokens = word_tokens(raw.decode("utf-8", errors="ignore"))
        hashes = ngram_hashes(tokens, SHINGLE_WORDS)
        for value in hashes:
            out.setdefault(value, path.name)
        sources.append({
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "lexical_tokens": len(tokens),
            "shingles": len(hashes),
        })
    return out, sources


def scan_items(items: list[tuple[str, dict[str, str]]],
               cal: dict[bytes, str]) -> dict:
    """items: (item_id, {field_name: text}). Returns per-benchmark tallies."""
    scanned = 0
    shingles_scanned = 0
    hit_items = []
    for item_id, fields in items:
        scanned += 1
        item_hits = 0
        item_fields = {}
        for field, text in fields.items():
            tokens = word_tokens(text)
            hashes = ngram_hashes(tokens, SHINGLE_WORDS)
            shingles_scanned += len(hashes)
            flags = [value in cal for value in hashes]
            hits = sum(flags)
            if hits:
                run = longest_true_run(flags)
                matched = sorted({cal[value] for value in hashes if value in cal})
                matched_shingles = [tokens[i:i + SHINGLE_WORDS]
                                    for i, flag in enumerate(flags) if flag]
                numeric = sum(all(token.isdigit() for token in shingle)
                              for shingle in matched_shingles)
                samples = []
                for shingle in matched_shingles:
                    joined = " ".join(shingle)
                    if joined not in samples:
                        samples.append(joined)
                    if len(samples) == 3:
                        break
                item_fields[field] = {
                    "hits": hits,
                    "hits_all_numeric_tokens": numeric,
                    "longest_exact_run_words": run + SHINGLE_WORDS - 1,
                    "calibration_sources": matched,
                    "sample_matched_shingles": samples,
                }
                item_hits += hits
        if item_hits:
            hit_items.append({"id": item_id, "hits": item_hits, "fields": item_fields})
    return {
        "items_scanned": scanned,
        "shingles_scanned": shingles_scanned,
        "items_with_hits": len(hit_items),
        "total_hits": sum(h["hits"] for h in hit_items),
        "hit_items": hit_items,
    }


def load_mmlupro(receipt_path: Path) -> tuple[dict, list[tuple[str, dict[str, str]]]]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("task_count") != 70 or len(receipt["tasks"]) != 70:
        raise SystemExit("MMLU-Pro receipt does not carry the pinned 70-item draw")
    items = []
    for task in receipt["tasks"]:
        item_text = task["question"] + "\n" + "\n".join(task["options"])
        items.append((f"question_id={task['question_id']}", {
            "item_text": item_text,
            "rendered_prompt": task["system_prompt"] + "\n" + task["user_prompt"],
        }))
    source = {
        "kind": "local pinned receipt",
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "suite_sha256": receipt["suite_sha256"],
        "upstream": receipt["source"],
        "fields_scanned": ["item_text (question + options)",
                           "rendered_prompt (system + five-shot user prompt)"],
    }
    return source, items


def load_humanevalplus(parquet_path: Path, repo: str, revision: str,
                       filename: str) -> tuple[dict, list[tuple[str, dict[str, str]]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    items = [(row["task_id"], {
        "prompt": row["prompt"],
        "canonical_solution": row["canonical_solution"],
        "test": row["test"],
    }) for row in rows]
    source = {
        "kind": "huggingface dataset",
        "dataset": repo,
        "revision": revision,
        "filename": filename,
        "file_sha256": sha256_file(parquet_path),
        "fields_scanned": ["prompt", "canonical_solution", "test"],
    }
    return source, items


def load_gpqa_diamond(csv_path: Path, repo: str, revision: str,
                      filename: str) -> tuple[dict, list[tuple[str, dict[str, str]]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    items = [(row["Record ID"], {
        "item_text": "\n".join([
            row["Question"], row["Correct Answer"], row["Incorrect Answer 1"],
            row["Incorrect Answer 2"], row["Incorrect Answer 3"],
        ]),
    }) for row in rows]
    source = {
        "kind": "huggingface dataset (gated)",
        "dataset": repo,
        "revision": revision,
        "filename": filename,
        "file_sha256": sha256_file(csv_path),
        "fields_scanned": ["item_text (question + correct answer + 3 incorrect answers)"],
    }
    return source, items


def load_retention_suite() -> tuple[dict, list[tuple[str, dict[str, str]]]]:
    import random

    from task_retention import gen_arithmetic, gen_code, gen_format, gen_tool

    rng = random.Random(RETENTION_SEED)
    tasks = (gen_arithmetic(rng, RETENTION_PER_KIND) + gen_code(rng, RETENTION_PER_KIND)
             + gen_format(rng, RETENTION_PER_KIND) + gen_tool(rng, RETENTION_PER_KIND))
    for index, task in enumerate(tasks):
        task["index"] = index
    suite_sha256 = sha256_bytes(canonical_bytes(tasks))
    if suite_sha256 != RETENTION_SUITE_SHA256:
        raise SystemExit(
            f"regenerated retention suite hash {suite_sha256} does not match the "
            f"published suite_sha256 {RETENTION_SUITE_SHA256}"
        )
    items = [(f"{task['kind']}/{task['index']}", {"prompt": task["prompt"]})
             for task in tasks]
    source = {
        "kind": "deterministic regeneration",
        "generator": "tools/task_retention.py",
        "seed": RETENTION_SEED,
        "per_kind": RETENTION_PER_KIND,
        "suite_sha256": suite_sha256,
        "suite_sha256_matches_published_runs": True,
        "fields_scanned": ["prompt"],
    }
    return source, items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-dir", type=Path, default=DEFAULT_CAL_DIR)
    parser.add_argument("--mmlupro-receipt", type=Path,
                        default=Path("receipts/public-capability-suite-mmlupro-70.json"))
    parser.add_argument("--humanevalplus-parquet", type=Path, required=True)
    parser.add_argument("--humanevalplus-revision", required=True)
    parser.add_argument("--gpqa-csv", type=Path, required=True)
    parser.add_argument("--gpqa-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    print("hashing every calibration lexical-token 12-gram", flush=True)
    cal, calibration_sources = cal_shingles(args.cal_dir)
    print(f"  {len(cal)} distinct {SHINGLE_WORDS}-token shingles", flush=True)

    scanner_path = Path(__file__).resolve().parent / "near_duplicate_scan.py"
    benchmarks = {}
    loaders = {
        "mmlu_pro_70": lambda: load_mmlupro(args.mmlupro_receipt),
        "humanevalplus": lambda: load_humanevalplus(
            args.humanevalplus_parquet, "evalplus/humanevalplus",
            args.humanevalplus_revision, args.humanevalplus_parquet.name),
        "gpqa_diamond": lambda: load_gpqa_diamond(
            args.gpqa_csv, "Idavidrein/gpqa", args.gpqa_revision, args.gpqa_csv.name),
        "task_retention_40": load_retention_suite,
    }
    for name, loader in loaders.items():
        source, items = loader()
        result = scan_items(items, cal)
        if result["items_with_hits"]:
            hit_fields = sorted({field for item in result["hit_items"]
                                 for field in item["fields"]})
            numeric = sum(v["hits_all_numeric_tokens"]
                          for item in result["hit_items"]
                          for v in item["fields"].values())
            verdict = (
                f"{result['items_with_hits']}/{result['items_scanned']} items share at least "
                f"one exact normalized {SHINGLE_WORDS}-token n-gram with the conversion "
                f"calibration corpora ({result['total_hits']} matched shingles, "
                f"{numeric} of them consisting solely of numeric tokens, confined to "
                f"field(s): {', '.join(hit_fields)})"
            )
        else:
            verdict = (
                f"all {result['items_scanned']} items scanned "
                f"({result['shingles_scanned']} {SHINGLE_WORDS}-token shingles); zero exact "
                f"normalized {SHINGLE_WORDS}-token n-gram overlaps with the conversion "
                f"calibration corpora"
            )
        benchmarks[name] = {"source": source, **result, "verdict": verdict}
        print(f"{name}: items={result['items_scanned']} "
              f"shingles={result['shingles_scanned']} "
              f"items_with_hits={result['items_with_hits']} "
              f"total_hits={result['total_hits']}", flush=True)

    receipt = {
        "schema": "qwen38-benchmark-leakage-scan/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": (
            "hash every normalized lexical-token 12-gram (Unicode words plus one token per "
            "Han/Kana character, NFKC + casefold, blake2b-128) of every conversion "
            "calibration corpus, then test every benchmark item text shingle for membership; "
            "offset-independent, all positions"
        ),
        "limitation": "lexical overlap audit; does not detect semantic paraphrases",
        "scanner": {
            "algorithm_file": "tools/near_duplicate_scan.py",
            "algorithm_sha256": sha256_file(scanner_path),
            "driver_file": "tools/benchmark_leakage_scan.py",
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "shingle_words": SHINGLE_WORDS,
            "digest": "blake2b-128",
        },
        "calibration": {
            "dir": str(args.cal_dir),
            "files": calibration_sources,
            "distinct_shingles": len(cal),
        },
        "benchmarks": benchmarks,
    }
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=1) + "\n")
    tmp.replace(args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
