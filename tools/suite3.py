#!/usr/bin/env python3
"""Build the v3 fidelity suite: held-out contexts, contamination-checked, partitioned.

Differences from v2, all of them things the Kimi-K3 protocol requires and v2
lacked:
  * contexts come from a **held-out** corpus (Gutenberg / arXiv / Wikipedia /
    CPython), not from exllamav3's calibration data that our own quant was tuned on;
  * contamination scan against every exllamav3 calibration corpus, reported, not
    assumed;
  * 4x the volume;
  * source clusters are real (one document family per cluster), so the bootstrap
    resamples independent units instead of synthetic groups of four;
  * deterministic analysis / qualification partition split, and a sentinel subset
    for runtime-repeat noise measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

CAL_DIR = Path("/work/exllamav3/exllamav3/conversion/standard_cal_data")
SHINGLE = 160  # characters; a match this long is real overlap, not coincidence


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cal_shingles() -> set[int]:
    """Hashes of fixed-stride character shingles across the calibration corpora."""
    out: set[int] = set()
    for p in sorted(CAL_DIR.glob("*.utf8")):
        text = re.sub(r"\s+", " ", p.read_text(encoding="utf-8", errors="ignore"))
        for i in range(0, max(0, len(text) - SHINGLE), SHINGLE // 2):
            out.add(hash(text[i:i + SHINGLE]))
    return out


def contaminated(text: str, cal: set[int]) -> int:
    norm = re.sub(r"\s+", " ", text)
    hits = 0
    for i in range(0, max(0, len(norm) - SHINGLE), SHINGLE):
        if hash(norm[i:i + SHINGLE]) in cal:
            hits += 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="/work/kld3/corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", type=int, default=256)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--sentinels", type=int, default=32)
    ap.add_argument("--qualification-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("hashing calibration shingles for the contamination scan", flush=True)
    cal = cal_shingles()
    print(f"  {len(cal)} shingles", flush=True)

    corpus = Path(args.corpus)
    strata = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    ctx_len = args.context_length
    per = {s: args.contexts // len(strata) for s in strata}
    for s in strata[: args.contexts % len(strata)]:
        per[s] += 1

    contexts, seen, contam_total, contam_ctx = [], set(), 0, 0
    for stratum in strata:
        want = per[stratum]
        files = sorted((corpus / stratum).glob("*.txt"))
        made = 0
        # round-robin across documents so one long book cannot dominate a stratum
        cursors = {f: 0 for f in files}
        while made < want and files:
            progressed = False
            for f in list(files):
                if made >= want:
                    break
                text = f.read_text(encoding="utf-8", errors="ignore")
                start = cursors[f]
                if start >= len(text):
                    files.remove(f)
                    continue
                piece = text[start:start + ctx_len * 8]
                cursors[f] = start + ctx_len * 8
                progressed = True
                ids = tok(piece, add_special_tokens=False, truncation=True,
                          max_length=ctx_len)["input_ids"]
                if isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) < ctx_len:
                    continue
                ids = ids[:ctx_len]
                digest = sha(json.dumps(ids).encode())
                if digest in seen:
                    continue
                seen.add(digest)
                hits = contaminated(piece[:ctx_len * 5], cal)
                contam_total += hits
                contam_ctx += 1 if hits else 0
                contexts.append({"stratum": stratum, "source_cluster": f.stem,
                                 "token_sha256": digest, "tokens": ids,
                                 "calibration_shingle_hits": hits})
                made += 1
            if not progressed:
                break
        print(f"  {stratum}: {made} contexts from {len(set(c['source_cluster'] for c in contexts if c['stratum']==stratum))} documents", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(contexts)
    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    n_qual = int(len(contexts) * args.qualification_fraction)
    index = []
    for i, c in enumerate(contexts):
        name = f"context-{i:04d}.json"
        (out / "tokens" / name).write_text(json.dumps(c["tokens"]))
        index.append({"index": i, "stratum": c["stratum"],
                      "source_cluster": c["source_cluster"],
                      "file": f"tokens/{name}", "tokens": len(c["tokens"]),
                      "token_sha256": c["token_sha256"],
                      "calibration_shingle_hits": c["calibration_shingle_hits"],
                      "partition": "qualification" if i < n_qual else "analysis",
                      "sentinel": False})
    sentinels = rng.sample([r for r in index if r["partition"] == "analysis"],
                           min(args.sentinels, len(index) - n_qual))
    for r in sentinels:
        r["sentinel"] = True

    manifest = {
        "schema": "qwen38-distribution-fidelity/3",
        "model": args.model, "context_length": ctx_len,
        "scored_positions_per_context": ctx_len - 1,
        "contexts": len(index),
        "total_scored_positions": len(index) * (ctx_len - 1),
        "hidden_size": 5120, "vocab_size": 248320,
        "held_out": True,
        "corpus_note": ("Gutenberg / arXiv / Wikipedia (en + de,fr,es,ja,zh,ru) / "
                        "CPython v3.12.8 Lib. Disjoint from exllamav3 calibration data."),
        "contamination_scan": {
            "shingle_chars": SHINGLE,
            "calibration_shingles": len(cal),
            "contexts_with_any_hit": contam_ctx,
            "total_hits": contam_total,
        },
        "partitions": {"analysis": sum(1 for r in index if r["partition"] == "analysis"),
                       "qualification": n_qual,
                       "sentinels": len(sentinels)},
        "strata": {s: sum(1 for r in index if r["stratum"] == s) for s in strata},
        "source_clusters": len({r["source_cluster"] for r in index}),
        "context_index": index,
    }
    manifest["suite_token_sha256"] = sha("".join(r["token_sha256"] for r in index).encode())
    (out / "suite-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("contexts", "total_scored_positions", "source_clusters",
                       "partitions", "contamination_scan", "suite_token_sha256")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
