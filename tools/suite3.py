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
from pathlib import Path
from near_duplicate_scan import ngram_hashes, word_tokens

CAL_DIR = Path("/work/exllamav3/exllamav3/conversion/standard_cal_data")
SHINGLE_WORDS = 12


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()



def cal_shingles() -> set[bytes]:
    """Every normalized lexical-token n-gram in every calibration corpus."""
    out: set[bytes] = set()
    for path in sorted(CAL_DIR.glob("*.utf8")):
        tokens = word_tokens(path.read_text(encoding="utf-8", errors="ignore"))
        out.update(ngram_hashes(tokens, SHINGLE_WORDS))
    return out


def contaminated(text: str, cal: set[bytes]) -> int:
    """Count overlap independently of file offset or context-window alignment."""
    return sum(value in cal for value in ngram_hashes(word_tokens(text), SHINGLE_WORDS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="execute model-repository Python (unsafe; off by default)")
    ap.add_argument("--corpus", default="/work/kld3/corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", type=int, default=256)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--sentinels", type=int, default=32)
    ap.add_argument("--qualification-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--allow-shortfall", action="store_true",
                    help="accept fewer contexts than requested in a stratum instead of "
                         "failing; without it an under-filled stratum aborts the build")
    ap.add_argument("--exclude-suite", default=None,
                    help="path to an earlier suite-manifest.json; documents and token "
                         "hashes recorded there are refused, which is how a later suite "
                         "is made source-disjoint from the one used to pick a recipe")
    args = ap.parse_args()
    if args.contexts <= 0 or args.context_length <= 1:
        raise SystemExit("--contexts must be positive and --context-length must exceed one")
    if args.sentinels < 0:
        raise SystemExit("--sentinels may not be negative")
    if not 0 <= args.qualification_fraction < 1:
        raise SystemExit("--qualification-fraction must be in [0, 1)")
    model_root = Path(args.model)
    if not model_root.is_dir():
        raise SystemExit(
            "--model must be a reviewed local snapshot directory; download an immutable "
            "revision before building the suite"
        )
    tokenizer_file = model_root / "tokenizer.json"
    config_file = model_root / "config.json"
    if not tokenizer_file.is_file() or not config_file.is_file():
        raise SystemExit("local model snapshot must contain tokenizer.json and config.json")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    print("hashing every calibration lexical-token n-gram", flush=True)
    cal = cal_shingles()
    print(f"  {len(cal)} {SHINGLE_WORDS}-token shingles", flush=True)

    corpus = Path(args.corpus)
    strata = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    if not strata:
        raise SystemExit(f"no stratum directories under {corpus}")
    ctx_len = args.context_length
    per = {s: args.contexts // len(strata) for s in strata}
    for s in strata[: args.contexts % len(strata)]:
        per[s] += 1

    # Document identity travels with the suite: without it a reader cannot tell which
    # source a context came from, only which category, and cannot check that a later
    # suite is source-disjoint from this one.
    doc_identity: dict[str, dict] = {}
    for stratum in strata:
        for f in sorted((corpus / stratum).glob("*.txt")):
            raw = f.read_bytes()
            if f.stem in doc_identity:
                prior = doc_identity[f.stem]
                raise SystemExit(
                    f"duplicate source stem {f.stem!r}: {prior['stratum']}/{prior['file']} "
                    f"and {stratum}/{f.name}; source_cluster identities must be unique"
                )
            doc_identity[f.stem] = {"stratum": stratum, "file": f.name,
                                    "bytes": len(raw), "sha256": sha(raw)}


    excluded_docs: set[str] = set()
    excluded_tokens: set[str] = set()
    if args.exclude_suite:
        prior = json.loads(Path(args.exclude_suite).read_text())
        for name, meta in prior.get("documents", []):
            excluded_docs.add(name)
            if isinstance(meta, dict) and meta.get("sha256"):
                excluded_docs.add(meta["sha256"])
        excluded_docs.update(r["source_cluster"] for r in prior.get("context_index", []))
        excluded_tokens.update(r["token_sha256"] for r in prior.get("context_index", []))
        for name in list(doc_identity):
            if name in excluded_docs or doc_identity[name]["sha256"] in excluded_docs:
                del doc_identity[name]
        print(f"excluding {len(excluded_docs)} prior document identifiers and "
              f"{len(excluded_tokens)} prior context hashes", flush=True)
    # Seeding `seen` with the prior suite's context hashes means an excluded context
    # cannot re-enter even if the same passage is reachable through a different document.
    contexts, seen, contam_total, contam_ctx = [], set(excluded_tokens), 0, 0
    shortfall = {}
    for stratum in strata:
        want = per[stratum]
        files = [f for f in sorted((corpus / stratum).glob("*.txt"))
                 if f.stem in doc_identity]
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
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) < ctx_len:
                    continue
                ids = ids[:ctx_len]
                digest = sha(json.dumps(ids).encode())
                if digest in seen:
                    continue
                seen.add(digest)
                hits = contaminated(piece, cal)
                contam_total += hits
                contam_ctx += 1 if hits else 0
                contexts.append({"stratum": stratum, "source_cluster": f.stem,
                                 "token_sha256": digest, "tokens": ids,
                                 "calibration_shingle_hits": hits})
                made += 1
            if not progressed:
                break
        if made < want:
            shortfall[stratum] = (made, want)
        print(f"  {stratum}: {made}/{want} contexts from "
              f"{len(set(c['source_cluster'] for c in contexts if c['stratum']==stratum))} "
              "documents", flush=True)
    # Under-filling silently is how the v3 suite ended up claiming nine Wikipedia
    # languages while holding German and Russian only. Refuse instead, unless the caller
    # accepts the shortfall explicitly.
    if shortfall and not args.allow_shortfall:
        detail = ", ".join(f"{s}: {got}/{want}" for s, (got, want) in sorted(shortfall.items()))
        raise SystemExit(
            f"stratum shortfall ({detail}); fetch more sources or pass --allow-shortfall"
        )

    rng = random.Random(args.seed)
    # Partition by SOURCE CLUSTER, not by context. Shuffling contexts and slicing put
    # every cluster on both sides of the split: measured on the v3 suite, all 27
    # qualification clusters also appeared in analysis, so the "held-out" partition
    # shared documents with the partition used to pick the recipe and could not serve as
    # a post-selection test. Whole clusters move together here, and the split is
    # stratified by stratum so neither side loses a domain.
    by_stratum: dict[str, dict[str, list[dict]]] = {}
    for c in contexts:
        by_stratum.setdefault(c["stratum"], {}).setdefault(c["source_cluster"], []).append(c)
    qual_clusters: set[str] = set()
    for stratum, clusters in sorted(by_stratum.items()):
        names = sorted(clusters)
        rng.shuffle(names)
        stratum_contexts = sum(len(v) for v in clusters.values())
        target = stratum_contexts * args.qualification_fraction
        taken = 0
        # names[:-1] keeps at least one cluster of every stratum on the analysis side,
        # so neither partition silently loses a domain.
        for name in names[:-1]:
            if taken >= target:
                break
            qual_clusters.add(name)
            taken += len(clusters[name])
    ordered = list(contexts)
    rng.shuffle(ordered)
    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    index = []
    for i, c in enumerate(ordered):
        name = f"context-{i:04d}.json"
        (out / "tokens" / name).write_text(json.dumps(c["tokens"]))
        index.append({"index": i, "stratum": c["stratum"],
                      "source_cluster": c["source_cluster"],
                      "file": f"tokens/{name}", "tokens": len(c["tokens"]),
                      "token_sha256": c["token_sha256"],
                      "calibration_shingle_hits": c["calibration_shingle_hits"],
                      "partition": ("qualification" if c["source_cluster"] in qual_clusters
                                    else "analysis"),
                      "sentinel": False})
    n_qual = sum(1 for r in index if r["partition"] == "qualification")
    analysis_clusters = {r["source_cluster"] for r in index if r["partition"] == "analysis"}
    leak = analysis_clusters & qual_clusters
    if leak:
        raise SystemExit(f"partition leak: clusters on both sides: {sorted(leak)}")
    sentinels = rng.sample([r for r in index if r["partition"] == "analysis"],
                           min(args.sentinels, len(index) - n_qual))
    for r in sentinels:
        r["sentinel"] = True

    manifest = {
        "schema": "qwen38-distribution-fidelity/5",
        "model": args.model,
        "model_identity": {
            "tokenizer_sha256": sha(tokenizer_file.read_bytes()),
            "config_sha256": sha(config_file.read_bytes()),
            "trust_remote_code": args.trust_remote_code,
        },
        "context_length": ctx_len,
        "scored_positions_per_context": ctx_len - 1,
        "contexts": len(index),
        "total_scored_positions": len(index) * (ctx_len - 1),
        "hidden_size": 5120, "vocab_size": 248320,
        "corpus_note": ("Gutenberg / arXiv / Wikipedia (en + de,fr,es,ja,zh,ru) / "
                        "CPython v3.12.8 Lib; calibration overlap is measured below."),
        "contamination_scan": {
            "shingle_words": SHINGLE_WORDS,
            "stride_words": 1,
            "digest": "blake2b-128",
            "normalization": "Unicode NFKC, casefold, every Unicode word token",
            "calibration_shingles": len(cal),
            "contexts_with_any_hit": contam_ctx,
            "total_hits": contam_total,
        },
        "partitions": {"analysis": sum(1 for r in index if r["partition"] == "analysis"),
                       "qualification": n_qual,
                       "sentinels": len(sentinels)},
        "strata": {s: sum(1 for r in index if r["stratum"] == s) for s in strata},
        "source_clusters": len({r["source_cluster"] for r in index}),
        # Recorded explicitly so a reader can check group-disjointness without
        # recomputing it, which is exactly the property the v3 suite lacked.
        "cluster_partition": {
            "analysis": sorted({r["source_cluster"] for r in index
                                if r["partition"] == "analysis"}),
            "qualification": sorted(qual_clusters),
            "overlap": sorted({r["source_cluster"] for r in index
                               if r["partition"] == "analysis"} & qual_clusters),
        },
        "documents": sorted(
            {r["source_cluster"]: doc_identity.get(r["source_cluster"], {})
             for r in index}.items()
        ),
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
