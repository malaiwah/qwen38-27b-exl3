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



def cal_shingles() -> tuple[set[bytes], list[dict]]:
    """Every normalized lexical-token n-gram in every calibration corpus."""
    out: set[bytes] = set()
    sources = []
    paths = sorted(CAL_DIR.glob("*.utf8"))
    if not paths:
        raise SystemExit(f"no calibration corpora found under {CAL_DIR}")
    for path in paths:
        raw = path.read_bytes()
        tokens = word_tokens(raw.decode("utf-8", errors="ignore"))
        hashes = ngram_hashes(tokens, SHINGLE_WORDS)
        out.update(hashes)
        sources.append({"path": str(path), "bytes": len(raw), "sha256": sha(raw),
                        "lexical_tokens": len(tokens), "shingles": len(hashes)})
    return out, sources


def contaminated(text: str, cal: set[bytes]) -> int:
    """Count overlap independently of file offset or context-window alignment."""
    return sum(value in cal for value in ngram_hashes(word_tokens(text), SHINGLE_WORDS))


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_prior_suite(value: str, option: str) -> dict:
    """Load and strictly validate a suite used as an exclusion boundary."""
    path = Path(value)
    try:
        raw = path.read_bytes()
        prior = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{option} cannot load {path}: {exc}") from exc
    if not isinstance(prior, dict):
        raise SystemExit(f"{option} manifest must be a JSON object: {path}")

    schema = prior.get("schema")
    rows = prior.get("context_index")
    declared_contexts = prior.get("contexts")
    suite_digest = prior.get("suite_token_sha256")
    if not isinstance(schema, str) or not schema:
        raise SystemExit(f"{option} manifest has no valid schema identity: {path}")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{option} manifest has no non-empty context_index: {path}")
    if type(declared_contexts) is not int or declared_contexts != len(rows):
        raise SystemExit(
            f"{option} manifest contexts does not equal context_index length: {path}"
        )
    if not _sha256(suite_digest):
        raise SystemExit(f"{option} manifest has no valid suite_token_sha256: {path}")

    token_hashes: list[str] = []
    source_clusters: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"{option} context_index[{position}] is not an object: {path}")
        if row.get("index") != position:
            raise SystemExit(
                f"{option} context_index[{position}] has invalid index {row.get('index')!r}: "
                f"{path}"
            )
        digest = row.get("token_sha256")
        cluster = row.get("source_cluster")
        if not _sha256(digest):
            raise SystemExit(
                f"{option} context_index[{position}] has invalid token_sha256: {path}"
            )
        if not isinstance(cluster, str) or not cluster:
            raise SystemExit(
                f"{option} context_index[{position}] has invalid source_cluster: {path}"
            )
        token_hashes.append(digest)
        source_clusters.add(cluster)
    if len(set(token_hashes)) != len(token_hashes):
        raise SystemExit(f"{option} manifest contains duplicate context token hashes: {path}")
    computed_suite_digest = sha("".join(token_hashes).encode())
    if computed_suite_digest != suite_digest:
        raise SystemExit(
            f"{option} suite_token_sha256 does not match its ordered context hashes: {path}"
        )

    document_rows = prior.get("documents")
    if not isinstance(document_rows, list):
        raise SystemExit(f"{option} manifest has no valid documents list: {path}")
    documents: dict[str, dict] = {}
    for position, entry in enumerate(document_rows):
        if not isinstance(entry, list) or len(entry) != 2:
            raise SystemExit(f"{option} documents[{position}] is not a [name, metadata] pair")
        name, metadata = entry
        if not isinstance(name, str) or not name or name in documents:
            raise SystemExit(
                f"{option} documents[{position}] has an invalid or duplicate name: {path}"
            )
        if not isinstance(metadata, dict) or not _sha256(metadata.get("sha256")):
            raise SystemExit(
                f"{option} documents[{position}] has no valid source sha256: {path}"
            )
        documents[name] = metadata
    missing_documents = source_clusters - documents.keys()
    if missing_documents:
        raise SystemExit(
            f"{option} context sources missing from documents: {sorted(missing_documents)}"
        )

    identity = {
        "path": str(path.resolve()),
        "manifest_sha256": sha(raw),
        "identity": {
            "schema": schema,
            "suite_token_sha256": suite_digest,
            "contexts": declared_contexts,
            "context_length": prior.get("context_length"),
            "model": prior.get("model"),
            "model_identity": prior.get("model_identity"),
        },
    }
    return {
        "identity": identity,
        "token_hashes": set(token_hashes),
        "documents": documents,
        "source_clusters": source_clusters,
    }


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
                         "failing; without it the exact requested count is mandatory")
    ap.add_argument("--exclude-suite", default=None,
                    help="earlier suite-manifest.json for whole-source disjointness: "
                         "exclude its source documents and exact context token hashes")
    ap.add_argument("--exclude-suite-tokens", default=None,
                    help="earlier suite-manifest.json for token-only exclusion: exclude "
                         "exact prior context token hashes while retaining and potentially "
                         "reusing its source documents for later windows")
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
    if not getattr(tok, "is_fast", False):
        raise SystemExit(
            "exact non-overlapping window advance requires a fast tokenizer with "
            "offset mapping support"
        )

    print("hashing every calibration lexical-token n-gram", flush=True)
    cal, calibration_sources = cal_shingles()
    print(f"  {len(cal)} {SHINGLE_WORDS}-token shingles", flush=True)

    corpus = Path(args.corpus)
    strata = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    if not strata:
        raise SystemExit(f"no stratum directories under {corpus}")
    ctx_len = args.context_length
    per = {s: args.contexts // len(strata) for s in strata}
    for s in strata[: args.contexts % len(strata)]:
        per[s] += 1

    # Scan every document before selection. A single all-position normalized 12-token
    # calibration match makes the entire document ineligible.
    all_doc_identity: dict[str, dict] = {}
    document_texts: dict[str, str] = {}
    document_calibration_hits: dict[str, int] = {}
    document_shingles_scanned = 0
    for stratum in strata:
        for f in sorted((corpus / stratum).glob("*.txt")):
            raw = f.read_bytes()
            if f.stem in all_doc_identity:
                prior = all_doc_identity[f.stem]
                raise SystemExit(
                    f"duplicate source stem {f.stem!r}: {prior['stratum']}/{prior['file']} "
                    f"and {stratum}/{f.name}; source_cluster identities must be unique"
                )
            text = raw.decode("utf-8", errors="ignore")
            words = word_tokens(text)
            hashes = ngram_hashes(words, SHINGLE_WORDS)
            hits = sum(value in cal for value in hashes)
            document_shingles_scanned += len(hashes)
            all_doc_identity[f.stem] = {
                "stratum": stratum,
                "file": f.name,
                "bytes": len(raw),
                "sha256": sha(raw),
                "lexical_tokens": len(words),
                "lexical_shingles": len(hashes),
            }
            document_texts[f.stem] = text
            document_calibration_hits[f.stem] = hits
            del raw, text, words, hashes

    whole_prior = (
        load_prior_suite(args.exclude_suite, "--exclude-suite")
        if args.exclude_suite else None
    )
    token_only_prior = (
        load_prior_suite(args.exclude_suite_tokens, "--exclude-suite-tokens")
        if args.exclude_suite_tokens else None
    )
    whole_prior_tokens = whole_prior["token_hashes"] if whole_prior else set()
    token_only_prior_tokens = (
        token_only_prior["token_hashes"] if token_only_prior else set()
    )
    duplicate_prior_hashes = whole_prior_tokens & token_only_prior_tokens
    if duplicate_prior_hashes:
        raise SystemExit(
            "duplicate context token hashes across --exclude-suite and "
            f"--exclude-suite-tokens ({len(duplicate_prior_hashes)} duplicates)"
        )
    excluded_tokens = whole_prior_tokens | token_only_prior_tokens
    whole_names = set(whole_prior["documents"]) if whole_prior else set()
    whole_hashes = (
        {metadata["sha256"] for metadata in whole_prior["documents"].values()}
        if whole_prior else set()
    )

    excluded_document_records = []
    eligible_doc_identity: dict[str, dict] = {}
    calibration_excluded: set[str] = set()
    whole_source_excluded: set[str] = set()
    for name, metadata in sorted(all_doc_identity.items()):
        reasons = []
        hits = document_calibration_hits[name]
        if hits:
            reasons.append("calibration_12_token_overlap")
            calibration_excluded.add(name)
        if name in whole_names or metadata["sha256"] in whole_hashes:
            reasons.append("prior_suite_whole_source")
            whole_source_excluded.add(name)
        if reasons:
            excluded_document_records.append({
                "source_cluster": name,
                **metadata,
                "calibration_shingle_hits": hits,
                "reasons": reasons,
            })
        else:
            eligible_doc_identity[name] = metadata
    if whole_prior:
        print(f"excluding {len(whole_source_excluded)} prior source documents and "
              f"{len(whole_prior_tokens)} prior context hashes", flush=True)
    if token_only_prior:
        print(f"excluding {len(token_only_prior_tokens)} prior context hashes only; "
              "source documents remain eligible", flush=True)
    print(f"pre-excluded {len(calibration_excluded)} documents with calibration overlap",
          flush=True)
    document_texts = {
        name: document_texts[name] for name in eligible_doc_identity
    }

    # Seeding `seen` with prior context hashes prevents an exact context from re-entering
    # while token-only exclusion deliberately leaves clean later source windows available.
    contexts, seen, contam_total, contam_ctx = [], set(excluded_tokens), 0, 0
    prior_token_rejections = {"whole_source": 0, "token_only": 0}
    current_duplicate_rejections = 0
    decoded_overlap_rejections = 0
    decoded_overlap_hits = 0
    candidate_windows_tokenized = 0
    candidate_source_characters_consumed = 0
    offset_probe_retries = 0
    incomplete_document_tails = 0
    shortfall = {}
    for stratum in strata:
        want = per[stratum]
        files = [f for f in sorted((corpus / stratum).glob("*.txt"))
                 if f.stem in eligible_doc_identity]
        made = 0
        # Round-robin across documents so one long book cannot dominate a stratum.
        # Each source file was decoded once during the document scan. Offset mappings
        # advance the in-memory cursor only through the final emitted token, rather than
        # discarding the unused suffix of an oversized character probe.
        cursors = {f: 0 for f in files}
        probe_chars = {f: max(ctx_len * 4, 1) for f in files}
        source_window_indexes = {f: 0 for f in files}
        while made < want and files:
            progressed = False
            for f in list(files):
                if made >= want:
                    break
                text = document_texts[f.stem]
                start = cursors[f]
                if start >= len(text):
                    files.remove(f)
                    continue
                progressed = True
                remaining = len(text) - start
                piece_chars = min(probe_chars[f], remaining)
                ids = None
                offsets = None
                while True:
                    piece = text[start:start + piece_chars]
                    encoded = tok(
                        piece,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=ctx_len,
                        return_offsets_mapping=True,
                    )
                    candidate_ids = encoded["input_ids"]
                    candidate_offsets = encoded["offset_mapping"]
                    if candidate_ids and isinstance(candidate_ids[0], list):
                        candidate_ids = candidate_ids[0]
                        candidate_offsets = candidate_offsets[0]
                    if len(candidate_ids) >= ctx_len:
                        ids = candidate_ids[:ctx_len]
                        offsets = candidate_offsets[:ctx_len]
                        break
                    if piece_chars == remaining:
                        cursors[f] = len(text)
                        files.remove(f)
                        incomplete_document_tails += 1
                        break
                    piece_chars = min(piece_chars * 2, remaining)
                    offset_probe_retries += 1
                if ids is None or offsets is None:
                    continue
                if len(offsets) != ctx_len:
                    raise SystemExit(
                        f"tokenizer returned {len(offsets)} offsets for {len(ids)} tokens"
                    )
                final_offset = offsets[-1]
                if (
                    not isinstance(final_offset, (list, tuple))
                    or len(final_offset) != 2
                    or type(final_offset[1]) is not int
                    or not 0 < final_offset[1] <= len(piece)
                ):
                    raise SystemExit(
                        f"tokenizer returned invalid final offset {final_offset!r} "
                        f"for source {f.stem}"
                    )
                consumed = final_offset[1]
                end = start + consumed
                cursors[f] = end
                probe_chars[f] = consumed + max(64, consumed // 16)
                source_window_index = source_window_indexes[f]
                source_window_indexes[f] += 1
                candidate_windows_tokenized += 1
                candidate_source_characters_consumed += consumed
                digest = sha(json.dumps(ids).encode())
                if digest in seen:
                    if digest in token_only_prior_tokens:
                        prior_token_rejections["token_only"] += 1
                    elif digest in whole_prior_tokens:
                        prior_token_rejections["whole_source"] += 1
                    else:
                        current_duplicate_rejections += 1
                    continue
                decoded = tok.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                hits = contaminated(decoded, cal)
                if hits:
                    decoded_overlap_rejections += 1
                    decoded_overlap_hits += hits
                    continue
                seen.add(digest)
                contexts.append({
                    "stratum": stratum,
                    "source_cluster": f.stem,
                    "source_window_index": source_window_index,
                    "source_char_start": start,
                    "source_char_end": end,
                    "token_sha256": digest,
                    "tokens": ids,
                    "calibration_shingle_hits": 0,
                })
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
    if not args.allow_shortfall and len(contexts) != args.contexts:
        raise SystemExit(
            f"suite underfill: emitted {len(contexts)} of exactly {args.contexts} "
            "requested contexts"
        )
    if any(context["calibration_shingle_hits"] for context in contexts):
        raise SystemExit("internal invariant failed: emitted a contaminated context")

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
                      "source_window_index": c["source_window_index"],
                      "source_char_start": c["source_char_start"],
                      "source_char_end": c["source_char_end"],
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

    token_prior_document_hashes = (
        {metadata["sha256"] for metadata in token_only_prior["documents"].values()}
        if token_only_prior else set()
    )
    reusable_sources_in_corpus = sorted(
        name for name, metadata in all_doc_identity.items()
        if metadata["sha256"] in token_prior_document_hashes
    )
    reused_emitted_sources = sorted({
        row["source_cluster"] for row in index
        if all_doc_identity[row["source_cluster"]]["sha256"] in token_prior_document_hashes
    })
    token_only_record = None
    if token_only_prior:
        token_only_record = {
            **token_only_prior["identity"],
            "policy": ("seed exact prior context token SHA-256 hashes; retain prior source "
                       "documents so later clean windows may be emitted"),
            "excluded_token_hashes": len(token_only_prior_tokens),
        }
    whole_source_record = None
    if whole_prior:
        whole_source_record = {
            **whole_prior["identity"],
            "policy": ("exclude prior documents by source_cluster or document SHA-256 and "
                       "seed exact prior context token SHA-256 hashes"),
            "excluded_token_hashes": len(whole_prior_tokens),
            "matched_source_documents": len(whole_source_excluded),
        }

    manifest = {
        "schema": "qwen38-distribution-fidelity/6",
        "model": args.model,
        "model_identity": {
            "tokenizer_sha256": sha(tokenizer_file.read_bytes()),
            "config_sha256": sha(config_file.read_bytes()),
            "trust_remote_code": args.trust_remote_code,
        },
        "context_length": ctx_len,
        "scored_positions_per_context": ctx_len - 1,
        "contexts_requested": args.contexts,
        "allow_shortfall": args.allow_shortfall,
        "contexts": len(index),
        "total_scored_positions": len(index) * (ctx_len - 1),
        "hidden_size": 5120, "vocab_size": 248320,
        "corpus": str(corpus.resolve()),
        "corpus_note": ("Gutenberg / arXiv / Wikipedia (en + de,fr,es,ja,zh,ru) / "
                        "CPython v3.12.8 Lib; calibration-overlapping documents are "
                        "pre-excluded."),
        "windowing": {
            "selection": "deterministic sorted-document round-robin within each stratum",
            "file_reads": "each source file is read and UTF-8 decoded exactly once",
            "tokenizer_offsets": "fast-tokenizer return_offsets_mapping",
            "cursor_advance": (
                "advance from source_char_start to the exclusive character end offset "
                "of token 2048 (or configured context_length); unused probe suffix is "
                "retained for the next candidate"
            ),
            "candidate_windows_adjacent": True,
            "candidate_windows_non_overlapping": True,
            "rejected_candidates_create_disclosed_window_index_gaps": True,
            "initial_probe_characters": ctx_len * 4,
            "next_probe_policy": "previous consumed characters + max(64, consumed / 16)",
            "candidate_windows_tokenized": candidate_windows_tokenized,
            "candidate_source_characters_consumed":
                candidate_source_characters_consumed,
            "offset_probe_retries": offset_probe_retries,
            "incomplete_document_tails": incomplete_document_tails,
            "context_index_span_fields": [
                "source_window_index", "source_char_start", "source_char_end"
            ],
        },
        "document_scan": {
            "policy": ("scan each complete <stratum>/*.txt lexical-token stream at every "
                       "position before selection; exclude the whole document on any "
                       "calibration match"),
            "input_glob": "<stratum>/*.txt",
            "decoding": "UTF-8 with malformed bytes ignored",
            "normalization": ("Unicode NFKC, casefold, alphanumeric/underscore words, "
                              "one token per Han/Kana character"),
            "shingle_words": SHINGLE_WORDS,
            "stride_words": 1,
            "digest": "blake2b-128",
            "documents_discovered": len(all_doc_identity),
            "documents_scanned": len(all_doc_identity),
            "lexical_shingles_scanned": document_shingles_scanned,
            "documents_excluded_calibration_overlap": len(calibration_excluded),
            "documents_excluded_prior_whole_source": len(whole_source_excluded),
            "documents_excluded_by_both": len(
                calibration_excluded & whole_source_excluded
            ),
            "documents_excluded_any_reason": len(excluded_document_records),
            "documents_eligible": len(eligible_doc_identity),
            "excluded_documents": excluded_document_records,
        },
        "prior_suite_exclusion": {
            "token_hash": {
                "algorithm": "SHA-256",
                "serialization": "UTF-8 JSON token-id array using json.dumps defaults",
            },
            "whole_source": whole_source_record,
            "token_only": token_only_record,
            "excluded_token_hashes_total": len(excluded_tokens),
            "candidate_contexts_rejected_exact_token_hash": prior_token_rejections,
            "candidate_contexts_rejected_duplicate_current_suite":
                current_duplicate_rejections,
        },
        "source_reuse_disclosure": {
            "token_only_exclusion_enabled": token_only_prior is not None,
            "policy": (
                "Prior source documents are retained and may supply later windows; only "
                "exact prior context token hashes are excluded."
                if token_only_prior else
                "No token-only prior suite was supplied, so no prior-source reuse claim "
                "is made."
            ),
            "matching_prior_source_documents_present_in_corpus":
                reusable_sources_in_corpus,
            "matching_prior_source_documents_emitted": reused_emitted_sources,
            "contexts_emitted_from_matching_prior_source_documents": sum(
                1 for row in index if row["source_cluster"] in reused_emitted_sources
            ),
        },
        "contamination_scan": {
            "shingle_words": SHINGLE_WORDS,
            "stride_words": 1,
            "digest": "blake2b-128",
            "normalization": "Unicode NFKC, casefold, every Unicode word token",
            "calibration_shingles": len(cal),
            "calibration_sources": calibration_sources,
            "document_policy": "whole-document pre-exclusion on any match",
            "emitted_context_policy": "decoded exact token sequence rejected on any match",
            "decoded_candidates_rejected": decoded_overlap_rejections,
            "decoded_candidate_hits_rejected": decoded_overlap_hits,
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
            {r["source_cluster"]: all_doc_identity[r["source_cluster"]]
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
