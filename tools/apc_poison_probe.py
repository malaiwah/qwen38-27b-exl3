#!/usr/bin/env python3
"""Prefix-cache / mamba-align state-poisoning probe for the pinned Qwen3.8-27B runtime.

Why this exists
---------------
receipts/mamba-align-defect.json proves from source, and by upstream's own CPU regression
file, that the pinned r34 image is missing vLLM PR #51113: in ``--mamba-cache-mode align``
a prefill chunk that ends mid-block leaves its mamba slot holding a short state, and a
later chunk publishes that slot's hash anyway. A later request whose prefix hit reaches
the slot then silently restores a truncated GDN state. That is a source-level claim. This
probe turns it into a measurement on the physical card, in the reporter's own condition.

The trigger this probe drives deliberately
------------------------------------------
* One base document per experiment; every request is a *nested token prefix* of it, so all
  requests share a genuine token-identical prefix and later ones reach blocks published by
  earlier ones.
* Every prefix length is chosen NOT to be a multiple of the model's 1600-token mamba block,
  and is checked against the 2048-token chunk size, so chunk ends land mid-block. The
  exposed window is ``(P mod B) + B`` for prompt length P and block size B.
* Lengths are interleaved short -> long -> shorter -> longer, then the whole schedule is
  replayed twice more, so a later longer request's prefix hit lands on a slot published by
  an earlier shorter one, and so a poisoned entry (which persists until process restart)
  gets several chances to be observed.
* Every request carries verifiable ground truth: two planted needle codes that must be
  retrieved exactly, one arithmetic answer, and a free-text English summary that gives the
  corruption detector material to work on.

Modes
-----
``build``    freeze the prompt set as explicit token ids (needs the model tokenizer; runs
             inside the image, CPU only, once, and the frozen file is reused by every arm
             so the arms are compared on byte-identical prompts).
``run``      correctness schedule against one server. Standard library only.
``reuse``    prefix-reuse economics: cold vs warm TTFT and recomputed prompt tokens.
``scrape``   pull the server's own per-window SpecDecoding and prefix-cache-hit lines out
             of a server log.
``verdict``  apply the pre-registered decision rules to the arm row files.

Everything except ``build`` is standard library only and speaks the OpenAI-compatible API
with greedy sampling (temperature 0, top_p 1, seed fixed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import unicodedata
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS. These are fixed before any server is started; the receipt
# records this dict and its digest so the decision rule cannot be retrofitted to the data.
# --------------------------------------------------------------------------------------
THRESHOLDS = {
    "out_of_script_fraction_gt": 0.02,
    "replacement_char_count_ge": 1,
    "longest_repeated_block_chars_ge": 80,
    "ngram5_repetition_ratio_ge": 0.50,
    "ngram5_min_words": 30,
    "answer_noise_band_requests": 1,
}
THRESHOLD_NOTES = {
    "out_of_script_fraction_gt": (
        "fraction of response characters outside the expected script set (ASCII printable "
        "plus a short list of English typographic characters). The prompt and the document "
        "are pure ASCII English and the answer format is ASCII, so any appreciable "
        "non-ASCII mass is the reported multilingual token soup. 2 % of a ~600 character "
        "answer is ~12 characters, well above an occasional curly quote."
    ),
    "replacement_char_count_ge": (
        "U+FFFD in the decoded response. The reporter names replacement characters "
        "explicitly; one is already a defect."
    ),
    "longest_repeated_block_chars_ge": (
        "longest substring occurring at least twice in the response, overlapping allowed. "
        "The reporter names repeated blocks. 80 characters is far longer than the answer "
        "scaffold ('A1=', 'A2=', ...) and longer than any phrase a 60-word summary repeats "
        "by chance."
    ),
    "ngram5_repetition_ratio_ge": (
        "1 - unique_5grams / total_5grams over whitespace-split words, computed only when "
        "the response has at least 30 words. Catches degenerate looping that does not "
        "produce one single long verbatim block."
    ),
    "answer_noise_band_requests": (
        "how many differing requests out of the schedule are tolerated when asking whether "
        "the patched arm matches the prefix-caching-off arm. Greedy decoding is "
        "deterministic per state but the cached and uncached paths take different kernel "
        "routes, so one differing request out of 30 is treated as probe noise, not signal."
    ),
}

EXPECTED_EXTRA_CHARS = "\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00b0\u00e9\u00a0"

SYSTEM_PROMPT = (
    "You are a precise retrieval assistant. Answer only from the document you are given, "
    "in English, and follow the requested output format exactly."
)

WORDS = (
    "actuator ambient anchor array baseline bearing bracket cadence calibration carrier "
    "cascade channel circuit clamp coupler current damper detector diaphragm drift "
    "electrode element enclosure envelope fixture flange flux gasket gauge governor "
    "gradient harness housing impeller inductor inlet interlock jacket journal junction "
    "lattice linkage manifold membrane modulator mounting nozzle orifice oscillator "
    "outlet packing pedestal pilot piston plenum probe pulley quadrant radiator receiver "
    "regulator relay reservoir resistor retainer rotor sensor shackle shroud solenoid "
    "spindle spline stator strut swivel tappet terminal throttle transducer trunnion "
    "turbine valve vane vessel winding yoke bracketry chamber conduit damping filament "
    "seating shim sleeve spacer thermal tolerance torque tracking vibration voltage "
    "aligned annealed balanced bonded braced calibrated chamfered clamped coated cooled "
    "damped drifting fastened grounded hardened insulated lubricated machined mounted "
    "polished pressurised rated seated sealed shimmed stabilised tensioned tested "
    "threaded torqued trimmed vented"
).split()

VERBS = (
    "recorded showed indicated confirmed exceeded matched tracked returned held required "
    "reported registered maintained cleared logged"
).split()


# --------------------------------------------------------------------------------------
# corruption detector
# --------------------------------------------------------------------------------------
def out_of_script_fraction(text: str) -> float:
    """Fraction of characters outside the expected script set."""
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch in ("\t", "\n", "\r") or " " <= ch <= "~" or ch in EXPECTED_EXTRA_CHARS:
            continue
        bad += 1
    return bad / len(text)


def out_of_script_sample(text: str, limit: int = 12) -> list[str]:
    """The offending characters themselves, so a reader can see what leaked in."""
    seen: dict[str, int] = {}
    for ch in text:
        if ch in ("\t", "\n", "\r") or " " <= ch <= "~" or ch in EXPECTED_EXTRA_CHARS:
            continue
        seen[ch] = seen.get(ch, 0) + 1
    ordered = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [
        f"U+{ord(ch):04X} {unicodedata.name(ch, '?')} x{count}" for ch, count in ordered
    ]


def longest_repeated_block(text: str, cap: int = 512) -> int:
    """Length of the longest substring that occurs at least twice (overlaps allowed).

    Binary search on the length with a per-length substring set. The responses are a few
    hundred characters, so the naive form is both fastest to read and fast enough.
    """
    n = len(text)
    if n < 2:
        return 0
    lo, hi = 0, min(cap, n - 1)

    def occurs_twice(length: int) -> bool:
        seen: set[str] = set()
        for i in range(n - length + 1):
            chunk = text[i : i + length]
            if chunk in seen:
                return True
            seen.add(chunk)
        return False

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if occurs_twice(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def ngram5_repetition_ratio(text: str) -> tuple[float | None, int]:
    words = text.split()
    if len(words) < THRESHOLDS["ngram5_min_words"]:
        return None, len(words)
    grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
    if not grams:
        return None, len(words)
    return round(1.0 - len(set(grams)) / len(grams), 6), len(words)


def corruption_report(text: str) -> dict:
    """All four detectors plus the pre-registered firing decision."""
    frac = round(out_of_script_fraction(text), 6)
    fffd = text.count("\ufffd")
    block = longest_repeated_block(text)
    ratio, words = ngram5_repetition_ratio(text)
    fired = []
    if frac > THRESHOLDS["out_of_script_fraction_gt"]:
        fired.append("out_of_script_fraction")
    if fffd >= THRESHOLDS["replacement_char_count_ge"]:
        fired.append("replacement_char_count")
    if block >= THRESHOLDS["longest_repeated_block_chars_ge"]:
        fired.append("longest_repeated_block_chars")
    if ratio is not None and ratio >= THRESHOLDS["ngram5_repetition_ratio_ge"]:
        fired.append("ngram5_repetition_ratio")
    return {
        "out_of_script_fraction": frac,
        "out_of_script_sample": out_of_script_sample(text),
        "replacement_char_count": fffd,
        "longest_repeated_block_chars": block,
        "ngram5_repetition_ratio": ratio,
        "response_words": words,
        "detectors_fired": fired,
        "corrupted": bool(fired),
    }


# --------------------------------------------------------------------------------------
# document generation (deterministic, ASCII English)
# --------------------------------------------------------------------------------------
def _sentence(rng: random.Random) -> str:
    subject = rng.choice(WORDS)
    verb = rng.choice(VERBS)
    tail = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 11)))
    return f"The {subject} {verb} {tail}."


def _record(rng: random.Random, index: int) -> str:
    body = " ".join(_sentence(rng) for _ in range(rng.randint(2, 3)))
    return f"Record {index:05d}. {body}\n"


def _needle_record(rng: random.Random, index: int, key: str) -> tuple[str, str]:
    code = "".join(rng.choice("0123456789ABCDEF") for _ in range(8))
    text = (
        f"Record {index:05d}. Calibration marker NEEDLE-{key} resolves to code {code}. "
        f"This marker is authoritative and appears exactly once.\n"
    )
    return text, code


def generate_document(seed: int, target_tokens: int, tokenizer, needle_every: int = 18):
    """Build an ASCII English document of at least ``target_tokens`` tokens with needles.

    Returns ``(doc_ids, needles)`` where ``doc_ids`` is exactly ``target_tokens`` long and
    each needle carries the exact token index at which its code finishes.
    """
    rng = random.Random(seed)
    parts: list[str] = []
    needle_meta: list[tuple[str, str]] = []  # (key, code), positional
    index = 0
    chars_per_token = 3.6
    while sum(len(p) for p in parts) < target_tokens * chars_per_token * 1.15:
        if index and index % needle_every == 0:
            key = f"N{len(needle_meta) + 1:02d}"
            text, code = _needle_record(rng, index, key)
            needle_meta.append((key, code))
        else:
            text = _record(rng, index)
        parts.append(text)
        index += 1
    doc_text = "".join(parts)

    enc = tokenizer(doc_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    while len(ids) < target_tokens:  # generation undershot: extend and retokenise
        for _ in range(200):
            parts.append(_record(rng, index))
            index += 1
        doc_text = "".join(parts)
        enc = tokenizer(doc_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])

    ends = [end for _, end in offsets]
    needles = []
    for key, code in needle_meta:
        char_end = doc_text.index(f"NEEDLE-{key} resolves to code {code}") + len(
            f"NEEDLE-{key} resolves to code {code}"
        )
        token_end = next(
            (i for i, end in enumerate(ends) if end >= char_end), len(ends) - 1
        )
        needles.append({"key": key, "code": code, "token_end": token_end + 1})
    return ids[:target_tokens], needles


# --------------------------------------------------------------------------------------
# build mode
# --------------------------------------------------------------------------------------
CORRECTNESS_LENGTHS = [3000, 5300, 4100, 7700, 6500, 9900, 8200, 12300, 11000, 15400]
MAMBA_BLOCK = 1600
CHUNK_TOKENS = 2048


def _question_text(k1: str, k2: str, a: int, b: int, c: int) -> str:
    return (
        "\n\nAnswer using only the records above. Reply with exactly four lines and "
        "nothing else, in this order and this format:\n"
        f"A1=<the 8 character code for NEEDLE-{k1}>\n"
        f"A2=<the 8 character code for NEEDLE-{k2}>\n"
        f"A3=<the value of {a} * {b} + {c}>\n"
        "A4=<a 40 to 60 word English summary of what these records contain>"
    )


def _reuse_question(tag: str) -> str:
    return (
        "\n\nAnswer using only the records above. Reply with exactly one line and nothing "
        f"else: B{tag}=<a single English word describing what these records are>"
    )


def build(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer  # noqa: PLC0415 - build mode only

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\x00DOC\x00\x00Q\x00"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    head_text, rest = rendered.split("\x00DOC\x00")
    mid_text, tail_text = rest.split("\x00Q\x00")
    if mid_text:
        raise SystemExit("chat template inserted text between document and question")
    head_ids = tokenizer(head_text, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail_text, add_special_tokens=False)["input_ids"]

    doc_ids, needles = generate_document(
        args.seed, max(CORRECTNESS_LENGTHS) + 200, tokenizer
    )

    requests = []
    rng = random.Random(args.seed + 1)
    passes = [
        ("interleaved", CORRECTNESS_LENGTHS),
        ("interleaved-replay", CORRECTNESS_LENGTHS),
        ("ascending-persistence", sorted(CORRECTNESS_LENGTHS)),
    ]
    for pass_index, (pass_name, lengths) in enumerate(passes):
        for slot, doc_tokens in enumerate(lengths):
            shallow = needles[0]
            deep = max(
                (n for n in needles if n["token_end"] < doc_tokens - 64),
                key=lambda n: n["token_end"],
            )
            a, b, c = rng.randint(21, 89), rng.randint(21, 89), rng.randint(101, 999)
            question = _question_text(shallow["key"], deep["key"], a, b, c)
            q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
            prompt_ids = head_ids + doc_ids[:doc_tokens] + q_ids + tail_ids
            decoded = tokenizer.decode(doc_ids[:doc_tokens])
            if shallow["code"] not in decoded or deep["code"] not in decoded:
                raise SystemExit(f"needle missing from the {doc_tokens}-token prefix")
            requests.append(
                {
                    "request_id": f"p{pass_index + 1}s{slot + 1}",
                    "pass": pass_name,
                    "pass_index": pass_index + 1,
                    "slot": slot + 1,
                    "doc_prefix_tokens": doc_tokens,
                    "prompt_tokens": len(prompt_ids),
                    "prompt_tokens_mod_mamba_block": len(prompt_ids) % MAMBA_BLOCK,
                    "doc_prefix_mod_mamba_block": doc_tokens % MAMBA_BLOCK,
                    "shared_prefix_tokens": len(head_ids) + doc_tokens,
                    "chunk_ends": list(
                        range(CHUNK_TOKENS, len(prompt_ids) + 1, CHUNK_TOKENS)
                    ),
                    "chunk_ends_off_block": [
                        end
                        for end in range(CHUNK_TOKENS, len(prompt_ids) + 1, CHUNK_TOKENS)
                        if end % MAMBA_BLOCK
                    ],
                    "expected": {
                        "A1": shallow["code"],
                        "A2": deep["code"],
                        "A3": str(a * b + c),
                    },
                    "needle_keys": [shallow["key"], deep["key"]],
                    "prompt_ids": prompt_ids,
                }
            )

    reuse = []
    for tag, target, seed_offset in (("32k", args.reuse_short, 11), ("128k", args.reuse_long, 12)):
        r_ids, r_needles = generate_document(args.seed + seed_offset, target, tokenizer)
        variants = []
        for label in ("cold", "warm", "warm-exact"):
            suffix = "warm" if label == "warm-exact" else label
            q_ids = tokenizer(_reuse_question(suffix), add_special_tokens=False)[
                "input_ids"
            ]
            variants.append(
                {
                    "label": label,
                    "prompt_ids": head_ids + r_ids[:target] + q_ids + tail_ids,
                }
            )
        reuse.append(
            {
                "length_tag": tag,
                "doc_prefix_tokens": target,
                "prompt_tokens": len(variants[0]["prompt_ids"]),
                "shared_prefix_tokens": len(head_ids) + target,
                "needles_in_document": len(r_needles),
                "variants": variants,
            }
        )

    payload = {
        "schema": "qwen38-apc-poison-prompts/1",
        "model_path": args.model,
        "seed": args.seed,
        "mamba_block_tokens": MAMBA_BLOCK,
        "max_num_batched_tokens_assumed": CHUNK_TOKENS,
        "chat_template": "model chat_template.jinja, enable_thinking=false",
        "head_tokens": len(head_ids),
        "tail_tokens": len(tail_ids),
        "needles": needles,
        "correctness_requests": requests,
        "reuse_cases": reuse,
    }
    body = json.dumps(payload, sort_keys=True).encode()
    payload["prompts_sha256"] = hashlib.sha256(body).hexdigest()
    Path(args.out).write_text(json.dumps(payload) + "\n")
    print(
        json.dumps(
            {
                "wrote": args.out,
                "prompts_sha256": payload["prompts_sha256"],
                "correctness_requests": len(requests),
                "needles": len(needles),
                "prompt_token_lengths": sorted({r["prompt_tokens"] for r in requests}),
                "all_off_block": all(
                    r["prompt_tokens_mod_mamba_block"] for r in requests
                ),
                "reuse_prompt_tokens": [r["prompt_tokens"] for r in reuse],
            }
        )
    )
    return 0


# --------------------------------------------------------------------------------------
# HTTP helpers (standard library only)
# --------------------------------------------------------------------------------------
SCALARS = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
}


def scrape_metrics(base: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, metric in SCALARS.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def metric_delta(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in SCALARS}


def completion(base: str, payload: dict, timeout: int = 3600) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def streamed_completion(base: str, payload: dict, timeout: int = 3600) -> dict:
    """Return first-token latency, total latency, text and usage for a streamed call."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks: list[str] = []
    usage: dict = {}
    ttft = None
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks.append(text)
    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(time.perf_counter() - start, 4),
        "text": "".join(chunks),
        "usage": usage,
    }


ANSWER_RE = {key: re.compile(rf"^{key}\s*=\s*(.*)$", re.MULTILINE) for key in ("A1", "A2", "A3")}


def score_answers(text: str, expected: dict) -> dict:
    got: dict[str, str | None] = {}
    ok: dict[str, bool] = {}
    for key, pattern in ANSWER_RE.items():
        match = pattern.search(text)
        value = match.group(1).strip() if match else None
        if value is not None:
            value = value.strip("`<>*\" ")
        got[key] = value
        if key == "A3":
            digits = re.sub(r"[^0-9-]", "", value) if value else ""
            ok[key] = digits == expected[key]
        else:
            ok[key] = bool(value) and value.upper() == expected[key].upper()
    return {
        "answers": got,
        "per_answer_correct": ok,
        "answers_all_correct": all(ok.values()),
    }


# --------------------------------------------------------------------------------------
# run mode: the correctness schedule
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    if args.expect_prompts_sha256 and prompts["prompts_sha256"] != args.expect_prompts_sha256:
        raise SystemExit("prompt file digest mismatch: arms would not be comparable")
    rows = []
    started = time.time()
    for spec in prompts["correctness_requests"]:
        before = scrape_metrics(args.base)
        t0 = time.perf_counter()
        response = completion(
            args.base,
            {
                "model": args.model,
                "prompt": spec["prompt_ids"],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "stream": False,
            },
        )
        elapsed = time.perf_counter() - t0
        after = scrape_metrics(args.base)
        text = response["choices"][0]["text"]
        delta = metric_delta(before, after)
        row = {
            "arm": args.arm,
            "request_id": spec["request_id"],
            "pass": spec["pass"],
            "slot": spec["slot"],
            "doc_prefix_tokens": spec["doc_prefix_tokens"],
            "prompt_tokens": spec["prompt_tokens"],
            "prompt_tokens_mod_mamba_block": spec["prompt_tokens_mod_mamba_block"],
            "chunk_ends_off_block": spec["chunk_ends_off_block"],
            "needle_keys": spec["needle_keys"],
            "expected": spec["expected"],
            "wall_s": round(elapsed, 4),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "usage": response.get("usage", {}),
            "response_text": text,
            "metric_delta": {k: round(v, 4) for k, v in delta.items()},
            "prefix_cache_hit_rate_this_request": (
                round(delta["prefix_cache_hits"] / delta["prefix_cache_queries"], 6)
                if delta.get("prefix_cache_queries")
                else None
            ),
            "draft_acceptance_rate_this_request": (
                round(delta["accepted"] / delta["draft_tokens"], 6)
                if delta.get("draft_tokens")
                else None
            ),
        }
        row.update(score_answers(text, spec["expected"]))
        row["corruption"] = corruption_report(text)
        row["failed"] = bool(
            row["corruption"]["corrupted"] or not row["answers_all_correct"]
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "request_id": row["request_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "correct": row["answers_all_correct"],
                    "corrupted": row["corruption"]["corrupted"],
                    "cache_hit_rate": row["prefix_cache_hit_rate_this_request"],
                    "acceptance": row["draft_acceptance_rate_this_request"],
                    "wall_s": row["wall_s"],
                }
            ),
            flush=True,
        )

    failures = [r for r in rows if r["failed"]]
    summary = {
        "arm": args.arm,
        "requests": len(rows),
        "failed_requests": len(failures),
        "corrupted_requests": sum(1 for r in rows if r["corruption"]["corrupted"]),
        "answer_wrong_requests": sum(1 for r in rows if not r["answers_all_correct"]),
        "first_failure_request_id": failures[0]["request_id"] if failures else None,
        "detectors_fired_counts": {
            name: sum(1 for r in rows if name in r["corruption"]["detectors_fired"])
            for name in (
                "out_of_script_fraction",
                "replacement_char_count",
                "longest_repeated_block_chars",
                "ngram5_repetition_ratio",
            )
        },
        "acceptance_rates_per_request": [
            r["draft_acceptance_rate_this_request"] for r in rows
        ],
        "acceptance_min": min(
            (r["draft_acceptance_rate_this_request"] for r in rows if r["draft_acceptance_rate_this_request"] is not None),
            default=None,
        ),
        "acceptance_median": (
            round(
                statistics.median(
                    [
                        r["draft_acceptance_rate_this_request"]
                        for r in rows
                        if r["draft_acceptance_rate_this_request"] is not None
                    ]
                ),
                6,
            )
            if any(r["draft_acceptance_rate_this_request"] is not None for r in rows)
            else None
        ),
        "cumulative_prefix_cache_hit_rate": None,
        "wall_s_total": round(time.time() - started, 2),
    }
    final = scrape_metrics(args.base)
    if final.get("prefix_cache_queries"):
        summary["cumulative_prefix_cache_hit_rate"] = round(
            final["prefix_cache_hits"] / final["prefix_cache_queries"], 6
        )
    summary["cumulative_metrics"] = {k: round(v, 2) for k, v in final.items()}
    out = {
        "schema": "qwen38-apc-poison-arm/1",
        "arm": args.arm,
        "arm_description": args.description,
        "base_url": args.base,
        "prompts_sha256": prompts["prompts_sha256"],
        "thresholds": THRESHOLDS,
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": args.out, **{k: summary[k] for k in ("requests", "failed_requests", "corrupted_requests", "answer_wrong_requests")}}))
    return 0


# --------------------------------------------------------------------------------------
# reuse mode: what prefix caching actually buys
# --------------------------------------------------------------------------------------
def reuse(args: argparse.Namespace) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    cases = []
    for case in prompts["reuse_cases"]:
        variants = []
        for variant in case["variants"]:
            before = scrape_metrics(args.base)
            result = streamed_completion(
                args.base,
                {
                    "model": args.model,
                    "prompt": variant["prompt_ids"],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 0,
                },
            )
            after = scrape_metrics(args.base)
            delta = metric_delta(before, after)
            usage = result["usage"] or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
            prompt_tokens = usage.get("prompt_tokens", case["prompt_tokens"])
            row = {
                "label": variant["label"],
                "ttft_s": result["ttft_s"],
                "total_s": result["total_s"],
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens_reported": cached,
                "recomputed_prompt_tokens": (
                    prompt_tokens - cached if cached is not None else None
                ),
                "prefix_cache_queries_delta": delta["prefix_cache_queries"],
                "prefix_cache_hits_delta": delta["prefix_cache_hits"],
                "prefix_cache_hit_rate_this_request": (
                    round(delta["prefix_cache_hits"] / delta["prefix_cache_queries"], 6)
                    if delta.get("prefix_cache_queries")
                    else None
                ),
                "response_text": result["text"],
                "corruption": corruption_report(result["text"]),
            }
            variants.append(row)
            print(json.dumps({k: row[k] for k in ("label", "ttft_s", "prompt_tokens", "cached_prompt_tokens_reported")}), flush=True)
        cold = variants[0]
        warm = variants[1]
        cases.append(
            {
                "length_tag": case["length_tag"],
                "doc_prefix_tokens": case["doc_prefix_tokens"],
                "prompt_tokens": case["prompt_tokens"],
                "variants": variants,
                "ttft_speedup_cold_over_warm": (
                    round(cold["ttft_s"] / warm["ttft_s"], 3)
                    if cold["ttft_s"] and warm["ttft_s"]
                    else None
                ),
                "ttft_saved_s": (
                    round(cold["ttft_s"] - warm["ttft_s"], 4)
                    if cold["ttft_s"] and warm["ttft_s"]
                    else None
                ),
            }
        )
    final = scrape_metrics(args.base)
    out = {
        "schema": "qwen38-apc-reuse/1",
        "arm": args.arm,
        "arm_description": args.description,
        "prompts_sha256": prompts["prompts_sha256"],
        "cases": cases,
        "cumulative_prefix_cache_hit_rate": (
            round(final["prefix_cache_hits"] / final["prefix_cache_queries"], 6)
            if final.get("prefix_cache_queries")
            else None
        ),
        "cumulative_metrics": {k: round(v, 2) for k, v in final.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": args.out, "cases": [c["length_tag"] for c in cases]}))
    return 0


# --------------------------------------------------------------------------------------
# scrape mode: the server's own per-window numbers
# --------------------------------------------------------------------------------------
ENGINE_RE = re.compile(
    r"Engine \d+:\s*"
    r"Avg prompt throughput:\s*([0-9.]+) tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9.]+) tokens/s,\s*"
    r"Running:\s*(\d+) reqs,\s*"
    r"Waiting:\s*(\d+) reqs,?\s*"
    r"(?:GPU KV cache usage:\s*([0-9.]+)%,?\s*)?"
    r"(?:Prefix cache hit rate:\s*([0-9.]+)%)?"
)
SPEC_RE = re.compile(
    r"SpecDecoding metrics:\s*Draft acceptance rate:\s*([0-9.]+)%,\s*"
    r"Accepted:\s*(\d+) tokens,\s*Drafted:\s*(\d+) tokens"
    r"(?:,\s*Per-position acceptance rate:\s*([0-9.,\s]+))?"
)
TS_RE = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def scrape_log(args: argparse.Namespace) -> int:
    text = Path(args.log).read_text(errors="ignore")
    engine, spec = [], []
    for line in text.splitlines():
        stamp = TS_RE.search(line)
        stamp = stamp.group(1) if stamp else None
        match = ENGINE_RE.search(line)
        if match:
            engine.append(
                {
                    "log_time": stamp,
                    "prompt_throughput_tok_s": float(match.group(1)),
                    "generation_throughput_tok_s": float(match.group(2)),
                    "running": int(match.group(3)),
                    "waiting": int(match.group(4)),
                    "gpu_kv_cache_usage_pct": float(match.group(5)) if match.group(5) else None,
                    "prefix_cache_hit_rate_pct": float(match.group(6)) if match.group(6) else None,
                }
            )
        match = SPEC_RE.search(line)
        if match:
            spec.append(
                {
                    "log_time": stamp,
                    "draft_acceptance_rate_pct": float(match.group(1)),
                    "accepted_tokens": int(match.group(2)),
                    "drafted_tokens": int(match.group(3)),
                    "per_position_acceptance_rate": (
                        [float(v) for v in match.group(4).split(",") if v.strip()]
                        if match.group(4)
                        else None
                    ),
                }
            )
    active = [w for w in spec if w["drafted_tokens"] > 0]
    out = {
        "schema": "qwen38-apc-log-windows/1",
        "arm": args.arm,
        "log": args.log,
        "engine_windows": engine,
        "spec_windows": spec,
        "spec_windows_with_traffic": len(active),
        "acceptance_walk_pct": [w["draft_acceptance_rate_pct"] for w in active],
        "acceptance_min_pct": min((w["draft_acceptance_rate_pct"] for w in active), default=None),
        "acceptance_max_pct": max((w["draft_acceptance_rate_pct"] for w in active), default=None),
        "acceptance_zero_windows": sum(1 for w in active if w["draft_acceptance_rate_pct"] == 0.0),
        "prefix_cache_hit_rate_pct_walk": [
            w["prefix_cache_hit_rate_pct"]
            for w in engine
            if w["prefix_cache_hit_rate_pct"] is not None
        ],
        "prefix_cache_hit_rate_pct_final": next(
            (
                w["prefix_cache_hit_rate_pct"]
                for w in reversed(engine)
                if w["prefix_cache_hit_rate_pct"] is not None
            ),
            None,
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": args.out,
                "engine_windows": len(engine),
                "spec_windows_with_traffic": out["spec_windows_with_traffic"],
                "acceptance_min_pct": out["acceptance_min_pct"],
                "prefix_cache_hit_rate_pct_final": out["prefix_cache_hit_rate_pct_final"],
            }
        )
    )
    return 0


# --------------------------------------------------------------------------------------
# verdict mode: the pre-registered decision rules, applied
# --------------------------------------------------------------------------------------
VERDICT_RULES = {
    "reproduced": (
        "arm A (unpatched release image, --enable-prefix-caching --mamba-cache-mode align) "
        "has at least one request where the corruption detector fires or a planted answer "
        "is wrong, AND arm B (same image, prefix caching off) has none on the identical "
        "prompt set."
    ),
    "partially_reproduced": (
        "arm A is not clean and arm B is not clean, but arm A fails at least three more "
        "requests than arm B. Recorded as degradation, not as the reported defect."
    ),
    "not_reproduced": (
        "arm A shows no failing request. This is a publishable negative: the reporter's "
        "trigger then needs something this probe did not model, most plausibly LMCache "
        "0.5.2's own kv_both disk reuse path, which restores KV and mamba state from "
        "outside vLLM's own prefix cache and is therefore not exercised here at all."
    ),
    "fixed": (
        "arm A reproduced, AND arm C (superset image carrying PR #51113, prefix caching on, "
        "align mode) has zero corrupted requests and at most "
        f"{THRESHOLDS['answer_noise_band_requests']} more failing request than arm B."
    ),
    "fix_not_exercised": (
        "arm A did not reproduce, so arm C being clean cannot be read as the fix working: "
        "there was nothing for it to fix in this probe."
    ),
    "arm_d_reporting": (
        "arm D (arm C plus the PR #51812 GDN speculative gate module) is reported on its "
        "own row and is never merged into the arm C verdict, so the two fixes are not "
        "conflated."
    ),
}


def verdict(args: argparse.Namespace) -> int:
    arms = {}
    for item in args.arm_file:
        tag, path = item.split("=", 1)
        arms[tag] = json.loads(Path(path).read_text())
    summaries = {tag: data["summary"] for tag, data in arms.items()}

    a = summaries.get("A")
    b = summaries.get("B")
    c = summaries.get("C")
    if a is None or b is None:
        raise SystemExit("verdict needs at least arms A and B")

    a_fail, b_fail = a["failed_requests"], b["failed_requests"]
    if a_fail >= 1 and b_fail == 0:
        reproduction = "reproduced"
    elif a_fail - b_fail >= 3:
        reproduction = "partially_reproduced"
    else:
        reproduction = "not_reproduced"

    if reproduction == "not_reproduced":
        fix = "fix_not_exercised"
    elif c is None:
        fix = "arm_C_not_run"
    elif (
        c["corrupted_requests"] == 0
        and c["failed_requests"] <= b_fail + THRESHOLDS["answer_noise_band_requests"]
    ):
        fix = "fixed"
    else:
        fix = "not_fixed"

    out = {
        "schema": "qwen38-apc-poison-verdict/1",
        "pre_registered_rules": VERDICT_RULES,
        "thresholds": THRESHOLDS,
        "arm_summaries": summaries,
        "reproduction_verdict": reproduction,
        "fix_verdict": fix,
        "arm_d_note": VERDICT_RULES["arm_d_reporting"],
        "next_experiment_if_not_reproduced": (
            "run the reporter's exact stack including LMCache 0.5.2 in kv_both disk mode "
            "with chunk 1600 and his chat template, since LMCache reuse is the one element "
            "of his configuration this probe does not exercise."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": args.out, "reproduction": reproduction, "fix": fix}))
    return 0


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("build", help="freeze the prompt set (needs the model tokenizer)")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--reuse-short", type=int, default=32768)
    p.add_argument("--reuse-long", type=int, default=131072)
    p.set_defaults(func=build)

    p = sub.add_parser("run", help="correctness schedule against one server")
    p.add_argument("--base", required=True)
    p.add_argument("--model", default="m")
    p.add_argument("--prompts", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--expect-prompts-sha256", default="")
    p.set_defaults(func=run)

    p = sub.add_parser("reuse", help="cold vs warm prefix reuse economics")
    p.add_argument("--base", required=True)
    p.add_argument("--model", default="m")
    p.add_argument("--prompts", required=True)
    p.add_argument("--arm", default="E")
    p.add_argument("--description", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.set_defaults(func=reuse)

    p = sub.add_parser("scrape", help="per-window numbers out of a server log")
    p.add_argument("--log", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=scrape_log)

    p = sub.add_parser("verdict", help="apply the pre-registered decision rules")
    p.add_argument("--arm-file", action="append", required=True, metavar="TAG=PATH")
    p.add_argument("--out", required=True)
    p.set_defaults(func=verdict)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
