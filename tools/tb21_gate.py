#!/usr/bin/env python3
"""TB2.1 fidelity gate (docs/46 §6): five checks against a live endpoint.

Runs before any scored TB pass on a new serving config. Each check reports
PASS / FAIL / SKIP with evidence in the receipt; exit is nonzero iff any
check FAILs (SKIP-with-reason never fails the gate).

  1. liveness      - /health 200 + /v1/models served id.
  2. generation    - 8-token proof, finish_reason present.
  3. repeatability - 8 deterministic frozen prompts (temperature 0, seed
                     passed when the server accepts it) run TWICE in the SAME
                     server process; token-identical both times. Cross-restart
                     identity is NOT claimable on this stack (greedy text
                     differs across engine restarts 7/8, identical
                     within-process 8/8 - receipts/scratch-arena.json); this
                     gate only ever claims within-process repeatability.
  4. needle        - deterministic ~N-token document (--needle-tokens, default
                     the 262,144 native window minus a 2,048-token reserve for
                     chat template + question + answer), needle at depth 0.5,
                     exact-match retrieval required. Token estimation via the
                     server's /tokenize when present, else the chars/4
                     heuristic (method named in the receipt).
                     --tp-smoke relaxes to a 32k document (the 2x G0 gate
                     needs fast turnaround); --skip-needle for rehearsal.
  5. MTP sanity    - /metrics counter deltas across check 3: draft/accepted
                     present and acceptance > 0.3 when a speculative config is
                     active; SKIP with reason when absent (BF16 arm without
                     MTP, TP smoke, or an endpoint without /metrics).

Both /metrics snapshots are stored verbatim in the receipt with parsed deltas
(counter-delta discipline, docs/46 §3). HTTP failures are recorded outcomes.
Stdlib only; runs on the driver VM's python3.12 with no venv.

The mock preflight double (tools/tb_mock_endpoint.py, id owned_by
"mock-preflight-endpoint") has no /health, no /metrics, and its replies embed
a global call counter, so checks 3-5 SKIP against it with the reason recorded;
checks 1-2 run for real. A gate receipt produced against the mock is
self-identifying and never backs a scored pass.

Usage:
  tb21_gate.py --base-url http://10.0.0.3:8000 --out receipts/tb21-gate-1x-quant-dp1.json \
      [--api-key K] [--tp-smoke] [--skip-needle] [--needle-tokens 262144] [--seed 46]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import re
import ssl
import sys
import time
import urllib.parse
from datetime import datetime, timezone

SCHEMA = "qwen38-tb21-gate/1"
MOCK_OWNER = "mock-preflight-endpoint"
CROSS_RESTART_NOTE = (
    "Within-process repeatability only. Cross-restart bit-exactness is NOT "
    "claimable on this stack: greedy text differs across engine restarts 7/8 "
    "and is identical within a process 8/8 (exl3_gemm autotune selects kernel "
    "configs by measured time per process) - receipts/scratch-arena.json."
)

WORDS = (
    "ledger granite harbor lattice ember quorum vessel timber orchard nickel "
    "meadow cipher walnut breeze summit copper hollow prairie anchor tundra "
    "marble falcon lantern gravel monsoon pigment thicket saffron drizzle "
    "boulder canyon iodine juniper kestrel liquorice mineral nectar obsidian "
    "paprika quartz russet sequoia tamarind umber vertex willow xenon yarrow"
).split()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- transport


class Endpoint:
    def __init__(self, base_url: str, api_key: str | None, timeout: float):
        u = urllib.parse.urlsplit(base_url.rstrip("/"))
        self.scheme = u.scheme or "http"
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if self.scheme == "https" else 80)
        self.root = u.path[:-3] if u.path.endswith("/v1") else u.path
        self.api_key = api_key
        self.timeout = timeout

    def call(self, method: str, path: str, payload=None,
             timeout: float | None = None) -> dict:
        """One exchange; never raises. Returns
        {status, body(str), json, error, latency_s}."""
        hdrs = {"Accept": "*/*", "Connection": "close",
                "User-Agent": "tb21-gate/1"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        body = None
        if payload is not None:
            body = json.dumps(payload)
            hdrs["Content-Type"] = "application/json"
        t0 = time.perf_counter()
        try:
            if self.scheme == "https":
                conn = http.client.HTTPSConnection(
                    self.host, self.port, timeout=timeout or self.timeout,
                    context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(
                    self.host, self.port, timeout=timeout or self.timeout)
            conn.request(method, self.root + path, body=body, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            out = {"status": resp.status,
                   "body": raw.decode("utf-8", "replace"),
                   "latency_s": round(time.perf_counter() - t0, 4),
                   "error": None}
            try:
                out["json"] = json.loads(raw)
            except ValueError:
                out["json"] = None
            conn.close()
            return out
        except Exception as exc:
            return {"status": 0, "body": "", "json": None,
                    "latency_s": round(time.perf_counter() - t0, 4),
                    "error": f"{type(exc).__name__}: {exc}"}

    def chat(self, model: str, prompt: str, max_tokens: int,
             seed: int | None = None, timeout: float | None = None,
             no_think: bool = False) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if seed is not None:
            payload["seed"] = seed
        if no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return self.call("POST", "/v1/chat/completions", payload,
                         timeout=timeout)


# ---------------------------------------------------------------- metrics

METRIC_FAMILIES = {
    "prompt_tokens": r"^vllm:prompt_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "generation_tokens": r"^vllm:generation_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "draft_tokens": r"^vllm:spec_decode_num_draft_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "accepted_tokens": r"^vllm:spec_decode_num_accepted_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_queries": r"^vllm:(?:gpu_)?prefix_cache_queries_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_hits": r"^vllm:(?:gpu_)?prefix_cache_hits_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
}


def metrics_snapshot(ep: Endpoint) -> dict:
    out = ep.call("GET", "/metrics")
    if out["error"] or out["status"] != 200:
        return {"available": False, "utc": utcnow(),
                "reason": out["error"] or f"HTTP {out['status']}"}
    text = out["body"]
    sums = {}
    lines = {}
    for fam, pat in METRIC_FAMILIES.items():
        matches = list(re.finditer(pat, text, re.M))
        if matches:
            sums[fam] = sum(float(m.group(2)) for m in matches)
            lines[fam] = [m.group(0) for m in matches]
    return {"available": True, "utc": utcnow(), "verbatim": text,
            "sums": sums, "counter_lines_verbatim": lines}


# ---------------------------------------------------------------- checks


def synth_words(rng: random.Random, n_chars: int) -> str:
    out, total = [], 0
    while total < n_chars:
        w = rng.choice(WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)[:n_chars]


def frozen_prompts(seed: int) -> list[str]:
    prompts = []
    for i in range(8):
        rng = random.Random(f"tb21-gate-frozen:{seed}:{i}")
        prompts.append(
            f"[tb21-gate frozen prompt {i}, seed {seed}] Continue the "
            f"following text in the same style:\n\n{synth_words(rng, 800)}"
        )
    return prompts


def check_liveness(ep: Endpoint) -> tuple[dict, str | None, bool]:
    """Returns (check, served_model_id, is_mock)."""
    health = ep.call("GET", "/health")
    models = ep.call("GET", "/v1/models")
    model_id, owned_by = None, None
    if models["status"] == 200 and models["json"]:
        data = (models["json"].get("data") or [{}])[0]
        model_id = data.get("id")
        owned_by = data.get("owned_by")
    is_mock = owned_by == MOCK_OWNER
    ev = {
        "health_status": health["status"],
        "health_error": health["error"],
        "health_latency_s": health["latency_s"],
        "models_status": models["status"],
        "served_model_id": model_id,
        "owned_by": owned_by,
        "models_payload": models["json"],
    }
    if model_id is None:
        status = "FAIL"
        ev["why"] = "/v1/models did not return a served model id"
    elif health["status"] == 200:
        status = "PASS"
    elif is_mock:
        status = "PASS"
        ev["why"] = ("/health absent by design on the mock preflight double "
                     "(tools/tb_mock_endpoint.py serves only /v1/models and "
                     "chat completions); /v1/models proves liveness")
    else:
        status = "FAIL"
        ev["why"] = (f"/health returned {health['status']} "
                     f"({health['error']}) on a non-mock endpoint")
    return ({"name": "liveness", "status": status, "evidence": ev},
            model_id, is_mock)


def check_generation(ep: Endpoint, model: str) -> dict:
    # no_think=True: this is a graded liveness probe, not a capability probe.
    # Without it a reasoning model can spend the whole token budget on its
    # <think> preamble and never reach `content`, which is a false FAIL, not
    # a serving defect (confirmed live: max_tokens=8 with thinking on emits
    # content=None, finish_reason="length"; the same request with
    # enable_thinking=False emits content="ready", finish_reason="stop").
    out = ep.chat(model, "Reply with the single word: ready", max_tokens=16,
                  no_think=True)
    ev = {"http_status": out["status"], "latency_s": out["latency_s"],
          "error": out["error"], "no_think": True}
    status = "FAIL"
    if out["status"] == 200 and out["json"]:
        ch = (out["json"].get("choices") or [{}])[0]
        msg = (ch.get("message") or {}).get("content")
        reasoning = (ch.get("message") or {}).get("reasoning")
        ev.update(content=msg, reasoning=reasoning,
                  finish_reason=ch.get("finish_reason"),
                  usage=out["json"].get("usage"))
        # Accept reasoning-only output too: enable_thinking=False is a
        # request, not a guarantee every template honours, and a non-empty
        # reasoning field under a real finish_reason is still proof the
        # engine is generating - the thing this check exists to establish.
        if (msg or reasoning) and ch.get("finish_reason") is not None:
            status = "PASS"
        else:
            ev["why"] = "missing content, missing reasoning, or missing finish_reason"
    else:
        ev["why"] = "non-200 or unparseable response"
        ev["body_head"] = out["body"][:300]
    return {"name": "generation_proof", "status": status, "evidence": ev}


def check_repeatability(ep: Endpoint, model: str, seed: int,
                        is_mock: bool) -> dict:
    if is_mock:
        return {"name": "frozen_prompt_repeatability", "status": "SKIP",
                "evidence": {"reason": (
                    "mock preflight double: replies embed a global call "
                    "counter (terminus_reply(call_index) in "
                    "tb_mock_endpoint.py), so two runs can never be "
                    "token-identical by design; repeatability is only "
                    "meaningful against the real engine"),
                    "note": CROSS_RESTART_NOTE}}
    prompts = frozen_prompts(seed)
    seed_supported = True
    probe = ep.chat(model, prompts[0], max_tokens=4, seed=seed)
    if probe["status"] == 400:
        seed_supported = False  # server rejects the seed param; rerun without

    def run_once(tag: str) -> list[dict]:
        rows = []
        for i, p in enumerate(prompts):
            out = ep.chat(model, p, max_tokens=64,
                          seed=seed if seed_supported else None)
            if out["status"] != 200 or not out["json"]:
                rows.append({"i": i, "ok": False, "status": out["status"],
                             "error": out["error"],
                             "body_head": out["body"][:300]})
                continue
            ch = (out["json"].get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content") or ""
            rows.append({
                "i": i, "ok": True,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "completion_tokens":
                    (out["json"].get("usage") or {}).get("completion_tokens"),
                "finish_reason": ch.get("finish_reason"),
                "content_head": content[:120],
            })
        return rows

    run_a = run_once("A")
    run_b = run_once("B")
    matches = []
    for a, b in zip(run_a, run_b):
        matches.append(bool(
            a.get("ok") and b.get("ok")
            and a["sha256"] == b["sha256"]
            and a["completion_tokens"] == b["completion_tokens"]))
    all_ok = all(r.get("ok") for r in run_a + run_b)
    identical = all(matches) and len(matches) == 8
    ev = {
        "prompts": 8,
        "prompt_synthesis": f"seeded word filler, seed {seed}, 800 chars each",
        "temperature": 0.0,
        "seed_param_supported": seed_supported,
        "identity_criterion": "sha256(content) equal AND "
                              "usage.completion_tokens equal, per prompt "
                              "(text equality implies token identity under "
                              "one tokenizer)",
        "matches": matches,
        "run_a": run_a,
        "run_b": run_b,
        "combined_sha256_run_a": hashlib.sha256(
            "".join(r.get("sha256", "x") for r in run_a).encode()).hexdigest(),
        "combined_sha256_run_b": hashlib.sha256(
            "".join(r.get("sha256", "x") for r in run_b).encode()).hexdigest(),
        "note": CROSS_RESTART_NOTE,
    }
    status = "PASS" if (all_ok and identical) else "FAIL"
    if not all_ok:
        ev["why"] = "one or more requests failed outright"
    elif not identical:
        ev["why"] = f"non-identical prompts: {[i for i, m in enumerate(matches) if not m]}"
    return {"name": "frozen_prompt_repeatability", "status": status,
            "evidence": ev}


def build_needle_doc(seed: int, target_chars: int) -> tuple[str, str, str]:
    """Deterministic document with the needle at char-depth 0.5."""
    rng = random.Random(f"tb21-gate-needle:{seed}")
    code = "".join(str(rng.randint(0, 9)) for _ in range(9))
    needle = f"The vault access code is {code}."
    half = max(0, (target_chars - len(needle)) // 2)
    front = synth_words(random.Random(f"tb21-needle-front:{seed}"), half)
    back = synth_words(random.Random(f"tb21-needle-back:{seed}"),
                       target_chars - half - len(needle))
    doc = f"{front}\n{needle}\n{back}"
    return doc, needle, code


def check_needle(ep: Endpoint, model: str, args, is_mock: bool) -> dict:
    name = "needle_retrieval"
    if args.skip_needle:
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": "--skip-needle (rehearsal run)"}}
    if is_mock:
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": (
                    "mock preflight double returns canned Terminus-2 JSON "
                    "regardless of the prompt; retrieval is not meaningful "
                    "without a real model")}}
    window = 32768 if args.tp_smoke else args.needle_tokens
    reserve = 2048  # chat template + question + answer headroom
    doc_target_tokens = window - reserve

    # chars-per-token: /tokenize when present, else chars/4 (named). This is
    # only a STARTING estimate - the trim loop below makes the actual send
    # size self-correcting against the real tokenizer, so a calibration miss
    # here costs one extra /tokenize round trip, never a 400.
    sample = synth_words(random.Random("tb21-gate-cpt"), 8000)
    tk = ep.call("POST", "/tokenize", {"model": model, "prompt": sample})
    if tk["status"] == 200 and tk["json"] and tk["json"].get("count"):
        cpt = len(sample) / int(tk["json"]["count"])
        cpt_method = (f"/tokenize calibration: {len(sample)} chars -> "
                      f"{tk['json']['count']} tokens")
    else:
        cpt = 4.0
        cpt_method = "chars/4 heuristic (/tokenize unavailable)"

    # Deliberately UNDERSHOOT the first probe. A short-sample chars/token ratio
    # underestimates tokens on a 1.2M-char document by ~0.4 %, so a first probe
    # aimed straight at the budget tokenizes ~1,100 tokens OVER the window -
    # harmless (a /tokenize call runs no forward pass and the send is trimmed
    # before it happens) but it makes transformers emit "Token indices sequence
    # length is longer than the specified maximum ... will result in indexing
    # errors" into the serving log of a qualified run, where it reads like a
    # correctness problem and is not one. Starting 1.5 % low and growing into
    # the budget keeps the log clean and still lands within ~0.5 % of the
    # window. Measured: probe0 263,243 > 262,144 before this, under it after.
    UNDERSHOOT = 0.985
    doc, needle, code = build_needle_doc(
        args.seed, int(doc_target_tokens * cpt * UNDERSHOOT))

    # Fit-to-budget: /tokenize the ACTUAL doc and correct against the real
    # tokenizer rather than trusting the char/token estimate. Bounded at 6
    # tries; each re-tokenizes using the doc's OWN measured ratio, so it
    # converges in one or two passes from either direction.
    measured_tokens = None
    trims = []
    for attempt in range(6):
        tk2 = ep.call("POST", "/tokenize", {"model": model, "prompt": doc},
                      timeout=max(ep.timeout, 300))
        if not (tk2["status"] == 200 and tk2["json"] and tk2["json"].get("count")):
            break  # /tokenize itself failed; fall through and let the real
                   # chat request surface the error, unchanged from before.
        measured_tokens = int(tk2["json"]["count"])
        trims.append({"attempt": attempt, "doc_chars": len(doc),
                      "measured_tokens": measured_tokens})
        doc_cpt = len(doc) / measured_tokens
        if measured_tokens > doc_target_tokens:
            # Shrink by the measured overshoot, converted back to chars via the
            # doc's own just-measured ratio, plus 0.5 % so it does not hover at
            # the edge.
            cut_chars = int((measured_tokens - doc_target_tokens) * doc_cpt * 1.005)
            doc, needle, code = build_needle_doc(
                args.seed, max(1000, len(doc) - cut_chars))
            trims[-1]["cut_chars"] = cut_chars
            continue
        # Under budget: grow into it only while the gap is worth another round
        # trip, and aim 0.3 % short so growth can never cross the window.
        gap = doc_target_tokens - measured_tokens
        if gap <= doc_target_tokens * 0.005 or attempt == 5:
            break
        add_chars = int(gap * doc_cpt * 0.997)
        doc, needle, code = build_needle_doc(args.seed, len(doc) + add_chars)
        trims[-1]["add_chars"] = add_chars

    prompt = (
        f"{doc}\n\nQuestion: what is the vault access code stated in the "
        "document above? Answer with the digits only."
    )
    # no_think=True for the same reason as generation_proof: this is an
    # exact-match retrieval grade, and reasoning tokens on a 260k-token
    # document would burn the answer budget before any digits appear.
    out = ep.chat(model, prompt, max_tokens=64, no_think=True,
                  timeout=max(ep.timeout, 1800))
    ev = {
        "mode": "tp-smoke 32k" if args.tp_smoke else "full window",
        "window_tokens": window,
        "reserve_tokens": reserve,
        "doc_target_tokens": doc_target_tokens,
        "doc_chars": len(doc),
        "chars_per_token": round(cpt, 4),
        "token_estimation_method": cpt_method,
        "trim_to_fit_attempts": trims,
        "doc_measured_tokens_via_tokenize": measured_tokens,
        "needle_depth": 0.5,
        "needle": needle,
        "expected_code": code,
        "no_think": True,
        "http_status": out["status"],
        "latency_s": out["latency_s"],
        "error": out["error"],
    }
    if out["status"] != 200 or not out["json"]:
        ev["why"] = "request refused or failed (recorded outcome)"
        ev["body_head"] = out["body"][:500]
        return {"name": name, "status": "FAIL", "evidence": ev}
    ch = (out["json"].get("choices") or [{}])[0]
    answer = (ch.get("message") or {}).get("content") or ""
    ev["answer"] = answer
    ev["usage"] = out["json"].get("usage")
    ev["retrieved_exact"] = code in answer
    return {"name": name,
            "status": "PASS" if code in answer else "FAIL",
            "evidence": ev}


def _token_len(ep: Endpoint, model: str, text: str) -> int | None:
    """Exact token length via /tokenize, or None when the endpoint has no such route."""
    out = ep.call("POST", "/tokenize", {"model": model, "prompt": text},
                  timeout=max(ep.timeout, 300))
    if out["status"] == 200 and out["json"] and out["json"].get("count"):
        return int(out["json"]["count"])
    return None


def build_reuse_doc(seed: int, total_chars: int,
                    needle_char_offsets: list[int]) -> tuple[str, list[dict]]:
    """Deterministic filler carrying one needle sentence at each requested char offset."""
    rng = random.Random(f"tb21-gate-reuse:{seed}")
    parts: list[str] = []
    needles: list[dict] = []
    cursor = 0
    for i, offset in enumerate(sorted(needle_char_offsets)):
        label = f"N{i + 1}"
        code = "".join(str(rng.randint(0, 9)) for _ in range(9))
        sentence = f" The {label} vault access code is {code}. "
        gap = max(0, offset - cursor)
        if gap:
            filler = synth_words(random.Random(f"tb21-reuse-fill:{seed}:{i}"), gap)
            parts.append(filler)
            cursor += len(filler)
        parts.append(sentence)
        cursor += len(sentence)
        needles.append({"label": label, "code": code,
                        "sentence": sentence.strip(), "char_end": cursor})
    tail = max(0, total_chars - cursor)
    if tail:
        parts.append(synth_words(random.Random(f"tb21-reuse-tail:{seed}"), tail))
    return "".join(parts), needles


def check_connector_reuse(ep: Endpoint, model: str, args, is_mock: bool) -> dict:
    """Check 6: reuse a prefix ACROSS a mamba block boundary and score the answer.

    Why this check exists, in one measured sentence: on 2026-08-17 this gate returned PASS
    on all five of its other checks - the 262,144-token needle included - against a server
    that the frozen poisoning probe had scored 38/38 CORRUPTED minutes earlier
    (receipts/lmcache-l1-2x.json, receipts/tb21-gate-2x-lmcache-l1.json). None of the five
    reuses a cached prefix across a mamba block boundary, so none of them can see the
    hybrid+KV-connector state-loss defect class. A gate PASS is not fidelity whenever a KV
    connector is attached.

    Geometry, in tokens, all multiples of the served attention/mamba block B
    (--reuse-block-tokens, the value vLLM logs as "Setting attention block size to N
    tokens"):

      request A  document truncated at 4B + 500, asks for needle N1 at depth ~1.5B
                 -> publishes cache blocks up to 4B and is its own control
      request B  the SAME document truncated at 4B + 900, so it shares a >4B prefix with A
                 and its recorded hit floors to a block multiple; asks for N1 (a needle far
                 from any boundary, which must survive) AND for N2 at depth 4B - 200, which
                 lies inside [3B, 4B) - the LAST block of the hit, i.e. exactly the
                 divergence window between the full-attention group's hit and the last
                 position where a recurrent-state checkpoint exists.

    A correct engine answers both needles in B. An engine that takes the full-attention
    group's hit as the model-wide computed prefix, or a connector that supplies KV without
    the matching SSM/GDN state, loses N2 while keeping N1.

    Status is NOT_EXERCISED - never PASS - if B did not actually reuse at least one block,
    because a check that never touched the path proves nothing about it.
    """
    name = "connector_prefix_reuse"
    if not args.check_connector_reuse:
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": (
                    "opt-in check, off by default so every previously published gate "
                    "receipt stays comparable. MANDATORY BY POLICY whenever the server "
                    "runs with --kv-transfer-config (any KV connector attached).")}}
    if is_mock:
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": "mock preflight double returns canned JSON; "
                                       "prefix reuse is not meaningful without a real model"}}
    block = args.reuse_block_tokens
    a_tokens = 4 * block + 500
    b_tokens = 4 * block + 900
    n1_depth = int(1.5 * block)
    n2_depth = 4 * block - 200
    ev: dict = {
        "block_tokens": block,
        "target_geometry_tokens": {
            "request_a_doc": a_tokens, "request_b_doc": b_tokens,
            "needle_1_depth": n1_depth, "needle_2_depth": n2_depth,
            "needle_2_window": [3 * block, 4 * block],
        },
    }

    sample = synth_words(random.Random("tb21-gate-reuse-cpt"), 8000)
    n = _token_len(ep, model, sample)
    if n:
        cpt = len(sample) / n
        ev["token_estimation_method"] = f"/tokenize calibration: 8000 chars -> {n} tokens"
    else:
        cpt = 4.0
        ev["token_estimation_method"] = "chars/4 heuristic (/tokenize unavailable)"
    ev["chars_per_token"] = round(cpt, 4)

    doc, needles = build_reuse_doc(
        args.seed, int(b_tokens * cpt),
        [int(n1_depth * cpt), int(n2_depth * cpt)])
    a_doc = doc[:int(a_tokens * cpt)]
    b_doc = doc[:int(b_tokens * cpt)]
    # measured, not assumed: where the needles and the two documents actually land
    ev["measured_tokens"] = {
        "request_a_doc": _token_len(ep, model, a_doc),
        "request_b_doc": _token_len(ep, model, b_doc),
        "needle_1_depth": _token_len(ep, model, doc[:needles[0]["char_end"]]),
        "needle_2_depth": _token_len(ep, model, doc[:needles[1]["char_end"]]),
    }
    n2 = ev["measured_tokens"]["needle_2_depth"]
    ev["needle_2_lands_in_the_divergence_window"] = (
        None if n2 is None else bool(3 * block <= n2 < 4 * block))
    ev["needles"] = [{"label": x["label"], "code": x["code"]} for x in needles]

    ask_one = ("\n\nQuestion: what is the N1 vault access code stated in the document "
               "above? Answer with the digits only.")
    ask_both = ("\n\nAnswer using only the document above, on exactly two lines and "
                "nothing else:\nN1=<the N1 vault access code>\nN2=<the N2 vault access "
                "code>")
    long_timeout = max(ep.timeout, 900)

    a_out = ep.chat(model, a_doc + ask_one, max_tokens=48, no_think=True,
                    timeout=long_timeout)
    a_ans = ((a_out["json"] or {}).get("choices") or [{}])[0]
    a_text = ((a_ans.get("message") or {}).get("content") or "")
    ev["request_a"] = {"http_status": a_out["status"], "latency_s": a_out["latency_s"],
                       "error": a_out["error"], "answer": a_text,
                       "n1_retrieved": needles[0]["code"] in a_text}

    m_before = metrics_snapshot(ep)
    b_out = ep.chat(model, b_doc + ask_both, max_tokens=64, no_think=True,
                    timeout=long_timeout)
    m_after = metrics_snapshot(ep)
    b_ans = ((b_out["json"] or {}).get("choices") or [{}])[0]
    b_text = ((b_ans.get("message") or {}).get("content") or "")
    hits = None
    if m_before.get("available") and m_after.get("available"):
        hits = round(m_after["sums"].get("prefix_cache_hits", 0.0)
                     - m_before["sums"].get("prefix_cache_hits", 0.0), 3)
    ev["request_b"] = {
        "http_status": b_out["status"], "latency_s": b_out["latency_s"],
        "error": b_out["error"], "answer": b_text,
        "n1_retrieved": needles[0]["code"] in b_text,
        "n2_retrieved": needles[1]["code"] in b_text,
        "usage": (b_out["json"] or {}).get("usage"),
        "prefix_cache_hits_delta": hits,
        "prefix_cache_hits_delta_scope": "request B only, between two named snapshots",
    }
    ev["reuse_exercised"] = bool(hits is not None and hits >= block)

    if a_out["status"] != 200 or b_out["status"] != 200:
        ev["why"] = "request refused or failed (recorded outcome)"
        return {"name": name, "status": "FAIL", "evidence": ev}
    if not ev["request_a"]["n1_retrieved"]:
        ev["why"] = ("request A could not retrieve its own needle without any reuse: the "
                     "check's own control failed, so nothing about reuse is measurable")
        return {"name": name, "status": "FAIL", "evidence": ev}
    if not ev["reuse_exercised"]:
        ev["why"] = (f"request B reused fewer than one {block}-token block "
                     f"(prefix_cache_hits delta {hits}); the boundary this check exists to "
                     "cross was never crossed. Enable prefix caching, and check that the "
                     "block size passed as --reuse-block-tokens matches the server's own "
                     "\"Setting attention block size to N tokens\" line.")
        return {"name": name, "status": "NOT_EXERCISED", "evidence": ev}
    ok = ev["request_b"]["n1_retrieved"] and ev["request_b"]["n2_retrieved"]
    if not ok:
        ev["why"] = (
            "request B reused a prefix across the block boundary and then answered wrong. "
            "Losing N2 while keeping N1 is the hybrid+connector signature: attention KV is "
            "valid across the reused prefix but the recurrent (Mamba/GDN) state at the "
            "resume boundary is not restored, so content inside the last reused block is "
            "gone. See receipts/lmcache-l1-2x.json."
            if ev["request_b"]["n1_retrieved"] else
            "request B reused a prefix across the block boundary and lost both needles: "
            "total state loss, not a bounded divergence window.")
    return {"name": name, "status": "PASS" if ok else "FAIL", "evidence": ev}


def check_mtp(before: dict, after: dict, is_mock: bool) -> dict:
    name = "mtp_sanity"
    if is_mock:
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": "mock preflight double has no /metrics"}}
    if not (before.get("available") and after.get("available")):
        return {"name": name, "status": "SKIP",
                "evidence": {"reason": (
                    "/metrics unavailable on this endpoint: "
                    f"{before.get('reason') or after.get('reason')}")}}
    b, a = before["sums"], after["sums"]
    if "draft_tokens" not in a or "accepted_tokens" not in a:
        return {"name": name, "status": "SKIP",
                "evidence": {
                    "reason": ("speculative counters absent from /metrics - "
                               "no active speculative config on this arm "
                               "(e.g. BF16 without MTP, or TP smoke)"),
                    "families_present": sorted(a.keys()),
                }}
    draft = a["draft_tokens"] - b.get("draft_tokens", 0.0)
    accepted = a["accepted_tokens"] - b.get("accepted_tokens", 0.0)
    ev = {
        "window": "counter deltas across check 3 (frozen-prompt runs)",
        "draft_delta": draft,
        "accepted_delta": accepted,
        "acceptance": round(accepted / draft, 4) if draft > 0 else None,
        "threshold": 0.3,
        "counter_lines_before": {
            k: before["counter_lines_verbatim"].get(k)
            for k in ("draft_tokens", "accepted_tokens")},
        "counter_lines_after": {
            k: after["counter_lines_verbatim"].get(k)
            for k in ("draft_tokens", "accepted_tokens")},
    }
    if draft <= 0:
        ev["why"] = ("speculative counters present but no draft tokens were "
                     "produced during check 3 - speculative path inactive")
        return {"name": name, "status": "FAIL", "evidence": ev}
    ok = (accepted / draft) > 0.3
    if not ok:
        ev["why"] = "acceptance <= 0.3 with active speculative config"
    return {"name": name, "status": "PASS" if ok else "FAIL", "evidence": ev}


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True,
                    help="http://HOST:8000 (with or without /v1)")
    ap.add_argument("--out", required=True, help="receipt JSON path")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None,
                    help="served model id (default: first from /v1/models)")
    ap.add_argument("--needle-tokens", type=int, default=262144,
                    help="context window the needle doc targets "
                         "(doc = window - 2048 reserve; default 262144)")
    ap.add_argument("--tp-smoke", action="store_true",
                    help="G0 fast path: 32k needle window")
    ap.add_argument("--skip-needle", action="store_true",
                    help="rehearsal: skip check 4")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request timeout (needle/tokenize get more)")
    ap.add_argument("--check-connector-reuse", action="store_true",
                    help="check 6: reuse a prefix across a mamba block boundary and score "
                         "the answer. OFF by default so previously published receipts stay "
                         "comparable; MANDATORY BY POLICY whenever the server runs with "
                         "--kv-transfer-config (see receipts/lmcache-l1-2x.json: the other "
                         "five checks all PASSED against a server measured 38/38 corrupted)")
    ap.add_argument("--reuse-block-tokens", type=int, default=1600,
                    help="served attention/mamba block size for check 6; must match the "
                         "server's own \"Setting attention block size to N tokens\" line "
                         "(default 1600, this model's value)")
    args = ap.parse_args()

    ep = Endpoint(args.base_url, args.api_key, args.timeout)
    started = utcnow()
    checks = []

    c1, model_id, is_mock = check_liveness(ep)
    checks.append(c1)
    model = args.model or model_id
    print(f"[gate] 1 liveness: {c1['status']} (model={model}, "
          f"mock={is_mock})", flush=True)

    if model:
        c2 = check_generation(ep, model)
    else:
        c2 = {"name": "generation_proof", "status": "FAIL",
              "evidence": {"why": "no served model id to address"}}
    checks.append(c2)
    print(f"[gate] 2 generation: {c2['status']}", flush=True)

    m_before = metrics_snapshot(ep)
    c3 = (check_repeatability(ep, model, args.seed, is_mock) if model else
          {"name": "frozen_prompt_repeatability", "status": "FAIL",
           "evidence": {"why": "no served model id"}})
    m_after = metrics_snapshot(ep)
    checks.append(c3)
    print(f"[gate] 3 repeatability: {c3['status']}", flush=True)

    c4 = (check_needle(ep, model, args, is_mock) if model else
          {"name": "needle_retrieval", "status": "FAIL",
           "evidence": {"why": "no served model id"}})
    checks.append(c4)
    print(f"[gate] 4 needle: {c4['status']}", flush=True)

    c5 = check_mtp(m_before, m_after, is_mock)
    checks.append(c5)
    print(f"[gate] 5 mtp: {c5['status']}", flush=True)

    c6 = (check_connector_reuse(ep, model, args, is_mock) if model else
          {"name": "connector_prefix_reuse", "status": "FAIL",
           "evidence": {"why": "no served model id"}})
    checks.append(c6)
    print(f"[gate] 6 connector-reuse: {c6['status']}", flush=True)

    failed = [c["name"] for c in checks if c["status"] == "FAIL"]
    not_exercised = [c["name"] for c in checks if c["status"] == "NOT_EXERCISED"]
    receipt = {
        "schema": SCHEMA,
        "generated_utc": utcnow(),
        "started_utc": started,
        "argv": sys.argv,
        "base_url": args.base_url,
        "endpoint_identity": {
            "served_model_id": model_id,
            "is_mock_preflight_double": is_mock,
        },
        "seed": args.seed,
        "cross_restart_note": CROSS_RESTART_NOTE,
        "metrics_before_check3": m_before,
        "metrics_after_check3": m_after,
        "checks": checks,
        "verdict": "FAIL" if failed else "PASS",
        "failed_checks": failed,
        "not_exercised_checks": not_exercised,
        "verdict_scope": (
            "PASS means every check that RAN passed. It is not a fidelity claim: with a KV "
            "connector attached, checks 1-5 all passed against a server independently "
            "measured 38/38 corrupted (receipts/lmcache-l1-2x.json). Read check 6, and if "
            "it is SKIP or NOT_EXERCISED, read the verdict as untested for prefix-reuse "
            "correctness."),
        "finished_utc": utcnow(),
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f, indent=1)
    print(f"[gate] receipt -> {args.out}", flush=True)
    print(f"[gate] verdict: {receipt['verdict']}"
          + (f" (failed: {', '.join(failed)})" if failed else "")
          + (f" (not exercised: {', '.join(not_exercised)})" if not_exercised else ""),
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
