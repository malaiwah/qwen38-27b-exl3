#!/usr/bin/env python3
"""TB2.1 synthetic concurrency ladder (docs/46 §4.3).

Sweeps concurrency x workload shape against an OpenAI-compatible endpoint and
finds the knee by the owner-fixed headroom rule: the largest concurrency where
median per-request tok/s >= 50 % of single-stream (C1) per-request tok/s for
the same shape. Also reports, per docs/46 §4.3, the eligible rung with the
maximum aggregate tok/s ("require per-request >= 50 % of single-stream, then
maximise aggregate").

Design points, all recorded in the receipt:
  - stdlib only (asyncio + a hand-rolled HTTP/1.1 client, Connection: close);
    runs on the driver VM's python3.12 with no venv.
  - Shapes: 512-token prompt / 256 out, 4k / 1k, 30k / 2k. Prompts are
    synthesized deterministically from --seed and are UNIQUE per
    (shape, conc, repeat, slot) so prefix caching cannot fake throughput.
  - Rungs: 1,2,4,8,16,32,64; 96 and 128 join automatically when --max-conc
    allows. If the knee lands on the last rung the receipt flags knee_at_cap:
    raise --max-conc (and the server's --max-num-seqs) and re-sweep - the
    server cap must never be the binding constraint during the search.
  - >= --repeats repeats per cell; medians reported.
  - Per repeat, slot 0 is a STREAMED request (SSE) whose first body byte gives
    TTFT under that cell's real concurrency; the other slots are non-streaming
    so usage.completion_tokens is authoritative. TTFT p50/p95 are computed
    over the streamed samples (n = repeats; named in the receipt).
  - /metrics (prometheus text) is snapshotted BEFORE and AFTER each cell; both
    snapshots are stored verbatim plus parsed counter deltas
    (prompt/generation/draft/accepted tokens, prefix-cache queries/hits) and
    the KV-usage gauge lines. Counters are read as deltas between named
    snapshots per the standing discipline; a missing /metrics (e.g. the mock
    preflight double) is recorded, never fatal.
  - HTTP 4xx/5xx are recorded outcomes (status + body head), never crashes.

Usage:
  tb21_ladder.py --base-url http://10.0.0.3:8000 --out receipts/tb21-ladder-1x-quant-dp1.json \
      [--api-key K] [--max-conc 64] [--repeats 3] [--shapes 512x256,4kx1k,30kx2k] \
      [--rungs 1,2,4] [--seed 46] [--request-timeout 900]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import ssl
import statistics
import sys
import time
import urllib.parse
from datetime import datetime, timezone

SCHEMA = "qwen38-tb21-ladder/1"

# (key, prompt_tokens, max_tokens) - docs/46 §4.3.
SHAPES = [
    ("512x256", 512, 256),
    ("4kx1k", 4096, 1024),
    ("30kx2k", 30720, 2048),
]
BASE_RUNGS = [1, 2, 4, 8, 16, 32, 64]
EXT_RUNGS = [96, 128]

WORDS = (
    "ledger granite harbor lattice ember quorum vessel timber orchard nickel "
    "meadow cipher walnut breeze summit copper hollow prairie anchor tundra "
    "marble falcon lantern gravel monsoon pigment thicket saffron drizzle "
    "boulder canyon iodine juniper kestrel liquorice mineral nectar obsidian "
    "paprika quartz russet sequoia tamarind umber vertex willow xenon yarrow "
    "zephyr basalt cobalt delta echo fjord glacier heron islet jetty krill"
).split()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_base(base_url: str) -> dict:
    u = urllib.parse.urlsplit(base_url.rstrip("/"))
    path = u.path
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return {
        "scheme": u.scheme or "http",
        "host": u.hostname or "127.0.0.1",
        "port": u.port or (443 if u.scheme == "https" else 80),
        "root": path,
    }


class HTTPOutcome:
    __slots__ = ("status", "reason", "headers", "body", "t_first_body", "error")

    def __init__(self):
        self.status = 0
        self.reason = ""
        self.headers = {}
        self.body = b""
        self.t_first_body = None  # perf_counter of first body byte
        self.error = None


async def _read_body(reader, headers, sink):
    te = headers.get("transfer-encoding", "").lower()
    if "chunked" in te:
        while True:
            line = await reader.readuntil(b"\r\n")
            size = int(line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                # trailers until blank line
                while True:
                    t = await reader.readuntil(b"\r\n")
                    if t in (b"\r\n", b""):
                        return
            data = await reader.readexactly(size + 2)
            sink(data[:-2])
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            piece = await reader.read(min(65536, remaining))
            if not piece:
                return
            remaining -= len(piece)
            sink(piece)
    else:
        while True:
            piece = await reader.read(65536)
            if not piece:
                return
            sink(piece)


async def http_call(
    ep: dict,
    method: str,
    path: str,
    payload=None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> HTTPOutcome:
    """One HTTP/1.1 exchange, Connection: close. Never raises; errors land in
    outcome.error. t_first_body is the perf_counter of the first body byte
    (== TTFT for an SSE stream, since the server flushes per token)."""
    out = HTTPOutcome()
    body = b"" if payload is None else json.dumps(payload).encode()
    lines = [
        f"{method} {ep['root']}{path} HTTP/1.1",
        f"Host: {ep['host']}:{ep['port']}",
        "Connection: close",
        "Accept: */*",
        "User-Agent: tb21-ladder/1",
    ]
    if api_key:
        lines.append(f"Authorization: Bearer {api_key}")
    if payload is not None:
        lines.append("Content-Type: application/json")
        lines.append(f"Content-Length: {len(body)}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    async def _inner():
        sslctx = None
        if ep["scheme"] == "https":
            sslctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            ep["host"], ep["port"], ssl=sslctx, limit=2**20
        )
        try:
            writer.write(raw)
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            head_lines = head.decode("latin-1").split("\r\n")
            parts = head_lines[0].split(" ", 2)
            out.status = int(parts[1])
            out.reason = parts[2] if len(parts) > 2 else ""
            for hl in head_lines[1:]:
                if ":" in hl:
                    k, v = hl.split(":", 1)
                    out.headers[k.strip().lower()] = v.strip()
            chunks = []

            def sink(piece: bytes):
                if piece and out.t_first_body is None:
                    out.t_first_body = time.perf_counter()
                chunks.append(piece)

            await _read_body(reader, out.headers, sink)
            out.body = b"".join(chunks)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    try:
        await asyncio.wait_for(_inner(), timeout=timeout)
    except Exception as exc:  # timeout, refused, protocol - all recorded
        out.error = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------- prompts


def synth_words(rng: random.Random, n_chars: int) -> str:
    out = []
    total = 0
    while total < n_chars:
        w = rng.choice(WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)[:n_chars]


def mk_prompt(seed: int, shape: str, conc: int, rep: int, slot: int,
              target_tokens: int, cpt: float) -> str:
    tag = f"tb21-ladder seed={seed} cell={shape}/c{conc}/r{rep}/s{slot}"
    header = (
        f"[{tag}] You are a text continuation engine. Continue the following "
        "text in the same style, at length, without stopping early.\n\n"
    )
    rng = random.Random(f"{seed}:{shape}:{conc}:{rep}:{slot}")
    n_chars = max(64, int(target_tokens * cpt) - len(header))
    return header + synth_words(rng, n_chars)


async def calibrate_cpt(ep, model, api_key, timeout) -> tuple[float, str]:
    """chars-per-token: /tokenize when the server has it, else 4.0 heuristic."""
    sample = synth_words(random.Random("tb21-cpt-sample"), 8000)
    out = await http_call(
        ep, "POST", "/tokenize", {"model": model, "prompt": sample},
        api_key=api_key, timeout=timeout,
    )
    if out.status == 200 and not out.error:
        try:
            count = int(json.loads(out.body)["count"])
            if count > 0:
                return len(sample) / count, (
                    f"/tokenize calibration: {len(sample)} chars -> {count} tokens"
                )
        except Exception:
            pass
    return 4.0, "chars/4 heuristic (/tokenize unavailable on this endpoint)"


# ---------------------------------------------------------------- metrics

METRIC_FAMILIES = {
    "prompt_tokens": r"^vllm:prompt_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "generation_tokens": r"^vllm:generation_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "draft_tokens": r"^vllm:spec_decode_num_draft_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "accepted_tokens": r"^vllm:spec_decode_num_accepted_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_queries": r"^vllm:(?:gpu_)?prefix_cache_queries_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_hits": r"^vllm:(?:gpu_)?prefix_cache_hits_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
}
KV_USAGE_RE = re.compile(
    r"^vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)(\{[^}]*\})?\s+", re.M
)


def parse_metric_sums(text: str) -> dict:
    sums = {}
    for fam, pat in METRIC_FAMILIES.items():
        vals = [float(m.group(2)) for m in re.finditer(pat, text, re.M)]
        if vals:
            sums[fam] = sum(vals)
    return sums


def kv_usage_lines(text: str) -> list[str]:
    return [text[m.start():text.index("\n", m.start())].strip()
            for m in KV_USAGE_RE.finditer(text)]


async def get_metrics(ep, api_key, timeout) -> dict:
    out = await http_call(ep, "GET", "/metrics", api_key=api_key, timeout=timeout)
    if out.error or out.status != 200:
        return {
            "available": False,
            "reason": out.error or f"HTTP {out.status} {out.reason}",
            "utc": utcnow(),
        }
    text = out.body.decode("utf-8", "replace")
    return {
        "available": True,
        "utc": utcnow(),
        "verbatim": text,
        "sums": parse_metric_sums(text),
        "kv_usage_lines": kv_usage_lines(text),
    }


def metric_deltas(before: dict, after: dict) -> dict:
    if not (before.get("available") and after.get("available")):
        return {"available": False,
                "reason": "one or both snapshots unavailable"}
    d = {"available": True}
    fams = set(before["sums"]) | set(after["sums"])
    for fam in sorted(fams):
        b = before["sums"].get(fam)
        a = after["sums"].get(fam)
        d[fam] = None if (b is None or a is None) else a - b
    d["kv_usage_before"] = before.get("kv_usage_lines", [])
    d["kv_usage_after"] = after.get("kv_usage_lines", [])
    return d


# ---------------------------------------------------------------- requests


def parse_sse_events(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except ValueError:
            pass
    return events


async def one_request(ep, model, prompt, max_tokens, streamed, args,
                      t_cell0: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # vLLM extra param; mock/others ignore it
    }
    if streamed:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    out = await http_call(ep, "POST", "/v1/chat/completions", payload,
                          api_key=args.api_key, timeout=args.request_timeout)
    t1 = time.perf_counter()
    row = {
        "mode": "stream" if streamed else "plain",
        "start_offset_s": round(t0 - t_cell0, 4),
        "duration_s": round(t1 - t0, 4),
        "http_status": out.status,
    }
    if out.error:
        row.update(ok=False, outcome="transport-error", error=out.error)
        return row
    if out.status >= 400:
        row.update(ok=False, outcome=f"http-{out.status}",
                   body_head=out.body[:300].decode("utf-8", "replace"))
        return row
    text = out.body.decode("utf-8", "replace")
    ctype = out.headers.get("content-type", "")
    usage = None
    finish = None
    if streamed and "text/event-stream" in ctype:
        events = parse_sse_events(text)
        content_chunks = 0
        for ev in events:
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    content_chunks += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if ev.get("usage"):
                usage = ev["usage"]
        row["ttft_s"] = (round(out.t_first_body - t0, 4)
                         if out.t_first_body else None)
        row["ttft_method"] = "first SSE body byte"
        if usage is None:
            usage = {"completion_tokens": content_chunks, "prompt_tokens": None}
            row["token_count_method"] = "SSE content-chunk count (no usage in stream)"
    else:
        try:
            obj = json.loads(text)
        except ValueError:
            row.update(ok=False, outcome="unparseable-body",
                       body_head=text[:300])
            return row
        usage = obj.get("usage") or {}
        ch = (obj.get("choices") or [{}])[0]
        finish = ch.get("finish_reason")
        if streamed:
            # server ignored stream=true (the mock does): e2e latency stands in
            row["ttft_s"] = round(t1 - t0, 4)
            row["ttft_method"] = ("server returned non-streamed response; "
                                  "ttft == e2e latency (recorded, not comparable)")
    ctok = usage.get("completion_tokens") or 0
    row.update(
        ok=True,
        outcome="completed",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=ctok,
        finish_reason=finish,
        per_request_tok_s=round(ctok / (t1 - t0), 3) if t1 > t0 and ctok else 0.0,
    )
    return row


def median(vals):
    return round(statistics.median(vals), 3) if vals else None


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, math.ceil(p / 100.0 * len(s)) - 1)
    return round(s[k], 4)


async def run_cell(ep, model, shape_key, ptok, mtok, conc, args, cpt) -> dict:
    before = await get_metrics(ep, args.api_key, args.request_timeout)
    repeats = []
    for rep in range(args.repeats):
        prompts = [
            mk_prompt(args.seed, shape_key, conc, rep, slot, ptok, cpt)
            for slot in range(conc)
        ]
        t_cell0 = time.perf_counter()
        rows = await asyncio.gather(*[
            one_request(ep, model, prompts[slot], mtok,
                        streamed=(slot == 0), args=args, t_cell0=t_cell0)
            for slot in range(conc)
        ])
        wall = time.perf_counter() - t_cell0
        ok_rows = [r for r in rows if r.get("ok")]
        agg = sum(r["completion_tokens"] or 0 for r in ok_rows) / wall if wall > 0 else 0.0
        repeats.append({
            "wall_s": round(wall, 4),
            "aggregate_tok_s": round(agg, 3),
            "ok": len(ok_rows),
            "refused_or_errored": len(rows) - len(ok_rows),
            "requests": rows,
        })
    after = await get_metrics(ep, args.api_key, args.request_timeout)

    all_ok = [r for rep in repeats for r in rep["requests"] if r.get("ok")]
    per_req = [r["per_request_tok_s"] for r in all_ok if r.get("per_request_tok_s")]
    ttfts = [r["ttft_s"] for rep in repeats for r in rep["requests"]
             if r.get("ttft_s") is not None]
    cell = {
        "shape": shape_key,
        "prompt_tokens_target": ptok,
        "max_tokens": mtok,
        "concurrency": conc,
        "repeats": repeats,
        "median_aggregate_tok_s": median([r["aggregate_tok_s"] for r in repeats]),
        "median_per_request_tok_s": median(per_req),
        "ttft_p50_s": pctl(ttfts, 50),
        "ttft_p95_s": pctl(ttfts, 95),
        "ttft_samples_n": len(ttfts),
        "ok_requests": len(all_ok),
        "refused_or_errored": sum(rep["refused_or_errored"] for rep in repeats),
        "metrics_before": before,
        "metrics_after": after,
        "metric_deltas": metric_deltas(before, after),
    }
    return cell


def compute_knee(cells: list[dict]) -> dict:
    """Owner rule: largest conc with per-request >= 50 % of single-stream;
    docs/46 variant also reported: among eligible, max aggregate."""
    by_conc = {c["concurrency"]: c for c in cells}
    c1 = by_conc.get(1)
    single = c1["median_per_request_tok_s"] if c1 else None
    if not single:
        return {"computable": False,
                "reason": "no successful single-stream (C1) measurement"}
    eligible = [
        c for c in cells
        if c["median_per_request_tok_s"] is not None
        and c["ok_requests"] > 0
        and c["median_per_request_tok_s"] >= 0.5 * single
    ]
    if not eligible:
        return {"computable": False, "single_stream_tok_s": single,
                "reason": "no rung met the 50 % rule"}
    largest = max(eligible, key=lambda c: c["concurrency"])
    max_agg = max(eligible, key=lambda c: (c["median_aggregate_tok_s"] or 0,
                                           c["concurrency"]))
    top_rung = max(c["concurrency"] for c in cells)
    return {
        "computable": True,
        "rule": "largest concurrency with median per-request tok/s >= 50 % of C1; "
                "max-aggregate-among-eligible also reported (docs/46 §4.3)",
        "single_stream_per_request_tok_s": single,
        "knee_concurrency": largest["concurrency"],
        "knee_per_request_tok_s": largest["median_per_request_tok_s"],
        "knee_aggregate_tok_s": largest["median_aggregate_tok_s"],
        "knee_max_aggregate_concurrency": max_agg["concurrency"],
        "knee_at_cap": largest["concurrency"] == top_rung,
        "knee_at_cap_advice": (
            "knee landed on the top rung: raise --max-conc (and the server's "
            "--max-num-seqs) and re-sweep - the cap must never bind the search"
            if largest["concurrency"] == top_rung else None
        ),
    }


async def amain(args) -> int:
    ep = parse_base(args.base_url)
    started = utcnow()

    models_out = await http_call(ep, "GET", "/v1/models",
                                 api_key=args.api_key, timeout=30)
    if models_out.error or models_out.status != 200:
        print(f"FATAL: /v1/models unreachable: "
              f"{models_out.error or models_out.status}", file=sys.stderr)
        return 2
    models_obj = json.loads(models_out.body)
    served = models_obj["data"][0]
    model_id = args.model or served["id"]

    cpt, cpt_method = await calibrate_cpt(ep, model_id, args.api_key, 60)
    print(f"[ladder] endpoint={args.base_url} model={model_id} "
          f"cpt={cpt:.3f} ({cpt_method})", flush=True)

    rungs = ([int(x) for x in args.rungs.split(",")] if args.rungs else
             [r for r in BASE_RUNGS + EXT_RUNGS if r <= args.max_conc])
    shape_keys = args.shapes.split(",")
    shapes = [s for s in SHAPES if s[0] in shape_keys]
    if len(shapes) != len(shape_keys):
        print(f"FATAL: unknown shape in --shapes {args.shapes}; "
              f"known: {[s[0] for s in SHAPES]}", file=sys.stderr)
        return 2

    tables = {}
    for shape_key, ptok, mtok in shapes:
        cells = []
        for conc in rungs:
            print(f"[ladder] cell {shape_key} C{conc} "
                  f"({args.repeats} repeats)...", flush=True)
            cell = await run_cell(ep, model_id, shape_key, ptok, mtok,
                                  conc, args, cpt)
            cells.append(cell)
            print(f"[ladder]   agg={cell['median_aggregate_tok_s']} tok/s  "
                  f"per-req={cell['median_per_request_tok_s']} tok/s  "
                  f"ttft_p50={cell['ttft_p50_s']}s  "
                  f"refused={cell['refused_or_errored']}", flush=True)
        tables[shape_key] = {"cells": cells, "knee": compute_knee(cells)}

    receipt = {
        "schema": SCHEMA,
        "generated_utc": utcnow(),
        "started_utc": started,
        "argv": sys.argv,
        "base_url": args.base_url,
        "endpoint_identity": {
            "served_model_id": served["id"],
            "owned_by": served.get("owned_by"),
            "models_payload": models_obj,
        },
        "seed": args.seed,
        "prompt_synthesis": {
            "method": "seeded word filler, unique per (shape,conc,repeat,slot) "
                      "so prefix caching cannot inflate throughput",
            "chars_per_token": round(cpt, 4),
            "chars_per_token_method": cpt_method,
        },
        "request_params": {
            "temperature": 0.0,
            "ignore_eos": True,
            "note": "ignore_eos is a vLLM extra body param so every request "
                    "decodes the full max_tokens; endpoints without it may "
                    "stop at EOS (finish_reason recorded per request)",
        },
        "rungs": rungs,
        "repeats": args.repeats,
        "shapes": [{"key": k, "prompt_tokens": p, "max_tokens": m}
                   for k, p, m in shapes],
        "knee_rule": "largest concurrency where median per-request tok/s >= "
                     "50 % of single-stream (owner-fixed); "
                     "max-aggregate-among-eligible reported alongside",
        "ttft_note": "TTFT from one streamed request per repeat running INSIDE "
                     "the cell's concurrent batch (slot 0); p50/p95 over "
                     "ttft_samples_n samples",
        "metrics_note": "/metrics snapshotted before+after each cell, both "
                        "verbatim; deltas per the counter-delta discipline "
                        "(docs/46 §3)",
        "tables": tables,
        "finished_utc": utcnow(),
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f, indent=1)
    print(f"[ladder] receipt -> {args.out}", flush=True)

    for shape_key, t in tables.items():
        knee = t["knee"]
        if knee.get("computable"):
            print(f"[ladder] {shape_key}: knee C{knee['knee_concurrency']} "
                  f"(per-req {knee['knee_per_request_tok_s']} tok/s, "
                  f"agg {knee['knee_aggregate_tok_s']} tok/s, "
                  f"single {knee['single_stream_per_request_tok_s']} tok/s)"
                  + ("  [AT CAP - extend sweep]" if knee["knee_at_cap"] else ""),
                  flush=True)
        else:
            print(f"[ladder] {shape_key}: knee not computable: "
                  f"{knee.get('reason')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True,
                    help="http://HOST:8000 (with or without /v1)")
    ap.add_argument("--out", required=True, help="receipt JSON path")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None,
                    help="served model id (default: first from /v1/models)")
    ap.add_argument("--max-conc", type=int, default=64,
                    help="top rung; 96/128 join when raised (default 64)")
    ap.add_argument("--rungs", default=None,
                    help="explicit comma list overriding the default ladder")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shapes", default="512x256,4kx1k,30kx2k")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
