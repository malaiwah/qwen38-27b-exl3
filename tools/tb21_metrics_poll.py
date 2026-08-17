#!/usr/bin/env python3
"""Continuous /metrics poller for TB2.1 speed-run passes (docs/46).

The ladder and gate tools snapshot /metrics before+after each cell/check -
a delta, not a timeline. This tool is the timeline: it polls /metrics at a
fixed interval for the whole duration of a run (a TB pass, a ladder sweep,
anything), and appends one JSON line per sample to a JSONL file - streamed,
not buffered, so a kill loses at most one in-flight sample and the file is
valid to read while the poller is still running.

Each sample carries the same nine counter families the ladder and gate
already track (prompt/generation/draft/accepted tokens, prefix-cache
queries/hits, running/waiting/KV-usage), parsed from the Prometheus text,
so post-analysis does not need to re-parse it for the common case.
``--verbatim`` additionally stores the full raw /metrics text per sample -
off by default so a multi-hour pass stays a reasonable file size. Read-only:
this tool never changes server state and is safe to run alongside any other
traffic, including another tool's own before/after snapshots.

Usage:
    tb21_metrics_poll.py --base-url http://10.0.0.3:8000 \\
        --out receipts/tb21-metrics-1x-hyd-pass1.jsonl --interval 5
    # runs until SIGINT/SIGTERM (or --duration N seconds), then prints and
    # writes a summary receipt beside the .jsonl (same path, .summary.json)
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import urllib.error
import urllib.request

METRIC_FAMILIES = {
    "prompt_tokens": r"^vllm:prompt_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "generation_tokens": r"^vllm:generation_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "draft_tokens": r"^vllm:spec_decode_num_draft_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "accepted_tokens": r"^vllm:spec_decode_num_accepted_tokens_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_queries": r"^vllm:(?:gpu_)?prefix_cache_queries_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "prefix_cache_hits": r"^vllm:(?:gpu_)?prefix_cache_hits_total(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "num_requests_running": r"^vllm:num_requests_running(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "num_requests_waiting": r"^vllm:num_requests_waiting(\{[^}]*\})?\s+([0-9eE+.\-]+)",
    "gpu_cache_usage_perc": r"^vllm:gpu_cache_usage_perc(\{[^}]*\})?\s+([0-9eE+.\-]+)",
}


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_metrics(text: str) -> dict:
    sums: dict[str, float] = {}
    for name, pat in METRIC_FAMILIES.items():
        rx = re.compile(pat, re.MULTILINE)
        total = 0.0
        hit = False
        for m in rx.finditer(text):
            try:
                total += float(m.group(2))
                hit = True
            except ValueError:
                pass
        if hit:
            sums[name] = total
    return sums


def fetch(url: str, api_key: str | None, timeout: float) -> tuple[int, str, str | None, float]:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, body, None, time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), str(e), time.monotonic() - t0
    except Exception as e:  # noqa: BLE001 - a poller must never crash on one bad tick
        return 0, "", str(e), time.monotonic() - t0


class Stop:
    requested = False


def install_signal_handlers(stop: Stop) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        stop.requested = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def write_summary(out_jsonl: str, samples: list[dict], started: str,
                  finished: str, interval: float, base_url: str) -> str:
    ok = [s for s in samples if s["http_status"] == 200]

    # vLLM's counters are per-process: a server restart resets them to zero.
    # A stream that spans a restart (an A/B arm swap, a crash, a redeploy)
    # therefore has NO meaningful first-to-last delta - naively subtracting
    # gives a NEGATIVE token count, which is either absurd on its face or,
    # worse, quietly misread. Observed in the 2026-08-17 rehearsal, which
    # spanned three arm swaps. So: detect resets, segment into monotonic
    # runs, report per-segment deltas, and refuse the single naive delta
    # whenever a reset exists.
    MONOTONIC = ("prompt_tokens", "generation_tokens", "draft_tokens",
                 "accepted_tokens", "prefix_cache_queries", "prefix_cache_hits")
    segments: list[list[dict]] = []
    current: list[dict] = []
    for s in ok:
        if current:
            prev = current[-1]["parsed"]
            cur = s["parsed"]
            reset = any(k in prev and k in cur and cur[k] < prev[k]
                        for k in MONOTONIC)
            if reset:
                segments.append(current)
                current = []
        current.append(s)
    if current:
        segments.append(current)

    def seg_delta(seg: list[dict]) -> dict | None:
        if len(seg) < 2:
            return None
        a, b = seg[0]["parsed"], seg[-1]["parsed"]
        return {k: round(b.get(k, 0) - a.get(k, 0), 3)
                for k in METRIC_FAMILIES if k in a and k in b}

    seg_reports = []
    for i, seg in enumerate(segments):
        d = seg_delta(seg)
        rep = {
            "segment": i,
            "samples": len(seg),
            "first_utc": seg[0].get("utc"),
            "last_utc": seg[-1].get("utc"),
            "deltas": d,
        }
        if d and d.get("draft_tokens"):
            rep["mtp_acceptance"] = round(
                d.get("accepted_tokens", 0) / d["draft_tokens"], 4)
        if d and d.get("prefix_cache_queries"):
            rep["prefix_cache_hit_rate"] = round(
                d.get("prefix_cache_hits", 0) / d["prefix_cache_queries"], 4)
        seg_reports.append(rep)

    summary = {
        "schema": "qwen38-tb21-metrics-poll-summary/2",
        "base_url": base_url,
        "started_utc": started,
        "finished_utc": finished,
        "interval_s": interval,
        "samples_total": len(samples),
        "samples_ok": len(ok),
        "samples_failed": len(samples) - len(ok),
        "samples_failed_note": "failed scrapes are expected while a server is "
                               "down (arm swap, restart); they are recorded, "
                               "not discarded",
        "jsonl_path": out_jsonl,
        "counter_resets_detected": len(segments) - 1,
        "monotonic_segments": seg_reports,
        "first_ok_parsed": ok[0]["parsed"] if ok else None,
        "last_ok_parsed": ok[-1]["parsed"] if ok else None,
        "deltas_first_to_last": (
            seg_delta(ok) if len(segments) == 1 else None
        ),
        "deltas_first_to_last_note": (
            "single monotonic run: first-to-last delta is valid"
            if len(segments) == 1 else
            f"REFUSED: {len(segments) - 1} counter reset(s) detected "
            "(server restarted mid-stream), so a first-to-last delta is "
            "meaningless - use monotonic_segments instead"
        ),
    }
    summary_path = out_jsonl.rsplit(".jsonl", 1)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    return summary_path


def amain(args) -> int:
    stop = Stop()
    install_signal_handlers(stop)
    url = args.base_url.rstrip("/") + "/metrics"
    started = utcnow()
    t_start = time.monotonic()
    samples: list[dict] = []
    n = 0
    print(f"[metrics-poll] {url} every {args.interval}s -> {args.out} "
          f"(SIGINT/SIGTERM to stop)", flush=True)
    with open(args.out, "a") as f:
        while True:
            if stop.requested:
                break
            if args.duration and (time.monotonic() - t_start) >= args.duration:
                break
            status, body, error, latency = fetch(url, args.api_key, args.timeout)
            sample = {
                "utc": utcnow(),
                "elapsed_s": round(time.monotonic() - t_start, 3),
                "http_status": status,
                "latency_s": round(latency, 4),
                "error": error,
                "parsed": parse_metrics(body) if status == 200 else {},
                "raw_metrics_text": body if args.verbatim else None,
            }
            f.write(json.dumps(sample) + "\n")
            f.flush()
            samples.append(sample)
            n += 1
            if n % 12 == 0 or status != 200:
                print(f"[metrics-poll] sample {n} @ {sample['elapsed_s']}s "
                      f"status={status} parsed_families={len(sample['parsed'])}",
                      flush=True)
            for _ in range(int(args.interval * 10)):
                if stop.requested:
                    break
                time.sleep(0.1)
    finished = utcnow()
    summary_path = write_summary(args.out, samples, started, finished,
                                 args.interval, args.base_url)
    print(f"[metrics-poll] stopped: {n} samples over "
          f"{round(time.monotonic() - t_start, 1)}s -> {args.out}", flush=True)
    print(f"[metrics-poll] summary -> {summary_path}", flush=True)
    return 0


def summarize_only(path: str) -> int:
    """Rebuild a summary from an existing JSONL stream.

    The live summary is written by the SIGTERM/SIGINT handler, so an unclean
    kill (SIGKILL, or a dropped SSH transport that never forwards the signal)
    leaves the stream complete but the summary missing - observed exactly
    once, in the 2026-08-17 rehearsal. The stream is the source of truth by
    design, so a summary must always be reconstructible from it rather than
    depending on process teardown.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                # A kill mid-write can truncate the final line; the design
                # tolerates losing that one sample rather than the file.
                print(f"[metrics-poll] skipped one unparseable trailing line",
                      flush=True)
    if not samples:
        print(f"[metrics-poll] no samples in {path}", flush=True)
        return 1
    started = samples[0].get("utc", "unknown")
    finished = samples[-1].get("utc", "unknown")
    # interval inferred from the stream rather than assumed
    interval = 0.0
    if len(samples) >= 2:
        span = samples[-1].get("elapsed_s", 0) - samples[0].get("elapsed_s", 0)
        interval = round(span / max(1, len(samples) - 1), 3)
    summary_path = write_summary(path, samples, started, finished, interval,
                                 "(rebuilt from stream; base_url not recorded per-sample)")
    print(f"[metrics-poll] rebuilt summary from {len(samples)} samples "
          f"(inferred interval {interval}s) -> {summary_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", help="required unless --summarize-only")
    ap.add_argument("--out", required=True, help="JSONL path, appended to")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (default 5)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run until signalled)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="per-request timeout (default 10s - a poller must "
                         "not let one slow scrape stall the cadence)")
    ap.add_argument("--verbatim", action="store_true",
                    help="store the full raw /metrics text per sample, not "
                         "just the parsed families (larger file; off by "
                         "default so a long rehearsal/pass stays a "
                         "reasonable size - the parsed families cover the "
                         "documented counter-delta discipline)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="do not poll; rebuild the summary from an existing "
                         "--out JSONL (use after an unclean kill)")
    args = ap.parse_args()
    if args.summarize_only:
        return summarize_only(args.out)
    if not args.base_url:
        ap.error("--base-url is required unless --summarize-only")
    return amain(args)


if __name__ == "__main__":
    raise SystemExit(main())
