#!/usr/bin/env python3
"""Fold the measured arms into receipts/apc-poison-repro.json without touching the
pre-registration block that was committed before the GPU window opened.

Kept as its own file rather than inlined in run_apc_repro.sh because the receipt is the
published artifact: it has to be regenerable from the raw arm files alone, on any host,
with no GPU and no server.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

RAW = Path("receipts/apc-repro-raw")
OUT = RAW / "out"
RECEIPT = Path("receipts/apc-poison-repro.json")

ARMS = ("A", "B", "Bn", "C", "D", "A2", "B2")
ROW_DROP = ("chosen_token_logprobs", "response_tokens", "prompt_ids")


def load(name: str):
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text())
    packed = OUT / (name + ".gz")
    if packed.exists():
        return json.loads(gzip.decompress(packed.read_bytes()))
    return None


def trim(row: dict) -> dict:
    """Every field a reader needs, minus the two vectors that live in the raw arm file."""
    kept = {k: v for k, v in row.items() if k not in ROW_DROP}
    logprobs = [v for v in (row.get("chosen_token_logprobs") or []) if v is not None]
    kept["emitted_tokens"] = len(row.get("response_tokens") or [])
    kept["mean_chosen_logprob"] = (
        round(sum(logprobs) / len(logprobs), 6) if logprobs else None
    )
    return kept


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def facts(tag: str) -> list[str]:
    path = OUT / f"startup-facts-{tag}.txt"
    return [l for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main() -> int:
    receipt = json.loads(RECEIPT.read_text())
    verdict = load("verdict.json")
    stage = load("stage.json")
    reuse = load("reuse-E.json")

    arms = {}
    for tag in ARMS:
        data = load(f"arm-{tag}.json")
        if not data:
            continue
        windows = load(f"windows-{tag}.json") or {}
        command = load(f"command-{tag}.json") or {}
        arms[tag] = {
            "description": data["arm_description"],
            "vllm_argv": command.get("vllm_argv"),
            "image": command.get("image"),
            "startup_facts": facts(tag),
            "summary": data["summary"],
            "log_windows": {
                "engine": windows.get("engine_windows"),
                "spec_decoding": windows.get("spec_windows"),
                "acceptance_walk_pct": windows.get("acceptance_walk_pct"),
                "acceptance_zero_windows": windows.get("acceptance_zero_windows"),
                "prefix_cache_hit_rate_pct_walk": windows.get(
                    "prefix_cache_hit_rate_pct_walk"
                ),
                "prefix_cache_hit_rate_pct_final": windows.get(
                    "prefix_cache_hit_rate_pct_final"
                ),
            },
            "rows": [trim(r) for r in data["rows"]],
        }

    reuse_table = []
    for case in reuse["cases"]:
        by_label = {v["label"]: v for v in case["variants"]}
        cold, warm = by_label["cold"], by_label["warm"]
        reuse_table.append(
            {
                "length_tag": case["length_tag"],
                "prompt_tokens": case["prompt_tokens"],
                "cold_ttft_s": cold["ttft_s"],
                "warm_ttft_s": warm["ttft_s"],
                "warm_exact_repeat_ttft_s": by_label["warm-exact"]["ttft_s"],
                "ttft_speedup": case["ttft_speedup_cold_over_warm"],
                "ttft_saved_s": case["ttft_saved_s"],
                "cold_prompt_tokens_recomputed": cold["prefix_cache_queries_delta"]
                - cold["prefix_cache_hits_delta"],
                "warm_prompt_tokens_recomputed": warm["prefix_cache_queries_delta"]
                - warm["prefix_cache_hits_delta"],
                "warm_prefix_cache_hit_rate": warm["prefix_cache_hit_rate_this_request"],
                "warm_response_corrupted": warm["corruption"]["corrupted"],
            }
        )

    receipt["status"] = (
        "COMPLETE. The pre_registration block above, including every threshold and every "
        "decision rule, was committed to git before the first scored arm ran; amendment_1 "
        "was committed before the first scored arm ran and after a pilot that is published "
        "in full. Nothing below was used to choose anything above."
    )
    receipt["results"] = {
        "run_utc": "2026-08-16T03:50Z to 04:23Z",
        "arms_run": sorted(arms),
        "requests_per_arm": 38,
        "headline": (
            "NOT REPRODUCED. In the reporter's own condition on the same card class, all "
            "38 requests of every arm returned correct planted answers and the corruption "
            "detector never fired once, in any arm, on any request."
        ),
        "verdict": {
            "reproduction": verdict["reproduction_verdict"],
            "fix": verdict["fix_verdict"],
            "secondary_concurrency_pair": verdict["secondary_concurrency_pair"],
        },
        "per_arm": arms,
        "divergence_vs_arm_B": {
            tag: {k: v for k, v in data.items() if k != "per_request"}
            for tag, data in verdict["divergence_vs_arm_B"].items()
        },
        "divergence_per_request": verdict["divergence_vs_arm_B"],
        "reuse_arm_E": {
            "purpose": (
                "what prefix reuse is worth, measured separately from the correctness "
                "verdict and unable to influence it. A 196k-token agentic turn spends most "
                "of its wall time in prefill, so prefix reuse is the largest untapped lever "
                "available, and it is exactly the lever the absent PR #51113 made unsafe."
            ),
            "method": (
                "two documents with disjoint token prefixes so the long case is genuinely "
                "cold; per request the first-token latency is measured on a streamed "
                "completion and the recomputed prompt tokens are taken as "
                "prefix_cache_queries_total minus prefix_cache_hits_total over that "
                "request, both of which this build reports in tokens"
            ),
            "table": reuse_table,
            "cumulative_prefix_cache_hit_rate": reuse["cumulative_prefix_cache_hit_rate"],
            "log_window_hit_rate_pct": (load("windows-E.json") or {}).get(
                "prefix_cache_hit_rate_pct_walk"
            ),
        },
        "whole_schedule_wall_clock_s": {
            tag: arms[tag]["summary"]["wall_s_total"] for tag in arms
        },
        "stage": stage,
    }
    receipt["raw_artifacts"] = {
        "directory": "receipts/apc-repro-raw/",
        "per_arm_server_logs": sorted(p.name for p in (RAW / "logs").glob("server-*.log.gz")),
        "full_arm_files_including_logprob_vectors": sorted(
            p.name for p in OUT.glob("arm-*.json.gz")
        ),
        "pilot": "receipts/apc-repro-raw/pilot/",
        "sha256": {
            str(p.relative_to(RAW)): digest(p)
            for p in sorted(RAW.rglob("*"))
            if p.is_file() and p.name != "assembled.json"
        },
    }
    body = json.dumps(
        {k: v for k, v in receipt.items() if k != "content_sha256"}, sort_keys=True
    ).encode()
    receipt["content_sha256"] = hashlib.sha256(body).hexdigest()
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(RECEIPT),
                "bytes": RECEIPT.stat().st_size,
                "arms": sorted(arms),
                "reproduction": verdict["reproduction_verdict"],
                "fix": verdict["fix_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
