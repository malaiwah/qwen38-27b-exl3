#!/usr/bin/env python3
"""Extract the per-module proxy-error ladder from a conversion log.

exllamav3 prints an rmse per module at the bitrate it assigned, so one conversion yields one
point per module rather than a curve. Teeing the log is free and the points accumulate across
builds; this collects them into a single JSON keyed by (role, bits) with the distribution over
layers, which is what an error-driven allocator needs as input.

    ladder_from_log.py --log convert-ctx.log --label ctx --out ladder-ctx.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# The converter labels the number `proxy_err` for calibrated modules and `rmse` only for the
# uncalibrated fallback path (vision tower, MTP without a Hessian). Matching `rmse` alone -
# as this did until it was checked against convert-ctx.log - silently drops all 400 calibrated
# body modules and keeps only the 118 fallback ones, i.e. exactly the modules an allocator has
# no use for. The flag field between the number and `g_sc` is two characters (`o.`/`of`), not
# the literal `of`.
LINE = re.compile(
    r"Quantized:\s+(?P<key>\S+)\s+bpw:\s+(?P<bpw>[\d.]+)\s+"
    r"(?:proxy_err|rmse)\s*:\s*(?P<rmse>[\d.eE+-]+)"
    r"(?:\s+\S{2})?(?:\s+g_sc:\s*(?P<gsc>[\d.]+))?"
)


def role_of(key: str) -> str:
    if key.startswith("mtp") or ".mtp." in key:
        return "mtp"
    if "visual" in key or key.startswith("model.visual"):
        return "vision"
    if "lm_head" in key:
        return "lm_head"
    for name in ("gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                 "q_proj", "k_proj", "v_proj", "o_proj"):
        if key.endswith(name):
            return name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_module: dict[str, dict] = {}
    for line in Path(args.log).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        key = m.group("key")
        bits = round(float(m.group("bpw")))
        rmse = float(m.group("rmse"))
        buckets[(role_of(key), bits)].append(rmse)
        per_module[key] = {"bits": bits, "rmse": rmse,
                           "g_scale": float(m.group("gsc")) if m.group("gsc") else None}

    summary = {}
    for (role, bits), vals in sorted(buckets.items()):
        summary[f"{role}@K{bits}"] = {
            "modules": len(vals), "mean_rmse": statistics.fmean(vals),
            "median_rmse": statistics.median(vals),
            "min_rmse": min(vals), "max_rmse": max(vals),
        }
    out = {"schema": "qwen38-error-ladder/1", "label": args.label, "source_log": args.log,
           "note": ("One rmse per module at its assigned bitrate. A full epsilon(K) curve needs "
                    "the same module converted at several widths; these points accumulate "
                    "across builds and are the input an error-driven allocator needs."),
           "by_role_and_bits": summary, "per_module": per_module}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {"modules": v["modules"], "mean_rmse": round(v["mean_rmse"], 6)}
                      for k, v in summary.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
