#!/usr/bin/env python3
"""Group nsys cuda_gpu_kern_sum CSV rows into kernel families and report shares.

Round-2 analysis (receipts/nsys-round2-2026-08-19.md) grouped kernels manually.
This tool reproduces that grouping from any _cuda_gpu_kern_sum.csv so deep-context
traces can be compared against the round-2 shallow-context baseline identically.

Usage:
  python3 tools/parse-nsys-kernels.py receipts/traces/<tag>_cuda_gpu_kern_sum.csv
  python3 tools/parse-nsys-kernels.py --compare old.csv new.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# Kernel-name patterns, checked in order. First match wins.
# These mirror the manual grouping in receipts/nsys-round2-2026-08-19.md.
FAMILIES = [
    # b12x dense GEMMs — the FP4/FP6 trellis matmuls. Identified by the
    # cutlass DenseGemmKernel template with f4E2M1FN (FP4) or f6E2M3FN (FP6).
    ("b12x FP4 GEMM (5120)",      re.compile(r"DenseGemmKernel.*o5120.*f4E2M1FN")),
    ("b12x FP4 GEMM (17408)",     re.compile(r"DenseGemmKernel.*o17408.*f4E2M1FN")),
    ("b12x FP4 GEMM (6144)",      re.compile(r"DenseGemmKernel.*o6144.*f4E2M1FN")),
    ("b12x FP4 GEMM (other)",     re.compile(r"DenseGemmKernel.*f4E2M1FN")),
    ("b12x FP6 GEMM",             re.compile(r"DenseGemmKernel.*f6E2M3FN")),
    ("b12x GEMM (other)",         re.compile(r"DenseGemmKernel")),
    # EXL3 fused gemm (K6 lm_head, etc.)
    ("exl3_gemm_kernel",          re.compile(r"exl3_gemm_kernel")),
    # Attention
    ("unified_attention",         re.compile(r"unified_attention")),
    ("flash_attn\|flashinfer",    re.compile(r"flash_attn|flash_attn_varlen|flashinfer")),
    ("attention (other)",         re.compile(r"attention|attn")),
    # Activation quantization support
    ("act-quant: MaxNan/amax",    re.compile(r"MaxNan|amax")),
    ("act-quant: Bf16ToFp4",      re.compile(r"Bf16ToFp4|Fp4")),
    ("act-quant: copy/convert",   re.compile(r"CopyKernel|copy.*quant|quantize.*copy")),
    ("act-quant: div/scale",      re.compile(r"elementwise.*div|scale.*kernel|div.*kernel")),
    ("act-quant (other)",         re.compile(r"quantiz|dequantiz|cast.*type|ConvertKernel")),
    # Tensor concat (CatArrayBatchedCopy)
    ("CatArrayBatchedCopy",       re.compile(r"CatArrayBatchedCopy")),
    # GDN fused
    ("GDN fused_sigmoid_gating",  re.compile(r"fused_sigmoid_gating_delta_rule|gated_delta_rule")),
    # Silu
    ("silu act_and_mul",          re.compile(r"silu|act_and_mul")),
    # RMSNorm
    ("rms_norm",                  re.compile(r"rms_norm|RMSNorm")),
    # Sampling
    ("sampling",                  re.compile(r"sample|topk|topp|softmax.*sample")),
    # Misc
    ("embedding",                 re.compile(r"embed")),
    ("reshape/index/copy",        re.compile(r"reshape|index|copy|gather|scatter|ConcatKernel")),
    ("reduction",                 re.compile(r"reduce|Reduction")),
    ("empty/cache/launch",        re.compile(r"empty|cache|launch|synchronize|event")),
]


def classify(name: str) -> str:
    for label, pat in FAMILIES:
        if pat.search(name):
            return label
    return "uncategorised"


def parse_csv(path: Path) -> dict[str, dict]:
    """Return {family: {pct, total_ns, count}} sorted by pct descending."""
    rows = []
    with open(path, newline="") as f:
        # nsys CSVs sometimes have a header line before the actual header
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not header and row and row[0].strip() == "Time (%)":
                header = row
                continue
            if header and len(row) == len(header):
                try:
                    pct = float(row[0])
                    total_ns = float(row[1])
                    count = int(row[2])
                    name = row[-1] if len(row) > 8 else ""
                except (ValueError, IndexError):
                    continue
                rows.append((pct, total_ns, count, name))

    by_fam: dict[str, dict] = defaultdict(lambda: {"pct": 0.0, "total_ns": 0.0, "count": 0, "samples": []})
    for pct, total_ns, count, name in rows:
        fam = classify(name)
        by_fam[fam]["pct"] += pct
        by_fam[fam]["total_ns"] += total_ns
        by_fam[fam]["count"] += count
        if len(by_fam[fam]["samples"]) < 2:
            by_fam[fam]["samples"].append(name[:120])

    # merge small families
    merged = {}
    misc = {"pct": 0.0, "total_ns": 0.0, "count": 0, "samples": []}
    for fam, d in by_fam.items():
        if d["pct"] < 0.3 and fam not in ("uncategorised",):
            misc["pct"] += d["pct"]
            misc["total_ns"] += d["total_ns"]
            misc["count"] += d["count"]
        else:
            merged[fam] = d
    if misc["pct"] > 0.1:
        merged["(other <0.3% each)"] = misc

    return dict(sorted(merged.items(), key=lambda kv: kv[1]["pct"], reverse=True))


def print_table(label: str, data: dict[str, dict]) -> None:
    print(f"\n{'='*72}")
    print(f" {label}")
    print(f"{'='*72}")
    print(f" {'family':<32} {'share':>7} {'ns':>14} {'count':>8}")
    print(f" {'-'*32} {'-'*7} {'-'*14} {'-'*8}")
    total_pct = 0.0
    for fam, d in data.items():
        print(f" {fam:<32} {d['pct']:>6.1f}% {d['total_ns']:>14,.0f} {d['count']:>8,}")
        total_pct += d["pct"]
    print(f" {'-'*32} {'-'*7} {'-'*14} {'-'*8}")
    print(f" {'TOTAL':<32} {total_pct:>6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    args = ap.parse_args()

    if args.compare:
        old = parse_csv(Path(args.compare[0]))
        new = parse_csv(Path(args.compare[1]))
        print_table(args.compare[0], old)
        print_table(args.compare[1], new)
        print(f"\n{'='*72}")
        print(" DELTA (new - old)")
        print(f"{'='*72}")
        print(f" {'family':<32} {'old':>7} {'new':>7} {'delta':>8}")
        print(f" {'-'*32} {'-'*7} {'-'*7} {'-'*8}")
        allf = sorted(set(old) | set(new), key=lambda f: new.get(f, {}).get("pct", 0), reverse=True)
        for f in allf:
            o = old.get(f, {}).get("pct", 0)
            n = new.get(f, {}).get("pct", 0)
            print(f" {f:<32} {o:>6.1f}% {n:>6.1f}% {n-o:>+7.1f}%")
    elif args.csv:
        data = parse_csv(Path(args.csv))
        print_table(args.csv, data)
        print("\nUncategorised samples (for extending FAMILIES):")
        if "uncategorised" in data:
            for s in data["uncategorised"]["samples"]:
                print(f"  {s}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
