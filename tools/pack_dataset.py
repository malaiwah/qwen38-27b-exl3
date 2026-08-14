#!/usr/bin/env python3
"""Package the v3 fidelity artifacts as a reproducible HF dataset.

What goes in, and why each piece is necessary for someone else to check our
numbers without trusting us:

  tokens/                 the exact token IDs. Retokenising the source text does
                          NOT reproduce the evaluation input, so without these the
                          measurement is not reproducible at all.
  suite-manifest.json     per-context stratum, source cluster, partition, sentinel
                          flag, token SHA-256, contamination-scan result.
  reference-hidden/       BF16 hidden states after the final RMSNorm, [2047, 5120]
                          per context. This is the expensive artifact: with it,
                          anyone can score a new candidate without owning or running
                          the BF16 model.
  candidate-hidden/<id>/  the same tensors for every candidate we measured, so our
                          published numbers can be recomputed exactly, on CPU if
                          desired, with no GPU and no model downloads.
  sentinel-hidden/        two extra captures of the same runtime over the 32 sentinel
                          contexts: the measurement noise floor.
  lm-head/weight.safetensors  the BF16 head used for replay, extracted from
                          Qwen/Qwen3.8-27B (Apache-2.0). Both operands are replayed
                          through this one matrix, so it is part of the metric
                          definition.
  reports/                every report/paired/qualification JSON we published.
  checksums.txt           SHA-256 of every file above.

Deliberately NOT included: the raw source text. The token IDs freeze the input, and
the fetcher scripts plus source URLs are in the research repo, so re-deriving is
possible without us redistributing third-party corpora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SRC = Path("/var/tmp/work/kld3")
CAND = {"k4-online-k6": "hidden-k4", "unsloth-nvfp4": "hidden-nvfp4", "qwen-fp8": "hidden-fp8"}
SENT = {"repeat-02": "hidden-k4-r2", "repeat-03": "hidden-k4-r3"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def copytree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/tmp/work/dataset")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    shutil.copy2(SRC / "suite/suite-manifest.json", out / "suite-manifest.json")
    counts["tokens"] = copytree(SRC / "suite/tokens", out / "tokens")
    counts["reference-hidden"] = copytree(SRC / "hidden-bf16", out / "reference-hidden")
    for label, d in CAND.items():
        counts[f"candidate-hidden/{label}"] = copytree(SRC / d, out / "candidate-hidden" / label)
    for label, d in SENT.items():
        counts[f"sentinel-hidden/{label}"] = copytree(SRC / d, out / "sentinel-hidden" / label)

    (out / "lm-head").mkdir(exist_ok=True)
    head = Path("/var/tmp/work/kld2/lm_head.safetensors")
    if head.exists():
        shutil.copy2(head, out / "lm-head/weight.safetensors")
        counts["lm-head"] = 1

    rep = out / "reports"
    rep.mkdir(exist_ok=True)
    n = 0
    for pat in ("report-*.json", "paired-*.json", "noise-*.json", "qualification-*.json"):
        for f in sorted(SRC.glob(pat)):
            shutil.copy2(f, rep / f.name)
            n += 1
    counts["reports"] = n

    lines, total = [], 0
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "checksums.txt":
            lines.append(f"{sha256_file(f)}  {f.relative_to(out)}")
            total += f.stat().st_size
    (out / "checksums.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"files": len(lines), "gb": round(total / 1e9, 2), "counts": counts}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
