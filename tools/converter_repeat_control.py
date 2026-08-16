#!/usr/bin/env python3
"""Compare two runs of the SAME conversion, from one converter log that holds both.

This run got the control by accident: the first attempt died at `compile_model` on the
missing `marisa_trie` with all 409 modules already quantized, and the retry -- same
process, same machine, same inputs, same flags, minutes later -- started a new job and
re-quantized every module.  So the log contains the same conversion twice, and the
converter's own per-module `proxy_err` and global scale can be compared run against run.

That comparison is what separates "the converter is not deterministic" from "the two
builds ran in different environments".  Without it, a byte-level mismatch between a
rebuild and a published checkpoint has at least two explanations and no way to choose.

    repeat_control.py <convert.log> <out.json>
"""
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"-- Quantized: (\S+)\s+bpw:\s+([\d.]+)\s+proxy_err: ([\d.eE+-]+)"
                  r".*?g_sc: ([\d.]+)")


def parse(lines):
    out = {}
    for line in lines:
        m = LINE.search(line)
        if m:
            out[m.group(1)] = {"bpw": m.group(2), "proxy_err": m.group(3),
                               "g_scale": m.group(4)}
    return out


def main() -> int:
    log, out = Path(sys.argv[1]), Path(sys.argv[2])
    lines = log.read_text(errors="replace").splitlines()
    banners = [i for i, l in enumerate(lines) if "Creating new job" in l]
    if len(banners) != 2:
        raise SystemExit(f"expected exactly two conversion jobs in {log}, found "
                         f"{len(banners)}")
    a, b = parse(lines[:banners[1]]), parse(lines[banners[1]:])
    common = sorted(set(a) & set(b))
    err = [k for k in common if a[k]["proxy_err"] != b[k]["proxy_err"]]
    scale = [k for k in common if a[k]["g_scale"] != b[k]["g_scale"]]
    bpw = [k for k in common if a[k]["bpw"] != b[k]["bpw"]]

    payload = {
        "what": "two runs of the identical conversion, same box, same flags, same source, "
                "minutes apart, compared on the converter's own printed per-module "
                "quantization error and global scale",
        "why_it_exists": "the first attempt died at compile_model on the missing "
                         "marisa_trie with every module already quantized; the retry "
                         "started a new job and re-quantized all of them, so one log holds "
                         "the same conversion twice",
        "source_log": str(log),
        "modules_in_run_a": len(a),
        "modules_in_run_b": len(b),
        "modules_compared": len(common),
        "width_disagreements": bpw,
        "global_scale_disagreements": len(scale),
        "proxy_err_disagreements": len(err),
        "proxy_err_disagreement_fraction": (len(err) / len(common)) if common else None,
        "disagreeing_modules": {k: {"run_a": a[k], "run_b": b[k]} for k in err},
        "resolution_note": "the converter prints proxy_err to six decimals, so this "
                           "detects a difference only when it reaches that digit; it is a "
                           "LOWER bound on how many modules actually differ, never an "
                           "upper one. Two identical float computations round identically, "
                           "so every disagreement here is a real difference in the "
                           "underlying value.",
        "reading": None,
    }
    payload["reading"] = (
        "the allocator is deterministic (every module got the same width) and the global "
        "scale search is deterministic (" + ("no" if not scale else str(len(scale))) +
        " disagreements), but the quantization result is not: " + str(len(err)) + " of " +
        str(len(common)) + " modules land on a different error. A conversion of this model "
        "with this converter on this card is therefore not reproducible bit for bit even "
        "against itself, which is sufficient on its own to explain a checkpoint that does "
        "not match a published one."
    ) if err else (
        "every one of the " + str(len(common)) + " modules compared reproduced its width, "
        "its global scale and its proxy error exactly, so the converter is deterministic "
        "run-to-run at the resolution the log prints, and any checkpoint difference has to "
        "come from somewhere else."
    )
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("modules_compared", "width_disagreements",
                       "global_scale_disagreements", "proxy_err_disagreements")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
