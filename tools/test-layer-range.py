#!/usr/bin/env python3
"""Unit checks for the FP4/FP6 layer-index range filter in exl3.py.

Runs on the host with no GPU, no torch and no vLLM: the parser/filter block is
extracted from the patch and executed in an isolated namespace.  This exists
because the filter decides which weights get converted, and a silent off-by-one
would quietly change both memory footprint and fidelity.

    python3 tools/test-layer-range.py
"""
import logging
import re
import sys

PATCH = "/home/mbelleau/vllm-exl3-multiprecision.py"


def load_block():
    src = open(PATCH).read()
    start = src.index("def _parse_layer_range")
    end = src.index("def _layer_precision")
    tail_start = end
    tail_end = src.index("_FP4_CONVERSION_MODULE = None", tail_start)
    seg = src[start:tail_end]
    ns = {
        "os": __import__("os"),
        "re": re,
        "logger": logging.getLogger("test-layer-range"),
        "_FP4_LAYER_PATTERNS": [],
        "_FP6_LAYER_PATTERNS": ["mlp.gate_up_proj"],
    }
    exec(seg, ns)
    return ns


def main() -> int:
    ns = load_block()
    pr = ns["_parse_layer_range"]
    wr = ns["_within_layer_range"]
    li = ns["_layer_index"]
    lp = ns["_layer_precision"]
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    ck("empty spec -> None", pr("") is None and pr("   ") is None)
    ck("lo-hi parsed", pr("0-31") == (0, 31))
    ck("reversed range normalised", pr("31-0") == (0, 31))
    ck("bare index = single layer", pr("5") == (5, 5))
    ck("malformed -> None, no raise", pr("junk") is None)

    ck("index extracted", li("model.language_model.layers.42.mlp.gate_up_proj") == 42)
    ck("no index for lm_head", li("lm_head") is None)

    ck("lower bound inclusive", wr("model.language_model.layers.0.mlp.gate_up_proj", (0, 28)))
    ck("upper bound inclusive", wr("model.language_model.layers.28.mlp.gate_up_proj", (0, 28)))
    ck("just past upper excluded", not wr("model.language_model.layers.29.mlp.gate_up_proj", (0, 28)))
    ck("indexless prefix unrestricted", wr("lm_head", (0, 28)))
    ck("vision tower unrestricted", wr("visual.blocks.3.mlp.fc1", (0, 28)))

    ns["_FP6_LAYER_RANGE"] = (0, 28)
    ck("in-range gate_up -> fp6", lp("model.language_model.layers.10.mlp.gate_up_proj") == "fp6")
    ck("out-of-range gate_up -> skip", lp("model.language_model.layers.40.mlp.gate_up_proj") == "skip")
    ck("unmatched pattern -> skip", lp("model.language_model.layers.10.mlp.down_proj") == "skip")
    ns["_FP6_LAYER_RANGE"] = None
    ck("no range = every layer", lp("model.language_model.layers.40.mlp.gate_up_proj") == "fp6")

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
