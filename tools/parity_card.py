#!/usr/bin/env python3
"""Fill the K6-parity card from its receipt, and refuse to emit a card with a hole in it.

The renderer preserves two invariants:

1. Run-specific measurements come from `receipts/k6-parity-kld.json`; byte-axis
   anchors come from `receipts/cross-candidate-byte-accounting.json`. Any
   `{{TOKEN}}` left after substitution is fatal.
2. The historical receipt remains immutable, including its pre-registered
   BEATS/MATCHES/MISSES branch. Publication text always withdraws that branch
   because it used invalid cross-engine KL subtraction, and reports only the
   observed complete-pipeline comparison.

    parity_card.py --receipt receipts/k6-parity-kld.json \
        --byte-accounting receipts/cross-candidate-byte-accounting.json \
        --body tools/parity-card-body.md --out MODEL_CARD-K6-parity.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GIB = 2 ** 30


def corrected_headline(mean: float, q6k_mean: float) -> str:
    return (
        f"Near-equal file bytes: EXL3 measures {mean:.6f} and GGUF `Q6_K` "
        f"{q6k_mean:.6f} on the same suite, with engine confounding explicit"
    )
CORRECTED_OUTCOME = (
    "The pre-registered BEATS/MATCHES/MISSES rule subtracted a cross-engine "
    "BF16 control from candidate KL. That operation is invalid: KL is neither "
    "additive nor a metric. The receipt remains immutable, but its parity "
    "verdict is withdrawn. The observable result is a complete-pipeline "
    "comparison: this EXL3 build measures lower KL than the llama.cpp `Q6_K` "
    "pipeline on these contexts, while the engine mismatch prevents a "
    "format-only or byte-gap attribution."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument(
        "--byte-accounting",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "receipts/cross-candidate-byte-accounting.json",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    r = json.loads(a.receipt.read_text())
    byte_accounting = json.loads(a.byte_accounting.read_text())
    if byte_accounting.get("schema") != "qwen38-cross-candidate-byte-accounting/1":
        raise SystemExit("unexpected byte-accounting schema")
    s, b = r["score"], r["build"]
    outcome = r["verdict"]["outcome"]
    if outcome not in {
        "BEATS_Q6K",
        "MATCHES_Q6K",
        "MISSES",
        "CROSS_ENGINE_COMPLETE_PIPELINE_ONLY",
    }:
        raise SystemExit("unknown historical verdict outcome %r" % outcome)
    roles = b["quantization_manifest_roles"]
    payload_b = b["measured_role_payload_bytes"]
    body_b = (payload_b - roles["embed_tokens"]["bytes"] - roles["lm_head"]["bytes"]
              - roles["vision_tower"]["bytes"] - roles["mtp_draft"]["bytes"])
    mean = s["token_mean_kld"]
    q6k_mean = r["paired"]["q6k"]["a_mean"]
    q6k_body_gib = byte_accounting["rows"]["GGUF Q6_K"]["body_bytes"] / GIB
    hyd_payload = byte_accounting["rows"]["hydrated K5/K6"]["tensor_bytes"]
    check = r["prediction_check"]

    def paired_sentence(row: dict, other: str) -> str:
        d = row["difference_a_minus_b"]            # a minus b; b is this build
        lo, hi = row["ci95"]
        verb = "better than" if d > 0 else "worse than"
        return ("this build is **%.6f %s** %s, paired per context: %+.6f "
                "[%+.6f, %+.6f] over %d contexts and %d source clusters, winning %d of %d"
                % (abs(d), verb, other, -d, -hi, -lo, row["contexts"], row["clusters"],
                   row["b_wins"], row["contexts"]))

    reg = check["registered_primary"]
    where = ("below `Q6_K`'s measured complete-pipeline value %.6f; the "
             "comparison is cross-engine and does not isolate format" % q6k_mean
             if mean < q6k_mean else
             "above `Q6_K`'s measured complete-pipeline value %.6f" % q6k_mean)
    tok = {
        "HEADLINE": corrected_headline(mean, q6k_mean),
        "OUTCOME_SENTENCE": CORRECTED_OUTCOME,
        "MEAN": "%.6f" % mean,
        "CI": "[%.6f, %.6f]" % tuple(s["ci95"]),
        "P999": "%.6f" % s["p999_kld"],
        "MAX": "%.6f" % s["max_kld"],
        "TOP1": "%.3f %%" % (100 * s["top1_agreement"]),
        "MEASURED_PAYLOAD": "{:,}".format(payload_b),
        "MEASURED_PAYLOAD_GIB": "%.3f" % (payload_b / GIB),
        "MEASURED_TREE": "{:,}".format(b["tree"]["total_bytes"]),
        "MEASURED_BODY_GIB": "%.3f" % (body_b / GIB),
        "BODY_DEFICIT_GIB": "%.3f" % (q6k_body_gib - body_b / GIB),
        "BODY_DEFICIT_PCT": "%.1f" % (100 * (q6k_body_gib - body_b / GIB) / (body_b / GIB)),
        "MTP_GIB": "%.4f" % (roles["mtp_draft"]["bytes"] / GIB),
        "PARITY_GIB": "%.3f" % (payload_b / GIB),
        "PARITY_SURPLUS_GIB": "%.3f" % ((payload_b - hyd_payload) / GIB),
        "CONVERT_LOG_LINES": "{:,}".format(r["toolchain"]["convert_log"]["lines"]),
        "PAIRED_HYD": paired_sentence(r["paired"]["hyd"], "hydrated"),
        "PAIRED_Q6K": paired_sentence(r["paired"]["q6k"], "GGUF `Q6_K` as measured"),
        "PREDICTION_CHECK_SENTENCE":
            ("the registered primary was %.6f, so the measurement is %.2fx the prediction and "
             "%s the registered interval [%.6f, %.6f]; it sits %s. The body deficit against "
             "`Q6_K` at this file size is %.3f GiB (%.1f %% of ours), recorded before the "
             "conversion ran."
             % (reg, mean / reg,
                "inside" if check.get("registered_interval_contains_measured") else "outside",
                check["registered_interval"][0], check["registered_interval"][1], where,
                q6k_body_gib - body_b / GIB,
                100 * (q6k_body_gib - body_b / GIB) / (body_b / GIB))),
    }

    text = a.body.read_text()
    # Drop the drafting note first: it mentions {{TOKENS}} in prose and would otherwise trip
    # the unfilled-token check below.
    text = re.sub(r"<!--\s*FRAMING-INDEPENDENT BODY.*?-->\n\n", "", text, flags=re.S)
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: tok.get(m.group(1), m.group(0)), text)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)))
    if left:
        print("REFUSING to write a card with unfilled tokens: %s" % left, file=sys.stderr)
        return 1
    a.out.write_text(text)
    print("wrote %s (%d bytes); verdict %s; mean %s\n%s"
          % (a.out, len(text), outcome, tok["MEAN"], tok["PREDICTION_CHECK_SENTENCE"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
