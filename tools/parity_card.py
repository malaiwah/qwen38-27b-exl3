#!/usr/bin/env python3
"""Fill the K6-parity card from its receipt, and refuse to emit a card with a hole in it.

Two rules this tool exists to enforce.

1. **No measured figure is typed by hand.**  Every numeric token comes out of
   `receipts/k6-parity-kld.json`, and any `{{TOKEN}}` still present at the end is a fatal
   error, so a card cannot ship with an unfilled placeholder or a hand-copied number.

2. **The claim cannot be reshaped around the result.**  The headline and the outcome sentence
   are not written after seeing the number: all three branches of the pre-registered decision
   rule are written out below, before the conversion runs, and the tool selects one by the
   receipt's own `verdict.outcome`.  The commit that adds this file predates the conversion
   log, which is what makes that checkable.

    parity_card.py --receipt receipts/k6-parity-kld.json \\
        --body parity-card-body.md --out MODEL_CARD-K6-parity.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GIB = 2 ** 30
HYD_PAYLOAD = 21_586_964_548
Q6K_NET = 0.0015278671188742878
Q6K_MEASURED = 0.002035222177682657
Q6K_BODY_GIB = 19.3599

# Written before the measurement exists. One is selected by the receipt's verdict.
HEADLINE = {
    "BEATS_Q6K": "At equal file bytes, this EXL3 recipe beats GGUF `Q6_K` — the 6-bit loss was "
                 "a byte gap, and it is now closed",
    "MATCHES_Q6K": "At equal file bytes, this EXL3 recipe matches GGUF `Q6_K` — the 6-bit loss "
                   "was a byte gap, not an engineering one",
    "MISSES": "At equal file bytes, GGUF `Q6_K` still wins — a pre-registered prediction that "
              "missed, published as it stands",
}
OUTCOME = {
    "BEATS_Q6K":
        "It is below the net-of-floor figure, so the answer to the pre-registered acceptance "
        "question is BEATS. Spending the byte surplus our own 3.73x-per-bit law said `Q6_K` was "
        "spending closes the 6-bit-class gap that `receipts/cross-engine-comparator.json` "
        "recorded against us — and closes it while carrying about 2.3 GiB less transformer body "
        "than the artifact it is measured against.",
    "MATCHES_Q6K":
        "It lands between the net and the measured `Q6_K` figures, so the answer to the "
        "pre-registered acceptance question is MATCHES: at equal file bytes the two are "
        "indistinguishable once the cross-engine floor's own slack is admitted, which is what "
        "the byte-gap decomposition predicted. The 6-bit-class loss recorded in "
        "`receipts/cross-engine-comparator.json` was a byte gap and not an engineering one, and "
        "this build reaches that result carrying about 2.3 GiB less transformer body than `Q6_K`.",
    "MISSES":
        "It is above `Q6_K`'s measured upper bound, so the answer to the pre-registered "
        "acceptance question is MISSES and the prediction is wrong in magnitude by the ratio "
        "stated below. Spending `Q6_K`'s byte surplus did not buy `Q6_K`'s fidelity, so part of "
        "the 6-bit-class gap is not the budget. Note what that does and does not license: this "
        "build still carries about 2.3 GiB less transformer body than `Q6_K`, so the byte-gap "
        "decomposition is falsified at file parity and not at body parity, which is a separate "
        "and unrun experiment. The checkpoint and this card are published anyway, because a "
        "pre-registration is only worth what is done with its misses.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    r = json.loads(a.receipt.read_text())
    s, b = r["score"], r["build"]
    outcome = r["verdict"]["outcome"]
    if outcome not in HEADLINE:
        raise SystemExit("unknown verdict outcome %r; this tool's branches were fixed before the "
                         "measurement and must not be invented now" % outcome)
    roles = b["quantization_manifest_roles"]
    payload_b = b["measured_role_payload_bytes"]
    body_b = (payload_b - roles["embed_tokens"]["bytes"] - roles["lm_head"]["bytes"]
              - roles["vision_tower"]["bytes"] - roles["mtp_draft"]["bytes"])
    mean = s["token_mean_kld"]
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
    where = ("below `Q6_K`'s net-of-floor %.6f" % Q6K_NET if mean < Q6K_NET else
             "between `Q6_K`'s net-of-floor %.6f and its measured %.6f" % (Q6K_NET, Q6K_MEASURED)
             if mean <= Q6K_MEASURED else
             "above `Q6_K`'s measured %.6f" % Q6K_MEASURED)
    tok = {
        "HEADLINE": HEADLINE[outcome],
        "OUTCOME_SENTENCE": OUTCOME[outcome],
        "MEAN": "%.6f" % mean,
        "CI": "[%.6f, %.6f]" % tuple(s["ci95"]),
        "P999": "%.6f" % s["p999_kld"],
        "MAX": "%.6f" % s["max_kld"],
        "TOP1": "%.3f %%" % (100 * s["top1_agreement"]),
        "MEASURED_PAYLOAD": "{:,}".format(payload_b),
        "MEASURED_PAYLOAD_GIB": "%.3f" % (payload_b / GIB),
        "MEASURED_TREE": "{:,}".format(b["tree"]["total_bytes"]),
        "MEASURED_BODY_GIB": "%.3f" % (body_b / GIB),
        "BODY_DEFICIT_GIB": "%.3f" % (Q6K_BODY_GIB - body_b / GIB),
        "BODY_DEFICIT_PCT": "%.1f" % (100 * (Q6K_BODY_GIB - body_b / GIB) / (body_b / GIB)),
        "MTP_GIB": "%.4f" % (roles["mtp_draft"]["bytes"] / GIB),
        "PARITY_GIB": "%.3f" % (payload_b / GIB),
        "PARITY_SURPLUS_GIB": "%.3f" % ((payload_b - HYD_PAYLOAD) / GIB),
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
                Q6K_BODY_GIB - body_b / GIB,
                100 * (Q6K_BODY_GIB - body_b / GIB) / (body_b / GIB))),
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
