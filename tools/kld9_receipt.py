#!/usr/bin/env python3
"""Turn one pre-registered conversion + its shard-0 score into its receipt.

Three conditions were pre-registered in `receipts/preregistration-kld9-window.json`
before any of them was converted, each with its predicted bytes, its predicted mean
KLD and a decision rule stated as thresholds:

  * `s16v`     -- S16-V, the first sub-4-bit width ever converted in this family;
                  docs/34's blocking item for the 16 GB class.
  * `altcal`   -- the published hydrated recipe at identical widths and byte sizes
                  with a deliberately different calibration corpus.
  * `k6parity` -- hydrated with MLP gate/up promoted K5->K6, at byte parity with
                  GGUF `Q6_K`.

This tool never invents a threshold.  Every number it compares against is read out
of the committed pre-registration, and it asserts that each threshold it uses is
literally present in that file's own decision-rule prose before it applies it -- so
a receipt cannot quietly move the goalposts after seeing the score.  Fidelity
numbers are copied from `tools/fidelity.py` reports and never recomputed here.

Realised widths are read back out of the conversion log rather than assumed: the
per-module `bpw` and `proxy_err` the converter printed are aggregated per role, and
for `s16v` they are additionally compared against `receipts/error-driven-ladder.json`
-- whose K3 rung is missing for every module below the 52M big-module threshold, and
absent entirely from any KLD measurement until this conversion.

    kld9_receipt.py --cond k6parity --tree /var/tmp/work/kld9/qwen38-k6parity \\
        --report /var/tmp/work/kld9/reports/report-k6parity.json \\
        --convert-log /var/tmp/work/kld9/convert-k6parity.log \\
        --build-script /var/tmp/work/kld9/build_k6parity.sh \\
        --paired hyd=/var/tmp/work/kld9/reports/paired-hyd-vs-k6parity.json \\
        --paired q6k=/var/tmp/work/kld9/reports/paired-q6k-vs-k6parity.json \\
        --out receipts/k6-parity-kld.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity  # noqa: E402  -- for bootstrap(), so intervals here match the harness exactly

REPO = Path(__file__).resolve().parent.parent
PREREG = REPO / "receipts/preregistration-kld9-window.json"
GIB = 2 ** 30

REPORT_ROW = ("token_mean_kld", "token_median_kld", "p95_kld", "p99_kld", "p999_kld",
              "max_kld", "mean_jsd_bits", "top1_agreement", "contexts",
              "scored_positions", "suite_token_sha256", "head_sha256", "filter")

# Which pre-registered condition, and where its answer belongs.
CONDS = {
    "s16v": {
        "index": 0,
        "schema": "qwen38-sixteen-flip-kld/1",
        "question": "what does the first sub-4-bit width in this family measure on shard 0 of "
                    "the v5 suite, and does sub-4-bit fidelity justify a 16 GB SKU?",
        "roles_expected": {"full_attn": 4, "linear_attn": 3, "mlp_gate": 3, "mlp_up": 3,
                           "mlp_down": 3},
        # docs/34 6.1 pins the head at K4 and the whole MTP draft at K4; the override regex is
        # deliberately body-only so -mb 4 governs the draft. Checked, not assumed.
        "head_mtp_expected": {"lm_head": 4.0, "mtp": 4.0},
    },
    "altcal": {
        "index": 1,
        "schema": "qwen38-calibration-corpus-kld/1",
        "question": "at identical widths and identical byte sizes, does the CONTENT of the "
                    "calibration corpus change measured fidelity?",
        "roles_expected": {"full_attn": 6, "linear_attn": 6, "mlp_gate": 5, "mlp_up": 5,
                           "mlp_down": 6},
        "head_mtp_expected": {"lm_head": 6.0},
    },
    "k6parity": {
        "index": 2,
        "schema": "qwen38-k6-parity-kld/1",
        "question": "at byte parity with GGUF Q6_K, does our recipe match or beat it -- i.e. is "
                    "the 6-bit-class loss a byte gap rather than an engineering gap?",
        "roles_expected": {"full_attn": 6, "linear_attn": 6, "mlp_gate": 6, "mlp_up": 6,
                           "mlp_down": 6},
        "head_mtp_expected": {"lm_head": 6.0},
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(("git",) + a, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def role_of(name: str) -> str:
    for key, r in (("mlp.gate_proj", "mlp_gate"), ("mlp.up_proj", "mlp_up"),
                   ("mlp.down_proj", "mlp_down"), ("linear_attn", "linear_attn"),
                   ("self_attn", "full_attn"), ("lm_head", "lm_head")):
        if key in name:
            return r
    return "other"


QLINE = re.compile(
    r"^ -- Quantized: (\S+)\s+bpw:\s+([0-9.]+)\s+"
    r"(?:proxy_err:\s+([0-9.eE+-]+)|rmse\s+:\s+([0-9.eE+-]+))")


def parse_convert_log(p: Path) -> dict:
    """Per-module realised bpw and proxy error, aggregated per role."""
    mods = {}
    for line in p.read_text(errors="replace").splitlines():
        m = QLINE.match(line)
        if not m:
            continue
        name, bpw, proxy, rmse = m.group(1), float(m.group(2)), m.group(3), m.group(4)
        if name.endswith(("norm", ".0")) and proxy is None and rmse is None:
            continue
        mods[name] = {"bpw": bpw,
                      "proxy_err": float(proxy) if proxy is not None else None,
                      "rmse": float(rmse) if rmse is not None else None}
    body = {n: v for n, v in mods.items()
            if n.startswith("model.language_model.layers") and v["proxy_err"] is not None}
    per_role = {}
    for n, v in body.items():
        per_role.setdefault(role_of(n), []).append(v)
    def stats(vs):
        e = sorted(v["proxy_err"] for v in vs)
        b = sorted({v["bpw"] for v in vs})
        return {"modules": len(vs), "bpw_values": b,
                "proxy_err_min": e[0], "proxy_err_median": e[len(e) // 2],
                "proxy_err_max": e[-1],
                "proxy_err_mean": sum(e) / len(e)}
    return {
        "modules_quantized": len(mods),
        "body_modules_with_proxy_err": len(body),
        "per_role": {r: stats(vs) for r, vs in sorted(per_role.items())},
        "head_and_mtp": {n: v for n, v in mods.items()
                         if not n.startswith("model.language_model.layers")},
        "per_module": {n: v for n, v in sorted(mods.items())},
    }


def ladder_check(realised: dict) -> dict:
    """S16-V only: the conversion log's realised K3 proxy errors against the ladder.

    `receipts/error-driven-ladder.json` laddered 400 body modules over five widths, but
    the rung set depends on size: modules below 52M parameters were laddered K4..K8 and
    have NO measured K3 point.  This conversion produces one for every module.
    """
    lad = json.loads((REPO / "receipts/error-driven-ladder.json").read_text())["modules"]
    have, extrap, out = [], [], {}
    for n, v in realised["per_module"].items():
        if not n.startswith("model.language_model.layers") or v["proxy_err"] is None:
            continue
        m = lad.get(n)
        if m is None:
            continue
        rung = m["ladder"].get(str(int(v["bpw"])))
        if rung is not None:
            have.append((n, v["proxy_err"], rung))
        elif int(v["bpw"]) == 3 and m["ladder"].get("4") and m["ladder"].get("5"):
            e4, e5 = m["ladder"]["4"], m["ladder"]["5"]
            extrap.append((n, v["proxy_err"], e4 * (e4 / e5)))
    def ratios(rows):
        if not rows:
            return None
        r = sorted(a / b for _, a, b in rows)
        return {"n": len(r), "min": r[0], "median": r[len(r) // 2], "max": r[-1],
                "mean": sum(r) / len(r)}
    return {
        "what": "the converter's own realised proxy error at the width it actually used, "
                "against the laddered value for the same module at the same width",
        "measured_rung_available": ratios(have),
        "k3_rung_extrapolated_by_the_ladder_law": ratios(extrap),
        "k3_note": "for modules below the ladder's 52M big-module threshold the K3 rung was "
                   "never measured; the comparison uses this run's own extrapolation "
                   "eps(3) = eps(4) * eps(4)/eps(5), the same one the pre-registration used, "
                   "so a ratio near 1.0 is the first direct evidence that the 3.73x-per-bit "
                   "law holds one rung below where it was fitted",
    }


MODEL_LOAD = re.compile(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds")


def resident_weights(capture_log: Path, serialized_payload: int | None) -> dict | None:
    """vLLM's own model-loading footprint, read out of the capture log.

    docs/34 models resident weights as `payload + 0.35 GiB loader allowance` and labels the
    0.35 a prediction.  Every capture prints the real number, so each conversion scored here
    also measures that term for its own build -- which for S16-V is the other half of the
    16 GB arithmetic, and the only part of it that a 32 GB card can measure.
    """
    if capture_log is None or not capture_log.exists():
        return None
    m = None
    for line in capture_log.read_text(errors="replace").splitlines():
        m = MODEL_LOAD.search(line) or m
        if m:
            break
    if not m:
        return None
    gib = float(m.group(1))
    out = {"model_loading_gib": gib, "model_loading_seconds": float(m.group(2)),
           "source": "vLLM gpu_model_runner, quoted from the capture log",
           "embedding_form": "BF16 on disk and BF16 resident -- the fidelity protocol runs no "
                             "int8 embedding overlay, so this is the un-overlaid figure"}
    if serialized_payload:
        out["serialized_payload_gib"] = round(serialized_payload / GIB, 4)
        out["loader_delta_gib"] = round(gib - serialized_payload / GIB, 4)
        out["loader_delta_note"] = ("measured counterpart of docs/34's +0.35 GiB loader "
                                    "allowance, which is labelled a prediction there")
    return out


def sixteen_gb_arithmetic(rw: dict | None) -> dict | None:
    """Re-run docs/34's 16 GB resident line with the measured loader term.

    receipts/vram-class-verdict.json fits resident weights as `payload + 0.35 GiB` and labels
    the 0.35 a prediction.  This build's own capture measures it.  Everything else in the line
    -- the int8-embedding payload, the two budget figures -- is quoted from that receipt.
    """
    if not rw or "loader_delta_gib" not in rw:
        return None
    v = json.loads((REPO / "receipts/vram-class-verdict.json").read_text())
    cand = v["class_16_gib"]["s16_v_candidate"]
    int8_payload = 12_440_105_028      # cand.resident_payload_gib's own formula input
    int8_gib = int8_payload / GIB
    delta = rw["loader_delta_gib"]
    resident = int8_gib + delta
    budgets = {
        "docs34_section3_basis_12_70": 12.70,
        "measured_utilisation_0_955_basis_12_49": 12.49,
    }
    return {
        "what": "docs/34's 16 GB resident-weight line with the loader allowance measured "
                "instead of predicted",
        "int8_embedding_payload_bytes": int8_payload,
        "int8_embedding_payload_gib": round(int8_gib, 4),
        "predicted_resident_gib": cand["resident_weights_gib"]["value"],
        "predicted_loader_allowance_gib": 0.35,
        "measured_loader_delta_gib": delta,
        "resident_with_measured_loader_gib": round(resident, 4),
        "budgets_gib": budgets,
        "headroom_gib": {k: round(b - resident, 4) for k, b in budgets.items()},
        "fits": {k: resident <= b for k, b in budgets.items()},
        "assumption": "the loader delta is measured with the BF16 embedding resident, because "
                      "the fidelity protocol runs no int8 overlay, and is carried across to the "
                      "int8-embedding form unchanged. That assumes the loader's own overhead "
                      "does not scale with the embedding form.",
        "what_it_does_not_settle": "this is still a 32 GB board. Activation, non-torch and "
                                   "CUDA-graph terms in the 16 GB budget are carried from a "
                                   "32 GB profile, which is docs/34's flip item (2).",
    }


def tree_digests(tree: Path) -> dict:
    files, total = {}, 0
    for p in sorted(tree.rglob("*")):
        if p.is_file():
            b = p.stat().st_size
            total += b
            files[str(p.relative_to(tree))] = {"bytes": b, "sha256": sha256_file(p)}
    return {"dir": str(tree), "file_count": len(files), "total_bytes": total,
            "total_gib": round(total / GIB, 4), "files": files}


def report_row(p: Path) -> dict:
    d = json.loads(p.read_text())
    row = {k: d[k] for k in REPORT_ROW if k in d}
    row["report"] = str(p)
    row["schema"] = d["schema"]
    row["ci95"] = [d["context_bootstrap"]["ci95_low"], d["context_bootstrap"]["ci95_high"]]
    row["clusters"] = d["context_bootstrap"]["clusters"]
    row["bootstrap_samples"] = d["context_bootstrap"]["samples"]
    row["reference"] = d["reference"]
    row["candidate"] = d["candidate"]
    return row


def paired_row(p: Path) -> dict:
    d = json.loads(p.read_text())
    bd = d["bootstrap_difference"]
    return {
        "paired_report": str(p), "schema": d["schema"],
        "a_label": d["a"]["label"], "a_mean": d["a"]["mean"],
        "b_label": d["b"]["label"], "b_mean": d["b"]["mean"],
        "contexts": d["contexts"],
        "difference_a_minus_b": d["difference_a_minus_b"],
        "median_difference": d["median_difference"],
        "ci95": [bd["ci95_low"], bd["ci95_high"]],
        "clusters": bd["clusters"], "bootstrap_samples": bd["samples"],
        "a_wins": d["a_wins"], "b_wins": d["b_wins"],
        "interval_contains_zero": bd["ci95_low"] <= 0.0 <= bd["ci95_high"],
        "sign": ("b better than a" if d["difference_a_minus_b"] > 0 else
                 "a better than b" if d["difference_a_minus_b"] < 0 else "exactly equal"),
        "reading": "difference is a MINUS b on the same 512 contexts, so a positive value "
                   "means b carries more KLD and is worse",
    }


def per_stratum_paired(a_path: Path, b_path: Path, samples: int = 10000,
                       seed: int = 1) -> dict:
    """Paired per-context differences split by suite stratum, same bootstrap as the harness.

    `fidelity.py paired` reports one interval over all 512 contexts.  A calibration-corpus
    change is expected to act unevenly across domains, so the stratum split is where the
    pre-registered sub-prediction lives.  `fidelity.bootstrap` is imported rather than
    reimplemented, so the resampling is bit-for-bit the tool's own.
    """
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    for field in ("suite_token_sha256", "filter", "head_sha256"):
        if a.get(field) != b.get(field):
            raise SystemExit("stratum pairing disagrees on %s" % field)
    ai = {c["index"]: c for c in a["per_context"]}
    bi = {c["index"]: c for c in b["per_context"]}
    if set(ai) != set(bi):
        raise SystemExit("stratum pairing needs the same context set")
    by: dict = {}
    for i in sorted(ai):
        by.setdefault(ai[i]["stratum"], []).append(
            (ai[i]["mean_kld"] - bi[i]["mean_kld"], ai[i]["source_cluster"]))
    out = {}
    for st, rows in sorted(by.items()):
        d = [r[0] for r in rows]
        bs = fidelity.bootstrap(d, [r[1] for r in rows], samples, seed)
        out[st] = {
            "contexts": len(d),
            "difference_a_minus_b": bs["mean"],
            "ci95": [bs["ci95_low"], bs["ci95_high"]],
            "clusters": bs["clusters"], "bootstrap_samples": samples,
            "b_worse_contexts": sum(1 for x in d if x > 0),
            "interval_contains_zero": bs["ci95_low"] <= 0.0 <= bs["ci95_high"],
        }
    out["_reading"] = ("difference is a MINUS b per context, averaged inside the stratum; "
                       "positive means b carries more KLD in that stratum")
    return out


def require_in_prose(prose: str, needle: str, errors: list, what: str) -> None:
    if needle not in prose:
        errors.append("threshold %s (%s) is not present in the pre-registered rule prose"
                      % (needle, what))


def verdict_s16v(mean: float, prereg: dict, cond: dict, errors: list) -> dict:
    rule = cond["registered_decision_rule"]
    k4 = prereg["published_anchors"]["shard_0_reports"]["k4"]["token_mean_kld"]
    require_in_prose(rule["YES"], "%.6f" % k4, errors, "K4 shard-0 mean")
    require_in_prose(rule["NO"], "0.030", errors, "docs/34 6.4 range floor")
    if mean <= k4:
        out = "YES"
    elif mean <= 0.030:
        out = "BORDERLINE"
    else:
        out = "NO"
    return {
        "outcome": out,
        "rule_applied": rule[out],
        "thresholds_used": {"k4_shard0_mean": k4, "docs34_range_floor": 0.030},
        "answers": "docs/34's flip item (1): a measured KLD for one sub-4-bit width in this "
                   "family on shard 0 of the v5 suite, with its tail row. That item is no "
                   "longer absent.",
        "does_not_answer": "items (2) startup on a physical 16 GB board, (3) needle retrieval "
                           "and a text-plus-image request at the 4.2 MP cap on a K3 body, and "
                           "(4) the weighted non-termination check. A 16 GB SKU still requires "
                           "all three, and no 16 GB board has been started in this project.",
    }


def verdict_altcal(paired: dict, prereg: dict, cond: dict, errors: list) -> dict:
    rule = cond["registered_decision_rule"]
    ctrl = prereg["published_anchors"]["converter_draw_noise_floor"]
    envelope = max(abs(ctrl["ci95"][0]), abs(ctrl["ci95"][1]))
    require_in_prose(rule["content_does_not_matter"], "2.9e-05", errors, "control envelope")
    row = paired["hyd"]
    d = abs(row["difference_a_minus_b"])
    if row["interval_contains_zero"] and d <= 2.9e-05:
        out = "content_does_not_matter"
    elif not row["interval_contains_zero"]:
        out = "content_matters"
    else:
        out = "inconclusive"
    small = prereg["published_anchors"]["smallest_margin_this_project_draws_conclusions_from"]
    return {
        "outcome": out,
        "rule_applied": rule.get(out, "the paired interval contains zero but the point "
                                      "difference exceeds the same-corpus control envelope, so "
                                      "this run can neither close the lever nor claim an "
                                      "effect: it bounds the content effect at this resolution "
                                      "and no more."),
        "thresholds_used": {"control_envelope_abs": 2.9e-05,
                            "control_ci95_half_width": envelope,
                            "control_paired_difference": ctrl["paired_difference"]},
        "magnitude_in_context": {
            "abs_difference": d,
            "as_fraction_of_the_smallest_margin_we_conclude_from":
                d / small["hydrated_vs_error_driven"],
            "as_fraction_of_a_one_bit_attention_move":
                d / small["hydrated_vs_context_one_bit_attention"],
            "as_multiple_of_the_same_corpus_converter_draw":
                d / abs(ctrl["paired_difference"]),
        },
    }


def verdict_k6parity(mean: float, prereg: dict, cond: dict, errors: list) -> dict:
    rule = cond["registered_decision_rule"]
    q = prereg["published_anchors"]["gguf_q6_k"]
    net, meas = q["net_of_engine_floor_estimate"], q["token_mean_kld"]
    require_in_prose(rule["BEATS_Q6K"], "%.6f" % net, errors, "Q6_K net of floor")
    require_in_prose(rule["MISSES"], "%.6f" % meas, errors, "Q6_K measured")
    if mean < net:
        out = "BEATS_Q6K"
    elif mean <= meas:
        out = "MATCHES_Q6K"
    else:
        out = "MISSES"
    return {
        "outcome": out,
        "rule_applied": rule[out],
        "thresholds_used": {"q6k_net_of_floor_estimate": net, "q6k_measured": meas},
        "cross_engine_caveat": "the Q6_K row was captured in llama.cpp against the same 512 "
                               "contexts and the same suite token, so the comparison carries "
                               "the cross-engine term measured at %.6f mean "
                               "(receipts/gguf-report-engine-floor.json). A GGUF number is an "
                               "upper bound on its own quantization error and the net figure is "
                               "a naive subtraction, not an identity, because KL is not "
                               "additive. The hydrated pairing carries no such term."
                               % prereg["published_anchors"]["cross_engine_floor"]["token_mean_kld"],
    }


def prediction_check(cond: dict, mean: float, paired: dict, key: str) -> dict:
    pk = cond["predicted_mean_kld"]
    reg = pk.get("registered_primary", pk.get("registered_point"))
    out = {"registered_primary": reg, "measured": mean,
           "ratio_measured_over_registered": mean / reg if reg else None}
    if key == "altcal":
        # this condition's registered interval is on the paired DIFFERENCE, not on the mean
        d = -paired["hyd"]["difference_a_minus_b"]   # published minus alt -> alt minus published
        lo, hi = pk["registered_interval"]
        out["registered_interval_is_on"] = ("the paired difference, alt minus published "
                                            "hydrated; positive means the alt corpus is worse")
        out["registered_interval"] = [lo, hi]
        out["registered_direction"] = pk["registered_direction"]
        out["measured_paired_difference_alt_minus_published"] = d
        out["registered_interval_contains_measured"] = lo <= d <= hi
        out["registered_direction_correct"] = d > 0
        return out
    for k in ("registered_range", "registered_interval"):
        if k in pk:
            lo, hi = pk[k]
            out[k] = pk[k]
            out[k + "_contains_measured"] = lo <= mean <= hi
    if "point_estimate_surrogate" in pk:
        lo, hi = pk["point_estimate_surrogate"]["bracket"]
        out["surrogate_bracket"] = [lo, hi]
        out["surrogate_bracket_contains_measured"] = lo <= mean <= hi
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDS))
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--convert-log", type=Path, required=True)
    ap.add_argument("--build-script", type=Path, required=True)
    ap.add_argument("--paired", action="append", default=[],
                    metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--capture-log", type=Path)
    ap.add_argument("--stratum-baseline", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="report to split against by suite stratum; repeatable")
    ap.add_argument("--wd-bytes", type=Path,
                    help="du -sb output for the deleted conversion working directory")
    ap.add_argument("--extra", type=Path, help="JSON merged into the payload")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    errors: list = []
    prereg = json.loads(PREREG.read_text())
    spec = CONDS[args.cond]
    cond = prereg["conditions"][spec["index"]]

    report = report_row(args.report)
    mean = report["token_mean_kld"]
    if report["suite_token_sha256"] != prereg["protocol"]["suite_token_sha256"]:
        errors.append("suite token mismatch: this score is not on the pre-registered suite")
    if report["head_sha256"] != prereg["protocol"]["head_sha256"]:
        errors.append("head mismatch: this score did not go through the shared head")
    if report["scored_positions"] != prereg["protocol"]["scored_positions"]:
        errors.append("scored positions %d != pre-registered %d"
                      % (report["scored_positions"], prereg["protocol"]["scored_positions"]))

    paired = {}
    for spec_str in args.paired:
        label, _, path = spec_str.partition("=")
        paired[label] = paired_row(Path(path))
        if abs(paired[label]["b_mean"] - mean) > 1e-15:
            errors.append("paired %s does not have this run's report as b (b_mean %r vs %r)"
                          % (label, paired[label]["b_mean"], mean))

    realised = parse_convert_log(args.convert_log)
    for role, want in spec["roles_expected"].items():
        got = realised["per_role"].get(role, {}).get("bpw_values")
        if got != [float(want)]:
            errors.append("realised width for %s is %s, pre-registered %s"
                          % (role, got, want))
    for key, want in spec.get("head_mtp_expected", {}).items():
        got = {n: v["bpw"] for n, v in realised["head_and_mtp"].items()
               if n.startswith(key) and v["bpw"] < 16.0}
        if not got:
            errors.append("no quantized %s module found in the conversion log" % key)
        bad = {n: b for n, b in got.items() if b != want}
        if bad:
            errors.append("realised %s widths %s, pre-registered %s" % (key, bad, want))

    tree = tree_digests(args.tree)
    predicted = cond["predicted_bytes"]
    pred_payload = predicted.get("serialized_payload")
    manifest_path = args.tree / "quantization_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    measured_payload = (sum(v["bytes"] for v in manifest["roles"].values())
                        if manifest else None)

    if args.cond == "s16v":
        verdict = verdict_s16v(mean, prereg, cond, errors)
    elif args.cond == "altcal":
        verdict = verdict_altcal(paired, prereg, cond, errors)
    else:
        verdict = verdict_k6parity(mean, prereg, cond, errors)

    payload = {
        "schema": spec["schema"],
        "question": spec["question"],
        "answered_here": verdict["outcome"],
        "preregistration": {
            "file": "receipts/preregistration-kld9-window.json",
            "file_sha256": sha256_file(PREREG),
            "registered_in_commit": git("log", "-1", "--format=%H", "--",
                                        "receipts/preregistration-kld9-window.json"),
            "registered_at_utc": git("log", "-1", "--format=%cI", "--",
                                     "receipts/preregistration-kld9-window.json"),
            "condition": cond["name"],
            "why_this_matters": "the prediction, the thresholds and the decision rule were "
                                "committed and pushed before this conversion ran. This tool "
                                "reads them back out of that file and asserts each threshold "
                                "appears in its prose, so the verdict cannot have been chosen "
                                "after the number was known.",
            "registered_prediction": cond["predicted_mean_kld"],
            "registered_decision_rule": cond["registered_decision_rule"],
        },
        "protocol": prereg["protocol"],
        "toolchain": dict(prereg["toolchain"],
                          build_script={"path": str(args.build_script),
                                        "sha256": sha256_file(args.build_script)},
                          convert_log={"path": str(args.convert_log),
                                       "sha256": sha256_file(args.convert_log),
                                       "lines": len(args.convert_log.read_text(
                                           errors="replace").splitlines())}),
        "build": {
            "predicted_bytes": predicted,
            "measured_role_payload_bytes": measured_payload,
            "measured_role_payload_gib": (round(measured_payload / GIB, 4)
                                          if measured_payload else None),
            "payload_prediction_error_bytes": (measured_payload - pred_payload
                                               if measured_payload and pred_payload else None),
            "quantization_manifest_roles": manifest["roles"] if manifest else None,
            "tree": tree,
            "realised_widths": realised,
            "working_directory_bytes": (args.wd_bytes.read_text().split()[0]
                                        if args.wd_bytes and args.wd_bytes.exists() else None),
        },
        "score": report,
        "tail_row": {k: report[k] for k in
                     ("token_median_kld", "p95_kld", "p99_kld", "p999_kld", "max_kld",
                      "top1_agreement") if k in report},
        "paired": paired,
        "per_stratum_paired": {
            label: per_stratum_paired(Path(path), args.report)
            for label, _, path in (spec.partition("=") for spec in args.stratum_baseline)
        },
        "prediction_check": prediction_check(cond, mean, paired, args.cond),
        "verdict": verdict,
        "preservation": {"registered_policy": cond["preservation_policy"]},
    }
    if args.cond == "s16v":
        payload["ladder_extension"] = ladder_check(realised)
    if args.capture_log and args.capture_log.exists():
        payload["score"]["capture_log_sha256"] = sha256_file(args.capture_log)
        rw = resident_weights(args.capture_log, measured_payload or pred_payload)
        if rw:
            payload["resident_weights_measured"] = rw
            if args.cond == "s16v":
                payload["sixteen_gb_arithmetic"] = sixteen_gb_arithmetic(rw)
    if args.extra:
        # --extra adds narrative, it never overwrites a measured field: a key collision here
        # silently replaced the paired block once, so it is now fatal.
        add = json.loads(args.extra.read_text())
        clash = sorted(set(add) & set(payload))
        if clash:
            raise SystemExit("--extra would overwrite measured fields %s" % clash)
        payload.update(add)
    payload["errors"] = errors
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"})

    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "mean": mean,
                      "verdict": verdict["outcome"],
                      "prediction_check": payload["prediction_check"],
                      "errors": errors}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
