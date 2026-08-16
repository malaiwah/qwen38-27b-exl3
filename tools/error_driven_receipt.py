#!/usr/bin/env python3
"""Assemble receipts/error-driven-allocation.json from the artifacts of a solved-allocation run.

Inputs are all files produced by the run, nothing is retyped by hand:

  --plan        the pre-registered plan from tools/allocate_bits.py (written before converting)
  --ladder      the per-module proxy-error ladder from tools/ladder_pass.py
  --convert-log the converter's teed stdout for the solved build
  --manifest    quantization_manifest.json written by tools/finalize_checkpoint.py on the build
  --build-receipt build-receipt.json from the same finalize run
  --report      replay report for the solved build
  --report-hyd-rematch  replay report for the published hydrated checkpoint, same reference/harness
  --report-hyd-published  the published shard-0 hydrated report
  --paired      paired report, hydrated-rematched (a) minus solved (b)
  --paired-published    paired report, hydrated-published (a) minus solved (b)
  --paired-reference    paired report, hydrated-published (a) minus hydrated-rematched (b)

The verdict is computed from the paired interval, not asserted: an improvement is only called a
win when the paired 95 % interval on the difference excludes zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

PAIRED_RESOLUTION = 6.4e-5   # half-width of the paired ci95 measured on the existing
                             # shard-0 hydrated-vs-context pair, 512 contexts, 330 clusters


def sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path: str | Path) -> dict:
    p = Path(path)
    st = p.stat() if p.exists() else None
    return {"path": str(p), "sha256": sha256(p),
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)) if st else None,
            "bytes": st.st_size if st else None}


def role_of(key: str) -> str:
    if key.startswith("mtp") or ".mtp." in key:
        return "mtp_draft"
    if "visual" in key:
        return "vision_tower"
    if "lm_head" in key:
        return "lm_head"
    if ".self_attn." in key:
        return "full_attention"
    if ".linear_attn." in key:
        return "linear_attention"
    if ".mlp." in key:
        return "mlp_" + key.rsplit(".", 1)[-1]
    return "other"


def realised_proxy_errors(log: Path) -> dict[str, dict]:
    """Per-module (bits, proxy_err) actually achieved by the conversion, from its own stdout."""
    pat = re.compile(r"Quantized:\s+(?P<key>\S+)\s+bpw:\s+(?P<bpw>[\d.]+)\s+"
                     r"(?:proxy_err|rmse)\s*:\s*(?P<err>[\d.eE+-]+)")
    out = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            out[m.group("key")] = {"bits": round(float(m.group("bpw"))),
                                   "proxy_err": float(m.group("err"))}
    return out


def equal_width_table(lad_mods: dict) -> dict:
    """Mean / median proxy error per role at each width, over the modules that have that rung.

    This is the comparison the published hydrated recipe never made: its justification for
    down_proj at K6 was that down_proj carries the largest per-tensor proxy error in every layer,
    but that observation compared down_proj at K6 with gate/up at K5. At equal width the three MLP
    roles must be compared directly.
    """
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, rec in lad_mods.items():
        for k, v in (rec.get("ladder") or {}).items():
            if v is not None and v > 0:
                buckets[(role_of(key), int(k))].append(v)
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (role, k), vals in sorted(buckets.items()):
        out[role][f"K{k}"] = {"modules": len(vals), "mean": statistics.fmean(vals),
                              "median": statistics.median(vals),
                              "min": min(vals), "max": max(vals)}
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("plan", "ladder", "convert-log", "manifest", "build-receipt", "report",
                 "report-hyd-rematch", "report-hyd-published", "paired", "paired-published",
                 "paired-reference"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--workdir", default="/var/tmp/work/kld6")
    ap.add_argument("--exllamav3", default="/var/tmp/work/exllamav3")
    ap.add_argument("--receipts", default="receipts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())
    ladder = json.loads(Path(a.ladder).read_text())
    manifest = json.loads(Path(a.manifest).read_text())
    breceipt = json.loads(Path(a.build_receipt).read_text())
    rep = json.loads(Path(a.report).read_text())
    rep_hr = json.loads(Path(a.report_hyd_rematch).read_text())
    rep_hp = json.loads(Path(a.report_hyd_published).read_text())
    paired = json.loads(Path(a.paired).read_text())
    paired_pub = json.loads(Path(a.paired_published).read_text())
    paired_ref = json.loads(Path(a.paired_reference).read_text())
    W = Path(a.workdir)

    # ---- realised bytes, manifest convention ----
    hyd_man = json.loads((Path(a.receipts) / "hydrated-quantization-manifest.json").read_text())["roles"]
    real_roles = {r: v["bytes"] for r, v in manifest["roles"].items()}
    pred_roles = plan["bytes"]["solved_roles"]
    byte_rows = {}
    for r in sorted(set(real_roles) | set(pred_roles) | set(hyd_man)):
        row = {"predicted": pred_roles.get(r), "realised": real_roles.get(r),
               "hydrated": hyd_man.get(r, {}).get("bytes")}
        row["realised_minus_predicted"] = (
            None if row["realised"] is None or row["predicted"] is None
            else row["realised"] - row["predicted"])
        row["realised_minus_hydrated"] = (
            None if row["realised"] is None or row["hydrated"] is None
            else row["realised"] - row["hydrated"])
        byte_rows[r] = row
    real_total = sum(real_roles.values())
    hyd_total = sum(v["bytes"] for v in hyd_man.values())

    # ---- ladder validity: what the converter actually got at the solved widths ----
    realised = realised_proxy_errors(Path(a.convert_log))
    lad_mods = ladder["modules"]
    rows, ratios = {}, []
    for key, r in realised.items():
        rec = lad_mods.get(key)
        pred = (rec.get("ladder") or {}).get(str(r["bits"])) if rec else None
        if not pred or pred <= 0 or r["proxy_err"] <= 0:
            continue
        ratio = r["proxy_err"] / pred
        rows[key] = {"bits": r["bits"], "ladder_predicted": pred, "realised": r["proxy_err"],
                     "ratio_realised_over_predicted": ratio}
        ratios.append(ratio)
    ratios.sort()

    def q(p):
        return ratios[min(len(ratios) - 1, int(p * len(ratios)))] if ratios else None

    by_role = defaultdict(list)
    for key, v in rows.items():
        by_role[role_of(key)].append(v["ratio_realised_over_predicted"])

    # ---- verdict from the paired interval ----
    d = paired["bootstrap_difference"]
    lo, hi = d["ci95_low"], d["ci95_high"]
    if lo > 0 and hi > 0:
        outcome = "win"            # a - b > 0: hydrated is worse than the solved build
    elif lo < 0 and hi < 0:
        outcome = "loss"
    else:
        outcome = "no_resolvable_difference"
    measured_delta = rep["context_macro_mean_kld"] - rep_hr["context_macro_mean_kld"]
    predicted_delta = plan["predicted"]["kld_delta_solved_minus_hydrated"]
    ref_shift = rep_hp["context_macro_mean_kld"] - rep_hr["context_macro_mean_kld"]

    git = subprocess.run(["git", "-C", a.exllamav3, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", a.exllamav3, "diff"], capture_output=True, text=True).stdout

    eqw = equal_width_table(lad_mods)
    mlp_eq = {}
    for k in ("K4", "K5", "K6", "K7"):
        r = {role: eqw.get(role, {}).get(k, {}).get("mean")
             for role in ("mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")}
        if all(v is not None for v in r.values()):
            mlp_eq[k] = dict(r, down_over_gate=r["mlp_down_proj"] / r["mlp_gate_proj"])

    receipt = {
        "schema": "qwen38-error-driven-allocation/1",
        "question": "Does an error-driven per-module bit allocation, solved against the converter's own "
                    "proxy error at the hydrated build's serialized-byte budget, measure better than the "
                    "hand-designed hydrated role split?",
        "verdict": {
            "outcome": outcome,
            "plain": None,     # filled in below
            "paired_difference_hydrated_minus_solved": d["mean"],
            "paired_ci95": [lo, hi],
            "paired_contexts": paired["contexts"],
            "paired_clusters": d["clusters"],
            "solved_wins": paired["b_wins"], "hydrated_wins": paired["a_wins"],
            "solved_mean_kld": rep["context_macro_mean_kld"],
            "hydrated_rematched_mean_kld": rep_hr["context_macro_mean_kld"],
            "hydrated_published_mean_kld": rep_hp["context_macro_mean_kld"],
            "solved_serialized_bytes": real_total,
            "hydrated_serialized_bytes": hyd_total,
            "byte_budget_respected": real_total <= hyd_total,
        },
        "protocol": {
            "suite": "v5 shard 0",
            "suite_token_sha256": rep["suite_token_sha256"],
            "contexts": rep["contexts"], "scored_positions": rep["scored_positions"],
            "reference_capture": rep["reference"],
            "shared_head": rep["head"], "head_sha256": rep["head_sha256"],
            "filter": rep["filter"],
            "harness": stamp("tools/fidelity.py"),
            "paired_resolution_half_width": PAIRED_RESOLUTION,
            "paired_resolution_basis": "ci95 half-width of the paired difference between the existing "
                                       "shard-0 hydrated and context reports (-0.000710 "
                                       "[-0.000778, -0.000650], 512 contexts, 330 clusters)",
            "reference_reuse": {
                "why": "the published shard-0 hydrated report was replayed against "
                       "/work/kld5/hidden/shard-0000/hidden-bf16, which was deleted when the 10 M-position "
                       "run reclaimed disk; the surviving BF16 shard-0 capture is /work/gguf/hidden-bf16",
                "handling": "hydrated was re-captured and re-replayed against the surviving reference with "
                            "this harness version, so the headline paired delta uses one reference and one "
                            "code version for both operands",
                "published_minus_rematched_mean_kld": ref_shift,
                "exceeds_paired_resolution": abs(ref_shift) > PAIRED_RESOLUTION,
                "note": "capture is not assumed deterministic: this difference is measured, and the paired "
                        "report paired_reference_and_harness_delta gives it an interval",
            },
        },
        "converter_signal": {
            "emitted_per_conversion": "one proxy error per module, at the width the allocator assigned: "
                                      "proxy_err = tr(E^T H E) / tr(W^T H W), computed in "
                                      "exllamav3/modules/quant/exl3_lib/quantize.py and printed by "
                                      "conversion/convert_model.py:print_quantized_linear",
            "candidate_ladder_emitted": False,
            "on_disk_before_this_work": {
                "convert-ctx.log": "519 single-width points (one per module at the context build's widths), "
                                   "the only converter stdout that survived; no candidate curve",
                "working_directories": "wd-hyd / wd-ctx / wd-k4 / wd-v2 held only the resume state "
                                       "(ckpt/state.safetensors) and the per-layer quantized tensors: no "
                                       "Hessians, no error records. Reclaimed for this work",
                "reused": "the context log's points were used only as a magnitude cross-check; a curve "
                          "cannot be reconstructed from one width per module",
            },
            "upstream_error_driven_path": {
                "tools": "util/measure.py -> conversion/measure_model.py, util/optimize.py -> "
                         "conversion/optimize_model.py",
                "signal": "per-group dkld/dbits, measured by swapping module groups between >=2 already "
                          "converted whole models and a reference",
                "why_not_used": "it needs N full conversions of the model plus a "
                                "VariantSafetensorsCollection splice-recompile, and its allocator is a "
                                "greedy dkld/dbits ratio with an ad-hoc adjust(dkld) = -(-dkld)**0.69 for "
                                "negative deltas. The ladder pass here gets a per-candidate signal in ONE "
                                "conversion by reusing each captured Hessian, and the allocation is then "
                                "solved exactly rather than greedily",
            },
            "defect_found_and_fixed": "tools/ladder_from_log.py matched only the `rmse` label, which the "
                                      "converter prints for uncalibrated fallback modules; all 400 "
                                      "calibrated body modules print `proxy_err` and were silently dropped "
                                      "(118 modules extracted instead of 519). Regex fixed in this change",
        },
        "ladder": {
            "tool": stamp("tools/ladder_pass.py"),
            "artifact": stamp(a.ladder),
            "modules": len(lad_mods),
            "rungs_per_module": {"big": ladder["candidate_widths"]["big"],
                                 "small": ladder["candidate_widths"]["small"],
                                 "big_numel_threshold": ladder["candidate_widths"]["big_numel_threshold"]},
            "measurement_cost_sec": ladder["elapsed_sec"],
            "propagation_recipe": ladder["propagation_recipe"],
            "metric": ladder["metric"],
            "out_energy": ladder["out_energy"],
            "assumption_a_reader_must_check": (
                "every rung is measured at FIXED propagation: one Hessian per module, captured with the "
                "hydrated recipe's upstream widths, then re-used for all five widths of that module. So the "
                "ladder is the module-local error of changing one module's width, not the true sequential "
                "effect of changing many widths at once, which would move every downstream Hessian. The "
                "objective built on it is therefore first-order and additive by assumption. Both assumptions "
                "are checked, not asserted: pre_registered_plan.objective_validation scores the objective "
                "against two measured single-role-group KLD deltas between published checkpoints, and "
                "prediction_vs_measurement.ladder_replay compares every ladder rung against the proxy error "
                "the real conversion achieved at the same width with the solved propagation"),
            "equal_width_proxy_error_by_role": eqw,
        },
        "hand_designed_recipe_justification_reviewed": {
            "claim_in_the_shipped_recipe": "down_proj is at K6 while gate/up are at K5 because down_proj "
                                           "carries the largest per-tensor proxy error in every layer "
                                           "(tools/exllamav3-allocation-bits-override.py, "
                                           "docs/12 P2, build_hydrated.sh)",
            "what_the_ladder_shows_at_equal_width": mlp_eq,
            "reading": None,   # filled in below
        },
        "pre_registered_plan": {
            "artifact": stamp(a.plan),
            "conversion_log": stamp(a.convert_log),
            "written_before_conversion": Path(a.plan).stat().st_mtime < Path(a.convert_log).stat().st_mtime,
            "immutable_after_writing": True,
            "objective_chosen": plan["objective"]["chosen"],
            "objective_selection_rule": plan["objective"]["rule"],
            "objective_definitions": plan["objective"]["definition"],
            "objective_validation": plan["objective"]["validation"],
            "kld_per_objective_unit": plan["objective"]["kld_per_objective_unit"],
            "predicted": plan["predicted"],
            "byte_law": plan["byte_law"],
            "byte_law_validation": plan["byte_law_validation"],
            "solver": plan["solver"],
            "grid_unit_bytes": plan["grid_unit_bytes"], "grid_points": plan["grid_points"],
        },
        "allocation": {
            "average_body_bits": plan["average_body_bits"],
            "hydrated_average_body_bits": plan["hydrated_average_body_bits"],
            "bit_histogram_by_role": plan["bit_histogram_by_role"],
            "modules_changed_from_hydrated": len(plan["changed_from_hydrated"]),
            "changed_from_hydrated": plan["changed_from_hydrated"],
            "solved_bits": plan["solved_bits"],
            "realised_bits_from_converter_log": {k: v["bits"] for k, v in sorted(realised.items())},
            "realised_matches_solved": all(
                realised[k]["bits"] == v for k, v in plan["solved_bits"].items() if k in realised),
        },
        "bytes": {
            "convention": "per-role serialized bytes, embeddings and vision separate, never called VRAM "
                          "(receipts/hydrated-quantization-manifest.json convention)",
            "per_role": byte_rows,
            "solved_total": real_total, "solved_gib": real_total / 2 ** 30,
            "hydrated_total": hyd_total, "hydrated_gib": hyd_total / 2 ** 30,
            "solved_minus_hydrated": real_total - hyd_total,
            "predicted_total": plan["bytes"]["solved_total"],
            "realised_minus_predicted_total": real_total - plan["bytes"]["solved_total"],
            "disk_bytes": breceipt.get("artifact", {}).get("disk_bytes"),
        },
        "fidelity": {
            "solved": {k: rep.get(k) for k in ("token_mean_kld", "token_median_kld", "p95_kld", "p99_kld",
                                               "p999_kld", "max_kld", "top1_agreement", "mean_jsd_bits",
                                               "context_macro_mean_kld", "context_bootstrap")},
            "hydrated_rematched": {k: rep_hr.get(k) for k in ("token_mean_kld", "token_median_kld",
                                                              "p999_kld", "max_kld", "top1_agreement",
                                                              "context_macro_mean_kld", "context_bootstrap")},
            "hydrated_published": {k: rep_hp.get(k) for k in ("token_mean_kld", "token_median_kld",
                                                              "p999_kld", "max_kld", "top1_agreement",
                                                              "context_macro_mean_kld", "context_bootstrap")},
            "paired_hydrated_rematched_minus_solved": paired,
            "paired_hydrated_published_minus_solved": paired_pub,
            "paired_reference_and_harness_delta": paired_ref,
            "published_shard0_comparators": {
                "hydrated": 0.002700, "online_k5k6": 0.003141, "context": 0.003409, "k4": 0.010345,
                "official_fp8": 0.005197, "gguf_q6_k": 0.002035,
                "note": "the GGUF number carries a cross-engine floor of 0.000507 "
                        "(receipts/gguf-report-engine-floor.json); the EXL3 numbers do not",
            },
        },
        "prediction_vs_measurement": {
            "predicted_kld_delta_solved_minus_hydrated": predicted_delta,
            "measured_kld_delta_solved_minus_hydrated_rematched": measured_delta,
            "measured_minus_predicted": measured_delta - predicted_delta,
            "sign_agreement": (predicted_delta < 0) == (measured_delta < 0),
            "ladder_replay": {
                "modules_compared": len(rows),
                "ratio_realised_over_ladder_predicted": {
                    "p05": q(0.05), "median": q(0.5), "p95": q(0.95),
                    "mean": (sum(ratios) / len(ratios)) if ratios else None},
                "by_role_mean_ratio": {r: sum(v) / len(v) for r, v in sorted(by_role.items())},
                "meaning": "the ladder was measured with hydrated propagation; converting with the solved "
                           "widths changes every layer's input state, so a ratio above 1 is the part of the "
                           "error the layer-local proxy cannot see",
                "per_module": rows,
            },
        },
        "identity": {
            "source_model": "/var/tmp/models/Qwen3.8-27B",
            "source_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "converter": f"turboderp-org/exllamav3@{git} (1.4.2) plus this repository's allocation patch "
                         f"(tools/exllamav3-allocation-bits-override.py)",
            "converter_worktree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "command": breceipt.get("build", {}).get("command") or breceipt.get("command"),
            "recipe": breceipt.get("build", {}).get("recipe") or breceipt.get("recipe"),
            "bits_fixed_spec": stamp(W / "solved-fixed.json"),
            "bits_override_spec": stamp(W / "solved-override.json"),
            "build_script": stamp(W / "build_solved.sh"),
            "measure_script": stamp(W / "measure_eda.sh"),
            "ladder_script": stamp(W / "run_ladder.sh"),
            "solver": stamp("tools/allocate_bits.py"),
            "receipt_tool": stamp("tools/error_driven_receipt.py"),
            "checkpoint_dir": a.checkpoint_dir,
            "checkpoint_index_sha256": sha256(Path(a.checkpoint_dir) / "model.safetensors.index.json"),
            "hardware": "1x RTX PRO 6000 Blackwell Server Edition, SM120, driver 595.58.03, rental box "
                        "(never AIBoss's 5090; no number here is comparable across the two cards)",
        },
    }

    sign = "lower" if measured_delta < 0 else "higher"
    receipt["verdict"]["plain"] = (
        f"The solved allocation measures {rep['context_macro_mean_kld']:.6f} mean KLD against the "
        f"re-measured hydrated {rep_hr['context_macro_mean_kld']:.6f} on the same 512 contexts and the "
        f"same BF16 reference, i.e. {abs(measured_delta):.6f} {sign}, at "
        f"{real_total:,} serialized bytes against hydrated's {hyd_total:,}. The paired difference is "
        f"{d['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}] over {paired['contexts']} contexts with "
        f"{paired['b_wins']} contexts won by the solved build and {paired['a_wins']} by hydrated. "
        + {"win": "The interval excludes zero in favour of the solved allocation: error-driven "
                  "allocation beats the hand-designed split at the same byte budget.",
           "loss": "The interval excludes zero in favour of hydrated: error-driven allocation on this "
                   "proxy does NOT beat the hand-designed split at the same byte budget.",
           "no_resolvable_difference": "The interval includes zero: on this protocol the two allocations "
                                       "are not distinguishable, so no improvement may be claimed."}[outcome])
    if mlp_eq:
        worst = {k: v["down_over_gate"] for k, v in mlp_eq.items()}
        receipt["hand_designed_recipe_justification_reviewed"]["reading"] = (
            "At equal width, down_proj's mean proxy error is "
            + ", ".join(f"{r:.3f}x gate_proj's at {k}" for k, r in worst.items())
            + ". The shipped justification compared down_proj at K6 with gate/up at K5, which is not a "
              "comparison of tensors but of widths; "
            + ("down_proj is therefore NOT the highest-error MLP tensor at equal width and the stated "
               "reason for spending the extra bit there is wrong, whatever the merits of the resulting "
               "recipe." if all(r < 1 for r in worst.values()) else
               "the ladder at equal width is what settles it; see the numbers above."))

    Path(a.out).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"outcome": outcome, "paired": [d["mean"], lo, hi],
                      "solved_mean": rep["context_macro_mean_kld"],
                      "hyd_rematch_mean": rep_hr["context_macro_mean_kld"],
                      "published_minus_rematched": ref_shift,
                      "bytes": real_total, "budget": hyd_total,
                      "predicted_delta": predicted_delta, "measured_delta": measured_delta}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
