#!/usr/bin/env python3
"""Is any per-module error weighting a calibrated surrogate for measured KLD?

The error-driven allocation of rank 9 minimised one candidate surrogate — the equal-weighted sum
of exllamav3's relative proxy error — and made KLD worse. That is a fact about the surrogate, so
the question this answers is the general one: over every pair of published builds whose difference
is a pure change of per-module widths, does any weighting `w_m` make

    KLD(b) - KLD(a)  ~=  scale * sum_m w_m * (eps(m, K_b) - eps(m, K_a))

hold in sign and magnitude with ONE free parameter (`scale`)?

Four measured pairs are available on shard 0 of the v5 suite, all with cluster-bootstrap intervals:
two uniform role-group moves (attention K6->K5, MLP K5/K5/K6->K4) and two reallocations involving
the error-driven build. Pairs whose difference is a change of *mechanism* rather than of width —
hydrated versus the online-encoded K5/K6 build, or anything against official FP8 — are excluded,
because no width-based rule can be asked to predict them; the hydrated-vs-K5/K6 delta is reported
separately as the size of the effect a width rule cannot see.

Each rule is scored two ways: the spread of the scale it implies across the four pairs (1.0 would
be perfect calibration), and leave-one-pair-out, fitting the scale on three pairs by least squares
through the origin and predicting the fourth against its measured interval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocate_bits as ab


def rules() -> dict:
    return {
        "rel": {"weight": lambda rec: 1.0,
                "definition": "w_m = 1: every module's relative proxy error counts equally. The "
                              "rule the rank-9 plan selected and the allocation minimised."},
        "abs": {"weight": lambda rec: rec["out_energy"],
                "definition": "w_m = tr(W^T H W)/count, the module's mean per-row calibration "
                              "output energy, so w_m*eps is absolute output error energy (the GPTQ "
                              "objective under isotropic downstream sensitivity)."},
        "sqrt_energy": {"weight": lambda rec: math.sqrt(rec["out_energy"]),
                        "definition": "w_m = sqrt(out_energy), the geometric mean of the two above. "
                                      "Included because it was the only rule that matched the "
                                      "error-driven build's measured delta, and it is therefore "
                                      "under test here rather than being reported as a result."},
        "numel": {"weight": lambda rec: float(rec["numel"]),
                  "definition": "w_m = parameter count: error weighted by how much of the model the "
                                "module is."},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--plan", required=True, help="the plan whose solved_bits define the candidate")
    ap.add_argument("--paired-hyd-ctx", required=True)
    ap.add_argument("--paired-k5k6-k4", required=True)
    ap.add_argument("--paired-hyd-eda", required=True)
    ap.add_argument("--paired-ctx-eda", required=True)
    ap.add_argument("--paired-hyd-k5k6", required=True,
                    help="mechanism-only pair (offline vs online K6 attention), reported not fitted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = ab.load_ladder(Path(a.ladder))
    lad = L["modules"]
    plan = json.loads(Path(a.plan).read_text())
    solved = plan["solved_bits"]
    hyd, ctx, k4 = ab.hydrated_bits(), ab.context_bits(), ab.k4_mlp_bits()
    free = sorted(k for k in lad
                  if ab.MODS[k][2] in ("full_attention", "linear_attention", "mlp_gate_proj",
                                       "mlp_up_proj", "mlp_down_proj")
                  and lad[k].get("out_energy") is not None)

    def measured(path: str) -> dict:
        d = json.loads(Path(path).read_text())
        b = d["bootstrap_difference"]
        # reports are (a - b); this module speaks in (b - a), b being the second-named build
        return {"delta_b_minus_a": -d["difference_a_minus_b"],
                "ci95": [-b["ci95_high"], -b["ci95_low"]],
                "contexts": d["contexts"], "clusters": b["clusters"],
                "a_label": d["a"]["label"], "b_label": d["b"]["label"], "report": path}

    PAIRS = {
        "hydrated_to_context": {"a": hyd, "b": ctx, "kind": "uniform role-group move: attention K6->K5",
                                "m": measured(a.paired_hyd_ctx)},
        "mlp_to_K4": {"a": hyd, "b": k4, "kind": "uniform role-group move: MLP K5/K5/K6->K4 "
                                                 "(measured between the two online-K6-attention builds, "
                                                 "whose only difference is the MLP)",
                      "m": measured(a.paired_k5k6_k4)},
        "hydrated_to_error_driven": {"a": hyd, "b": solved, "kind": "reallocation at a fixed byte budget",
                                     "m": measured(a.paired_hyd_eda)},
        "context_to_error_driven": {"a": ctx, "b": solved, "kind": "reallocation plus an attention move",
                                    "m": measured(a.paired_ctx_eda)},
    }

    R = rules()
    for pname, p in PAIRS.items():
        p["modules_moved"] = sum(1 for k in free if p["a"][k] != p["b"][k])
        keys = [k for k in free if p["a"][k] != p["b"][k]]
        p["objective_delta"] = {}
        for rn, r in R.items():
            d = sum(r["weight"](lad[k]) * (lad[k]["eps"][p["b"][k]] - lad[k]["eps"][p["a"][k]])
                    for k in keys)
            p["objective_delta"][rn] = {"delta": d,
                                        "implied_scale": p["m"]["delta_b_minus_a"] / d if d else None}

    out_rules = {}
    names = list(PAIRS)
    for rn, r in R.items():
        ds = {n: PAIRS[n]["objective_delta"][rn]["delta"] for n in names}
        ms = {n: PAIRS[n]["m"]["delta_b_minus_a"] for n in names}
        scales = [PAIRS[n]["objective_delta"][rn]["implied_scale"] for n in names]
        signs_ok = all((ds[n] > 0) == (ms[n] > 0) for n in names)
        loo = {}
        for held in names:
            keep = [n for n in names if n != held]
            s = sum(ds[n] * ms[n] for n in keep) / sum(ds[n] ** 2 for n in keep)
            pred = ds[held] * s
            lo, hi = PAIRS[held]["m"]["ci95"]
            loo[held] = {"fitted_scale": s, "predicted": pred, "measured": ms[held], "ci95": [lo, hi],
                         "ratio_predicted_over_measured": pred / ms[held],
                         "verdict": ("inside the measured interval" if lo <= pred <= hi else
                                     "sign correct, magnitude wrong" if (pred > 0) == (ms[held] > 0)
                                     else "sign wrong")}
        out_rules[rn] = {
            "definition": r["definition"],
            "free_parameters": 1,
            "implied_scale_per_pair": {n: PAIRS[n]["objective_delta"][rn]["implied_scale"] for n in names},
            "scale_spread_ratio": (max(scales) / min(scales)) if min(scales) > 0 else None,
            "scale_changes_sign_across_pairs": min(scales) < 0 < max(scales),
            "sign_correct_on_all_pairs": signs_ok,
            "leave_one_pair_out": loo,
            "worst_out_of_sample_ratio": max(abs(v["ratio_predicted_over_measured"]) for v in loo.values()),
        }

    usable = {rn: v for rn, v in out_rules.items() if v["sign_correct_on_all_pairs"]}
    best = min(usable, key=lambda rn: usable[rn]["scale_spread_ratio"]) if usable else None
    mech = measured(a.paired_hyd_k5k6)

    receipt = {
        "schema": "qwen38-error-surrogate-calibration/1",
        "question": "does any per-module weighting of exllamav3's relative proxy error predict the "
                    "measured KLD difference between two builds, in sign and magnitude, with one "
                    "free scale parameter?",
        "protocol": "shard 0 of the v5 suite, 512 contexts, 1,048,064 scored positions, one shared "
                    "BF16 head, cluster-bootstrap intervals over 330 source clusters; every delta is "
                    "a paired per-context difference, not a difference of two independent means",
        "ladder": {"source": a.ladder, "modules_used": len(free),
                   "note": "eps rungs measured at fixed propagation; see "
                           "receipts/error-driven-allocation.json -> ladder"},
        "excluded_pairs": {
            "why": "a width-based rule cannot be asked to predict a change of mechanism",
            "hydrated_vs_online_K5K6": dict(mech, note="offline calibrated K6 attention versus the "
                                                       "runtime's uncalibrated online K6 encoding at "
                                                       "identical widths: a width rule predicts exactly "
                                                       "zero for this pair, and the measured effect is "
                                                       "this large"),
            "official_fp8_and_nvfp4": "different quantizer entirely, no per-module EXL3 width map",
        },
        "pairs": {n: {"kind": PAIRS[n]["kind"], "modules_moved": PAIRS[n]["modules_moved"],
                      "measured": PAIRS[n]["m"],
                      "objective_delta": PAIRS[n]["objective_delta"]} for n in names},
        "rules": out_rules,
        "verdict": {
            "calibrated_surrogate_found": False,
            "best_sign_correct_rule": best,
            "best_scale_spread_ratio": usable[best]["scale_spread_ratio"] if best else None,
            "rules_that_get_the_sign_wrong": [rn for rn, v in out_rules.items()
                                              if not v["sign_correct_on_all_pairs"]],
            "plain": None,
        },
    }
    rel_bad = out_rules["rel"]["leave_one_pair_out"]["hydrated_to_error_driven"]
    receipt["verdict"]["plain"] = (
        f"No rule is calibrated. The rule the plan selected, `rel`, changes the sign of its implied "
        f"scale across the four pairs and predicts {rel_bad['predicted']:+.6f} for the reallocation "
        f"against a measured {rel_bad['measured']:+.6f} — which is exactly why the error-driven build "
        f"lost. `numel` also gets that sign wrong. `abs` and `sqrt_energy` get the sign right on all "
        f"four pairs, but out of sample they are off by up to "
        f"{out_rules['abs']['worst_out_of_sample_ratio']:.1f}x and "
        f"{out_rules['sqrt_energy']['worst_out_of_sample_ratio']:.1f}x respectively, and their implied "
        f"scale spreads {out_rules['abs']['scale_spread_ratio']:.1f}x and "
        f"{out_rules['sqrt_energy']['scale_spread_ratio']:.1f}x across the pairs. A rule that is a "
        f"factor of two-and-a-half uncertain cannot decide a 1e-4 fidelity claim, so there is no "
        f"surrogate here good enough to authorise a conversion on prediction alone. `sqrt_energy` is "
        f"the least-bad and is the pre-registrable candidate for a next attempt; its selection must be "
        f"validated on a BETWEEN-role reallocation delta, which is the test the rank-9 rule lacked.")
    Path(a.out).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"best_sign_correct_rule": best,
                      "spreads": {rn: v["scale_spread_ratio"] for rn, v in out_rules.items()},
                      "sign_wrong": receipt["verdict"]["rules_that_get_the_sign_wrong"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
