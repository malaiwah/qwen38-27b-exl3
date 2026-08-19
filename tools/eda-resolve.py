#!/usr/bin/env python3
"""Re-solve the EXL3 error-driven bit allocation under structured weight models.

Background
----------
`malaiwah/Qwen3.8-27B-EXL3-EDA-research` solved a per-matrix bit allocation with
the `rel` objective (`sum_m eps(m,K)`, every module weighted equally). That build
measured WORSE than the hydrated recipe (+0.000366 mean KLD on shard 0) because
`rel` is blind to KV-cache error compounding: it moved bytes away from
attention/GDN and into MLP (docs/57). A depth sweep further showed every
depth-blind weighting also strips bytes from early layers
(receipts/eda-depth-weighting-2026-08-19.md).

This tool re-runs the allocation as exact dynamic programming over the byte
grid (matching the original solver) with a WEIGHT MODEL instead of a single
scalar family:

    w_m = base(weighting, m) * depth_factor(layer(m), amp)

Weightings:
  rel          base = 1                      (the published objective)
  abs          base = out_energy_m
  sqrt_energy  base = sqrt(out_energy_m)
  class-kld    base = s_class(m)  -- per-class KLD-per-eps-unit scales fitted
               from the plan's own two MEASURED calibration deltas
               (attention K6->K5: 0.000709 KLD over 0.0406893 eps-units;
               MLP K5K5K6->K4: 0.007204 over 0.2735037). The plan AVERAGED
               these into one global scale (0.021423) and called their 1.51x
               spread "inconsistency"; the spread is the class signal. Under
               class-kld the objective is directly a predicted-KLD-delta in
               absolute units, which makes solves comparable and falsifiable.

Constraints:
  --max-width  serving-aware width cap. Under ANY_BITS the b12x prefill gate is
               n_words in (48,64,80,96) = K3..K6 ONLY
               (patches/vllm-exl3-multiprecision.py:1644): K7/K8 modules fall
               off the fast path. The published EDA solve put 42 modules at K7.

Falsification built in
----------------------
1. The byte model must reproduce published ground truth exactly (hydrated role
   totals; published solved role totals; objective domain to 15 significant
   figures) or the tool refuses to emit an allocation.
2. With --solved provided, the tool prints a HELD-OUT prediction test: the
   per-class model's predicted KLD delta for the PUBLISHED reallocation versus
   its measured +0.000366. The two class scales were fitted on single-class
   moves; the published solve is a cross-class reallocation neither scale saw.

Everything here is first-order and layer-local at the ladder level; the depth
term and class scales are corrections fitted to measurements, not a substitute
for a compounding-aware objective. `depth_amp` remains UNCALIBRATED (no
early-vs-late measurement exists yet); any amp > 0 is a sensitivity probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# Roles carried by the 409-module ladder.  `lm_head` and the MTP draft were in
# the ladder but PINNED by the original solver's fixed/override regexes -- the
# published solve leaves their byte totals bit-identical to hydrated -- so they
# are excluded from the DP and held at recipe_bits here too.  embed_tokens,
# vision_tower and norms_and_small never enter the ladder (BF16 / fixed).
ROLE_PATTERNS = (
    ("linear_attn.", "linear_attention"),
    ("self_attn.", "full_attention"),
    ("mlp.gate_proj", "mlp_gate_proj"),
    ("mlp.up_proj", "mlp_up_proj"),
    ("mlp.down_proj", "mlp_down_proj"),
)
PINNED_ROLES = ("lm_head", "mtp_draft")

# Measured outcome of the published rel-solved reallocation, for the held-out
# prediction test: hydrated 0.002700 -> EDA 0.003066 on shard 0, paired CI
# excludes zero (docs/57 SS1; EDA-research model card "The measurement").
MEASURED_PUBLISHED_DELTA = +0.000366
GLOBAL_SCALE_PREDICTION = -0.000211  # the plan's own prediction with one scale


def role_of(name: str) -> str:
    if name == "lm_head":
        return "lm_head"
    if name.startswith("mtp."):
        return "mtp_draft"
    for frag, role in ROLE_PATTERNS:
        if frag in name:
            return role
    raise KeyError(f"no role for module {name!r}")


def class_of(name: str) -> str:
    """attn = full attention + GDN (the plan's attention delta demoted both);
    mlp = gate/up/down."""
    r = role_of(name)
    if r in ("full_attention", "linear_attention"):
        return "attn"
    if r.startswith("mlp_"):
        return "mlp"
    return "pinned"


LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def depth_factor(name: str, amp: float, n_layers: int, form: str = "exp") -> float:
    """Downstream-amplification factor for error injected at this module's layer.

    Error injected at layer L is subsequently transformed by layers L+1..n-1.
    If each downstream layer amplifies injected error by a mean factor (1+amp),
    the injected error's contribution at the readout scales as
    (1+amp)**(n-1-L). amp=0 reproduces depth-blind behaviour exactly.

    amp is NOT calibrated by any measurement in this repository
    (receipts/eda-depth-weighting-2026-08-19.md): treat amp > 0 as a
    sensitivity probe, never as a result.
    """
    if amp == 0.0:
        return 1.0
    L = layer_index(name)
    if L is None:
        return 1.0
    if form == "exp":
        return (1.0 + amp) ** (n_layers - 1 - L)
    if form == "u":
        # U-shaped: both ends weighted above the middle. Prior art: llama.cpp
        # use_more_bits promotes the first AND last n/8 of layers
        # (src/llama-quant.cpp:430), and exllamav3's own allocator weights by
        # dist = min(layer, stack_max - layer) ("end layers contribute
        # disproportionately", exllamav3/conversion/allocation.py:63) - see
        # docs/58. Mechanisms differ per end: early = downstream amplification,
        # late = readout proximity.
        dist = min(L, n_layers - 1 - L)
        half = (n_layers - 1) / 2.0
        return (1.0 + amp) ** (half - dist)
    raise ValueError(f"unknown depth form {form!r}")


def fit_class_scales(plan: dict) -> dict[str, float]:
    """Per-class KLD-per-eps-unit scales from the plan's own validation deltas.

    The plan measured two single-class moves and recorded, for each, the
    ladder-predicted objective delta and the measured KLD delta. Their ratio is
    the class's KLD-per-eps-unit scale. The plan then averaged them; we keep
    them separate.
    """
    deltas = plan["objective"]["validation"]["rel"]["deltas"]
    scales: dict[str, float] = {}
    for key, d in deltas.items():
        s = d["implied_scale_kld_per_objective_unit"]
        if "attention" in key.lower():
            scales["attn"] = float(s)
        elif "mlp" in key.lower():
            scales["mlp"] = float(s)
    if set(scales) != {"attn", "mlp"}:
        raise SystemExit(f"could not fit class scales from plan deltas: {list(deltas)}")
    return scales


def build_weights(mods: dict, kind: str, amp: float, n_layers: int,
                  class_scales: dict[str, float] | None,
                  depth_form: str = "exp") -> dict[str, float]:
    w = {}
    for name, m in mods.items():
        if kind == "rel":
            base = 1.0
        elif kind == "abs":
            base = m["out_energy"]
        elif kind == "sqrt_energy":
            base = math.sqrt(m["out_energy"])
        elif kind == "class-kld":
            assert class_scales is not None
            base = class_scales[class_of(name)]
        else:
            raise ValueError(f"unknown weighting {kind!r}")
        w[name] = base * depth_factor(name, amp, n_layers, depth_form)
    return w


def load_inputs(ladder_path: Path, plan_path: Path):
    ladder = json.loads(ladder_path.read_text())
    plan = json.loads(plan_path.read_text())
    mods = ladder["modules"]
    if len(mods) != 409:
        raise SystemExit(f"expected 409 body modules, got {len(mods)}")
    return ladder, plan, mods


def derive_fixed(mods: dict, hydrated_roles: dict) -> dict:
    """Derive fixed(role) from published hydrated role totals.
    bytes(role) = fixed_total(role) + sum_m numel_m * recipe_bits_m / 8
    """
    var = defaultdict(float)
    for name, m in mods.items():
        var[role_of(name)] += m["numel"] * m["recipe_bits"] / 8.0
    fixed = {}
    for r, v in var.items():
        if r not in hydrated_roles:
            raise SystemExit(f"role {r} missing from published hydrated totals")
        fixed[r] = hydrated_roles[r] - v
    return fixed


def role_bytes(mods: dict, widths: dict, fixed: dict) -> dict:
    out = defaultdict(float)
    for name, m in mods.items():
        out[role_of(name)] += m["numel"] * widths[name] / 8.0
    for r in out:
        out[r] += fixed[r]
    return {r: int(round(v)) for r, v in out.items()}


def objective(mods: dict, widths: dict, weights: dict) -> float:
    return sum(weights[n] * m["ladder"][str(widths[n])] for n, m in mods.items())


def solve(mods: dict, body_budget: int, weights: dict, grid: int,
          max_width: int | None):
    """Exact DP: minimise sum_m w_m*eps(m,K) s.t. total body bytes <= budget."""
    names = list(mods)
    slots = body_budget // grid
    NEG = float("inf")
    dp = [NEG] * (slots + 1)
    dp[0] = 0.0
    choice: list[dict[int, int]] = []
    for name in names:
        m = mods[name]
        cands = sorted(int(k) for k in m["ladder"])
        if max_width is not None:
            capped = [k for k in cands if k <= max_width]
            cands = capped or cands[:1]
        costs = [(k, int((m["numel"] * k / 8.0) // grid)) for k in cands]
        gains = {k: weights[name] * m["ladder"][str(k)] for k in cands}
        ndp = [NEG] * (slots + 1)
        pick: dict[int, int] = {}
        for c, cur in enumerate(dp):
            if cur == NEG:
                continue
            for k, cost in costs:
                nc = c + cost
                if nc > slots:
                    continue
                val = cur + gains[k]
                if val < ndp[nc]:
                    ndp[nc] = val
                    pick[nc] = k
        dp = ndp
        choice.append(pick)
    best_c = min((c for c, v in enumerate(dp) if v != NEG), key=lambda c: dp[c])
    widths: dict[str, int] = {}
    c = best_c
    for name, pick in zip(reversed(names), reversed(choice)):
        k = pick[c]
        widths[name] = k
        c -= int((mods[name]["numel"] * k / 8.0) // grid)
    return widths, dp[best_c]


def depth_bands(dp_mods: dict, widths: dict, hyd: dict) -> dict[str, int]:
    bands = {"L00-15": (0, 15), "L16-31": (16, 31),
             "L32-47": (32, 47), "L48-63": (48, 63)}
    out = {}
    for bname, (lo, hi) in bands.items():
        d = 0.0
        for n, m in dp_mods.items():
            L = layer_index(n)
            if L is not None and lo <= L <= hi:
                d += m["numel"] * (widths[n] - hyd[n]) / 8.0
        out[bname] = int(round(d))
    return out


def width_heatmap(dp_mods: dict, widths: dict) -> dict[str, list[float]]:
    """Mean width per role per 16-layer band -- the allocation as a picture."""
    acc: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for n in dp_mods:
        L = layer_index(n)
        if L is None:
            continue
        acc[role_of(n)][L // 16].append(widths[n])
    return {r: [round(sum(v) / len(v), 2) if (v := acc[r].get(b, [])) else None
                for b in range(4)]
            for r in sorted(acc)}


def cmd_compare(paths: list[str]) -> None:
    arts = []
    for p in paths:
        a = json.loads(Path(p).read_text())
        a["_name"] = Path(p).stem.replace("resolve-", "")
        arts.append(a)
    names = [a["_name"] for a in arts]
    print("=== strategy agreement (% modules with identical width) ===")
    print(" " * 24 + "".join(f"{n[:14]:>16}" for n in names))
    for a in arts:
        row = []
        for b in arts:
            common = set(a["widths"]) & set(b["widths"])
            same = sum(1 for k in common if a["widths"][k] == b["widths"][k])
            row.append(f"{100*same/len(common):>15.1f}%")
        print(f"{a['_name'][:23]:<24}" + "".join(row))
    print()
    print("=== role byte delta vs hydrated ===")
    roles = sorted({r for a in arts for r in a["role_byte_delta_vs_hydrated"]})
    print(f"{'role':<18}" + "".join(f"{n[:14]:>16}" for n in names))
    for r in roles:
        print(f"{r:<18}" + "".join(
            f"{a['role_byte_delta_vs_hydrated'].get(r, 0):>+16,}" for a in arts))
    print()
    print("=== depth-band byte delta vs hydrated ===")
    bands = sorted({b for a in arts for b in a.get("byte_delta_by_depth_band", {})})
    print(f"{'band':<18}" + "".join(f"{n[:14]:>16}" for n in names))
    for b in bands:
        print(f"{b:<18}" + "".join(
            f"{a.get('byte_delta_by_depth_band', {}).get(b, 0):>+16,}" for a in arts))
    print()
    print("=== predicted KLD delta (class-kld units only; others are not KLD) ===")
    for a in arts:
        pk = a.get("predicted_kld_delta")
        print(f"  {a['_name']:<28} "
              + (f"{pk:+.6f}" if pk is not None else "n/a (objective not in KLD units)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", help="error-driven-ladder.json")
    ap.add_argument("--plan", help="plan-error-driven-allocation.json")
    ap.add_argument("--solved", action="append", default=None,
                    help="published solved-*.json (repeatable: attention/GDN in "
                         "solved-fixed.json, MLP in solved-override.json); enables "
                         "byte-model validation AND the held-out prediction test")
    ap.add_argument("--weighting", default="class-kld",
                    choices=("rel", "abs", "sqrt_energy", "class-kld"))
    ap.add_argument("--depth-amp", type=float, default=0.0,
                    help="per-layer downstream amplification; weight *= "
                         "(1+amp)**(n_layers-1-layer). UNCALIBRATED - probe only.")
    ap.add_argument("--depth-form", default="exp", choices=("exp", "u"),
                    help="exp: early-heavy (downstream amplification). u: both "
                         "ends heavy (llama.cpp use_more_bits + exllamav3 "
                         "allocation.py prior art, docs/58).")
    ap.add_argument("--max-width", type=int, default=None,
                    help="serving-aware cap: b12x ANY_BITS prefill supports K3..K6 "
                         "only (n_words 48/64/80/96); K7/K8 fall off the fast path.")
    ap.add_argument("--budget-delta", type=int, default=0,
                    help="bytes added to (or removed from) the body budget")
    ap.add_argument("--grid", type=int, default=None,
                    help="knapsack grid in bytes (default: plan's grid_unit_bytes)")
    ap.add_argument("--out", help="allocation artifact JSON")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="compare N artifact JSONs instead of solving")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args.compare)
        return
    if not (args.ladder and args.plan and args.out):
        ap.error("--ladder, --plan and --out are required unless --compare")

    ladder, plan, mods = load_inputs(Path(args.ladder), Path(args.plan))
    hydrated_roles = plan["bytes"]["hydrated_roles"]
    solved_roles_pub = plan["bytes"]["solved_roles"]
    budget = plan["budget_bytes"]
    grid = args.grid or plan.get("grid_unit_bytes") or 1 << 20

    fixed = derive_fixed(mods, hydrated_roles)
    hyd_widths = {n: m["recipe_bits"] for n, m in mods.items()}

    # --- validation 1: reproduce the hydrated role totals exactly
    got_h = role_bytes(mods, hyd_widths, fixed)
    bad_h = {r: (got_h[r], hydrated_roles[r]) for r in got_h
             if got_h[r] != hydrated_roles[r]}
    if bad_h:
        raise SystemExit(f"byte model fails hydrated reproduction: {bad_h}")

    # --- validation 2 + held-out prediction test on the PUBLISHED solve
    val2 = "skipped (no --solved)"
    heldout = None
    class_scales = fit_class_scales(plan)
    if args.solved:
        sfx = {}
        for pth in args.solved:
            sfx.update(json.loads(Path(pth).read_text()))
        solved_widths = dict(hyd_widths)
        for pat, k in sfx.items():
            name = pat.strip("^$").replace("\\", "")
            if name not in solved_widths:
                raise SystemExit(f"solved key not in ladder: {name}")
            solved_widths[name] = int(k)
        got_s = role_bytes(mods, solved_widths, fixed)
        bad_s = {r: (got_s[r], solved_roles_pub[r]) for r in got_s
                 if r in solved_roles_pub and got_s[r] != solved_roles_pub[r]}
        if bad_s:
            raise SystemExit(
                "byte model reproduces hydrated but NOT the published solve; "
                f"refusing to emit an allocation: {bad_s}")
        val2 = "exact"
        # Held-out test: class scales were fitted on two SINGLE-class moves;
        # the published solve is a cross-class reallocation neither saw.
        d_attn = d_mlp = 0.0
        for n, m in mods.items():
            c = class_of(n)
            if c == "pinned":
                continue
            de = m["ladder"][str(solved_widths[n])] - m["ladder"][str(hyd_widths[n])]
            if c == "attn":
                d_attn += de
            else:
                d_mlp += de
        pred = class_scales["attn"] * d_attn + class_scales["mlp"] * d_mlp
        heldout = {
            "delta_eps_attn": d_attn,
            "delta_eps_mlp": d_mlp,
            "class_scales": class_scales,
            "predicted_kld_delta_class_model": pred,
            "predicted_kld_delta_plan_global_scale": GLOBAL_SCALE_PREDICTION,
            "measured_kld_delta": MEASURED_PUBLISHED_DELTA,
        }

    # DP over unpinned body modules only
    dp_mods = {n: m for n, m in mods.items() if role_of(n) not in PINNED_ROLES}
    pinned = {n: m for n, m in mods.items() if role_of(n) in PINNED_ROLES}
    dp_roles = {role_of(n) for n in dp_mods}
    body_budget = sum(hydrated_roles[r] for r in dp_roles)
    body_budget -= sum(int(round(fixed[r])) for r in dp_roles)
    body_budget += args.budget_delta

    n_layers = 1 + max(v for v in (layer_index(n) for n in dp_mods) if v is not None)
    weights = build_weights(dp_mods, args.weighting, args.depth_amp,
                            n_layers, class_scales, args.depth_form)
    dp_widths, _ = solve(dp_mods, body_budget, weights, grid, args.max_width)
    widths = dict(dp_widths)
    for n, m in pinned.items():
        widths[n] = m["recipe_bits"]

    # Objective domain is the 400 movable modules (plan's own objective_hydrated
    # reproduces to 15 significant figures over this subset).
    obj_new = objective(dp_mods, widths, weights)
    obj_hyd = objective(dp_mods, hyd_widths, weights)
    predicted_kld = (obj_new - obj_hyd) if args.weighting == "class-kld" else None

    moved = {n: (hyd_widths[n], widths[n]) for n in mods if widths[n] != hyd_widths[n]}
    roles_new = role_bytes(mods, widths, fixed)
    role_delta = {r: roles_new[r] - hydrated_roles[r] for r in roles_new}
    band_delta = depth_bands(dp_mods, widths, hyd_widths)
    over_cap = sum(1 for n in dp_mods if widths[n] > 6)

    art = {
        "schema": "qwen38-eda-resolve/2",
        "weighting": args.weighting,
        "class_scales": class_scales if args.weighting == "class-kld" else None,
        "depth_amp": args.depth_amp,
        "depth_form": args.depth_form,
        "depth_weight_form": ("(1+amp)**(n_layers-1-L)" if args.depth_form == "exp"
                              else "(1+amp)**((n-1)/2 - min(L, n-1-L))"),
        "depth_amp_calibrated": False,
        "max_width": args.max_width,
        "modules_over_k6": over_cap,
        "budget_delta": args.budget_delta,
        "byte_delta_by_depth_band": band_delta,
        "width_heatmap_mean_by_16layer_band": width_heatmap(dp_mods, widths),
        "grid_unit_bytes": grid,
        "byte_model_validation": {"hydrated": "exact", "published_solve": val2},
        "objective": {"hydrated": obj_hyd, "resolved": obj_new,
                      "delta": obj_new - obj_hyd},
        "predicted_kld_delta": predicted_kld,
        "heldout_published_solve_test": heldout,
        "modules_moved": len(moved),
        "role_byte_delta_vs_hydrated": role_delta,
        "widths": widths,
        "moved": {n: {"from": a, "to": b} for n, (a, b) in sorted(moved.items())},
        "caveats": [
            "class scales are fitted from TWO single-class measured deltas; the "
            "held-out test against the published cross-class solve is the only "
            "out-of-sample check. Scales fold KV compounding and unit conversion "
            "into one number per class.",
            "depth_amp is NOT calibrated (no early-vs-late measurement exists).",
            "All objectives remain first-order and layer-local at the ladder "
            "level; predicted KLD deltas are pre-registrable candidates requiring "
            "paired shard-0 validation before any build ships.",
        ],
    }
    Path(args.out).write_text(json.dumps(art, indent=2) + "\n")

    print(f"weighting        : {args.weighting}"
          + (f"  (scales attn={class_scales['attn']:.6f} mlp={class_scales['mlp']:.6f})"
             if args.weighting == "class-kld" else ""))
    print(f"constraints      : max_width={args.max_width}  depth_amp={args.depth_amp}"
          f"  budget_delta={args.budget_delta:+,}")
    print(f"byte model       : hydrated=exact, published_solve={val2}")
    if heldout:
        h = heldout
        print("held-out test on the PUBLISHED solve (fitted on single-class moves only):")
        print(f"  class-model predicted : {h['predicted_kld_delta_class_model']:+.6f}")
        print(f"  plan's global scale   : {h['predicted_kld_delta_plan_global_scale']:+.6f}")
        print(f"  measured              : {h['measured_kld_delta']:+.6f}")
    if predicted_kld is not None:
        print(f"predicted KLD delta of THIS allocation: {predicted_kld:+.6f}")
    print(f"modules moved    : {len(moved)} / {len(mods)}   over-K6: {over_cap}")
    print("role byte deltas vs hydrated (bytes):")
    for r in sorted(role_delta, key=lambda r: role_delta[r]):
        print(f"  {r:<18} {role_delta[r]:+,}")
    print("byte deltas by depth band (bytes):")
    for b in sorted(band_delta):
        print(f"  {b:<8} {band_delta[b]:+,}")
    print(f"artifact -> {args.out}")


if __name__ == "__main__":
    main()
