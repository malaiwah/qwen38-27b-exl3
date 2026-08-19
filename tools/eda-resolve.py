#!/usr/bin/env python3
"""Re-solve the EXL3 error-driven bit allocation under a chosen error weighting.

Background
----------
`malaiwah/Qwen3.8-27B-EXL3-EDA-research` solved a per-matrix bit allocation with
the `rel` objective (`sum_m eps(m,K)`, every module weighted equally).  That build
measured WORSE than the hydrated recipe (+0.000366 mean KLD on shard 0, 470/512
contexts) because `rel` is blind to KV-cache error compounding: it moved bytes
away from attention/GDN and into MLP, the opposite of what the attribution work
measured (docs/57).  Only `abs` (weight = out_energy) and `sqrt_energy`
(weight = sqrt(out_energy)) recovered the correct sign on all four calibration
pairs.

This tool re-runs the allocation with a selectable weighting.  It is exact
dynamic programming over the byte grid, not greedy -- matching the original
solver.

Inputs (both published, no GPU required)
----------------------------------------
* ladder: `malaiwah/qwen38-27b-fidelity-suite-v5` (dataset) ->
  `captures/shard-0000/error-driven-ladder.json`; 409 body modules, each with
  `numel`, `out_energy`, `recipe_bits`, and `ladder{width: proxy_err}`.
  `proxy_err = tr(E^T H E) / tr(W^T H W)` -- exllamav3's own per-module
  Hessian-weighted relative quantization error.
  `out_energy = tr(W^T H W) / count` -- mean per-calibration-row output energy.
* plan: `malaiwah/Qwen3.8-27B-EXL3-EDA-research` ->
  `allocation/plan-error-driven-allocation.json`; supplies the byte law, the
  budget, the published per-role byte totals, and the measured
  KLD-per-objective-unit scale.

Byte law (docs/34 SS2): `bytes(role,K) = fixed(role) + params(role)*K/8`.
`fixed(role)` is not published directly, so it is DERIVED from the published
hydrated role totals and then VALIDATED by reproducing the published *solved*
role totals exactly.  If that validation fails the tool refuses to emit an
allocation -- a byte model that cannot reproduce a known solve must not be
trusted to produce a new one.

Honesty note carried into the output
------------------------------------
The objective is first-order and layer-local: the proxy sees no error
accumulation between layers, and the ladder itself was measured under
hydrated-recipe propagation.  `sqrt_energy` was moreover selected knowing the
`rel` build's answer, and its implied KLD scale is a factor ~2.47 uncertain
(worst leave-one-out ratio).  A predicted KLD delta from this tool is a
pre-registrable candidate, never a result: it must be validated as a PAIRED
comparison against the hydrated recipe on shard 0, shipping only if the paired
interval excludes zero in the new allocation's favour.
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


def role_of(name: str) -> str:
    if name == "lm_head":
        return "lm_head"
    if name.startswith("mtp."):
        return "mtp_draft"
    for frag, role in ROLE_PATTERNS:
        if frag in name:
            return role
    raise KeyError(f"no role for module {name!r}")


def weight_fn(kind: str):
    if kind == "rel":
        return lambda out_energy: 1.0
    if kind == "abs":
        return lambda out_energy: out_energy
    if kind == "sqrt_energy":
        return lambda out_energy: math.sqrt(out_energy)
    raise ValueError(f"unknown weighting {kind!r}")


LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_index(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def depth_factor(name: str, amp: float, n_layers: int) -> float:
    """Downstream-amplification factor for error injected at this module's layer.

    Error injected at layer L is subsequently transformed by layers L+1..n-1.
    If each downstream layer amplifies injected error by a mean factor (1+amp),
    the injected error's contribution at the readout scales as
    (1+amp)**(n-1-L) -- exponential in REMAINING depth, so early layers are
    weighted more heavily than late ones.

    amp = 0.0 reproduces the depth-blind behaviour exactly (factor 1 for every
    module), which is what the published solver and every weighting in the
    plan's objective table do: the ladder's proxy_err is module-local, measured
    with fixed hydrated propagation, so it carries no inter-layer accumulation
    (plan `predicted.note`, docs/57 SS1).

    amp is NOT calibrated by any measurement in this repository. Every
    layer-range fidelity experiment we have (receipts/layer-range-dial-2026-08-19.md)
    converted ranges starting at layer 0, so there is no early-vs-late contrast
    to fit against, and that receipt's 13-layer KLD was predicted rather than
    measured. Treat any amp > 0 as a sensitivity probe, never as a result.
    """
    if amp == 0.0:
        return 1.0
    L = layer_index(name)
    if L is None:
        return 1.0
    return (1.0 + amp) ** (n_layers - 1 - L)


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
    count = defaultdict(int)
    for name, m in mods.items():
        r = role_of(name)
        var[r] += m["numel"] * m["recipe_bits"] / 8.0
        count[r] += 1
    fixed = {}
    for r, v in var.items():
        if r not in hydrated_roles:
            raise SystemExit(f"role {r} missing from published hydrated totals")
        fixed[r] = hydrated_roles[r] - v
    return fixed, count


def role_bytes(mods: dict, widths: dict, fixed: dict) -> dict:
    out = defaultdict(float)
    for name, m in mods.items():
        r = role_of(name)
        out[r] += m["numel"] * widths[name] / 8.0
    for r in out:
        out[r] += fixed[r]
    return {r: int(round(v)) for r, v in out.items()}


def objective(mods: dict, widths: dict, wfn, amp: float = 0.0, n_layers: int = 64) -> float:
    tot = 0.0
    for name, m in mods.items():
        eps = m["ladder"][str(widths[name])]
        tot += wfn(m["out_energy"]) * depth_factor(name, amp, n_layers) * eps
    return tot


def solve(mods: dict, fixed: dict, body_budget: int, wfn, grid: int,
          amp: float = 0.0, n_layers: int = 64):
    """Exact DP: minimise sum_m w_m*eps(m,K) subject to total body bytes <= budget.

    Byte cost per module is numel*K/8; the grid quantises the knapsack axis.
    """
    names = list(mods)
    slots = body_budget // grid
    NEG = float("inf")
    # dp[c] = best objective using processed modules with c grid-cells spent
    dp = [NEG] * (slots + 1)
    dp[0] = 0.0
    choice: list[dict[int, int]] = []
    for name in names:
        m = mods[name]
        cands = sorted(int(k) for k in m["ladder"])
        costs = [(k, int((m["numel"] * k / 8.0) // grid)) for k in cands]
        df = depth_factor(name, amp, n_layers)
        gains = {k: wfn(m["out_energy"]) * df * m["ladder"][str(k)] for k in cands}
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
    # backtrack
    widths: dict[str, int] = {}
    c = best_c
    for name, pick in zip(reversed(names), reversed(choice)):
        k = pick[c]
        widths[name] = k
        c -= int((mods[name]["numel"] * k / 8.0) // grid)
    return widths, dp[best_c]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", required=True, help="error-driven-ladder.json")
    ap.add_argument("--plan", required=True, help="plan-error-driven-allocation.json")
    ap.add_argument("--solved", action="append", default=None,
                    help="published solved-*.json (repeatable: the original solve "
                         "splits attention/GDN into solved-fixed.json and MLP into "
                         "solved-override.json); enables byte-model validation")
    ap.add_argument("--weighting", default="sqrt_energy",
                    choices=("rel", "abs", "sqrt_energy"))
    ap.add_argument("--grid", type=int, default=None,
                    help="knapsack grid in bytes (default: plan's grid_unit_bytes)")
    ap.add_argument("--depth-amp", type=float, default=0.0,
                    help="per-layer downstream amplification of injected error; "
                         "module weight *= (1+amp)**(n_layers-1-layer). 0 (default) "
                         "reproduces the depth-blind published behaviour. UNCALIBRATED "
                         "- sensitivity probe only.")
    ap.add_argument("--out", required=True, help="allocation artifact JSON")
    args = ap.parse_args()

    ladder, plan, mods = load_inputs(Path(args.ladder), Path(args.plan))
    hydrated_roles = plan["bytes"]["hydrated_roles"]
    solved_roles_pub = plan["bytes"]["solved_roles"]
    budget = plan["budget_bytes"]
    grid = args.grid or plan.get("grid_unit_bytes") or 1 << 20

    fixed, counts = derive_fixed(mods, hydrated_roles)
    hyd_widths = {n: m["recipe_bits"] for n, m in mods.items()}

    # --- validation 1: reproduce the hydrated role totals exactly
    got_h = role_bytes(mods, hyd_widths, fixed)
    bad_h = {r: (got_h[r], hydrated_roles[r]) for r in got_h if got_h[r] != hydrated_roles[r]}
    if bad_h:
        raise SystemExit(f"byte model fails hydrated reproduction: {bad_h}")

    # --- validation 2: reproduce the published SOLVED role totals exactly
    val2 = "skipped (no --solved)"
    if args.solved:
        sfx = {}
        for pth in args.solved:
            sfx.update(json.loads(Path(pth).read_text()))
        # keys are ^regex$ of the module name
        solved_widths = dict(hyd_widths)
        for pat, k in sfx.items():
            name = pat.strip("^$").replace("\\", "")
            if name not in solved_widths:
                raise SystemExit(f"solved-fixed key not in ladder: {name}")
            solved_widths[name] = int(k)
        got_s = role_bytes(mods, solved_widths, fixed)
        bad_s = {r: (got_s[r], solved_roles_pub[r]) for r in got_s
                 if r in solved_roles_pub and got_s[r] != solved_roles_pub[r]}
        if bad_s:
            raise SystemExit(
                "byte model reproduces hydrated but NOT the published solve; "
                f"refusing to emit an allocation: {bad_s}")
        val2 = "exact"

    # The DP ranges only over unpinned body modules; pinned modules keep
    # recipe_bits and their bytes are removed from the movable budget.
    dp_mods = {n: m for n, m in mods.items() if role_of(n) not in PINNED_ROLES}
    pinned = {n: m for n, m in mods.items() if role_of(n) in PINNED_ROLES}
    dp_roles = {role_of(n) for n in dp_mods}
    body_budget = sum(hydrated_roles[r] for r in dp_roles)
    body_budget -= sum(int(round(fixed[r])) for r in dp_roles)

    wfn = weight_fn(args.weighting)
    n_layers = 1 + max(v for v in (layer_index(n) for n in dp_mods) if v is not None)
    dp_widths, obj_dp = solve(dp_mods, fixed, body_budget, wfn, grid,
                              args.depth_amp, n_layers)
    widths = dict(dp_widths)
    for n, m in pinned.items():
        widths[n] = m["recipe_bits"]
    # Objective domain is the 400 movable modules, NOT all 409: recomputing the
    # plan's own objective_hydrated over this subset reproduces its published
    # value to 15 significant figures (0.07535511617344567 vs 0.07535511617344577),
    # which pins the domain beyond doubt. lm_head and the MTP draft carry no
    # out_energy in the ladder and are pinned anyway.
    obj_new = objective(dp_mods, widths, wfn, args.depth_amp, n_layers)
    obj_hyd = objective(dp_mods, hyd_widths, wfn, args.depth_amp, n_layers)

    moved = {n: (hyd_widths[n], widths[n]) for n in mods if widths[n] != hyd_widths[n]}
    roles_new = role_bytes(mods, widths, fixed)
    role_delta = {r: roles_new[r] - hydrated_roles[r] for r in roles_new}

    # byte delta by depth quartile, so a depth-weighted solve's effect is visible
    bands = {"L00-15": (0, 15), "L16-31": (16, 31), "L32-47": (32, 47), "L48-63": (48, 63)}
    band_delta = {}
    for bname, (lo, hi) in bands.items():
        d = 0.0
        for n, m in dp_mods.items():
            L = layer_index(n)
            if L is None or not (lo <= L <= hi):
                continue
            d += m["numel"] * (widths[n] - hyd_widths[n]) / 8.0
        band_delta[bname] = int(round(d))

    scale = plan["objective"].get("kld_per_objective_unit")
    art = {
        "schema": "qwen38-eda-resolve/1",
        "weighting": args.weighting,
        "depth_amp": args.depth_amp,
        "depth_weight_form": "(1+amp)**(n_layers-1-layer_index)",
        "depth_amp_calibrated": False,
        "byte_delta_by_depth_band": band_delta,
        "grid_unit_bytes": grid,
        "budget_bytes": budget,
        "body_budget_bytes": body_budget,
        "byte_model_validation": {"hydrated": "exact", "published_solve": val2},
        "objective": {
            "definition": {"rel": "sum_m eps", "abs": "sum_m out_energy*eps",
                           "sqrt_energy": "sum_m sqrt(out_energy)*eps"}[args.weighting],
            "hydrated": obj_hyd,
            "resolved": obj_new,
            "delta_resolved_minus_hydrated": obj_new - obj_hyd,
        },
        "kld_per_objective_unit_from_plan": scale,
        "modules_moved": len(moved),
        "role_byte_delta_vs_hydrated": role_delta,
        "widths": widths,
        "moved": {n: {"from": a, "to": b} for n, (a, b) in sorted(moved.items())},
        "caveats": [
            "depth_amp is NOT calibrated by any measurement in this repo: every "
            "layer-range fidelity experiment converted ranges starting at layer 0 "
            "(receipts/layer-range-dial-2026-08-19.md), so there is no early-vs-late "
            "contrast to fit, and that receipt's 13-layer KLD was predicted not "
            "measured. Any amp>0 allocation is a sensitivity probe.",
            "First-order and layer-local: the proxy sees no inter-layer error "
            "accumulation, and the ladder was measured under hydrated-recipe "
            "propagation.",
            "sqrt_energy was selected knowing the rel build's outcome; its implied "
            "KLD scale is ~2.47x uncertain (worst leave-one-out ratio).",
            "A predicted KLD delta here is a pre-registrable candidate, not a "
            "result. Validate as a PAIRED shard-0 comparison against hydrated and "
            "ship only if the paired interval excludes zero in its favour.",
        ],
    }
    Path(args.out).write_text(json.dumps(art, indent=2) + "\n")

    print(f"weighting        : {args.weighting}")
    print(f"byte model       : hydrated=exact, published_solve={val2}")
    print(f"objective hyd    : {obj_hyd:.9g}")
    print(f"objective solved : {obj_new:.9g}  (delta {obj_new - obj_hyd:+.9g})")
    print(f"modules moved    : {len(moved)} / {len(mods)}")
    print("role byte deltas vs hydrated (bytes):")
    for r in sorted(role_delta, key=lambda r: role_delta[r]):
        print(f"  {r:<18} {role_delta[r]:+,}")
    print("byte deltas by depth band (bytes):")
    for b in sorted(band_delta):
        print(f"  {b:<8} {band_delta[b]:+,}")
    print(f"artifact -> {args.out}")


if __name__ == "__main__":
    main()
