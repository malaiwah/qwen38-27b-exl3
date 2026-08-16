#!/usr/bin/env python3
"""Error-driven per-module bit allocation for Qwen3.8-27B, at a fixed serialized-byte budget.

The converter allocates bits by a static priority order, not by measured error
([docs/04](../docs/04-exllamav3-toolchain.md) Gap 1). It does, however, print a
per-module proxy error - `tr(E^T H E) / tr(W^T H W)`, the Hessian-weighted relative
quantization error - at whatever width it assigned. `tools/ladder_pass.py` turns that
one point per module into a curve by re-quantizing each module at every candidate
width against the same captured Hessian. This tool consumes that ladder and solves

    minimise  sum_m  w_m * eps(m, K_m)      subject to   sum_role bytes(role) <= budget

exactly, by dynamic programming over the byte grid. The byte law is
[docs/34](../docs/34-vram-class-profiles.md) §2,

    bytes(role, K) = fixed(role) + params(role) * K / 8

applied per module (`fixed(m) = 2*(in+out)` fp16 `suh`/`svh` bytes plus one int32
codebook scalar, plus the role's BF16/F16 companions), and it is asserted byte-for-byte
against both published manifests before anything is solved: a byte model that cannot
reproduce the two shipped checkpoints has no business proposing a third.

Every module byte cost is an integer multiple of 655,360 B = the smallest module's
params/8, so the DP grid is exact rather than rounded - 25,664 grid points at the
hydrated budget, which makes the optimum provable instead of greedy.

Two objectives are offered because the proxy error is *relative*, and turning it into a
model-wide sum requires an assumption about per-module sensitivity:

  * `rel`  - w_m = 1. Every module's relative output perturbation counts the same.
             This is the assumption behind uniform-bitrate quantization.
  * `abs`  - w_m = tr(W^T H W)/count, the module's mean per-row calibration output
             energy, so w_m * eps is absolute output error energy. This is the GPTQ
             objective under isotropic downstream sensitivity.

Neither is a law, so both are scored against two *measured* KLD deltas between
published checkpoints that differ in exactly one role group (attention K6 -> K5, and
the MLP stack K5/K5/K6 -> K4), and the objective whose implied error-to-KLD scale is
consistent across those two independent deltas is the one used. That scale also
converts the solved objective delta into a predicted KLD, which is what gets
pre-registered before a conversion is allowed to run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

UNIT = 655_360           # bytes: params/8 of the smallest quantized module (k_proj/v_proj)
FULL_ATTN_LAYERS = [i for i in range(64) if i % 4 == 3]
LIN_ATTN_LAYERS = [i for i in range(64) if i % 4 != 3]

# BF16/F16 companions that do not move with K, from docs/34 §2
COMPANION = {
    "full_attention": 16 * 2 * 256 * 2,
    "linear_attention": 48 * 2 * (48 * 5120 * 2) + 48 * (10240 * 4 * 2 + 48 * 2 + 48 * 2 + 128 * 2),
    "mtp_draft": 52_224,
    "mlp_gate_proj": 0, "mlp_up_proj": 0, "mlp_down_proj": 0, "lm_head": 0,
}
FLAT_ROLES = {"embed_tokens": 2_542_796_800, "vision_tower": 921_460_192,
              "norms_and_small": 1_320_960}


def inventory() -> dict[str, tuple[int, int, str]]:
    """module key -> (in_features, out_features, role). Shapes are the source model's."""
    mods: dict[str, tuple[int, int, str]] = {}
    for i in range(64):
        p = f"model.language_model.layers.{i}"
        if i in FULL_ATTN_LAYERS:
            for nm, inf, out in (("q_proj", 5120, 12288), ("k_proj", 5120, 1024),
                                 ("v_proj", 5120, 1024), ("o_proj", 6144, 5120)):
                mods[f"{p}.self_attn.{nm}"] = (inf, out, "full_attention")
        else:
            for nm, inf, out in (("in_proj_qkv", 5120, 10240), ("in_proj_z", 5120, 6144),
                                 ("out_proj", 6144, 5120)):
                mods[f"{p}.linear_attn.{nm}"] = (inf, out, "linear_attention")
        for nm, inf, out in (("gate_proj", 5120, 17408), ("up_proj", 5120, 17408),
                             ("down_proj", 17408, 5120)):
            mods[f"{p}.mlp.{nm}"] = (inf, out, "mlp_" + nm)
    for nm, inf, out in (("self_attn.q_proj", 5120, 12288), ("self_attn.k_proj", 5120, 1024),
                         ("self_attn.v_proj", 5120, 1024), ("self_attn.o_proj", 6144, 5120),
                         ("mlp.gate_proj", 5120, 17408), ("mlp.up_proj", 5120, 17408),
                         ("mlp.down_proj", 17408, 5120)):
        mods[f"mtp.layers.0.{nm}"] = (inf, out, "mtp_draft")
    mods["mtp.input_layer.eh_proj"] = (5120, 10240, "mtp_draft")
    mods["lm_head"] = (5120, 248320, "lm_head")
    return mods


MODS = inventory()


def fixed_bytes(key: str) -> int:
    inf, out, _ = MODS[key]
    return 2 * (inf + out) + 4


def numel(key: str) -> int:
    inf, out, _ = MODS[key]
    return inf * out


def role_bytes(bits: dict[str, int], vision_bytes: int | None = None) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for k, (_, _, role) in MODS.items():
        out[role] += fixed_bytes(k) + numel(k) * bits[k] // 8
    for role, c in COMPANION.items():
        out[role] += c
    for role, b in FLAT_ROLES.items():
        out[role] = b
    if vision_bytes is not None:
        out["vision_tower"] = vision_bytes
    return dict(out)


def hydrated_bits() -> dict[str, int]:
    bits = {}
    for k, (_, _, role) in MODS.items():
        if role in ("full_attention", "linear_attention"):
            bits[k] = 6
        elif role in ("mlp_gate_proj", "mlp_up_proj"):
            bits[k] = 5
        elif role == "mlp_down_proj":
            bits[k] = 6
        elif role == "lm_head":
            bits[k] = 6
        else:
            bits[k] = (6 if "self_attn" in k else
                       5 if ("gate_proj" in k or "up_proj" in k) else
                       6 if "down_proj" in k else 4)
    return bits


def context_bits() -> dict[str, int]:
    bits = hydrated_bits()
    for k, (_, _, role) in MODS.items():
        if role in ("full_attention", "linear_attention"):
            bits[k] = 5
        elif role == "mtp_draft" and "self_attn" in k:
            bits[k] = 5
    return bits


def k4_mlp_bits() -> dict[str, int]:
    """The published K4 build's MLP: all K4. Used only as an error-model check."""
    bits = hydrated_bits()
    for k, (_, _, role) in MODS.items():
        if role.startswith("mlp_") and k.startswith("model."):
            bits[k] = 4
    return bits


def assert_byte_law(receipts: Path) -> dict:
    """Fail closed unless the byte model reproduces both published manifests exactly."""
    checks = {}
    for label, fn, name in (("hydrated", hydrated_bits, "hydrated-quantization-manifest.json"),
                            ("context", context_bits, "context-quantization-manifest.json")):
        man = json.loads((receipts / name).read_text())["roles"]
        model = role_bytes(fn(), vision_bytes=man["vision_tower"]["bytes"])
        bad = {r: (model[r], man[r]["bytes"]) for r in man if model[r] != man[r]["bytes"]}
        if bad:
            raise SystemExit(f"byte law does not reproduce {label} manifest: {bad}")
        checks[label] = {"manifest": name, "roles": len(man),
                         "total_bytes": sum(v["bytes"] for v in man.values()), "exact": True}
    return checks


def load_ladder(path: Path) -> dict:
    d = json.loads(path.read_text())
    lad = {}
    for key, rec in d["modules"].items():
        if not rec.get("ladder"):
            continue
        eps = {int(k): float(v) for k, v in rec["ladder"].items() if v is not None and v >= 0}
        if not eps:
            continue
        lad[key] = {"eps": eps, "numel": rec["numel"], "out_energy": rec.get("out_energy"),
                    "recipe_bits": rec.get("recipe_bits"), "q_fallback": rec.get("q_fallback", False)}
    return {"meta": {k: v for k, v in d.items() if k != "modules"}, "modules": lad}


def weights(lad: dict, objective: str) -> dict[str, float]:
    if objective == "rel":
        return {k: 1.0 for k in lad}
    if objective == "abs":
        return {k: float(v["out_energy"]) for k, v in lad.items()}
    raise SystemExit(f"unknown objective {objective}")


def total_error(lad: dict, w: dict[str, float], bits: dict[str, int],
                keys: list[str] | None = None) -> float:
    keys = keys if keys is not None else list(lad)
    t = 0.0
    for k in keys:
        eps = lad[k]["eps"]
        b = bits[k]
        if b not in eps:
            raise SystemExit(f"no ladder rung for {k} at K{b}")
        t += w[k] * eps[b]
    return t


def solve(lad: dict, w: dict[str, float], free: list[str], budget_units: int,
          min_bits: int) -> tuple[dict[str, int], float]:
    """Exact DP over the 655,360-byte grid. Returns (bits for free modules, objective)."""
    n = len(free)
    U = budget_units
    units = [numel(k) // 8 // UNIT for k in free]
    for k, u in zip(free, units):
        if numel(k) // 8 != u * UNIT:
            raise SystemExit(f"{k}: params/8 is not a multiple of {UNIT}")
    cands = [sorted(kk for kk in lad[k]["eps"] if kk >= min_bits) for k in free]
    INF = float("inf")
    dp = np.full(U + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((n, U + 1), dtype=np.int8)
    for j in range(n):
        ndp = np.full(U + 1, INF)
        cj, uj, wj, ej = choice[j], units[j], w[free[j]], lad[free[j]]["eps"]
        for ci, Kc in enumerate(cands[j]):
            c = uj * Kc
            if c > U:
                continue
            vals = dp[: U + 1 - c] + wj * ej[Kc]
            tgt = ndp[c:]
            better = vals < tgt
            tgt[better] = vals[better]
            cj[c:][better] = ci
        dp = ndp
    u_best = int(np.argmin(dp))
    if not math.isfinite(dp[u_best]):
        raise SystemExit("no feasible allocation within budget")
    bits, u = {}, u_best
    for j in range(n - 1, -1, -1):
        Kc = cands[j][int(choice[j, u])]
        bits[free[j]] = Kc
        u -= units[j] * Kc
    assert u == 0, u
    return bits, float(dp[u_best])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--receipts", default=str(Path(__file__).resolve().parent.parent / "receipts"))
    ap.add_argument("--out", required=True, help="plan JSON")
    ap.add_argument("--fixed-out", default=None, help="EXL3_BITS_FIXED spec (attention)")
    ap.add_argument("--override-out", default=None, help="EXL3_BITS_OVERRIDE spec (MLP)")
    ap.add_argument("--objective", default="auto", choices=("auto", "rel", "abs"))
    ap.add_argument("--min-bits", type=int, default=3)
    ap.add_argument("--budget-bytes", type=int, default=None,
                    help="serialized payload ceiling; default = hydrated manifest total")
    args = ap.parse_args()

    receipts = Path(args.receipts)
    byte_checks = assert_byte_law(receipts)

    hyd = hydrated_bits()
    hyd_roles = role_bytes(hyd)
    budget = args.budget_bytes or sum(hyd_roles.values())

    L = load_ladder(Path(args.ladder))
    lad = L["modules"]
    # free variables: the budgeted body. Head and MTP are recipe-pinned, identical to hydrated.
    free = [k for k in lad if MODS[k][2] in ("full_attention", "linear_attention",
                                             "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")]
    free.sort()
    missing = [k for k, (_, _, r) in MODS.items()
               if r in ("full_attention", "linear_attention", "mlp_gate_proj", "mlp_up_proj",
                        "mlp_down_proj") and k not in lad]
    if missing:
        raise SystemExit(f"ladder is missing {len(missing)} body modules, e.g. {missing[:4]}")

    pinned_bytes = sum(fixed_bytes(k) + numel(k) * hyd[k] // 8
                       for k, (_, _, r) in MODS.items() if r in ("lm_head", "mtp_draft"))
    pinned_bytes += COMPANION["lm_head"] + COMPANION["mtp_draft"] + sum(FLAT_ROLES.values())
    body_fixed = sum(fixed_bytes(k) for k in free) + COMPANION["full_attention"] \
        + COMPANION["linear_attention"]
    body_variable_budget = budget - pinned_bytes - body_fixed
    if body_variable_budget % UNIT:
        raise SystemExit(f"budget residue {body_variable_budget % UNIT} B is not a multiple of {UNIT}")
    U = body_variable_budget // UNIT

    # ---- error model validation on two measured single-role-group KLD deltas ----
    ctx, k4 = context_bits(), k4_mlp_bits()
    attn_keys = [k for k in free if MODS[k][2] in ("full_attention", "linear_attention")]
    mlp_keys = [k for k in free if MODS[k][2].startswith("mlp_")]
    MEASURED = {
        "attention_K6_to_K5": {"delta_kld": 0.003409 - 0.002700,
                               "from": "receipts shard-0 report-hyd 0.002700 -> report-ctx 0.003409",
                               "keys": "attn", "a": hyd, "b": ctx},
        "mlp_K5K5K6_to_K4": {"delta_kld": 0.010345 - 0.003141,
                             "from": "receipts shard-0 report-k5k6 0.003141 -> report-k4 0.010345 "
                                     "(identical online-K6 attention and K6 head; MTP is not in the "
                                     "scored forward pass)",
                             "keys": "mlp", "a": hyd, "b": k4},
    }
    validation = {}
    for obj in ("rel", "abs"):
        w = weights(lad, obj)
        rows = {}
        for name, spec in MEASURED.items():
            keys = attn_keys if spec["keys"] == "attn" else mlp_keys
            pred = total_error(lad, w, spec["b"], keys) - total_error(lad, w, spec["a"], keys)
            rows[name] = {"predicted_objective_delta": pred, "measured_kld_delta": spec["delta_kld"],
                          "implied_scale_kld_per_objective_unit": spec["delta_kld"] / pred,
                          "source": spec["from"]}
        scales = [r["implied_scale_kld_per_objective_unit"] for r in rows.values()]
        ratio = max(scales) / min(scales)
        validation[obj] = {"deltas": rows, "scale_ratio_across_the_two_deltas": ratio,
                           "scale_geometric_mean": math.exp(sum(math.log(s) for s in scales) / len(scales))}

    chosen = args.objective
    if chosen == "auto":
        chosen = min(validation, key=lambda o: validation[o]["scale_ratio_across_the_two_deltas"])
    w = weights(lad, chosen)
    scale = validation[chosen]["scale_geometric_mean"]

    solved_free, obj_val = solve(lad, w, free, U, args.min_bits)
    bits = dict(hyd)
    bits.update(solved_free)
    roles = role_bytes(bits)
    total = sum(roles.values())
    if total > budget:
        raise SystemExit(f"solved allocation is over budget: {total} > {budget}")

    obj_hyd = total_error(lad, w, hyd, free)
    delta_obj = obj_val - obj_hyd
    per_role_bits = defaultdict(lambda: defaultdict(int))
    for k, b in bits.items():
        per_role_bits[MODS[k][2]][b] += 1

    plan = {
        "schema": "qwen38-error-driven-allocation-plan/1",
        "written_before_conversion": True,
        "budget_bytes": budget,
        "budget_source": "receipts/hydrated-quantization-manifest.json role total (serialized payload)",
        "byte_law": "bytes(role,K) = fixed(role) + params(role)*K/8, docs/34 §2, per module",
        "byte_law_validation": byte_checks,
        "grid_unit_bytes": UNIT,
        "grid_points": U,
        "solver": "exact dynamic programming over the byte grid (not greedy)",
        "ladder": {"source": str(args.ladder), "modules": len(lad),
                   "meta": L["meta"]},
        "objective": {"chosen": chosen, "requested": args.objective,
                      "rule": "the objective whose implied KLD-per-objective-unit scale is most "
                              "consistent across the two measured single-role-group deltas",
                      "definition": {"rel": "sum_m eps(m,K)",
                                     "abs": "sum_m out_energy_m * eps(m,K)"},
                      "validation": validation,
                      "kld_per_objective_unit": scale},
        "predicted": {
            "objective_hydrated": obj_hyd,
            "objective_solved": obj_val,
            "objective_delta_solved_minus_hydrated": delta_obj,
            "kld_delta_solved_minus_hydrated": delta_obj * scale,
            "hydrated_measured_shard0_mean_kld": 0.002700,
            "predicted_shard0_mean_kld": 0.002700 + delta_obj * scale,
            "note": "first-order and layer-local: the proxy sees no error accumulation between "
                    "layers, and the ladder was measured with hydrated-recipe propagation",
        },
        "bytes": {
            "hydrated_roles": hyd_roles, "hydrated_total": sum(hyd_roles.values()),
            "solved_roles": roles, "solved_total": total,
            "headroom_bytes": budget - total,
            "solved_gib": total / 2 ** 30,
        },
        "bit_histogram_by_role": {r: dict(sorted(v.items())) for r, v in sorted(per_role_bits.items())},
        "average_body_bits": sum(numel(k) * bits[k] for k in free) / sum(numel(k) for k in free),
        "hydrated_average_body_bits": sum(numel(k) * hyd[k] for k in free) / sum(numel(k) for k in free),
        "solved_bits": {k: bits[k] for k in sorted(bits)},
        "hydrated_bits": {k: hyd[k] for k in sorted(hyd)},
        "changed_from_hydrated": {k: [hyd[k], bits[k]] for k in sorted(free) if bits[k] != hyd[k]},
    }
    Path(args.out).write_text(json.dumps(plan, indent=1))

    # EXL3 spec files: exact-key regexes, mirroring build_hydrated.sh's split of the
    # mechanism (attention pinned before allocation, MLP overridden after it).
    def esc(k): return "^" + re.escape(k) + "$"
    if args.fixed_out:
        spec = {esc(k): bits[k] for k in sorted(bits)
                if MODS[k][2] in ("full_attention", "linear_attention")
                or (MODS[k][2] == "mtp_draft" and "self_attn" in k)}
        Path(args.fixed_out).write_text(json.dumps(spec, indent=1))
    if args.override_out:
        spec = {esc(k): bits[k] for k in sorted(bits)
                if MODS[k][2].startswith("mlp_")
                or (MODS[k][2] == "mtp_draft" and ".mlp." in k)}
        Path(args.override_out).write_text(json.dumps(spec, indent=1))

    print(json.dumps({
        "objective": chosen,
        "scale_ratio": {o: validation[o]["scale_ratio_across_the_two_deltas"] for o in validation},
        "objective_delta": delta_obj,
        "predicted_kld": plan["predicted"]["predicted_shard0_mean_kld"],
        "solved_total_bytes": total, "budget_bytes": budget, "headroom": budget - total,
        "avg_body_bits": plan["average_body_bits"],
        "changed_modules": len(plan["changed_from_hydrated"]),
    }, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
