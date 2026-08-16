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

# no third-party imports: this must run on the box's bare python3, which has no numpy

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
    mods["mtp.fc"] = (5120, 10240, "mtp_draft")   # the MTP input projection, at the -mb width
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


# Candidate widths, and the measured law that lets ONE width stand in for the whole curve.
#
# Measured on the five-rung ladder of all 400 body modules (receipts/error-driven-ladder.json):
#
#     log eps(m, K) = a_m + s(K)
#
# one constant per module and ONE universal shape, pinned s(5) = 0. Fitted on 8,000 ordered rung
# pairs, the shape's implied per-bit ratio is not constant - it declines smoothly from 3.860 at
# K3->K4 to 3.559 at K7->K8, i.e. each further bit buys slightly less - and the decline shows up
# in every module class with very little scatter (the 192 MLP modules put sd 0.003-0.011 on each
# rung pair). Predicting a held-out width from ONE measured width and this shape lands within
# 1.175 % mean absolute error, p95 3.86 %, with no systematic bias by rung; the single-constant
# geometric approximation eps = c_m * 3.7294**-K is 3.03 % / 6.88 % and biased at the ends.
#
# What it buys: the allocation solved from ONE K5 rung per module plus this shape agreed with the
# five-rung solve on 396 of 400 modules - the four disagreements one bit at a threshold - and
# recovered 99.98 % of the objective improvement at identical bytes. The input is then an ordinary
# teed conversion log, ~25 min of GPU, instead of a ~2 h five-rung measurement pass. Rerunning it
# on convert-ctx.log, a log from a DIFFERENT recipe that already existed before this work, also
# reproduced 396 of 400 and 100.0 % of the improvement.
#
# The shape is measured, not derived. A pure information argument gives 4.0 per bit; nothing here
# reaches it, the deficit widens with width, and we do not know why. Extrapolating four or five
# bits from one anchor is where it degrades: three of 400 modules exceed 20 % error, all of them
# K8->K4 or K7->K3. Per-class constants and the depth check are in docs/37.
LAW_SHAPE = {3: 2.6834, 4: 1.3327, 5: 0.0, 6: -1.3150, 7: -2.6089, 8: -3.8783}
CAND_BIG = (3, 4, 5, 6, 7)
CAND_SMALL = (4, 5, 6, 7, 8)
BIG_NUMEL = 52_000_000

LOG_LINE = re.compile(r"Quantized:\s+(?P<key>\S+)\s+bpw:\s+(?P<bpw>[\d.]+)\s+"
                      r"(?:proxy_err|rmse)\s*:\s*(?P<err>[\d.eE+-]+)")


def ladder_from_log(path: Path, shape: dict[int, float]) -> dict:
    """Synthesize the candidate ladder from ONE ordinary teed conversion log.

    A conversion prints one proxy error per module, at the width the allocator gave it. With the
    shape above that is enough: a_m follows from the single point and every other width is
    exp(a_m + s(K)). `out_energy` is not recoverable from a log, so only the `rel` objective is
    available - which is the one the two measured KLD deltas select anyway.
    """
    lad = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = LOG_LINE.search(line)
        if not m:
            continue
        key, k, err = m.group("key"), round(float(m.group("bpw"))), float(m.group("err"))
        if key not in MODS or err <= 0 or k not in shape:
            continue
        cands = CAND_BIG if numel(key) >= BIG_NUMEL else CAND_SMALL
        a = math.log(err) - shape[k]
        lad[key] = {"eps": {kk: math.exp(a + shape[kk]) for kk in cands if kk in shape},
                    "numel": numel(key), "out_energy": None, "recipe_bits": k,
                    "q_fallback": False, "measured_rung": k, "measured_proxy_err": err}
    if not lad:
        raise SystemExit(f"no per-module proxy errors found in {path}")
    return {"meta": {"source": str(path), "synthesized": True,
                     "law": "log eps(m,K) = a_m + s(K), a_m from the log's single rung",
                     "shape": {str(k): v for k, v in sorted(shape.items())},
                     "candidate_widths": {"big": list(CAND_BIG), "small": list(CAND_SMALL),
                                          "big_numel_threshold": BIG_NUMEL}},
            "modules": lad}


def weights(lad: dict, objective: str) -> dict[str, float]:
    """Objective weight per module. Recipe-pinned modules (head, MTP) are laddered at their
    assigned width only and carry no output energy; they are never free variables and never
    appear in a delta, so they get no weight rather than a fabricated one."""
    if objective == "rel":
        return {k: 1.0 for k in lad}
    if objective == "abs":
        return {k: float(v["out_energy"]) for k, v in lad.items()
                if v.get("out_energy") is not None}
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
    """Exact DP over the 655,360-byte grid. Returns (bits for free modules, objective).

    Every module's byte cost is `units * K` grid steps, so the grid is exact and the optimum is
    provable rather than greedy. Reachability pruning keeps the inner loop to the states that any
    prefix of the module list can actually occupy.
    """
    n = len(free)
    U = budget_units
    units = [numel(k) // 8 // UNIT for k in free]
    for k, u in zip(free, units):
        if numel(k) // 8 != u * UNIT:
            raise SystemExit(f"{k}: params/8 is not a multiple of {UNIT}")
    cands = [sorted(kk for kk in lad[k]["eps"] if kk >= min_bits) for k in free]
    if any(not c for c in cands):
        raise SystemExit("a module has no candidate width at or above --min-bits")
    INF = float("inf")
    dp = [INF] * (U + 1)
    dp[0] = 0.0
    choice = [bytearray(U + 1) for _ in range(n)]
    lo = hi = 0                                  # reachable window after the modules done so far
    for j in range(n):
        uj, wj, ej, cj = units[j], w[free[j]], lad[free[j]]["eps"], choice[j]
        nlo = lo + uj * cands[j][0]
        nhi = min(U, hi + uj * cands[j][-1])
        if nlo > U:
            raise SystemExit("no feasible allocation within budget")
        ndp = [INF] * (U + 1)
        for ci, Kc in enumerate(cands[j]):
            c = uj * Kc
            val = wj * ej[Kc]
            for u in range(max(nlo, lo + c), min(nhi, hi + c) + 1):
                v = dp[u - c] + val
                if v < ndp[u]:
                    ndp[u] = v
                    cj[u] = ci
        dp, lo, hi = ndp, nlo, nhi
    u_best, best = -1, INF
    for u in range(lo, hi + 1):
        if dp[u] < best:
            best, u_best = dp[u], u
    if u_best < 0:
        raise SystemExit("no feasible allocation within budget")
    bits, u = {}, u_best
    for j in range(n - 1, -1, -1):
        Kc = cands[j][choice[j][u]]
        bits[free[j]] = Kc
        u -= units[j] * Kc
    assert u == 0, u
    return bits, best


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ladder", help="five-rung ladder JSON from tools/ladder_pass.py")
    src.add_argument("--ladder-from-log", metavar="LOG",
                     help="an ordinary teed conversion log: one rung per module, expanded to the "
                          "candidate widths by the geometric law (see LAW_R)")
    ap.add_argument("--law-shape", default=None, metavar="JSON",
                    help="override the measured width->log-error shape used by --ladder-from-log, "
                         "as a JSON object {\"4\": 1.3327, ...}; default is LAW_SHAPE")
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

    if args.ladder:
        L = load_ladder(Path(args.ladder))
    else:
        shape = ({int(k): float(v) for k, v in json.loads(args.law_shape).items()}
                 if args.law_shape else LAW_SHAPE)
        L = ladder_from_log(Path(args.ladder_from_log), shape)
        if args.objective == "abs":
            raise SystemExit("--objective abs needs per-module output energies, which a conversion "
                             "log does not carry; use --ladder or --objective rel")
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
    have_energy = all(lad[k].get("out_energy") is not None for k in free)
    for obj in (("rel", "abs") if have_energy else ("rel",)):
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
