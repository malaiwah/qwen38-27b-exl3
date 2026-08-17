#!/usr/bin/env python3
"""Add the b12x-gate Amdahl bound to receipts/kernel-amdahl-bound.json.

No new harness was needed: the gate's per-call kernel deltas were already
measured on THIS box with THIS method (CUDA events over 20-call CUDA graphs,
real hydrated weights) and live in receipts/kernel-gap-gemm-bandwidth.json.
Rebuilding a standalone trellis harness would have re-measured the same thing.
"""
import json

R = "/home/mbelleau/research"
bw = json.load(open(f"{R}/receipts/kernel-gap-gemm-bandwidth.json"))
out = json.load(open(f"{R}/receipts/kernel-amdahl-bound.json"))

lm = bw["run1"]["matrices"]["lm_head K6 5120x248320"]["m"]["4"]
kp = bw["run2"]["attn.k_proj 5120x1024"]["m"]["4"]
vp = bw["run2"]["attn.v_proj 5120x1024"]["m"]["4"]

# calls per decode step, all from docs/47 F6/F10.3 (measured profile, not guessed)
CALLS = {"lm_head": 4, "k_proj": 16, "v_proj": 16}
IDLE = 0.23
STEP_MS = {"low": 23.0, "mid": 26.0, "high": 28.0}

comps = []
for tag, rec, n in (("lm_head 5120x248320", lm, CALLS["lm_head"]),
                    ("attn.k_proj 5120x1024", kp, CALLS["k_proj"]),
                    ("attn.v_proj 5120x1024", vp, CALLS["v_proj"])):
    b, e = rec["b12x_graph_us"], rec["exl3_graph_us"]
    comps.append({
        "shape": tag,
        "b12x_graph_us": b,
        "exl3_gemm_graph_us": e,
        "delta_us_per_call": round(b - e, 3),
        "per_call_k": round(b / e, 4),
        "calls_per_step": n,
        "step_b12x_us": round(b * n, 2),
        "step_exl3_us": round(e * n, 2),
        "step_saving_us": round((b - e) * n, 2),
    })

tot_b12x = sum(c["step_b12x_us"] for c in comps)
saved = sum(c["step_saving_us"] for c in comps)
k_eff = tot_b12x / (tot_b12x - saved)


def amdahl(s, k):
    x = s * (1 - 1 / k)
    return x / (1 - x)


mid = STEP_MS["mid"] * 1000.0
s_busy = tot_b12x / (mid * (1 - IDLE))
s_wall = tot_b12x / mid
bound_busy = amdahl(s_busy, k_eff)
bound_wall = amdahl(s_wall, k_eff)
rng = {k: round(100 * (saved / (v * 1000.0 - saved)), 3) for k, v in STEP_MS.items()}

out["gate_bound"] = {
    "status": "MEASURED (reused) + bound derived here",
    "why_no_new_harness": (
        "The task allowed reporting only the transpose if a faithful standalone gate harness were "
        "not tractable in the time available. It turned out not to be needed: the gate's per-call "
        "kernel deltas were ALREADY measured on this same box with this same method -- CUDA events "
        "over 20-call CUDA graphs, real hydrated K6 trellis weights, autotune warmed, GPU verified "
        "idle -- and are recorded in receipts/kernel-gap-gemm-bandwidth.json. Building a second "
        "trellis harness would have re-measured an existing measurement. What was missing was the "
        "Amdahl arithmetic on top of it, which is what this section adds."
    ),
    "source_receipt": "receipts/kernel-gap-gemm-bandwidth.json",
    "source_method": bw["method"],
    "row_count": "m=4, the served MTP verify width (same basis as the transpose numbers above)",
    "components": comps,
    "call_counts_basis": (
        "lm_head 4x/step -- docs/47 F6: the 953.5 MB head runs once for the target and once per MTP "
        "depth. k_proj/v_proj 16x each -- docs/47 F10.3: 16 full-attention layers, q/k/v serialized "
        "as separate trellises. Body pass only; any draft-model attn k/v would ADD saving, so this "
        "is a conservative count."
    ),
    "arithmetic": {
        "step_b12x_us_total": round(tot_b12x, 2),
        "step_saving_us_total": round(saved, 2),
        "saving_detail": (
            f"4 x {comps[0]['delta_us_per_call']} + 16 x {comps[1]['delta_us_per_call']} + "
            f"16 x {comps[2]['delta_us_per_call']} = {saved:.2f} us/step"
        ),
        "effective_k": round(k_eff, 4),
        "effective_k_detail": f"{tot_b12x:.2f} / ({tot_b12x:.2f} - {saved:.2f}) = {k_eff:.4f}",
        "cross_check": (
            "issue #406's own body states 'per-call GPU deltas explain only ~0.6 ms/step'; this "
            f"arithmetic gives {saved/1000:.3f} ms/step, confirming it independently"
        ),
        "share_of_gpu_busy": round(s_busy, 6),
        "share_of_gpu_busy_detail": (
            f"{tot_b12x:.2f} us / ({mid:.0f} us step x (1 - 0.23) GPU-busy) = {s_busy:.6f}"
        ),
        "bound_on_gpu_busy_pct": round(100 * bound_busy, 3),
        "bound_on_gpu_busy_detail": (
            f"1 - 1/{k_eff:.4f} = {1-1/k_eff:.6f}; x = {s_busy:.6f} x {1-1/k_eff:.6f} = "
            f"{s_busy*(1-1/k_eff):.6f}; bound = {100*bound_busy:.3f} %"
        ),
        "share_of_wall_step": round(s_wall, 6),
        "bound_on_wall_step_pct": round(100 * bound_wall, 3),
        "bound_on_wall_step_detail": (
            f"x = {s_wall:.6f} x {1-1/k_eff:.6f} = {s_wall*(1-1/k_eff):.6f}; "
            f"bound = {100*bound_wall:.3f} %"
        ),
        "bound_over_step_range_pct": {
            "identity": "gain = saved / (step - saved)",
            "step_23ms": rng["low"],
            "step_26ms_local_measured": rng["mid"],
            "step_28ms": rng["high"],
        },
    },
    "VERDICT": (
        f"The gate's measured per-call kernel deltas bound the end-to-end single-stream decode gain "
        f"at +{100*bound_wall:.2f} % on a wall-clock step (+{100*bound_busy:.2f} % of decode GPU busy "
        f"time), i.e. +{rng['high']:.2f} .. +{rng['low']:.2f} % across the 23-28 ms step range. "
        f"This is BELOW the +3 % lower end of the '+3-15 %' bracket published in issue #406. The "
        "+15.4 % observed locally is therefore NOT attributable to kernel time: it is the removal of "
        "eager-Python dispatch for the 4 per-step lm_head calls, on a box where proot ptraces every "
        "launch. The bracket must be replaced by this bound."
    ),
    "consistency_with_s28": (
        f"a ceiling of +{100*bound_wall:.2f} % is fully consistent with s28's bare-metal +1.44 % "
        "median at a 2.84 % within-arm CV -- indeed at a true effect of ~2.2 % the paired repeat "
        f"requirement is 7.8489 x (2.84/2.23)^2 = {7.8489*(2.84/(100*bound_wall))**2:.1f}, i.e. ~13 "
        "repeats/cell, against the 5 that were run"
    ),
    "INFERENCE_flag": (
        "[INFERENCE] every percentage in this section is an Amdahl BOUND from measured per-call "
        "deltas and measured call counts; none was observed as a throughput delta. The step-time "
        "denominator (23-28 ms) is itself a local proot-inflated measurement, so the wall-clock "
        "bound is the softer of the two figures."
    ),
}
out["not_measured_and_why"]["b12x_gate_kernel_level"] = (
    "RESOLVED -- see gate_bound: no new harness was built because the gate's per-call kernel deltas "
    "were already measured on this box with this method (receipts/kernel-gap-gemm-bandwidth.json); "
    "this receipt adds the Amdahl arithmetic on top of them."
)
json.dump(out, open(f"{R}/receipts/kernel-amdahl-bound.json", "w"), indent=1)
print("saved_us=%.2f k_eff=%.4f" % (saved, k_eff))
print("bound_busy=%.3f%% bound_wall=%.3f%% range=%s" % (100 * bound_busy, 100 * bound_wall, rng))
print("s_busy=%.4f s_wall=%.4f tot_b12x=%.1fus" % (s_busy, s_wall, tot_b12x))
