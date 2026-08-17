#!/usr/bin/env python3
"""Build receipts/kernel-amdahl-bound.json from the measured harness output."""
import hashlib
import json
import subprocess
import sys

RESEARCH = "/home/mbelleau/research"
SWEEP = "/var/tmp/work/kg_amdahl_ba2.json"
KNAMES = "/var/tmp/work/kg_kernel_names.json"

sweep = json.load(open(SWEEP))
knames = json.load(open(KNAMES))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def amdahl(s, k):
    """End-to-end gain when a component of share s is sped up by k.

    new_total = (1-s) + s/k  = 1 - s*(1-1/k)
    gain      = old/new - 1  = s*(1-1/k) / (1 - s*(1-1/k))
    """
    x = s * (1.0 - 1.0 / k)
    return x / (1.0 - x)


S = 0.052            # in_proj_ba share of decode GPU busy time (docs/47 F6)
IDLE = 0.23          # GPU idle share inside the same profiled window (docs/47 F6)

hot = {m: v for m, v in sweep["shapes"].items()}
cold = {m: v for m, v in sweep["cold_brackets"].items()}

k_hot = hot["m4"]["arms"]["tn_linear_served"]["mean_us"] / hot["m4"]["arms"]["nn_mm_transposed"]["mean_us"]
k_cold = cold["m4"]["cold_tn_us"] / cold["m4"]["cold_nn_us"]

bound_hot = amdahl(S, k_hot)
bound_cold = amdahl(S, k_cold)
# wall-clock basis: the same component is a smaller share of the STEP because
# 23 % of the profiled window is GPU-idle (CPU gaps), and Amdahl only accelerates
# the busy part.
S_wall = S * (1.0 - IDLE)
bound_wall = amdahl(S_wall, k_cold)

table = {}
for k in (1.5, 2.0, 3.0, 4.4):
    x = S * (1 - 1 / k)
    table[f"k={k}"] = {
        "one_minus_1_over_k": round(1 - 1 / k, 6),
        "s_times_that": round(x, 6),
        "gain_pct": round(100 * x / (1 - x), 3),
    }

# §28 repeat-count arithmetic. Paired design (each cell is its own control),
# n = (z_{1-a/2} + z_{1-b})^2 * (sigma/delta)^2 with sigma the within-arm CV.
Z = 1.959964 + 0.841621
CV = 2.84


def repeats(delta_pct):
    import math
    return Z ** 2 * (CV / delta_pct) ** 2, math.ceil(Z ** 2 * (CV / delta_pct) ** 2)


r150 = repeats(1.50)
r144 = repeats(1.44)
rbound = repeats(100 * bound_cold)

saved_hot_us = 48 * (hot["m4"]["arms"]["tn_linear_served"]["mean_us"] - hot["m4"]["arms"]["nn_mm_transposed"]["mean_us"])
saved_cold_us = 48 * (cold["m4"]["cold_tn_us"] - cold["m4"]["cold_nn_us"])

out = {
    "schema": "qwen38-kernel-amdahl-bound/1",
    "generated_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip(),
    "purpose": (
        "Close the re-scope mandated by docs/46 s28 + docs/29: measure the patch at the KERNEL "
        "level, then state the server-level consequence as an Amdahl BOUND instead of trying to "
        "observe it against a 2.84 % within-arm CV."
    ),
    "box": {
        "host": "main-omp-session (1x RTX PRO 6000 Blackwell SE, 96 GB)",
        "gpu_was_idle_before_run": True,
        "container": "proot rootfs via tools/ggrun.sh (no container runtime available)",
        "why_cuda_events": (
            "proot ptraces every syscall and inflates host-side launch/dispatch, so end-to-end "
            "throughput from this box is NOT reportable as a performance result (docs/47 F6 caveat). "
            "CUDA events around CUDA-graph replays time DEVICE execution only, which proot does not "
            "touch, and graph replay removes per-call Python/ioctl dispatch entirely."
        ),
    },
    "patches_under_test": {
        "transpose": {
            "what": "keep a pre-transposed KxN copy for tiny-N tall-K unquantized linears; "
                    "UnquantizedLinearMethod.apply uses torch.mm(x, weight_kn) when present",
            "file": "vllm/model_executor/layers/linear.py",
            "fork_commit": "3b35c04c6 (malaiwah/vllm-voipmonitor branch kernel-gap/tiny-n-mm-transpose)",
            "diff_used": "/tmp/kernel-gap-transpose.patch",
            "sha256": sha("/tmp/kernel-gap-transpose.patch"),
            "provenance_check": (
                "byte-identical to `git diff 3b35c04c6~1..3b35c04c6` regenerated from /var/tmp/vllm-gg "
                "(same sha256), so the timed code path is the committed patch"
            ),
            "env_flag": "VLLM_TINY_N_MM_TRANSPOSE (default 1)",
        },
        "b12x_gate": {
            "what": "one clause in _b12x_trellis_k6_supported rejecting n_packed <5120 or >32768, "
                    "routing lm_head (N=248320) and the tiny k/v projections to exl3_gemm",
            "file": "vllm/model_executor/layers/quantization/exl3.py",
            "diff_used": "receipts/kernel-gap-gate-ab.patch",
            "sha256": sha(f"{RESEARCH}/receipts/kernel-gap-gate-ab.patch"),
            "kernel_level_status": "NOT MEASURED IN THIS RECEIPT -- see not_measured_and_why",
        },
    },
    "MEASURED": {
        "method": sweep["method"],
        "device": sweep["device"],
        "torch": sweep["torch"],
        "cuda": sweep["cuda"],
        "clock_handling": {
            "before_any_gpu_work": sweep["clocks_before_any_gpu_work"],
            "after_ramp_before_timing": sweep["clocks_after_ramp_before_timing"],
            "after": sweep["clocks_after"],
            "note": (
                "the card idles at 180 MHz SM / 405 MHz mem; 300 8192^2 bf16 matmuls are run "
                "BEFORE any timing so both arms are measured on a ramped card, and the arms are "
                "interleaved sample-by-sample so residual drift is common-mode"
            ),
        },
        "shape": sweep["shape"],
        "served_shape_is_m4": (
            "docs/47 F6 profiled 480 in_proj_ba calls over 9 decode steps = 48 per step, m<=4; the "
            "target body pass verifies 1 target + 3 MTP draft tokens, so m=4 is the served width"
        ),
        "hot_L2_per_call_us": {
            m: {
                "tn_served_F_linear": v["arms"]["tn_linear_served"],
                "nn_patch_torch_mm": v["arms"]["nn_mm_transposed"],
                "speedup_k": v["speedup_k"],
            }
            for m, v in hot.items()
        },
        "cold_weight_bracket_per_call_us": cold,
        "kernel_selection_by_name": knames,
        "numerics": {
            m: v["relerr"] for m, v in hot.items()
        },
    },
    "findings": {
        "F_A_the_cliff_is_m_2_to_4_only": {
            "statement": (
                "The served TN path is not uniformly slow at tiny N. It falls off a cliff for "
                "m in {2,3,4} ONLY. At m=1 cuBLAS uses a GEMV (dot_kernel + reduce_1Block) and the "
                "served path is FASTER than the patch; at m>=5 both arms get the modern "
                "nvjet_sm120 TMA split-K kernel and are within 2-16 % of each other."
            ),
            "evidence_us_mean": {
                m: {
                    "tn": hot[m]["arms"]["tn_linear_served"]["mean_us"],
                    "nn": hot[m]["arms"]["nn_mm_transposed"]["mean_us"],
                    "k": hot[m]["speedup_k"]["mean_ratio"],
                }
                for m in hot
            },
        },
        "F_B_mechanism_is_a_declined_split_K": {
            "statement": (
                "Named from the profiler, not inferred: at m in {2,4} the TN arm gets "
                "cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_TN_align8 with NO "
                "split-K reduction (single kernel, 20.1-20.3 us). The NN arm gets the same CUTLASS "
                "family's NN_align8 variant at 4.47-4.49 us PLUS a cublasLt::splitKreduce_kernel at "
                "1.21 us. cuBLASLt declines to split K for the TN layout at N=96/K=5120, so one "
                "16x16 tile walks all 5120 of K serially. This confirms docs/47 F6's attribution "
                "('cutlass_80_wmma...16x16') independently, and identifies the cause as a heuristic "
                "layout preference rather than a missing kernel."
            ),
        },
        "F_C_the_patch_regresses_at_m_equals_1": {
            "statement": (
                "The patch as committed has NO row-count guard, so any configuration that calls this "
                "linear at m=1 (speculative decoding disabled) pays 6.30 us instead of 4.72 us -- a "
                "33 % per-call REGRESSION (k=0.75). The served MTP profile is m=4 so the shipped "
                "configuration is unaffected, but the gate should be m>=2, or better, layout chosen "
                "per row count."
            ),
            "hot_m1_us": {
                "tn": hot["m1"]["arms"]["tn_linear_served"]["mean_us"],
                "nn": hot["m1"]["arms"]["nn_mm_transposed"]["mean_us"],
                "k": hot["m1"]["speedup_k"]["mean_ratio"],
            },
        },
        "F_D_cold_bracket_reproduces_the_in_model_cost": {
            "statement": (
                "With an L2-flushing 192 MB read before every call, the cost-by-difference for the "
                f"served arm at m=4 is {cold['m4']['cold_tn_us']} us -- which brackets docs/47 F6's "
                "in-model figure of 28 us/call that the 5.2 % share was computed from. The hot-L2 "
                f"figure {hot['m4']['arms']['tn_linear_served']['mean_us']} us reproduces the earlier "
                "standalone probe (receipts/kernel-gap-ba-probe.json: 20.08 us) to 3 significant "
                "figures. Two independent probes, two matched numbers."
            ),
            "method_error_selfcheck": (
                f"at m=1 the difference method returns {cold['m1']['cold_tn_us']} us against a hot "
                f"measurement of {hot['m1']['arms']['tn_linear_served']['mean_us']} us; the "
                "difference method therefore carries a systematic error of order 0.4 us, which is "
                "why the cold numbers are quoted as a BRACKET and not as the primary result"
            ),
        },
        "F_E_the_patch_is_slightly_less_accurate": {
            "statement": (
                "Not free: because the NN layout is the one cuBLASLt splits K for, the patch path "
                "reduces bf16 partials through splitKreduce. Relative error against an fp32 "
                f"reference at m=4 is {hot['m4']['relerr']['nn_vs_fp32']} for the patch vs "
                f"{hot['m4']['relerr']['tn_vs_fp32']} for the served path -- both ~0.2 %, but the "
                "patch is the worse of the two and the results are NOT bit-identical "
                f"(max abs delta {hot['m4']['relerr']['tn_vs_nn_max_abs']} on outputs of magnitude "
                "~72). A KLD spot-check remains required before shipping."
            ),
        },
    },
    "amdahl": {
        "identity": "gain = s*(1 - 1/k) / (1 - s*(1 - 1/k))",
        "derivation": (
            "let baseline total = 1 with component share s. After a k-fold speedup of that "
            "component the total is (1-s) + s/k = 1 - s*(1-1/k). Throughput gain is "
            "old/new - 1 = 1/(1 - s*(1-1/k)) - 1 = s*(1-1/k)/(1 - s*(1-1/k))."
        ),
        "component_share_s": S,
        "share_basis": (
            "docs/47 F6, torch-profiled decode of the hydrated model: cuBLAS in_proj_ba = 5.2 % of "
            "decode GPU BUSY time (480 calls / 9 steps), 1.34 ms of a 23-28 ms step"
        ),
        "bound_at_measured_k": {
            "hot_L2": {
                "k": round(k_hot, 4),
                "s_times_1_minus_1_over_k": round(S * (1 - 1 / k_hot), 6),
                "bound_pct": round(100 * bound_hot, 3),
                "arithmetic": (
                    f"1 - 1/{k_hot:.4f} = {1-1/k_hot:.6f}; x = 0.052 * {1-1/k_hot:.6f} = "
                    f"{S*(1-1/k_hot):.6f}; bound = {S*(1-1/k_hot):.6f} / "
                    f"{1-S*(1-1/k_hot):.6f} = {bound_hot:.6f} = {100*bound_hot:.3f} %"
                ),
            },
            "cold_bracket": {
                "k": round(k_cold, 4),
                "s_times_1_minus_1_over_k": round(S * (1 - 1 / k_cold), 6),
                "bound_pct": round(100 * bound_cold, 3),
                "arithmetic": (
                    f"1 - 1/{k_cold:.4f} = {1-1/k_cold:.6f}; x = 0.052 * {1-1/k_cold:.6f} = "
                    f"{S*(1-1/k_cold):.6f}; bound = {S*(1-1/k_cold):.6f} / "
                    f"{1-S*(1-1/k_cold):.6f} = {bound_cold:.6f} = {100*bound_cold:.3f} %"
                ),
            },
            "HEADLINE": (
                f"the measured kernel speedup at the served m=4 shape is {k_hot:.2f}x (hot L2) to "
                f"{k_cold:.2f}x (cold weight), which bounds the end-to-end single-stream decode gain "
                f"at +{100*bound_hot:.2f} % .. +{100*bound_cold:.2f} % of decode GPU time. This is a "
                "CEILING, not an observed throughput gain."
            ),
        },
        "bound_on_wall_clock_basis": {
            "why": (
                "Amdahl only accelerates GPU-busy time. The same profiled window is 23 % GPU-IDLE "
                "(eager draft passes + sampler D2H syncs), so on a wall-clock step the component "
                "share is 0.052 * (1 - 0.23) and the bound is smaller still."
            ),
            "s_wall": round(S_wall, 6),
            "bound_pct": round(100 * bound_wall, 3),
            "arithmetic": (
                f"s_wall = 0.052 * 0.77 = {S_wall:.6f}; x = {S_wall:.6f} * {1-1/k_cold:.6f} = "
                f"{S_wall*(1-1/k_cold):.6f}; bound = {100*bound_wall:.3f} %"
            ),
            "caveat": "the 23 % idle share is itself proot-inflated locally, so this is a lower bracket",
        },
        "bound_table_over_k": {
            "note": "s = 5.2 % held fixed; reproduces docs/46 s28's table exactly",
            "rows": table,
        },
        "per_step_saving_us": {
            "hot_L2": round(saved_hot_us, 1),
            "cold_bracket": round(saved_cold_us, 1),
            "arithmetic": (
                f"48 calls/step * ({hot['m4']['arms']['tn_linear_served']['mean_us']} - "
                f"{hot['m4']['arms']['nn_mm_transposed']['mean_us']}) us = {saved_hot_us:.1f} us hot; "
                f"48 * ({cold['m4']['cold_tn_us']} - {cold['m4']['cold_nn_us']}) = {saved_cold_us:.1f} us cold"
            ),
            "cross_check": (
                f"the cold saving {saved_cold_us/1000:.2f} ms/step independently matches docs/47 "
                "F10.2's predicted '~1.2 ms/step recovered', derived there from the 28 us/call figure"
            ),
        },
    },
    "reconciliation_with_s28": {
        "what_s28_measured": {
            "receipt": "receipts/kernel-gap-bare-metal-ab.json",
            "setup": "real docker, no proot, single GPU on the 8x host, 5 repeats/cell, rungs C1-C8",
            "valid_cells": 11,
            "median_delta_per_request_pct": 1.44,
            "mean_delta_per_request_pct": 2.24,
            "range_pct": "-0.55 .. +8.94",
            "within_arm_cv_median_pct": 2.84,
            "within_arm_cv_worst_pct": "13.9 - 18.0 (short-prompt cells)",
            "sign_test_p": 0.065,
            "verdict": "no magnitude supportable",
        },
        "same_result_two_scales": (
            f"The kernel measurement says the component got {k_hot:.2f}-{k_cold:.2f}x faster. Amdahl "
            f"turns that into a CEILING of +{100*bound_hot:.2f}..{100*bound_cold:.2f} % on decode GPU "
            f"time, and +{100*bound_wall:.2f} % once the 23 % GPU-idle share of the step is admitted. "
            f"s28 observed +1.44 % (median) / +2.24 % (mean) per-request against a 2.84 % within-arm "
            "CV. A ceiling of ~+4 % and an observation of +1.4..2.2 % +/- 2.8 % are not in conflict: "
            "the observation's own noise band spans the entire interval between zero and the "
            "ceiling. s28 did not fail to find the effect because the effect is absent; it failed "
            "because a >=2.84 % ruler cannot resolve a <=4.24 % quantity, let alone the ~1-3 % the "
            "wall-clock basis predicts. One result, two scales."
        ),
        "repeats_that_would_have_been_needed": {
            "formula": (
                "paired design (each cell is its own control): "
                "n = (z_{1-alpha/2} + z_{1-beta})^2 * (sigma/delta)^2, "
                "alpha=0.05 two-sided, power 0.80, sigma = within-arm CV = 2.84 %"
            ),
            "z_sum": round(Z, 6),
            "z_sum_squared": round(Z ** 2, 6),
            "at_delta_1.50_pct": {
                "arithmetic": f"{Z**2:.4f} * (2.84/1.50)^2 = {Z**2:.4f} * {(CV/1.5)**2:.4f} = {r150[0]:.2f}",
                "n": r150[1],
                "matches_s28_estimate_of_29": r150[1] == 29,
            },
            "at_delta_1.44_pct_the_observed_median": {
                "arithmetic": f"{Z**2:.4f} * (2.84/1.44)^2 = {r144[0]:.2f}",
                "n": r144[1],
            },
            "at_the_amdahl_ceiling": {
                "delta_pct": round(100 * bound_cold, 3),
                "arithmetic": f"{Z**2:.4f} * (2.84/{100*bound_cold:.2f})^2 = {rbound[0]:.2f}",
                "n": rbound[1],
                "reading": (
                    "s28 ran 5 repeats per cell. That is just enough to resolve an effect sitting "
                    f"exactly at the ceiling ({rbound[1]} needed) and nowhere near enough for the "
                    f"magnitude actually observed ({r144[1]} needed). The design was marginal by "
                    "construction, which is precisely why the method now measures the kernel."
                ),
            },
        },
        "standing_method": (
            "docs/29 / docs/46 s28: for any patch whose Amdahl ceiling is below the harness CV, "
            "measure the kernel and publish the bound. Do not spend ladder time trying to observe it."
        ),
    },
    "MEASURED_vs_INFERENCE": {
        "MEASURED_on_this_box": [
            "per-call device time of both code paths at m in {1,2,3,4,5,6,8,16}, 200 CUDA-event "
            "samples per arm per shape, 48 calls per sample, rel. SEM 0.012-0.068 %",
            "the CUTLASS/cuBLAS kernel actually selected for each arm at m in {1,2,4,8}, by name, "
            "from the torch CUDA profiler",
            "the cold-weight (L2-flushed) cost by difference at the same shapes",
            "relative error of both arms against an fp32 reference, and their mutual max abs delta",
            "SM/mem clocks and temperature before ramp, after ramp, and after the run",
            "sha256 and git provenance of the transpose diff under test",
        ],
        "MEASURED_ELSEWHERE_and_reused": [
            "s = 5.2 % component share of decode GPU busy time -- docs/47 F6, "
            "receipts/kernel-gap-profiled-decode.json (torch profile of the real model)",
            "23 % GPU-idle share of the profiled decode window -- same source",
            "48 in_proj_ba calls per decode step, m<=4 -- same source",
            "s28's +1.44 % median / 2.84 % CV -- receipts/kernel-gap-bare-metal-ab.json (8x host)",
        ],
        "INFERENCE": [
            "[INFERENCE] every end-to-end percentage in this receipt is an Amdahl BOUND computed "
            "from a measured kernel ratio and a separately measured component share. None of them "
            "was observed as a throughput delta, and none may be quoted as one.",
            "[INFERENCE] the bound assumes the 5.2 % share is unchanged by the patch other than "
            "through the component itself (no secondary effects on cache residency, kernel "
            "occupancy overlap, or launch scheduling).",
            "[INFERENCE] the cold-weight bracket is a difference of two graph measurements, not a "
            "direct cold-call measurement; its systematic error is bounded at ~0.4 us by the m=1 "
            "self-check above.",
            "[INFERENCE] the m=1 regression is measured per-call but its end-to-end cost in a "
            "non-speculative configuration has not been computed -- the 5.2 % share was measured "
            "under the MTP profile and does not transfer.",
            "[INFERENCE] the required-repeat counts assume the within-arm CV is stationary across "
            "repeats and that the paired differences are approximately normal.",
        ],
    },
    "not_measured_and_why": {
        "b12x_gate_kernel_level": (
            "IN PROGRESS at the time of this commit -- the transpose result is landed first per the "
            "standing rule to commit measured numbers as soon as they exist. A faithful standalone "
            "harness for the gate needs real packed K6 trellis + suh/svh/mcg tensors for lm_head "
            "(N=248320) pulled from the hydrated checkpoint and driven through both "
            "_b12x_trellis_linear and exl3_gemm; see field gate_attempt if present."
        ),
    },
    "artifacts": {
        "harness": "tools/kernel-amdahl-ba.py (this repo) == /var/tmp/work/kg_amdahl_ba.py",
        "kernel_name_probe": "tools/kernel-name-probe.py == /var/tmp/work/kg_kernel_names.py",
        "raw_sweep_json": "receipts/kernel-amdahl-ba-raw.json",
        "raw_kernel_names_json": "receipts/kernel-amdahl-kernel-names.json",
    },
    "issue_comments": {},
}

json.dump(out, open(f"{RESEARCH}/receipts/kernel-amdahl-bound.json", "w"), indent=1)
print("k_hot=%.4f k_cold=%.4f" % (k_hot, k_cold))
print("bound_hot=%.3f%% bound_cold=%.3f%% bound_wall=%.3f%%" % (100 * bound_hot, 100 * bound_cold, 100 * bound_wall))
print("table:", json.dumps(table))
print("repeats: 1.50->%s  1.44->%s  ceiling(%.2f%%)->%s" % (r150[1], r144[1], 100 * bound_cold, rbound[1]))
print("saved hot=%.1fus cold=%.1fus" % (saved_hot_us, saved_cold_us))
