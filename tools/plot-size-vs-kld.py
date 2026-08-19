#!/usr/bin/env python3
"""Weight/disk size vs measured KLD, with VRAM capacity bands.

Design rule this chart obeys
----------------------------
**One panel, one protocol.** KLD is only comparable within a single measurement
protocol (same reference model, same context length, same scored positions, same
LM head). Panel A therefore contains ONLY points measured on our own 512-context
shard-0 suite. Third-party numbers get their own panel with their own axis, and
the caption says why they cannot share Panel A. Mixing them would be the exact
error this project has repeatedly corrected.

Panel C plots the allocation solver's frontier as a RELATIVE shape only: the
objective is not in KLD units (its absolute extrapolation covers 59.8-68.6% of
measured KLD, and it is wrong-signed for cross-class reallocation --
receipts/eda-resolve-2026-08-19.md), so it carries no y-axis tick labels.

VRAM bands are WEIGHT budgets, not total footprints. Serving also needs KV cache,
activations and CUDA/allocator overhead: on our 32 GB card, 31.40 GiB is usable and
the fidelity profile spends 9.29 GiB of it on KV at 238,400 context. Each band is
drawn at capacity minus a stated serving reserve so the "fits" reading is honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# Panel A: our own 512-ctx shard-0 suite. Every KLD here shares one protocol.
#   size_gib = serialized weight payload on disk.
#   Runtime-converted profiles (balanced, all-FP4, all-FP6, self_attn-FP4) are
#   the SAME checkpoint served differently, so they stack at hydrated's x.
# ---------------------------------------------------------------------------
OURS = [
    # label,                      size_gib,                    kld,      kind
    ("K5K6-hydrated (offline)",   21_586_964_548 / GIB,        0.002700, "ckpt"),
    ("K5K6-online",               20_672_081_988 / GIB,        0.003141, "ckpt"),
    ("EDA-research (rel-solved)", 21_586_964_548 / GIB,        0.003066, "ckpt"),
    ("fidelity (as served)",      21_586_964_548 / GIB,        0.003405, "serve"),
    ("balanced (gate_up MXFP6)",  21_586_964_548 / GIB,        0.005672, "serve"),
    ("all-FP6 (runtime)",         21_586_964_548 / GIB,        0.010699, "serve"),
    ("self_attn-FP4 (runtime)",   21_586_964_548 / GIB,        0.011534, "serve"),
    ("throughput all-FP4",        21_586_964_548 / GIB,        0.063759, "serve"),
    ("RTN FP8attn+NVFP4mlp",      21_957_305_088 / GIB,        0.022121, "ckpt"),
    ("GPTQ FP8attn+NVFP4mlp",     21_957_305_088 / GIB,        0.028548, "ckpt"),
]
CRITERION_KLD = 0.012   # our north-star criterion 3

# ---------------------------------------------------------------------------
# Panel B: third-party harness (Discord leaderboard). DIFFERENT PROTOCOL.
# Sizes as published there; KLD on their scale only.
# ---------------------------------------------------------------------------
THIRD_PARTY = [
    ("gptq-nvfp4-mixed-32",  23.96, 0.002642),
    ("gptq-nvfp4-mixed-48",  22.22, 0.002662),
    ("gptq-nvfp4-mixed-64",  20.47, 0.002666),
    ("gptq-mxfp8-mixed",     43.38, 0.002670),
    ("EXL3-EDA-research",    20.10, 0.007461),
    ("EXL3-K5K6 (online)",   19.25, 0.008170),
    ("nvfp4-gptq-v18",       32.98, 0.009034),
    ("modelopt-nvfp4",       18.32, 0.350262),
]

# ---------------------------------------------------------------------------
# VRAM bands. Capacity is nameplate; reserve is what serving needs beyond
# weights (KV + activations + allocator). Blackwell-first as requested.
# ---------------------------------------------------------------------------
BANDS = [
    (12, "12 GB - RTX 5070", 4.0),
    (16, "16 GB - RTX 5080 / 5070 Ti", 5.0),
    (24, "24 GB - RTX PRO 4000 Blackwell", 6.0),
    (32, "32 GB - RTX 5090", 9.3),
    (48, "48 GB - RTX PRO 5000 Blackwell", 12.0),
    (96, "96 GB - RTX PRO 6000 Blackwell", 20.0),
]


def solver_frontier(ladder: Path, plan: Path, solved: list[str], tool: Path,
                    n: int = 13) -> list[tuple[float, float]]:
    """Sweep the body budget and record (total serialized GiB, objective)."""
    import subprocess
    import tempfile
    pts = []
    plan_j = json.loads(plan.read_text())
    base_total = plan_j["budget_bytes"]
    for i in range(n):
        delta = int((-0.30 + 0.60 * i / (n - 1)) * 4_000_000_000)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            out = tf.name
        cmd = ["python3", str(tool), "--ladder", str(ladder), "--plan", str(plan),
               "--weighting", "class-kld", "--max-width", "6",
               "--depth-form", "late", "--depth-amp", "0.05",
               "--budget-delta", str(delta), "--out", out]
        for s in solved:
            cmd += ["--solved", s]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            continue
        a = json.loads(Path(out).read_text())
        pts.append(((base_total + delta) / GIB, a["objective"]["resolved"]))
        Path(out).unlink(missing_ok=True)
    return sorted(pts)


# Measured KV cost on our card: 9.29 GiB of KV serves 238,400 context on the
# fidelity profile => 0.00003897 GiB/token. Usable VRAM is 31.40 GiB of the 32 GB
# nameplate (driver/context overhead), measured.
USABLE_GIB = 31.40
KV_GIB_PER_TOKEN = 9.29 / 238_400
CTX_LINES = [32_768, 131_072, 199_104, 238_400, 262_144]

# Contexts we have actually served and gated. Runtime-converted profiles differ
# in RESIDENT size from their disk payload, so these override any estimate.
MEASURED_CTX = {
    "fidelity (as served)": 238_400,
    "balanced (gate_up MXFP6)": 199_104,
    "throughput all-FP4": 249_600,
    "self_attn-FP4 (runtime)": 238_400,
    "all-FP6 (runtime)": 99_000,
}


def chart_5090_zoom(path: str) -> None:
    """One card, one question: at what context does each artifact still fit?"""
    fig, ax = plt.subplots(figsize=(14.5, 9.0))

    # Weight budget shrinks as target context grows: budget = usable - KV(ctx).
    for ctx in CTX_LINES:
        budget = USABLE_GIB - ctx * KV_GIB_PER_TOKEN
        ax.axvline(budget, color="tab:blue", ls="--", lw=1.5, alpha=0.75)
        ax.text(budget, 0.0019, f" {ctx:,} ctx\n weights <= {budget:.2f} GiB",
                rotation=90, va="bottom", ha="right", fontsize=9.2,
                color="tab:blue", weight="bold")
    max_budget = USABLE_GIB - CTX_LINES[0] * KV_GIB_PER_TOKEN
    ax.axvspan(0, max_budget, color="tab:blue", alpha=0.05)

    ax.axhline(CRITERION_KLD, color="crimson", ls="--", lw=1.8, alpha=0.9)
    ax.text(18.15, CRITERION_KLD * 1.07, "criterion 3: KLD <= 0.012",
            color="crimson", fontsize=11, weight="bold")

    styles = {"ckpt": dict(marker="o", s=170, edgecolor="black", linewidth=1.2),
              "serve": dict(marker="^", s=140, edgecolor="black", linewidth=0.9)}
    order = sorted(OURS, key=lambda r: r[2])
    lo, hi = 0.00215, 0.092
    for i, (label, size, kld, kind) in enumerate(order):
        c = "tab:green" if kld <= CRITERION_KLD else "tab:red"
        ax.scatter([size], [kld], color=c, zorder=6, **styles[kind])
        frac = i / max(len(order) - 1, 1)
        y_lab = lo * (hi / lo) ** frac
        # Measured served context where we have one; otherwise a weights-only
        # estimate. Runtime-converted profiles change the RESIDENT footprint
        # (FP4/FP6 conversion shrinks weights in VRAM), so a disk-derived
        # estimate would be wrong for them - hence measured values take priority.
        if label in MEASURED_CTX:
            ctx_txt = f"serves {MEASURED_CTX[label]:,} ctx (measured)"
        else:
            room = USABLE_GIB - size
            ctx_txt = f"~{max(int(room / KV_GIB_PER_TOKEN), 0):,} ctx (est, weights-only)"
        ax.annotate(f"{label}   KLD {kld:.6f}   →  {ctx_txt}",
                    xy=(size, kld), xytext=(22.9, y_lab), fontsize=10.0,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.0, alpha=0.8,
                                    shrinkA=0, shrinkB=5,
                                    connectionstyle="arc3,rad=0.05"))

    ax.set_yscale("log")
    ax.set_xlim(18, 29.5)
    ax.set_ylim(0.0020, 0.10)
    ax.set_xlabel("serialized weight payload on disk (GiB)", fontsize=12)
    ax.set_ylabel("mean KLD vs BF16  (log scale)", fontsize=12)
    ax.set_title("RTX 5090 32 GB, zoomed: the weight budget IS a function of target "
                 "context\n"
                 f"usable {USABLE_GIB} GiB of 32 GB nameplate (measured) · KV costs "
                 f"{KV_GIB_PER_TOKEN*1e6:.1f} MiB per 1k tokens (measured: 9.29 GiB "
                 "at 238,400 ctx)\n"
                 "dashed lines = the weight ceiling for each target context; "
                 "everything left of a line fits at that context",
                 fontsize=12.5, weight="bold", loc="left")
    ax.grid(alpha=0.25, which="both")
    fig.text(0.5, 0.007,
             "All KLD values on ONE protocol (512-context shard-0 suite, 1,048,064 "
             "scored positions, shared BF16 head). 'fits ~N ctx' is weights-only "
             "arithmetic from the measured KV rate; it ignores the ~0.4 GiB boot "
             "margin a profile needs, so treat it as an upper bound.",
             ha="center", fontsize=9.0, color="#333333")
    fig.savefig(path, dpi=155, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder")
    ap.add_argument("--plan")
    ap.add_argument("--solved", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom-out", default=None,
                    help="also write the RTX 5090 32 GB zoom panel here")
    args = ap.parse_args()
    if args.zoom_out:
        chart_5090_zoom(args.zoom_out)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.9, 1.5], hspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # ---------------- Panel A ----------------
    for cap, label, reserve in BANDS:
        budget = cap - reserve
        if budget <= 0:
            continue
        axA.axvspan(0, budget, color="tab:blue", alpha=0.045)
        axA.axvline(budget, color="tab:blue", ls=":", lw=1.1, alpha=0.65)
        axA.text(budget, 0.0982, f"{label}\nweights <= {budget:.1f} GiB",
                 rotation=90, va="top", ha="right", fontsize=7.6,
                 color="tab:blue", alpha=0.95)

    axA.axhline(CRITERION_KLD, color="crimson", ls="--", lw=1.5, alpha=0.85)
    axA.text(6.6, CRITERION_KLD * 1.10,
             "criterion 3: KLD <= 0.012", color="crimson", fontsize=9.5, weight="bold")

    styles = {"ckpt": dict(marker="o", s=115, edgecolor="black", linewidth=1.0),
              "serve": dict(marker="^", s=95, edgecolor="black", linewidth=0.7)}
    for label, size, kld, kind in OURS:
        c = "tab:green" if kld <= CRITERION_KLD else "tab:red"
        axA.scatter([size], [kld], color=c, zorder=5, **styles[kind])
    # Most runtime profiles share hydrated's x, so labels would pile up. Place
    # them in a clean column at the right with leader lines, log-spaced by rank
    # so the ordering stays readable.
    order = sorted(OURS, key=lambda r: r[2])
    x_lab = 24.6
    lo, hi = 0.00205, 0.098
    n = len(order)
    for i, (label, size, kld, kind) in enumerate(order):
        frac = i / max(n - 1, 1)
        y_lab = lo * (hi / lo) ** frac
        c = "tab:green" if kld <= CRITERION_KLD else "tab:red"
        axA.annotate(f"{label}   {kld:.6f}", xy=(size, kld),
                     xytext=(x_lab, y_lab), fontsize=9.0, va="center",
                     color="black",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.9, alpha=0.75,
                                     shrinkA=0, shrinkB=4,
                                     connectionstyle="arc3,rad=0.06"))
    axA.set_yscale("log")
    axA.set_xlim(6, 78)
    axA.set_ylim(0.0018, 0.12)
    axA.set_xlabel("serialized weight payload on disk (GiB)")
    axA.set_ylabel("mean KLD vs BF16  (log scale)")
    axA.set_title("A. OUR artifacts, all on ONE protocol: 512-context shard-0 suite, "
                  "1,048,064 scored positions, shared BF16 LM head\n"
                  "circles = distinct checkpoints; triangles = the same hydrated "
                  "checkpoint served with runtime conversion (same download, "
                  "different fidelity)", fontsize=10.5, loc="left")
    axA.grid(alpha=0.25, which="both")

    # ---------------- Panel B ----------------
    for label, size, kld in THIRD_PARTY:
        axB.scatter([size], [kld], color="tab:purple", marker="s", s=70,
                    edgecolor="black", linewidth=0.6, zorder=5)
        pass
    tp = sorted(THIRD_PARTY, key=lambda r: r[2])
    lo_b, hi_b = 0.0021, 0.30
    for i, (label, size, kld) in enumerate(tp):
        frac = i / max(len(tp) - 1, 1)
        y_lab = lo_b * (hi_b / lo_b) ** frac
        axB.annotate(f"{label}   {kld:.6f}", xy=(size, kld), xytext=(45.5, y_lab),
                     fontsize=8.6, va="center",
                     arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.8,
                                     alpha=0.65, shrinkA=0, shrinkB=4,
                                     connectionstyle="arc3,rad=0.05"))
    axB.set_yscale("log")
    axB.set_xlim(15, 62)
    axB.set_xlabel("published size (GB, their figures)")
    axB.set_ylabel("KLD, THEIR harness")
    axB.set_title("B. Third-party leaderboard - SEPARATE PANEL BY NECESSITY. Their "
                  "protocol (context length, scored positions, reference) is "
                  "undisclosed;\ntheir K5K6 row scores 0.008170 where our suite "
                  "scores the same weights 0.003141, so the axes differ by roughly "
                  "2.6x. Only WITHIN-panel ordering is evidence.",
                  fontsize=10.5, loc="left")
    axB.grid(alpha=0.25, which="both")

    # ---------------- Panel C ----------------
    if args.ladder and args.plan:
        pts = solver_frontier(Path(args.ladder), Path(args.plan), args.solved,
                              Path(__file__).with_name("eda-resolve.py"))
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            axC.plot(xs, ys, "-o", color="tab:orange", ms=4.5, lw=1.6)
            axC.set_xlabel("serialized weight payload budget (GiB)")
            axC.set_ylabel("solver objective\n(relative - NOT KLD)")
            axC.set_yticklabels([])
            axC.axvline(21_586_964_548 / GIB, color="k", ls=":", lw=1.2)
            axC.text(21_586_964_548 / GIB, max(ys),
                     " hydrated budget", fontsize=8.5, va="top")
            axC.grid(alpha=0.25)
    axC.set_title("C. Allocation-solver frontier (class-kld, K<=6, depth-form=late "
                  "amp 0.05): SHAPE ONLY.\nThe objective is not in KLD units - its "
                  "absolute extrapolation covers 59.8-68.6% of measured KLD and it "
                  "is wrong-signed for cross-class reallocation, so no y ticks.",
                  fontsize=10.5, loc="left")

    fig.suptitle("Qwen3.8-27B: weight size vs measured KLD, with Blackwell VRAM "
                 "weight budgets", fontsize=14, weight="bold", y=0.985)
    fig.text(0.01, 0.005,
             "VRAM bands are WEIGHT budgets = nameplate capacity minus a serving "
             "reserve (KV cache + activations + allocator). Our measured example: "
             "31.40 GiB usable of 32 GB, with 9.29 GiB of KV at 238,400 context. "
             "No competitor checkpoint has been measured on our suite, which is why "
             "Panel A contains only our artifacts.",
             fontsize=8.0, color="#333333")
    fig.savefig(args.out, dpi=155, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
