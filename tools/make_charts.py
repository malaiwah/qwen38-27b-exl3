#!/usr/bin/env python3
"""Charts for the Qwen3.8-27B-K4 model card: fidelity versus weight footprint.

Every point is measured on this box unless the series is explicitly marked
external. Emits light and dark SVG + PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("/var/tmp/work/charts")

# ---- measured here: v2 hidden-state-replay protocol, 74 contexts / 151,478 positions
#      weights = safetensors weight bytes actually resident (decimal GB)
MEASURED = [
    # label, weights GB, mean KLD, top-1 %, colour key
    # v3 protocol: held-out corpus, 136 analysis contexts, 278,392 scored positions
    ("Qwen3.8-27B BF16\n(reference)", 55.56, 0.0, 100.00, "ref"),
    ("malaiwah/Qwen3.8-27B-EXL3-K5K6\n(iteration 2)", 21.82, 0.008157, 96.97, "ours2"),
    ("Qwen/Qwen3.8-27B-FP8", 30.61, 0.013126, 96.22, "fp8"),
    ("malaiwah/Qwen3.8-27B-K4\n(iteration 1)", 19.21, 0.030736, 94.50, "ours"),
    ("unsloth/Qwen3.8-27B-NVFP4", 22.91, 0.094978, 90.53, "nvfp4"),
]
# CI bounds for the three quantized points (source-cluster bootstrap, v3)
CI = {"ours2": (0.006066, 0.010667),
      "fp8": (0.009807, 0.017087),
      "ours": (0.022384, 0.040731),
      "nvfp4": (0.068584, 0.126882)}

# ---- external context: Quesma, Qwen3.6-27B, different model generation and protocol
#      https://quesma.com/blog/qwen-quantization-quality/
EXTERNAL = [
    ("UD-IQ2_XXS", 9.6, 0.28), ("UD-IQ2_M", 11.0, 0.13), ("UD-Q2_K_XL", 12.0, 0.12),
    ("UD-IQ3_XXS", 12.2, 0.086), ("Q3_K_S", 12.6, 0.083), ("Q3_K_M", 13.8, 0.050),
    ("UD-Q3_K_XL", 14.8, 0.038), ("IQ4_XS", 15.7, 0.018), ("Q4_0", 16.1, 0.035),
    ("Q4_K_S", 16.1, 0.019), ("IQ4_NL", 16.3, 0.018), ("Q4_K_M", 17.1, 0.017),
    ("UD-Q4_K_XL", 17.9, 0.013), ("Q5_K_S", 19.3, 0.0058), ("Q5_K_M", 19.8, 0.0052),
    ("UD-Q5_K_XL", 20.4, 0.0046), ("Q6_K", 22.9, 0.0021), ("UD-Q6_K_XL", 26.0, 0.0014),
    ("Q8_0", 29.0, 0.00049), ("UD-Q8_K_XL", 35.8, 0.00038),
    ("FP8 (vLLM)", 30.9, 0.017), ("NVFP4 (NVIDIA)", 21.9, 0.039),
    ("NVFP4 (Unsloth)", 23.3, 0.044),
]

CEILING = 21.92  # nvidia/Qwen3.6-27B-NVFP4 weight footprint: the memory ceiling we hold to

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d0d0d0", muted="#8a8a8a",
                  ours="#57a773", ours2="#0b5d1e", fp8="#0969da", nvfp4="#a40e26", ref="#57606a",
                  ext="#b9bcc0"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e",
                 ours="#2ea043", ours2="#56d364", fp8="#58a6ff", nvfp4="#f85149", ref="#8b949e",
                 ext="#3d444d"),
}


def draw(theme_name: str) -> None:
    c = THEMES[theme_name]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=c["bg"],
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    for a in (ax, bx):
        a.set_facecolor(c["bg"])
        for s in a.spines.values():
            s.set_color(c["grid"])
        a.tick_params(colors=c["muted"], labelsize=9)
        a.grid(True, which="both", color=c["grid"], alpha=0.45, lw=0.6)
        a.set_axisbelow(True)

    # ---------------- panel A: KLD vs weight footprint
    ax.set_yscale("log")
    ex_x = [w for _, w, _ in EXTERNAL]
    ex_y = [k for _, _, k in EXTERNAL]
    ax.scatter(ex_x, ex_y, s=26, color=c["ext"], zorder=2, label="external context")
    ext_labels = {"FP8 (vLLM)": (0, -13, "center"),
                  "UD-Q4_K_XL": (-8, -4, "right"),
                  "Q8_0": (0, -13, "center")}
    for name, w, k in EXTERNAL:
        if name in ext_labels:
            dx, dy, ha = ext_labels[name]
            ax.annotate(name, (w, k), color=c["muted"], fontsize=7,
                        xytext=(dx, dy), textcoords="offset points", ha=ha)

    ax.axvline(CEILING, color=c["nvfp4"], lw=1.1, ls=(0, (5, 3)), alpha=0.8, zorder=1)
    ax.annotate("NVFP4-equivalent\nmemory ceiling 21.9 GB", (CEILING, 0.42),
                color=c["nvfp4"], fontsize=8, xytext=(7, -2), textcoords="offset points",
                va="top", ha="left")

    place = {"fp8": (12, 10, "left"), "ours": (-14, -8, "right"), "nvfp4": (-13, 9, "right"),
             "ours2": (4, -30, "center")}
    for label, w, kld, _top1, key in MEASURED:
        if key == "ref":
            ax.scatter([w], [3.0e-4], s=150, marker="*", color=c["ref"], zorder=5)
            ax.annotate("BF16 reference\n(KLD = 0 by definition)", (w, 3.0e-4),
                        color=c["fg"], fontsize=8.5, xytext=(-10, 10),
                        textcoords="offset points", ha="right", weight="bold")
            continue
        lo, hi = CI[key]
        ax.plot([w, w], [lo, hi], color=c[key], lw=2.2, alpha=0.55, zorder=4)
        ax.scatter([w], [kld], s=185 if key.startswith("ours") else 110,
                   marker="D" if key.startswith("ours") else "o",
                   color=c[key], edgecolor=c["bg"], linewidth=1.4, zorder=6)
        dx, dy, ha = place[key]
        ax.annotate(f"{label}\n{kld:.4f}", (w, kld), color=c[key], fontsize=9,
                    weight="bold" if key.startswith("ours") else "normal",
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.set_xlabel("resident weight footprint (decimal GB)", color=c["fg"], fontsize=10)
    ax.set_ylabel("mean KL divergence from BF16  (log scale, lower is better)",
                  color=c["fg"], fontsize=10)
    ax.set_title("Distribution fidelity vs memory — held-out corpus, 136 contexts, 278,392 positions",
                 color=c["fg"], fontsize=11.5, weight="bold", loc="left")
    ax.set_xlim(7, 59)
    ax.set_ylim(2.2e-4, 0.5)
    handles = [
        Line2D([], [], marker="D", ls="", color=c["ours2"], label="EXL3 K5/K6 (iteration 2)"),
        Line2D([], [], marker="D", ls="", color=c["ours"], label="EXL3 K4 (iteration 1)"),
        Line2D([], [], marker="o", ls="", color=c["fp8"], label="Qwen official FP8"),
        Line2D([], [], marker="o", ls="", color=c["nvfp4"], label="Unsloth NVFP4"),
        Line2D([], [], marker="*", ls="", color=c["ref"], label="BF16 reference"),
        Line2D([], [], lw=2.2, color=c["ours"], alpha=0.55, label="bootstrap 95 % CI"),
        Line2D([], [], marker="o", ls="", color=c["ext"],
               label="Qwen3.6-27B GGUF/FP8/NVFP4 (external, other protocol)"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True,
                    facecolor=c["bg"], edgecolor=c["grid"], labelcolor=c["fg"])
    leg.get_frame().set_alpha(0.72)

    # ---------------- panel B: top-1 agreement vs weight footprint
    pts = [(l, w, t, k) for l, w, _, t, k in MEASURED if k != "ref"]
    for label, w, top1, key in pts:
        bx.scatter([w], [top1], s=185 if key.startswith("ours") else 110,
                   marker="D" if key.startswith("ours") else "o",
                   color=c[key], edgecolor=c["bg"], linewidth=1.4, zorder=5)
        bx.annotate(f"{top1:.2f} %", (w, top1), color=c[key], fontsize=9.5,
                    weight="bold" if key.startswith("ours") else "normal",
                    xytext=(0, 12 if key != "ours2" else -20), textcoords="offset points", ha="center")
    bx.axhline(100.0, color=c["ref"], lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    bx.annotate("BF16 = 100 %", (55.0, 100.0), color=c["ref"], fontsize=8,
                xytext=(0, -13), textcoords="offset points", ha="right")
    bx.axvline(CEILING, color=c["nvfp4"], lw=1.1, ls=(0, (5, 3)), alpha=0.8)
    bx.set_xlabel("resident weight footprint (decimal GB)", color=c["fg"], fontsize=10)
    bx.set_ylabel("top-1 agreement with BF16 (%)", color=c["fg"], fontsize=10)
    bx.set_title("Greedy-token agreement", color=c["fg"], fontsize=11.5,
                 weight="bold", loc="left")
    bx.set_xlim(17, 33)
    bx.set_ylim(89.5, 100.8)

    fig.text(0.005, 0.012,
             "Measured on 1x RTX PRO 6000 Blackwell, TP1, GG r34 image. Held-out corpus (Gutenberg/arXiv/Wikipedia/CPython, "
             "0 calibration-contamination hits). Exact full-vocabulary two-pass KL(BF16 || candidate) through one shared "
             "BF16 LM head, float64, source-cluster bootstrap; runtime-repeat noise floor 0.000000. "
             "Grey: Qwen3.6-27B external series, quesma.com/blog/qwen-quantization-quality.",
             color=c["muted"], fontsize=7.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("svg", "png"):
        p = OUT / f"fidelity-vs-size-{theme_name}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=170 if ext == "png" else None)
        print("wrote", p, p.stat().st_size // 1024, "KB")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for t in THEMES:
        draw(t)
