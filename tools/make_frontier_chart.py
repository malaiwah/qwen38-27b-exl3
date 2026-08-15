#!/usr/bin/env python3
"""Card chart 3: the operating frontier on a 32 GB card.

Each point is a build measured on the same held-out suite, placed by the context it actually
serves on a 5090-sized budget (multimodal profiling on, MTP-3) against its divergence from
BF16. The dashed line is official FP8's divergence: everything below it is closer to the
unquantized model than Qwen's own FP8 release, and the K4 build is the only one that reaches
native 262,144.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/var/tmp/work/charts")
FP8_KLD = 0.013126

# label, max context with MTP-3 on a 32 GB card, mean KLD, resident GiB, key, verified needle
POINTS = [
    ("K5/K6 hydrated, attention K6 offline", 180224, 0.007406, 20.31, "hyd", None),
    ("K5/K6 online K6 (flexible)", 180224, 0.008157, 20.32, "k6", None),
    ("context edition + int8 embeddings", 262144, 0.009738, 18.13, "ctx", 227334),
    ("K5/K6 online K5", 196608, 0.012135, 19.82, "k5", None),
    ("K4 build", 262144, 0.030736, 17.89, "k4", None),
]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d0d0d0", muted="#8a8a8a", fp8="#b08800",
                  hyd="#0b5d1e", k6="#0969da", ctx="#8250df", k5="#bc4c00", k4="#a40e26"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e", fp8="#d29922",
                 hyd="#56d364", k6="#58a6ff", ctx="#bc8cff", k5="#ffa657", k4="#f85149"),
}


def draw(theme: str) -> None:
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.2, 5.9), facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.45, zorder=0)

    # Points cluster tightly on the context axis, so identity goes in the legend rather than
    # in labels that would overlap each other.
    for label, ctx, kld, gib, key, needle in POINTS:
        marker = "*" if needle else "o"
        size = 420 if needle else 170
        flat = label.replace("\n", " ")
        ax.scatter([ctx], [kld], s=size, marker=marker, color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=1.2,
                   label=f"{flat} — {gib:.2f} GiB, KLD {kld:.6f}"
                         + (", needle verified" if needle else ""))

    ax.axhline(FP8_KLD, color=c["fp8"], lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.annotate("official FP8: 0.013126 at 28.51 GiB — below this line is closer to BF16",
                (8000, FP8_KLD), textcoords="offset points", xytext=(0, 6),
                fontsize=8.5, color=c["fp8"])
    ax.axvline(262144, color=c["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("native 262,144", (262144, 0.0055), rotation=90, va="bottom", ha="right",
                fontsize=8.5, color=c["muted"], xytext=(-5, 0), textcoords="offset points")

    ax.set_yscale("log")
    ax.set_xlim(0, 300000)
    ax.set_ylim(0.005, 0.05)
    ax.set_xticks([65536, 131072, 196608, 262144])
    ax.set_xticklabels(["64k", "128k", "192k", "262k"])
    ax.set_yticks([0.006, 0.008, 0.01, 0.013126, 0.02, 0.03, 0.04])
    ax.set_yticklabels(["0.006", "0.008", "0.010", "0.0131", "0.020", "0.030", "0.040"])
    ax.set_xlabel("context served on a 32 GB card, vision enabled, tokens "
                  "(MTP-3 on, except the starred point which needs MTP off)",
                  color=c["fg"], fontsize=10)
    ax.set_ylabel("mean KL divergence from BF16 (log, lower is better)", color=c["fg"],
                  fontsize=10)
    ax.set_title("Native 262,144 context at 26 % below official FP8 divergence, on a 32 GB card",
                 color=c["fg"], fontsize=11.5, pad=12)
    leg = ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92, labelspacing=0.7)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])
    fig.text(0.005, 0.030,
             "Measured on one RTX PRO 6000 with the engine budget capped to 31.2 GiB, which is "
             "what a 32 GB RTX 5090 gives vLLM at utilisation 0.97.",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.012,
             "The star marks the only build whose long context is verified by generation: "
             "exact needle retrieval at three depths from 227,334-token prompts, serving at "
             "--max-model-len 262144.",
             color=c["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("svg", "png"):
        p = OUT / f"context-frontier-{theme}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=170 if ext == "png" else None)
        print("wrote", p, f"{p.stat().st_size // 1024} KB")
    plt.close(fig)


if __name__ == "__main__":
    for t in THEMES:
        draw(t)
