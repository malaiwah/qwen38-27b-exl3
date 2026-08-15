#!/usr/bin/env python3
"""Card chart: measured fidelity/capacity profiles around a 32 GB budget.

The hydrated, online and K4 points use an independent real RTX 5090 run with
MTP-3. The context-edition point also uses MTP-3, but on an RTX PRO 6000 whose
vLLM engine budget was capped to 30.24 GiB and whose image profile was capped
to 8.4 MP; it awaits a hard-limit RTX 5090 rerun. The chart labels that
difference instead of presenting unlike hardware runs as one comparison.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/var/tmp/work/charts")
FP8_KLD = 0.012798

# label, demonstrated/configured context, overlap-corrected v3 mean KLD,
# resident GiB, color, marker
POINTS = [
    ("K5/K6 hydrated — RTX 5090, MTP-3", 185600, 0.007172, 20.31, "hyd", "o"),
    ("K5/K6 online K6 — RTX 5090, MTP-3", 185600, 0.007945, 20.32, "k6", "o"),
    ("context + int8 embed — capped budget, MTP-3, 8.4 MP; 5090 pending",
     262144, 0.009459, 18.41, "ctx", "*"),
    ("K5/K6 online K5 — RTX 5090, MTP-3; needle verified",
     206400, 0.011801, 19.82, "k5", "*"),
    ("K4 — RTX 5090, MTP-3", 262144, 0.029679, 17.89, "k4", "o"),
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

    # Identity and unlike runtime profiles belong in the legend, not overlapping labels.
    for label, ctx, kld, gib, key, marker in POINTS:
        size = 420 if marker == "*" else 170
        ax.scatter([ctx], [kld], s=size, marker=marker, color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=1.2,
                   label=f"{label} — {gib:.2f} GiB, KLD {kld:.6f}")

    ax.axhline(FP8_KLD, color=c["fp8"], lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.annotate("official FP8: 0.012798 at 28.51 GiB — below this line is closer to BF16",
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
    ax.set_yticks([0.006, 0.008, 0.01, FP8_KLD, 0.02, 0.03, 0.04])
    ax.set_yticklabels(["0.006", "0.008", "0.010", "0.0128", "0.020", "0.030", "0.040"])
    ax.set_xlabel("demonstrated/configured context (tokens; see per-point runtime profile)",
                  color=c["fg"], fontsize=10)
    ax.set_ylabel("overlap-corrected v3 mean KL divergence (log, lower is better)",
                  color=c["fg"], fontsize=10)
    ax.set_title("Fidelity/capacity profiles: hard-limit RTX 5090 results and one capped-budget candidate",
                 color=c["fg"], fontsize=11.5, pad=12)
    leg = ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, labelspacing=0.7)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])
    fig.text(0.005, 0.030,
             "Circles: real RTX 5090, MTP-3. Purple star: MTP-3 + 8.4 MP cap on a "
             "96 GB SM120 with vLLM capped to 30.24 GiB; physical 5090 validation pending.",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.012,
             "Stars mark generation proof: online K5 retrieved at 205,021 prompt tokens; "
             "context retrieved at 261,794 and in a 236,824-token seven-megapixel request.",
             color=c["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("svg", "png"):
        p = OUT / f"context-frontier-{theme}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=170 if ext == "png" else None)
        if ext == "svg":
            p.write_text(
                "\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n"
            )
        print("wrote", p, f"{p.stat().st_size // 1024} KB")
    plt.close(fig)


if __name__ == "__main__":
    for t in THEMES:
        draw(t)
