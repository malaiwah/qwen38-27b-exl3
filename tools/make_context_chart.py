#!/usr/bin/env python3
"""Card chart 2: context per card, calibrated against a real RTX 5090.

An earlier version of this chart was wrong in two ways and claimed native 262,144 context
was reachable on a 32 GB card. An independent tester on an actual RTX 5090 showed it is
not, and the two errors were:

  * the KV cost per token omitted MTP. With `num_speculative_tokens: 3` the engine needs
    9.13 GiB of KV for 262,144 tokens, not 8.18 GiB - 37.4 KB/token instead of 33.5.
  * usable VRAM was taken as 31.84 GiB. A 5090 reports 31.39 GiB usable.

Coefficients here come from that hardware: KV 37,396 B/token with MTP3, fixed overhead
2.75 GiB at --max-num-seqs 4 (activation + non-torch + decode graphs), and per-card usable
VRAM rather than marketing capacity. Solid markers are contexts that were **run** on the
5090, including an exact long-context retrieval check.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/var/tmp/work/charts")
GIB = float(1 << 30)
TARGET = 262144
KV_BYTES_PER_TOKEN = 37396.0   # with MTP3, from the 5090's own 9.13 GiB requirement
FIXED_GIB = 2.75               # --max-num-seqs 4, decode-only CUDA graphs

# label, resident GiB, mean KLD, colour key, demonstrated context on the 5090 (or None)
WIDTHS = [
    ("K5/K6, attention K6  ·  KLD 0.008157", 20.24, "k6", 185600),
    ("K5/K6, attention K5  ·  KLD 0.012135", 19.40, "k5", 206400),
    ("K4 build, attention K6  ·  KLD 0.030736", 17.89, "k4", 262144),
]

CARDS = [
    (15.90, "16 GB usable 15.9 — RTX 5070 Ti / 5080"),
    (23.88, "24 GB usable 23.9 — RTX PRO 4000 Blackwell"),
    (31.39, "32 GB usable 31.4 — RTX 5090 / RTX PRO 4500"),
]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d0d0d0", muted="#8a8a8a",
                  k6="#0969da", k5="#0b5d1e", k4="#a40e26", card="#8a8a8a",
                  target="#b08800"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e",
                 k6="#58a6ff", k5="#56d364", k4="#f85149", card="#8b949e",
                 target="#d29922"),
}


def need_gib(tokens: float, weights: float) -> float:
    return weights + FIXED_GIB + tokens * KV_BYTES_PER_TOKEN / GIB


def draw(theme: str) -> None:
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.6, 6.2), facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.45, zorder=0)

    xs = [0, 65536, 131072, 196608, 262144, 300000]
    for wi, (label, w, key, demo) in enumerate(WIDTHS):
        ax.plot(xs, [need_gib(x, w) for x in xs], color=c[key], lw=2.0, zorder=3,
                label=label)
        if demo:
            ax.scatter([demo], [need_gib(demo, w)], s=150, marker="o", color=c[key],
                       zorder=4, edgecolor=c["bg"], linewidth=1.2)
            ax.annotate(f"{demo:,} run", (demo, need_gib(demo, w)),
                        textcoords="offset points", xytext=(0, -17 - 13 * (wi % 2)),
                        ha="center",
                        fontsize=8.5, color=c[key], weight="bold",
                        bbox=dict(facecolor=c["bg"], edgecolor="none", pad=0.8))

    for gib, name in CARDS:
        ax.axhline(gib, color=c["card"], lw=1.3, ls="-", alpha=0.75, zorder=1)
        ax.annotate(name, (1500, gib), textcoords="offset points", xytext=(0, 5),
                    ha="left", fontsize=8.5, color=c["card"],
                    bbox=dict(facecolor=c["bg"], edgecolor="none", pad=1.2))
        if gib < 30:
            continue
        safe = gib * 0.95
        ax.axhline(safe, color=c["card"], lw=1.0, ls=(0, (5, 4)), alpha=0.9, zorder=1)
        ax.annotate("utilisation 0.95 — vision-safe ceiling (0.98 OOMs on a "
                    "3,264-token image)", (1500, safe), textcoords="offset points",
                    xytext=(0, -13), ha="left", fontsize=8, color=c["card"],
                    bbox=dict(facecolor=c["bg"], edgecolor="none", pad=1.0))
        # Model ceilings at the vision-safe utilisation, stacked so they cannot collide
        # with the demonstrated markers sitting on the same line.
        for i, (label, w, key, _demo) in enumerate(WIDTHS):
            room = safe - w - FIXED_GIB
            if room <= 0:
                continue
            tok = min(room * GIB / KV_BYTES_PER_TOKEN, 300000)
            ax.annotate(f"{tok/1000:.0f}k model", (tok, safe), textcoords="offset points",
                        xytext=(0, 10 + 13 * i), ha="center", fontsize=8.5, color=c[key],
                        bbox=dict(facecolor=c["bg"], edgecolor="none", pad=0.8))

    ax.axvline(TARGET, color=c["target"], lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.annotate("native context 262,144", (TARGET, 15.2), rotation=90, va="bottom",
                ha="right", color=c["target"], fontsize=9,
                xytext=(-5, 0), textcoords="offset points",
                bbox=dict(facecolor=c["bg"], edgecolor="none", pad=1.0))

    ax.set_xlim(0, xs[-1])
    ax.set_ylim(14.5, 34)
    ax.set_xticks([0, 65536, 131072, 196608, 262144])
    ax.set_xticklabels(["0k", "64k", "128k", "192k", "262k"])
    ax.set_xlabel("context served, tokens (fp8 KV, MTP-3 speculative decoding on)",
                  color=c["fg"], fontsize=10)
    ax.set_ylabel("GPU memory required, GiB", color=c["fg"], fontsize=10)
    ax.set_title("Native 262,144 needs the K4 build on a 32 GB card; K5/K6 reaches "
                 "~206k there", color=c["fg"], fontsize=11.5, pad=12)
    leg = ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])
    fig.text(0.005, 0.048,
             "Calibrated on an independent RTX 5090 (31.39 GiB usable, TP1, FP8 E4M3 KV, "
             "MTP-3, decode-only CUDA graphs, vision enabled): KV costs 37.4 KB/token and "
             "262,144 tokens need 9.13 GiB.",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.030,
             "Circles are contexts actually run there; the K5 point passed an exact "
             "long-context retrieval check at 205,021 tokens. Lines are the linear model "
             "(weights + 2.75 GiB + KV).",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.012,
             "Attention width is the launch-time env var VLLM_EXL3_ONLINE_TRELLIS_BITS, so "
             "one K5/K6 download serves both of its lines. Without MTP, KV drops to "
             "33.5 KB/token and every line moves right by ~11 %.",
             color=c["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    for ext in ("svg", "png"):
        p = OUT / f"context-per-card-{theme}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=170 if ext == "png" else None)
        print("wrote", p, f"{p.stat().st_size // 1024} KB")
    plt.close(fig)


if __name__ == "__main__":
    for t in THEMES:
        draw(t)
