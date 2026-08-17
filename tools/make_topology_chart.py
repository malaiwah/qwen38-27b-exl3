#!/usr/bin/env python3
"""The 8x topology chart: DP8 vs TP2xDP4 vs TP4xDP2, and why TP8 is absent.

This regenerates the card asset that previously existed with no committed
generator, and it adds the two things that were missing when it was first drawn:
the third measured arm (TP4xDP2) and the fact that the fourth arm is
ARCHITECTURALLY IMPOSSIBLE rather than merely unmeasured.

Panel A - per-request tok/s, which is what a single agent feels. The crossover
is the point: TP wins below the knee and DP wins above it, and the winner at C1
even swaps between shapes (TP2xDP4 at 512x256, TP4xDP2 at 4kx1k).

Panel B - aggregate tok/s, which is what a fleet feels, with each arm's knee
marked. The knees land at 4 x the DATA-PARALLEL degree - C32 / C16 / C8 - which
is the law docs/46 s27 states; TP width inside a replica lowers the knee in
proportion instead of raising it.

Honesty features baked into the figure rather than left to a caption:
  * The short-prompt low-concurrency cells carry a measured +/-14-18 % within-arm
    CV (docs/46 s28), so those markers are drawn hollow and the band is annotated.
    Individual crossings there are noise, not signal.
  * TP8's refusal is printed with its arithmetic, so a reader cannot mistake the
    missing arm for an omission.

Usage:
    make_topology_chart.py --out-prefix assets/tb21-topology [--receipts-dir receipts]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a1a", "grid": "#c9c9c9", "panel_bg": "#f7f7f7",
              "series": ["#1f6feb", "#d1242f", "#1a7f37"]},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d", "panel_bg": "#161b22",
             "series": ["#58a6ff", "#ff7b72", "#3fb950"]},
}

# label -> (receipt, data-parallel degree, tensor-parallel width)
ARMS = [
    ("DP8", "tb21-ladder-8x-dp8.json", 8, 1),
    ("TP2xDP4", "tb21-ladder-8x-tp2dp4.json", 4, 2),
    ("TP4xDP2", "tb21-ladder-8x-tp4dp2.json", 2, 4),
]
SHAPES = ["512x256", "4kx1k"]           # the two shapes ALL THREE arms ran
NOISY = {("512x256", 1), ("512x256", 2), ("512x256", 4)}   # measured CV 13.9-18.0 %

TP8_NOTE = ("TP8 is impossible, not unmeasured: EXL3 shards trellis tensors on 128-element\n"
            "boundaries and lm_head is 248320 = 128 x 1940 with 1940 = 2^2 x 5 x 97, so a TP8\n"
            "slice of 31040 = 128 x 242.5 is refused at load. Max TP width is 4 on any GPU count.")


def load(receipts: Path, fname: str) -> dict:
    d = json.loads((receipts / fname).read_text())
    out = {}
    for shape, tb in d["tables"].items():
        rows = {}
        for cell in tb["cells"]:
            aggs = [r["aggregate_tok_s"] for r in cell["repeats"]]
            pers = [q["per_request_tok_s"] for r in cell["repeats"]
                    for q in r["requests"] if q.get("ok") and q.get("per_request_tok_s")]
            if not pers or st.median(aggs) <= 0:
                continue                      # void cell (see docs/46 s28)
            rows[cell["concurrency"]] = (st.median(aggs), st.median(pers))
        knee = (tb.get("knee") or {}).get("knee_concurrency")
        out[shape] = (rows, knee)
    return out


def style(ax, c: dict) -> None:
    ax.set_facecolor(c["panel_bg"])
    ax.grid(True, which="both", color=c["grid"], lw=0.5, alpha=0.6)
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=8)
    ax.xaxis.label.set_color(c["fg"])
    ax.yaxis.label.set_color(c["fg"])
    ax.title.set_color(c["fg"])


def draw(theme: str, data: dict, out_prefix: Path) -> Path:
    c = THEMES[theme]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), facecolor=c["bg"])

    for col, shape in enumerate(SHAPES):
        ax_per, ax_agg = axes[0][col], axes[1][col]
        for i, (label, _f, dp, tp) in enumerate(ARMS):
            rows, knee = data[label][shape]
            xs = sorted(rows)
            per = [rows[x][1] for x in xs]
            agg = [rows[x][0] for x in xs]
            col_i = c["series"][i]
            ax_per.plot(xs, per, "-", color=col_i, lw=1.8,
                        label=f"{label}  (DP={dp}, TP={tp})")
            ax_agg.plot(xs, agg, "-", color=col_i, lw=1.8, label=label)
            # hollow markers where our own CV says the cell is untrustworthy
            for x, yp, ya in zip(xs, per, agg):
                noisy = (shape, x) in NOISY
                for ax, y in ((ax_per, yp), (ax_agg, ya)):
                    ax.plot([x], [y], "o", ms=5, color=col_i,
                            mfc=("none" if noisy else col_i), mew=1.4)
            if knee in rows:
                ax_agg.plot([knee], [rows[knee][0]], "*", ms=17, color=col_i,
                            mec=c["fg"], mew=0.7, zorder=5)
                ax_agg.annotate(f"knee C{knee} = 4x{dp}", (knee, rows[knee][0]),
                                textcoords="offset points", xytext=(6, -14),
                                color=col_i, fontsize=8, fontweight="bold")

        for ax, ylab, title in ((ax_per, "per-request tok/s", f"{shape} - what ONE agent feels"),
                                (ax_agg, "aggregate tok/s", f"{shape} - what the FLEET feels")):
            style(ax, c)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("client concurrency")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, facecolor=c["panel_bg"], edgecolor=c["grid"],
                      labelcolor=c["fg"])
        ax_per.annotate("hollow markers = within-arm CV 13.9-18.0 %\n(docs/46 s28) - crossings here are noise",
                        (0.02, 0.03), xycoords="axes fraction", fontsize=7,
                        color=c["fg"], alpha=0.85)

    fig.suptitle("Qwen3.8-27B EXL3 K5/K6 hydrated - 8x topology ladder, one host, ForceP2P on\n"
                 "DP for throughput, TP for single-stream latency; the knee tracks the DATA-PARALLEL degree",
                 color=c["fg"], fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, TP8_NOTE, ha="center", va="bottom", color=c["fg"],
             fontsize=7.5, family="monospace", alpha=0.9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))

    out = out_prefix.with_name(out_prefix.name + f"-{theme}.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=c["bg"], bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), facecolor=c["bg"], bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts-dir", type=Path, default=Path("receipts"))
    ap.add_argument("--out-prefix", type=Path, required=True)
    a = ap.parse_args()

    data = {label: load(a.receipts_dir, f) for label, f, _dp, _tp in ARMS}
    for label, _f, _dp, _tp in ARMS:
        missing = [s for s in SHAPES if s not in data[label]]
        if missing:
            raise SystemExit(f"{label} is missing shape(s) {missing}; refusing to draw a partial ladder")
    for theme in THEMES:
        print(f"wrote {draw(theme, data, a.out_prefix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
