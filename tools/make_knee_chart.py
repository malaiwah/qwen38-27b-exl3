#!/usr/bin/env python3
"""Concurrency / knee curves for the TB2.1 speed-run campaign (docs/46).

One line per (weight-size, shape) series: x = concurrency (log2), left y =
per-request tok/s (the owner's knee metric, log scale), right y = aggregate
tok/s (log scale). The knee point - largest concurrency where per-request
tok/s stays >= 50 % of single-stream - is marked explicitly on each series,
because that is the number the campaign actually uses to set -n; aggregate
is shown for context, not as the thing being optimised.

Every point comes from a `tb21_ladder.py` receipt (schema
qwen38-tb21-ladder/1). Multiple receipts overlay as separate weight-size
series on the same axes, so a card can show the flagship quant next to
whatever other sizes have been measured (BF16, other tiers) even when they
have fewer data points - a partial series is plotted as far as it goes and
labelled by its own last measured rung, never interpolated past real data.

Usage:
    make_knee_chart.py --receipt LABEL=path/to/tb21-ladder-*.json [--receipt ...]
                        --out-prefix /var/tmp/work/charts/tb21-knee
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "bg": "#ffffff", "fg": "#1a1a1a", "grid": "#c9c9c9",
        "panel_bg": "#f7f7f7",
        "series": ["#1f6feb", "#d1242f", "#1a7f37", "#8250df", "#bf8700"],
    },
    "dark": {
        "bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d",
        "panel_bg": "#161b22",
        "series": ["#58a6ff", "#ff7b72", "#3fb950", "#bc8cff", "#e3b341"],
    },
}

SHAPE_TITLES = {
    "512x256": "512-in / 256-out (short turn)",
    "4kx1k": "4,096-in / 1,024-out (typical turn)",
    "30kx2k": "30,720-in / 2,048-out (long-context turn)",
}


def load_series(spec: str) -> tuple[str, dict]:
    label, path = spec.split("=", 1)
    return label, json.loads(Path(path).read_text())


def style_axes(ax, c: dict) -> None:
    ax.set_facecolor(c["panel_bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=8.5)
    ax.grid(True, which="both", color=c["grid"], lw=0.6, alpha=0.4, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def draw_shape_panel(ax, c: dict, shape_key: str,
                     series: list[tuple[str, dict, str]]) -> bool:
    """One shape's per-request-tok/s-vs-concurrency panel. Returns True if
    at least one series had data for this shape (so callers can skip empty
    axes rather than publish a blank panel)."""
    style_axes(ax, c)
    plotted = False
    for label, receipt, color in series:
        table = receipt.get("tables", {}).get(shape_key)
        if not table or not table.get("cells"):
            continue
        plotted = True
        cells = sorted(table["cells"], key=lambda x: x["concurrency"])
        xs = [cell["concurrency"] for cell in cells]
        ys = [cell["median_per_request_tok_s"] for cell in cells]
        ax.plot(xs, ys, "-o", color=color, ms=4.5, lw=1.6,
                label=label, zorder=4)

        knee = table.get("knee", {})
        if knee.get("computable"):
            kx = knee["knee_concurrency"]
            ky = knee["knee_per_request_tok_s"]
            ax.scatter([kx], [ky], marker="D", s=70, facecolor="none",
                      edgecolor=color, linewidth=2.0, zorder=6)
            at_cap = " (at sweep cap - extend)" if knee.get("knee_at_cap") else ""
            ax.annotate(f"knee C{kx}{at_cap}", (kx, ky),
                       textcoords="offset points", xytext=(6, -12),
                       fontsize=7.6, color=color, zorder=6)

        single = cells[0]["median_per_request_tok_s"]
        ax.axhline(single * 0.5, color=color, lw=0.8, ls=":", alpha=0.55,
                  zorder=2)

    if plotted:
        ax.set_title(SHAPE_TITLES.get(shape_key, shape_key),
                    color=c["fg"], fontsize=9.6, pad=8)
        ax.set_xlabel("concurrency (requests)", color=c["fg"], fontsize=8.6)
        ax.set_ylabel("per-request tok/s (log)", color=c["fg"], fontsize=8.6)
    return plotted


def draw_aggregate_panel(ax, c: dict, shape_key: str,
                         series: list[tuple[str, dict, str]]) -> bool:
    """Companion aggregate-throughput panel for the same shape - context,
    not the optimisation target. Dashed to visually subordinate it to the
    per-request panel above."""
    style_axes(ax, c)
    plotted = False
    for label, receipt, color in series:
        table = receipt.get("tables", {}).get(shape_key)
        if not table or not table.get("cells"):
            continue
        plotted = True
        cells = sorted(table["cells"], key=lambda x: x["concurrency"])
        xs = [cell["concurrency"] for cell in cells]
        ys = [cell["median_aggregate_tok_s"] for cell in cells]
        ax.plot(xs, ys, "--s", color=color, ms=3.6, lw=1.2, alpha=0.85,
                label=f"{label} (aggregate)", zorder=3)
    if plotted:
        ax.set_xlabel("concurrency (requests)", color=c["fg"], fontsize=8.6)
        ax.set_ylabel("aggregate tok/s (log)", color=c["fg"], fontsize=8.6)
    return plotted


def draw(theme: str, series: list[tuple[str, dict]], out_prefix: Path) -> Path:
    c = THEMES[theme]
    shape_keys = SHAPE_TITLES.keys()
    colored = [(label, receipt, c["series"][i % len(c["series"])])
              for i, (label, receipt) in enumerate(series)]

    fig, axes = plt.subplots(2, len(list(shape_keys)), figsize=(13.5, 7.2),
                             squeeze=False)
    fig.patch.set_facecolor(c["bg"])

    any_plotted = False
    for col, shape_key in enumerate(shape_keys):
        top = draw_shape_panel(axes[0][col], c, shape_key, colored)
        bot = draw_aggregate_panel(axes[1][col], c, shape_key, colored)
        any_plotted = any_plotted or top or bot
        if not (top or bot):
            for row in (0, 1):
                axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, 1.02), ncol=min(len(handles), 5),
                  fontsize=8.6, facecolor=c["panel_bg"],
                  edgecolor=c["grid"], labelcolor=c["fg"])

    fig.suptitle(
        "TB2.1 speed-run: per-request throughput vs. concurrency, by shape "
        "and weight size\ndiamond = knee (largest concurrency with "
        "per-request \u2265 50 % of single-stream); dotted line = that 50 % "
        "floor",
        color=c["fg"], fontsize=10.2, y=1.10)
    fig.text(0.5, 0.005,
            "Every point from a tb21_ladder.py receipt (docs/46); knee rule "
            "and shapes fixed before any sweep ran.",
            ha="center", color=c["fg"], fontsize=7.6, alpha=0.75)

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = out_prefix.with_name(f"{out_prefix.name}-{theme}.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=c["bg"], bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, facecolor=c["bg"], bbox_inches="tight", dpi=150)
    plt.close(fig)
    if not any_plotted:
        raise SystemExit("no series had data for any shape - check --receipt paths")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--receipt", action="append", required=True,
                    metavar="LABEL=PATH",
                    help="repeatable: a weight-size series, e.g. "
                         "'hydrated (20.3 GiB)=receipts/tb21-ladder-1x-hyd.json'")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    series = [load_series(s) for s in args.receipt]
    for theme in THEMES:
        out = draw(theme, series, args.out_prefix)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
