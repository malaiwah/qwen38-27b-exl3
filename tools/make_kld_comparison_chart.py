#!/usr/bin/env python3
"""Two-protocol quant-family comparison for the Hugging Face cards.

The two panels are deliberately NOT one chart. Left is our v5 held-out suite,
body-only through one shared BF16 head, x = weights actually resident under
vLLM. Right is turboderp's published chart labels for his own protocol
(OpenWebText, 8 x 8192 formatted tokens, his own BF16 reference), x = quantized
decoder weight excluding embeddings and including the output head. Nothing is
drawn across the two axes: our builds appear on his panel only as a vertical
decoder-weight marker, with no y-value, because we have never run his protocol.

Emits light and dark SVG + PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path("/var/tmp/work/charts")
RECEIPTS = Path(__file__).resolve().parent.parent / "receipts"

# ---------------------------------------------------------------- our protocol
# v5 held-out suite, 10,480,640 scored positions, 842 source clusters
# (receipts/kld5-10M-*.json). Body-only, both operands through one shared BF16
# head. Intervals are 95% source-cluster bootstrap. Resident GiB is weights
# measured resident under vLLM -- not serialized bytes on disk.
# label, resident GiB, mean, ci_low, ci_high, p99.9, colour key,
# mean-label (dx, dy, ha), tail-label (dx, dy, ha), interval cap width.
# hydrated and online sit 0.01 GiB apart with overlapping intervals, so their
# caps get different widths to stay tellable apart. The x values are measured
# and are never nudged to make room.
OURS = [
    ("hydrated K5/K6", 20.31, 0.002760, 0.002540, 0.003020, 0.1413, "hyd",
     (0, -20, "center"), (-8, -3, "right"), 9.0),
    ("online K5/K6", 20.32, 0.003210, 0.002982, 0.003480, 0.1527, "k6",
     (11, 6, "left"), (8, 4, "left"), 4.0),
    ("context edition", 18.41, 0.003509, 0.003220, 0.003852, 0.1803, "ctx",
     (0, 11, "center"), (0, 10, "center"), 6.0),
    ("official Qwen FP8", 28.51, 0.005294, 0.004927, 0.005728, 0.2560, "fp8",
     (-9, 8, "right"), (0, 10, "center"), 6.0),
    ("K4", 17.89, 0.010604, 0.009640, 0.011746, 0.5933, "k4",
     (11, 2, "left"), (0, 10, "center"), 6.0),
]
OUR_POSITIONS = 10_480_640
OUR_CLUSTERS = 842
TAIL_POSITIONS = 2_096_128  # receipts/kld5-2M-tail-*.json, p99.9 window

# --------------------------------------------------------------- his protocol
# turboderp/Qwen3.8-27B-exl3 published chart labels: his printed numbers, read
# off his rasters, not a published data file. Never mixed with ours.
# family key -> (display name, marker,
#                [(x GiB, mean, median, point label, (dx, dy, ha))])
HIS = [
    ("exl3", "EXL3 (bpw)", "o", [
        (6.6, 0.351, 0.1268, "2.00", (0, 10, "center")),
        (8.1, 0.299, 0.0783, "2.50", (0, 10, "center")),
        (9.5, 0.112, 0.0273, "3.00", (0, 10, "center")),
        (12.4, 0.052, 0.0086, "4.00", (0, 10, "center")),
        (15.0, 0.014, 0.0022, "5.00", (0, -14, "center")),
        (17.8, 0.007, 0.0010, "6.00", (-7, -6, "right")),
    ]),
    ("udq", "GGUF UD", "s", [
        (9.6, 0.237, 0.0916, "Q2_K_XL", (0, 10, "center")),
        (12.0, 0.089, 0.0238, "Q3_K_XL", (0, 10, "center")),
        (16.0, 0.039, 0.0069, "Q4_K_XL", (0, -14, "center")),
        (18.0, 0.019, 0.0032, "Q5_K_XL", (7, 5, "left")),
        (23.1, 0.009, 0.0012, "Q6_K_XL", (0, 10, "center")),
    ]),
    ("iq", "GGUF-IQ", "D", [(13.9, 0.055, 0.0131, "IQ4_XS", (0, 11, "center"))]),
    ("nvfp4", "NVFP4 (Unsloth)", "^",
     [(17.9, 0.041, 0.0103, "NVFP4", (0, 11, "center"))]),
    ("fp8", "FP8 (Qwen)", "v", [(25.1, 0.023, 0.0035, "FP8", (0, 11, "center"))]),
]
HIS_FLOOR_MEAN = 0.0052    # his synthetic noise floor, mean
HIS_FLOOR_MEDIAN = 0.0007  # his synthetic noise floor, median
HIS_POSITIONS = 8 * 8192   # 65,536 formatted OpenWebText positions

# Roles that make up his x-axis: decoder linear storage, embeddings excluded,
# output head included. Vision, MTP draft and norms are not decoder weight.
HIS_AXIS_ROLES = ("full_attention", "linear_attention", "mlp_gate_proj",
                  "mlp_up_proj", "mlp_down_proj", "lm_head")
# Our builds that have an authoritative per-tensor manifest, so their placement
# on his x-axis is exact rather than estimated. A build without a manifest is
# omitted rather than guessed at.
# On his axis our builds are annotations, not data, so they wear neutral
# annotation colours: reusing a family colour would invite the reader to attach
# them to that family's curve.
# label, manifest, colour key, label side, dash pattern
OUR_ON_HIS_AXIS = [
    ("hydrated", "hydrated-quantization-manifest.json", "ann", "right",
     (0, (7, 3))),
    ("context edition", "context-quantization-manifest.json", "tail", "left",
     (0, (2, 2))),
]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d0d0d0", muted="#8a8a8a",
                  hyd="#0b5d1e", k6="#0969da", ctx="#8250df", k4="#a40e26",
                  fp8="#b08800", exl3="#0b5d1e", udq="#0969da", iq="#8250df",
                  nvfp4="#a40e26", tail="#57606a", floor="#8a8a8a",
                  ann="#1a1a1a"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e",
                 hyd="#56d364", k6="#58a6ff", ctx="#bc8cff", k4="#f85149",
                 fp8="#d29922", exl3="#56d364", udq="#58a6ff", iq="#bc8cff",
                 nvfp4="#f85149", tail="#8b949e", floor="#8b949e",
                 ann="#e6edf3"),
}


def decoder_weight_gib(manifest: Path) -> tuple[int, float]:
    """Bytes and GiB of decoder linear weight on his axis, from a manifest."""
    roles = json.loads(manifest.read_text())["roles"]
    total = sum(roles[r]["bytes"] for r in HIS_AXIS_ROLES)
    return total, total / 1024 ** 3


def style_axes(ax, c: dict) -> None:
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.45, zorder=0)


def draw_ours(ax, c: dict) -> None:
    style_axes(ax, c)
    for (label, gib, mean, lo, hi, p999, key,
         (dx, dy, ha), (tdx, tdy, tha), cap) in OURS:
        # The vertical spine shows how far this build's tail runs past its mean.
        ax.plot([gib, gib], [mean, p999], color=c[key], lw=1.0, alpha=0.38,
                zorder=2)
        # The bootstrap interval is only a few points tall on a range that has
        # to reach the tail, so the mean marker stays small enough for the caps
        # to clear it.
        ax.errorbar([gib], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor=c[key], elinewidth=2.0, capsize=cap, capthick=1.9,
                    zorder=3)
        ax.scatter([gib], [mean], s=62, marker="o", color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=0.9)
        ax.scatter([gib], [p999], s=150, marker="v", facecolor=c["bg"],
                   edgecolor=c[key], linewidth=1.5, zorder=4)
        ax.annotate(f"{label}\n{mean:.6f} at {gib:.2f} GiB", (gib, mean),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=8.5, color=c["fg"], zorder=5)
        ax.annotate(f"p99.9 {p999:.4f}", (gib, p999), textcoords="offset points",
                    xytext=(tdx, tdy), ha=tha, fontsize=7.5, color=c[key],
                    zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(16.8, 30.5)
    ax.set_ylim(0.0016, 1.25)
    ax.set_xticks([18, 20, 22, 24, 26, 28, 30])
    ax.set_yticks([0.002, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0])
    ax.set_yticklabels(["0.002", "0.003", "0.005", "0.010", "0.030", "0.050",
                        "0.100", "0.300", "0.500", "1.000"])
    ax.set_xlabel("weights measured resident under vLLM (GiB) — resident "
                  "weights, not serialized disk bytes", color=c["fg"],
                  fontsize=10)
    ax.set_ylabel("our v5 KL divergence, nats/token (log, lower is better)",
                  color=c["fg"], fontsize=10)
    ax.set_title(
        "Our protocol: v5 held-out suite, body-only through one shared BF16 head\n"
        f"mean over {OUR_POSITIONS:,} positions; p99.9 over the "
        f"{TAIL_POSITIONS:,}-position tail window",
        color=c["fg"], fontsize=10.5, pad=10)

    handles = [
        Line2D([], [], marker="o", ls="none", color=c["tail"], markersize=6,
               markeredgecolor=c["bg"],
               label=f"mean, 95% source-cluster bootstrap bars ({OUR_POSITIONS:,} positions)"),
        Line2D([], [], marker="v", ls="none", markerfacecolor=c["bg"],
               markeredgecolor=c["tail"], color=c["tail"], markersize=9,
               label=f"p99.9 tail, histogram-bin estimate ({TAIL_POSITIONS:,} positions)"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8,
                    framealpha=0.92)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])


def draw_his(ax, c: dict) -> None:
    style_axes(ax, c)
    for key, name, marker, points in HIS:
        xs = [p[0] for p in points]
        means = [p[1] for p in points]
        medians = [p[2] for p in points]
        if len(points) > 1:
            ax.plot(xs, means, color=c[key], lw=1.2, alpha=0.75, zorder=2)
            ax.plot(xs, medians, color=c[key], lw=1.2, alpha=0.75,
                    ls=(0, (4, 2)), zorder=2)
        ax.scatter(xs, means, s=95, marker=marker, color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=1.0)
        ax.scatter(xs, medians, s=85, marker=marker, facecolor=c["bg"],
                   edgecolor=c[key], linewidth=1.3, zorder=4)
        for x, mean, _median, plabel, (dx, dy, ha) in points:
            # The ladder lines, noise floors and decoder-weight markers all run
            # through this panel; a tight background box keeps labels legible.
            ax.annotate(plabel, (x, mean), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5,
                        color=c[key], zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=c["bg"],
                                  edgecolor="none", alpha=0.85))

    # His synthetic noise floor: the level below which his protocol cannot
    # resolve a difference at all.
    ax.axhline(HIS_FLOOR_MEAN, color=c["floor"], lw=1.3, ls=(0, (5, 3)),
               zorder=1)
    ax.annotate(f"his synthetic noise floor — mean {HIS_FLOOR_MEAN}",
                (6.2, HIS_FLOOR_MEAN), textcoords="offset points",
                xytext=(0, 5), fontsize=8, color=c["floor"], zorder=5)
    ax.axhline(HIS_FLOOR_MEDIAN, color=c["floor"], lw=1.1, ls=(0, (2, 3)),
               zorder=1)
    ax.annotate(f"his synthetic noise floor — median {HIS_FLOOR_MEDIAN}",
                (6.2, HIS_FLOOR_MEDIAN), textcoords="offset points",
                xytext=(0, 5), fontsize=8, color=c["floor"], zorder=5)

    # Our builds enter this panel as an x-position only. Drawing a y for them
    # here would be inventing a measurement on a protocol we never ran, so each
    # label sits on the outer side of its own line and says so.
    for label, fname, key, side, dash in OUR_ON_HIS_AXIS:
        manifest = RECEIPTS / fname
        if not manifest.exists():
            continue
        raw, gib = decoder_weight_gib(manifest)
        print(f"  our {label} on his axis: {raw:,} B = {gib:.4f} GiB")
        ax.axvline(gib, color=c[key], lw=1.5, ls=dash, zorder=3)
        ax.annotate(
            f"our {label} build\n"
            f"{gib:.2f} GiB decoder weight\n"
            "x only — y UNMEASURED here",
            (gib, 1.30), rotation=90, va="top",
            ha="left" if side == "right" else "right",
            textcoords="offset points", xytext=(4 if side == "right" else -4, 0),
            fontsize=7.5, color=c[key], zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(5.4, 26.6)
    ax.set_ylim(0.0005, 1.45)
    ax.set_xticks([6, 9, 12, 15, 18, 21, 24])
    ax.set_yticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.001", "0.003", "0.010", "0.030", "0.100", "0.300",
                        "1.000"])
    ax.set_xlabel("his axis: quantized decoder weight (GiB), embeddings "
                  "excluded, output head included", color=c["fg"], fontsize=10)
    ax.set_ylabel("his published KL divergence (log, lower is better)",
                  color=c["fg"], fontsize=10)
    ax.set_title(
        "His protocol: turboderp's published labels, his own BF16 reference\n"
        f"OpenWebText, 8 x 8192 = {HIS_POSITIONS:,} formatted positions; "
        "filled = mean, hollow = median",
        color=c["fg"], fontsize=10.5, pad=10)

    # One entry per family: the fill convention is in the title, so the legend
    # stays narrow enough to clear the decoder-weight markers on the left.
    handles = [Line2D([], [], marker=marker, ls="none", color=c[key],
                      markeredgecolor=c["bg"], markersize=8, label=name)
               for key, name, marker, _points in HIS]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                    framealpha=0.92, ncol=2, columnspacing=1.0,
                    handletextpad=0.4)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])


def draw(theme: str) -> list[Path]:
    c = THEMES[theme]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16.4, 7.4),
                                     facecolor=c["bg"])
    draw_ours(ax_l, c)
    draw_his(ax_r, c)

    fig.suptitle("Quantization families, two protocols, side by side — the "
                 "panels are not interchangeable",
                 color=c["fg"], fontsize=12.5, y=0.985)
    fig.text(0.005, 0.048,
             f"Left (ours): v5 held-out suite, {OUR_POSITIONS:,} scored "
             f"positions across {OUR_CLUSTERS} source clusters, body-only "
             "through one shared BF16 head, weights resident under vLLM; "
             f"p99.9 from the {TAIL_POSITIONS:,}-position tail window.",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.028,
             f"Right (his): turboderp's OpenWebText protocol, 8 x 8192 = "
             f"{HIS_POSITIONS:,} formatted positions, his own BF16 reference. "
             "His values are chart labels read off his published images, not a "
             "published data file.",
             color=c["muted"], fontsize=7.5)
    fig.text(0.005, 0.008,
             "The two panels are NOT interchangeable and no ratio between them "
             "is meaningful: different corpus, window, reference numerics, "
             "vocabulary handling and head placement. The same NVFP4 "
             "checkpoint reads 0.041 on his protocol and 0.0927 on ours.",
             color=c["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.065, 1, 0.965))

    written = []
    for ext in ("svg", "png"):
        p = OUT / f"kld-family-comparison-{theme}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=170 if ext == "png" else None)
        if ext == "svg":
            p.write_text(
                "\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n"
            )
        print("wrote", p, f"{p.stat().st_size // 1024} KB")
        written.append(p)
    plt.close(fig)
    return written


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for t in THEMES:
        draw(t)
