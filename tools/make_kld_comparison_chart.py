#!/usr/bin/env python3
"""Quant-family comparison for the Hugging Face cards.

The left column and the right panel are deliberately NOT one chart.

Left column is ONE measurement protocol for every family we have measured:
shard 0 of the v5 held-out suite, the same 512 contexts and the same 1,048,064
scored positions for every point, KL(BF16 reference || candidate) over the full
vocabulary with both operands through one shared BF16 head
(receipts/cross-engine-comparator.json). GGUF points were captured under
llama.cpp, while the reference and vLLM points were captured under vLLM. The
unquantized-BF16 cross-engine control is drawn as a diagnostic reference line;
it is not subtracted because KL is neither additive nor a metric.

The left column is split into two stacked sub-panels sharing the y-axis because
the two families do not have one common size measurement: we measured resident
weights under vLLM for our own builds and official FP8, and we have serialized
bytes for the GGUF files and for our published payloads. Resident weights and
serialized bytes are different quantities, so they get different x-axes rather
than being merged into one dishonest axis.

Right panel is turboderp's published chart labels for his own protocol
(OpenWebText, 8 x 8192 formatted tokens, his own BF16 reference), x = quantized
decoder weight excluding embeddings and including the output head. Nothing is
drawn across the two protocols: our builds appear on his panel only as a
vertical decoder-weight marker, with no y-value, because we have never run his
protocol.

Emits light and dark SVG + PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets"
RECEIPTS = Path(__file__).resolve().parent.parent / "receipts"

# ---------------------------------------------------------------- our protocol
# Every y-value in the left column comes from this one receipt, so the panel
# cannot drift from the published table. Shard 0 of the v5 ladder: the same 512
# contexts and 1,048,064 positions for all eight candidates, one shared BF16
# head, source-cluster bootstrap intervals.
COMPARATOR = json.loads((RECEIPTS / "cross-engine-comparator.json").read_text())
CAND = COMPARATOR["candidates"]
SUITE = COMPARATOR["suite"]
FLOOR = COMPARATOR["engine_floor"]
POSITIONS = SUITE["scored_positions"]
CONTEXTS = SUITE["contexts"]
CLUSTERS = SUITE["source_clusters"]

# Table name, marker label, colour key, engine (engine picks the marker shape).
# The marker label is deliberately short: every number lives in the panel's own
# table, so nothing has to be crammed next to a point.
IDENTITY = {
    "gguf-q8_0": ("GGUF Q8_0", "Q8_0", "gguf", "llama.cpp"),
    "gguf-q6_k": ("GGUF Q6_K", "Q6_K", "gguf", "llama.cpp"),
    "gguf-q5_k_xl": ("GGUF UD-Q5_K_XL", "UD-Q5_K_XL", "gguf", "llama.cpp"),
    "hyd": ("hydrated K5/K6", "hydrated", "hyd", "vLLM"),
    "k5k6": ("online K5/K6", "online", "k6", "vLLM"),
    "ctx": ("context edition", "context", "ctx", "vLLM"),
    "fp8": ("official Qwen FP8", "official FP8", "fp8", "vLLM"),
    "k4": ("K4", "K4", "k4", "vLLM"),
}
MARKER = {"vLLM": "o", "llama.cpp": "s"}

# Upper sub-panel: x = weights measured resident under vLLM. No GGUF point can
# appear here -- we never measured llama.cpp resident weights, and inventing one
# from serialized bytes would be a guess.
# Every value is read from the receipt; this table only carries placement.
# id, resident GiB, short label (dx, dy, ha, va), p99.9 label (dx, dy, ha, va),
# interval cap width.
# hydrated and online sit 0.01 GiB apart with overlapping intervals, so their
# caps get different widths and their labels opposite sides to stay tellable
# apart. The x values are measured and are never nudged to make room.
RESIDENT = [
    ("hyd", 20.31, (-10, -1, "right", "center"), (-8, 0, "right", "center"),
     9.0),
    ("k5k6", 20.32, (10, 2, "left", "center"), (8, 0, "left", "center"), 4.0),
    ("ctx", 18.41, (0, 9, "center", "bottom"), (0, 10, "center", "bottom"),
     6.0),
    ("fp8", 28.51, (-10, 0, "right", "center"), (-8, 0, "right", "center"), 6.0),
    ("k4", 17.89, (10, -1, "left", "center"), (0, 10, "center", "bottom"), 6.0),
]
RESIDENT_TABLE = (23.7, 0.0034)  # x, y of the table's top row

# Lower sub-panel: x = serialized bytes. GGUF x comes from the pinned file
# sizes in the comparator receipt; ours comes from the immutable payload of the
# published build receipt, which is the same kind of quantity -- bytes that
# exist on disk. Online K5/K6 ships BF16 attention and is quantized at load, so
# its serialized bytes are not its operating point, and K4 has no published
# payload receipt; both are therefore absent here rather than estimated.
# id, build receipt for our payload (None = GGUF, size from the comparator),
# short label (dx, dy, ha, va), p99.9 label (dx, dy, ha, va), cap width
SERIALIZED = [
    ("gguf-q8_0", None, (0, 9, "center", "bottom"),
     (-8, 0, "right", "center"), 6.0),
    ("gguf-q6_k", None, (9, 0, "left", "center"), (-8, 0, "right", "center"),
     6.0),
    ("gguf-q5_k_xl", None, (0, 9, "center", "bottom"),
     (0, 10, "center", "bottom"), 6.0),
    ("hyd", "hydrated-build-receipt.json", (10, 6, "left", "bottom"),
     (8, 0, "left", "center"), 9.0),
    ("ctx", "context-build-receipt.json", (9, 0, "left", "center"),
     (8, 0, "left", "center"), 6.0),
]
SERIALIZED_TABLE = (21.9, 0.85)


# Our five vLLM builds also have full-suite means over ten shards. The left
# column plots shard 0 only, because that is the shard every GGUF candidate was
# scored on; this table exists so the footer can state how far the full suite
# moves those five numbers instead of asserting it.
# id -> full-suite mean (receipts/kld5-10M-*.json)
FULL_SUITE = {
    "hyd": 0.002760, "k5k6": 0.003210, "ctx": 0.003509,
    "fp8": 0.005294, "k4": 0.010604,
}
FULL_SUITE_POSITIONS = 10_480_640

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
                  fp8="#b08800", gguf="#bf3989", exl3="#0b5d1e", udq="#0969da",
                  iq="#8250df", nvfp4="#a40e26", tail="#57606a", floor="#8a8a8a",
                  ann="#1a1a1a"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e",
                 hyd="#56d364", k6="#58a6ff", ctx="#bc8cff", k4="#f85149",
                 fp8="#d29922", gguf="#f778ba", exl3="#56d364", udq="#58a6ff",
                 iq="#bc8cff", nvfp4="#f85149", tail="#8b949e", floor="#8b949e",
                 ann="#e6edf3"),
}

# One shared y-range for both left sub-panels: it has to reach from below the
# cross-engine floor, which needs a clear lane for its own label, up to K4's
# p99.9 at the top.
LEFT_YLIM = (0.00030, 0.95)
LEFT_YTICKS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 0.5]


def decoder_weight_gib(manifest: Path) -> tuple[int, float]:
    """Bytes and GiB of decoder linear weight on his axis, from a manifest."""
    roles = json.loads(manifest.read_text())["roles"]
    total = sum(roles[r]["bytes"] for r in HIS_AXIS_ROLES)
    return total, total / 1024 ** 3


def serialized_gib(cid: str, receipt: str | None) -> tuple[int, float]:
    """Serialized bytes and GiB for one candidate, from its own receipt."""
    if receipt is None:
        raw = CAND[cid]["serialized_bytes"]
    else:
        raw = json.loads((RECEIPTS / receipt).read_text())["immutable_payload_bytes"]
    return raw, raw / 1024 ** 3


def style_axes(ax, c: dict) -> None:
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.45, zorder=0)


def draw_point(ax, c: dict, cid: str, x: float, cap: float,
               mean_off, tail_off) -> None:
    """One candidate: mean with its interval, its p99.9 tail, and its name."""
    _name, short, key, engine = IDENTITY[cid]
    d = CAND[cid]
    mean, (lo, hi), p999 = d["mean_kld"], d["ci95"], d["p999_kld"]
    dx, dy, ha, va = mean_off
    tdx, tdy, tha, tva = tail_off
    # The vertical spine shows how far this candidate's tail runs past its mean.
    ax.plot([x, x], [mean, p999], color=c[key], lw=1.0, alpha=0.38, zorder=2)
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=c[key], elinewidth=2.0, capsize=cap, capthick=1.9,
                zorder=3)
    ax.scatter([x], [mean], s=70, marker=MARKER[engine], color=c[key], zorder=4,
               edgecolor=c["bg"], linewidth=0.9)
    ax.scatter([x], [p999], s=150, marker="v", facecolor=c["bg"],
               edgecolor=c[key], linewidth=1.5, zorder=4)
    ax.annotate(short, (x, mean), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=8.5, color=c[key], zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=c["bg"],
                          edgecolor="none", alpha=0.85))
    ax.annotate(f"p99.9 {p999:.4f}", (x, p999), textcoords="offset points",
                xytext=(tdx, tdy), ha=tha, va=tva, fontsize=7.5, color=c[key],
                zorder=5)


def draw_table(ax, c: dict, anchor: tuple[float, float], rows: list) -> None:
    """A monospaced value table inside a panel: every plotted number in text.

    rows is a list of (colour key or None, text). None is a header or note.
    """
    x, y = anchor
    for i, (key, text) in enumerate(rows):
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(0, -i * 10.2), ha="left", va="top", fontsize=7.0,
                    family="DejaVu Sans Mono",
                    color=c[key] if key else c["muted"], zorder=6,
                    bbox=dict(boxstyle="square,pad=0.22", facecolor=c["bg"],
                              edgecolor="none", alpha=0.92))


def draw_resident(ax, c: dict) -> None:
    """Upper left sub-panel: x = weights measured resident under vLLM."""
    style_axes(ax, c)
    rows: list[tuple[str | None, str]] = [
        (None, f"{'resident weights':<23}{'mean':<10}"
         f"{'p99.9':<8}{'top-1':<9}{'GiB':<7}")
    ]
    for cid, gib, mean_off, tail_off, cap in RESIDENT:
        name, _short, key, _engine = IDENTITY[cid]
        d = CAND[cid]
        draw_point(ax, c, cid, gib, cap, mean_off, tail_off)
        top1 = f"{d['top1_agreement'] * 100:.2f} %"
        rows.append((key, f"{name:<23}{d['mean_kld']:<10.6f}"
                          f"{d['p999_kld']:<8.4f}{top1:<9}{gib:<7.2f}"))
    rows.append((None, "circle = captured under vLLM, carries no engine term"))
    draw_table(ax, c, RESIDENT_TABLE, rows)

    ax.set_yscale("log")
    ax.set_xlim(17.1, 29.9)
    ax.set_ylim(*LEFT_YLIM)
    ax.set_xticks([18, 20, 22, 24, 26, 28])
    ax.set_yticks(LEFT_YTICKS)
    ax.set_yticklabels([f"{v:g}" for v in LEFT_YTICKS])
    ax.set_xlabel("weights measured resident under vLLM (GiB) — resident "
                  "weights, not serialized disk bytes", color=c["fg"],
                  fontsize=9.5)
    ax.set_ylabel("KL divergence on shard 0, nats/token\n(log, lower is better)",
                  color=c["fg"], fontsize=9.5)
    ax.set_title(
        "One protocol for every family: shard 0 of our v5 held-out suite\n"
        f"same {CONTEXTS} contexts, same {POSITIONS:,} scored positions, "
        f"{CLUSTERS} source clusters, one shared BF16 head for both operands\n"
        "upper: x = weights measured resident under vLLM (no GGUF point: "
        "llama.cpp resident weights never measured)",
        color=c["fg"], fontsize=9.4, pad=9)


def draw_serialized(ax, c: dict) -> None:
    """Lower left sub-panel: x = serialized bytes, same y as the panel above."""
    style_axes(ax, c)
    rows: list[tuple[str | None, str]] = [
        (None, f"{'serialized bytes':<18}{'mean':<10}"
         f"{'p99.9':<8}{'top-1':<9}{'GiB':<7}")
    ]
    for cid, receipt, mean_off, tail_off, cap in SERIALIZED:
        name, _short, key, _engine = IDENTITY[cid]
        raw, gib = serialized_gib(cid, receipt)
        d = CAND[cid]
        mean = d["mean_kld"]
        print(f"  {cid}: {raw:,} B serialized = {gib:.4f} GiB, "
              f"mean {mean:.6f}")
        draw_point(ax, c, cid, gib, cap, mean_off, tail_off)
        top1 = f"{d['top1_agreement'] * 100:.2f} %"
        rows.append((key, f"{name:<18}{mean:<10.6f}"
                          f"{d['p999_kld']:<8.4f}{top1:<9}{gib:<7.3f}"))
    rows.append((None, "square = llama.cpp; circle = vLLM"))
    rows.append((None, "cross-engine control is diagnostic, not subtractable"))
    draw_table(ax, c, SERIALIZED_TABLE, rows)

    # Cross-engine BF16 control: the two implementations disagree even on the
    # same unquantized weights. It demonstrates confounding but cannot be
    # algebraically removed from a candidate KL.
    fmean, fp999 = FLOOR["mean_kld"], FLOOR["p999_kld"]
    ax.axhline(fmean, color=c["floor"], lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.annotate(
        f"cross-engine BF16 control: llama.cpp ↔ vLLM on the SAME weights, "
        f"mean {fmean:.6f}; diagnostic only",
        (17.88, fmean), textcoords="offset points", xytext=(0, -11),
        va="top", fontsize=7.6, color=c["floor"], zorder=5)
    ax.axhline(fp999, color=c["floor"], lw=1.1, ls=(0, (2, 3)), zorder=1)
    ax.annotate(f"same control, p99.9 {fp999:.4f}", (17.88, fp999),
                textcoords="offset points", xytext=(0, 5), va="bottom",
                fontsize=7.6, color=c["floor"], zorder=5)


    ax.set_yscale("log")
    ax.set_xlim(17.8, 27.9)
    ax.set_ylim(*LEFT_YLIM)
    ax.set_xticks([18, 20, 22, 24, 26])
    ax.set_yticks(LEFT_YTICKS)
    ax.set_yticklabels([f"{v:g}" for v in LEFT_YTICKS])
    ax.set_xlabel("serialized bytes on disk (GiB) — GGUF files as published, "
                  "our immutable payload as published; never VRAM",
                  color=c["fg"], fontsize=9.5)
    ax.set_ylabel("KL divergence on shard 0, nats/token\n(log, lower is better)",
                  color=c["fg"], fontsize=9.5)
    ax.set_title(
        "lower: same suite, same y-axis, x = serialized bytes on disk\n"
        "online K5/K6 (BF16 attention, quantized at load) and K4 (no payload "
        "receipt) are absent rather than estimated",
        color=c["fg"], fontsize=9.4, pad=9)


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
        "A different protocol: turboderp's published labels, his own BF16 "
        "reference\n"
        f"OpenWebText, 8 x 8192 = {HIS_POSITIONS:,} formatted positions; "
        "filled = mean, hollow = median\n"
        "nothing here is comparable to the left column",
        color=c["fg"], fontsize=9.8, pad=9)

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
    fig = plt.figure(figsize=(16.8, 12.2), facecolor=c["bg"])
    # tight_layout cannot handle the right panel spanning both rows, so the
    # margins are set once, here, and the footer owns everything below them.
    gs = fig.add_gridspec(2, 2, left=0.070, right=0.995, top=0.888,
                          bottom=0.158, hspace=0.40, wspace=0.145)
    ax_res = fig.add_subplot(gs[0, 0])
    ax_ser = fig.add_subplot(gs[1, 0], sharey=ax_res)
    ax_his = fig.add_subplot(gs[:, 1])
    print(f"{theme}:")
    draw_resident(ax_res, c)
    draw_serialized(ax_ser, c)
    draw_his(ax_his, c)

    # The footer states this as a difference FROM the shard-0 values, so the
    # shard-0 value is the denominator: the same gap is 2.8 % of a full-run
    # mean and 2.9 % of a shard-0 mean, and the larger number is the safe one.
    drift = max(abs(FULL_SUITE[k] - CAND[k]["mean_kld"]) / CAND[k]["mean_kld"]
                for k in FULL_SUITE) * 100
    fmean = FLOOR["mean_kld"]

    fig.suptitle("Quantization families: every family we have measured, one "
                 "protocol, two size axes (left) — and one published protocol "
                 "we have never run (right)",
                 color=c["fg"], fontsize=12.5, y=0.986)
    footer = [
        f"Left column, ONE protocol: shard 0 of the v5 held-out suite — the "
        f"same {CONTEXTS} contexts, the same {POSITIONS:,} scored positions and "
        f"the same {CLUSTERS} source clusters for every candidate, "
        "KL(BF16 reference || candidate) over the full 248,320-entry vocabulary, "
        "both operands through one shared BF16 head.",
        "Receipt: receipts/cross-engine-comparator.json. Shard 0 is one tenth of "
        f"the suite: our five vLLM means over {FULL_SUITE_POSITIONS:,} positions "
        f"differ from these shard-0 values by at most {drift:.1f} % and change "
        "no ordering.",
        "Cross-engine control, measured on identical unquantized BF16 weights: "
        f"llama.cpp versus vLLM is {fmean:.6f} mean, "
        f"{FLOOR['top1_agreement'] * 100:.2f} % top-1, p99.9 "
        f"{FLOOR['p999_kld']:.4f} (receipts/gguf-report-engine-floor.json). "
        "It proves the engine is a confounder. KL is neither additive nor a "
        "metric, so the control is not subtracted and supplies no quantization-only bound.",
        "The two left sub-panels share a y-axis and deliberately do NOT share an "
        "x-axis: resident weights and serialized bytes are different quantities, "
        "and merging them would be a false axis.",
        "Our serialized bytes carry a BF16 embedding, the BF16 vision tower and "
        "the quantized MTP draft; the GGUF bytes are the published single-file "
        "GGUF. Serialized bytes are never VRAM, resident weights or KV.",
        f"Right panel, a different protocol: turboderp's OpenWebText run, "
        f"8 x 8192 = {HIS_POSITIONS:,} formatted positions, his own BF16 "
        "reference, values read off his published images rather than a data "
        "file. No ratio between the columns is meaningful — different corpus, "
        "window, reference numerics, vocabulary handling and head placement.",
    ]
    for i, line in enumerate(footer):
        fig.text(0.005, 0.108 - i * 0.0152, line, color=c["muted"],
                 fontsize=7.4)

    written = []
    for ext in ("svg", "png"):
        p = OUT / f"kld-family-comparison-{theme}.{ext}"
        fig.savefig(p, facecolor=c["bg"], dpi=150 if ext == "png" else None)
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
