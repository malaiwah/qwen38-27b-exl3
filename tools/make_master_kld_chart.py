#!/usr/bin/env python3
"""Every KL divergence we have measured, in one figure, four separate panels.

One image, four protocols, and NOT one axis between them. Each panel states its
own protocol and its own scored-position count in its title, because the only
dishonest thing this figure could do is let a reader slide a ruler from one
panel to another.

  A  v5 held-out suite, shard 0. The one panel where every family we have
     measured appears together: same 512 contexts, same 1,048,064 scored
     positions, KL(BF16 reference || candidate) over the full 248,320-entry
     vocabulary with both operands through one shared BF16 head. Mean with its
     source-cluster bootstrap interval AND its p99.9, on a log y-axis.
     x = serialized bytes on disk, the only size quantity that exists for both
     engines (see SIZE_MEASURE below). Candidates with no published serialized
     size get a separate lane at the right of the same panel rather than an
     invented x. GGUF points are captured under llama.cpp and therefore carry
     the measured cross-engine floor, which is drawn as a labelled reference
     line; vLLM points do not carry it.
  B  The same suite's ladder checkpoints, 1M/2M/5M/10M, for the five vLLM
     builds that ran all ten shards: how far the point estimate moves and what
     the interval does when the scored positions grow tenfold.
  C  The two prior suites, corrected v3 and source-disjoint v4. They get one
     y-axis EACH, deliberately different, with a hatched barrier between them,
     because their absolute values are not comparable to each other or to
     anything else in this figure.
  D  turboderp's published chart labels for his own protocol, read off his
     rasters. Our builds appear there only as vertical decoder-weight markers
     with no y-value, because we have never run his protocol.

Every one of OUR numbers is read from receipts/ at runtime and nothing about
them is transcribed into this file, so the figure cannot drift from the
evidence; the script prints each receipt it opened. Only placement (label
offsets, panel order, colours) lives here. Panel D's y-values are the sole
hard-coded numbers, because they are somebody else's printed labels rather
than a receipt of ours, and they are marked as such in the panel.

Emits light and dark SVG + PNG into assets/.
"""
from __future__ import annotations

import json
import textwrap
from math import log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = ROOT / "receipts"
OUT = ROOT / "assets"
# Footer wrap width, in characters: at 7.6 pt on a 19.2 in canvas this keeps
# every line inside the figure instead of running off its right edge.
FOOTER_WIDTH = 285

OPENED: list[str] = []


def receipt(name: str) -> dict:
    """Read one receipt and record that we read it."""
    p = RECEIPTS / name
    if name not in OPENED:
        OPENED.append(name)
    return json.loads(p.read_text())


def optional(name: str) -> dict | None:
    """A receipt that may not exist yet (NVFP4 on v5 is still being measured)."""
    return receipt(name) if (RECEIPTS / name).exists() else None


GIB = 1024 ** 3

# ---------------------------------------------------------------- panel A data
COMPARATOR = receipt("cross-engine-comparator.json")
CAND = dict(COMPARATOR["candidates"])
SUITE = COMPARATOR["suite"]
FLOOR = COMPARATOR["engine_floor"]
FLOOR_REPORT = receipt("gguf-report-engine-floor.json")

# NVFP4 on the v5 suite is being measured while this is written. Two ways it
# can arrive: as a row inside the comparator's candidate map, which the line
# above picks up with no code at all, or as a standalone shard-0 receipt in the
# ladder-cumulative shape. The second case is merged here, and only if that
# receipt scored the identical positions panel A is built on -- a run over a
# different position set would be a different protocol and does not belong in
# this panel.
NVFP4_SHARD = None if "nvfp4" in CAND else optional("kld5-1M-nvfp4.json")
NVFP4_NOTE = ("present in the comparator candidate map" if "nvfp4" in CAND
              else "no NVFP4 receipt on the v5 shard yet")
if NVFP4_SHARD is not None:
    if NVFP4_SHARD["scored_positions"] != SUITE["scored_positions"]:
        NVFP4_NOTE = (f"receipts/kld5-1M-nvfp4.json scored "
                      f"{NVFP4_SHARD['scored_positions']:,} positions, not the "
                      f"{SUITE['scored_positions']:,} of panel A: not plotted")
        NVFP4_SHARD = None
    else:
        boot = NVFP4_SHARD["context_bootstrap"]
        tail = NVFP4_SHARD.get("tail") or {}
        p999 = tail.get("quantiles", {}).get("p999", {}).get("estimate")
        CAND["nvfp4"] = {
            "family": "NVFP4 (unsloth)",
            "engine": "vLLM (pinned r34)",
            "mean_kld": NVFP4_SHARD["token_mean_kld"],
            "ci95": [boot["ci95_low"], boot["ci95_high"]],
            "top1_agreement": NVFP4_SHARD["top1_agreement"],
            "p999_kld": p999,
            "carries_cross_engine_term": False,
            "report": "receipts/kld5-1M-nvfp4.json",
        }
        NVFP4_NOTE = "merged from receipts/kld5-1M-nvfp4.json"

# The size measure for panel A, stated once and never mixed with another:
#
#   SERIALIZED BYTES ON DISK -- the bytes of the artifact you download.
#
# It is the only size quantity that exists on both sides of the engine split.
# Resident weights exist only for our vLLM builds (we never measured llama.cpp
# resident weights), and decoder-linear weight exists only where a per-tensor
# manifest was published, so either of those would silently drop one engine.
# Serialized bytes are never VRAM, never resident weights and never KV.
#
# GGUF x comes from the pinned single-file GGUF in the comparator receipt; ours
# comes from the immutable payload of the published build receipt -- the same
# kind of quantity, bytes that exist on disk. A candidate whose serialized size
# is not published, or whose serialized form is not its operating point, is NOT
# given an estimated x: it goes in the lane at the right of the same panel.
SIZE_MEASURE = "serialized bytes on disk (published artifact payload)"

# candidate id -> build receipt carrying our published payload bytes.
OUR_PAYLOAD = {
    "hyd": "hydrated-build-receipt.json",
    "ctx": "context-build-receipt.json",
}
# Why the rest have no x on this axis. Printed in the lane, not hidden.
NO_SIZE_REASON = {
    "k5k6": "serialized bytes are not its operating point",
    "fp8": "no published serialized-byte receipt of ours",
    "k4": "no published payload receipt",
    "nvfp4": "no published serialized-byte receipt of ours",
    "nvfp4-rtx5090": "no published serialized-byte receipt of ours",
}

# id -> (display name, short label, colour key). Engine, floor-carrying and
# every number come from the receipt, so a candidate added to the comparator
# later (NVFP4) needs nothing here but a name and a colour.
IDENTITY = {
    "gguf-q8_0": ("GGUF Q8_0", "Q8_0", "gguf"),
    "gguf-q6_k": ("GGUF Q6_K", "Q6_K", "gguf"),
    "gguf-q5_k_xl": ("GGUF UD-Q5_K_XL", "UD-Q5_K_XL", "gguf"),
    "hyd": ("hydrated K5/K6", "hydrated", "hyd"),
    "k5k6": ("online K5/K6", "online", "k6"),
    "ctx": ("context edition", "context", "ctx"),
    "fp8": ("official Qwen FP8", "official FP8", "fp8"),
    "k4": ("K4", "K4", "k4"),
    "nvfp4": ("Unsloth NVFP4", "NVFP4", "nvfp4"),
    "nvfp4-rtx5090": ("gittensor NVFP4 (RTX5090)", "NVFP4-5090", "nvfp4"),
}
# Label placement only: (dx, dy, ha, va) for the mean label and for the p99.9
# label, plus the interval cap width. No x or y value is ever nudged.
PLACE = {
    "gguf-q8_0": ((0, 10, "center", "bottom"), (-8, 0, "right", "center"), 6.0),
    "gguf-q6_k": ((9, -2, "left", "center"), (8, 0, "left", "center"), 6.0),
    "gguf-q5_k_xl": ((6, 8, "left", "bottom"), (0, 11, "center", "bottom"), 6.0),
    "hyd": ((0, -12, "center", "top"), (9, 0, "left", "center"), 9.0),
    "ctx": ((-10, -7, "right", "top"), (-9, 0, "right", "center"), 6.0),
    "k5k6": ((10, 0, "left", "center"), (0, 10, "center", "bottom"), 6.0),
    "fp8": ((10, 0, "left", "center"), (0, 10, "center", "bottom"), 6.0),
    "k4": ((10, 0, "left", "center"), (0, 10, "center", "bottom"), 6.0),
    "nvfp4": ((10, 0, "left", "center"), (0, 10, "center", "bottom"), 6.0),
    "nvfp4-rtx5090": ((10, 0, "left", "center"), (0, 10, "center", "bottom"), 6.0),
}
PLACE_DEFAULT = ((0, 10, "center", "bottom"), (0, 10, "center", "bottom"), 6.0)

# ---------------------------------------------------------------- panel B data
LADDER = receipt("kld5-ladder-convergence.json")
FULL = {c: receipt(f"kld5-10M-{c}.json") for c in ("hyd", "k5k6", "ctx", "fp8", "k4")}
CHECKPOINTS = ("1M", "2M", "5M", "10M")

# ---------------------------------------------------------------- panel C data
V3 = receipt("analysis-v3-contamination-corrected.json")
V4 = receipt("qualification-v4-contamination-corrected.json")
# v4's corrected block publishes surviving contexts but not the position count;
# its own qualification report gives positions per context exactly, so the
# count is derived from two receipts instead of being typed in here.
V4_QUAL = receipt("report-v4-hyd-qual.json")
V4_PER_CONTEXT = V4_QUAL["scored_positions"] // V4_QUAL["contexts"]
V3_POSITIONS = V3["as_served_metrics"]["scored_positions"]
V4_POSITIONS = V4["qualification_contexts_after"] * V4_PER_CONTEXT
# metric key -> (short label, colour key). Order is the plotting order.
PRIOR_ROWS = [
    ("hydrated", "hydrated", "hyd"),
    ("k5k6", "online", "k6"),
    ("context", "context", "ctx"),
    ("official_fp8", "FP8", "fp8"),
    ("k4", "K4", "k4"),
    ("nvfp4", "NVFP4", "nvfp4"),
]

# The measured answer to "why can't I compare a v3 number to a v5 one?", if it
# has been published: the same candidate's v3 mean over its v5 shard-0 mean,
# for every candidate. A band of factors rather than one factor is exactly what
# makes a cross-suite conversion impossible, so C1 prints it instead of only
# asserting the rule.
NVFP4_MEASUREMENT = optional("nvfp4-v5-measurement.json")
RATIOS = (NVFP4_MEASUREMENT or {}).get("suite_comparability_v3_vs_v5")

# NVFP4's own ten-shard run, if it has landed. It has no ladder checkpoints of
# its own, so it is drawn as a single 10M point and labelled as one rather than
# joined into a line that was never measured.
NVFP4_FULL = optional("kld5-10M-nvfp4.json")

# ---------------------------------------------------------------- panel E data
WINDOW = receipt("scored-window-offset.json")

# ---------------------------------------------------------------- panel D data
# turboderp/Qwen3.8-27B-exl3's published chart labels: HIS printed numbers,
# read off his rasters, not a published data file, documented in
# docs/35-external-protocol-comparability.md. The only hard-coded values in
# this script, and never mixed with ours.
# key -> (display name, marker, [(x GiB, mean, label, (dx, dy, ha))])
HIS = [
    ("exl3", "EXL3 (bpw)", "o", [
        (6.6, 0.351, "2.00", (0, 10, "center")),
        (8.1, 0.299, "2.50", (0, 10, "center")),
        (9.5, 0.112, "3.00", (0, 10, "center")),
        (12.4, 0.052, "4.00", (0, 10, "center")),
        (15.0, 0.014, "5.00", (0, -15, "center")),
        (17.8, 0.007, "6.00", (-7, -6, "right")),
    ]),
    ("udq", "GGUF UD", "s", [
        (9.6, 0.237, "Q2_K_XL", (0, 10, "center")),
        (12.0, 0.089, "Q3_K_XL", (0, 10, "center")),
        (16.0, 0.039, "Q4_K_XL", (0, -15, "center")),
        (18.0, 0.019, "Q5_K_XL", (7, 4, "left")),
        (23.1, 0.009, "Q6_K_XL", (0, 10, "center")),
    ]),
    ("iq", "GGUF-IQ", "D", [(13.9, 0.055, "IQ4_XS", (0, 11, "center"))]),
    ("nvfp4", "NVFP4 (Unsloth)", "^", [(17.9, 0.041, "NVFP4", (0, 11, "center"))]),
    ("fp8", "FP8 (Qwen)", "v", [(25.1, 0.023, "FP8", (0, 11, "center"))]),
]
HIS_FLOOR_MEAN = 0.0052     # his synthetic noise floor, mean
HIS_FLOOR_MEDIAN = 0.0007   # his synthetic noise floor, median
HIS_POSITIONS = 8 * 8192    # 65,536 formatted OpenWebText positions
# His x convention: decoder linear storage, embeddings excluded, output head
# included. Vision, MTP draft and norms are not decoder weight.
HIS_AXIS_ROLES = ("full_attention", "linear_attention", "mlp_gate_proj",
                  "mlp_up_proj", "mlp_down_proj", "lm_head")
# Our builds with an authoritative per-tensor manifest, so their placement on
# his x-axis is exact. They are annotations with NO y-value, in neutral
# annotation colours: a family colour would invite attaching them to a curve.
OUR_ON_HIS_AXIS = [
    ("hydrated", "hydrated-quantization-manifest.json", "ann", (0, (7, 3))),
    ("context edition", "context-quantization-manifest.json", "tail", (0, (2, 2))),
]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d0d0d0", muted="#8a8a8a",
                  hyd="#0b5d1e", k6="#0969da", ctx="#8250df", k4="#a40e26",
                  fp8="#b08800", gguf="#bf3989", exl3="#0b5d1e", udq="#0969da",
                  iq="#8250df", nvfp4="#a40e26", tail="#57606a", floor="#8a8a8a",
                  ann="#1a1a1a", warn="#a40e26"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", muted="#8b949e",
                 hyd="#56d364", k6="#58a6ff", ctx="#bc8cff", k4="#f85149",
                 fp8="#d29922", gguf="#f778ba", exl3="#56d364", udq="#58a6ff",
                 iq="#bc8cff", nvfp4="#f85149", tail="#8b949e", floor="#8b949e",
                 ann="#e6edf3", warn="#f85149"),
}


# ------------------------------------------------------------------- utilities
def style_axes(ax, c: dict) -> None:
    ax.set_facecolor(c["bg"])
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["fg"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.45, zorder=0)


def panel_title(ax, c: dict, text: str) -> None:
    ax.set_title(text, color=c["fg"], fontsize=9.4, pad=9)


def draw_table(ax, c: dict, anchor: tuple[float, float], rows: list,
               fontsize: float = 7.0, step: float = 10.2) -> None:
    """A monospaced value table in axes coordinates: plotted numbers as text.

    rows is a list of (colour key or None, text). None is a header or note.
    """
    x, y = anchor
    for i, (key, text) in enumerate(rows):
        ax.annotate(text, (x, y), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, -i * step),
                    ha="left", va="top", fontsize=fontsize,
                    family="DejaVu Sans Mono",
                    color=c[key] if key else c["muted"], zorder=6,
                    bbox=dict(boxstyle="square,pad=0.22", facecolor=c["bg"],
                              edgecolor="none", alpha=0.92))


def identity(cid: str) -> tuple[str, str, str]:
    return IDENTITY.get(cid, (cid, cid, "ann"))


def serialized_gib(cid: str) -> float | None:
    """Serialized GiB for one candidate, or None if none is published."""
    if cid in OUR_PAYLOAD:
        return receipt(OUR_PAYLOAD[cid])["immutable_payload_bytes"] / GIB
    raw = CAND[cid].get("serialized_bytes")
    return raw / GIB if raw else None


def decoder_weight_gib(manifest: str) -> float:
    roles = json.loads((RECEIPTS / manifest).read_text())["roles"]
    if manifest not in OPENED:
        OPENED.append(manifest)
    return sum(roles[r]["bytes"] for r in HIS_AXIS_ROLES) / GIB


def sized() -> list[tuple[str, float]]:
    """Comparator candidates that have a published serialized size, by size."""
    out = [(cid, serialized_gib(cid)) for cid in CAND]
    return sorted([(cid, g) for cid, g in out if g is not None], key=lambda r: r[1])


def unsized() -> list[str]:
    """Comparator candidates with no published serialized size, by mean KLD."""
    out = [cid for cid in CAND if serialized_gib(cid) is None]
    return sorted(out, key=lambda cid: CAND[cid]["mean_kld"])


def panel_a_ylim() -> tuple[float, float]:
    """The data range of panel A, with room below for the floor's own label."""
    vals = [FLOOR["mean_kld"]]
    for d in CAND.values():
        vals += [d["mean_kld"], d["ci95"][0], d.get("p999_kld") or d["mean_kld"]]
        if "engine_floor_subtracted_estimate" in d:
            vals.append(d["engine_floor_subtracted_estimate"])
    return min(vals) * 0.55, max(vals) * 1.15


def reserve_top(ax, points: float, above: float | None = None) -> None:
    """Grow a log y-axis so `points` of blank space sit above `above`.

    A value table is a fixed number of points tall, not a fixed fraction of the
    data range, so multiplying the top value by a constant either wastes half
    the panel or lets the table land on a curve as soon as the range changes --
    which is exactly what a new candidate does. This converts the table's height
    into the decades it occupies on this particular axis, and counts whatever
    room is already free above the highest thing this panel draws, so a tall
    shared axis is not stretched twice.
    """
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72.0
    if points <= 0 or height_pt <= points:
        return
    lo, hi = ax.get_ylim()
    decades = log10(hi / lo)
    if above is not None and above < hi:
        points -= height_pt * log10(hi / above) / decades
        if points <= 0:
            return
    ax.set_ylim(lo, hi * 10 ** (decades * points / (height_pt - points)))


def draw_candidate(ax, c: dict, cid: str, x: float, name_label: bool = True,
                   tail_label: bool = True) -> None:
    """Mean with its bootstrap interval, its p99.9, and the spine between them.

    Labels are off where the panel already names the candidate on its x-axis
    and prints every number in its own value table: once is enough, and in a
    narrow lane a second copy is what collides.
    """
    _name, short, key = identity(cid)
    d = CAND[cid]
    mean, (lo, hi), p999 = d["mean_kld"], d["ci95"], d.get("p999_kld")
    carries = d.get("carries_cross_engine_term", False)
    marker = "s" if carries else "o"
    (dx, dy, ha, va), (tdx, tdy, tha, tva), cap = PLACE.get(cid, PLACE_DEFAULT)

    if p999:
        ax.plot([x, x], [mean, p999], color=c[key], lw=1.0, alpha=0.38, zorder=2)
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=c[key], elinewidth=2.0, capsize=cap, capthick=1.9,
                zorder=3)
    ax.scatter([x], [mean], s=72, marker=marker, color=c[key], zorder=4,
               edgecolor=c["bg"], linewidth=0.9)
    if p999:
        ax.scatter([x], [p999], s=150, marker="v", facecolor=c["bg"],
                   edgecolor=c[key], linewidth=1.5, zorder=4)
    if carries:
        # The naive floor-subtracted estimate: an estimate, not an identity.
        net = d["engine_floor_subtracted_estimate"]
        ax.plot([x, x], [net, mean], color=c[key], lw=1.0, ls=(0, (1, 2)),
                alpha=0.75, zorder=2)
        ax.scatter([x], [net], s=72, marker="s", facecolor=c["bg"],
                   edgecolor=c[key], linewidth=1.5, zorder=4)
    if name_label:
        ax.annotate(short, (x, mean), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=8.5, color=c[key], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=c["bg"],
                              edgecolor="none", alpha=0.85))
    if p999 and tail_label:
        ax.annotate(f"p99.9 {p999:.4f}", (x, p999), textcoords="offset points",
                    xytext=(tdx, tdy), ha=tha, va=tva, fontsize=7.5,
                    color=c[key], zorder=5)
    elif not p999 and tail_label:
        # A candidate whose run carries no tail histogram gets no invented tail.
        ax.annotate("p99.9 not published", (x, mean), textcoords="offset points",
                    xytext=(tdx, tdy + 12), ha=tha, va=tva, fontsize=7.0,
                    color=c["muted"], zorder=5)


def draw_floor(ax, c: dict, label: bool) -> None:
    lo, hi = FLOOR["ci95"]
    ax.axhspan(lo, hi, color=c["floor"], alpha=0.16, zorder=1)
    ax.axhline(FLOOR["mean_kld"], color=c["floor"], lw=1.3, ls=(0, (5, 3)),
               zorder=2)
    if label:
        ax.annotate(
            f"cross-engine floor {FLOOR['mean_kld']:.6f}, "
            f"{FLOOR['top1_agreement'] * 100:.2f} % top-1, "
            f"{FLOOR_REPORT['scored_positions']:,} positions\n"
            "llama.cpp vs vLLM on the SAME unquantized BF16 weights — "
            "squares contain this term, circles do not",
            (0.012, FLOOR["mean_kld"]), xycoords=("axes fraction", "data"),
            textcoords="offset points", xytext=(0, 5), ha="left", va="bottom",
            fontsize=7.0, color=c["floor"], zorder=5)


def draw_size_panel(ax, c: dict) -> None:
    style_axes(ax, c)
    rows = sized()
    for cid, gib in rows:
        draw_candidate(ax, c, cid, gib, tail_label=False)
    draw_floor(ax, c, label=True)

    xs = [g for _, g in rows]
    ax.set_xlim(min(xs) - 1.15, max(xs) + 1.35)
    ax.set_yscale("log")
    ax.set_ylim(*panel_a_ylim())
    # The table carries every candidate of this run, including the ones drawn
    # in the lane, so no number has to be crammed next to a marker. Only the
    # room this panel's own drawing does not already leave has to be added:
    # the shared y-axis is set by the tallest tail in the lane, which is often
    # well above anything plotted here.
    drawn_max = max(max(CAND[cid]["p999_kld"] or CAND[cid]["mean_kld"],
                        CAND[cid]["ci95"][1]) for cid, _g in rows)
    reserve_top(ax, 9.4 * (len(CAND) + 3) + 34, above=drawn_max)
    ax.set_xlabel(f"{SIZE_MEASURE}, GiB  —  never VRAM, resident weights or KV",
                  color=c["fg"], fontsize=9)
    ax.set_ylabel("KL(BF16 reference || candidate), nats/token (log)",
                  color=c["fg"], fontsize=9.5)

    table = [(None, f"{'candidate':<15}{'GiB':>6}  {'mean':>8}  "
                    f"{'95 % interval':>19}  {'p99.9':>7}  {'top-1':>7}")]
    for cid, gib in rows + [(cid, None) for cid in unsized()]:
        name, _short, key = identity(cid)
        d = CAND[cid]
        lo, hi = d["ci95"]
        tail = d.get("p999_kld")
        table.append((key, f"{name:<15}"
                           f"{(f'{gib:6.2f}' if gib else '     —'):>6}  "
                           f"{d['mean_kld']:8.6f}  [{lo:.6f}, {hi:.6f}]  "
                           f"{(f'{tail:.4f}' if tail else 'n/a'):>7}  "
                           f"{d['top1_agreement'] * 100:6.2f} %"))
    table.append((None, "— in the GiB column: no published serialized size, so the "
                        "candidate is in the lane at the right, not given an x"))
    table.append((None, "hollow square = measured value minus the engine floor; "
                        "KL is not additive, so it is an estimate, not an identity"))
    draw_table(ax, c, (0.012, 0.985), table, fontsize=6.6, step=9.4)

    handles = [
        Line2D([], [], marker="o", ls="none", color=c["fg"], markersize=7,
               label="vLLM (pinned r34) — no engine term"),
        Line2D([], [], marker="s", ls="none", color=c["fg"], markersize=7,
               label="llama.cpp — carries the cross-engine floor"),
        Line2D([], [], marker="s", ls="none", markerfacecolor=c["bg"],
               markeredgecolor=c["fg"], markersize=7,
               label="naive floor-subtracted estimate"),
        Line2D([], [], marker="v", ls="none", markerfacecolor=c["bg"],
               markeredgecolor=c["fg"], markersize=8, label="p99.9 of the same run"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7.6,
                    framealpha=0.93, labelspacing=0.45)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])

    panel_title(ax, c, "A — v5 held-out suite, shard 0: one protocol, every family "
                       f"we have measured\n{SUITE['contexts']} contexts · "
                       f"{SUITE['scored_positions']:,} scored positions · "
                       f"{SUITE['source_clusters']} source clusters — identical for "
                       "every candidate")


def draw_lane_panel(ax, c: dict) -> None:
    """Same panel, same protocol, same y-axis: candidates with no published size."""
    style_axes(ax, c)
    ids = unsized()
    for i, cid in enumerate(ids):
        draw_candidate(ax, c, cid, i, name_label=False, tail_label=False)
        ax.annotate(NO_SIZE_REASON.get(cid, "no published serialized size"),
                    (i, 0.015), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(-10, 0),
                    rotation=90, ha="center", va="bottom", fontsize=6.1,
                    color=c["muted"], zorder=5)
    draw_floor(ax, c, label=False)
    ax.set_xlim(-0.7, len(ids) - 0.3)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([identity(cid)[1] for cid in ids], fontsize=7.6,
                       rotation=30, ha="right")
    ax.set_xlabel("no comparable x", color=c["muted"], fontsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_visible(False)
    panel_title(ax, c, "same run,\nsame y-axis\nno published size")


def draw_ladder_panel(ax, c: dict) -> None:
    style_axes(ax, c)
    rows = LADDER["rows"]
    cands = []
    for r in rows:
        if r["candidate"] not in cands:
            cands.append(r["candidate"])
    drift = 0.0
    ends = []  # (x, y, text, colour key, minimum vertical room in points)
    for cid in cands:
        _name, short, key = identity(cid)
        pts = sorted((r for r in rows if r["candidate"] == cid),
                     key=lambda r: r["scored_positions"])
        xs = [r["scored_positions"] for r in pts]
        ys = [r["token_mean_kld"] for r in pts]
        drift = max(drift, (max(ys) - min(ys)) / min(ys) * 100)
        ax.plot(xs, ys, color=c[key], lw=1.5, marker="o", markersize=5,
                markeredgecolor=c["bg"], markeredgewidth=0.8, zorder=4)
        ends.append((xs[-1], ys[-1], f"{short} {ys[-1]:.6f}", key, 10.5))
    if NVFP4_FULL is not None:
        _n, short, key = identity("nvfp4")
        x, y = NVFP4_FULL["scored_positions"], NVFP4_FULL["token_mean_kld"]
        ax.scatter([x], [y], s=80, marker="D", color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=0.9)
        ends.append((x, y, f"{short} {y:.6f}\n(10M only, no ladder)", key, 18.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    xs_all = sorted({r["scored_positions"] for r in rows})
    ax.set_xlim(xs_all[0] * 0.82, xs_all[-1] * 3.3)
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f"{cp}\n{x:,}" for cp, x in zip(CHECKPOINTS, xs_all)],
                       fontsize=7.6)
    ax.minorticks_off()
    ax.set_xlabel("cumulative scored positions at each ladder checkpoint",
                  color=c["fg"], fontsize=9)
    ax.set_ylabel("mean KL divergence, nats/token (log)", color=c["fg"],
                  fontsize=9.5)
    # The value table gets its own lane above the curves instead of sitting on
    # top of K4's line. A candidate with a 10M run but no checkpoints (NVFP4)
    # still has to fit under that lane.
    ys_all = [r["token_mean_kld"] for r in rows]
    if NVFP4_FULL is not None:
        ys_all.append(NVFP4_FULL["token_mean_kld"])
    ax.set_ylim(min(ys_all) * 0.72, max(ys_all) * 1.18)
    reserve_top(ax, 9.8 * (len(cands) + 3) + 8)
    ylo, yhi = ax.get_ylim()
    # The plotted band can be under one decade wide, where the default log
    # ticks leave a single labelled gridline. These are the decade and
    # half-decade steps that actually fall inside it.
    yticks = [t for t in (0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05,
                          0.1, 0.2, 0.3)
              if ylo <= t <= yhi]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.3f}" for t in yticks], fontsize=8.4)

    # End labels are placed only once the scale is final: two candidates can sit
    # a couple of points apart on this axis, so each label is pushed up just
    # enough to clear the one below it. The marker never moves, only its text.
    pt = ax.figure.dpi / 72.0
    dy = 0.0
    prev_top = None
    for x, y, text, key, room in sorted(ends, key=lambda e: e[1]):
        py = ax.transData.transform((0, y))[1] / pt
        if prev_top is not None and py + dy < prev_top:
            dy = prev_top - py
        else:
            dy = 0.0
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(8, dy),
                    ha="left", va="center", fontsize=7.8 if room < 12 else 7.4,
                    color=c[key], zorder=5)
        prev_top = py + dy + room

    first = {r["candidate"]: r for r in rows if r["checkpoint"] == CHECKPOINTS[0]}
    last = {r["candidate"]: r for r in rows if r["checkpoint"] == CHECKPOINTS[-1]}
    table = [(None, f"{'candidate':<10}{'1M mean':>10}{'10M mean':>11}"
                    f"{'move':>8}{'rel 95 % width':>17}")]
    for cid in cands:
        _name, short, key = identity(cid)
        a, b = first[cid], last[cid]
        move = (b["token_mean_kld"] - a["token_mean_kld"]) / a["token_mean_kld"] * 100
        table.append((key, f"{short:<10}{a['token_mean_kld']:10.6f}"
                           f"{b['token_mean_kld']:11.6f}{move:+7.1f} %"
                           f"{a['ci95_width_relative'] * 100:8.1f} %"
                           f" -> {b['ci95_width_relative'] * 100:.1f} %"))
    table.append((None, "the interval is governed by source diversity, not by "
                        "scored positions: it does not shrink"))
    table.append((None, f"every mean moves by less than {drift:.1f} % of its own "
                        "value across a tenfold increase; no ordering changes"))
    draw_table(ax, c, (0.012, 0.985), table, fontsize=6.9, step=9.8)

    panel_title(ax, c, "B — same v5 suite, ladder checkpoints: does more of the same "
                       "corpus move the answer?\nsame protocol as A · cumulative "
                       f"{min(xs_all):,} -> {max(xs_all):,} scored positions · "
                       f"{FULL['hyd']['source_clusters']} source clusters at 10M\n"
                       "vLLM builds only: no GGUF candidate ran all ten shards")


def draw_prior_panel(ax, c: dict, data: dict, title: str, positions: int,
                     contexts: int, note: str, ratios: dict | None = None) -> None:
    """One prior suite on its OWN y-axis. Never shared with anything."""
    style_axes(ax, c)
    metrics = data["metrics"]
    rows = [(k, s, key) for k, s, key in PRIOR_ROWS if k in metrics]
    for i, (mkey, short, ckey) in enumerate(rows):
        m = metrics[mkey]
        mean = m["mean_kld"]
        lo, hi = m["ci95"]["ci95_low"], m["ci95"]["ci95_high"]
        ax.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor=c[ckey], elinewidth=2.0, capsize=5.0, capthick=1.8,
                    zorder=3)
        ax.scatter([i], [mean], s=64, marker="o", color=c[ckey], zorder=4,
                   edgecolor=c["bg"], linewidth=0.9)
        ax.annotate(f"{mean:.6f}", (i, hi), textcoords="offset points",
                    xytext=(0, 7), ha="center", va="bottom", fontsize=7.0,
                    color=c[ckey], zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([s for _k, s, _c in rows], fontsize=7.6, rotation=30,
                       ha="right")
    lo = min(metrics[k]["ci95"]["ci95_low"] for k, _s, _c in rows)
    hi = max(metrics[k]["ci95"]["ci95_high"] for k, _s, _c in rows)
    # The ratio band, where it is drawn, needs its own clear lane above the data.
    ax.set_ylim(lo * 0.55, hi * 1.35)
    if ratios:
        rows_needed = 3 + -(-len(ratios["per_candidate"]) // 3)
        reserve_top(ax, 9.0 * rows_needed + 10)
    ylo, yhi = ax.get_ylim()
    # Each prior suite gets the ticks its own range needs. The two tick sets
    # come out different, which is the point: these are two separate axes.
    yticks = [t for t in (0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2)
              if ylo <= t <= yhi]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.3f}" for t in yticks], fontsize=8.4)
    ax.minorticks_off()
    ax.set_ylabel("mean KLD, this suite only (log)", color=c["fg"], fontsize=8.6)
    ax.annotate(note, (0.5, 0.02), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=6.9, color=c["muted"])

    if ratios:
        # Measured proof of the rule this panel exists to enforce: the same
        # candidate's value in this suite over its value in the v5 shard-0 run
        # of panel A. A BAND of factors, not one factor, is precisely why no
        # cross-suite conversion exists -- while the ordering survives intact.
        per = ratios["per_candidate"]
        order = sorted(per, key=lambda k: per[k]["ratio_v3_over_v5"])
        lo_r, hi_r = ratios["ratio_range"]
        table = [(None, "measured: this suite's mean over the SAME candidate's "
                        "v5 shard-0 mean (panel A)")]
        cells = [f"{identity(k)[1]} {per[k]['ratio_v3_over_v5']:.2f}x" for k in order]
        for i in range(0, len(cells), 3):
            table.append((None, "  ".join(f"{cell:<17}" for cell in cells[i:i + 3])))
        table.append((None, f"{lo_r:.2f} x to {hi_r:.2f} x — a "
                            f"{ratios['ratio_spread_end_to_end']:.2f} x spread, so "
                            "NO single conversion factor exists"))
        table.append((None, ("ordering identical in both suites"
                             if ratios["ordering_identical"] else
                             "ordering differs between the suites")
                            + ": only absolute values fail to carry"))
        draw_table(ax, c, (0.012, 0.985), table, fontsize=6.4, step=9.0)
    panel_title(ax, c, f"{title}\n{contexts} contexts, {positions:,} scored positions "
                       "— OWN y-axis")


def draw_his_panel(ax, c: dict) -> None:
    style_axes(ax, c)
    for key, name, marker, pts in HIS:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, color=c[key], lw=1.3, alpha=0.75, zorder=3)
        ax.scatter(xs, ys, s=64, marker=marker, color=c[key], zorder=4,
                   edgecolor=c["bg"], linewidth=0.8, label=name)
        for x, y, lbl, (dx, dy, ha) in pts:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=7.2, color=c[key], zorder=5)
    ax.axhline(HIS_FLOOR_MEAN, color=c["floor"], lw=1.3, ls=(0, (5, 3)), zorder=2)
    ax.annotate(f"his synthetic noise floor, mean {HIS_FLOOR_MEAN:.4f} "
                f"(median {HIS_FLOOR_MEDIAN:.4f})", (0.012, HIS_FLOOR_MEAN),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 5), ha="left", va="bottom", fontsize=7.2,
                color=c["floor"], zorder=5)

    # Our builds appear here with NO y-value: an x placement and nothing else.
    # The labels hang from the empty top band so they never cross his curves.
    for label, manifest, ckey, dash in OUR_ON_HIS_AXIS:
        gib = decoder_weight_gib(manifest)
        ax.axvline(gib, color=c[ckey], lw=1.2, ls=dash, alpha=0.85, zorder=2)
        ax.annotate(f"{label} {gib:.2f} GiB", (gib, 0.62), rotation=90,
                    ha="right", va="top", textcoords="offset points",
                    xytext=(-4, 0), fontsize=7.0, color=c[ckey], zorder=5)
    ax.annotate("dashed verticals: OUR builds' decoder weight on his x-axis —\n"
                "an x placement and nothing else; we have never run this protocol",
                (0.012, 0.985), xycoords="axes fraction", ha="left", va="top",
                fontsize=7.0, color=c["muted"], zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(5.4, 27.0)
    ax.set_ylim(0.0045, 0.85)
    ax.set_xlabel("quantized decoder weight, GiB — HIS convention: embeddings "
                  "excluded, output head included", color=c["fg"], fontsize=9)
    ax.set_ylabel("his published mean KLD (log)", color=c["fg"], fontsize=9.5)
    leg = ax.legend(loc="upper right", fontsize=7.6, framealpha=0.93,
                    labelspacing=0.4)
    leg.get_frame().set_facecolor(c["bg"])
    leg.get_frame().set_edgecolor(c["grid"])
    for t in leg.get_texts():
        t.set_color(c["fg"])
    panel_title(ax, c, "D — a protocol we have never run: turboderp's published chart "
                       f"labels, read off his images\nOpenWebText, 8 x 8192 = "
                       f"{HIS_POSITIONS:,} formatted positions, his own BF16 reference, "
                       "his own head in the loop")


def draw_barrier(fig, c: dict, box) -> None:
    """The hatched gap between the two prior suites: not one axis, not one suite."""
    fig.patches.append(Rectangle(
        (box[0], box[1]), box[2], box[3], transform=fig.transFigure,
        facecolor="none", edgecolor=c["warn"], hatch="////", lw=0.0,
        alpha=0.30, zorder=1))
    fig.text(box[0] + box[2] / 2, box[1] + box[3] / 2,
             "NOT ONE\nAXIS", rotation=90, ha="center", va="center",
             fontsize=7.4, color=c["warn"], zorder=2)


def draw(theme: str) -> list[Path]:
    c = THEMES[theme]
    fig = plt.figure(figsize=(19.2, 14.4), facecolor=c["bg"])
    outer = fig.add_gridspec(2, 2, left=0.045, right=0.995, top=0.878,
                             bottom=0.222, hspace=0.42, wspace=0.135,
                             width_ratios=[1.34, 1.0], height_ratios=[1.16, 1.0])
    top_left = outer[0, 0].subgridspec(1, 2, width_ratios=[3.35, 1.0], wspace=0.045)
    ax_size = fig.add_subplot(top_left[0, 0])
    ax_lane = fig.add_subplot(top_left[0, 1], sharey=ax_size)
    ax_ladder = fig.add_subplot(outer[0, 1])
    bottom_left = outer[1, 0].subgridspec(1, 2, width_ratios=[6.0, 4.0], wspace=0.34)
    ax_v3 = fig.add_subplot(bottom_left[0, 0])
    ax_v4 = fig.add_subplot(bottom_left[0, 1])
    ax_his = fig.add_subplot(outer[1, 1])

    draw_size_panel(ax_size, c)
    draw_lane_panel(ax_lane, c)
    draw_ladder_panel(ax_ladder, c)
    draw_prior_panel(
        ax_v3, c, V3, "C1 — corrected v3 suite (superseded protocol)",
        V3_POSITIONS, V3["analysis_contexts_after"],
        f"{V3['analysis_contexts_before']} contexts minus "
        f"{len(V3['excluded_source_clusters'])} contaminated source clusters\n"
        "positions: as_served_metrics.scored_positions of that receipt",
        ratios=RATIOS)
    draw_prior_panel(
        ax_v4, c, V4, "C2 — source-disjoint v4 qualification",
        V4_POSITIONS, V4["qualification_contexts_after"],
        f"{V4['qualification_contexts_before']} contexts minus "
        f"{len(V4['excluded_source_clusters'])} contaminated source clusters\n"
        f"positions: {V4['qualification_contexts_after']} contexts x "
        f"{V4_PER_CONTEXT:,} each\n(receipts/report-v4-hyd-qual.json)")
    draw_his_panel(ax_his, c)

    # The barrier sits in the wspace between C1 and C2, in figure coordinates,
    # so the two prior suites can never read as one axis.
    b3 = ax_v3.get_position()
    b4 = ax_v4.get_position()
    draw_barrier(fig, c, (b3.x1 + (b4.x0 - b3.x1) * 0.08, b3.y0,
                          (b4.x0 - b3.x1) * 0.36, b3.height))

    fig.suptitle("Every KL divergence we have measured — four protocols, four "
                 "panels, and deliberately no axis between them",
                 color=c["fg"], fontsize=13.5, y=0.985)
    fig.text(0.045, 0.952,
             "Each panel names its protocol and its scored-position count. "
             "Values inside one panel are comparable to each other and to "
             "nothing else in this image.",
             color=c["muted"], fontsize=9.0)

    w = WINDOW["findings"]
    ladder_first = min(r["scored_positions"] for r in LADDER["rows"])
    ladder_last = max(r["scored_positions"] for r in LADDER["rows"])
    footer = [
        f"A — v5 held-out suite, shard 0: {SUITE['contexts']} contexts, "
        f"{SUITE['scored_positions']:,} scored positions, {SUITE['source_clusters']} "
        "source clusters, identical for every candidate; KL(BF16 reference || "
        "candidate) over all 248,320 vocabulary entries, both operands through one "
        "shared BF16 head; source-cluster bootstrap, 10,000 resamples. "
        "receipts/cross-engine-comparator.json + the per-candidate reports it names.",
        f"B — same suite, cumulative {ladder_first:,} -> {ladder_last:,} scored "
        f"positions over ten shards ({FULL['hyd']['contexts']:,} contexts, "
        f"{FULL['hyd']['source_clusters']} clusters at 10M), five vLLM builds. "
        "receipts/kld5-ladder-convergence.json, receipts/kld5-10M-*.json.  "
        f"C1 — corrected v3: {V3['analysis_contexts_after']} contexts, "
        f"{V3_POSITIONS:,} positions.  C2 — source-disjoint v4: "
        f"{V4['qualification_contexts_after']} contexts, {V4_POSITIONS:,} positions.  "
        f"D — turboderp's protocol: {HIS_POSITIONS:,} positions, his corpus, his "
        "reference, his head.",
        f"RULE 1 — the engine term is not shared. Every GGUF value in A is measured "
        f"under llama.cpp and contains the {FLOOR['mean_kld']:.6f} cross-engine floor "
        f"({FLOOR['top1_agreement'] * 100:.2f} % top-1, p99.9 {FLOOR['p999_kld']:.4f}, "
        f"{FLOOR_REPORT['scored_positions']:,} positions, "
        "receipts/gguf-report-engine-floor.json); every vLLM value does not. The "
        "squares are therefore upper bounds, the hollow squares subtract the floor "
        "naively, and KL is not additive, so those are estimates and not identities.",
        "RULE 2 — no cross-panel ratio is meaningful. A/B, C1, C2 and D differ in "
        "corpus, context length, scored-position selection, reference numerics, "
        "vocabulary handling and head placement. Dividing a number in one panel by a "
        "number in another produces a quantity that measures none of them. C1 and C2 "
        "are not comparable to each other either, which is why they have separate "
        "y-axes and a barrier rather than a shared one.",
        f"Scored-window control (receipts/scored-window-offset.json): {w['offset_at_256']}; "
        f"{w['offset_at_1024']}. {w['uniformity']}",
        "Panel A x is serialized bytes on disk — the pinned single-file GGUF for the "
        "squares, the published immutable payload of our build receipts for the "
        "circles. Serialized bytes are never VRAM, never resident weights and never "
        "KV. Candidates without a published serialized size are placed in the lane at "
        "A's right edge instead of being given an estimated x. Panel D's y-values are "
        "read off published images, not a data file; our builds appear there as an x "
        "placement with no y-value at all.",
    ]
    wrapped: list[str] = []
    for para in footer:
        wrapped += textwrap.wrap(para, width=FOOTER_WIDTH)
    for i, line in enumerate(wrapped):
        fig.text(0.005, 0.176 - i * 0.0116, line, color=c["muted"], fontsize=7.6)

    written = []
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        p = OUT / f"kld-all-measurements-{theme}.{ext}"
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
    for t in THEMES:
        draw(t)
    print("\nreceipts read at runtime (every one of our values comes from these):")
    for name in OPENED:
        print("  receipts/" + name)
    print(f"\npanel A candidates ({len(CAND)}): {', '.join(sorted(CAND))}")
    print(f"  NVFP4 on the v5 shard: {NVFP4_NOTE}")
    print("  NVFP4 ten-shard run: "
          + (f"receipts/kld5-10M-nvfp4.json, mean "
             f"{NVFP4_FULL['token_mean_kld']:.6f} over "
             f"{NVFP4_FULL['scored_positions']:,} positions, plotted in panel B"
             if NVFP4_FULL is not None else "not present yet"))
