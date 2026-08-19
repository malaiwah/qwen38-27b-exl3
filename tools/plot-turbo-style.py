#!/usr/bin/env python3
"""KLD-vs-weight-size charts in the visual language of turboderp's exl3 cards.

Reproduces the three diagram types published on
https://huggingface.co/turboderp/Qwen3.8-27B-exl3 using OUR measurements:

  1. kld-spread-vs-size.png  log-y median with a shaded p25-p75 band, mean as a
                             dotted overlay, one series per format family.
  2. kld-mean-vs-size.png    the same points on a linear mean axis.
  3. kld-histograms.png      per-token KLD histograms, log x, one curve per config.

Two things are deliberately NOT copied from those cards:

* **No noise floor line.** turboderp draws the divergence of the BF16 reference
  against itself under bf16-rounding-scale perturbation, and expresses every
  result as a multiple of it. We have never measured ours, so drawing one would
  be fabrication. The mechanism is cheap (perturb the reference at rounding
  scale, re-score the same suite) and is tracked as a todo; until then the
  charts carry no floor and no "x floor" axis.
* **No shared y-axis with his numbers.** His suite is openwebtext 8x8192
  formatted; ours is the 512-context shard-0 suite at 2047 scored positions per
  context. Same methodology, different data -> incomparable magnitudes. Only the
  x-axis is shared, because his "quantized weight size |W_q| (excl. embeddings,
  incl. output head)" is exactly our trellis payload definition.

X-axis derivation. Sizes are computed from exact format definitions, not
estimated: trellis K_n stores n_words/16 = n bits per weight; NVFP4 is 4 bits
plus one FP8 scale per 16 elements (4.5); MXFP6 is 6 bits plus one E8M0 scale
per 32 (6.25); FP8 per-channel is 8.0. Summing role numel x bits over the 409
quantized matrices reproduces the known 16.82 GiB K5K6 payload to 0.2%, which
is what validates the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GIB = 1024 ** 3

# --- turboderp's palette, sampled from the published PNGs -------------------
BG, AXBG, GRID = "#16181d", "#1f232b", "#333844"
FG, MUTED = "#dfe3ea", "#9aa1ad"
GOLD, GREEN, PINK, BLUE, ORANGE = "#f2c81e", "#c3d96e", "#ee3a72", "#4c9ede", "#e8763a"

BITS = {"K5": 5.0, "K6": 6.0, "NVFP4": 4.5, "MXFP6": 6.25, "FP8": 8.0}


def role(name: str) -> str:
    if "mtp" in name: return "mtp"
    if "lm_head" in name: return "lm_head"
    if "mlp.gate_proj" in name or "mlp.up_proj" in name: return "gate_up"
    if "mlp.down_proj" in name: return "down"
    if "self_attn" in name: return "self_attn"
    if "linear_attn" in name: return "gdn"
    raise KeyError(name)


def load_numel(ladder: Path) -> dict[str, int]:
    mods = json.loads(ladder.read_text())["modules"]
    out: dict[str, int] = {}
    for n, m in mods.items():
        out[role(n)] = out.get(role(n), 0) + m["numel"]
    return out


def size_gib(numel: dict[str, int], assign: dict[str, str]) -> float:
    return sum(numel[r] * BITS[assign[r]] / 8 for r in numel) / GIB


# --- families: colour, and the per-role format assignment of each config ----
BASE = {"gate_up": "K5", "down": "K6", "self_attn": "K6",
        "gdn": "K6", "lm_head": "K6", "mtp": "K6"}


def var(**kw) -> dict[str, str]:
    a = dict(BASE); a.update(kw); return a


def all_of(fmt: str) -> dict[str, str]:
    return {r: fmt for r in BASE}


# tag -> (display label, family, assignment)
CONFIGS = {
    "alltrellis-anybits":    ("fidelity (K5K6)",          "EXL3 trellis", var()),
    "alltrellis-fp8kv":      ("fp8-KV",                   "EXL3 trellis", var()),
    "depth-early":           ("FP6 band L0-12",           "FP6 depth band", var()),
    "depth-mid":             ("FP6 band L26-38",          "FP6 depth band", var()),
    "depth-late":            ("FP6 band L51-63",          "FP6 depth band", var()),
    "gateup-fp6-anybits":    ("balanced (gate_up FP6)",   "runtime MXFP6", var(gate_up="MXFP6")),
    "allfp6-int6emb":        ("all-FP6",                  "runtime MXFP6", all_of("MXFP6")),
    "selfattn-fp4":          ("self_attn FP4",            "runtime NVFP4", var(self_attn="NVFP4")),
    "attrib-down":           ("mlp.down FP4",             "runtime NVFP4", var(down="NVFP4")),
    "attrib-gdn":            ("GDN FP4",                  "runtime NVFP4", var(gdn="NVFP4")),
    "attrib-gateup":         ("gate_up FP4",              "runtime NVFP4", var(gate_up="NVFP4")),
    "attrib-gateupdown":     ("all-MLP FP4",              "runtime NVFP4", var(gate_up="NVFP4", down="NVFP4")),
    "allfp4-262k":           ("throughput (all-FP4)",     "runtime NVFP4", all_of("NVFP4")),
    "rtn-fp8attn-nvfp4mlp":  ("RTN FP8attn+FP4mlp",       "compressed-tensors",
                              var(gate_up="NVFP4", down="NVFP4", self_attn="FP8", gdn="FP8", lm_head="FP8")),
    "gptq-fp8attn-nvfp4mlp": ("GPTQ FP8attn+FP4mlp",      "compressed-tensors",
                              var(gate_up="NVFP4", down="NVFP4", self_attn="FP8", gdn="FP8", lm_head="FP8")),
}

FAMILY = {
    "EXL3 trellis":       GOLD,
    "FP6 depth band":     BLUE,
    "runtime MXFP6":      GREEN,
    "runtime NVFP4":      ORANGE,
    "compressed-tensors": PINK,
}


def hist_quantiles(tail: dict, qs) -> dict[float, float]:
    """Exact-enough quantiles from the log-spaced kld_tail histogram.

    counts[0] is the underflow bin below edges[0]; counts[i] covers
    [edges[i-1], edges[i]) for i in 1..len(edges)-1; counts[-1] is overflow.
    Recovers the reported median to ~2%, which is well inside the band width.
    """
    e = np.asarray(tail["bin_edges"], float)
    c = np.asarray(tail["counts"], float)
    interior = c[1:len(e)]
    centres = np.sqrt(e[:-1] * e[1:])
    cum = np.cumsum(interior)
    n = float(tail["total_count"])
    out = {}
    for q in qs:
        i = int(np.searchsorted(cum, q * n - c[0]))
        out[q] = float(centres[min(i, len(centres) - 1)])
    return out


def collect(reports: Path, ladder: Path):
    numel = load_numel(ladder)
    pts = []
    for tag, (label, fam, assign) in CONFIGS.items():
        p = reports / f"report-{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        q = hist_quantiles(d["kld_tail"], [0.25, 0.75])
        pts.append({
            "tag": tag, "label": label, "family": fam,
            "gib": size_gib(numel, assign),
            "mean": d["token_mean_kld"], "median": d["token_median_kld"],
            "p25": q[0.25], "p75": q[0.75], "p99": d["p99_kld"],
            "tail": d["kld_tail"],
        })
    return sorted(pts, key=lambda r: r["gib"])


def style(ax):
    ax.set_facecolor(AXBG)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=11)


def frame(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(BG)
    style(ax)
    return fig, ax


def titles(fig, sub):
    fig.text(0.5, 0.955, "Qwen3.8-27B on one RTX 5090", ha="center",
             color=FG, fontsize=19)
    fig.text(0.5, 0.917, sub, ha="center", color=MUTED, fontsize=12)


def caption(fig, lines, y=0.152):
    fig.text(0.082, y, "\n".join(lines), ha="left", va="top",
             color=MUTED, fontsize=9.5, linespacing=1.55)


def legend(ax, fams):
    h = [Line2D([], [], color=FAMILY[f], marker="o", mfc=FAMILY[f],
                mec="white", mew=0.9, ms=8, lw=2.0, label=f) for f in fams]
    lg = ax.legend(handles=h, loc="upper right", frameon=False,
                   labelcolor=FG, fontsize=11.5)
    return lg


def place(ax, pts, key, fontsize=8.0, force_up=False):
    """Annotate every point, fanning labels within same-size clusters.

    Five configurations sit at exactly 16.85 GiB -- one checkpoint allocated
    differently -- so x-jitter would fabricate a size difference that does not
    exist. Instead, points are grouped by size and each group's labels are
    fanned vertically around the group, alternating which side of the axis they
    sit on so adjacent groups do not collide. Leader lines keep the association.
    """
    groups: dict[float, list[dict]] = {}
    for p in pts:
        groups.setdefault(round(p["gib"], 1), []).append(p)
    for gi, gx in enumerate(sorted(groups)):
        members = sorted(groups[gx], key=lambda r: r[key])
        n = len(members)
        # Single points take one of four quadrants so neighbouring groups in a
        # crowded size band do not stack; clusters fan vertically to one side.
        for k, p in enumerate(members):
            if n == 1:
                dx = (78, -78, 44, -44)[gi % 4]
                dy = (30, -34, -34, 30)[gi % 4]
                if force_up:
                    dy = (30, 52, 30, 52)[gi % 4]
            else:
                dx = 104 if gi % 2 == 0 else -104
                # On a linear axis a downward fan runs past the zero baseline,
                # so clusters stack upward instead.
                dy = (k * 30 + 22) if force_up else (k - (n - 1) / 2) * 30
            ax.annotate(f"{p['label']}\n{p['median']:.4f} | {p['mean']:.4f}",
                        (p["gib"], p[key]), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=fontsize,
                        color=FAMILY[p["family"]], weight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.24", fc="#12141a",
                                  ec=GRID, lw=0.5, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=FAMILY[p["family"]],
                                        lw=0.6, alpha=0.5,
                                        shrinkA=1, shrinkB=3))


def plot_spread(pts, out: Path):
    fig, ax = frame((14.6, 10.2))
    fams = [f for f in FAMILY if any(p["family"] == f for p in pts)]
    # Per-point p25-p75 whiskers. turboderp shades a band because each of his
    # families is a monotone bpw ladder; ours are not (five configs share one
    # size), and fill_between across a non-monotone relation draws nonsense.
    for p in pts:
        ax.vlines(p["gib"], p["p25"], p["p75"], color=FAMILY[p["family"]],
                  lw=7, alpha=0.22, zorder=2)
    for f in fams:
        s = sorted([p for p in pts if p["family"] == f], key=lambda r: r["gib"])
        x = [p["gib"] for p in s]
        if len(s) > 1:
            ax.plot(x, [p["mean"] for p in s], color=FAMILY[f], lw=1.0, ls=":",
                    marker="D", ms=4.2, alpha=0.8, zorder=3)
        ax.plot(x, [p["median"] for p in s], color=FAMILY[f], lw=0, marker="o",
                mfc=FAMILY[f], mec="white", mew=0.9, ms=8.5, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("quantized weight size  $|W_q|$ / GiB   (excl. embeddings, incl. output head)",
                  color=FG, fontsize=12.5, labelpad=9)
    ax.set_ylabel(r"per-token KL divergence,  $D_{\mathrm{KL}}(p_{\mathrm{BF16}}\,\|\,p_{\mathrm{quant}})$",
                  color=FG, fontsize=12.5, labelpad=9)
    ax.set_xlim(13.1, 19.6)
    ax.axhline(0.012, color="#8f96a3", ls=(0, (2, 3)), lw=1.1, zorder=1)
    ax.text(19.5, 0.0126, "criterion: mean KLD <= 0.012", color="#8f96a3",
            fontsize=9.5, va="bottom", ha="right")
    place(ax, pts, "median")
    legend(ax, fams)
    titles(fig, "512-context shard-0 suite, 2047 scored positions x 512 contexts, full 248,320 vocab")
    caption(fig, [
        "Per-token KL divergence between each served configuration and the BF16 reference. Circles are the median token; the vertical bar behind each is its",
        "p25-p75 spread; the dotted line joins the means within a family. Means run far above medians because the distribution is heavy-tailed -- divergence",
        "concentrates in the few tokens where the reference itself is undecided. Labels read median | mean. Sizes come from exact format definitions",
        "(trellis K_n = n bits, NVFP4 = 4+8/16, MXFP6 = 6+8/32, FP8 = 8), which reproduce the known 16.82 GiB K5K6 payload to 0.2%. Five configurations share",
        "16.85 GiB: they are one checkpoint allocated differently, so no x-jitter is applied. No self-noise floor is drawn -- we have not measured ours.",
    ])
    fig.subplots_adjust(left=0.082, right=0.975, top=0.888, bottom=0.235)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def plot_mean_linear(pts, out: Path):
    fig, ax = frame((14.6, 9.6))
    fams = [f for f in FAMILY if any(p["family"] == f for p in pts)]
    for f in fams:
        s = sorted([p for p in pts if p["family"] == f], key=lambda r: r["gib"])
        ax.plot([p["gib"] for p in s], [p["mean"] for p in s], color=FAMILY[f],
                lw=1.3, ls=":", marker="o", mfc=FAMILY[f], mec="white",
                mew=0.9, ms=9, zorder=4)
    ax.set_xlabel("quantized weight size  $|W_q|$ / GiB   (excl. embeddings, incl. output head)",
                  color=FG, fontsize=12.5, labelpad=9)
    ax.set_ylabel(r"mean KL divergence,  $D_{\mathrm{KL}}(p_{\mathrm{BF16}}\,\|\,p_{\mathrm{quant}})$",
                  color=FG, fontsize=12.5, labelpad=9)
    ax.set_xlim(13.1, 19.6)
    ax.axhline(0.012, color="#8f96a3", ls=(0, (2, 3)), lw=1.1, zorder=1)
    ax.text(19.5, 0.0128, "criterion: mean KLD <= 0.012", color="#8f96a3",
            fontsize=9.5, va="bottom", ha="right")
    ax.set_ylim(0, 0.076)
    place(ax, pts, "mean", force_up=True)
    legend(ax, fams)
    titles(fig, "512-context shard-0 suite -- mean on a linear axis; the same points as the spread chart")
    caption(fig, [
        "The mean on a linear axis makes the cliff visible: every runtime NVFP4 configuration except self_attn-only lands above the 0.012 criterion, while",
        "every trellis configuration sits an order of magnitude below it. all-FP4 reaches 0.063759 at 13.63 GiB -- 18.7x the fidelity profile for 3.2 GiB saved.",
    ], y=0.112)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.885, bottom=0.205)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


HIST_COLOUR = {
    "alltrellis-anybits":   GOLD,
    "gateup-fp6-anybits":   GREEN,
    "allfp6-int6emb":       "#7fb069",
    "selfattn-fp4":         ORANGE,
    "rtn-fp8attn-nvfp4mlp": PINK,
    "allfp4-262k":          "#b8503a",
}


def plot_hist(pts, out: Path, show=tuple(HIST_COLOUR)):
    fig, ax = frame((13.6, 9.0))
    for p in pts:
        if p["tag"] not in show:
            continue
        e = np.asarray(p["tail"]["bin_edges"], float)
        c = np.asarray(p["tail"]["counts"], float)[1:len(e)]
        centres = np.sqrt(e[:-1] * e[1:])
        m = c > 0
        ax.plot(centres[m], c[m], color=HIST_COLOUR[p["tag"]], lw=1.9,
                label=p["label"])
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 3.0)
    ax.set_xlabel("per-token KL divergence (log)", color=FG, fontsize=12.5, labelpad=9)
    ax.set_ylabel("tokens per bin", color=FG, fontsize=12.5, labelpad=9)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=11)
    titles(fig, "512-context shard-0 suite -- 1,048,064 scored positions per configuration")
    caption(fig, [
        "Per-token KL divergence against the BF16 reference, one histogram per configuration over the 561 log-spaced bins recorded in each report.",
        "The whole distribution shifts right as precision drops -- the degradation is not confined to a tail. Underflow (KLD < 1e-12) is excluded:",
        "16,708 of 1,048,064 positions for the fidelity profile, where the quantized model reproduces the reference distribution exactly.",
    ], y=0.115)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.885, bottom=0.215)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="receipts/kld-reports")
    ap.add_argument("--ladder", default="/tmp/ladder.json")
    ap.add_argument("--outdir", default="charts")
    a = ap.parse_args()
    pts = collect(Path(a.reports), Path(a.ladder))
    print(f"{len(pts)} configurations")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    plot_spread(pts, out / "kld-spread-vs-size.png")
    plot_mean_linear(pts, out / "kld-mean-vs-size.png")
    plot_hist(pts, out / "kld-histograms.png")


if __name__ == "__main__":
    main()
