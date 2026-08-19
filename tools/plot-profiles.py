#!/usr/bin/env python3
"""Regenerate the three serving-profile charts from ONE source of truth.

Why this exists
---------------
The original three charts (`profiles-tradeoff`, `profiles-throughput`,
`fidelity-vs-quants`) were produced ad hoc with no generator committed, so they
silently went stale: every number in them predated the K5 corruption cure
(fidelity prefill 1,966 -> 2,987.7, +52%) and the balanced+ANY_BITS promotion
(3,251 -> 3,925.2). A chart nobody can regenerate is a chart that will be wrong
again, so the numbers now live in one table here and all three charts derive
from it.

It also fixes a protocol defect in `fidelity-vs-quants`: that chart placed
512-context served measurements on the SAME axis as v5-suite numbers
(10.48 M positions), including third-party quantisations, with a footnote calling
the difference "sample size". It is not a sample-size difference, it is a
different protocol - the same class of comparison this project retracted from its
HF model cards. Cross-protocol bars are now split into their own panel with the
gap stated numerically.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH. Update here; all three charts follow.
# All rows: RTX 5090, 600 W, n=3 boots, KLD on the 512-context shard-0 suite
# (1,048,064 scored positions, shared BF16 LM head).
# ---------------------------------------------------------------------------
PROFILES = {
    #                prefill  fox    essay  ctx      kld       kld_lo    kld_hi    p99
    "fidelity":   dict(pp=2987.7, pp_sd=4.4,  fox=228.3, essay=104.1, ctx=238_400,
                       kld=0.003405, lo=0.003166, hi=0.003672, p99=0.034889,
                       note="all-trellis K5/K6 + ANY_BITS"),
    "balanced":   dict(pp=3925.2, pp_sd=13.1, fox=215.6, essay=103.7, ctx=199_104,
                       kld=0.005672, lo=0.005302, hi=0.006087, p99=0.059908,
                       note="trellis + gate_up MXFP6"),
    "throughput": dict(pp=9638.9, pp_sd=18.3, fox=187.4, essay=94.3, ctx=249_600,
                       kld=0.063759, lo=None, hi=None, p99=0.7010,
                       note="all-FP4 runtime conversion"),
}
COLORS = {"fidelity": "#2e8b57", "balanced": "#e8a33d", "throughput": "#cf4457"}

# Measured but deliberately not shipped as profiles (same 512-ctx protocol).
NOT_SHIPPED = [
    ("self_attn-FP4", 0.011534, 238_400),
    ("all-FP6",       0.010699, 99_000),
]

# Same-protocol requant artifacts (512-ctx suite).
REQUANT = [
    ("RTN FP8attn+NVFP4mlp",  0.022121),
    ("GPTQ FP8attn+NVFP4mlp", 0.028548),
]

# DIFFERENT PROTOCOL - v5 suite, 10.48 M positions. Never on a shared axis.
V5_SUITE = [
    ("this ckpt, offline",        0.002700, False),
    ("Qwen3.8-27B-FP8 (official)", 0.005294, True),
    ("unsloth NVFP4",             0.031059, True),
]

TARGETS = dict(pp=7000, fox=190, essay=83, ctx=238_400, kld=0.012)


def chart_tradeoff(path: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.axhspan(0.002, TARGETS["kld"], xmin=0, xmax=1, color="tab:green", alpha=0.0)
    # target zone
    ax.add_patch(plt.Rectangle((TARGETS["pp"], 0.0028), 20000, TARGETS["kld"] - 0.0028,
                               color="tab:green", alpha=0.10, zorder=0))
    ax.axvline(TARGETS["pp"], color="tab:green", ls="--", lw=1.4, alpha=0.8)
    ax.axhline(TARGETS["kld"], color="tab:green", ls="--", lw=1.4, alpha=0.8)

    for name, d in PROFILES.items():
        ax.scatter([d["pp"]], [d["kld"]], s=330, color=COLORS[name],
                   edgecolor="black", linewidth=1.1, zorder=6, label=f"{name} — {d['note']}")
        if d["lo"]:
            ax.plot([d["pp"], d["pp"]], [d["lo"], d["hi"]], color="black", lw=1.4, zorder=5)
        ax.plot([d["pp"] - d["pp_sd"], d["pp"] + d["pp_sd"]], [d["kld"]] * 2,
                color="black", lw=1.4, zorder=5)
        ax.annotate(f"{name}\n{d['pp']:,.1f} tok/s · KLD {d['kld']:.6f}\n{d['ctx']:,} ctx",
                    (d["pp"], d["kld"]), textcoords="offset points",
                    xytext=(-12, -62), ha="center", fontsize=9.6,
                    color=COLORS[name], weight="bold")
    for label, kld, ctx in NOT_SHIPPED:
        ax.scatter([2050 if "self" in label else 4400], [kld], s=95, facecolor="none",
                   edgecolor="dimgray", linewidth=1.3, marker="s", zorder=5)
        ax.annotate(f"{label}\n({ctx:,} ctx)",
                    (2050 if "self" in label else 4400, kld),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=8.4, color="dimgray")
    ax.text(TARGETS["pp"] * 1.06, 0.0104,
            f"TARGET ZONE\nprefill ≥ {TARGETS['pp']:,}  &  KLD ≤ {TARGETS['kld']}\n"
            "still empty — nothing reaches both",
            color="darkgreen", fontsize=10.5, weight="bold", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1500, 20000)
    ax.set_ylim(0.0028, 0.13)
    ax.set_xticks([2000, 3000, 4000, 6000, 7000, 10000, 15000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))
    ax.set_yticks([0.003, 0.005, 0.01, 0.012, 0.02, 0.05, 0.1])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.set_xlabel("Prefill throughput — 2051-token prompt, tok/s  (higher is better) →",
                  fontsize=11.5)
    ax.set_ylabel("← KLD vs BF16  (lower = more faithful)", fontsize=11.5)
    ax.set_title("Qwen3.8-27B EXL3 K5K6 — pick a serving profile: speed vs fidelity\n"
                 "RTX 5090 32 GB · 600 W · n=3 boots · KLD on 512 contexts vs BF16",
                 fontsize=13.5, weight="bold")
    ax.legend(loc="upper left", fontsize=9.6, framealpha=0.95)
    ax.grid(alpha=0.25, which="both")
    fig.text(0.5, 0.012,
             "hollow squares = measured but not shipped as a profile   ·   "
             "error bars = 95% CI (KLD) and ±SD over 3 boots (prefill)",
             ha="center", fontsize=8.6, color="#444444")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def chart_throughput(path: str) -> None:
    names = list(PROFILES)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.6))
    panels = [
        ("Prefill\n(tok/s, 2051-tok prompt)", "pp", "pp_sd", TARGETS["pp"], "{:,.1f}"),
        ("Decode — short prompt\n(tok/s, “fox”)", "fox", None, TARGETS["fox"], "{:.1f}"),
        ("Decode — long generation\n(tok/s, “essay”, 500 tok)", "essay", None,
         TARGETS["essay"], "{:.1f}"),
        ("Max context\n(tokens served)", "ctx", None, TARGETS["ctx"], "{:,.0f}"),
    ]
    for ax, (title, key, sdkey, target, fmt) in zip(axes, panels):
        vals = [PROFILES[n][key] for n in names]
        errs = [PROFILES[n][sdkey] if sdkey else 0 for n in names]
        bars = ax.bar(names, vals, yerr=errs, capsize=4,
                      color=[COLORS[n] for n in names], edgecolor="black", linewidth=0.7)
        ax.axhline(target, color="black", ls="--", lw=1.2)
        ax.text(-0.45, target, f" ≥ {target:,} target", fontsize=8.6, va="bottom")
        for b, n, v in zip(bars, names, vals):
            ok = v >= target
            ax.annotate(("✓ " if ok else "✗ ") + fmt.format(v),
                        (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=10.2, weight="bold",
                        color="darkgreen" if ok else "#b3243c")
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Qwen3.8-27B EXL3 K5K6 — what each vLLM-GG serving profile delivers "
                 "  (RTX 5090 32 GB, 600 W, n=3 boots)", fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def chart_fidelity(path: str) -> None:
    """Two panels: one protocol each. Never a shared axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 9.6),
                                   gridspec_kw=dict(height_ratios=[3.1, 1.5], hspace=0.42))

    rows = [(f"PROFILE={n}\n(served)", PROFILES[n]["kld"], PROFILES[n]["lo"],
             PROFILES[n]["hi"], COLORS[n]) for n in PROFILES]
    rows += [(f"{l}\n(served, not shipped)", k, None, None, "#9a9a9a")
             for l, k, _ in NOT_SHIPPED]
    rows += [(f"{l}\n(requant checkpoint)", k, None, None, "#6a7fb5") for l, k in REQUANT]
    rows.sort(key=lambda r: r[1])

    y = np.arange(len(rows))
    for i, (label, kld, lo, hi, color) in enumerate(rows):
        ax1.barh(i, kld, color=color, edgecolor="black", linewidth=0.7, height=0.68)
        if lo:
            ax1.plot([lo, hi], [i, i], color="black", lw=1.5)
        ax1.annotate(f" {kld:.6f}", (kld, i), va="center", fontsize=10, weight="bold")
    ax1.axvline(TARGETS["kld"], color="tab:green", ls="--", lw=1.6)
    ax1.text(TARGETS["kld"] * 1.04, len(rows) - 0.6, "0.012 fidelity bar",
             color="darkgreen", fontsize=10.5, weight="bold")
    ax1.set_yticks(y, [r[0] for r in rows], fontsize=9.4)
    ax1.set_xscale("log")
    ax1.set_xlim(0.002, 0.11)
    ax1.invert_yaxis()
    ax1.set_xlabel("KLD vs BF16 — lower is more faithful (log scale)", fontsize=11)
    ax1.set_title("A. ONE protocol: 512-context shard-0 suite, 1,048,064 scored "
                  "positions, shared BF16 LM head.\nEvery bar here is directly "
                  "comparable to every other bar here.",
                  fontsize=11.8, weight="bold", loc="left")
    ax1.grid(axis="x", alpha=0.25, which="both")

    for i, (label, kld, third) in enumerate(V5_SUITE):
        ax2.barh(i, kld, color="#8c8c8c" if third else "#2e8b57",
                 hatch="//" if third else None, edgecolor="black",
                 linewidth=0.7, height=0.6)
        ax2.annotate(f" {kld:.6f}", (kld, i), va="center", fontsize=10, weight="bold")
    ax2.set_yticks(range(len(V5_SUITE)), [r[0] for r in V5_SUITE], fontsize=9.4)
    ax2.set_xscale("log")
    ax2.set_xlim(0.002, 0.11)
    ax2.invert_yaxis()
    ax2.set_xlabel("KLD vs BF16 — v5-suite protocol (log scale)", fontsize=11)
    ax2.set_title("B. DIFFERENT protocol: v5 suite, 10,485,760 positions — 10x panel A. "
                  "Kept separate deliberately.\nThe same weights score 0.002700 here and "
                  "0.003405 in panel A; a 26% gap from protocol alone, so a bar in A "
                  "cannot be read against a bar in B.",
                  fontsize=11.8, weight="bold", loc="left")
    ax2.grid(axis="x", alpha=0.25, which="both")

    fig.suptitle("Fidelity of this checkpoint's serving profiles and requants",
                 fontsize=14.5, weight="bold", y=0.975)
    fig.text(0.5, 0.006,
             "hatched = third-party quantisation.  Requant checkpoints are our own "
             "(FP8 attention + NVFP4 MLP, RTN and GPTQ calibration).  "
             "No third-party checkpoint has been measured on panel A's protocol.",
             ha="center", fontsize=8.8, color="#444444")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="charts")
    args = ap.parse_args()
    chart_tradeoff(f"{args.outdir}/profiles-tradeoff.png")
    chart_throughput(f"{args.outdir}/profiles-throughput.png")
    chart_fidelity(f"{args.outdir}/fidelity-vs-quants.png")


if __name__ == "__main__":
    main()
