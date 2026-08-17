#!/usr/bin/env python3
"""Two card charts the ladder receipts support but make_knee_chart.py does not.

Panel A - DP scaling efficiency at MATCHED PER-REPLICA LOAD. A DP-N replica set
only receives 1/N of the client concurrency, so "twice the GPUs" only means
"twice the work offered" when the client concurrency is doubled too. Comparing
DP2@C against DP4@2C against DP8@4C is therefore the only honest scaling axis,
and it is plotted against the linear reference rather than described in prose.

Read the caveat printed on the panel: our tiers were measured on DIFFERENT
hosts and, for the 4x tier, through a different harness path (localhost, because
that VM was VPC-isolated). docs/46 §24 records that this makes the cross-tier
ratio unreliable at the 5-17 % scale of the effect - which is why no scaling law
is published and why the 4x TB row stays blank. The panel exists to show the
scatter honestly, not to fit a curve through it.

Panel B - the decode roofline, with the honest denominator. The vendor spec
(1.792 TB/s) is not achievable on this SKU; a pure-stream kernel measures
1462-1525 GB/s (docs/47 F1). Against that ceiling, and with the corrected
per-step traffic of 20.5 GB (the 953 MB head streams once per draft sampling,
docs/47 F6), decode runs at ~85 % of achievable rather than the 55-65 % of spec
this project used to quote. Bars, not prose, because the whole point is that
two different denominators were in play.

Usage:
    make_scaling_charts.py --out-prefix /var/tmp/work/charts/tb21-scaling
                           [--receipts-dir receipts]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a1a", "grid": "#c9c9c9", "panel_bg": "#f7f7f7",
              "series": ["#1f6feb", "#d1242f", "#1a7f37", "#8250df", "#bf8700"]},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d", "panel_bg": "#161b22",
             "series": ["#58a6ff", "#ff7b72", "#3fb950", "#bc8cff", "#e3b341"]},
}

# docs/47 F1/F6 - all measured on the same RTX PRO 6000 Blackwell SKU.
SPEC_BW = 1792.0          # vendor spec, GB/s [uncalibrated]
CEIL_LO, CEIL_HI = 1462.0, 1525.0   # measured pure-stream ceiling (F1)
TRAFFIC_MEASURED = 20.5   # GB per decode step (F6)
TRAFFIC_ASSUMED = 17.4    # what this project used to assume
# RECONCILED 2026-08-17 with docs/47's author. The earlier "985-1165 GB/s" was
# the brief's own 55-65 %-of-spec re-expressed (0.55x1792, 0.65x1792) and
# relabelled "effective" without recomputing; the "~85 %" headline was the
# favourable endpoint of a plausible pairing, unlabelled. Recomputed from the
# fixed 20.5 GB/step numerator across real operating points, wall-clock:
#   (135.8 tok/s, 2.7 accepted/step) -> 1033 GB/s = 69 % of the 1494 midpoint
#   (129.5 tok/s, 2.1 accepted/step) -> 1267 GB/s = 85 %
# Note the inversion: the LOWER-acceptance point implies the HIGHER bandwidth
# share, because per-step traffic is fixed while tokens per step vary.
EFF_LO, EFF_HI = 1033.0, 1267.0


def agg(path: Path, shape: str) -> dict[int, float]:
    """Aggregate tok/s by concurrency, from one tb21_ladder receipt."""
    d = json.loads(path.read_text())
    return {c["concurrency"]: c["median_aggregate_tok_s"]
            for c in d["tables"][shape]["cells"]}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--receipts-dir", default="receipts")
    args = ap.parse_args()
    R = Path(args.receipts_dir)

    tiers = {2: R / "tb21-ladder-2x-dp2.json", 4: R / "tb21-ladder-4x-dp4.json",
             8: R / "tb21-ladder-8x-dp8.json"}
    missing = [str(p) for p in tiers.values() if not p.exists()]
    if missing:
        raise SystemExit("missing receipts: " + ", ".join(missing))

    for theme_name, C in THEMES.items():
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.2), facecolor=C["bg"])
        for ax in (axA, axB):
            ax.set_facecolor(C["panel_bg"])
            ax.tick_params(colors=C["fg"])
            for s in ax.spines.values():
                s.set_color(C["grid"])
            ax.grid(True, color=C["grid"], alpha=0.35, linewidth=0.6)

        # ---- Panel A: scaling at matched per-replica load ----
        per_replica = [2, 4, 8, 16]
        for si, shape in enumerate(("512x256", "4kx1k")):
            data = {n: agg(p, shape) for n, p in tiers.items()}
            for ti, (lo, hi) in enumerate(((2, 4), (4, 8))):
                xs, ys = [], []
                for pr in per_replica:
                    a, b = data[lo].get(pr * lo), data[hi].get(pr * hi)
                    if a and b:
                        xs.append(pr)
                        ys.append(b / a / 2 * 100)
                if xs:
                    axA.plot(xs, ys, marker="o" if si == 0 else "s",
                             linestyle="-" if si == 0 else "--",
                             color=C["series"][ti], linewidth=1.9, markersize=7,
                             label=f"DP{lo}->DP{hi}  {'512-in/256-out' if si==0 else '4k-in/1k-out'}")
        axA.axhline(100, color=C["fg"], linestyle=":", linewidth=1.3, alpha=0.8)
        axA.text(2.1, 100.8, "linear (2x GPUs = 2x aggregate)", color=C["fg"],
                 fontsize=8.5, alpha=0.85)
        axA.set_xscale("log", base=2)
        axA.set_xticks(per_replica)
        axA.set_xticklabels([str(p) for p in per_replica])
        axA.set_xlabel("requests per replica (matched load)", color=C["fg"])
        axA.set_ylabel("achieved / linear  (%)", color=C["fg"])
        axA.set_title("A - DP scaling at matched per-replica load", color=C["fg"], fontsize=11)
        axA.legend(fontsize=8, facecolor=C["panel_bg"], edgecolor=C["grid"],
                   labelcolor=C["fg"], loc="lower left")
        axA.text(0.5, -0.235,
                 "Tiers were measured on DIFFERENT hosts, and the 4x tier through a different "
                 "harness path (localhost; that VM was VPC-isolated).\ndocs/46 §24: the ratio is "
                 "unreliable at the 5-17 % scale of the effect, so NO scaling law is published and "
                 "the 4x TB row stays blank.",
                 transform=axA.transAxes, ha="center", va="top", color=C["fg"],
                 fontsize=7.6, alpha=0.82)

        # ---- Panel B: the decode roofline, honest denominator ----
        eff_mid = (EFF_LO + EFF_HI) / 2
        ceil_mid = (CEIL_LO + CEIL_HI) / 2
        bars = [
            ("vendor spec\n(NOT achievable)", SPEC_BW, C["series"][1], 0.0),
            ("measured pure-stream\nceiling (F1)", ceil_mid, C["series"][2], (CEIL_HI - CEIL_LO) / 2),
            ("our decode, effective\n(F6 numerator)", eff_mid, C["series"][0], (EFF_HI - EFF_LO) / 2),
        ]
        xs = list(range(len(bars)))
        axB.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.58,
                edgecolor=C["grid"])
        for x, (_, v, _, err) in zip(xs, bars):
            if err:
                axB.errorbar([x], [v], yerr=[[err], [err]], fmt="none",
                             ecolor=C["fg"], capsize=6, alpha=0.9)
            axB.text(x, v + err + 34, f"{v:,.0f}" + (f"\n±{err:,.0f}" if err else ""),
                     ha="center", color=C["fg"], fontsize=9.5, linespacing=1.25)
        # Percentages use the ceiling MIDPOINT for both ends, matching the
        # reconciled 69-85 % in docs/47; the ceiling's own +/-32 GB/s is shown
        # as an error bar rather than smeared into the headline range.
        lo_pct, hi_pct = EFF_LO / ceil_mid * 100, EFF_HI / ceil_mid * 100
        spec_pct = eff_mid / SPEC_BW * 100
        axB.set_xticks(xs)
        axB.set_xticklabels([b[0] for b in bars], color=C["fg"], fontsize=9)
        axB.set_ylabel("memory bandwidth (GB/s)", color=C["fg"])
        axB.set_ylim(0, SPEC_BW * 1.20)
        axB.set_title(f"B - decode is {lo_pct:.0f}-{hi_pct:.0f} % of ACHIEVABLE "
                      f"(and {spec_pct:.0f} % of a spec nobody can reach)",
                      color=C["fg"], fontsize=11)
        axB.text(0.5, -0.26,
                 f"Both terms of the old \"55-65 % of roofline\" claim were wrong, in the same "
                 f"direction. Numerator: {TRAFFIC_MEASURED} GB/step, not ~{TRAFFIC_ASSUMED} - the "
                 f"953 MB head streams\nonce per draft sampling (docs/47 F6). Denominator: a "
                 f"pure-stream kernel measures {CEIL_LO:,.0f}-{CEIL_HI:,.0f} GB/s, not the "
                 f"{SPEC_BW:,.0f} spec (F1). Effective range is\n{EFF_LO:,.0f}-{EFF_HI:,.0f} GB/s "
                 f"across real operating points at 2.1-2.7 accepted/step. Counterintuitively the LOWER-acceptance "
                 f"point gives the HIGHER share:\nper-step traffic is fixed while tokens per step vary. The 1.792 TB/s figure is banned as headroom.",
                 transform=axB.transAxes, ha="center", va="top", color=C["fg"],
                 fontsize=7.4, alpha=0.85, linespacing=1.5)

        fig.suptitle("Qwen3.8-27B EXL3 K5/K6 - multi-GPU scaling and the decode roofline, measured",
                     color=C["fg"], fontsize=12.5)
        fig.tight_layout(rect=(0, 0.16, 1, 0.94))
        for ext in ("svg", "png"):
            out = f"{args.out_prefix}-{theme_name}.{ext}"
            fig.savefig(out, facecolor=C["bg"], dpi=150 if ext == "png" else None)
            print(f"wrote {out}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
