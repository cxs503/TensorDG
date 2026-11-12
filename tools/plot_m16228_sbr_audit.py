#!/usr/bin/env python3
"""m16228 audit figure: CASE-DG-SOLID-BODY-ROTATION-2D (m16200) evidence map.

Plots every number the subject actually wrote to disk (artifacts/*.json),
the declared 5 % acceptance gate, and the auditor's own two independent runs.
The declared acceptance grid (100x100) is drawn as a hatched "NOT RUN" band
because no artifact and no test execution exists for it.

Inputs (read-only): artifacts copied into runs/m16228/.
Output: docs/figs/m16228_sbr_audit.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "m16228"
OUT = ROOT / "docs" / "figs" / "m16228_sbr_audit.png"

TOL = 0.05  # subject-declared acceptance gate (L1 relative < 5 %)


def load(name: str) -> dict:
    return json.loads((RUNS / name).read_text())["case"]


def main() -> None:
    series = {}
    for deg in (1, 2):
        pts = []
        for name in sorted(RUNS.glob(f"case_dg_solid_body_rotation_2d_p{deg}_n*.json")):
            c = load(name.name)
            pts.append((c["ncells"], c["l1_relative"], "subject artifact"))
        series[deg] = sorted(pts)

    repro = load("auditor_repro_p2n25.json")
    novel = load("auditor_novel_p1n20.json")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # ---- left: L1 error vs grid, log-log, declared gate
    for deg, pts in series.items():
        ns = [p[0] for p in pts]
        es = [p[1] for p in pts]
        ax.loglog(ns, es, "o-", label=f"P{deg} (subject artifacts, n<=32)")
        for n, e in zip(ns, es):
            ax.annotate(f"{e*100:.2f}%", (n, e), textcoords="offset points",
                        xytext=(4, 6), fontsize=7)
    ax.loglog([repro["ncells"]], [repro["l1_relative"]], "k*", ms=14,
              label="auditor re-run P2 n=25 (bit-identical)")
    ax.loglog([novel["ncells"]], [novel["l1_relative"]], "ks", ms=6,
              label="auditor novel config P1 n=20 (not in any artifact)")
    ax.axhline(TOL, color="crimson", ls="--", lw=1.2, label="declared gate L1 < 5%")
    ax.axvspan(90, 115, color="0.85", hatch="//", alpha=0.6, lw=0)
    ax.text(100, 0.30, "declared\n100x100\nNOT RUN", ha="center", va="center",
            fontsize=8, color="crimson", fontweight="bold")
    ax.set_xlabel("cells per direction n")
    ax.set_ylabel("relative L1 error (disk-restricted, r=0.28)")
    ax.set_title("CASE-DG-SOLID-BODY-ROTATION-2D: what was actually run")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="lower left")

    # ---- right: peak retention (the second declared metric) + mass drift bar
    labels, vals = [], []
    for deg, pts in series.items():
        for n, _e, _s in pts:
            labels.append(f"P{deg}\nn={n}")
            vals.append(load(f"case_dg_solid_body_rotation_2d_p{deg}_n{n}.json")["peak_retention_nodal"])
    colors = ["#4c72b0" if l.startswith("P1") else "#55a868" for l in labels]
    bars = ax2.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax2.axhline(1 - TOL, color="crimson", ls="--", lw=1.2, label="declared gate peak > 95%")
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v*100:.2f}%",
                 ha="center", fontsize=7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylim(0.80, 1.02)
    ax2.set_ylabel("peak retention (nodal)")
    ax2.set_title("second declared metric: peak decay after one rotation")
    ax2.grid(True, axis="y", ls=":", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("m16228 audit of m16200 — mass drift |rel| = 0 in all runs (<=3.9e-16), "
                 "i.e. conservation is exact but the acceptance grid was never executed",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()