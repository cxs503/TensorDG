"""Figures for the C6 (divergence) gate report.

Numbers are read from on-disk artefacts, never typed by hand:

* ``validation/riemann2d_c6/scan.json``   -- cluster CFL x M scan (gate calibration)
* ``validation/riemann2d_c6/local_probe.json`` -- local 16^2 / 8^2 probes at
  t = 0.05 and t = 0.25 produced by
  ``benchmarks/riemann_2d_divergence_scan.py``-compatible calls
* ``validation/riemann2d_c6/recorded_divergence.json`` -- the numbers quoted in
  ``docs/CASE-DG-CKS1998-2D-RIEMANN-CONSERVATION.md`` (previous mission), kept in
  a separate file so it is never confused with this mission's measurements.

Usage: PYTHONPATH=. python tools/c6_report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUNDLE = Path("validation/riemann2d_c6")
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def _load(name):
    path = BUNDLE / name
    return json.loads(path.read_text()) if path.exists() else None


def main():
    probe = _load("local_probe.json")
    scan = _load("scan.json")
    recorded = _load("recorded_divergence.json")
    made = []

    # --- figure 1: measured mass change vs the a-priori C6 bound (log scale)
    if probe:
        rows = probe["rows"]
        labels = [r["label"] for r in rows]
        change = np.abs([r["mass_change"] for r in rows])
        bound = np.array([r["mass_flux_bound"] for r in rows])
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        idx = np.arange(len(rows))
        ax.bar(idx - 0.2, bound, width=0.38, color="#bbbbbb",
               label="C6 a-priori bound ($5\\rho_{max,0}s_{max,0}Pt$)")
        ax.bar(idx + 0.2, change, width=0.38, color="#1f77b4",
               label="measured $|\\Delta M|$")
        if recorded:
            ax.scatter([len(rows) - 0.5], [recorded["mass_change"]], marker="X", s=110,
                       color="#d62728", zorder=5,
                       label="recorded 32$^2$ t=0.25 (prev. mission)")
        ax.set_yscale("log")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("mass   [log scale]")
        ax.set_title("C6 gate: measured mass change vs a-priori bound "
                     "(Lax-Liu cfg 3, DG P1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        png = OUT / "cks1998_c6_mass_vs_bound.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        made.append(str(png))

    # --- figure 2: cluster CFL x M scan (calibration of the safety factors)
    if scan and scan.get("rows"):
        rows = [r for r in scan["rows"] if r["c6_mass_bound_ratio"] is not None]
        if rows:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
            for tvb in sorted({r["tvb_constant"] for r in rows}):
                subset = sorted((r for r in rows if r["tvb_constant"] == tvb),
                                key=lambda r: r["cfl"])
                axes[0].plot([r["cfl"] for r in subset],
                             [r["c6_mass_bound_ratio"] for r in subset], "o-",
                             label=f"TVB M={tvb:g}")
                axes[1].plot([r["cfl"] for r in subset],
                             [r["c6_extremum_factor"] for r in subset], "s-",
                             label=f"TVB M={tvb:g}")
            axes[0].axhline(1.0, color="k", ls="--", lw=1, label="C6 limit")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("CFL")
            axes[0].set_ylabel(r"$|\Delta M|/M_{\rm flux\ bound}$")
            axes[1].axhline(scan["gates"]["C6_extremum_factor"], color="k", ls="--",
                            lw=1, label="C6 limit")
            axes[1].set_xlabel("CFL")
            axes[1].set_ylabel(r"max extremum factor")
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            first = rows[0]
            fig.suptitle("CKS1998 2-D Riemann C6 calibration: P{deg} {n}x{n}, "
                         "t={t}".format(deg=first["degree"], n=first["ncells"],
                                        t=first["final_time"]))
            fig.tight_layout()
            png = OUT / "cks1998_c6_cfl_scan.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            made.append(str(png))

    print("figures:", made)
    return made


if __name__ == "__main__":
    main()