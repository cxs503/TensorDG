"""Generate figures for the CASE-DG-ADVECT-ORDER unified report.

Reads the assembled metrics and produces three PNG figures.

    PYTHONPATH=. python benchmarks/plot_advorder_unified.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("validation/advection_order_unified")
FIG = pathlib.Path("docs/figures")
METRICS = OUT / "metrics.json"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text())

    p1p4 = m["p1p4"]
    p0 = m["p0"]
    comparison = p1p4["comparison"]
    error_splits = p1p4["error_split"]

    # Figure 1: convergence rates for P1-P4
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in comparison:
        d = row["degree"]
        ax.loglog(row["meshes"], row["got_l2"], "o-", label=f"P{d} ({row['integrator']})")
        ax.loglog(row["meshes"], row["ref_l2"], "x--", label=f"P{d} ref ({row['ref_column']})")
    ax.set_xlabel("Number of cells N")
    ax.set_ylabel("L2 error")
    ax.legend()
    ax.set_title("CASE-DG-ADVECT-ORDER: Convergence rates P1-P4")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "advorder_unified_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'advorder_unified_convergence.png'}")

    # Figure 2: error split (E_time vs E_space)
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in error_splits:
        ax.plot(s["E_space"], s["E_time"], "o", label=f"P{s['degree']} {s['integrator']}")
    xs = np.logspace(-10, -2, 50)
    ax.plot(xs, 0.1 * xs, "k--", alpha=0.5, label="E_time = 0.1 E_space")
    ax.set_xlabel("E_space")
    ax.set_ylabel("E_time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Time vs Space error split (cfl=0.2)")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "advorder_unified_error_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'advorder_unified_error_split.png'}")

    # Figure 3: P0 asymptotic constant convergence
    if "derived" in p0:
        fig, ax = plt.subplots(figsize=(6, 5))
        # Reconstruct from derived data
        # We know the analytic ladder goes from N=20 to N=40960
        # Use the derived values to show the convergence
        ncells = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]
        # Reconstruct N*L2 from the derived constant and local orders
        # The constant at finest is ~14.013, asymptotic is ~14.0165
        const_asymptotic = p0["closed_form"]["asymptotic_constant"]
        const_measured = p0["derived"]["constant_measured_finest"]

        # Use the analytic data from the original P0 metrics
        p0_orig = json.loads(pathlib.Path("validation/advection_order_p0/metrics.json").read_text())
        analytic = p0_orig["analytic"]
        ncells_actual = [a["ncells"] for a in analytic]
        n_times_l2 = [a["n_times_l2"] for a in analytic]

        ax.semilogx(ncells_actual, n_times_l2, "o-", label="N * L2(N)")
        ax.axhline(const_asymptotic, color="r", linestyle="--",
                   label=f"Closed form: {const_asymptotic:.4f}")
        ax.set_xlabel("Number of cells N")
        ax.set_ylabel("N * L2 error")
        ax.legend()
        ax.set_title("P0: Asymptotic error constant convergence")
        ax.grid(True, alpha=0.3)
        fig.savefig(FIG / "advorder_unified_p0_constant.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {FIG / 'advorder_unified_p0_constant.png'}")

    print("All figures generated.")


if __name__ == "__main__":
    main()
