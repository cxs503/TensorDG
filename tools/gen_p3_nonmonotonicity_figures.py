"""Generate figures for the P3 non-monotonicity analysis.

Produces:
- fig_p3_nonmonotonicity.png: convergence curves for P1/P2/P3 with P3 anomaly highlighted
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT = Path("validation/xval_dg_round2/artifacts/multifield_audit.json")
OUTDIR = Path("validation/xval_dg_round2/artifacts")


def _load():
    with open(AUDIT) as f:
        return json.load(f)


def _get_errors(data, degree, field="rho"):
    """Get sorted (ncells, error) pairs for a degree."""
    errs = []
    for leg in data["legs"]:
        if leg["degree"] == degree and leg["cfl"] == 0.4:
            errs.append((leg["ncells"], leg[field]["rel_l1_percent"]))
    errs.sort()
    return errs


def gen_figures():
    data = _load()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Convergence curves for P1/P2/P3
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"1": "#1f77b4", "2": "#2ca02c", "3": "#d62728"}
    markers = {"1": "o", "2": "s", "3": "^"}

    for deg in [1, 2, 3]:
        errs = _get_errors(data, deg)
        ncells = [e[0] for e in errs]
        errors = [e[1] for e in errs]
        ax.loglog(ncells, errors, f"{markers[str(deg)]}-", color=colors[str(deg)],
                  label=f"P{deg}", linewidth=2, markersize=8)

    # Reference lines for theoretical orders
    n_ref = np.array([200, 800])
    for deg, label in [(1, "O(h²)"), (2, "O(h³)"), (3, "O(h⁴)")]:
        e0 = _get_errors(data, deg)[0][1]
        n0 = _get_errors(data, deg)[0][0]
        e_ref = e0 * (n_ref / n0) ** (-(deg + 1))
        ax.loglog(n_ref, e_ref, "--", color=colors[str(deg)], alpha=0.3, label=f"{label} (smooth)")

    # Highlight the P3 anomaly at N=800
    p2_n800 = _get_errors(data, 2)[-1][1]
    p3_n800 = _get_errors(data, 3)[-1][1]
    ax.annotate(f"P3 > P2\n({p3_n800:.3f}% > {p2_n800:.3f}%)",
                xy=(800, p3_n800), xytext=(500, p3_n800 * 1.5),
                fontsize=10, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax.set_xlabel("Number of cells (N)", fontsize=12)
    ax.set_ylabel("Relative L1 error (%)", fontsize=12)
    ax.set_title("Shu-Osher: DG convergence with WENO limiter\n"
                 "P3 non-monotonicity at N=800 (limiter mode-zeroing)", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_p3_nonmonotonicity.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTDIR / 'fig_p3_nonmonotonicity.png'}")

    # Figure 2: EOC comparison
    fig, ax = plt.subplots(figsize=(6, 4))

    eocs = {}
    for deg in [1, 2, 3]:
        errs = _get_errors(data, deg)
        deg_eocs = []
        for i in range(1, len(errs)):
            e1, n1 = errs[i-1][1], errs[i-1][0]
            e2, n2 = errs[i][1], errs[i][0]
            eoc = -math.log(e2 / e1) / math.log(n2 / n1)
            deg_eocs.append(eoc)
        eocs[deg] = deg_eocs

    x = np.arange(1, 3)  # refinement steps
    width = 0.25
    for deg in [1, 2, 3]:
        ax.bar(x + (deg - 2) * width, eocs[deg], width, label=f"P{deg}",
               color=colors[str(deg)], edgecolor="white")

    # Theoretical orders
    ax.axhline(y=2, color="#1f77b4", linestyle=":", alpha=0.5, label="Theoretical P1")
    ax.axhline(y=3, color="#2ca02c", linestyle=":", alpha=0.5, label="Theoretical P2")
    ax.axhline(y=4, color="#d62728", linestyle=":", alpha=0.5, label="Theoretical P3")

    ax.set_xticks(x)
    ax.set_xticklabels(["200→400", "400→800"])
    ax.set_ylabel("Effective Order of Convergence (EOC)", fontsize=11)
    ax.set_title("EOC comparison: P3 anomaly in second refinement", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate the P3 anomaly
    ax.annotate(f"EOC={eocs[3][1]:.2f}\n< P2: {eocs[2][1]:.2f}",
                xy=(2 + width, eocs[3][1]), xytext=(2.5, eocs[3][1] + 0.3),
                fontsize=9, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_p3_eoc_anomaly.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTDIR / 'fig_p3_eoc_anomaly.png'}")


if __name__ == "__main__":
    gen_figures()
