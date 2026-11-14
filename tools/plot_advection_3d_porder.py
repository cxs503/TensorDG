#!/usr/bin/env python
"""Generate convergence plots for the 3-D advection p-order study."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-3d-porder" / "porder_convergence_3d.json"
FIG_DIR = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-3d-porder"


def main():
    data = json.loads(ARTIFACT.read_text())
    results = data["study_3d"]["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: convergence plot
    ax = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    markers = ["o", "s", "^"]
    for idx, (key, r) in enumerate(sorted(results.items())):
        p = r["degree"]
        cells = np.array(r["cells"], dtype=float)
        errs = np.array(r["errors_rel_l2"])
        ax.loglog(cells, errs, f"{markers[idx]}-", color=colors[idx],
                  label=f"P{p} (observed: {r['observed_order_lsq']:.3f})")
        # Reference slope
        ref_cells = np.array([cells[0], cells[-1]])
        ref_errs = errs[0] * (ref_cells / cells[0]) ** (-(p + 1))
        ax.loglog(ref_cells, ref_errs, "--", color=colors[idx], alpha=0.4,
                  label=f"P{p} ref slope p+1={p+1}")

    ax.set_xlabel("Cells per dimension (N)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("3-D Linear Advection: Convergence Study")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: order comparison bar chart
    ax = axes[1]
    degrees = [r["degree"] for r in sorted(results.values(), key=lambda x: x["degree"])]
    observed = [r["observed_order_lsq"] for r in sorted(results.values(), key=lambda x: x["degree"])]
    theoretical = [r["theoretical_order"] for r in sorted(results.values(), key=lambda x: x["degree"])]
    deviations = [r["order_deviation"] for r in sorted(results.values(), key=lambda x: x["degree"])]

    x_pos = np.arange(len(degrees))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, theoretical, width, label="Theoretical (p+1)",
                   color="#3498db", alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, observed, width, label="Observed",
                   color="#e74c3c", alpha=0.7)

    # Add deviation labels
    for i, (d, dev) in enumerate(zip(degrees, deviations)):
        ok = abs(dev) <= 0.35
        ax.text(i, max(observed[i], theoretical[i]) + 0.1,
                f"Δ={dev:+.3f} {'PASS' if ok else 'FAIL'}",
                ha="center", fontsize=9,
                color="green" if ok else "red")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"P{d}" for d in degrees])
    ax.set_ylabel("Order of Accuracy")
    ax.set_title("3-D: Observed vs Theoretical Order")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = FIG_DIR / "advection_3d_porder_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
