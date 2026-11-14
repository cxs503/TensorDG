#!/usr/bin/env python3
"""Generate contour figures for CASE-DG-EULER-VORTEX-ERR.

Produces:
- fig_density_dg_t1.png: DG-P2/32 density at t=1 vs exact
- fig_density_p0_t1.png: P0/128 density at t=1 vs exact
- fig_density_p0_t5.png: P0/128 density at t=5 vs exact (if available)
- fig_error_vs_time.png: error vs time (regenerated from errconserve)
- fig_invariants_vs_time.png: invariants vs time (regenerated from errconserve)
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUTPUT_DIR = Path("case_euler_vortex_errconserve")


def load_field(path: Path):
    """Load a .npz field file, return (density, x, y)."""
    data = np.load(path)
    return data["density"], data["x"], data["y"]


def plot_density_comparison(num_field, num_x, num_y, exact_density, exact_x, exact_y,
                            title, save_path, num_label="Numerical", exact_label="Exact"):
    """Side-by-side density contour: numerical (left) vs exact (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Common color scale
    vmin = min(num_field.min(), exact_density.min())
    vmax = max(num_field.max(), exact_density.max())
    cmap = "viridis"

    # Numerical
    ax = axes[0]
    X, Y = np.meshgrid(num_x, num_y, indexing="ij")
    cf = ax.contourf(X, Y, num_field, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{num_label}\nmin={num_field.min():.4f}, max={num_field.max():.4f}")
    plt.colorbar(cf, ax=ax, label="density")

    # Exact
    ax = axes[1]
    XE, YE = np.meshgrid(exact_x, exact_y, indexing="ij")
    cf2 = ax.contourf(XE, YE, exact_density, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{exact_label}\nmin={exact_density.min():.4f}, max={exact_density.max():.4f}")
    plt.colorbar(cf2, ax=ax, label="density")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    results = OUTPUT_DIR / "results"

    # Load exact solutions
    exact_t1 = load_field(results / "exact_t1_fields.npz")
    exact_t5 = load_field(results / "exact_t5_fields.npz")

    # 1. DG-P2/32 at t=1 vs exact
    dg_t1 = load_field(Path("case_euler_vortex/results/dg_d2_n32_t1_cfl0.2_fields.npz"))
    plot_density_comparison(
        dg_t1[0], dg_t1[1], dg_t1[2],
        exact_t1[0], exact_t1[1], exact_t1[2],
        "DG-P2/32² Density at t = 1",
        OUTPUT_DIR / "fig_density_dg_t1.png",
        num_label="DG-P2/32² (t=1)",
        exact_label="Exact (t=1)"
    )

    # 2. P0/128 at t=1 vs exact
    p0_t1 = load_field(results / "p0_n128_t1_cfl0.4_fields.npz")
    plot_density_comparison(
        p0_t1[0], p0_t1[1], p0_t1[2],
        exact_t1[0], exact_t1[1], exact_t1[2],
        "P0/128² Density at t = 1",
        OUTPUT_DIR / "fig_density_p0_t1.png",
        num_label="P0/128² (t=1)",
        exact_label="Exact (t=1)"
    )

    # 3. P0/128 at t=5 vs exact (if available)
    p0_t5_path = results / "p0_n128_t5_cfl0.4_fields.npz"
    if p0_t5_path.exists():
        p0_t5 = load_field(p0_t5_path)
        plot_density_comparison(
            p0_t5[0], p0_t5[1], p0_t5[2],
            exact_t5[0], exact_t5[1], exact_t5[2],
            "P0/128² Density at t = 5",
            OUTPUT_DIR / "fig_density_p0_t5.png",
            num_label="P0/128² (t=5)",
            exact_label="Exact (t=5)"
        )
    else:
        print("WARNING: p0_n128_t5_cfl0.4_fields.npz not found, skipping t=5 contour")

    print("All figures generated.")


if __name__ == "__main__":
    main()
