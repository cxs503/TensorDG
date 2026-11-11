"""Generate publication-quality figures for the Burgers sine benchmark."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from tensordg.burgers_1d import (
    MEAN, AMPLITUDE, initial_profile, exact_profile,
    solve_burgers, burgers_error,
)
from benchmarks._modal_1d import sample

rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

OUT = Path("artifacts/burgers_sine_benchmark")
OUT.mkdir(parents=True, exist_ok=True)


def plot_profile_comparison():
    """Plot numerical vs exact at t=0.1 for P2 N=128."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: full profile
    ax = axes[0]
    for N, ls in [(16, "--"), (64, "-."), (128, "-")]:
        x_c, coeff, _ = solve_burgers(N, 2, 0.1, 0.1)
        xs, un = sample(coeff, 801)
        ax.plot(xs, un, ls, label=f"P2 N={N}", alpha=0.8)

    xt = torch.as_tensor(xs, dtype=torch.float64)
    ue = exact_profile(xt, 0.1).numpy()
    ax.plot(xs, ue, "k:", lw=2, label="Exact (characteristics)")

    # Initial condition
    xi = torch.linspace(0, 1, 401, dtype=torch.float64)
    ui = initial_profile(xi).numpy()
    ax.plot(xi.numpy(), ui, "gray", lw=1, alpha=0.5, label="t=0 (initial)")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=0.1)")
    ax.set_title("Burgers sine: profile comparison (P2 DG)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Right: error distribution for N=128
    ax = axes[1]
    x_c, coeff, _ = solve_burgers(128, 2, 0.1, 0.1)
    xs, un = sample(coeff, 801)
    xt = torch.as_tensor(xs, dtype=torch.float64)
    ue = exact_profile(xt, 0.1).numpy()
    ax.semilogy(xs, np.abs(un - ue) + 1e-16, "b-", label="|u_num - u_exact|")
    ax.set_xlabel("x")
    ax.set_ylabel("Pointwise absolute error")
    ax.set_title("P2 N=128 pointwise error at t=0.1")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_profile_comparison.png")
    plt.close(fig)
    print("  Saved fig1_profile_comparison.png")


def plot_convergence_orders():
    """Plot convergence rates for P1, P2, P3."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cells = [8, 16, 32, 64, 128]
    colors = {"P1": "#d62728", "P2": "#2ca02c", "P3": "#1f77b4"}
    expected_slopes = {"P1": 2, "P2": 3, "P3": 4}

    for pname, color in colors.items():
        errors = []
        for N in cells:
            p = int(pname[1])
            x, coeff, _ = solve_burgers(N, p, 0.1, 0.1)
            m = burgers_error(x, coeff, 0.1)
            errors.append(m["relative_l2"])

        hs = [1.0 / N for N in cells]
        ax.loglog(hs, errors, "o-", color=color, label=f"{pname} (measured)")

        # Reference slope
        p_deg = int(pname[1])
        e_ref = errors[0]
        h_ref = hs[0]
        ref_line = [e_ref * (h / h_ref) ** (p_deg + 1) for h in hs]
        ax.loglog(hs, ref_line, "--", color=color, alpha=0.5,
                  label=f"O(h^{p_deg+1}) reference")

    ax.set_xlabel("Mesh size h = 1/N")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Burgers sine: h-convergence at t=0.1")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(OUT / "fig2_convergence.png")
    plt.close(fig)
    print("  Saved fig2_convergence.png")


def plot_time_evolution():
    """Show wave steepening at different times."""
    fig, ax = plt.subplots(figsize=(8, 5))

    times = [0.0, 0.05, 0.1, 0.15, 0.19]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

    for t, c in zip(times, colors):
        xt = torch.linspace(0, 1, 1001, dtype=torch.float64)
        if t == 0.0:
            ue = initial_profile(xt).numpy()
        else:
            ue = exact_profile(xt, t).numpy()
        label = f"t={t:.2f}" if t > 0 else "t=0 (initial)"
        ax.plot(xt.numpy(), ue, color=c, lw=1.5, label=label)

    t_shock = 1.0 / (AMPLITUDE * 2 * math.pi * 2 * math.pi)
    ax.axvline(x=0.75, color="red", ls=":", alpha=0.5, label=f"shock location")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"Burgers sine: wave steepening (t_c = {t_shock:.4f})")
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_time_evolution.png")
    plt.close(fig)
    print("  Saved fig3_time_evolution.png")


def plot_time_error():
    """Show error vs time for fixed mesh."""
    fig, ax = plt.subplots(figsize=(8, 5))

    times = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.17, 0.19]
    errors = []
    for t in times:
        x, coeff, _ = solve_burgers(64, 2, t, 0.1)
        m = burgers_error(x, coeff, t)
        errors.append(m["relative_l2"] * 100)

    ax.semilogy(times, errors, "bo-", ms=6)
    t_shock = 1.0 / (AMPLITUDE * 2 * math.pi * 2 * math.pi)
    ax.axvline(x=t_shock, color="red", ls="--", alpha=0.7,
               label=f"t_c = {t_shock:.4f} (shock)")
    ax.set_xlabel("Final time t")
    ax.set_ylabel("Relative L2 error (%)")
    ax.set_title("P2 N=64: error growth vs time")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fig4_error_vs_time.png")
    plt.close(fig)
    print("  Saved fig4_error_vs_time.png")


if __name__ == "__main__":
    print("Generating Burgers sine benchmark figures...")
    plot_profile_comparison()
    plot_convergence_orders()
    plot_time_evolution()
    plot_time_error()
    print("Done.")
