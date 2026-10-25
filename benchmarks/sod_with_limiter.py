"""Sod shock tube with TVB/WENO limiters - demonstration benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import sample
from tensordg.euler_dg_1d import solve_euler_dg_1d


def _reconstruct(x, coeff, samples=4):
    """Reconstruct point values from modal coefficients."""
    centers = x.detach().cpu().numpy()
    values = coeff.detach().cpu().numpy()
    cells, modes, _ = values.shape
    reference = 2.0 * (np.arange(samples) + 0.5) / samples - 1.0
    phi = []
    for k in range(modes):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        phi.append(np.polynomial.legendre.legval(reference, polynomial))
    phi = np.stack(phi, axis=1)
    numerical = np.einsum("cmv,qm->cqv", values, phi).reshape(-1, 3)
    points = (centers[:, None] + (centers[-1] - centers[0]) / (cells - 1) / 2 * reference).reshape(-1)
    primitive = {
        "density": numerical[:, 0],
        "velocity": numerical[:, 1] / numerical[:, 0],
        "pressure": 0.4 * (numerical[:, 2] - 0.5 * numerical[:, 1] ** 2 / numerical[:, 0]),
    }
    return points, primitive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--limiter", choices=["none", "tvb", "weno"], default="tvb")
    parser.add_argument("--tvb-constant", type=float, default=1.0)
    parser.add_argument("--output-dir", default="sod_with_limiter")
    args = parser.parse_args()

    left_state = (1.0, 0.0, 1.0)
    right_state = (0.125, 0.0, 0.1)

    limiter = args.limiter if args.limiter != "none" else None
    
    x, coeff, metadata = solve_euler_dg_1d(
        left_state, right_state, ncells=args.cells, degree=args.degree,
        final_time=args.time, cfl=args.cfl, limiter=limiter,
        tvb_constant=args.tvb_constant,
    )
    
    errors = riemann_dg_errors(x, coeff, args.time, left_state, right_state)
    points, numerical = _reconstruct(x, coeff)
    exact = sample(points, args.time, left_state, right_state)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ("density", "velocity", "pressure"),
                              (r"Density $\rho$", "Velocity $u$", "Pressure $p$")):
        ax.plot(points, exact[key], "k-", linewidth=1.8, label="Exact Riemann")
        ax.plot(points, numerical[key], "r-", linewidth=1.0, alpha=0.8,
                label=f"DG P{args.degree} / HLLC / {args.limiter}")
        ax.set_ylabel(label)
        err_pct = errors[key]["relative_l2_percent"]
        ax.set_title(f"{label}: relative L2 error = {err_pct:.4f}%")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Sod shock tube | {args.cells} cells, P{args.degree}, t={args.time:g}, limiter={args.limiter}")
    fig.savefig(output / "sod_reference_profiles.png", dpi=160)
    plt.close(fig)

    # Save metrics
    metrics = {
        "case": "sod",
        "ncells": args.cells,
        "degree": args.degree,
        "final_time": args.time,
        "cfl": args.cfl,
        "limiter": args.limiter,
        "tvb_constant": args.tvb_constant,
        "steps": metadata["steps"],
        "errors": errors,
    }

    with open(output / "sod_limiter_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Sod shock tube with {args.limiter} limiter:")
    print(f"  cells={args.cells}, degree={args.degree}, time={args.time}")
    print(f"  steps={metadata['steps']}")
    print(f"  density L2 error: {errors['density']['relative_l2_percent']:.4f}%")
    print(f"  velocity L2 error: {errors['velocity']['relative_l2_percent']:.4f}%")
    print(f"  pressure L2 error: {errors['pressure']['relative_l2_percent']:.4f}%")
    print(f"Results saved to {output}/")


if __name__ == "__main__":
    main()
