"""Compare Sod/Lax shock tube with and without TVB/WENO limiters."""
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


def run_case(case_name, left_state, right_state, ncells, degree, final_time, cfl, limiter, tvb_constant):
    """Run a single case and return errors + metadata."""
    x, coeff, metadata = solve_euler_dg_1d(
        left_state, right_state, ncells=ncells, degree=degree,
        final_time=final_time, cfl=cfl, limiter=limiter, tvb_constant=tvb_constant,
    )
    errors = riemann_dg_errors(x, coeff, final_time, left_state, right_state)
    points, numerical = _reconstruct(x, coeff)
    exact = sample(points, final_time, left_state, right_state)
    return {
        "x": x, "coeff": coeff, "metadata": metadata,
        "errors": errors, "points": points, "numerical": numerical, "exact": exact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["sod", "lax"], default="sod")
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="limiter_comparison")
    args = parser.parse_args()

    if args.case == "sod":
        left_state = (1.0, 0.0, 1.0)
        right_state = (0.125, 0.0, 0.1)
        final_time = args.time if args.time else 0.2
    else:  # lax
        left_state = (0.445, 0.698, 3.528)
        right_state = (0.5, 0.0, 0.571)
        final_time = args.time if args.time else 0.13

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Run three configurations: no limiter, TVB, WENO
    configs = [
        ("no_limiter", None, 1.0),
        ("tvb", "tvb", 1.0),
        ("weno", "weno", 1.0),
    ]

    results = {}
    for name, limiter, tvb_const in configs:
        print(f"Running {args.case} with {name}...")
        results[name] = run_case(
            args.case, left_state, right_state,
            args.cells, args.degree, final_time, args.cfl,
            limiter, tvb_const,
        )

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    colors = {"no_limiter": "red", "tvb": "blue", "weno": "green"}
    
    for key in ("density", "velocity", "pressure"):
        ax = axes[("density", "velocity", "pressure").index(key)]
        ax.plot(results["no_limiter"]["points"], results["no_limiter"]["exact"][key],
                "k-", linewidth=2.0, label="Exact")
        for name in ["no_limiter", "tvb", "weno"]:
            err_pct = results[name]["errors"][key]["relative_l2_percent"]
            ax.plot(results[name]["points"], results[name]["numerical"][key],
                    "-", color=colors[name], linewidth=1.0, alpha=0.7,
                    label=f"{name} ({err_pct:.3f}%)")
        ax.set_ylabel(key.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("x")
    fig.suptitle(f"{args.case.upper()} shock tube | {args.cells} cells, P{args.degree}, t={final_time:g}")
    fig.savefig(output / f"{args.case}_limiter_comparison.png", dpi=160)
    plt.close(fig)

    # Save metrics
    metrics = {
        "case": args.case,
        "ncells": args.cells,
        "degree": args.degree,
        "final_time": final_time,
        "cfl": args.cfl,
    }
    for name in ["no_limiter", "tvb", "weno"]:
        metrics[name] = {
            "errors": results[name]["errors"],
            "steps": results[name]["metadata"]["steps"],
            "limiter": results[name]["metadata"]["limiter"],
        }

    with open(output / f"{args.case}_limiter_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{args.case.upper()} limiter comparison:")
    print(f"{'Limiter':<15} {'Density L2':<12} {'Velocity L2':<12} {'Pressure L2':<12} {'Steps':<8}")
    print("-" * 60)
    for name in ["no_limiter", "tvb", "weno"]:
        d_err = results[name]["errors"]["density"]["relative_l2_percent"]
        v_err = results[name]["errors"]["velocity"]["relative_l2_percent"]
        p_err = results[name]["errors"]["pressure"]["relative_l2_percent"]
        steps = results[name]["metadata"]["steps"]
        print(f"{name:<15} {d_err:<12.4f} {v_err:<12.4f} {p_err:<12.4f} {steps:<8}")

    print(f"\nResults saved to {output}/")


if __name__ == "__main__":
    main()
