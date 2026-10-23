"""Compare the high-order DG Lax shock tube with the exact Riemann solution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import sample, star_state
from tensordg.euler_dg_1d import solve_euler_dg_1d

LEFT = (0.445, 0.698, 3.528)
RIGHT = (0.5, 0.0, 0.571)


def _reconstruct(x, coeff, samples=4):
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
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--degree", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.13)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="lax_1d_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    x, coeff, metadata = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=args.cells, degree=args.degree,
        final_time=args.time, cfl=args.cfl, device=args.device,
    )
    errors = riemann_dg_errors(x, coeff, args.time, LEFT, RIGHT)
    points, numerical = _reconstruct(x, coeff)
    exact = sample(points, args.time, LEFT, RIGHT)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ("density", "velocity", "pressure"),
                              (r"Density $\rho$", "Velocity $u$", "Pressure $p$")):
        ax.plot(points, exact[key], "k-", linewidth=1.8, label="Exact Riemann")
        ax.plot(points, numerical[key], "r-", linewidth=1.0, alpha=0.8,
                label=f"DG P{args.degree} / HLLC")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: relative L2 error = {errors[key]['relative_l2_percent']:.3f}%")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Lax shock tube | {args.cells} cells, t={args.time:g}")
    fig.savefig(output / "lax_reference_profiles.png", dpi=160)
    plt.close(fig)
    np.savetxt(
        output / "profiles.csv",
        np.column_stack([points]
                        + [numerical[key] for key in ("density", "velocity", "pressure")]
                        + [exact[key] for key in ("density", "velocity", "pressure")]),
        delimiter=",", comments="",
        header="x,density,velocity,pressure,density_exact,velocity_exact,pressure_exact",
    )
    max_error = max(value["relative_l2"] for value in errors.values())
    p_star, u_star = star_state(LEFT, RIGHT)
    record = {
        "benchmark": "lax_1d", **metadata, "errors": errors,
        "max_relative_l2": max_error, "max_relative_l2_percent": 100.0 * max_error,
        "acceptance_threshold": args.max_relative_error,
        "passed": max_error < args.max_relative_error,
        "reference": "exact general Riemann solution with 8-point quadrature",
        "star_pressure": p_star, "star_velocity": u_star,
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"Lax DG P{args.degree}: cells={args.cells} steps={metadata['steps']}")
    print(f"star state: p*={p_star:.12f}, u*={u_star:.12f}")
    for key, value in errors.items():
        print(f"{key:8s}: relative L2 = {value['relative_l2_percent']:.6f}%")
    if not record["passed"]:
        raise RuntimeError("A Lax primitive-field error exceeds the requested threshold")
    print(f"PASS: all field errors < {100.0 * args.max_relative_error:g}%")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
