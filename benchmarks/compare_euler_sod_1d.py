"""Compare the Sod P0 result with the exact Riemann solution and plot curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.euler_sod_1d import solve_sod
from benchmarks.sod_exact import cell_averages, sample, star_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=3200)
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="sod_comparison_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()
    if args.time <= 0.0:
        raise ValueError("the exact Riemann comparison requires --time > 0")

    x, fields, metadata = solve_sod(
        ncells=args.cells, final_time=args.time,
        cfl=args.cfl, device=args.device,
    )
    x_np = x.detach().cpu().numpy()
    _, reference = cell_averages(args.cells, args.time)
    errors = {}
    for key in ("density", "velocity", "pressure"):
        numerical = fields[key].detach().cpu().numpy()
        absolute = np.sqrt(np.mean((numerical - reference[key]) ** 2))
        norm = np.sqrt(np.mean(reference[key] ** 2))
        errors[key] = {
            "absolute_l2": float(absolute),
            "relative_l2": float(absolute / norm),
            "relative_l2_percent": float(100.0 * absolute / norm),
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dense_x = np.linspace(0.0, 1.0, 4001)
    dense_exact = sample(dense_x, args.time)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    for ax, key, label in zip(
        axes, ("density", "velocity", "pressure"),
        (r"Density $\rho$", "Velocity $u$", "Pressure $p$"),
    ):
        numerical = fields[key].detach().cpu().numpy()
        ax.plot(dense_x, dense_exact[key], "k-", linewidth=1.8, label="Exact Riemann")
        ax.plot(x_np, numerical, "r-", linewidth=1.0, alpha=0.8, label="DG P0 / FV")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"{label}: relative cell-averaged L2 error = {errors[key]['relative_l2_percent']:.3f}%")
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Sod shock tube | {args.cells} cells, t={args.time:g}")
    fig.savefig(output / "sod_reference_profiles.png", dpi=160)
    plt.close(fig)

    max_error = max(value["relative_l2"] for value in errors.values())
    p_star, u_star = star_state()
    record = {
        "benchmark": "euler_sod_1d", **metadata, "errors": errors,
        "max_relative_l2": max_error,
        "max_relative_l2_percent": 100.0 * max_error,
        "acceptance_threshold": args.max_relative_error,
        "passed": max_error < args.max_relative_error,
        "reference": "exact Sod Riemann solution, 8-point cell averages",
        "star_pressure": p_star, "star_velocity": u_star,
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    data = np.column_stack(
        [x_np]
        + [fields[key].detach().cpu().numpy() for key in ("density", "velocity", "pressure")]
        + [reference[key] for key in ("density", "velocity", "pressure")],
    )
    np.savetxt(
        output / "profiles.csv", data, delimiter=",", comments="",
        header="x,density,velocity,pressure,density_reference,velocity_reference,pressure_reference",
    )

    print(f"Sod exact comparison | cells={args.cells} steps={metadata['steps']}")
    print(f"star state: p*={p_star:.12f}, u*={u_star:.12f}")
    for key, value in errors.items():
        print(f"{key:8s}: relative L2 = {value['relative_l2_percent']:.6f}%")
    print(f"results: {output}")
    if not record["passed"]:
        raise RuntimeError(
            f"Maximum relative L2 error exceeds {100.0 * args.max_relative_error:g}%",
        )
    print(f"PASS: all field errors < {100.0 * args.max_relative_error:g}%")


if __name__ == "__main__":
    main()
