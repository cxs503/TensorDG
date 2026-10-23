"""Run the TensorDG smooth periodic advection convergence benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg import l2_error, solve_advection


def _reconstruct(coeff, samples_per_cell: int = 5):
    values = coeff.detach().cpu().numpy()
    cells, modes = values.shape
    reference = 2.0 * (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 1.0
    basis = []
    for k in range(modes):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(np.polynomial.legendre.legval(reference, polynomial))
    phi = np.stack(basis, axis=1)
    numerical = (values @ phi.T).reshape(-1)
    x = (np.arange(cells * samples_per_cell) + 0.5) / (cells * samples_per_cell)
    return x, numerical


def _write_results(
    output, degree, speed, final_time, records, coeff,
    max_relative_error=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        passed = (
            records[-1]["relative_l2"] < max_relative_error
            if max_relative_error is not None else None
        )
        json.dump(
            {"benchmark": "linear_advection_1d", "degree": degree,
             "final_time": final_time, "speed": speed, "runs": records,
             "acceptance_threshold": max_relative_error, "passed": passed},
            handle, indent=2,
        )
    cells = np.array([record["ncells"] for record in records])
    errors = np.array([record["absolute_l2"] for record in records])
    np.savetxt(
        output / "convergence.csv",
        np.column_stack((cells, errors, errors * math.sqrt(2.0))),
        delimiter=",", comments="", header="cells,absolute_l2,relative_l2",
    )

    line_x, numerical = _reconstruct(coeff)
    exact = np.sin(2.0 * math.pi * (line_x - speed * final_time))
    np.savetxt(
        output / "profile.csv", np.column_stack((line_x, numerical, exact)),
        delimiter=",", comments="", header="x,numerical,exact",
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(line_x, exact, "k-", linewidth=2.0, label="Exact")
    axes[0].plot(line_x, numerical, "r--", linewidth=1.4, label="DG numerical")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"1D periodic advection | P{degree}, {cells[-1]} cells, t={final_time:g}")
    axes[1].semilogy(line_x, np.maximum(np.abs(numerical - exact), 1e-16))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(output / "advection_1d_profile.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(cells, errors, "o-", label=f"DG P{degree}")
    reference = errors[-1] * (cells / cells[-1]) ** (-(degree + 1))
    ax.loglog(cells, reference, "k--", label=f"Reference slope {degree + 1}")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Absolute L2 error")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Mesh convergence")
    fig.savefig(output / "advection_1d_convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--cells", type=int, nargs="+", default=[20, 40, 80, 160])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-relative-error", type=float, default=None,
        help="If set, require the finest run to be below this relative L2 error.",
    )
    args = parser.parse_args()
    print(f"1D periodic linear advection | DG degree={args.degree} | PyTorch")
    print("cells       L2 error   relative L2       EOC       steps")
    previous_error, previous_h = None, None
    records = []
    centers = coeff = None
    for n in args.cells:
        centers, coeff, meta = solve_advection(
            ncells=n, degree=args.degree, final_time=args.time,
            speed=args.speed, cfl=args.cfl, device=args.device,
        )
        error = l2_error(
            centers, coeff, final_time=args.time, speed=args.speed,
        )
        h = 1.0 / n
        eoc = (
            math.log(previous_error / error) / math.log(previous_h / h)
            if previous_error else None
        )
        relative = error * math.sqrt(2.0)
        eoc_display = eoc if eoc is not None else float("nan")
        print(f"{n:5d}  {error:12.5e}  {relative:12.5e}  {eoc_display:7.3f}  {meta['steps']:7d}")
        records.append({
            "ncells": n, "absolute_l2": error, "relative_l2": relative,
            "relative_l2_percent": 100.0 * relative, "eoc": eoc,
            "steps": meta["steps"], "dt": meta["dt"],
        })
        previous_error, previous_h = error, h

    if args.output_dir:
        _write_results(
            args.output_dir, args.degree, args.speed, args.time,
            records, coeff, args.max_relative_error,
        )
        print(f"results: {args.output_dir}")
    if args.max_relative_error is not None:
        finest = records[-1]
        if finest["relative_l2"] >= args.max_relative_error:
            raise RuntimeError(
                f"Finest relative L2 error exceeds {100.0 * args.max_relative_error:g}%",
            )
        print(f"PASS: finest relative L2 < {100.0 * args.max_relative_error:g}%")


if __name__ == "__main__":
    main()
