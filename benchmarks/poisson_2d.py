"""Run and visualize the manufactured two-dimensional SIPG Poisson case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks._modal_2d import (
    reconstruct, sample_centerline, save_centerline, save_contours,
)
from tensordg import poisson_error_2d, solve_poisson_2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--degree", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--penalty", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="poisson_2d_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    x, y, coeff, metadata = solve_poisson_2d(
        ncells=args.cells, degree=args.degree,
        penalty=args.penalty, device=args.device,
    )
    metrics = poisson_error_2d(x, y, coeff)
    xx, yy, numerical = reconstruct(coeff)
    exact = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    line_x, line_numerical, fixed = sample_centerline(coeff)
    line_exact = np.sin(math.pi * line_x) * np.sin(math.pi * fixed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    title = f"2D Poisson SIPG | P{args.degree}, {args.cells}x{args.cells} cells"
    save_contours(output / "poisson_2d_contours.png", xx, yy, numerical, exact, title)
    save_centerline(
        output / "poisson_2d_centerline.png", line_x, line_numerical,
        line_exact, fixed, "2D Poisson",
    )
    record = {"benchmark": "poisson_2d", **metadata, **metrics}
    record["relative_l2_percent"] = 100.0 * metrics["relative_l2"]
    record["acceptance_threshold"] = args.max_relative_error
    record["passed"] = metrics["relative_l2"] < args.max_relative_error
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    np.savetxt(
        output / "centerline.csv",
        np.column_stack((line_x, line_numerical, line_exact)),
        delimiter=",", header="x,numerical,exact", comments="",
    )

    print(title)
    print(f"DOFs={metadata['dofs']} matrix symmetry error={metadata['matrix_symmetry_error']:.3e}")
    print(f"absolute L2={metrics['absolute_l2']:.8e}")
    print(f"relative L2={100.0 * metrics['relative_l2']:.6f}%")
    print(f"results: {output}")
    if not record["passed"]:
        raise RuntimeError(
            f"Relative L2 error exceeds {100.0 * args.max_relative_error:g}%",
        )
    print(f"PASS: relative L2 < {100.0 * args.max_relative_error:g}%")


if __name__ == "__main__":
    main()
