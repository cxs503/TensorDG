"""Run and visualize the verified two-dimensional periodic advection case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks._modal_2d import (
    reconstruct, sample_centerline, save_centerline, save_contours,
)
from tensordg import error_2d, solve_advection_2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=24)
    parser.add_argument("--degree", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--velocity", type=float, nargs=2, default=[1.0, 0.5])
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="advection_2d_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    velocity = tuple(args.velocity)
    x, y, coeff, metadata = solve_advection_2d(
        ncells=args.cells, degree=args.degree, final_time=args.time,
        velocity=velocity, cfl=args.cfl, device=args.device,
    )
    metrics = error_2d(x, y, coeff, args.time, velocity)
    xx, yy, numerical = reconstruct(coeff)
    ax, ay = velocity
    exact = 1.0 + np.sin(2.0 * math.pi * (xx - ax * args.time)) * np.sin(
        2.0 * math.pi * (yy - ay * args.time),
    )
    # y=0.25 gives a non-degenerate transported sine profile at the default
    # final time (y=0.5 would make the exact default profile identically one).
    line_x, line_numerical, fixed = sample_centerline(coeff, fixed=0.25)
    line_exact = 1.0 + np.sin(
        2.0 * math.pi * (line_x - ax * args.time),
    ) * np.sin(2.0 * math.pi * (fixed - ay * args.time))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    title = f"2D periodic advection | P{args.degree}, {args.cells}x{args.cells} cells"
    save_contours(output / "advection_2d_contours.png", xx, yy, numerical, exact, title)
    save_centerline(
        output / "advection_2d_centerline.png", line_x, line_numerical,
        line_exact, fixed, "2D periodic advection",
    )
    record = {"benchmark": "linear_advection_2d", **metadata, **metrics}
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
    print(f"steps={metadata['steps']} dt={metadata['dt']:.6e}")
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
