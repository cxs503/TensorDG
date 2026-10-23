"""Run the smooth nonlinear Burgers DG benchmark with exact characteristics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks._modal_1d import sample, save_convergence, save_profile
from tensordg.burgers_1d import burgers_error, exact_profile, solve_burgers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="burgers_1d_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    records = []
    previous_error = previous_h = None
    for cells in args.cells:
        x, coeff, metadata = solve_burgers(
            cells, args.degree, args.time, args.cfl, args.device,
        )
        metrics = burgers_error(x, coeff, args.time)
        h = 1.0 / cells
        eoc = (
            math.log(previous_error / metrics["relative_l2"]) / math.log(previous_h / h)
            if previous_error else None
        )
        records.append({**metadata, **metrics, "eoc": eoc,
                        "relative_l2_percent": 100.0 * metrics["relative_l2"]})
        previous_error, previous_h = metrics["relative_l2"], h
        print(f"cells={cells:4d} steps={metadata['steps']:5d} relative L2={100.0 * metrics['relative_l2']:.6f}%")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    line_x, numerical = sample(coeff, 801)
    exact = exact_profile(torch.as_tensor(line_x, dtype=coeff.dtype), args.time).numpy()
    save_profile(output / "burgers_1d_profile.png", line_x, numerical, exact,
                 f"Smooth Burgers | P{args.degree}, {args.cells[-1]} cells, t={args.time:g}")
    save_convergence(output / "burgers_1d_convergence.png", args.cells,
                     [record["relative_l2"] for record in records], args.degree)
    np.savetxt(output / "profile.csv", np.column_stack((line_x, numerical, exact)),
               delimiter=",", comments="", header="x,numerical,exact")
    passed = records[-1]["relative_l2"] < args.max_relative_error
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"benchmark": "burgers_1d", "runs": records,
                   "acceptance_threshold": args.max_relative_error, "passed": passed}, handle, indent=2)
    if not passed:
        raise RuntimeError("Burgers relative L2 error exceeds the requested threshold")
    print(f"PASS: finest relative L2 < {100.0 * args.max_relative_error:g}%")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
