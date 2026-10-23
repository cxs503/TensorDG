"""Run the 2D isentropic-vortex Euler benchmark and compare with its exact solution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks._modal_2d import save_centerline, save_contours
from tensordg.euler_2d import solve_isentropic_vortex, vortex_error, vortex_primitive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=192)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--strength", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="isentropic_vortex_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    x, y, fields, metadata = solve_isentropic_vortex(
        ncells=args.cells, final_time=args.time, cfl=args.cfl,
        strength=args.strength, device=args.device,
    )
    errors = vortex_error(x, y, fields["conservative"], args.time, args.strength)
    xx, yy = np.meshgrid(x.cpu().numpy(), y.cpu().numpy())
    exact = vortex_primitive(
        torch.as_tensor(xx, dtype=x.dtype), torch.as_tensor(yy, dtype=x.dtype),
        args.time, args.strength,
    )
    exact_np = tuple(field.numpy() for field in exact)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_contours(
        output / "vortex_density_contours.png", xx, yy,
        fields["density"].cpu().numpy(), exact_np[0],
        f"Isentropic vortex density | P0, {args.cells}x{args.cells} cells",
    )

    # Cut through the translated vortex center for a nontrivial density curve.
    target_y = float(args.time)
    row = int(torch.argmin(torch.abs(y - target_y)).item())
    line_x = x.cpu().numpy()
    actual_y = float(y[row].item())
    line_exact = vortex_primitive(x, torch.full_like(x, actual_y), args.time, args.strength)[0].numpy()
    line_numerical = fields["density"][row].cpu().numpy()
    save_centerline(
        output / "vortex_density_centerline.png", line_x, line_numerical,
        line_exact, actual_y, "Isentropic vortex density", ylabel=r"Density $\rho$",
    )
    np.savetxt(
        output / "density_centerline.csv",
        np.column_stack((line_x, line_numerical, line_exact)),
        delimiter=",", comments="", header="x,numerical_density,exact_density",
    )
    max_error = max(value["relative_l2"] for value in errors.values())
    record = {
        "benchmark": "isentropic_vortex_2d", **metadata, "errors": errors,
        "max_relative_l2": max_error, "max_relative_l2_percent": 100.0 * max_error,
        "acceptance_threshold": args.max_relative_error,
        "passed": max_error < args.max_relative_error,
        "reference": "analytical isentropic vortex with 4x4 exact cell quadrature",
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"Isentropic vortex: cells={args.cells} steps={metadata['steps']}")
    for key, value in errors.items():
        print(f"{key:10s}: relative L2 = {value['relative_l2_percent']:.6f}%")
    if not record["passed"]:
        raise RuntimeError("An isentropic-vortex field error exceeds the requested threshold")
    print(f"PASS: all field errors < {100.0 * args.max_relative_error:g}%")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
