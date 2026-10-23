"""Run and visualize the SIPG/Crank--Nicolson heat-equation benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks._modal_1d import sample, save_profile
from tensordg.heat_1d import exact_temperature, heat_error, solve_heat_1d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--penalty", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="heat_1d_results")
    parser.add_argument("--max-relative-error", type=float, default=0.03)
    args = parser.parse_args()

    x, coeff, metadata = solve_heat_1d(
        args.cells, args.degree, args.time, args.time_steps,
        args.penalty, args.device,
    )
    metrics = heat_error(x, coeff, args.time)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    line_x, numerical = sample(coeff, 801)
    exact = exact_temperature(torch.as_tensor(line_x, dtype=coeff.dtype), args.time).numpy()
    save_profile(output / "heat_1d_profile.png", line_x, numerical, exact,
                 f"Heat equation | P{args.degree}, {args.cells} cells, t={args.time:g}",
                 ylabel="Temperature")
    np.savetxt(output / "profile.csv", np.column_stack((line_x, numerical, exact)),
               delimiter=",", comments="", header="x,numerical,exact")

    # Recompute along one fixed-dt trajectory so the space-time plot does not
    # contain artifacts from independently rounded time-step counts.
    step_rows = np.unique(np.linspace(0, metadata["steps"], 41, dtype=int))
    times = step_rows * metadata["dt"]
    numerical_xt = []
    for step, time in zip(step_rows, times):
        _, history_coeff, _ = solve_heat_1d(
            args.cells, args.degree, float(time), int(step) if step > 0 else None,
            args.penalty, args.device,
        )
        _, values = sample(history_coeff, 201)
        numerical_xt.append(values)
    numerical_xt = np.asarray(numerical_xt)
    # The space-time grid uses the same 201-point sampling as the numerical rows.
    line_xt = np.linspace(0.0, 1.0, 201)
    exact_xt = exact_temperature(
        torch.as_tensor(line_xt[None, :], dtype=coeff.dtype),
        torch.as_tensor(times[:, None], dtype=coeff.dtype),
    ).numpy()
    xx, tt = np.meshgrid(line_xt, times)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    vmin = min(numerical_xt.min(), exact_xt.min())
    vmax = max(numerical_xt.max(), exact_xt.max())
    levels = np.linspace(vmin, vmax, 41)
    for ax, field, title in zip(axes[:2], (numerical_xt, exact_xt), ("DG numerical", "Exact")):
        image = ax.contourf(xx, tt, field, levels=levels, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(image, ax=ax, shrink=0.86)
    image = axes[2].contourf(xx, tt, np.abs(numerical_xt - exact_xt), levels=40, cmap="magma")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(image, ax=axes[2], shrink=0.86)
    fig.suptitle("Heat equation space-time solution")
    fig.savefig(output / "heat_1d_space_time.png", dpi=160)
    plt.close(fig)

    passed = metrics["relative_l2"] < args.max_relative_error
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"benchmark": "heat_1d", **metadata, **metrics,
                   "relative_l2_percent": 100.0 * metrics["relative_l2"],
                   "acceptance_threshold": args.max_relative_error, "passed": passed}, handle, indent=2)
    print(f"Heat P{args.degree}: cells={args.cells} steps={metadata['steps']}")
    print(f"relative L2={100.0 * metrics['relative_l2']:.6f}%")
    if not passed:
        raise RuntimeError("Heat-equation relative L2 error exceeds the requested threshold")
    print(f"PASS: relative L2 < {100.0 * args.max_relative_error:g}%")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
