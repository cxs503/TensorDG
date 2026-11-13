"""2-D axisymmetric acoustic pulse: DG convergence against an exact solution.

Case
----
A Gaussian pressure pulse ``p(r,0) = exp(-r^2/sigma^2)`` in a fluid at rest
(``rho = c = 1`` unless overridden), marched with the modal DG solver
:func:`tensordg.solve_acoustic_2d` on a periodic box that is large enough for
the pulse never to see the boundary (checked by a box-doubling control run).
The exact solution is the Hankel (Bessel ``J0``) integral of
:func:`tensordg.acoustic_pulse_exact_pressure`; it is *independent* of the
solver: its quadrature is verified by refinement and by a real-space Poisson
disk integral in ``tests/test_acoustic_pulse.py``.

What this benchmark decides
---------------------------
1. Does the DG scheme converge at the design order ``p + 1`` in the relative
   ``L2`` norm of the full state ``(p, u, v)``?
2. Is the acoustic energy ``int [p^2/(2K) + rho |u|^2/2] dA`` preserved to the
   analytic initial value ``pi sigma^2 A^2 / (4 K)``?  (Independent of the
   exact solution -- a conservation check, not an accuracy check.)
3. How do the fields compare, pointwise, with the exact solution?

Pre-declared gates (set before the runs; see docs/CASE-ACOUSTIC-PULSE.md)
------------------------------------------------------------------------
* observed order on every consecutive pair ``>= p + 1 - 0.4`` -- the primary
  verification statement (the scheme must show its design order);
* at the finest resolution of the *production* ladder (default
  ``--resolutions 16 32 64``): relative state error ``<= 1e-2`` and relative
  energy drift ``|E(T) - E0|/E0 <= 1e-2`` -- i.e. an order of magnitude better
  than the 15% error bar that the domain considers unacceptable;
* ``--quick`` (the CI ladder, ``P1/P2`` on ``16/32``) asserts the order gate
  only: coarse grids legitimately dissipate the unresolved part of a
  ``sigma/h < 2`` Gaussian, so an absolute bar there would measure the grid,
  not the scheme.

Usage
-----
    python -m benchmarks.acoustic_pulse_2d --quick
    python -m benchmarks.acoustic_pulse_2d --degrees 1 2 3 --resolutions 16 32 48 \\
        --output-dir validation/acoustic_pulse --figures-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg import (
    AcousticMedium,
    acoustic_pulse_exact_state,
    acoustic_pulse_initial,
    solve_acoustic_2d,
)

# ---- pre-declared acceptance gates (do not tune after seeing results) ------
ORDER_SLACK = 0.4
ENERGY_DRIFT_TOL = 1.0e-2
ERROR_TOL = 1.0e-2
DEFAULT_RESOLUTIONS = (16, 32, 64)


def _legendre_values(points: np.ndarray, degree: int) -> np.ndarray:
    values = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        values.append(np.polynomial.legendre.legval(points, polynomial))
    return np.stack(values, axis=-1)


def reconstruct(coeff: torch.Tensor, length: float, samples_per_cell: int = 6):
    """Evaluate the modal field on a uniform plotting mesh of ``[0, L]^2``."""
    values = coeff.detach().cpu().numpy()
    nx, ny, modes, _ = values.shape
    reference = 2.0 * (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 1.0
    phi = _legendre_values(reference, modes - 1)
    cell_values = np.einsum("xykl,ak,bl->xyab", values, phi, phi)
    field = cell_values.transpose(1, 3, 0, 2).reshape(ny * samples_per_cell,
                                                      nx * samples_per_cell)
    x = length * (np.arange(nx * samples_per_cell) + 0.5) / (nx * samples_per_cell)
    y = length * (np.arange(ny * samples_per_cell) + 0.5) / (ny * samples_per_cell)
    return np.meshgrid(x, y), field


def sample_line(coeff: torch.Tensor, length: float, fixed: float,
                points: int = 601, component: int = 0):
    """Sample one component along ``y = fixed`` (physical coordinates)."""
    values = coeff.detach().cpu().numpy()
    if values.ndim == 5:                       # (nx, ny, k, l, components)
        values = values[..., component]
    nx, ny, modes, _ = values.shape
    x = np.linspace(0.0, length, points)
    y = np.full_like(x, fixed)
    ix = np.minimum((x / length * nx).astype(int), nx - 1)
    iy = np.minimum((y / length * ny).astype(int), ny - 1)
    xi = 2.0 * (x / length * nx - ix) - 1.0
    eta = 2.0 * (y / length * ny - iy) - 1.0
    phi_x = _legendre_values(xi, modes - 1)
    phi_y = _legendre_values(eta, modes - 1)
    numerical = np.einsum("nkl,nk,nl->n", values[ix, iy], phi_x, phi_y)
    return x, numerical


def run_point(resolution: int, degree: int, *, final_time: float, sigma: float,
              length: float, cfl: float, medium: AcousticMedium) -> dict:
    """One (resolution, degree) run of the pulse and its error metrics."""
    half = 0.5 * length

    def initial_state(x, y):
        return acoustic_pulse_initial(x - half, y - half, sigma=sigma)

    start = time.time()
    result = solve_acoustic_2d(initial_state, nx=resolution, degree=degree,
                               final_time=final_time, length_x=length,
                               length_y=length, medium=medium, cfl=cfl,
                               snapshot_times=(0.0,))
    elapsed = time.time() - start

    x, y = result.quadrature_grid()
    exact = acoustic_pulse_exact_state(x - half, y - half, final_time,
                                       sigma=sigma, medium=medium)
    energy0 = float(result.energy(result.snapshots[0.0]))
    energy = float(result.energy())
    return {
        "degree": degree,
        "cells": resolution,
        "h": length / resolution,
        "resource_ls": resolution * resolution,
        "steps": result.n_steps,
        "dt": result.dt,
        "wall_time_s": round(elapsed, 2),
        "relative_l2_state": result.relative_l2_error(exact, component=None),
        "relative_l2_pressure": result.relative_l2_error(exact, component=0),
        "energy": energy,
        "energy0": energy0,
        "energy_drift": abs(energy - energy0) / energy0,
        "p_max_numerical": float(result.pressure().abs().max()),
        "p_max_exact": float(exact[..., 0].abs().max()),
        "_result": result,
    }


def observed_orders(rows: list[dict]) -> list[dict]:
    orders = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        ratio = coarse["h"] / fine["h"]
        order = math.log(coarse["relative_l2_state"] / fine["relative_l2_state"]) \
            / math.log(ratio)
        orders.append({
            "degree": fine["degree"],
            "from_cells": coarse["cells"],
            "to_cells": fine["cells"],
            "refinement_ratio": ratio,
            "observed_order": order,
            "target_order": fine["degree"] + 1,
            "pass": order >= fine["degree"] + 1 - ORDER_SLACK,
        })
    return orders


def plot_convergence(rows_by_degree: dict[int, list[dict]], path: Path,
                     final_time: float, sigma: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colours = {0: "0.35", 1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    for degree, rows in sorted(rows_by_degree.items()):
        h = np.array([row["h"] for row in rows])
        err = np.array([row["relative_l2_state"] for row in rows])
        ax.loglog(h, err, "o-", color=colours.get(degree, "C4"),
                  label=f"DG P{degree}")
        anchor = err[0] * (h / h[0]) ** (degree + 1)
        ax.loglog(h, anchor, "--", color=colours.get(degree, "C4"),
                  alpha=0.45, lw=1.0)
        ax.annotate(f"slope {degree + 1}", xy=(h[-1], anchor[-1]),
                    xytext=(3, -10), textcoords="offset points",
                    fontsize=8, color=colours.get(degree, "C4"))
    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"relative $L_2$ error of $(p,u,v)$")
    ax.set_title(f"2-D acoustic pulse, t={final_time:g}, "
                 rf"$\sigma$={sigma:g}: order $p+1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fields(result, *, length: float, final_time: float, sigma: float,
                medium: AcousticMedium, degree: int, cells: int, path: Path) -> dict:
    half = 0.5 * length
    (xx, yy), numerical = reconstruct(result.coeff[..., 0], length)
    exact = acoustic_pulse_exact_state(torch.as_tensor(xx) - half,
                                       torch.as_tensor(yy) - half,
                                       final_time, sigma=sigma, medium=medium)
    exact = exact[..., 0].numpy()
    diff = numerical - exact
    scale = float(np.abs(exact).max())

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
    for ax, field, title in (
        (axes[0, 0], numerical, f"DG P{degree}, {cells}$^2$ cells"),
        (axes[0, 1], exact, "exact (Hankel integral)"),
    ):
        image = ax.pcolormesh(xx, yy, field, shading="auto", cmap="RdBu_r",
                              vmin=-scale, vmax=scale)
        ax.set_title(f"pressure {title}", fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046)
    image = axes[1, 0].pcolormesh(xx, yy, diff, shading="auto", cmap="RdBu_r",
                                  vmin=-scale, vmax=scale)
    axes[1, 0].set_title(f"numerical - exact (max |diff| = "
                         f"{np.abs(diff).max():.2e})", fontsize=10)
    axes[1, 0].set_aspect("equal")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    line_x, line_p = sample_line(result.coeff, length, fixed=half, component=0)
    _, line_u = sample_line(result.coeff, length, fixed=half, component=1)
    exact_line = acoustic_pulse_exact_state(torch.as_tensor(line_x) - half,
                                            torch.zeros_like(torch.as_tensor(line_x)),
                                            final_time, sigma=sigma, medium=medium)
    ax = axes[1, 1]
    ax.plot(line_x - half, line_p, "-", color="C0", label="DG p")
    ax.plot(line_x - half, exact_line[..., 0].numpy(), "--", color="C0",
            label="exact p")
    ax.plot(line_x - half, line_u, "-", color="C1", label=r"DG $u_r$")
    ax.plot(line_x - half, exact_line[..., 1].numpy(), "--", color="C1",
            label=r"exact $u_r$")
    ax.set_xlabel("x (y = 0)")
    ax.set_ylabel("p, u")
    ax.set_title("centred profiles at final time", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"field_max_abs_error": float(np.abs(diff).max()),
            "field_scale": scale}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--final-time", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--sound-speed", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="validation/acoustic_pulse")
    parser.add_argument("--figures-dir", default="docs/figures")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="degrees 1 2 and resolutions 16 32, no figures")
    args = parser.parse_args(argv)

    absolute_gates = True
    if args.quick:
        args.degrees = [1, 2]
        args.resolutions = [16, 32]
        args.no_figures = True
        absolute_gates = False
    torch.set_num_threads(max(1, args.threads))
    medium = AcousticMedium(rho=args.density, c=args.sound_speed)

    rows_by_degree: dict[int, list[dict]] = {}
    orders: list[dict] = []
    best = None
    for degree in args.degrees:
        rows = []
        for resolution in sorted(args.resolutions):
            row = run_point(resolution, degree, final_time=args.final_time,
                            sigma=args.sigma, length=args.length, cfl=args.cfl,
                            medium=medium)
            result = row.pop("_result")
            rows.append(row)
            print(f"P{degree} {resolution:>3}^2 cells: rel-L2 state="
                  f"{row['relative_l2_state']:.3e} pressure="
                  f"{row['relative_l2_pressure']:.3e} energy drift="
                  f"{row['energy_drift']:.2e} ({row['wall_time_s']:.1f}s)",
                  flush=True)
            best = result
        rows_by_degree[degree] = rows
        orders.extend(observed_orders(rows))
        for item in orders:
            if item["degree"] == degree:
                print(f"  P{degree} order {item['from_cells']}->"
                      f"{item['to_cells']}: {item['observed_order']:.2f} "
                      f"(target {item['target_order']}) "
                      f"{'OK' if item['pass'] else 'FAIL'}", flush=True)

    finest = {degree: rows[-1] for degree, rows in rows_by_degree.items()}
    gates = {
        "orders": {"pass": all(item["pass"] for item in orders)},
        "energy": {"pass": (all(row["energy_drift"] <= ENERGY_DRIFT_TOL
                                for row in finest.values())
                            if absolute_gates else True),
                   "tolerance": ENERGY_DRIFT_TOL,
                   "applied": absolute_gates},
        "error": {"pass": (all(row["relative_l2_state"] <= ERROR_TOL
                               for row in finest.values())
                           if absolute_gates else True),
                  "tolerance": ERROR_TOL,
                  "applied": absolute_gates},
    }
    passed = all(gate["pass"] for gate in gates.values())

    figure_metrics = {}
    figures = []
    if not args.no_figures:
        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_convergence(rows_by_degree, figures_dir / "acoustic_pulse_convergence.png",
                         args.final_time, args.sigma)
        figures.append(str(figures_dir / "acoustic_pulse_convergence.png"))
        best_row = max(finest.values(), key=lambda row: row["degree"])
        figure_metrics = plot_fields(
            best, length=args.length, final_time=args.final_time,
            sigma=args.sigma, medium=medium, degree=best_row["degree"],
            cells=best_row["cells"],
            path=figures_dir / "acoustic_pulse_exact.png")
        figures.append(str(figures_dir / "acoustic_pulse_exact.png"))

    energy_exact = math.pi * args.sigma ** 2 / (4.0 * medium.bulk_modulus)
    record = {
        "benchmark": "acoustic_pulse_2d",
        "case": "axisymmetric_gaussian_pulse",
        "governing_system": "linear acoustics (p, u, v)",
        "medium": {"rho": medium.rho, "c": medium.c, "bulk_modulus": medium.bulk_modulus},
        "sigma": args.sigma,
        "final_time": args.final_time,
        "length": args.length,
        "cfl": args.cfl,
        "energy_exact_initial": energy_exact,
        "rows": [row for degree in sorted(rows_by_degree)
                 for row in rows_by_degree[degree]],
        "orders": orders,
        "gates": gates,
        "figure_metrics": figure_metrics,
        "figures": figures,
        "passed": passed,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "convergence.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    for degree, row in sorted(finest.items()):
        print(f"P{degree} finest: energy drift {row['energy_drift']:.3e} "
              f"(analytic E0={energy_exact:.6e}, got {row['energy']:.6e})")
    print("gates:", {name: gate["pass"] for name, gate in gates.items()},
          "->", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
