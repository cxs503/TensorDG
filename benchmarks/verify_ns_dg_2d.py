"""Verification driver for the compressible Navier--Stokes DG module.

Runs the three verified cases, writes machine-readable numbers to
``validation/ns_dg_2d/`` and the report figure to
``docs/figures/ns_dg_2d_shear_wave.png``.  No number is typed by hand: this
script is the only source of the numbers quoted in the report.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_ns_dg_2d
"""
import json
import os

import numpy as np
import torch

from tensordg.euler_2d import vortex_primitive
from tensordg.euler_dg_2d import legendre_operators, solve_euler_dg_2d_vortex
import tensordg.ns_dg_2d as ns  # noqa: F401  (public module under test)
from tensordg.ns_dg_2d import (
    _initial_coefficients,
    _rhs_ns,
    heat_conduction_exact,
    shear_wave_exact,
    solve_navier_stokes_dg_2d,
    solve_shear_wave_dg_2d,
)

AMPLITUDE = 0.01
WAVENUMBER = 2.0 * np.pi
MU = 0.05
FINAL_TIME = 0.1
CP = 1.4 / 0.4
KAPPA_CONDUCTION = 0.1 * CP / 0.71

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "validation", "ns_dg_2d")
FIG = os.path.join(ROOT, "docs", "figures")


def _operators(degree):
    return legendre_operators(degree, max(degree + 2, 3))


def _grid(ncells, degree):
    nodes, weights, basis, derivative, signs = _operators(degree)
    nx, ny = ncells
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, x=x, y=y, dx=dx, dy=dy, nx=nx, ny=ny)


def _quadrature_points(grid):
    nodes = grid["nodes"]
    nx, ny = grid["nx"], grid["ny"]
    xq = grid["x"][:, None, None, None] + 0.5 * grid["dx"] * nodes[None, None, :, None]
    yq = grid["y"][None, :, None, None] + 0.5 * grid["dy"] * nodes[None, None, None, :]
    shape = (nx, ny, nodes.numel(), nodes.numel())
    return xq.expand(*shape).clone(), yq.expand(*shape).clone()


def _nodal(coeff, grid):
    return torch.einsum("xyklv,ak,bl->xyabv", coeff, grid["basis"], grid["basis"])


def shear_wave_case(ncells, degree):
    _, _, coeff, metadata = solve_shear_wave_dg_2d(
        ncells=ncells, degree=degree, final_time=FINAL_TIME,
        amplitude=AMPLITUDE, wavenumber=WAVENUMBER, mu=MU,
    )
    grid = _grid(ncells, degree)
    xq, yq = _quadrature_points(grid)
    nodal = _nodal(coeff, grid)
    exact = shear_wave_exact(
        xq, yq, FINAL_TIME, AMPLITUDE, WAVENUMBER, MU, 1.0
    )["velocity_x"].expand_as(nodal[..., 0])
    numerical = nodal[..., 1] / nodal[..., 0]
    error = float((numerical - exact).pow(2).sum().sqrt() / exact.pow(2).sum().sqrt())
    # cell-centre profile for the figure
    centres = nodal[:, :, grid["nodes"].numel() // 2, grid["nodes"].numel() // 2, 1]
    centres = centres / nodal[:, :, grid["nodes"].numel() // 2,
                              grid["nodes"].numel() // 2, 0]
    y_centres = grid["y"].numpy()
    exact_centres = (AMPLITUDE * np.sin(WAVENUMBER * y_centres)
                     * np.exp(-MU * WAVENUMBER**2 * FINAL_TIME))
    return {
        "ncells": list(ncells),
        "degree": degree,
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "mu": metadata["mu"],
        "kappa": metadata["kappa"],
        "relative_l2_velocity_x": error,
        "profile_numerical": np.asarray(centres[0, :]).tolist(),
        "profile_exact": np.asarray(exact_centres).tolist(),
        "y_centres": y_centres.tolist(),
    }


def inviscid_limit_case():
    inviscid = solve_euler_dg_2d_vortex(ncells=6, degree=1, final_time=0.05, cfl=0.2)

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))

    viscous = solve_navier_stokes_dg_2d(
        initial, ncells=6, degree=1, domain=((-5.0, 5.0), (-5.0, 5.0)),
        final_time=0.05, cfl=0.2, mu=0.0,
    )
    reference = inviscid[2]
    scale = float(reference.abs().max())
    return {
        "relative_max_difference": float((reference - viscous[2]).abs().max()) / scale,
        "euler_steps": inviscid[3]["steps"],
        "navier_stokes_steps": viscous[3]["steps"],
        "scale": scale,
    }


def static_wall_case():
    ncells, degree = (2, 8), 1
    bottom, top, pressure = 1.0, 0.5, 1.0
    grid = _grid(ncells, degree)

    def state(x, y):
        exact = heat_conduction_exact(y, 0.0, 1.0, bottom, top, pressure)
        return (exact["density"], exact["velocity_x"], exact["velocity_y"],
                exact["pressure"])

    coeff = _initial_coefficients(
        state, grid["x"], grid["y"], grid["dx"], grid["dy"],
        grid["nodes"], grid["weights"], grid["basis"],
        torch.device("cpu"), torch.float64,
    )
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": bottom},
        "ymax": {"velocity": (0.0, 0.0), "temperature": top},
    }
    residual = _nodal(
        _rhs_ns(coeff, grid["basis"], grid["derivative"], grid["weights"],
                grid["signs"], grid["dx"], grid["dy"], (False, False), boundary,
                0.1, KAPPA_CONDUCTION, 2.0 * (degree + 1.0) ** 2),
        grid,
    )
    wall_flux = KAPPA_CONDUCTION * (bottom - top)
    return {
        "x_momentum_residual": float(residual[..., 1].abs().max()),
        "y_momentum_residual": float(residual[..., 2].abs().max()),
        "mass_residual": float(residual[..., 0].abs().max()),
        "energy_residual": float(residual[..., 3].abs().max()),
        "wall_conduction_flux": wall_flux,
        "energy_over_wall_flux": float(residual[..., 3].abs().max()) / wall_flux,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    inviscid = inviscid_limit_case()
    print("inviscid limit:", inviscid, flush=True)

    shear = []
    for degree, sizes in ((1, [(2, 6), (2, 8), (2, 12), (2, 16)]), (2, [(2, 6)])):
        for ncells in sizes:
            result = shear_wave_case(ncells, degree)
            shear.append(result)
            print("shear wave P%d %s -> rel L2 %.6e (steps %d)" % (
                degree, ncells, result["relative_l2_velocity_x"], result["steps"]),
                flush=True)

    orders = []
    p1 = [r for r in shear if r["degree"] == 1]
    for coarse, fine in zip(p1[:-1], p1[1:]):
        ratio = np.log(coarse["relative_l2_velocity_x"] / fine["relative_l2_velocity_x"])
        ratio /= np.log(fine["ncells"][1] / coarse["ncells"][1])
        orders.append({
            "between": [coarse["ncells"][1], fine["ncells"][1]],
            "observed_order": float(ratio),
            "design_order": 2.0,
        })
        print("order %s -> %.3f" % (orders[-1]["between"], ratio), flush=True)

    static = static_wall_case()
    print("static wall:", static, flush=True)

    summary = {
        "inviscid_limit": inviscid,
        "shear_wave": shear,
        "observed_orders_p1": orders,
        "static_wall": static,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = [r for r in p1 if r["ncells"][1] == 16][0]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    axes[0].plot(reference["profile_exact"], reference["y_centres"], "k-", lw=2.0,
                 label="exact  $u=A\\sin(ky)e^{-\\nu k^2 t}$")
    axes[0].plot(reference["profile_numerical"], reference["y_centres"], "o--",
                 ms=4, label="DG P1, 16 cells")
    axes[0].set_xlabel("$u$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("decaying shear wave, $t=%.2f$" % FINAL_TIME)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    sizes = np.array([r["ncells"][1] for r in p1], dtype=float)
    errors = np.array([r["relative_l2_velocity_x"] for r in p1], dtype=float)
    axes[1].loglog(sizes, errors, "s-", label="DG P1 measured")
    reference_curve = errors[0] * (sizes[0] / sizes) ** 2
    axes[1].loglog(sizes, reference_curve, "k--", label="slope 2 (design order)")
    axes[1].set_xlabel("cells in $y$")
    axes[1].set_ylabel("relative $L_2$ error of $u$")
    axes[1].set_title("viscous operator convergence")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    amplitude = AMPLITUDE
    times = np.linspace(0.0, 0.5, 200)
    axes[2].plot(times, amplitude * np.exp(-MU * WAVENUMBER**2 * times), "k-",
                 label="analytic amplitude")
    decayed = [r["relative_l2_velocity_x"] for r in p1]
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("peak $u$")
    axes[2].set_title("viscous decay rate $\\nu k^2=%.2f$" % (MU * WAVENUMBER**2))
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].text(0.03, 0.55,
                 "errors: " + ", ".join("%.2e" % value for value in decayed),
                 fontsize=7, transform=axes[2].transAxes)

    figure.tight_layout()
    path = os.path.join(FIG, "ns_dg_2d_shear_wave.png")
    figure.savefig(path, dpi=150)
    print("figure ->", path, flush=True)
    print("summary ->", os.path.join(OUT, "summary.json"), flush=True)


if __name__ == "__main__":
    main()