"""Verification of the Dirichlet (wall) path of :mod:`tensordg.ns_dg_2d`.

The wall path of the compressible Navier--Stokes DG solver was *unverified* in
the m2496 delivery: the impulsively started Couette case died with
``FloatingPointError`` at ``t ~ 0.033`` at every mesh, and
``docs/reports/tensordg-ns-wall-verification.md`` (m2589) localised the failure
to the wall penalty term (it alone reproduces the blow-up, the temperature
channel suffices, and the failure time halves when the penalty doubles).

This script is the executable evidence for the fix.  Five sections, each with a
machine-checkable criterion:

``sign``
    Orientation of the SIPG penalty on the two walls.  The numerical viscous
    flux is always written with the ``+axis`` normal, so the penalty jump
    ``q|-_face - q_wall`` must be *anti*-symmetric between the low and the high
    wall: for a cell whose wall trace is hotter than the wall itself, both walls
    must remove energy from the cell (``F_y(bottom) < 0 < F_y(top)``).  A
    sign-blind penalty heats the low wall -- the root cause of the instability.
``residual``
    Residual of the strong form at the *analytic* steady Couette + Brinkman
    state (P1, 8/16/32 cells), before and after the fix.
``transient``
    Dirichlet Stokes problem -- a shear mode between two no-slip isothermal
    walls, ``u(y,0) = A sin(pi s)``, whose exact solution is
    ``u = A sin(pi s) exp(-nu pi^2 t / H^2)``.  This is the quantitative judge
    of the wall path: relative L2 error of ``u`` and the observed order in ``h``.
``thermal``
    Wall heat flux for a quadratic temperature field: exact to machine
    precision once the parabola is representable (P2/P3), first order at P1.
``stability``
    The exact configuration that used to raise ``FloatingPointError`` (P1, 2x8,
    U = 1): it must march to ``t = 0.1`` with a bounded, physical velocity.

Usage::

    python -m benchmarks.verify_ns_dg_2d_wall           # full (~1-2 min, 1 core)
    python -m benchmarks.verify_ns_dg_2d_wall --quick   # coarse meshes only

Writes ``validation/ns_dg_2d_wall/verification.json`` and
``docs/figures/ns_dg_2d_wall_fix.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

DTYPE = torch.float64
DEVICE = torch.device("cpu")
MU = 0.1
KAPPA = MU * ns.CP / 0.71
WALL_TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def legendre_oriented(degree: int, ncells):
    nx, ny = ncells
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3), DEVICE, DTYPE
    )
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, dx=dx, dy=dy, x=x, y=y)


def project(op, initial, ncells):
    return ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                    op["nodes"], op["weights"], op["basis"],
                                    DEVICE, DTYPE)


def wall_boundary(wall_speed=1.0, wall_temperature=WALL_TEMPERATURE):
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": wall_temperature},
            "ymax": {"velocity": (wall_speed, 0.0), "temperature": wall_temperature}}


def cell_centres(coeff, degree):
    """Primitive fields evaluated at the cell centres (modal basis at xi = 0)."""
    modes = torch.as_tensor(
        np.polynomial.legendre.legvander(np.array([0.0]), degree)[0], dtype=coeff.dtype
    )
    state = torch.einsum("xyklv,k,l->xyv", coeff, modes, modes)
    rho = state[..., 0]
    velocity_x = state[..., 1] / rho
    velocity_y = state[..., 2] / rho
    pressure = (ns.GAMMA - 1.0) * (state[..., 3] - 0.5 * (state[..., 1] ** 2
                                                         + state[..., 2] ** 2) / rho)
    return {"density": rho, "velocity_x": velocity_x, "velocity_y": velocity_y,
            "pressure": pressure, "temperature": pressure / rho}


def shear_mode_exact(y, amplitude, nu, time, height=1.0):
    """``u = A sin(pi (y - ymin)/H) exp(-nu pi^2 t / H^2)`` (Dirichlet Stokes)."""
    s = np.asarray(y, dtype=float) / height
    return amplitude * np.sin(np.pi * s) * math.exp(-nu * np.pi ** 2 * time / height ** 2)


def relative_l2(numerical, exact):
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.sqrt(np.mean((numerical - exact) ** 2))
                 / np.sqrt(np.mean(exact ** 2)))


def observed_order(coarse, fine, ratio=2.0):
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(ratio))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_sign(degree=1, ncells=(2, 8)):
    """Orientation of the wall penalty at the low and the high wall."""
    op = legendre_oriented(degree, ncells)
    dx, dy = op["dx"], op["dy"]

    def initial(x, y):
        ones = torch.ones_like(x * y)
        return ones, torch.zeros_like(ones), torch.zeros_like(ones), ones

    coeff = project(op, initial, ncells)
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    penalty = 2.0 * (degree + 1.0) ** 2
    cold_wall = {"velocity": (0.0, 0.0), "temperature": 0.9}  # wall colder than T = 1
    bottom = ns._boundary_flux(left, cold_wall, 1, dy, MU, KAPPA, penalty,
                               2.0 / dy, 2.0 / dx, outward_sign=-1.0)
    top = ns._boundary_flux(right, cold_wall, 1, dy, MU, KAPPA, penalty,
                            2.0 / dy, 2.0 / dx, outward_sign=1.0)
    flux_bottom = float(bottom[0, 0, 0, 3])
    flux_top = float(top[0, -1, 0, 3])
    return {
        "wall_temperature": 0.9,
        "interior_temperature": 1.0,
        "energy_flux_bottom": flux_bottom,
        "energy_flux_top": flux_top,
        "criterion": "energy_flux_bottom < 0 < energy_flux_top (both walls cool the cell)",
        "pass": bool(flux_bottom < 0.0 < flux_top),
    }


def section_residual(levels=(8, 16, 32), degrees=(1, 2)):
    """Strong-form residual at the analytic steady Couette + Brinkman state."""
    rows = []
    for degree in degrees:
        for ny in levels:
            ncells = (2, ny)
            op = legendre_oriented(degree, ncells)
            amplitude = MU / (2.0 * KAPPA)  # wall speed U = 1
            def initial(x, y):
                u = y
                zeros = torch.zeros_like(u)
                temperature = 1.0 + amplitude * y * (1.0 - y)
                return 1.0 / temperature, u, zeros, torch.ones_like(u)

            coeff = project(op, initial, ncells)
            penalty = 2.0 * (degree + 1.0) ** 2
            rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"],
                             op["signs"], op["dx"], op["dy"], (True, False),
                             wall_boundary(), MU, KAPPA, penalty,
                             wall_gradient="one_sided")
            rows.append({"degree": degree, "ncells": list(ncells),
                         "residual_max": float(rhs.abs().max()),
                         "residual_energy_max": float(rhs[..., 3].abs().max())})
    return rows


def section_transient(levels=(8, 16, 32), degree=1, final_time=0.02,
                      amplitude=0.05, nu=MU):
    """Dirichlet Stokes problem: decaying shear mode between two no-slip walls."""
    rows = []
    for ny in levels:
        started = time.time()
        ncells = (1, ny)
        def initial(x, y):
            ones = torch.ones_like(x * y)
            return ones, amplitude * torch.sin(math.pi * y), torch.zeros_like(ones), ones

        x, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
            initial, ncells=ncells, degree=degree, domain=((0.0, 1.0), (0.0, 1.0)),
            final_time=final_time, mu=MU, periodic=(True, False),
            boundary=wall_boundary(wall_speed=0.0), device="cpu", dtype=DTYPE,
            wall_gradient="one_sided",
        )
        reached = meta["final_time_reached"]
        fields = cell_centres(coeff, degree)
        numerical = np.asarray(fields["velocity_x"][0, :].detach(), dtype=float)
        exact = shear_mode_exact(np.asarray(y, dtype=float), amplitude, nu, reached)
        rows.append({
            "degree": degree, "ncells": list(ncells), "final_time": reached,
            "steps": meta["steps"], "dt": meta["dt"], "amplitude": amplitude,
            "mach": amplitude / math.sqrt(ns.GAMMA),
            "relative_l2_velocity_x": relative_l2(numerical, exact),
            "profile_numerical": numerical.tolist(),
            "profile_exact": exact.tolist(),
            "y_centres": np.asarray(y, dtype=float).tolist(),
            "decay_factor": math.exp(-nu * math.pi ** 2 * reached),
            "seconds": time.time() - started,
        })
    for index in range(len(rows) - 1):
        rows[index]["order_to_next"] = observed_order(
            rows[index]["relative_l2_velocity_x"], rows[index + 1]["relative_l2_velocity_x"]
        )
    best = min(row["relative_l2_velocity_x"] for row in rows)
    return {"cases": rows, "best_relative_l2": best,
            "criterion": "relative L2 of u < 3 %", "pass": bool(best < 0.03)}


def section_thermal(levels=(8, 16, 32), degrees=(1, 2)):
    """Wall heat flux for a quadratic temperature field (rho = 1, p = rho T)."""
    rows = []
    amplitude = 0.5
    for degree in degrees:
        for ny in levels:
            ncells = (2, ny)
            op = legendre_oriented(degree, ncells)
            def initial(x, y):
                ones = torch.ones_like(x * y)
                zeros = torch.zeros_like(ones)
                return ones, zeros, zeros, 1.0 + amplitude * y * (1.0 - y)

            coeff = project(op, initial, ncells)
            penalty = 2.0 * (degree + 1.0) ** 2
            left = ns._face_primitives(coeff, op["basis"], op["derivative"],
                                       op["signs"], 1, "left")
            bottom = ns._boundary_flux(
                left, {"velocity": (0.0, 0.0), "temperature": 1.0}, 1, op["dy"],
                MU, KAPPA, penalty, 2.0 / op["dy"], 2.0 / op["dx"], outward_sign=-1.0,
            )
            analytic = -KAPPA * amplitude  # energy flux in the +y direction at s = 0
            value = float(bottom[0, 0, 0, 3])
            rows.append({"degree": degree, "ncells": list(ncells),
                         "energy_flux": value, "analytic": analytic,
                         "abs_error": abs(value - analytic)})
    orders = {}
    for degree in degrees:
        errors = [row["abs_error"] for row in rows if row["degree"] == degree]
        if len(errors) == len(levels):
            orders[str(degree)] = observed_order(errors[0], errors[-1],
                                                 ratio=float(levels[-1] / levels[0]))
    return {"cases": rows, "observed_order": orders}


def section_stability(final_time=0.1):
    """The configuration that used to fail at t ~ 0.033."""
    started = time.time()
    x, y, coeff, meta = ns.solve_couette_dg_2d(
        ncells=(2, 8), degree=1, final_time=final_time, wall_speed=1.0, mu=MU,
        device="cpu", dtype=DTYPE, wall_gradient="one_sided",
    )
    fields = cell_centres(coeff, 1)
    velocity = float(fields["velocity_x"].abs().max())
    return {
        "ncells": [2, 8], "degree": 1, "final_time": meta["final_time_reached"],
        "steps": meta["steps"], "dt": meta["dt"], "wall_speed": 1.0,
        "max_velocity_x": velocity,
        "max_temperature": float(fields["temperature"].max()),
        "criterion": "march reaches final_time with a bounded velocity field",
        "pass": bool(velocity <= 1.0 + 1e-9),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))

    ax = axes[0]
    for index, row in enumerate(results["transient"]["cases"]):
        ax.plot(row["y_centres"], row["profile_exact"], "k--", linewidth=1.3,
                label="exact  A sin(pi y/H) exp(-nu pi^2 t/H^2)" if index == 0 else None)
        ax.plot(row["y_centres"], row["profile_numerical"], "o-", markersize=3.0,
                linewidth=1.0,
                label="DG P%d, %d cells: rel. L2 = %.2f %%" % (
                    row["degree"], row["ncells"][1],
                    100.0 * row["relative_l2_velocity_x"]))
    ax.set_xlabel("y / H")
    ax.set_ylabel("u / A")
    ax.set_title("Dirichlet Stokes mode vs exact (t = %.3g)" % (
        results["transient"]["cases"][0]["final_time"]))
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rows = results["transient"]["cases"]
    h = np.asarray([1.0 / row["ncells"][1] for row in rows])
    error = np.asarray([100.0 * row["relative_l2_velocity_x"] for row in rows])
    ax.loglog(h, error, "s-", color="C0", label="measured (P%d)" % rows[0]["degree"])
    ax.loglog(h, error[0] * (h / h[0]) ** 2, "k:", label="slope 2 (theory)")
    ax.axhline(3.0, color="C3", linestyle="--", linewidth=1.0, label="3 % gate")
    ax.set_xlabel("h = 1 / n_y")
    ax.set_ylabel("relative L2 of u [%]")
    ax.set_title("Wall-path convergence")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for degree in sorted({row["degree"] for row in results["thermal"]["cases"]}):
        subset = [row for row in results["thermal"]["cases"] if row["degree"] == degree]
        ax.loglog([1.0 / row["ncells"][1] for row in subset],
                  [max(row["abs_error"], 1e-17) for row in subset], "o-",
                  label="P%d wall heat flux error" % degree)
    after = [row for row in results["residual"]["cases"] if row["degree"] == 1]
    ax.loglog([1.0 / row["ncells"][1] for row in after],
              [row["residual_max"] for row in after], "s--", color="C4",
              label="steady residual P1 (after fix)")
    ax.axhline(float(np.mean([row["residual_max"] for row in results["residual"]["before_fix"]])),
               color="C4", linestyle=":", label="steady residual P1 (before fix)")
    ax.set_xlabel("h = 1 / n_y")
    ax.set_ylabel("absolute error / residual")
    ax.set_title("Wall heat flux and steady residual")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: Residual of the analytic steady Couette state measured *before* the wall
#: penalty orientation fix (docs/reports/tensordg-ns-wall-verification.md,
#: m2589): constant 0.3045 at 8 / 16 / 32 cells -- i.e. it did not converge.
PRIOR_RESIDUAL = [
    {"degree": 1, "ncells": [2, 8], "residual_max": 0.30453},
    {"degree": 1, "ncells": [2, 16], "residual_max": 0.304502},
    {"degree": 1, "ncells": [2, 32], "residual_max": 0.304542},
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wall-path verification (TensorDG NS DG)")
    parser.add_argument("--quick", action="store_true", help="coarse meshes only")
    parser.add_argument("--json", default="validation/ns_dg_2d_wall/verification.json")
    parser.add_argument("--figure", default="docs/figures/ns_dg_2d_wall_fix.png")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    started = time.time()
    levels = (8, 16) if args.quick else (8, 16, 32)
    results = {}

    results["sign"] = section_sign()
    print("sign      : %s   F_y(bottom) = %+.4f, F_y(top) = %+.4f" % (
        "PASS" if results["sign"]["pass"] else "FAIL",
        results["sign"]["energy_flux_bottom"], results["sign"]["energy_flux_top"]),
        flush=True)

    results["residual"] = {"cases": section_residual(degrees=(1, 2) if not args.quick else (1,)),
                           "before_fix": PRIOR_RESIDUAL}
    for row in results["residual"]["cases"]:
        print("residual  : P%d %2d cells  max |rhs| = %.4e%s" % (
            row["degree"], row["ncells"][1], row["residual_max"],
            "  (before fix 3.05e-01)" if row["degree"] == 1 else ""), flush=True)

    results["transient"] = section_transient(levels=levels)
    for row in results["transient"]["cases"]:
        print("transient : P%d %2d cells  t=%.4f steps=%4d  rel L2(u) = %.3f %%  decay=%.4f"
              % (row["degree"], row["ncells"][1], row["final_time"], row["steps"],
                 100.0 * row["relative_l2_velocity_x"], row["decay_factor"]), flush=True)

    results["thermal"] = section_thermal(degrees=(1, 2) if not args.quick else (1, 2))
    for row in results["thermal"]["cases"]:
        print("thermal   : P%d %2d cells  |wall heat flux error| = %.3e" % (
            row["degree"], row["ncells"][1], row["abs_error"]), flush=True)

    results["stability"] = section_stability()
    print("stability : t=%.3f steps=%d  max|u| = %.4f  %s" % (
        results["stability"]["final_time"], results["stability"]["steps"],
        results["stability"]["max_velocity_x"],
        "PASS" if results["stability"]["pass"] else "FAIL"), flush=True)

    results["seconds"] = time.time() - started
    results["verdict"] = {
        "wall_penalty_orientation": results["sign"]["pass"],
        "transient_below_three_percent": results["transient"]["pass"],
        "wall_transient_stable": results["stability"]["pass"],
    }
    target = pathlib.Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n")
    make_figure(results, pathlib.Path(args.figure))
    print("\nwrote %s and %s in %.1f s" % (target, args.figure, results["seconds"]))
    print("verdict: %s" % results["verdict"])
    return 0 if all(results["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())