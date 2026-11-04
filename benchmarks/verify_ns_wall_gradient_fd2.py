"""Verification of the Dirichlet-wall channel of ``tensordg.ns_dg_2d``.

Reproduces every number of ``docs/reports/tensordg-ns-wall-gradient.md`` and
writes ``validation/ns_dg_2d_wall_gradient/summary.json`` plus
``docs/figures/ns_dg_2d_wall_gradient_fd2.png``.

Sections
--------
``flux``      wall heat-flux / traction error of the analytic steady Couette
              state for the legacy interior-only wall gradient (``br1``) and the
              second-order one-sided stencil (``fd2``), degrees 1 and 2;
``residual``  strong-form residual of the *projected* analytic steady state
              (shows why that state is not the fixed point of the scheme);
``transient`` impulsively started Couette against the closed-form Fourier
              series u(y, t) = U y/H + (2U/pi) sum (-1)^n/n sin(n pi y/H)
              exp(-nu n^2 pi^2 t / H^2), i.e. an *analytic* benchmark for the
              wall channel -- the case that used to blow up at t ~ 0.033;
``stability`` marches several (degree, mesh) combinations to t = 0.1 and
              reports whether the state stayed physical.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_ns_wall_gradient_fd2 [section ...]
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import torch

from tensordg import ns_dg_2d as ns

MU = 0.1          # dynamic viscosity of the Couette benchmarks
PRANDTL = 0.71
WALL_SPEED = 1.0
WALL_TEMPERATURE = 1.0
KAPPA = MU * ns.CP / PRANDTL
ANALYTIC_ENERGY_FLUX = -MU * WALL_SPEED**2 / 2.0   # -kappa dT/dy at the wall
BRINKMAN_AMPLITUDE = MU * WALL_SPEED**2 / (2.0 * KAPPA)

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "validation" / "ns_dg_2d_wall_gradient" / "summary.json"
FIGURE_PATH = ROOT / "docs" / "figures" / "ns_dg_2d_wall_gradient_fd2.png"


def couette_projected(ncells=(2, 8), degree=1):
    """The analytic steady Couette state projected onto the DG space.

    ``u = U y/H``, ``p = 1``, ``rho = 1/T`` with the Brinkman temperature
    ``T = T_w + A y (1 - y)``, ``A = mu U^2 / (2 kappa)``.
    """
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, max(degree + 2, 3), "cpu", torch.float64
    )
    nx, ny = ncells
    dx = 1.0 / nx
    dy = 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy

    def initial(x, y):
        # the quadrature grids broadcast; adding 0*(x+y) gives every component
        # the full (x, y, xi, eta) shape
        fill = torch.zeros_like(x + y)
        temperature = WALL_TEMPERATURE + BRINKMAN_AMPLITUDE * y * (1.0 - y) + fill
        return (
            1.0 / temperature,
            WALL_SPEED * y + fill,
            fill.clone(),
            torch.ones_like(x + y),
        )

    coeff = ns._initial_coefficients(
        initial, x, y, dx, dy, nodes, weights, basis, "cpu", torch.float64
    )
    boundary = {
        "xmin": None,
        "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": WALL_TEMPERATURE},
        "ymax": {"velocity": (WALL_SPEED, 0.0), "temperature": WALL_TEMPERATURE},
    }
    grid = {
        "basis": basis,
        "derivative": derivative,
        "weights": weights,
        "signs": signs,
        "dx": dx,
        "dy": dy,
        "x": x,
        "y": y,
        "nx": nx,
        "ny": ny,
    }
    return grid, coeff, boundary


def wall_fluxes(ncells=(2, 8), degree=1, wall_gradient="fd2"):
    """Total (traction, heat flux) on both walls of the projected Couette state."""
    grid, coeff, boundary = couette_projected(ncells, degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    h = grid["dy"]
    right = ns._face_primitives(
        coeff, grid["basis"], grid["derivative"], grid["signs"], 1, "right"
    )
    left = ns._face_primitives(
        coeff, grid["basis"], grid["derivative"], grid["signs"], 1, "left"
    )
    centre = None
    wall_penalty = penalty
    if wall_gradient == "fd2":
        centre_u, centre_v, centre_t = ns._cell_centre_primitives(coeff)
        centre = {"u": centre_u, "v": centre_v, "T": centre_t}
        wall_penalty = ns.WALL_PENALTY
    kwargs = dict(normal_scale=2.0 / h, tangential_scale=2.0 / grid["dx"])
    top = ns._boundary_flux(
        right, boundary["ymax"], 1, h, MU, KAPPA, wall_penalty, centre=centre,
        side_name="right", **kwargs
    )
    bottom = ns._boundary_flux(
        left, boundary["ymin"], 1, h, MU, KAPPA, wall_penalty, centre=centre,
        side_name="left", **kwargs
    )
    return {
        "nx_momentum_top": float(top[0, -1, 0, 1]),
        "nx_momentum_bottom": float(bottom[0, 0, 0, 1]),
        "energy_top": float(top[0, -1, 0, 3]),
        "energy_bottom": float(bottom[0, 0, 0, 3]),
    }


def section_flux():
    rows = []
    for wall_gradient in ("br1", "fd2"):
        for degree, meshes in ((1, (8, 16, 32, 64)), (2, (4, 8, 16, 32))):
            errors = []
            for ny in meshes:
                flux = wall_fluxes((2, ny), degree, wall_gradient)
                errors.append(
                    {
                        "ncells_y": ny,
                        "energy_top": flux["energy_top"],
                        "energy_bottom": flux["energy_bottom"],
                        "energy_error_top": abs(flux["energy_top"] - ANALYTIC_ENERGY_FLUX),
                        "energy_error_bottom": abs(
                            flux["energy_bottom"] - ANALYTIC_ENERGY_FLUX
                        ),
                        "momentum_top": flux["nx_momentum_top"],
                        "momentum_error_top": abs(flux["nx_momentum_top"] + MU * WALL_SPEED),
                    }
                )
            bottom = [row["energy_error_bottom"] for row in errors]
            rows.append(
                {
                    "wall_gradient": wall_gradient,
                    "degree": degree,
                    "meshes": errors,
                    "observed_order_bottom": math.log2(bottom[0] / bottom[-1])
                    / math.log2(meshes[-1] / meshes[0]),
                }
            )
    return {
        "analytic_energy_flux": ANALYTIC_ENERGY_FLUX,
        "analytic_momentum_flux": -MU * WALL_SPEED,
        "cases": rows,
    }


def section_residual():
    rows = []
    for wall_gradient in ("br1", "fd2"):
        for degree, meshes in ((1, (8, 16, 32, 64)), (2, (4, 8, 16, 32))):
            for ny in meshes:
                grid, coeff, boundary = couette_projected((2, ny), degree)
                rhs = ns._rhs_ns(
                    coeff, grid["basis"], grid["derivative"], grid["weights"],
                    grid["signs"], grid["dx"], grid["dy"], (True, False), boundary,
                    MU, KAPPA, 2.0 * (degree + 1.0) ** 2, wall_gradient,
                )
                rows.append(
                    {
                        "wall_gradient": wall_gradient,
                        "degree": degree,
                        "ncells_y": ny,
                        "rhs_max": float(rhs.abs().max()),
                        "energy_max": float(rhs[..., 3].abs().max()),
                        "momentum_max": float(rhs[..., 1].abs().max()),
                        "wall_rows": float(
                            torch.cat((rhs[:, 0, :, :].flatten(), rhs[:, -1, :, :].flatten()))
                            .abs()
                            .max()
                        ),
                    }
                )
    return {"cases": rows}


def couette_series(y, time, speed=WALL_SPEED, nu=MU, height=1.0, terms=600):
    """Closed-form impulsively started Couette flow between two isothermal walls."""
    profile = speed * y / height
    for n in range(1, terms + 1):
        profile = profile + (2.0 * speed / math.pi) * (-1.0) ** n / n * torch.sin(
            n * math.pi * y / height
        ) * math.exp(-nu * n * n * math.pi**2 * time / height**2)
    return profile


def transient_case(degree, ny, final_time=0.1, wall_gradient="fd2"):
    started = time.time()
    record = {
        "degree": degree,
        "ncells_y": ny,
        "final_time": final_time,
        "wall_gradient": wall_gradient,
    }
    try:
        _, y, coeff, meta = ns.solve_couette_dg_2d(
            ncells=(2, ny), degree=degree, mu=MU, final_time=final_time,
            wall_gradient=wall_gradient,
        )
    except FloatingPointError as error:
        record.update(stable=False, error=str(error), seconds=time.time() - started)
        return record
    # *cell-centre* values: the quadrature-point value is not on the y grid
    centre_u, centre_v, centre_T = ns._cell_centre_primitives(coeff)
    mean = coeff[..., 0, 0, :]
    density = mean[..., 0].clamp_min(1e-12)
    numerical = centre_u
    exact = couette_series(
        y.view(1, ny).expand(1, ny), final_time, nu=MU / float(density.mean())
    )
    record.update(
        stable=True,
        steps=meta["steps"],
        dt=meta["dt"],
        relative_l2_velocity_x=float(
            torch.linalg.norm(numerical - exact) / torch.linalg.norm(exact)
        ),
        velocity_max=float(centre_u.abs().max()),
        velocity_y_max=float(centre_v.abs().max()),
        temperature_min=float(centre_T.min()),
        temperature_max=float(centre_T.max()),
        profile_numerical=[float(value) for value in numerical[0]],
        profile_exact=[float(value) for value in exact[0]],
        y_centres=[float(value) for value in y],
        seconds=time.time() - started,
    )
    return record


def section_transient():
    cases = []
    for degree, ny in ((1, 8), (1, 16), (2, 4)):
        cases.append(transient_case(degree, ny))
    # legacy wall gradient: the case m2496 reported as unstable
    cases.append(transient_case(1, 8, wall_gradient="br1"))
    return {"cases": cases}


def section_stability():
    cases = []
    for degree, ny in ((1, 8), (2, 4)):
        for cfl_viscous in (1.0, 4.0):
            record = {
                "degree": degree,
                "ncells_y": ny,
                "cfl_viscous": cfl_viscous,
            }
            try:
                _, _, _, meta = ns.solve_couette_dg_2d(
                    ncells=(2, ny), degree=degree, mu=MU, final_time=0.05,
                    cfl_viscous=cfl_viscous,
                )
                record.update(stable=True, steps=meta["steps"])
            except FloatingPointError as error:
                record.update(stable=False, error=str(error))
            cases.append(record)
    return {"cases": cases}


SECTIONS = {
    "flux": section_flux,
    "residual": section_residual,
    "transient": section_transient,
    "stability": section_stability,
}


def plot(data, path=FIGURE_PATH):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) wall heat-flux error
    axis = axes[0]
    styles = {("br1", 1): ("C0", "o", "--"), ("fd2", 1): ("C0", "o", "-"),
              ("br1", 2): ("C1", "s", "--"), ("fd2", 2): ("C1", "s", "-")}
    for case in data["flux"]["cases"]:
        colour, marker, style = styles[(case["wall_gradient"], case["degree"])]
        meshes = [row["ncells_y"] for row in case["meshes"]]
        errors = [row["energy_error_bottom"] for row in case["meshes"]]
        axis.loglog(
            meshes, errors, style, color=colour, marker=marker,
            label=f"{case['wall_gradient']} P{case['degree']} "
                  f"(slope {case['observed_order_bottom']:.2f})",
        )
    axis.set_xlabel("cells in y")
    axis.set_ylabel(r"$|\bar q_w - q_w^{exact}|$")
    axis.set_title("wall heat flux error (Couette)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)

    # (b) transient profile against the Fourier series
    axis = axes[1]
    for case in data["transient"]["cases"]:
        if not case.get("stable") or case["wall_gradient"] != "fd2":
            continue
        axis.plot(
            case["profile_exact"], case["y_centres"], "k-", lw=1,
            label="Fourier series" if case["degree"] == 1 and case["ncells_y"] == 8 else None,
        )
        axis.plot(
            case["profile_numerical"], case["y_centres"], "o--", ms=4,
            label=f"P{case['degree']} 2x{case['ncells_y']} "
                  f"({case['relative_l2_velocity_x'] * 100:.2f}%)",
        )
    axis.set_xlabel("u / U")
    axis.set_ylabel("y / H")
    axis.set_title("impulsively started Couette at t = 0.1")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    # (c) residual of the projected analytic steady state
    axis = axes[2]
    for case in data["residual"]["cases"]:
        colour = "C0" if case["degree"] == 1 else "C1"
        style = "--" if case["wall_gradient"] == "br1" else "-"
        axis.loglog(
            [case["ncells_y"]], [case["rhs_max"]], style, color=colour,
            marker="o" if case["degree"] == 1 else "s",
            label=None if case["ncells_y"] != 8 and case["ncells_y"] != 4 else
            f"{case['wall_gradient']} P{case['degree']}",
        )
    axis.set_xlabel("cells in y")
    axis.set_ylabel(r"$\max|\partial_t U|$")
    axis.set_title("residual of the projected steady state")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)

    figure.suptitle("TensorDG compressible Navier-Stokes: Dirichlet wall channel")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    print("wrote", path)


def main(argv):
    wanted = [name for name in argv if name in SECTIONS] or list(SECTIONS)
    unknown = [name for name in argv if name not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown sections: {unknown}; choose from {list(SECTIONS)}")
    started = time.time()
    data = {
        "benchmark": "TensorDG ns_dg_2d Dirichlet wall channel",
        "mu": MU,
        "prandtl": PRANDTL,
        "kappa": KAPPA,
        "wall_speed": WALL_SPEED,
        "wall_gradient_default": "fd2",  # m2831 closure; module default is "face"
        "wall_penalty_default": ns.WALL_PENALTY,
        "dtype": "float64",
        "torch": torch.__version__,
    }
    for name in wanted:
        print(f"== section {name} ==", flush=True)
        data[name] = SECTIONS[name]()
        JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f"   {time.time() - started:.1f}s", flush=True)
    data["seconds"] = time.time() - started
    JSON_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print("wrote", JSON_PATH)
    if {"flux", "residual", "transient"} <= set(data):
        plot(data)


if __name__ == "__main__":
    main(sys.argv[1:])