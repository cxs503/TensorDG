"""Verification sweep for :mod:`tensordg.advection_diffusion_1d`.

Produces ``validation/advection_diffusion_1d/evidence.json`` plus three figures.

Sections
--------
1. h-convergence of the L2 error for Pe in {1, 10, 100} and degree p in {1, 2, 3}
   -> measured EOC against the design order p+1.
2. cell-Peclet sweep at fixed dofs: which ``a*h/D`` still meets the 3% gate.
3. implementation controls: SIPG penalty collapse and the (non-discriminating)
   downwind advective flux.
4. face-flux consistency: max deviation of ``a u - D u'`` on cell faces from the
   exact constant flux, normalised by ``a*|g_R - g_L|``.
5. baseline comparison: first-order finite-volume upwind on the same cell count.

Run with ``python benchmarks/verify_advection_diffusion_1d.py`` (PYTHONPATH=.).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.advection_diffusion_1d import (
    advection_diffusion_error,
    exact_flux,
    exact_solution,
    face_fluxes,
    sample_solution,
    solve_advection_diffusion_1d,
)

OUT = Path("validation/advection_diffusion_1d")
FIG = Path("docs/figures")
PECLETS = (1.0, 10.0, 100.0)
DEGREES = (1, 2, 3)
CELLS = (8, 16, 32, 64, 128)


def _eoc(fine: float, coarse: float, ratio: float = 2.0) -> float:
    return math.log(coarse / fine) / math.log(ratio)


def finite_volume_upwind(ncells: int, peclet: float, speed: float = 1.0) -> tuple[float, float]:
    """First-order upwind finite-volume solution of (1).

    Unknowns are cell averages; the inflow ghost value is the Dirichlet data at
    x=0, the last cell is a Dirichlet cell with u=1.  Returns (L2 error, cell
    Peclet number).
    """
    h = 1.0 / ncells
    diffusivity = speed / peclet
    cell_peclet = speed * h / diffusivity
    matrix = np.zeros((ncells, ncells))
    rhs = np.zeros(ncells)
    flux = speed + diffusivity / h
    diffusive = diffusivity / h
    matrix[0, 0] = speed + 2.0 * diffusive
    if ncells > 1:
        matrix[0, 1] = -diffusive
    for row in range(1, ncells - 1):
        matrix[row, row - 1] = -flux
        matrix[row, row] = speed + 2.0 * diffusive
        matrix[row, row + 1] = -diffusive
    matrix[ncells - 1, ncells - 1] = 1.0
    rhs[ncells - 1] = 1.0
    u = np.linalg.solve(matrix, rhs)
    centers = (np.arange(ncells) + 0.5) * h
    exact = np.array([float(exact_solution(torch.tensor(x), peclet)) for x in centers])
    return float(np.sqrt(np.sum((u - exact) ** 2) * h)), cell_peclet


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    evidence: dict = {
        "case": "steady a*u' - D*u'' = 0 on [0,1], u(0)=0, u(1)=1, exact exponential layer",
        "convergence": {},
        "cell_peclet_sweep": [],
        "controls": {},
        "face_flux": {},
        "finite_volume_baseline": [],
    }

    # ---- 1. h-convergence --------------------------------------------------
    for peclet in PECLETS:
        for degree in DEGREES:
            rows = []
            for ncells in CELLS:
                centers, coeff, meta = solve_advection_diffusion_1d(
                    ncells=ncells, degree=degree, peclet=peclet,
                )
                err = advection_diffusion_error(centers, coeff, peclet)
                rows.append({
                    "ncells": ncells, "h": meta["h"], "dofs": meta["dofs"],
                    "absolute_l2": err["absolute_l2"],
                    "relative_l2": err["relative_l2"],
                })
            for i in range(len(rows) - 1):
                rows[i]["eoc"] = _eoc(rows[i + 1]["absolute_l2"], rows[i]["absolute_l2"])
            evidence["convergence"][f"pe{peclet:g}_p{degree}"] = rows

    # ---- 2. cell-Peclet sweep at fixed resolution --------------------------
    ncells = 64
    h = 1.0 / ncells
    for peclet in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
        for degree in (1, 2, 3):
            centers, coeff, _ = solve_advection_diffusion_1d(
                ncells=ncells, degree=degree, peclet=peclet,
            )
            err = advection_diffusion_error(centers, coeff, peclet)
            evidence["cell_peclet_sweep"].append({
                "peclet": peclet, "degree": degree, "ncells": ncells,
                "cell_peclet": peclet * h,
                "relative_l2": err["relative_l2"],
                "below_three_percent": err["relative_l2"] < 0.03,
            })

    # ---- 3. implementation controls ---------------------------------------
    centers, coeff, _ = solve_advection_diffusion_1d(ncells=32, degree=2, peclet=10.0)
    baseline = advection_diffusion_error(centers, coeff, 10.0)["relative_l2"]
    controls = {"baseline_pe10_p2_n32": baseline, "penalty_sweep": [], "downwind": []}
    for penalty in (20.0, 1.0, 0.1, 1e-2, 1e-4):
        centers, coeff, _ = solve_advection_diffusion_1d(
            ncells=32, degree=2, peclet=10.0, penalty=penalty,
        )
        controls["penalty_sweep"].append({
            "penalty": penalty,
            "relative_l2": advection_diffusion_error(centers, coeff, 10.0)["relative_l2"],
        })
    for peclet in (10.0, 100.0, 1000.0):
        for scheme in ("upwind", "downwind"):
            centers, coeff, _ = solve_advection_diffusion_1d(
                ncells=32, degree=2, peclet=peclet, advection_scheme=scheme,
            )
            controls["downwind"].append({
                "peclet": peclet, "cell_peclet": peclet / 32, "scheme": scheme,
                "relative_l2": advection_diffusion_error(centers, coeff, peclet)["relative_l2"],
            })
    evidence["controls"] = controls

    # ---- 4. face-flux consistency -----------------------------------------
    for peclet in PECLETS:
        for degree in DEGREES:
            rows = {}
            exact = exact_flux(peclet)
            for ncells in (8, 32, 128):
                centers, coeff, _ = solve_advection_diffusion_1d(
                    ncells=ncells, degree=degree, peclet=peclet,
                )
                fluxes = face_fluxes(coeff, peclet)
                rows[f"n{ncells}"] = float((fluxes - exact).abs().max().item())
            evidence["face_flux"][f"pe{peclet:g}_p{degree}"] = {
                "exact_flux": exact, "max_abs_deviation": rows,
            }

    # ---- 5. first-order FV baseline ---------------------------------------
    for ncells in CELLS:
        error, cell_peclet = finite_volume_upwind(ncells, 10.0)
        evidence["finite_volume_baseline"].append({
            "ncells": ncells, "h": 1.0 / ncells, "cell_peclet": cell_peclet,
            "absolute_l2": error,
        })

    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    # ---- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, peclet in zip(axes, PECLETS):
        for degree in DEGREES:
            rows = evidence["convergence"][f"pe{peclet:g}_p{degree}"]
            h = [row["h"] for row in rows]
            e = [row["absolute_l2"] for row in rows]
            ax.loglog(h, e, "o-", label=f"DG p={degree}")
        ax.set_xlabel("h")
        ax.set_ylabel(r"$\|u_h-u_{\rm exact}\|_{L^2}$")
        ax.set_title(f"Pe = {peclet:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Steady advection–diffusion: h-convergence against the exact exponential layer")
    fig.tight_layout()
    fig.savefig(FIG / "advection_diffusion_convergence.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, peclet, configs in zip(
        axes, (10.0, 100.0),
        ((16, 64), (32, 128)),
    ):
        exact_x = np.linspace(0.0, 1.0, 400)
        exact_y = [float(exact_solution(torch.tensor(x), peclet)) for x in exact_x]
        ax.plot(exact_x, exact_y, "k-", lw=2, label="exact")
        for ncells in configs:
            centers, coeff, _ = solve_advection_diffusion_1d(
                ncells=ncells, degree=2, peclet=peclet,
            )
            xq, values = sample_solution(centers, coeff)
            ax.plot(xq.numpy(), values.numpy(), ".-", label=f"DG P2, n={ncells}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Pe = {peclet:g}: layer resolution")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "advection_diffusion_profiles.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for degree in DEGREES:
        pts = [row for row in evidence["cell_peclet_sweep"] if row["degree"] == degree]
        axes[0].loglog(
            [row["cell_peclet"] for row in pts],
            [max(row["relative_l2"], 1e-16) for row in pts],
            "o-", label=f"DG p={degree}",
        )
    fv = evidence["finite_volume_baseline"]
    axes[0].loglog(
        [row["cell_peclet"] for row in fv], [row["absolute_l2"] for row in fv],
        "s--", color="grey", label="FV upwind (P0 baseline)",
    )
    axes[0].axhline(0.03, color="r", ls=":", label="3% gate")
    axes[0].set_xlabel(r"cell Péclet  $a h / D$")
    axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("n = 64 cells, Pe sweep")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    pens = controls["penalty_sweep"]
    axes[1].loglog([p["penalty"] for p in pens], [p["relative_l2"] for p in pens], "o-")
    axes[1].axhline(baseline, color="g", ls=":", label="reference penalty=20")
    axes[1].set_xlabel("SIPG penalty parameter")
    axes[1].set_ylabel("relative $L^2$ error")
    axes[1].set_title("implementation control: penalty collapse (Pe=10, P2, n=32)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "advection_diffusion_controls.png", dpi=150)
    plt.close(fig)

    summary = {
        "pe1_p1_eoc_tail": evidence["convergence"]["pe1_p1"][-2]["eoc"],
        "pe10_p2_eoc_tail": evidence["convergence"]["pe10_p2"][-2]["eoc"],
        "pe100_p3_eoc_tail": evidence["convergence"]["pe100_p3"][-2]["eoc"],
        "pe10_p2_n64_relative_l2": evidence["convergence"]["pe10_p2"][3]["relative_l2"],
        "pe100_p3_n64_relative_l2": evidence["convergence"]["pe100_p3"][3]["relative_l2"],
        "baseline_pe10_p2_n32": baseline,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return evidence


if __name__ == "__main__":
    main()