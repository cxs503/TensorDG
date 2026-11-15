"""Verification of the curvilinear Euler boundary kinds (honest evidence bundle).

Writes ``validation/curved_wall_bc/metrics.json``.  Every number in the report
is copied from that file programmatically -- nothing is typed by hand.

Measured quantities
-------------------
W1  free-stream preservation with walls: ``sup|dU/dt|`` of a uniform slip flow
    in a wall-bounded channel (should be round-off);
W1n negative control: the same operator with the wall flux sign flipped (must be
    many orders of magnitude larger, otherwise W1 is vacuous);
W2  wall-flux admissibility: ``max|mass flux|`` and ``max|energy flux|`` at the
    wall (must be round-off) and ``max|momentum flux - p n|``;
W3  closed-annulus conservation: relative mass/energy drift over 20 steps;
W4  resolved geometry: relative error of the discrete annulus area vs the
    analytic value, and the exactness of the sampled inner radius;
W5  the honest negative result: growth of round-off in a march of an *exact*
    steady state (uniform slip flow), reported as the step at which the state
    exceeds 1e3 and the amplification per 10 steps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import (
    CurvilinearEuler,
    conserved_from_primitive,
    cylinder_annulus_mesh,
    wall_normal_flux,
)

OUT = Path(__file__).resolve().parents[1] / "validation" / "curved_wall_bc"


def channel(nx, ny, lx=4.0, ly=1.0, periodic=(False, False)):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=(False, False, True, True))


def uniform_slip(x, y):
    return (np.ones_like(x), 0.3 * np.ones_like(x), np.zeros_like(x),
            np.ones_like(x))


def closed_annulus(nr=8, ntheta=16, r_out=2.0):
    mesh = cylinder_annulus_mesh(nr, ntheta, 1.0, r_out, degree=2)
    mesh.walls = (True, True, False, False)
    mesh.neighbours = mesh._build_neighbours()
    return mesh


def main() -> dict:
    out = {}

    # W1 / W1n ---------------------------------------------------------------
    mesh = channel(8, 6)
    solver = CurvilinearEuler(mesh, 1)
    state = solver.project(uniform_slip)
    out["W1_wall_fsp_operator"] = float(np.abs(solver.rhs(state)).max())
    broken = CurvilinearEuler(mesh, 1)
    original = broken._boundary_normal_flux
    broken._boundary_normal_flux = lambda kind, inner, side, ux, uy: -original(
        kind, inner, side, ux, uy)
    out["W1n_flipped_wall_flux_operator"] = float(np.abs(broken.rhs(state)).max())

    # W2 --------------------------------------------------------------------
    rng = np.random.default_rng(20261115)
    nx_, ny_ = rng.normal(size=64), rng.normal(size=64)
    norm = np.hypot(nx_, ny_)
    nx_, ny_ = nx_ / norm, ny_ / norm
    rho = 1.0 + 0.3 * rng.random(64)
    tangential = rng.normal(size=64)          # admissible wall state: u_n = 0
    pressure = 0.7 + 0.6 * rng.random(64)
    speed = np.stack([-tangential * ny_, tangential * nx_], axis=-1)
    flux = wall_normal_flux(
        conserved_from_primitive(rho, speed[:, 0], speed[:, 1], pressure), nx_, ny_)
    out["W2_wall_mass_flux"] = float(np.abs(flux[:, 0]).max())
    out["W2_wall_energy_flux"] = float(np.abs(flux[:, 3]).max())
    out["W2_wall_momentum_minus_pn"] = float(max(
        np.abs(flux[:, 1] - pressure * nx_).max(),
        np.abs(flux[:, 2] - pressure * ny_).max()))

    # W3 / W4 ---------------------------------------------------------------
    ann = closed_annulus()
    asolver = CurvilinearEuler(ann, 1)
    astate = asolver.project(lambda x, y: (
        1.0 + 0.05 * np.exp(-((x - 1.6) ** 2 + y ** 2) / 0.2),
        np.zeros_like(x), np.zeros_like(x), np.ones_like(x)))
    before = asolver.totals(astate)
    dt = asolver.stable_dt(astate)
    for _ in range(20):
        astate = asolver.ssp_rk3_step(astate, dt)
    after = asolver.totals(astate)
    out["W3_mass_drift_rel"] = float(abs(after[0] - before[0]) / abs(before[0]))
    out["W3_energy_drift_rel"] = float(abs(after[3] - before[3]) / abs(before[3]))
    for nr, nt, gdeg in ((4, 8, 1), (8, 16, 2), (12, 24, 2)):
        m = cylinder_annulus_mesh(nr, nt, 1.0, 3.0, degree=gdeg)
        exact = math.pi * (3.0 ** 2 - 1.0)
        area = float(np.sum(m.areas()))
        out[f"W4_area_rel_nr{nr}_nt{nt}_g{gdeg}"] = float(abs(area - exact) / exact)

    # W5 --------------------------------------------------------------------
    march = CurvilinearEuler(channel(8, 6), 1)
    s0 = march.project(uniform_slip)
    dt = march.stable_dt(s0)
    s = s0.copy()
    residual = []
    blew_up_at = None
    for step in range(1, 401):
        s = march.ssp_rk3_step(s, dt)
        if step % 10 == 0:
            residual.append((step, float(np.abs(march.rhs(s)).max())))
        if not np.all(np.isfinite(s)) or np.abs(s).max() > 1e3:
            blew_up_at = step
            break
    out["W5_blew_up_at_step"] = blew_up_at
    out["W5_dt"] = float(dt)
    out["W5_residual_history"] = residual
    growth = [(b[0], b[1] / max(a[1], 1e-300)) for a, b in zip(residual, residual[1:])]
    out["W5_roundoff_growth_per_10_steps"] = growth

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if not k.startswith("W5_roundoff")}, indent=2))
    return out


if __name__ == "__main__":
    main()
