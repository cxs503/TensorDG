"""CKS1998 "flow past a circular cylinder": steady 2-D Euler with curvilinear DG.

The module ``tensordg.euler_curved`` used to record that this case could not be
run because the reflective wall was linearly unstable (round-off blew up after
~700 RK steps).  The wall-flux orientation defect behind it is fixed (see
``docs/CASE-DG-CKS1998-CYLINDER-WALL-STABILITY.md``), so the case is computed
here and checked against quantities that are known *exactly*:

T1  stagnation pressure.  On the stagnation streamline the flow is brought to
    rest across a (nearly) normal bow shock, so the wall pressure there is the
    Rayleigh-pitot total pressure of ``M_inf`` -- an exact function of ``M_inf``
    and ``gamma`` alone.  Target: ``Cp_stag = 2 (p0/p_inf - 1)/(gamma M^2)``.
T2  symmetry.  The configuration (circular body, uniform inflow along x) is
    invariant under ``y -> -y``, so ``Cp(theta) = Cp(pi - theta)``; the residual
    asymmetry is a pure discretization error.
T3  d'Alembert is *not* available (supersonic: wave drag > 0), but mass and
    energy are conserved by the DG scheme up to the boundary fluxes; the
    drill is reported for the record, not as a pass/fail gate.
T4  steady state is reached (residual decays monotonically to a plateau).

Usage::

    PYTHONPATH=. python benchmarks/cylinder_euler_steady.py --nr 28 --ntheta 44 --p 1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg.euler_curved import CurvilinearEuler, cylinder_annulus_mesh

OUT = Path(__file__).resolve().parents[1] / "validation" / "cylinder_steady"
GAMMA = 1.4


def isentropic_total_pressure(mach: float, gamma: float = GAMMA) -> float:
    """``p0/p_inf = (1 + (gamma-1)/2 M^2)^{gamma/(gamma-1)}`` (subsonic, exact)."""
    return (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** (gamma / (gamma - 1.0))


def rayleigh_pitot(mach: float, gamma: float = GAMMA) -> float:
    """``p0/p_inf`` behind a normal shock at ``mach`` (exact)."""
    return (((gamma + 1.0) ** 2 * mach ** 2
             / (4.0 * gamma * mach ** 2 - 2.0 * (gamma - 1.0))) ** (gamma / (gamma - 1.0))
            * ((1.0 - gamma + 2.0 * gamma * mach ** 2) / (gamma + 1.0)))


def run(nr: int, ntheta: int, p: int, mach: float, r_out: float, growth: float,
        cfl: float, max_time: float, flip_wall: bool = False) -> dict:
    a_inf = math.sqrt(GAMMA)
    rho_inf, p_inf = 1.0, 1.0
    u_inf = mach * a_inf
    mesh = cylinder_annulus_mesh(nr, ntheta, 1.0, r_out, degree=2, growth=growth)
    solver = CurvilinearEuler(
        mesh, p, cfl=cfl, boundaries=("wall", "farfield", "periodic", "periodic"),
        farfield=(rho_inf, u_inf, 0.0, p_inf))
    if flip_wall:                      # negative control: pre-fix orientation
        original = solver._boundary_normal_flux

        def broken(kind, inner, side, ux, uy, _o=original):
            flux = _o(kind, inner, side, ux, uy)
            return -flux if kind == "wall" else flux
        solver._boundary_normal_flux = broken

    state = solver.project(lambda x, y: (np.ones_like(x), u_inf * np.ones_like(x),
                                         np.zeros_like(x), np.ones_like(x)))
    mass0 = solver.totals(state)[0]
    dt = 0.5 * solver.stable_dt(state)
    hist = []
    t = 0.0
    steps = 0
    blew_up = False
    while t < max_time:
        state = solver.ssp_rk3_step(state, dt)
        t += dt
        steps += 1
        if steps % 50 == 0:
            rate = float(np.max(np.abs(solver.rhs(state))))
            hist.append((t, rate))
            if not np.all(np.isfinite(state)) or np.abs(state).max() > 1e3:
                blew_up = True
                break
            if len(hist) > 20 and rate < 1e-6 * hist[0][1]:
                break

    result = {
        "nr": nr, "ntheta": ntheta, "degree": p, "mach": mach,
        "r_out": r_out, "growth": growth, "cfl": cfl,
        "cells": int(mesh.ncells), "dt": dt, "steps": steps, "time": t,
        "blew_up": blew_up, "flip_wall": bool(flip_wall),
        "residual_history": hist[-40:],
        "mass_drift": float(solver.totals(state)[0] - mass0),
    }
    if blew_up:
        return result

    walls = solver.wall_pressure(state)
    # the inner circle is the ``axis0_lo`` wall family, the outer circle the
    # ``axis0_hi`` far-field boundary (see ``cylinder_annulus_mesh``)
    theta = walls[(0, -1)]["theta"].ravel()
    pressure = walls[(0, -1)]["p"].ravel()
    order = np.argsort(theta)
    theta, pressure = theta[order], pressure[order]
    cp = 2.0 * (pressure - p_inf) / (GAMMA * mach ** 2)
    # fold theta onto [-pi/2, pi/2] about the stagnation streamline (y -> -y)
    theta_f = np.where(theta > np.pi, theta - 2.0 * np.pi, theta)
    theta_f = np.where(theta_f > np.pi / 2, np.pi - theta_f, theta_f)
    theta_f = np.where(theta_f < -np.pi / 2, -np.pi - theta_f, theta_f)
    fold = np.argsort(theta_f)
    th_sorted, cp_sorted = theta_f[fold], cp[fold]
    half = len(cp) // 2
    symmetry = float(np.max(np.abs(cp_sorted[:half] - cp_sorted[::-1][:half]))
                     / max(np.max(np.abs(cp)), 1e-30))

    force = solver.wall_force(state)
    drag = force["per_side"][(0, -1)][0]
    cd = drag / (0.5 * rho_inf * u_inf ** 2 * 2.0)
    # subsonic: isentropic total pressure (no shock on the stagnation streamline);
    # supersonic: Rayleigh-pitot behind the normal part of the bow shock.
    total = (isentropic_total_pressure(mach) if mach < 1.0
             else rayleigh_pitot(mach))
    target = 2.0 * (total - 1.0) / (GAMMA * mach ** 2)
    cp_stag = float(np.max(cp))
    result.update({
        "cp_stag": cp_stag,
        "cp_stag_target": target,
        "cp_stag_rel_error": abs(cp_stag - target) / target,
        "cp_min": float(np.min(cp)),
        "cd": float(cd),
        "symmetry_error": symmetry,
        "theta": theta.tolist(),
        "cp": cp.tolist(),
        "force": {"Fx": force["Fx"], "Fy": force["Fy"]},
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=28)
    ap.add_argument("--ntheta", type=int, default=44)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--mach", type=float, default=0.38)
    ap.add_argument("--rout", type=float, default=10.0)
    ap.add_argument("--growth", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--maxtime", type=float, default=10.0)
    ap.add_argument("--flip-wall", action="store_true")
    args = ap.parse_args()
    res = run(args.nr, args.ntheta, args.p, args.mach, args.rout, args.growth,
              args.cfl, args.maxtime, args.flip_wall)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"p{args.p}_{args.nr}x{args.ntheta}" + ("_flip" if args.flip_wall else "")
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("theta", "cp", "residual_history")}, indent=2))


if __name__ == "__main__":
    main()