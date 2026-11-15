"""Is the growth a spurious top-mode (checkerboard) effect or a general one?

Smooth vs. random perturbation, p=0 and p=1, on a fully periodic uniform mesh;
also checks whether the perturbed update conserves mass/momentum/energy.
"""
from __future__ import annotations

import numpy as np

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import CurvilinearEuler


def periodic_channel(nx, ny, lx=4.0, ly=1.0):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=(True, True), walls=(False,) * 4)


def rest(x, y):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def smooth_pert(x, y):
    return (1.0 + 1e-6 * np.sin(np.pi * x) * np.sin(2 * np.pi * y),
            np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def norm(solver, state, s0):
    return float(np.abs(state - s0).max())


def run(p, kind, steps=200):
    solver = CurvilinearEuler(periodic_channel(8, 6), p)
    s0 = solver.project(rest)
    if kind == "smooth":
        state = solver.project(lambda x, y: smooth_pert(x, y))
    else:
        rng = np.random.default_rng(7)
        state = s0 + rng.standard_normal(s0.shape) * 1e-6
    dt = solver.stable_dt(state)
    # conservation of the *difference* (should hold for any state: periodic + consistent)
    rate = solver.rhs(state)
    totals_rate = solver.totals(rate)
    print(f"p={p} kind={kind}: totals(dU/dt) = "
          f"[{', '.join(f'{v:+.3e}' for v in totals_rate)}]  (mass/mom/mom/E)")
    n0 = norm(solver, state, s0)
    ratios = []
    for n in range(steps):
        state = solver.ssp_rk3_step(state, dt)
        cur = norm(solver, state, s0)
        if not np.isfinite(cur) or cur > 1e6:
            print(f"   BLEW UP after {n} steps")
            break
        ratios.append(cur / n0)
    else:
        print(f"   norm ratio after 1/10/50/{steps} steps: "
              f"{ratios[0]:.3e} {ratios[9]:.3e} {ratios[49]:.3e} {ratios[-1]:.3e}")
    return ratios


if __name__ == "__main__":
    for p in (0, 1):
        for kind in ("smooth", "random"):
            run(p, kind)