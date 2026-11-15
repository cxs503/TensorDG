"""Diagnose the reflective-wall linear instability of CurvilinearEuler.

Read-only diagnostic: isolates (a) spatial vs temporal origin, (b) which boundary
kind carries the unstable mode.  Nothing here is a deliverable.
"""
from __future__ import annotations

import numpy as np

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import CurvilinearEuler, conserved_from_primitive


def channel(nx, ny, lx=4.0, ly=1.0, periodic=(False, False), walls=(False, False, True, True)):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=walls)


def uniform_slip(x, y):
    return (np.ones_like(x), 0.3 * np.ones_like(x), np.zeros_like(x), np.ones_like(x))


def growth(solver, state0, dt, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    pert = rng.standard_normal(state0.shape) * 1e-8
    state = state0 + pert
    rates = []
    for n in range(steps):
        prev = np.abs(state - state0).max()
        state = solver.ssp_rk3_step(state, dt)
        cur = np.abs(state - state0).max()
        if not np.all(np.isfinite(state)) or cur > 1e6:
            return rates, None, True
        rates.append(cur / prev if prev > 0 else np.nan)
    return rates, state - state0, False


def report(tag, solver, dt, steps=120):
    state0 = solver.project(uniform_slip)
    rates, delta, blew = growth(solver, state0, dt, steps=steps)
    print(f"[{tag}] dt={dt:.5f} blew_up={blew} "
          f"rate/step first5={np.round(rates[:5], 5)} last5={np.round(rates[-5:], 5)}")
    if delta is not None:
        amp = np.abs(delta).reshape(delta.shape[0], -1).max(axis=1)
        idx = np.argsort(amp)[::-1][:5]
        print(f"    top cells (flat idx, |amp|): {[(int(i), float(amp[i])) for i in idx]}")
        # cell coordinates
        xc = solver.xg.reshape(delta.shape[0], -1).mean(axis=1)
        yc = solver.yg.reshape(delta.shape[0], -1).mean(axis=1)
        print(f"    cell centres of top cells: "
              f"{[(round(float(xc[i]), 3), round(float(yc[i]), 3)) for i in idx]}")
    return rates, delta, blew


if __name__ == "__main__":
    # A) walls in eta, no-slip-through; x-boundaries = do-nothing outflow
    solver = CurvilinearEuler(channel(8, 6), 1)
    dt0 = solver.stable_dt(solver.project(uniform_slip))
    print("=== A: channel 8x6 P1, outflow-x + walls-y ===")
    for f in (1.0, 0.5, 0.25, 0.1):
        report(f"walls-dt*{f}", solver, dt0 * f)

    # B) same mesh but periodic in y, no walls -> isolates wall vs volume
    solver_py = CurvilinearEuler(channel(8, 6, periodic=(False, True),
                                         walls=(False, False, False, False)), 1)
    print("=== B: periodic-y (no walls) ===")
    report("periodic-y", solver_py, solver_py.stable_dt(solver_py.project(uniform_slip)))

    # C) fully periodic
    solver_pp = CurvilinearEuler(channel(8, 6, periodic=(True, True),
                                         walls=(False, False, False, False)), 1)
    print("=== C: fully periodic ===")
    report("periodic-xy", solver_pp, solver_pp.stable_dt(solver_pp.project(uniform_slip)))

    # D) walls in y only, x periodic
    solver_w = CurvilinearEuler(channel(8, 6, periodic=(True, False),
                                        walls=(False, False, True, True)), 1)
    print("=== D: periodic-x + walls-y (isolates walls) ===")
    dt_w = solver_w.stable_dt(solver_w.project(uniform_slip))
    for f in (1.0, 0.25):
        report(f"walls-only-dt*{f}", solver_w, dt_w * f)