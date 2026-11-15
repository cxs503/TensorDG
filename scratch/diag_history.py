"""Growth pattern of a tiny smooth perturbation (p=0 and p=1, periodic uniform mesh)."""
from __future__ import annotations

import numpy as np

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import CurvilinearEuler


def periodic_mesh(nx, ny, lx=4.0, ly=1.0):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=(True, True), walls=(False,) * 4)


def rest(x, y):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def perturbed(x, y):
    return (1.0 + 1e-6 * np.sin(np.pi * x) * np.sin(2 * np.pi * y),
            np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def run(p, mesh_shape=(8, 6), dt_factor=1.0, steps=300, label=""):
    solver = CurvilinearEuler(periodic_mesh(*mesh_shape), p)
    s0 = solver.project(rest)
    state = solver.project(perturbed)
    dt = solver.stable_dt(state) * dt_factor
    n0 = float(np.abs(state - s0).max())
    print(f"\n[{label or f'p={p} mesh={mesh_shape} dt*{dt_factor}'}] dt={dt:.5f} n0={n0:.3e}")
    for n in range(1, steps + 1):
        state = solver.ssp_rk3_step(state, dt)
        cur = float(np.abs(state - s0).max())
        if n % 25 == 0 or not np.isfinite(cur):
            print(f"   step {n:4d} t={n*dt:6.3f} ratio={cur/n0:.3e}")
        if not np.isfinite(cur) or cur > 1e3:
            print(f"   -> diverged at step {n}")
            return


if __name__ == "__main__":
    run(0, dt_factor=1.0, label="p=0 8x6")
    run(0, dt_factor=0.25, label="p=0 8x6 dt/4")
    run(1, dt_factor=1.0, label="p=1 8x6")
    run(1, mesh_shape=(8, 8), dt_factor=1.0, label="p=1 8x8 square")