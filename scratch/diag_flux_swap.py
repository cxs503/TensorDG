"""Which part of the operator injects energy: the Riemann flux or the assembly?

Substitutes the numerical flux with (a) Rusanov (maximally dissipative),
(b) the arithmetic-average physical flux (non-dissipative), and repeats the
tiny stationary entropy-wave march on a fully periodic uniform mesh.
"""
from __future__ import annotations

import numpy as np

import tensordg.euler_curved as ec
from tensordg.curved_geometry import CurvedMesh, IsoparametricMap


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


def rusanov(left, right, nx, ny, gamma=ec.GAMMA):
    flx, fly = ec.euler_flux(left, gamma)
    frx, fry = ec.euler_flux(right, gamma)
    fl = flx * nx[..., None] + fly * ny[..., None]
    fr = frx * nx[..., None] + fry * ny[..., None]
    rl = np.maximum(left[..., 0], 1e-12)
    rr = np.maximum(right[..., 0], 1e-12)
    ulx, uly = left[..., 1] / rl, left[..., 2] / rl
    urx, ury = right[..., 1] / rr, right[..., 2] / rr
    pl = np.maximum(0.4 * (left[..., 3] - 0.5 * rl * (ulx ** 2 + uly ** 2)), 1e-12)
    pr = np.maximum(0.4 * (right[..., 3] - 0.5 * rr * (urx ** 2 + ury ** 2)), 1e-12)
    sl = np.sqrt(gamma * pl / rl)
    sr = np.sqrt(gamma * pr / rr)
    alpha = np.maximum(np.abs(ulx * nx + uly * ny) + sl, np.abs(urx * nx + ury * ny) + sr)
    return 0.5 * (fl + fr) - 0.5 * alpha[..., None] * (right - left)


def central(left, right, nx, ny, gamma=ec.GAMMA):
    flx, fly = ec.euler_flux(left, gamma)
    frx, fry = ec.euler_flux(right, gamma)
    return 0.5 * ((flx + frx) * nx[..., None] + (fly + fry) * ny[..., None])


def march(tag, steps=400, p=0):
    solver = ec.CurvilinearEuler(periodic_mesh(8, 6), p)
    s0 = solver.project(rest)
    state = solver.project(perturbed)
    dt = solver.stable_dt(state)
    n0 = float(np.abs(state - s0).max())
    for n in range(1, steps + 1):
        state = solver.ssp_rk3_step(state, dt)
        cur = float(np.abs(state - s0).max())
        if not np.isfinite(cur) or cur > 1e3:
            print(f"[{tag}] diverged at step {n} (t={n*dt:.2f})")
            return
    print(f"[{tag}] survived {steps} steps, final ratio={cur/n0:.3e}")


if __name__ == "__main__":
    original = ec.hllc_normal
    for name, fn in (("hllc", original), ("rusanov", rusanov), ("central", central)):
        ec.hllc_normal = fn
        march(f"p=0 {name}")
        march(f"p=1 {name}", p=1)
    ec.hllc_normal = original