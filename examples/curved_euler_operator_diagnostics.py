"""Diagnostics: curved-mesh Euler DG operator — order gap and blow-up attribution.

Two questions, both answered with public API only (no private members touched):

E1  Is the reported reflective-wall instability actually a *boundary* effect?
    Same uniform base state, four boundary combinations, including a **fully
    periodic** mesh with no wall at all.

E2  Does the numerical flux drive it?
    ``hllc_normal`` is replaced by Rusanov and by a central flux (monkeypatch).

E3  Is it a high-order artefact?
    The same march at p = 0 (element-wise finite volume), p = 1, p = 2.

E4  Does a *stationary* (analytic) solution survive?
    A 1e-6 density perturbation with constant pressure and zero velocity is an
    exact steady state of the Euler equations (entropy wave at rest).

Runtime ~1 min on 1 core.  See docs/CASE-DG-CURVED-DESIGN-ORDER-RED.md.
"""
from __future__ import annotations

import numpy as np

import tensordg.euler_curved as ec
from tensordg.curved_geometry import CurvedMesh, IsoparametricMap

GAMMA = ec.GAMMA


def mesh(nx, ny, periodic=(True, True), walls=(False,) * 4, lx=4.0, ly=1.0):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=walls)


def rest(x, y):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def entropy_wave(x, y):
    """Exact steady state: rho perturbed, p and velocity untouched."""
    return (1.0 + 1e-6 * np.sin(np.pi * x) * np.sin(2 * np.pi * y),
            np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def rusanov(left, right, nx, ny, gamma=GAMMA):
    flx, fly = ec.euler_flux(left, gamma)
    frx, fry = ec.euler_flux(right, gamma)
    fl = flx * nx[..., None] + fly * ny[..., None]
    fr = frx * nx[..., None] + fry * ny[..., None]
    rl = np.maximum(left[..., 0], 1e-12)
    rr = np.maximum(right[..., 0], 1e-12)
    ul = (left[..., 1] / rl, left[..., 2] / rl)
    ur = (right[..., 1] / rr, right[..., 2] / rr)
    pl = np.maximum((gamma - 1.0) * (left[..., 3] - 0.5 * rl * (ul[0] ** 2 + ul[1] ** 2)), 1e-12)
    pr = np.maximum((gamma - 1.0) * (right[..., 3] - 0.5 * rr * (ur[0] ** 2 + ur[1] ** 2)), 1e-12)
    alpha = np.maximum(np.abs(ul[0] * nx + ul[1] * ny) + np.sqrt(gamma * pl / rl),
                       np.abs(ur[0] * nx + ur[1] * ny) + np.sqrt(gamma * pr / rr))
    return 0.5 * (fl + fr) - 0.5 * alpha[..., None] * (right - left)


def central(left, right, nx, ny, gamma=GAMMA):
    flx, fly = ec.euler_flux(left, gamma)
    frx, fry = ec.euler_flux(right, gamma)
    return 0.5 * ((flx + frx) * nx[..., None] + (fly + fry) * ny[..., None])


def march(tag, p=1, steps=200, periodic=(True, True), walls=(False,) * 4,
          perturbation="entropy", dt_factor=1.0):
    solver = ec.CurvilinearEuler(mesh(8, 6, periodic, walls), p)
    s0 = solver.project(rest)
    if perturbation == "entropy":
        state = solver.project(entropy_wave)
    else:
        rng = np.random.default_rng(3)
        state = s0 + rng.standard_normal(s0.shape) * 1e-6
    dt = solver.stable_dt(state) * dt_factor
    n0 = float(np.abs(state - s0).max())
    for n in range(1, steps + 1):
        state = solver.ssp_rk3_step(state, dt)
        cur = float(np.abs(state - s0).max())
        if not np.isfinite(cur) or cur > 1e3:
            print(f"[{tag}] diverged at step {n} (t={n * dt:.3f}), dt={dt:.5f}")
            return n
    print(f"[{tag}] survived {steps} steps (t={steps * dt:.3f}), "
          f"final ratio={cur / n0:.3e}, dt={dt:.5f}")
    return None


def main():
    print("== E1 boundary attribution (entropy wave, p=1) ==")
    march("E1a fully periodic", periodic=(True, True))
    march("E1b periodic-x + walls-y", periodic=(True, False),
          walls=(False, False, True, True))
    march("E1c outflow-x + walls-y", periodic=(False, False),
          walls=(False, False, True, True))
    march("E1d periodic-y, open-x", periodic=(False, True))

    print("\n== E2 flux attribution (entropy wave, p=1, fully periodic) ==")
    original = ec.hllc_normal
    for name, fn in (("hllc", original), ("rusanov", rusanov), ("central", central)):
        ec.hllc_normal = fn
        march(f"E2 {name}")
    ec.hllc_normal = original

    print("\n== E3 polynomial order (entropy wave, fully periodic) ==")
    for p in (0, 1, 2):
        march(f"E3 p={p}", p=p)

    print("\n== E4 random modal noise vs stationary entropy wave (p=0) ==")
    march("E4 random", p=0, perturbation="random")
    march("E4 entropy", p=0)
    march("E4 entropy dt/4", p=0, dt_factor=0.25)


if __name__ == "__main__":
    main()