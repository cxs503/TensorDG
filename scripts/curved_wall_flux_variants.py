"""Variant scan: which wall-flux form removes the positive eigenvalue?

Same mesh, same state, only the wall numerical flux is replaced (monkeypatched
*local* to this script): the rest of the operator is HEAD's.  Run with
``PYTHONPATH=. python scripts/curved_wall_flux_variants.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensordg.euler_curved as ec
from tensordg.euler_curved import (
    CurvilinearEuler,
    conserved_from_primitive,
    euler_flux,
    hllc_normal,
    primitive_from_conserved,
)

from curved_wall_stability_scan import channel, spectrum, uniform_slip


def _normal_flux(state, nx, ny, gamma):
    flux_x, flux_y = euler_flux(state, gamma)
    return flux_x * nx[..., None] + flux_y * ny[..., None]


def llf_wall(inner, nx, ny, gamma=1.4):
    rho, u, v, p = primitive_from_conserved(inner, gamma)
    un = u * nx + v * ny
    ut = -u * ny + v * nx
    ghost = conserved_from_primitive(rho, -un * nx - ut * ny, -un * ny + ut * nx, p, gamma)
    sound = np.sqrt(gamma * p / rho)
    alpha = np.abs(un) + sound
    return (0.5 * (_normal_flux(inner, nx, ny, gamma) + _normal_flux(ghost, nx, ny, gamma))
            - 0.5 * alpha[..., None] * (ghost - inner))


def pressure_only_wall(inner, nx, ny, gamma=1.4):
    rho, u, v, p = primitive_from_conserved(inner, gamma)
    out = np.zeros(inner.shape)
    out[..., 1] = p * nx
    out[..., 2] = p * ny
    return out


def full_reflect_wall(inner, nx, ny, gamma=1.4):
    rho, u, v, p = primitive_from_conserved(inner, gamma)
    ghost = conserved_from_primitive(rho, -u, -v, p, gamma)
    return hllc_normal(inner, ghost, nx, ny, gamma)


def freestream_wall(inner, nx, ny, gamma=1.4):
    ghost = conserved_from_primitive(1.0, 0.3, 0.0, 1.0, gamma)
    return hllc_normal(inner, np.broadcast_to(ghost, inner.shape), nx, ny, gamma)


VARIANTS = {
    "V0_mirror_hllc_head": None,
    "V1_llf_mirror": llf_wall,
    "V2_pressure_only": pressure_only_wall,
    "V3_full_reflect": full_reflect_wall,
    "V4_freestream_leak": freestream_wall,
}


def main():
    original = ec.wall_normal_flux
    for name, func in VARIANTS.items():
        ec.wall_normal_flux = original if func is None else func
        mesh = channel(6, 4, periodic=(True, False))
        solver = CurvilinearEuler(mesh, 1, cfl=0.2,
                                  boundaries=("periodic", "periodic", "wall", "wall"))
        state = solver.project(uniform_slip)
        res = spectrum(solver, state)
        print(f"{name:22s} maxRe={res['max_re']:+.4e} "
              f"residual={float(np.abs(solver.rhs(state)).max()):.2e} "
              f"hot={res['hot_cell']:3d} conc={res['hot_fraction']:.2f}")
    ec.wall_normal_flux = original


if __name__ == "__main__":
    main()