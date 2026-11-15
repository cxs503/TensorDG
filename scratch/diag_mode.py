"""Locate the positive-eigenvalue mode of the curvilinear Euler operator.

Fully periodic mesh (no boundary condition involved) around a uniform state:
the semi-discrete operator L(delta) = rhs(s0+delta) - rhs(s0) is linearised and
its growth is characterised by (a) per-modal-component energy, (b) Rayleigh
quotient, (c) power iteration.
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


def base(state_kind, x, y):
    if state_kind == "slip":
        return (np.ones_like(x), 0.3 * np.ones_like(x), np.zeros_like(x), np.ones_like(x))
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def linearise(solver, s0, delta):
    return solver.rhs(s0 + delta) - solver.rhs(s0)


def mode_energy(delta, solver):
    """Energy per modal component (k, l, c) in the DG mass norm."""
    m = solver.mass  # (K,K) 1-D mass matrix on [-1,1]
    dens = np.einsum("ak,bd,nabcd->nkcnl", m, m, delta) * 0  # placeholder shape check
    return dens


def main():
    for kind in ("slip", "rest"):
        solver = CurvilinearEuler(periodic_channel(8, 6), 1)
        s0 = solver.project(lambda x, y: base(kind, x, y))
        rng = np.random.default_rng(3)
        delta = rng.standard_normal(s0.shape) * 1e-9
        print(f"\n===== base={kind} (fully periodic 8x6 P1) =====")
        print("rhs(s0) max:", float(np.abs(solver.rhs(s0)).max()))

        # modal decomposition of the growth: which components carry the norm?
        # state layout (ncells, K, K, 4); K=2 for P1: (0,0)=mean,(1,0)=xi,(0,1)=eta
        L = linearise(solver, s0, delta)
        print("delta modal norms per (k,l):")
        for k in range(solver.K):
            for l in range(solver.K):
                print(f"   ({k},{l}) ||delta||={np.linalg.norm(delta[:, k, l]):.3e} "
                      f"||Ldelta||={np.linalg.norm(L[:, k, l]):.3e}")

        # power iteration on L (small dt step, renormalise)
        v = delta / np.linalg.norm(delta)
        for it in range(60):
            w = v + 1e-4 * linearise(solver, s0, v)
            n = np.linalg.norm(w)
            v = w / n
        Lv = linearise(solver, s0, v)
        lam = float(np.sum(v * Lv) / np.sum(v * v))
        print(f"Rayleigh growth rate lambda ~= {lam:.3f} (per unit time)")
        print("dominant mode per (k,l) norm after power iteration:")
        for k in range(solver.K):
            for l in range(solver.K):
                print(f"   ({k},{l}) ||v||={np.linalg.norm(v[:, k, l]):.3e}")
        # structure across cells: is it cell-alternating (k=0 component)?
        print("cell-mean component of v by cell (component rho), grid:",
              np.round(np.abs(v[:, 0, 0, 0]).reshape(8, 6).T, 4))
        print("xi-slope component |v[:,1,0,0]| max:", float(np.abs(v[:, 1, 0, 0]).max()))
        print("eta-slope component |v[:,0,1,0]| max:", float(np.abs(v[:, 0, 1, 0]).max()))


if __name__ == "__main__":
    main()