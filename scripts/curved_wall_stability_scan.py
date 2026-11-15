"""Localise the linear instability of the curvilinear DG boundary treatment.

The module ``tensordg.euler_curved`` honestly records that an explicit SSP-RK3
march of an *exact* steady state blows up after ~700 steps, and that the CKS1998
"flow past a circular cylinder" case is therefore not reproduced.  This script
answers *which boundary family* carries the positive real eigenvalue, by

1. assembling the finite-difference Jacobian of ``rhs`` about a uniform slip
   state for several boundary-kind combinations on the same mesh, and
2. reporting ``max Re eig`` plus where the corresponding eigenvector lives.

No solver code is modified here: the operator is probed through its public API
(``rhs``), so the numbers it writes are a property of HEAD, not of a patch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import CurvilinearEuler

OUT = Path(__file__).resolve().parents[1] / "validation" / "curved_wall_stability"


def channel(nx, ny, lx=4.0, ly=1.0, periodic=(False, False), degree=1):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, degree)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=(False, False, True, True))


def uniform_slip(x, y):
    return (np.ones_like(x), 0.3 * np.ones_like(x), np.zeros_like(x),
            np.ones_like(x))


def fd_jacobian(solver, state, eps=1.0e-8):
    """Dense Jacobian of ``solver.rhs`` about ``state`` (modal dof ordering)."""
    n = state.size
    jac = np.zeros((n, n))
    base = solver.rhs(state).reshape(-1)
    for k in range(n):
        perturbed = state.copy()
        perturbed.reshape(-1)[k] += eps
        jac[:, k] = (solver.rhs(perturbed).reshape(-1) - base) / eps
    return jac


def spectrum(solver, state, top=6):
    jac = fd_jacobian(solver, state)
    vals, vecs = np.linalg.eig(jac)
    order = np.argsort(-vals.real)
    lead = order[0]
    amp = np.abs(vecs[:, lead])
    cells = amp.reshape(solver.mesh.ncells, -1).max(axis=1)
    worst = int(np.argmax(cells))
    return {
        "max_re": float(vals.real.max()),
        "max_abs_re": float(np.abs(vals.real).max()),
        "top_re": [float(v) for v in vals.real[order[:top]]],
        "top_im": [float(v) for v in vals.imag[order[:top]]],
        "dofs": int(state.size),
        "hot_cell": worst,
        "hot_fraction": float(cells[worst] / max(cells.mean(), 1e-300)),
    }


COMBOS = {
    "periodic_both": dict(periodic=(True, True),
                          boundaries=("periodic", "periodic", "periodic", "periodic")),
    "wall_y_only": dict(periodic=(True, False),
                        boundaries=("periodic", "periodic", "wall", "wall")),
    "outflow_wall_y": dict(periodic=(False, False),
                           boundaries=("outflow", "outflow", "wall", "wall")),
    "farfield_wall_y": dict(periodic=(False, False),
                            boundaries=("farfield", "farfield", "wall", "wall")),
    "wall_both": dict(periodic=(False, False),
                      boundaries=("wall", "wall", "wall", "wall")),
}


def main() -> dict:
    nx, ny, p = 6, 4, 1
    out = {"mesh": f"{nx}x{ny}", "degree": p, "combos": {}}
    for name, spec in COMBOS.items():
        mesh = channel(nx, ny, periodic=spec["periodic"])
        needs_ff = any(k == "farfield" for k in spec["boundaries"])
        solver = CurvilinearEuler(
            mesh, p, cfl=0.2, boundaries=spec["boundaries"],
            farfield=(1.0, 0.3, 0.0, 1.0) if needs_ff else None)
        state = solver.project(uniform_slip)
        res = spectrum(solver, state)
        res["steady_residual"] = float(np.abs(solver.rhs(state)).max())
        out["combos"][name] = res
        print(f"{name:16s} maxRe={res['max_re']:+.4e} "
              f"residual={res['steady_residual']:.2e} "
              f"hot_cell={res['hot_cell']:3d} amp/concentration={res['hot_fraction']:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spectrum.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()