"""Bit-for-bit compatibility check of the non-periodic limiter path.

Compares the *pre-fix* package (extracted from git into ``--old``) with the
current working tree on three open-domain Sod runs (TVB / WENO / no limiter)
and one ``boundary="transmissive"`` periodic-solver run.  Any difference means
the periodic-stencil change leaked into a non-periodic code path.

    mkdir -p /tmp/oldpkg/tensordg
    for f in tensordg/*.py; do git show HEAD:$f > /tmp/oldpkg/$f; done
    PYTHONPATH=. python tools/cmp_periodic_seam_compat.py --old /tmp/oldpkg
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _initial_state(x):
    rho = np.where(np.mod(x.detach().cpu().numpy(), 1.0) < 0.5, 2.0, 1.0)
    return (torch.as_tensor(rho, dtype=x.dtype).reshape(x.shape),
            torch.ones_like(x), torch.ones_like(x))


def run(root: str):
    sys.path.insert(0, root)
    for name in [m for m in sys.modules if m.startswith("tensordg")]:
        del sys.modules[name]
    from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

    out = []
    for limiter in ("tvb", "weno", None):
        _, coeff, _ = solve_euler_dg_1d(
            (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=120, degree=2,
            final_time=0.1, cfl=0.4, limiter=limiter, tvb_constant=1.0)
        out.append((f"sod_{limiter}", coeff.detach().cpu().numpy()))
    _, coeff, _ = solve_periodic_euler_dg_1d(
        _initial_state, ncells=48, degree=2, final_time=0.02, limiter="tvb",
        boundary="transmissive")
    out.append(("transmissive_path_tvb", coeff.detach().cpu().numpy()))
    sys.path.pop(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="/tmp/oldpkg")
    args = parser.parse_args()
    old = dict(run(args.old))
    new = dict(run("/root/tp-agent/work/m3366"))
    ok = True
    for key in old:
        same = np.array_equal(old[key], new[key])
        diff = float(np.abs(old[key] - new[key]).max())
        ok &= same
        print(f"{key:26s} bit-identical={same} max|diff|={diff:.3e}")
    print("VERDICT:", "non-periodic path unchanged" if ok else "REGRESSION")


if __name__ == "__main__":
    main()