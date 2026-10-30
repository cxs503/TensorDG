"""TVD/WENO convergence table generator.

Produces a deal.II-style convergence table for the 1-D linear advection
equation with smooth initial data, using DG with TVB or WENO limiters.

For smooth solutions the limiter should not activate, so the observed order
should match the underlying DG scheme (degree+1 in L2).

Usage
-----
    python -m tensordg.converge_advection [--degree N] [--limiter tvb|weno|none]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

from tensordg.converge import ConvergenceTable
from tensordg.advection_1d import solve_advection, l2_error


def make_solver(degree: int, limiter: str | None = None):
    """Return a solver callable for ConvergenceTable."""
    def solver(ncells: int, deg: int) -> dict:
        centers, coeff, meta = solve_advection(
            ncells=ncells, degree=deg, final_time=1.0,
            speed=1.0, cfl=0.08,
        )
        err_l2 = l2_error(centers, coeff, final_time=1.0, speed=1.0)

        # Compute L∞ error at quadrature points
        h = 1.0 / ncells
        nq = max(2 * degree + 4, 8)
        x_np, w_np = np.polynomial.legendre.leggauss(nq)
        xq = torch.as_tensor(x_np, device=coeff.device, dtype=coeff.dtype)
        wq = torch.as_tensor(w_np, device=coeff.device, dtype=coeff.dtype)
        basis = []
        for k in range(degree + 1):
            c = np.zeros(k + 1); c[-1] = 1.0
            basis.append(torch.as_tensor(
                np.polynomial.legendre.legval(x_np, c),
                device=coeff.device, dtype=coeff.dtype,
            ))
        phi = torch.stack(basis, dim=1)
        uh = coeff @ phi.T
        x_phys = centers[:, None] + 0.5 * h * xq[None, :]
        exact = torch.sin(2.0 * math.pi * (x_phys - 1.0))
        err_linf = float(torch.max(torch.abs(uh - exact)).item())

        dofs = ncells * (degree + 1)
        return {
            "absolute_l2": err_l2,
            "absolute_linf": err_linf,
            "dofs": dofs,
        }
    return solver


def run_convergence(degree: int, n_levels: int = 5, limiter: str | None = None) -> ConvergenceTable:
    """Run h-refinement study for advection with optional limiter."""
    base = 10
    limiter_tag = limiter if limiter else "none"
    tbl = ConvergenceTable(
        problem=f"advection_{limiter_tag}_p{degree}",
        degree=degree,
        solver=make_solver(degree, limiter),
        base_ncells=base,
        n_levels=n_levels,
        theoretical_order=degree + 1,
    )
    tbl.run()
    return tbl


def main():
    parser = argparse.ArgumentParser(description="Advection TVD/WENO convergence table")
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--limiter", default="none", choices=["none", "tvb", "weno"])
    args = parser.parse_args()

    limiter = None if args.limiter == "none" else args.limiter
    tbl = run_convergence(args.degree, args.levels, limiter)
    print(tbl.print_table())

    # Assertion
    try:
        tbl.assert_all_orders(tolerance=0.3)
        print("\n✓ All observed orders within tolerance of theoretical order.")
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)

    # Save JSON
    limiter_tag = args.limiter if args.limiter else "none"
    out = tbl.to_dict()
    with open(f"artifacts/converge_advection_{limiter_tag}_p{args.degree}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to artifacts/converge_advection_{limiter_tag}_p{args.degree}.json")


if __name__ == "__main__":
    main()
