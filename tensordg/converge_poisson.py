"""SIPG Poisson convergence table generator.

Produces a deal.II-style convergence table for the manufactured Poisson
problem with SIPG discretisation at degrees 1, 2, 3.

Usage
-----
    python -m tensordg.converge_poisson [--degree N] [--levels N] [--backend dense|sparse]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from tensordg.converge import ConvergenceTable
from tensordg.poisson_2d import solve_poisson_2d, poisson_error_2d


def make_solver(degree: int, backend: str = "dense"):
    """Return a solver callable for ConvergenceTable."""
    def solver(ncells: int, deg: int) -> dict:
        x, y, coeff, meta = solve_poisson_2d(
            ncells=ncells, degree=deg, backend=backend,
        )
        err = poisson_error_2d(x, y, coeff)
        err["dofs"] = meta["dofs"]
        return err
    return solver


def run_convergence(degree: int, n_levels: int = 5, backend: str = "dense") -> ConvergenceTable:
    """Run h-refinement study for SIPG Poisson at given degree."""
    base = 4 if degree <= 2 else 8
    tbl = ConvergenceTable(
        problem=f"poisson_sipg_p{degree}",
        degree=degree,
        solver=make_solver(degree, backend),
        base_ncells=base,
        n_levels=n_levels,
        theoretical_order=degree + 1,
    )
    tbl.run()
    return tbl


def main():
    parser = argparse.ArgumentParser(description="SIPG Poisson convergence table")
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--backend", default="dense", choices=["dense", "sparse"])
    args = parser.parse_args()

    tbl = run_convergence(args.degree, args.levels, args.backend)
    print(tbl.print_table())

    # Assertion
    try:
        tbl.assert_all_orders(tolerance=0.3)
        print("\n✓ All observed orders within tolerance of theoretical order.")
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)

    # Save JSON
    out = tbl.to_dict()
    with open(f"artifacts/converge_poisson_p{args.degree}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to artifacts/converge_poisson_p{args.degree}.json")


if __name__ == "__main__":
    main()
