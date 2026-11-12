"""Fault injection: swap the incoming/outgoing characteristic at the ends.

The characteristic boundary flux of ``tensordg/elastic_1d.py`` pairs the
*entering* Riemann invariant (``w+`` at ``x=0``, ``w-`` at ``x=L``) with the
invariant supplied by the interior trace.  Swapping the two ('right' pairing
used on the left end and vice versa) silently turns both reflecting ends into
amplifying ones: the discrete energy grows without bound while the code still
looks like it is solving the elastic wave equation.

This script runs the clamped-free eigenmode case with the correct and with the
swapped pairing and records the energy growth, so that the accuracy/energy gate
of ``tests/test_elastic_1d_dalembert.py`` can be shown to have teeth (a scheme
with the wrong characteristic pairing cannot pass it).

Usage::

    python -m benchmarks.elastic_1d_bc_fault_injection --output-dir validation/elastic_1d
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import tensordg.elastic_1d as elastic


def run(ncells: int, degree: int, final_time: float, cfl: float, swap: bool):
    original = elastic._boundary_flux
    if swap:
        def swapped(v_int, s_int, side, rho, young, bc):
            other = "right" if side == "left" else "left"
            return original(v_int, s_int, other, rho, young, bc)

        elastic._boundary_flux = swapped
    try:
        centers, coeff, meta = elastic.solve_elastic_1d(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
            bc="fixed-free")
    finally:
        elastic._boundary_flux = original
    k = math.pi / 2.0
    energy_exact = 0.25 * k ** 2
    energy = elastic.elastic_energy(centers, coeff)
    return {
        "cells": ncells,
        "steps": meta["steps"],
        "energy": energy,
        "energy_exact": energy_exact,
        "energy_ratio": energy / energy_exact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=40)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--output-dir", default="validation/elastic_1d")
    args = parser.parse_args()

    reference = run(args.cells, args.degree, args.final_time, args.cfl, False)
    fault = run(args.cells, args.degree, args.final_time, args.cfl, True)
    payload = {
        "case": "clamped-free eigenmode, characteristic pairing fault injection",
        "parameters": {
            "cells": args.cells, "degree": args.degree,
            "final_time": args.final_time, "cfl": args.cfl,
            "rho": 1.0, "young": 1.0,
        },
        "correct_pairing": reference,
        "swapped_pairing": fault,
        "detected": bool(fault["energy_ratio"] > 100.0 * reference["energy_ratio"]),
        "note": "energy of the discrete solution / analytic eigenmode energy; "
                "the wrong pairing feeds the outgoing invariant back into the "
                "domain and the amplitude grows geometrically in time.",
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bc_fault_injection.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"json: {path}")
    return 0 if payload["detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())