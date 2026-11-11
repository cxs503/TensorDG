"""Measure the law the two ``xfail`` pins of ``tests/test_ns_heat_conduction.py``
were written against, in that test file's own configuration.

The pins ask for (a) ``fine < 0.25 * coarse`` for the cold-wall energy residual
between ``ny=16`` and ``ny=128`` and (b) the same residual ``< 1e-6`` at
``ny=32`` for both wall-gradient reconstructions.  Both were documented as
evidence that "the isothermal-wall energy closure is inconsistent".  This script
measures the residual of the exact steady conduction state on the same mesh
family and reports the observed log-log order, so the pins can be re-stated
against the achievable rate ``p - 1`` instead of prose.

Run::

    PYTHONPATH=. python scratch/diag_wall_row_law.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns

MU, PRANDTL = 0.1, 0.71
KAPPA = MU * ns.CP / PRANDTL
T_BOTTOM, T_TOP, PRESSURE = 1.0, 0.5, 1.0
CELLS = (16, 32, 64, 128)
OUT = Path("validation/ns_heat_conduction/wall_row_law.json")


def _steady(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    temperature = T_BOTTOM + (T_TOP - T_BOTTOM) * (y + 0.0 * x)
    return (PRESSURE / temperature, torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def _wall_residual(ny, degree=1, wall_gradient="taylor", nx=2):
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": T_BOTTOM},
                "ymax": {"velocity": (0.0, 0.0), "temperature": T_TOP}}
    return ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                      (True, False), boundary, MU, KAPPA,
                      2.0 * (degree + 1.0) ** 2, wall_gradient)


def _order(cells, values):
    xs = [math.log(c) for c in cells]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return -sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


def main():
    table = {}
    for degree in (1, 2):
        for gradient in ("taylor", "face"):
            rows = []
            for ny in CELLS:
                rhs = _wall_residual(ny, degree, gradient)
                rows.append({"ny": ny,
                             "cold_wall_row": float(rhs[..., 3][:, -1].abs().max()),
                             "hot_wall_row": float(rhs[..., 3][:, 0].abs().max()),
                             "interior_rows": float(rhs[..., 3][:, 1:-1].abs().max())})
            key = f"p{degree}_{gradient}"
            table[key] = {
                "cases": rows,
                "cold_wall_order": _order(CELLS, [r["cold_wall_row"] for r in rows]),
                "ratio_128_over_16": rows[-1]["cold_wall_row"] / rows[0]["cold_wall_row"],
                "claim_pin_a_threshold": 0.25,
                "claim_pin_b_threshold": 1e-6,
                "fails_pin_a": rows[-1]["cold_wall_row"] >= 0.25 * rows[0]["cold_wall_row"],
                "fails_pin_b": rows[1]["cold_wall_row"] >= 1e-6,
            }
    payload = {"mu": MU, "prandtl": PRANDTL, "kappa": KAPPA, "cells": list(CELLS),
               "quantity": "energy residual of the exact steady conduction state, "
                           "max over the cold-wall row modes (np.mean-normalised)",
               "table": table}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    for key, data in table.items():
        print(f"{key}: order={data['cold_wall_order']:+.3f} "
              f"ratio(128/16)={data['ratio_128_over_16']:.3f} "
              + " ".join(f"{r['ny']}:{r['cold_wall_row']:.4f}" for r in data["cases"]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()