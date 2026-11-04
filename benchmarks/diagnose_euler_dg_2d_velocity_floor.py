"""Diagnose the velocity-channel error plateau of the 2-D high-order Euler DG vortex.

Motivation
----------
``docs/reports/tensordg-euler-dg-2d-design-order.md`` (m3024) validated P3 only
on ``6^2-16^2`` and left the velocity channels 12-13 % below the design order
``p+1 = 4``; a later mission (m3259) reported an *h-independent* velocity error
floor at ``16^2 -> 24^2`` but never resolved its origin.  This script attacks
the question with two independent knobs that were missing from that study:

1. **Frame (Galilean boost).**  The documented vortex case translates with
   ``u_bg = (1, 1)``; the same vortex with ``u_bg = (0, 0)`` is an equally exact
   steady solution of the same equations on the same mesh.  Comparing the two
   isolates defects that only exist for a *moving* solution.
2. **Quadrature order.**  The nonlinear physical flux of a P3 state is not a
   polynomial, so ``q = degree + 2`` is an under-integrated rule.  Sweeping
   ``q`` separates aliasing from everything else.

For every run the script records the relative ``L^2`` error of
``(rho, u, v, p)`` at the quadrature nodes (the metric used by
``benchmarks/verify_euler_dg_2d.py``), the *domain-mean* (zero-mode) part of
each error and the part left after removing it.

Run
---
    PYTHONPATH=. python -m benchmarks.diagnose_euler_dg_2d_velocity_floor

Writes ``validation/euler_dg_2d/velocity_floor.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tensordg import euler_dg_2d as dg
from tensordg.euler_dg_2d import vortex_primitive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "velocity_floor.json"
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def _reconstruct(coeff, basis):
    return torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)


def measure(cells, degree=3, quadrature=None, final_time=1.0, cfl=0.2,
            background=(1.0, 1.0), strength=5.0):
    """One vortex run plus its node-wise error decomposition."""

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, strength, background)

    started = time.time()
    _, _, coeff, meta = dg.solve_euler_dg_2d(
        initial, ncells=cells, degree=degree, domain=DOMAIN,
        final_time=final_time, cfl=cfl, quadrature=quadrature,
        periodic=(True, True),
    )
    nodes, weights, basis, _, _ = dg.legendre_operators(
        degree, meta["quadrature"]
    )
    nx, ny = coeff.shape[0], coeff.shape[1]
    dx, dy = 10.0 / nx, 10.0 / ny
    x = -5.0 + (torch.arange(nx, dtype=coeff.dtype) + 0.5) * dx
    y = -5.0 + (torch.arange(ny, dtype=coeff.dtype) + 0.5) * dy
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, final_time, strength, background)
    state = _reconstruct(coeff, basis)
    rho = state[..., 0]
    pressure = (dg.GAMMA - 1.0) * (
        state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
    )
    numerical = (rho, state[..., 1] / rho, state[..., 2] / rho, pressure)

    quad_weight = weights[:, None] * weights[None, :]
    cell_measure = 4.0 * nx * ny         # sum of all reference-cell weights
    errors = {}
    for name, num, ref in zip(FIELDS, numerical, exact):
        error = num - ref
        scale = torch.sqrt(
            (ref ** 2 * quad_weight).sum(dim=(2, 3)).sum() / cell_measure
        ).item()
        norm = torch.sqrt(
            (error ** 2 * quad_weight).sum(dim=(2, 3)).sum() / cell_measure
        ).item()
        per_cell_mean = (error * quad_weight).sum(dim=(2, 3)) / quad_weight.sum()
        mean = per_cell_mean.mean().item()
        centred = torch.sqrt(
            ((error - mean) ** 2 * quad_weight).sum(dim=(2, 3)).sum()
            / cell_measure
        ).item()
        errors[name] = {
            "relative_l2": norm / scale,
            "mean": mean,
            "relative_l2_mean": abs(mean) / scale,
            "relative_l2_centred": centred / scale,
        }
    return {
        "cells": cells,
        "degree": degree,
        "quadrature": meta["quadrature"],
        "cfl": cfl,
        "final_time": final_time,
        "background": list(background),
        "steps": meta["steps"],
        "wall_s": round(time.time() - started, 2),
        "errors": errors,
    }


def eoc(coarse, fine, field, key="relative_l2"):
    """Observed order between two rows of the same series."""
    ratio = coarse["errors"][field][key] / fine["errors"][field][key]
    return round(float(torch.log(torch.tensor(ratio))
                        / torch.log(torch.tensor(fine["cells"] / coarse["cells"]))), 3)


def ladder(rows, field, key="relative_l2"):
    return [eoc(rows[i], rows[i + 1], field, key) for i in range(len(rows) - 1)]


def main():
    series = {}
    # Frame discrimination: the two documented exact solutions of the same
    # equations on the same meshes differ only by a Galilean boost.
    for label, background in (("moving", (1.0, 1.0)), ("static", (0.0, 0.0)),
                              ("subsonic_boost", (0.2, 0.2))):
        rows = [measure(cells, background=background) for cells in (12, 16, 20)]
        series[label] = rows
        print(f"--- {label} background={background}")
        for row in rows:
            print(
                f"  cells={row['cells']:>2} q={row['quadrature']} "
                f"steps={row['steps']:>3} wall={row['wall_s']:>6.1f}s "
                + " ".join(
                    f"{name}={row['errors'][name]['relative_l2']:.4e}"
                    for name in FIELDS
                )
            )
        for name in FIELDS:
            print(f"     EOC {name:<11} {ladder(rows, name)}")
    # Quadrature discrimination at a fixed mesh.
    quad_rows = []
    for quadrature in (5, 6, 8):
        row = measure(20, quadrature=quadrature, background=(1.0, 1.0))
        quad_rows.append(row)
        print(
            f"quad q={quadrature} cells=20 "
            + " ".join(f"{n}={row['errors'][n]['relative_l2']:.4e}" for n in FIELDS)
        )
    payload = {
        "case": "2-D isentropic vortex (beta=5), tensor-product modal DG P3, "
                "HLLC, SSP-RK3, t=1.0, CFL 0.2",
        "domain": DOMAIN,
        "series": series,
        "quadrature_sweep": quad_rows,
        "eoc": {
            label: {name: ladder(rows, name) for name in FIELDS}
            for label, rows in series.items()
        },
        "zero_mode": {
            label: [
                {name: row["errors"][name]["mean"] for name in FIELDS}
                for row in rows
            ]
            for label, rows in series.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"json -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()