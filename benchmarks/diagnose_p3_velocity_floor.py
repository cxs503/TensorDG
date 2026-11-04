"""Decompose the velocity error of the 2-D Euler DG solver into its modes.

Motivation: the temporal-refinement study
(``benchmarks/temporal_asymptotics_euler_dg_2d.py``) shows that on the smooth
isentropic vortex the density and pressure channels of P3 converge at the design
order ``h^4`` from 16^2 to 24^2, while the velocity channels only improve by a
factor ~1.35 (order ~0.7) -- and the deficit survives every dt refinement, so it
is spatial.  This script asks *what kind* of error it is:

* conserved-variable errors ``(rho, rho u, rho v, E)`` versus primitive ones,
  because a discrepancy between them localises the defect in one channel;
* the Helmholtz split of the velocity error on the cell-centre lattice: its
  divergence (compressible part) and its curl (rotational part).  A purely
  rotational error means the extra error is a spurious vortical mode rather than
  a spurious acoustic one;
* where the largest velocity error sits (interior or the periodic seam).

The cell-centre lattice is the central Gauss node of each element, which for the
odd node counts used here sits exactly at the cell centre, so the lattice is
uniform with spacing ``dx`` and its Fourier derivatives are spectral (no finite
difference error).

Run::

    PYTHONPATH=. python -m benchmarks.diagnose_p3_velocity_floor --meshes 12 16 24 32

Outputs ``validation/euler_dg_2d/velocity_floor.json`` (the numbers used in
``docs/reports/tensordg-p3-asymptotic-order.md``).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _primitive, _reconstruct, vortex_primitive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "velocity_floor.json"
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
GAMMA = 1.4


def _nodes(ncells, nodes):
    """Cell centres, Gauss node abscissae, weights (``weight`` is 2-D)."""
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / ncells, (ymax - ymin) / ncells
    x = xmin + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dy
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    w = torch.as_tensor(np.polynomial.legendre.leggauss(nodes.numel())[1], dtype=torch.float64)
    return dx, dy, xq, yq, w[:, None] * w[None, :]


def _rms(field):
    return float(torch.sqrt((field ** 2).mean()))


def diagnose(ncells, degree=3, final_time=1.0, cfl=0.2):
    nodes, _w, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    started = time.time()
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time, cfl=cfl
    )
    dx, dy, xq, yq, weight = _nodes(ncells, nodes)
    rho_e, u_e, v_e, p_e = vortex_primitive(xq, yq, final_time)
    rho, u, v, p = _primitive(_reconstruct(coeff, basis))

    def rel_l2(num, ref):
        return float(torch.sqrt(((num - ref) ** 2 * weight).sum() / (ref ** 2 * weight).sum()))

    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    energy_e = p_e / (GAMMA - 1.0) + 0.5 * rho_e * (u_e ** 2 + v_e ** 2)
    errors = {
        "density": rel_l2(rho, rho_e),
        "pressure": rel_l2(p, p_e),
        "momentum_x": rel_l2(rho * u, rho_e * u_e),
        "momentum_y": rel_l2(rho * v, rho_e * v_e),
        "energy": rel_l2(energy, energy_e),
        "velocity_x": rel_l2(u, u_e),
        "velocity_y": rel_l2(v, v_e),
    }

    # Cell-centre lattice (central Gauss node == cell centre for odd node counts).
    j = nodes.numel() // 2
    du = (u - u_e)[:, :, j, j]
    dv = (v - v_e)[:, :, j, j]
    n = ncells
    big_n = float(n * n)
    du_hat = torch.fft.fft2(du)
    dv_hat = torch.fft.fft2(dv)
    kx = 2.0 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64)[:, None]
    ky = 2.0 * math.pi * torch.fft.fftfreq(n, d=dy, dtype=torch.float64)[None, :]
    div = 1j * (kx * du_hat + ky * dv_hat)
    curl = 1j * (kx * dv_hat - ky * du_hat)
    div_rms = float(torch.sqrt((div.abs() ** 2).sum() / big_n))
    curl_rms = float(torch.sqrt((curl.abs() ** 2).sum() / big_n))
    speed_err = _rms(torch.sqrt((u - u_e) ** 2 + (v - v_e) ** 2))

    centre = (u - u_e)[:, :, j, j]
    peak = np.unravel_index(int(centre.abs().argmax()), tuple(centre.shape))
    return {
        "degree": degree,
        "cells": ncells,
        "h": 10.0 / ncells,
        "steps": int(metadata["steps"]),
        "errors": errors,
        "velocity_error_rms_centre": speed_err,
        "divergence_rms": div_rms,
        "curl_rms": curl_rms,
        "rotational_fraction": curl_rms / (curl_rms + div_rms) if (curl_rms + div_rms) else None,
        "max_abs_velocity_error": float((u - u_e).abs().max()),
        "max_location_cell_centre": [int(peak[0]), int(peak[1])],
        "wall_seconds": round(time.time() - started, 1),
    }


def _add_eoc(rows):
    for i in range(1, len(rows)):
        rows[i]["eoc"] = {
            key: float(math.log(rows[i - 1]["errors"][key] / rows[i]["errors"][key])
                       / math.log(rows[i - 1]["h"] / rows[i]["h"]))
            for key in rows[i]["errors"]
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[12, 16, 24, 32])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--cfl", type=float, default=0.2)
    args = parser.parse_args()

    rows = []
    for ncells in args.meshes:
        row = diagnose(ncells, args.degree, cfl=args.cfl)
        rows.append(row)
        _add_eoc(rows)
        print(f"P{row['degree']} {ncells:3d}^2  "
              + "  ".join(f"{k} {v:.4e}" for k, v in row["errors"].items())
              + f"  rot_frac {row['rotational_fraction']:.3f} ({row['wall_seconds']}s)",
              flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"benchmark": "euler_dg_2d_velocity_floor",
                                   "cfl": args.cfl, "domain": DOMAIN,
                                   "rows": rows}, indent=1))

    print("\nper-interval EOC")
    for row in rows[1:]:
        print(f"-> {row['cells']}^2: " + " ".join(f"{k} {v:.2f}"
                                                  for k, v in row["eoc"].items()))
    print(f"json -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()