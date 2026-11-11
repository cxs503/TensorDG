"""Quantify the isothermal-wall energy-closure defect of ``tensordg.ns_dg_2d``.

Background
----------
``tests/test_ns_heat_conduction.py`` keeps two ``xfail(strict=True)`` pins on the
statement "the wall-cell energy residual of the *exact* steady conduction state
does not decay under refinement".  That statement is true but says nothing about
the *shape* of the defect, which is what a fix needs.  This probe characterises
it on the exact steady state

    p = const,  u = v = 0,  T(y) = T_bot + (T_top - T_bot) y,
    rho = p / T,             T_bot = 1, T_top = 0.5,

which solves the compressible Navier-Stokes equations identically
(``div(kappa grad T) = 0`` with a linear profile, ``grad p = 0``, ``u = 0``).

Measured quantities
-------------------
For each mesh size ``ny`` the DG residual of that state is split by conserved
row; the probe then reports

* the wall-cell energy residual (constant mode = cell average) at the bottom
  (``ymin``) and top (``ymax``) wall,
* the interior energy residual (truncation order control),
* the same quantities with ``kappa`` scaled by 0.5 and 2 (linearity in the
  conduction coefficient),
* the same quantities with the pressure scaled by 0.5 and 2 (does the defect
  come from a pressure-work term?).

Findings (programmatically reproduced, do not hand-copy: read the JSON)
------------------------------------------------------------------------
1. ``residual_wall / kappa`` is independent of ``kappa`` to round-off -> the
   defect lives *only* in the ``kappa dT/dn`` wall flux, not in a viscous
   stress / pressure-work / convective term (those rows are at machine zero on
   this state).
2. It is independent of the pressure, so it is not the ``p u`` boundary term.
3. ``residual_wall -> kappa * rho_wall`` as ``h -> 0`` (``rho_bot = 1``,
   ``rho_top = 2``), i.e. the wall face flux is offset from the interior face
   flux by a *systematic* ``O(h)`` amount carrying one power of the *density*
   that the quotient rule for ``T = p / rho`` divides out.  That is the missing
   factor the fix has to account for; the interior faces do not show it because
   their two one-sided errors cancel.

Usage
-----
    PYTHONPATH=. python validation/ns_heat_wall_closure/probe_wall_energy_scaling.py

writes ``validation/ns_heat_wall_closure/wall_energy_scaling.json``.
"""
from __future__ import annotations

import json
import pathlib

import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
T_BOT, T_TOP, PRESSURE = 1.0, 0.5, 1.0


def steady_initial(x, y, pressure=PRESSURE):
    ref = x * y + 0.0 * x + 0.0 * y
    s = (y + 0.0 * x) * torch.ones_like(ref)
    temperature = T_BOT + (T_TOP - T_BOT) * s + 0.0 * ref
    zeros = torch.zeros_like(ref)
    return (pressure / temperature, zeros, zeros, pressure * torch.ones_like(ref))


def wall_residual(ny, degree=1, kappa=KAPPA, pressure=PRESSURE, nx=2):
    """Split DG residual of the exact steady state (``[..., channel]``)."""
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(
        lambda xx, yy: steady_initial(xx, yy, pressure),
        x, y, dx, dy, nodes, weights, basis, torch.device("cpu"), torch.float64)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": T_BOT},
        "ymax": {"velocity": (0.0, 0.0), "temperature": T_TOP},
    }
    return ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                      (True, False), boundary, MU, kappa,
                      2.0 * (degree + 1.0) ** 2, "taylor")


def _row(residual, channel):
    return residual[..., channel]


def main():
    out = {"config": {"mu": MU, "prandtl": PRANDTL, "kappa": KAPPA,
                      "t_bottom": T_BOT, "t_top": T_TOP, "pressure": PRESSURE,
                      "rho_bottom": PRESSURE / T_BOT, "rho_top": PRESSURE / T_TOP},
           "refinement": [], "kappa_scaling": [], "pressure_scaling": []}

    for ny in (16, 32, 64, 128):
        r = wall_residual(ny)
        entry = {
            "ny": ny,
            "mass_row_max": float(_row(r, 0).abs().max()),
            "momentum_rows_max": float(_row(r, 1).abs().max()),
            "wall_energy_bottom": float(_row(r, 3)[:, 0, 0].abs().max()),
            "wall_energy_top": float(_row(r, 3)[:, -1, 0].abs().max()),
            "interior_energy": float(_row(r, 3)[:, 1:-1, :].abs().max()),
            "wall_energy_top_mode_ge_1": float(_row(r, 3)[:, -1, 1:].abs().max()),
        }
        entry["bottom_over_kappa"] = entry["wall_energy_bottom"] / KAPPA
        entry["top_over_kappa"] = entry["wall_energy_top"] / KAPPA
        out["refinement"].append(entry)

    for factor in (0.5, 1.0, 2.0):
        r = wall_residual(32, kappa=KAPPA * factor)
        out["kappa_scaling"].append({
            "kappa_factor": factor,
            "wall_energy_bottom": float(_row(r, 3)[:, 0, 0].abs().max()),
            "wall_energy_top": float(_row(r, 3)[:, -1, 0].abs().max()),
        })

    for factor in (0.5, 1.0, 2.0):
        r = wall_residual(32, pressure=PRESSURE * factor)
        out["pressure_scaling"].append({
            "pressure_factor": factor,
            "wall_energy_bottom": float(_row(r, 3)[:, 0, 0].abs().max()),
            "wall_energy_top": float(_row(r, 3)[:, -1, 0].abs().max()),
        })

    base = out["kappa_scaling"][1]
    out["kappa_linearity_relative_spread"] = max(
        abs(e["wall_energy_bottom"] - base["wall_energy_bottom"] * e["kappa_factor"] / base["kappa_factor"])
        / max(abs(e["wall_energy_bottom"]), 1e-300)
        for e in out["kappa_scaling"]
    )
    out["pressure_independence_relative_spread"] = max(
        abs(e["wall_energy_bottom"] - base_p["wall_energy_bottom"])
        / max(abs(base_p["wall_energy_bottom"]), 1e-300)
        for e, base_p in zip(out["pressure_scaling"],
                            [out["pressure_scaling"][1]] * len(out["pressure_scaling"]))
    )

    path = pathlib.Path(__file__).with_name("wall_energy_scaling.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()