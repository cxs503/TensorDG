"""Instrument the wall boundary flux actually used by the NS DG operator.

Wraps ``_boundary_flux`` to record the *physical* energy component of the total
numerical flux (inviscid - viscous) on the isothermal walls of the exact steady
conduction state, next to the exact value ``-kappa dT/dn``.
"""
from __future__ import annotations

import torch

import tensordg.ns_dg_2d as ns

MU, PRANDTL = 0.1, 0.71
KAPPA = MU * ns.CP / PRANDTL
TB, TT, P0 = 1.0, 0.5, 1.0
RECORD: list[dict] = []


def steady(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    T = TB + (TT - TB) * (y + 0.0 * x)
    return (P0 / T, torch.zeros_like(ref), torch.zeros_like(ref), P0 * torch.ones_like(ref))


def build(ny, degree=1):
    nx = 2
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    return coeff, basis, derivative, weights, signs, dx, dy


_orig = ns._boundary_flux


def spy(*args, **kwargs):
    out = _orig(*args, **kwargs)
    side_name = kwargs.get("side_name", "right")
    axis = kwargs.get("axis", 1)
    RECORD.append({
        "side": side_name,
        "axis": axis,
        "flux": out.clone(),
    })
    return out


ns._boundary_flux = spy


def main():
    print(f"kappa={KAPPA:.6f}  exact energy flux entering divergence"
          f" on bottom = -kappa*dT/dy = {KAPPA * 0.5:.6f}")
    for degree in (1, 2):
        for wg in ("taylor", "face"):
            print(f"\n=== degree={degree} wall_gradient={wg} ===")
            print(f"{'ny':>4} {'E_cold':>9} {'E_hot':>9} {'E_inner':>10}"
                  f" {'Fbot_e':>10} {'Ftop_e':>10} {'fbot[0]':>9} {'ftop[-1]':>9}")
            for ny in (8, 16, 32, 64):
                RECORD.clear()
                coeff, basis, derivative, weights, signs, dx, dy = build(ny, degree)
                pen = 2.0 * (degree + 1.0) ** 2
                boundary = {
                    "xmin": None, "xmax": None,
                    "ymin": {"velocity": (0.0, 0.0), "temperature": TB},
                    "ymax": {"velocity": (0.0, 0.0), "temperature": TT},
                }
                rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                                 (True, False), boundary, MU, KAPPA, pen, wg)
                e = rhs[..., 3]
                cold, hot = float(e[:, 0].abs().max()), float(e[:, -1].abs().max())
                inner = float(e[:, 1:-1].abs().max())
                walls = [r for r in RECORD if r["axis"] == 1]
                bot = [r for r in walls if r["side"] == "left"]
                top = [r for r in walls if r["side"] == "right"]
                fb = float(bot[0]["flux"][..., 3].mean()) if bot else float("nan")
                ft = float(top[0]["flux"][..., 3].mean()) if top else float("nan")
                fb0 = float(bot[0]["flux"][0, 0, 3]) if bot else float("nan")
                ftn = float(top[0]["flux"][-1, -1, 3]) if top else float("nan")
                print(f"{ny:>4} {cold:9.4f} {hot:9.4f} {inner:10.3e}"
                      f" {fb:10.6f} {ft:10.6f} {fb0:9.6f} {ftn:9.6f}")


if __name__ == "__main__":
    main()