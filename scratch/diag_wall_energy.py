"""Decompose the isothermal-wall energy residual of the exact steady conduction state.

Diagnostic only: prints, per wall row, the strong-form contributions
(face flux difference, volume inviscid divergence, viscous volume divergence)
and the wall heat-flux estimate used by each ``wall_gradient`` option.
"""
from __future__ import annotations

import torch

import tensordg.ns_dg_2d as ns

MU, PRANDTL = 0.1, 0.71
KAPPA = MU * ns.CP / PRANDTL
TB, TT, P0 = 1.0, 0.5, 1.0


def steady(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    T = TB + (TT - TB) * (y + 0.0 * x)
    return (P0 / T, torch.zeros_like(ref), torch.zeros_like(ref), P0 * torch.ones_like(ref))


def build(ny, degree=1, wall_gradient="taylor"):
    nx = 2
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    return coeff, basis, derivative, weights, signs, dx, dy


def main():
    print(f"kappa = {KAPPA:.6f}   exact dT/dy = {TT - TB}")
    for degree in (1, 2):
        for wg in ("taylor", "face"):
            print(f"\n=== degree={degree}  wall_gradient={wg} ===")
            print(f"{'ny':>4} {'E_cold':>10} {'E_hot':>10} {'E_inner':>10}"
                  f" {'wall_Ty_bot':>12} {'wall_Ty_top':>12} {'inner_Ty_face':>13}")
            for ny in (8, 16, 32, 64, 128):
                coeff, basis, derivative, weights, signs, dx, dy = build(ny, degree, wg)
                pen = 2.0 * (degree + 1.0) ** 2
                boundary = {
                    "xmin": None, "xmax": None,
                    "ymin": {"velocity": (0.0, 0.0), "temperature": TB},
                    "ymax": {"velocity": (0.0, 0.0), "temperature": TT},
                }
                rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                                 (True, False), boundary, MU, KAPPA, pen, wg)
                e = rhs[..., 3]
                # rhs shape: (nx, ny, k, l) modal
                cold = float(e[:, 0].abs().max())
                hot = float(e[:, -1].abs().max())
                inner = float(e[:, 1:-1].abs().max())
                # wall heat-flux estimate actually used: rebuild the face data
                right_y, left_y, _, _ = ns._axis_fluxes_ns(
                    coeff, basis, derivative, signs, weights, dx, dy, False,
                    boundary["ymin"], boundary["ymax"], MU, KAPPA, pen, 1, wg)
                # energy row of the *flux entering the divergence* = inviscid - viscous
                bot = float(left_y[:, 0, 3].mean())
                top = float(right_y[:, -1, 3].mean())
                # interior face flux between cell 0 and 1 (should equal exact flux)
                inner_f = float(right_y[0, :, 3].mean())
                print(f"{ny:>4} {cold:10.4f} {hot:10.4f} {inner:10.4e}"
                      f" {bot:12.6f} {top:12.6f} {inner_f:13.6f}")


if __name__ == "__main__":
    main()