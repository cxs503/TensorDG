"""Decompose the strong-form residual of the exact steady conduction state term by term."""
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


def terms(ny, degree=1, wg="taylor"):
    nx = 2
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    modes = torch.arange(degree + 1, dtype=torch.float64)
    scale = torch.outer(2 * modes + 1, 2 * modes + 1)
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    volume_x = torch.einsum("xyabv,a,b,ak,bl->xyklv", ns._physical_flux(at_quad, 0),
                            weights, weights, derivative, basis)
    volume_y = torch.einsum("xyabv,a,b,ak,bl->xyklv", ns._physical_flux(at_quad, 1),
                            weights, weights, basis, derivative)
    viscous_x, viscous_y = ns._viscous_volume(coeff, basis, derivative, weights, dx, dy, MU, KAPPA)
    pen = 2.0 * (degree + 1.0) ** 2
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": TB},
                "ymax": {"velocity": (0.0, 0.0), "temperature": TT}}
    rx, lx, _, _ = ns._axis_fluxes_ns(coeff, basis, derivative, signs, weights, dx, dy,
                                      True, boundary["xmin"], boundary["xmax"], MU, KAPPA, pen, 0, wg)
    ry, ly, _, _ = ns._axis_fluxes_ns(coeff, basis, derivative, signs, weights, dx, dy,
                                      False, boundary["ymin"], boundary["ymax"], MU, KAPPA, pen, 1, wg)
    for name, arr in (("right_y", ry), ("left_y", ly), ("volume_y", volume_y),
                      ("viscous_y", viscous_y), ("right_x", rx), ("volume_x", volume_x),
                      ("viscous_x", viscous_x)):
        print(f"  {name:>10} shape {tuple(arr.shape)}")
    sx = (scale / (2.0 * dx)).view(1, 1, *scale.shape, 1)
    sy = (scale / (2.0 * dy)).view(1, 1, *scale.shape, 1)
    ysign = signs.view(1, 1, 1, -1, 1)
    fx = rx.unsqueeze(2) - lx.unsqueeze(2) * signs.view(1, 1, -1, 1, 1) - volume_x + viscous_x
    fy = ry.unsqueeze(3) - ly.unsqueeze(3) * ysign - volume_y + viscous_y
    res = -(sx * fx) - (sy * fy)
    e = res[..., 3]
    # component-level, energy row
    print(f"  energy row |x-part| max {float((-(sx*fx))[...,3].abs().max()):.3e}"
          f"  |y-part| max {float((-(sy*fy))[...,3].abs().max()):.3e}")
    for tag, cell in (("cold(ny=0)", 0), ("hot(ny=-1)", -1)):
        row = fy[:, cell, ..., 3]
        print(f"  {tag}: raw fy modal l=0 {row[:, 0, 0].tolist()}")
        parts = {
            "right_y": -(sy * ry.unsqueeze(3))[:, cell, ..., 3],
            "-left_y*signs": +(sy * ly.unsqueeze(3) * ysign)[:, cell, ..., 3],
            "-volume_y": +(sy * volume_y)[:, cell, ..., 3],
            "+viscous_y": -(sy * viscous_y)[:, cell, ..., 3],
        }
        for key, val in parts.items():
            print(f"      {key:>16}: l=0 {val[:, 0, 0].tolist()}")
    return e


if __name__ == "__main__":
    for ny in (16,):
        print(f"### ny={ny} degree=1 taylor")
        terms(ny)