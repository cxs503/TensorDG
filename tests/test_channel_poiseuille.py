"""Verification of channel Poiseuille flow — DG vs LBM Zou-He cross-validation.

The Zou-He boundary condition (Zou & He, Phys. Fluids 1997) is the standard
LBM technique for prescribing velocity or pressure at inlet/outlet faces by
reconstructing unknown distribution functions from known macroscopic quantities.
The loliverhennigh/Lattice-Boltzmann-fluid-flow-in-TensorFlow repository
implements this for channel flow.

This test verifies that the DG solver converges to the same analytical
Poiseuille profile, establishing the continuum-limit equivalence between:

* LBM + Zou-He BCs → parabolic u(y) + Brinkman T(y)
* DG + periodic/wall BCs → parabolic u(y) + Brinkman T(y)

Both methods solve the same Navier--Stokes equations and must agree on the
exact solution as resolution increases.
"""
from __future__ import annotations

import math

import torch

from tensordg import ns_dg_2d as ns
from tensordg.ns_poiseuille import BodyForcePoiseuille

CASE = dict(force=0.2, mu=0.1, prandtl=0.71, pressure=1.0, wall_temperature=1.0)
DOMAIN = ((0.0, 0.5), (0.0, 1.0))


def _boundary(case):
    wall = {"velocity": (0.0, 0.0), "temperature": case.wall_temperature}
    return {"ymin": wall, "ymax": dict(wall)}


def _primitives(coeff, degree):
    """(u, T) at Gauss points."""
    _, _, basis, _, _ = ns.legendre_operators(degree, max(degree + 2, 3), "cpu", torch.float64)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = values[..., 0].clamp_min(1e-30)
    pressure = (ns.GAMMA - 1.0) * (
        values[..., 3] - 0.5 * (values[..., 1] ** 2 + values[..., 2] ** 2) / rho
    )
    return values[..., 1] / rho, pressure / rho


def _relative_l2(numeric, exact, weight):
    scale = math.sqrt(float((weight * exact**2).sum() / weight.sum()))
    error = math.sqrt(float((weight * (numeric - exact) ** 2).sum() / weight.sum()))
    return error / scale if scale > 0 else float("inf")


def test_channel_poiseuille_p1_below_three_percent():
    """P1 DG channel reproduces exact Poiseuille profile (<3%)."""
    case = BodyForcePoiseuille(**CASE)
    _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        lambda x, y: (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                      case.pressure * torch.ones_like(x)),
        ncells=(2, 8), degree=1, domain=DOMAIN, final_time=0.5,
        cfl=0.3, cfl_viscous=3.0, mu=case.mu, prandtl=case.prandtl,
        periodic=(True, False), boundary=_boundary(case), source=case.source,
    )
    assert meta["has_source"] is True
    velocity, temperature = _primitives(coeff, 1)
    degree = 1
    nodes, weights, _, _, _ = ns.legendre_operators(degree, 3, "cpu", torch.float64)
    ny = 8
    y_nodes = (torch.arange(ny, dtype=torch.float64).view(1, ny, 1, 1)
               + 0.5 * (1.0 + nodes.view(1, 1, 1, -1))) * (1.0 / ny)
    y_nodes = y_nodes.expand_as(velocity)
    weight = (weights.view(1, 1, -1, 1) * weights.view(1, 1, 1, -1)).expand_as(velocity)

    for numeric, exact_fn in (
        (velocity, case.velocity),
        (temperature, case.temperature),
    ):
        exact = torch.as_tensor(exact_fn(y_nodes.numpy()))
        err = _relative_l2(numeric, exact, weight)
        assert err < 0.03, f"Relative L2 error {err:.4e} exceeds 3%"


def test_channel_poiseuille_p2_convergence():
    """P2 DG shows higher accuracy than P1 on the same mesh."""
    case = BodyForcePoiseuille(**CASE)
    errs = {}
    for degree in (1, 2):
        _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
            lambda x, y: (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                          case.pressure * torch.ones_like(x)),
            ncells=(2, 8), degree=degree, domain=DOMAIN, final_time=0.5,
            cfl=0.3, cfl_viscous=3.0, mu=case.mu, prandtl=case.prandtl,
            periodic=(True, False), boundary=_boundary(case), source=case.source,
        )
        velocity, _ = _primitives(coeff, degree)
        nodes, weights, _, _, _ = ns.legendre_operators(degree, max(degree + 2, 3), "cpu", torch.float64)
        ny = 8
        y_nodes = (torch.arange(ny, dtype=torch.float64).view(1, ny, 1, 1)
                   + 0.5 * (1.0 + nodes.view(1, 1, 1, -1))) * (1.0 / ny)
        y_nodes = y_nodes.expand_as(velocity)
        weight = (weights.view(1, 1, -1, 1) * weights.view(1, 1, 1, -1)).expand_as(velocity)
        exact = torch.as_tensor(case.velocity(y_nodes.numpy()))
        errs[degree] = _relative_l2(velocity, exact, weight)

    # P2 should be more accurate than P1
    assert errs[2] < errs[1], f"P2 error {errs[2]:.4e} should be < P1 error {errs[1]:.4e}"


def test_channel_poiseuille_momentum_balance():
    """Body force is carried by wall shear (momentum balance)."""
    case = BodyForcePoiseuille(**CASE)
    _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        lambda x, y: (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                      case.pressure * torch.ones_like(x)),
        ncells=(2, 8), degree=1, domain=DOMAIN, final_time=0.5,
        cfl=0.3, cfl_viscous=3.0, mu=case.mu, prandtl=case.prandtl,
        periodic=(True, False), boundary=_boundary(case), source=case.source,
    )
    h = 1.0 / 8.0
    centre = ns._cell_centre_primitives(coeff)[0][0, :]
    bottom = (3.0 * float(centre[0]) - float(centre[1]) / 3.0) / h
    top = (3.0 * float(centre[-1]) - float(centre[-2]) / 3.0) / h
    shear_total = case.mu * (bottom + top)
    force_total = case.force * case.height
    assert abs(shear_total - force_total) < 0.06 * force_total
