"""Verification of the body-force (``source``) pathway of ``ns_dg_2d``.

The ``source`` hook of :func:`tensordg.ns_dg_2d.solve_navier_stokes_dg_2d` had no
test before mission m6454.  These tests pin it down against a closed-form
solution of the compressible Navier-Stokes equations, plane Poiseuille flow
driven by a constant volumetric force (:mod:`tensordg.ns_poiseuille`).

Cheap invariants first (no solver): the reference profile solves
``mu u'' + S = 0`` and ``k T'' = S u - mu (u')^2`` exactly, and the initial
acceleration of a fluid at rest is ``S/rho``.  Then one short marched case
checks that the DG channel converges to the reference profile and that the
momentum balance (body force vs the shear on both walls) closes.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tensordg import ns_dg_2d as ns
from tensordg.ns_poiseuille import BodyForcePoiseuille

CASE = dict(force=0.2, mu=0.1, prandtl=0.71, pressure=1.0, wall_temperature=1.0)
DOMAIN = ((0.0, 0.5), (0.0, 1.0))


def _boundary(case):
    wall = {"velocity": (0.0, 0.0), "temperature": case.wall_temperature}
    return {"ymin": wall, "ymax": dict(wall)}


def _primitives(coeff):
    """``(u, T)`` at the Gauss points of a DG state (x-averaged not applied)."""
    degree = coeff.shape[2] - 1
    _, _, basis, _, _ = ns.legendre_operators(degree, max(degree + 2, 3), "cpu",
                                             torch.float64)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = values[..., 0].clamp_min(1e-30)
    pressure = (ns.GAMMA - 1.0) * (
        values[..., 3] - 0.5 * (values[..., 1] ** 2 + values[..., 2] ** 2) / rho
    )
    return values[..., 1] / rho, pressure / rho, values[..., 2] / rho


def test_exact_profile_satisfies_the_steady_navier_stokes_equations():
    """``mu u'' + S = 0`` and ``k T'' = S u - mu u'^2`` hold for the reference."""
    case = BodyForcePoiseuille(**CASE)
    step = 1.0e-3
    for y in (0.15, 0.5, 0.83):
        u_yy = (case.velocity(y + step) - 2.0 * case.velocity(y)
                + case.velocity(y - step)) / step**2
        assert abs(case.mu * u_yy + case.force) < 1e-5 * case.force
        t_yy = (case.temperature(y + step) - 2.0 * case.temperature(y)
                + case.temperature(y - step)) / step**2
        balance = case.force * case.velocity(y) - case.mu * case.velocity_slope(y) ** 2
        scale = case.force * case.u_max + case.mu * case.wall_slope**2
        assert abs(case.kappa * t_yy - balance) < 1e-5 * scale
    # both walls are no-slip and isothermal, the pressure is uniform
    walls = torch.tensor([0.0, 1.0])
    assert float(np.abs(case.velocity(walls)).max()) < 1e-14
    assert float(np.abs(case.temperature(walls) - case.wall_temperature).max()) < 1e-14
    # the quartic temperature polynomial is symmetric about the mid-plane and
    # cools the core (the workless force removes energy that dissipation adds)
    assert float(case.temperature(np.array([0.5]))[0]) < case.wall_temperature
    assert abs(float(case.temperature(np.array([0.2]))[0])
               - float(case.temperature(np.array([0.8]))[0])) < 1e-12


def test_body_force_gives_the_expected_initial_acceleration():
    """At ``t -> 0`` the fluid at rest accelerates at ``S/rho`` (source sign/units)."""
    case = BodyForcePoiseuille(**CASE)
    def initial(x, y):
        return (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                case.pressure * torch.ones_like(x))

    final_time = 0.002
    _, _, coeff, meta = ns.solve_navier_stokes_dg_2d(
        initial, ncells=(2, 8), degree=1, domain=DOMAIN, final_time=final_time,
        cfl=0.3, cfl_viscous=3.0, mu=case.mu, prandtl=case.prandtl,
        periodic=(True, False), boundary=_boundary(case), source=case.source)
    velocity, _, velocity_y = _primitives(coeff)
    expected = case.force * meta["final_time_reached"]
    assert abs(velocity.mean().item() - expected) < 0.05 * expected
    assert float(velocity.min()) > 0.0                      # forced in +x only
    assert float(velocity_y.abs().max()) < 1e-8             # no cross-flow


def test_channel_converges_to_the_exact_poiseuille_profile():
    """The DG channel reproduces the exact parabolic/brinkman state (<3 %)."""
    case = BodyForcePoiseuille(**CASE)
    _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        case.initial, ncells=(2, 8), degree=1, domain=DOMAIN, final_time=0.5,
        cfl=0.3, cfl_viscous=3.0, mu=case.mu, prandtl=case.prandtl,
        periodic=(True, False), boundary=_boundary(case), source=case.source)
    assert meta["has_source"] is True
    velocity, temperature, _ = _primitives(coeff)
    degree = 1
    nodes, weights, _, _, _ = ns.legendre_operators(degree, 3, "cpu", torch.float64)
    y_nodes = (torch.arange(8, dtype=torch.float64).view(1, 8, 1, 1)
               + 0.5 * (1.0 + nodes.view(1, 1, 1, -1))) * (1.0 / 8.0)
    y_nodes = y_nodes.expand_as(velocity)
    weight = (weights.view(1, 1, -1, 1) * weights.view(1, 1, 1, -1)).expand_as(velocity)
    for numeric, exact in (
        (velocity, torch.as_tensor(case.velocity(y_nodes.numpy()))),
        (temperature, torch.as_tensor(case.temperature(y_nodes.numpy()))),
    ):
        scale = math.sqrt(float((weight * exact**2).sum() / weight.sum()))
        error = math.sqrt(float((weight * (numeric - exact) ** 2).sum() / weight.sum()))
        assert error / scale < 0.03, (error / scale, "above the 3% gate")

    # momentum balance: the body force is carried by the shear on both walls
    h = 1.0 / 8.0
    centre = ns._cell_centre_primitives(coeff)[0][0, :]
    bottom = (3.0 * float(centre[0]) - float(centre[1]) / 3.0) / h
    top = (3.0 * float(centre[-1]) - float(centre[-2]) / 3.0) / h
    shear_total = case.mu * (bottom + top)
    force_total = case.force * case.height
    assert abs(shear_total - force_total) < 0.06 * force_total
    assert float(temperature.min()) > 0.98 * case.wall_temperature