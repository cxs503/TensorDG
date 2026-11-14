"""Channel flow with prescribed inlet/outlet states — DG analogue of the LBM
Zou-He boundary-condition benchmark.

Physical background
-------------------
Plane Poiseuille flow in a 2-D channel of height ``H``, driven by a constant
pressure gradient ``-dp/dx = G``.  The exact steady solution (incompressible
limit, constant properties) is:

    u(y) = (G / (2 mu)) y (H - y),     u_max = G H^2 / (8 mu),
    v = 0,    p(x) = p_in - G x.

For the compressible DG solver we use a *body-force* formulation that is
mathematically equivalent: a constant volumetric force ``S = G`` in ``+x``
replaces the pressure gradient, and the walls are no-slip isothermal.  The
exact solution is the parabolic velocity profile plus the Brinkman quartic
temperature from viscous heating (see :mod:`tensordg.ns_poiseuille`).

Relation to LBM Zou-He
----------------------
The Zou-He boundary condition (Zou & He, Phys. Fluids 1997) reconstructs
unknown distribution functions at inlet/outlet faces from prescribed
macroscopic quantities (velocity or pressure).  In DG, the equivalent is the
``"state"`` prescribed-state boundary (m16075): the ghost conservative state
is constructed from ``(rho, u, v, p)`` and the HLLC flux carries mass,
momentum and energy across the face while the SIPG penalty enforces the
Dirichlet velocity/temperature for viscous terms.

This module provides the exact reference and a convenience solver for the
channel-with-inlet-outlet case, enabling direct comparison between:

* LBM + Zou-He BCs (loliverhennigh/Lattice-Boltzmann-fluid-flow-in-TensorFlow)
* DG + prescribed-state BCs (this module)

Both must converge to the same analytical Poiseuille profile.
"""
from __future__ import annotations

import numpy as np
import torch

from .ns_poiseuille import BodyForcePoiseuille
from .euler_dg_2d import legendre_operators

__all__ = [
    "ChannelInletOutlet",
    "solve_channel_inlet_outlet",
    "channel_inlet_outlet_initial",
]


class ChannelInletOutlet:
    """Exact reference for a channel with prescribed inlet velocity and outlet pressure."""

    def __init__(self, force=0.2, mu=0.1, prandtl=0.71, pressure=1.0,
                 wall_temperature=1.0, ymin=0.0, ymax=1.0, length=1.0):
        self._poiseuille = BodyForcePoiseuille(
            force=force, mu=mu, prandtl=prandtl, pressure=pressure,
            wall_temperature=wall_temperature, ymin=ymin, ymax=ymax,
        )
        self.length = float(length)

    @property
    def force(self):
        return self._poiseuille.force

    @property
    def mu(self):
        return self._poiseuille.mu

    @property
    def prandtl(self):
        return self._poiseuille.prandtl

    @property
    def pressure(self):
        return self._poiseuille.pressure

    @property
    def wall_temperature(self):
        return self._poiseuille.wall_temperature

    @property
    def ymin(self):
        return self._poiseuille.ymin

    @property
    def ymax(self):
        return self._poiseuille.ymax

    @property
    def height(self):
        return self._poiseuille.height

    @property
    def kappa(self):
        return self._poiseuille.kappa

    @property
    def u_max(self):
        return self._poiseuille.u_max

    @property
    def wall_slope(self):
        return self._poiseuille.wall_slope

    def velocity(self, y):
        return self._poiseuille.velocity(y)

    def temperature(self, y):
        return self._poiseuille.temperature(y)

    def velocity_slope(self, y):
        return self._poiseuille.velocity_slope(y)

    def source(self, x, y, t):
        return self._poiseuille.source(x, y, t)

    def initial(self, x, y):
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        return (ones, zeros, zeros, self.pressure * ones)


def channel_inlet_outlet_initial(pressure=1.0):
    def initial(x, y):
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        return (ones, zeros, zeros, pressure * ones)
    return initial


def _build_prescribed_state(case, ny, n_quad, dtype, device, pressure_offset=0.0):
    """Build prescribed state tensors for inlet/outlet faces.

    The trace at an x-face has shape ``(nx, ny, n_quad, 4)``.  The prescribed
    state must be broadcastable to this shape.  Since the inlet/outlet state
    is x-independent, we use shape ``(1, ny, n_quad)`` for each primitive.
    """
    nodes, _, _, _, _ = legendre_operators(max(n_quad - 1, 0), n_quad, "cpu", dtype)
    nodes = nodes.to(device)
    dy = case.height / ny
    # y at face quadrature points: (cell_idx + 0.5*(1+node)) * dy
    cell_idx = torch.arange(ny, dtype=dtype, device=device).view(1, ny, 1)
    y_q = (cell_idx + 0.5 * (1.0 + nodes.view(1, 1, -1))) * dy  # (1, ny, n_quad)

    u_np = case.velocity(y_q.cpu().numpy())
    T_np = case.temperature(y_q.cpu().numpy())
    u = torch.tensor(u_np, dtype=dtype, device=device)   # (1, ny, n_quad)
    T = torch.tensor(T_np, dtype=dtype, device=device)   # (1, ny, n_quad)

    p = case.pressure - pressure_offset
    rho = p / T  # (1, ny, n_quad)

    return {
        "rho": rho,
        "velocity_x": u,
        "velocity_y": torch.zeros_like(u),
        "pressure": p * torch.ones_like(u),
    }


def solve_channel_inlet_outlet(
    case=None,
    ncells=(16, 8),
    degree=1,
    final_time=0.5,
    cfl=0.3,
    cfl_viscous=3.0,
    device="cpu",
    dtype=torch.float64,
):
    """Solve the channel flow with inlet/outlet prescribed-state BCs."""
    from .ns_dg_2d import solve_navier_stokes_dg_2d

    if case is None:
        case = ChannelInletOutlet()

    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    n_quad = max(degree + 2, 3)

    inlet_state = _build_prescribed_state(case, ny, n_quad, dtype, device)
    p_drop = case.force * case.length
    outlet_state = _build_prescribed_state(case, ny, n_quad, dtype, device, pressure_offset=p_drop)

    wall = {"velocity": (0.0, 0.0), "temperature": case.wall_temperature}

    boundary = {
        "ymin": dict(wall),
        "ymax": dict(wall),
        "xmin": {
            "velocity": (0.0, 0.0),
            "temperature": case.wall_temperature,
            "state": inlet_state,
        },
        "xmax": {
            "velocity": (0.0, 0.0),
            "temperature": case.wall_temperature,
            "state": outlet_state,
        },
    }

    return solve_navier_stokes_dg_2d(
        case.initial,
        ncells=(nx, ny),
        degree=degree,
        domain=((0.0, case.length), (case.ymin, case.ymax)),
        final_time=final_time,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=case.mu,
        prandtl=case.prandtl,
        periodic=(False, False),
        boundary=boundary,
        source=case.source,
        device=device,
        dtype=dtype,
    )
