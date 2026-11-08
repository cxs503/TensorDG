"""Exact steady plane-Poiseuille flow driven by a volumetric body force.

The 2-D compressible Navier--Stokes channel of :mod:`tensordg.ns_dg_2d` has a
``source`` hook (``source(x, y, t) -> (4,)`` conserved-variable source per unit
volume) that, until ``docs/ns_couette_poiseuille_verification.md`` (mission
m6454), had no verification case at all.  This module supplies the exact
reference needed for it: a *pressure-gradient-free* plane Poiseuille flow in
which the driving force is applied volumetrically.

Derivation
----------
Let the channel be ``ymin <= y <= ymax`` (height ``H``), periodic in ``x``, with
no-slip isothermal walls at ``T_w``, and let the source be a constant force ``S``
per unit volume in ``+x`` with **zero** energy source (a *workless* body force,
i.e. an MMS-style manufactured exact solution; the physical body force that does
the work ``S u`` is the same momentum balance with a different temperature
profile).  Assume ``u = (u(y), 0)``, ``rho = rho(y)``, ``p = const``:

* continuity ``d(rho u)/dx = 0``                        -> satisfied;
* y-momentum  ``dp/dy = 0``                             -> ``p = const``;
* x-momentum  ``mu u'' + S = 0`` with ``u(ymin)=u(ymax)=0``:

      u(y) = (S / (2 mu)) (y - ymin) (ymax - y),
      u_max = S H^2 / (8 mu),    wall slope = 4 u_max / H;

* total energy.  The stress work divergence is
  ``d/dy (tau_xy u) = mu (u'' u + u'^2) = -S u + mu u'^2``, so with no energy
  source the steady energy balance ``k T'' = S u - mu u'^2`` gives the Brinkman
  quartic temperature (the work of the force balances part of the viscous
  dissipation; this is what makes the case sensitive to the ``u tau`` term of
  the viscous energy flux):

      T(y) = T_w + G(y) - G(ymax) (y - ymin)/H,
      G''(y) = (S u(y) - mu u'(y)^2) / k,   G(ymin) = G'(ymin) = 0,

  which vanishes at both walls by construction;

* density follows from the unit-gas-constant equation of state,
  ``rho = p / T``.

Every quantity above is a polynomial in ``y``, so the reference profile is
obtained by *exact* polynomial double integration (no transcription of
literature values, no free parameters): it is a closed-form solution of the
compressible Navier--Stokes equations for the discretisation being tested, and
its residual can be checked to machine precision (see
``tests/test_ns_couette_poiseuille.py``).

The case is the classical plane Poiseuille channel (the ``+S`` volumetric force
plays exactly the role of ``-dp/dx``) *plus* the compressible viscous-heating
coupling, so a converged DG solution exercises, at once, the body-force source
hook, the no-slip/isothermal wall closure, the viscous momentum flux and the
``u tau`` work term of the viscous energy flux.
"""
from __future__ import annotations

import numpy as np
import torch

from .ns_dg_2d import CP, CV, GAMMA  # noqa: F401  (re-exported for convenience)

__all__ = ["BodyForcePoiseuille"]


class BodyForcePoiseuille:
    """Exact steady channel flow of a calorically perfect gas under force ``force``.

    Parameters
    ----------
    force : float
        Volumetric body force in ``+x`` per unit volume (``S`` above); the
        conserving variables therefore receive ``(0, force, 0, 0)``.
    mu : float
        Dynamic viscosity.
    prandtl : float
        Prandtl number; the conductivity is ``kappa = mu cp / prandtl``.
    pressure : float
        Uniform pressure (unit gas constant, so ``rho = pressure / T``).
    wall_temperature : float
        Common isothermal wall temperature.
    ymin, ymax : float
        Channel walls.
    """

    def __init__(self, force=0.2, mu=0.1, prandtl=0.71, pressure=1.0,
                 wall_temperature=1.0, ymin=0.0, ymax=1.0):
        if not ymax > ymin:
            raise ValueError("ymax must exceed ymin")
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        if prandtl <= 0.0:
            raise ValueError("prandtl must be positive")
        if pressure <= 0.0 or wall_temperature <= 0.0:
            raise ValueError("pressure and wall_temperature must be positive")
        self.force = float(force)
        self.mu = float(mu)
        self.prandtl = float(prandtl)
        self.pressure = float(pressure)
        self.wall_temperature = float(wall_temperature)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.height = self.ymax - self.ymin
        self.kappa = self.mu * CP / self.prandtl
        self.u_max = self.force * self.height**2 / (8.0 * self.mu)
        self.wall_slope = 4.0 * self.u_max / self.height
        self._temperature = self._temperature_polynomial()

    # -- exact solution ---------------------------------------------------
    def _temperature_polynomial(self):
        """``T`` as a :class:`numpy.polynomial.Polynomial` in ``s=(y-ymin)/H``."""
        s = np.polynomial.Polynomial([0.0, 1.0])
        # u(s) = 4 u_max s (1 - s);  d/dy = (1/H) d/ds
        u_s = 4.0 * self.u_max * (s - s**2)
        du_ds = u_s.deriv()
        work = self.force * u_s                              # S u
        dissipation = self.mu * (du_ds / self.height) ** 2   # mu (du/dy)^2
        curvature = self.height**2 * (work - dissipation) / self.kappa  # d2T/ds2
        twice_integrated = curvature.integ().integ()
        correction = twice_integrated(1.0)
        polynomial = twice_integrated - correction * s + self.wall_temperature
        # exactly zero wall values (remove round-off from the polynomial algebra)
        residual = polynomial(0.0) - self.wall_temperature
        return polynomial - residual * (1.0 - s)

    def temperature_polynomial(self):
        """``T(s)`` as a polynomial in ``s = (y - ymin)/H`` (read-only copy)."""
        return self._temperature.copy()

    def velocity(self, y):
        """``u(y)``; ``y`` may be a float, a numpy array or a torch tensor."""
        y = np.asarray(y, dtype=float)
        return (self.force / (2.0 * self.mu)) * (y - self.ymin) * (self.ymax - y)

    def velocity_slope(self, y):
        """``du/dy``."""
        y = np.asarray(y, dtype=float)
        return (self.force / (2.0 * self.mu)) * (self.ymin + self.ymax - 2.0 * y)

    def temperature(self, y):
        """``T(y)``."""
        y = np.asarray(y, dtype=float)
        return self._temperature((y - self.ymin) / self.height)

    def density(self, y):
        """``rho(y) = p / T(y)``."""
        return self.pressure / self.temperature(y)

    # -- helpers for the solver -------------------------------------------
    def initial(self, x, y):
        """``(rho, u, v, p)`` tuple for :func:`solve_navier_stokes_dg_2d`."""
        shape = torch.broadcast_shapes(torch.as_tensor(x).shape, torch.as_tensor(y).shape)
        y_np = torch.as_tensor(y).detach().cpu().numpy()
        dtype = getattr(torch.as_tensor(x), "dtype", torch.float64)
        device = getattr(torch.as_tensor(x), "device", None)
        zeros = torch.zeros(shape, dtype=dtype, device=device)
        temperature = torch.as_tensor(self.temperature(y_np), dtype=dtype, device=device)
        velocity = torch.as_tensor(self.velocity(y_np), dtype=dtype, device=device)
        return (
            (self.pressure / temperature).expand(shape),
            velocity.expand(shape),
            zeros,
            torch.full(shape, self.pressure, dtype=dtype, device=device),
        )

    def source(self, x, y, time=None):
        """Conserved-variable source: momentum in ``+x`` only (workless force)."""
        zeros = torch.zeros_like(x)
        return (
            zeros,
            self.force * torch.ones_like(x),
            zeros,
            torch.zeros_like(x),
        )

    def state(self, y):
        """Exact primitive state as a dict (numpy arrays)."""
        return {
            "velocity_x": self.velocity(y),
            "velocity_y": np.zeros_like(np.asarray(y, dtype=float)),
            "pressure": np.full_like(np.asarray(y, dtype=float), self.pressure),
            "temperature": self.temperature(y),
            "density": self.density(y),
        }

    def case_summary(self):
        """Scalar description of the case (goes into the metrics JSON)."""
        return {
            "force": self.force,
            "mu": self.mu,
            "prandtl": self.prandtl,
            "kappa": self.kappa,
            "pressure": self.pressure,
            "wall_temperature": self.wall_temperature,
            "height": self.height,
            "u_max": self.u_max,
            "wall_slope": self.wall_slope,
            "reynolds": self.density(np.array([self.ymin])).item() * self.u_max
            * self.height / self.mu,
            "density_ratio": float(self.pressure / self.temperature(np.array([self.ymin + 0.5 * self.height])).item())
            / float(self.pressure / self.wall_temperature),
        }