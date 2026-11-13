"""2-D linear acoustics with a modal discontinuous Galerkin solver (TensorDG).

Why this module
---------------
The repository already carried a 2-D elastodynamic (velocity--stress) DG solver
(``tensordg.elastic_2d``) and a family of 1-D exact anchors, but no *acoustic*
(fluid) capability: a pressure--velocity system with an exact, cylindrically
spreading reference solution.  Linear acoustics is the parent model of every
wave-propagation chain in the ecosystem (aeroacoustics, ice-edge wave loading,
structural acoustics coupling), so it belongs here as a DG formulation with its
own verification face.

Governing system (first order, conservative)::

    dp/dt + K (du/dx + dv/dy) = 0
    du/dt + (1/rho) dp/dx     = 0
    dv/dt + (1/rho) dp/dy     = 0

with ``K = rho c^2`` the bulk modulus, ``rho`` the density and ``c`` the speed of
sound.  Eliminating the velocity gives ``p_tt = c^2 lap(p)``.

State layout: ``q = (p, u, v)``.

Numerics
--------
* modal (Legendre) DG on Cartesian tensor-product cells, degrees 0..N, reusing
  the operator pack :func:`tensordg.elastic_2d.legendre_operators_2d` (single
  source of truth for the 1-D operators -- no duplicated quadrature tables);
* the *exact* Godunov flux of the constant-coefficient linear system (the
  acoustic Riemann problem is closed form, so upwinding has no tunable
  parameter)::

      p* = (pL + pR)/2 + Z (uL - uR)/2 ,   u* = (uL + uR)/2 + (pL - pR)/(2 Z)

  with impedance ``Z = rho c``.  The tangential velocity is carried by the
  stationary (zero-speed) characteristic whose face flux is identically zero;
  it is therefore discretised with the consistent, non-dissipative average;
* strong-form volume term with exact face corrections (same discrete
  convention as ``elastic_2d``, including the ``2/h`` face scaling whose
  omission is documented in ``docs/CASE-DG-ELASTIC-2D-PSV.md``);
* SSP-RK3 (Shu--Osher 1988) in time with the standard DG CFL bound
  ``dt = CFL h / ((2p+1) c)``;
* periodic boundary conditions only (like ``elastic_2d``).  The acoustic-pulse
  benchmarks are run with the pulse far inside the box, so the boundary closure
  does not enter the error budget (checked by domain/box-size sensitivity).

Exact anchors
-------------
1. :func:`acoustic_plane_pulse_exact` -- d'Alembert solution of a Gaussian pulse
   travelling in ``+-x``.  The data are ``y``-invariant, for which the 2-D
   system degenerates *exactly* to 1-D, so this anchor isolates the DG error of
   one direction at closed-form accuracy (no quadrature at all).
2. :func:`acoustic_pulse_exact_pressure` / ``..._velocity`` -- the axisymmetric
   Gaussian pulse ``p(r,0) = A exp(-r^2/sigma^2)``, whose exact solution is the
   Hankel (Bessel ``J0``) integral::

       p(r,t) = int_0^inf F(k) J0(k r) cos(k c t) k dk ,
       F(k)   = (A sigma^2 / 2) exp(-k^2 sigma^2 / 4) ,

   and ``u_r = (1/(rho c)) int_0^inf F(k) J1(k r) sin(k c t) k dk``.  It is
   "exact up to quadrature": the truncation/resolution parameters are derived
   from the tail bound and the oscillation rate (see
   :func:`_hankel_quadrature`) so that the residual is < 1e-12.  The tests
   verify that by refinement and by the ``t = 0`` identity (which returns the
   initial Gaussian to 1e-13).  The whole state is anchored, not just the
   pressure: the radial velocity follows from the linearised momentum equation.

References
----------
* Cockburn, S. & Shu, C.-W. (1998), J. Comput. Phys. 141, 199 (DG).
* Shu, C.-W. & Osher, S. (1988), SIAM J. Numer. Anal. 25, 615 (SSP-RK3).
* Tam, C. K. W. & Webb, J. C. (1993), J. Comput. Phys. 107, 262 (2-D acoustic
  pulse benchmark family; axisymmetric Gaussian pulse).
* Morse, P. M. & Feshbach, H. (1953), *Methods of Theoretical Physics* II,
  chap. 11 (2-D wave equation, Hankel representation).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .elastic_2d import (
    _trace,
    legendre_operators_2d,
    modal_to_quadrature,
    quadrature_grid,
)

__all__ = [
    "Acoustic2DResult",
    "AcousticMedium",
    "acoustic_energy",
    "acoustic_flux_x",
    "acoustic_flux_y",
    "acoustic_plane_pulse_exact",
    "acoustic_pulse_exact_pressure",
    "acoustic_pulse_exact_state",
    "acoustic_pulse_exact_velocity",
    "acoustic_pulse_initial",
    "acoustic_riemann_flux_x",
    "acoustic_riemann_flux_y",
    "solve_acoustic_2d",
]

# state vector layout
_P, _U, _V = 0, 1, 2

# boundary-condition labels (periodic only, mirroring ``elastic_2d.BC_X``)
BC_X = ("periodic",)
BC_Y = ("periodic",)


class AcousticMedium:
    """Homogeneous acoustic medium: density ``rho`` and sound speed ``c``."""

    def __init__(self, rho: float = 1.0, c: float = 1.0):
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        if c <= 0.0:
            raise ValueError("c must be positive")
        self.rho = float(rho)
        self.c = float(c)

    @property
    def bulk_modulus(self) -> float:
        """``K = rho c^2``."""
        return self.rho * self.c ** 2

    @property
    def impedance(self) -> float:
        """Acoustic impedance ``Z = rho c``."""
        return self.rho * self.c

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AcousticMedium(rho={self.rho}, c={self.c}, "
                f"K={self.bulk_modulus:.6f}, Z={self.impedance:.6f})")


# ---------------------------------------------------------------------------
# physical and numerical fluxes
# ---------------------------------------------------------------------------
def acoustic_flux_x(q: torch.Tensor, medium: AcousticMedium) -> torch.Tensor:
    """Physical flux in ``x``: ``F(q) = (K u, p/rho, 0)``."""
    out = torch.zeros_like(q)
    out[..., _P] = medium.bulk_modulus * q[..., _U]
    out[..., _U] = q[..., _P] / medium.rho
    return out


def acoustic_flux_y(q: torch.Tensor, medium: AcousticMedium) -> torch.Tensor:
    """Physical flux in ``y``: ``G(q) = (K v, 0, p/rho)``."""
    out = torch.zeros_like(q)
    out[..., _P] = medium.bulk_modulus * q[..., _V]
    out[..., _V] = q[..., _P] / medium.rho
    return out


def acoustic_riemann_flux_x(left: torch.Tensor, right: torch.Tensor,
                            medium: AcousticMedium) -> torch.Tensor:
    """Exact Godunov flux ``F*.e_x`` between ``left`` (``-x``) and ``right`` (``+x``).

    ``p* = (pL+pR)/2 + Z (uL-uR)/2`` and ``u* = (uL+uR)/2 + (pL-pR)/(2Z)``: the
    two genuinely upwinded acoustic characteristics.  The tangential velocity
    ``v`` has zero normal characteristic speed and identically zero ``x``-flux,
    so the consistent average is used.
    """
    z = medium.impedance
    p_star = 0.5 * (left[..., _P] + right[..., _P]) \
        + 0.5 * z * (left[..., _U] - right[..., _U])
    u_star = 0.5 * (left[..., _U] + right[..., _U]) \
        + 0.5 * (left[..., _P] - right[..., _P]) / z
    out = torch.zeros_like(left)
    out[..., _P] = medium.bulk_modulus * u_star
    out[..., _U] = p_star / medium.rho
    return out


def acoustic_riemann_flux_y(left: torch.Tensor, right: torch.Tensor,
                            medium: AcousticMedium) -> torch.Tensor:
    """Exact Godunov flux ``F*.e_y`` between ``left`` (``-y``) and ``right`` (``+y``)."""
    z = medium.impedance
    p_star = 0.5 * (left[..., _P] + right[..., _P]) \
        + 0.5 * z * (left[..., _V] - right[..., _V])
    v_star = 0.5 * (left[..., _V] + right[..., _V]) \
        + 0.5 * (left[..., _P] - right[..., _P]) / z
    out = torch.zeros_like(left)
    out[..., _P] = medium.bulk_modulus * v_star
    out[..., _V] = p_star / medium.rho
    return out


# ---------------------------------------------------------------------------
# initial data and exact solutions
# ---------------------------------------------------------------------------
def acoustic_pulse_initial(x, y, *, sigma: float = 0.1,
                           amplitude: float = 1.0):
    """Axisymmetric Gaussian pressure pulse, fluid at rest.

    ``p(x, y, 0) = amplitude * exp(-(x^2 + y^2)/sigma^2)``, ``u = v = 0``.

    Returns a ``(..., 3)`` array ``(p, u, v)`` matching the type of ``x``
    (``torch.Tensor`` in, ``torch.Tensor`` out; numpy in, numpy out).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        p = amplitude * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / sigma ** 2)
        zeros = np.zeros_like(p)
        return np.stack([p, zeros, zeros], axis=-1)
    p = amplitude * torch.exp(-(x ** 2 + y ** 2) / sigma ** 2)
    zeros = torch.zeros_like(p)
    return torch.stack([p, zeros, zeros], dim=-1)


def acoustic_plane_pulse_exact(x, y, t: float, *, sigma: float = 0.1,
                               amplitude: float = 1.0,
                               medium: Optional[AcousticMedium] = None):
    """Closed-form d'Alembert solution for a ``y``-invariant Gaussian pulse.

    ``p = A/2 [f(x-ct) + f(x+ct)]`` with ``f(s) = exp(-s^2/sigma^2)`` and
    ``u = [f(x-ct) - f(x+ct)]/(2 rho c)``, ``v = 0`` -- the exact solution of the
    2-D acoustic system for data that do not vary in ``y`` (all ``y``
    derivatives vanish identically).  No quadrature is involved, so this anchor
    is exact to round-off and isolates the DG error of a single direction.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    medium = medium or AcousticMedium()
    c = medium.c
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x_np = np.asarray(x, dtype=float)
        f_minus = amplitude * np.exp(-((x_np - c * t) ** 2) / sigma ** 2)
        f_plus = amplitude * np.exp(-((x_np + c * t) ** 2) / sigma ** 2)
        p = 0.5 * (f_minus + f_plus)
        u = (f_minus - f_plus) / (2.0 * medium.rho * c)
        return np.stack([p, u, np.zeros_like(p)], axis=-1)
    f_minus = amplitude * torch.exp(-((x - c * t) ** 2) / sigma ** 2)
    f_plus = amplitude * torch.exp(-((x + c * t) ** 2) / sigma ** 2)
    p = 0.5 * (f_minus + f_plus)
    u = (f_minus - f_plus) / (2.0 * medium.rho * c)
    return torch.stack([p, u, torch.zeros_like(p)], dim=-1)


def _hankel_quadrature(sigma: float, t: float, r_max: float, c: float,
                       order: int = 10):
    """Nodes/weights for the ``int_0^inf`` Hankel integrals of a Gaussian pulse.

    *Truncation*: ``F(k) ~ exp(-k^2 sigma^2/4)``, so ``k_max = 12/sigma`` leaves
    a tail below ``exp(-36) ~ 2e-16``.
    *Resolution*: the fastest phase rate of the integrand with respect to ``k``
    is ``r_max + c|t|`` (from ``J0(k r)`` and ``cos(k c t)``); the panel width is
    ``1/(r_max + c|t|)`` so the phase turns by at most one radian per panel,
    where a 10-point Gauss--Legendre rule is super-algebraically accurate.
    """
    k_max = 12.0 / sigma
    rate = max(r_max + c * abs(t), 1.0)
    width = 1.0 / rate
    n_panel = max(1, int(math.ceil(k_max / width)))
    xi, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, k_max, n_panel + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    k = (mid[:, None] + half[:, None] * xi[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return k, weights


def _as_numpy_pair(x, y):
    x_np = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=float)
    y_np = np.asarray(y.detach().cpu() if torch.is_tensor(y) else y, dtype=float)
    return x_np, y_np


def _wrap_like(value: np.ndarray, template):
    if torch.is_tensor(template):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device)
    return value


def _bessel(kind: int):
    try:
        from scipy import special
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise RuntimeError("scipy is required for the Hankel exact solution") from exc
    return special.j0 if kind == 0 else special.j1


def acoustic_pulse_exact_pressure(x, y, t: float, *, sigma: float = 0.1,
                                  amplitude: float = 1.0,
                                  medium: Optional[AcousticMedium] = None,
                                  order: int = 10):
    """Exact pressure of the axisymmetric Gaussian pulse (Hankel integral).

    ``p(r,t) = int_0^inf F(k) J0(k r) cos(k c t) k dk`` with
    ``F(k) = (A sigma^2/2) exp(-k^2 sigma^2/4)``.

    Evaluated with composite Gauss--Legendre quadrature in ``k`` (see
    :func:`_hankel_quadrature`); the residual is < 1e-12 for the parameter
    ranges used here, which the tests verify by quadrature refinement and by the
    ``t = 0`` identity (returns the initial Gaussian to 1e-13).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    medium = medium or AcousticMedium()
    j0 = _bessel(0)
    x_np, y_np = _as_numpy_pair(x, y)
    r = np.sqrt(x_np ** 2 + y_np ** 2).ravel()
    k, w = _hankel_quadrature(sigma, t, float(r.max()) if r.size else 0.0,
                              medium.c, order=order)
    transform = 0.5 * amplitude * sigma ** 2 * np.exp(-k ** 2 * sigma ** 2 / 4.0)
    spectral = transform * np.cos(k * medium.c * t) * k * w
    out = np.empty_like(r)
    chunk = 4096
    for start in range(0, r.size, chunk):
        block = r[start:start + chunk]
        out[start:start + chunk] = j0(np.outer(block, k)) @ spectral
    return _wrap_like(out.reshape(x_np.shape), x)


def acoustic_pulse_exact_velocity(x, y, t: float, *, sigma: float = 0.1,
                                  amplitude: float = 1.0,
                                  medium: Optional[AcousticMedium] = None,
                                  order: int = 10):
    """Exact velocity ``(u, v)`` of the axisymmetric Gaussian pulse.

    ``u_r = (1/(rho c)) int_0^inf F(k) J1(k r) sin(k c t) k dk`` (from the
    linearised momentum equation, hence consistent with
    :func:`acoustic_pulse_exact_pressure`), then ``(u, v) = u_r (x, y)/r``.
    Both components vanish on the axis.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    medium = medium or AcousticMedium()
    j1 = _bessel(1)
    x_np, y_np = _as_numpy_pair(x, y)
    r = np.sqrt(x_np ** 2 + y_np ** 2).ravel()
    k, w = _hankel_quadrature(sigma, t, float(r.max()) if r.size else 0.0,
                              medium.c, order=order)
    transform = 0.5 * amplitude * sigma ** 2 * np.exp(-k ** 2 * sigma ** 2 / 4.0)
    spectral = transform * np.sin(k * medium.c * t) * k * w / (medium.rho * medium.c)
    u_r = np.empty_like(r)
    chunk = 4096
    for start in range(0, r.size, chunk):
        block = r[start:start + chunk]
        u_r[start:start + chunk] = j1(np.outer(block, k)) @ spectral
    radial_factor = np.where(r > 0.0, 1.0 / np.where(r > 0.0, r, 1.0), 0.0)
    u = (u_r * radial_factor * x_np.ravel()).reshape(x_np.shape)
    v = (u_r * radial_factor * y_np.ravel()).reshape(y_np.shape)
    return _wrap_like(np.stack([u, v], axis=-1), x)


def acoustic_pulse_exact_state(x, y, t: float, *, sigma: float = 0.1,
                               amplitude: float = 1.0,
                               medium: Optional[AcousticMedium] = None,
                               order: int = 10):
    """Exact ``(p, u, v)`` state of the axisymmetric Gaussian pulse."""
    p = acoustic_pulse_exact_pressure(x, y, t, sigma=sigma, amplitude=amplitude,
                                      medium=medium, order=order)
    vel = acoustic_pulse_exact_velocity(x, y, t, sigma=sigma, amplitude=amplitude,
                                        medium=medium, order=order)
    if torch.is_tensor(p):
        return torch.cat([p.unsqueeze(-1), torch.as_tensor(vel, dtype=p.dtype,
                                                          device=p.device)], dim=-1)
    return np.concatenate([np.asarray(p)[..., None], np.asarray(vel)], axis=-1)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def acoustic_energy(coeff: torch.Tensor, ops, medium: AcousticMedium,
                    hx: float, hy: float) -> torch.Tensor:
    """Total acoustic energy ``int [p^2/(2K) + rho (u^2+v^2)/2] dA``.

    This is the quadratic invariant of the linear acoustic system, so it is the
    conservation diagnostic of the scheme (an exact invariant of the continuous
    problem for periodic/closed boundaries).
    """
    q = modal_to_quadrature(coeff, ops["phi"])
    w = ops["weights"]
    w2 = torch.outer(w, w).reshape(1, 1, w.shape[0], w.shape[0])
    potential = q[..., _P] ** 2 / (2.0 * medium.bulk_modulus)
    kinetic = 0.5 * medium.rho * (q[..., _U] ** 2 + q[..., _V] ** 2)
    return ((potential + kinetic) * w2).sum() * (hx * hy / 4.0)


def _relative_l2_error_state(numerical: torch.Tensor, exact: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
    """Quadrature-weighted relative ``L2`` error of a ``(..., m)`` field."""
    w2 = torch.outer(weights, weights).reshape(1, 1, weights.numel(),
                                               weights.numel(), 1)
    diff = numerical - exact
    num = (diff ** 2 * w2).sum()
    den = (exact ** 2 * w2).sum()
    return torch.sqrt(num / den.clamp_min(1e-300))


def _relative_l2_error_scalar(numerical: torch.Tensor, exact: torch.Tensor,
                              weights: torch.Tensor) -> torch.Tensor:
    w2 = torch.outer(weights, weights).reshape(1, 1, weights.numel(),
                                               weights.numel())
    num = ((numerical - exact) ** 2 * w2).sum()
    den = (exact ** 2 * w2).sum()
    return torch.sqrt(num / den.clamp_min(1e-300))


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def _acoustic_rhs(coeff: torch.Tensor, ops, medium: AcousticMedium,
                  hx: float, hy: float, bc_x: str, bc_y: str) -> torch.Tensor:
    """Semi-discrete DG right-hand side in modal coefficients.

    Strong form with exact face corrections (same discrete convention as
    ``elastic_2d._rhs``): the volume term carries ``2/h`` from the reference
    cell map, and each face correction carries the same factor.  Omitting it
    under-weights the numerical-flux correction by ``O(h)`` and destroys
    consistency (see ``docs/CASE-DG-ELASTIC-2D-PSV.md``).
    """
    phi, proj, dmat = ops["phi"], ops["proj"], ops["dmat"]
    svec, mass_inv = ops["svec"], ops["mass_inv"]
    phi_left, phi_right = ops["phi_left"], ops["phi_right"]
    k = phi.shape[1]

    q = modal_to_quadrature(coeff, phi)
    cfx = torch.einsum("ka,lb,ijabm->ijklm", proj, proj,
                       acoustic_flux_x(q, medium))
    cfy = torch.einsum("ka,lb,ijabm->ijklm", proj, proj,
                       acoustic_flux_y(q, medium))

    # ---- volume ------------------------------------------------------------
    rhs = -(2.0 / hx) * torch.einsum("kn,ijnlm->ijklm", dmat, cfx) \
          -(2.0 / hy) * torch.einsum("ln,ijknm->ijklm", dmat, cfy)

    face_x, face_y = 2.0 / hx, 2.0 / hy
    mass = mass_inv.view(1, 1, k, k, 1)
    pr = phi_right.view(1, 1, k, 1, 1)
    pl = phi_left.view(1, 1, k, 1, 1)

    # ---- x faces -----------------------------------------------------------
    if bc_x != "periodic":  # pragma: no cover - only periodic x is supported
        raise ValueError(f"unsupported x boundary condition {bc_x!r}")
    u_l = _trace(coeff, phi_left, 0, phi)      # (nx, ny, nq_eta, 3)
    u_r = _trace(coeff, phi_right, 0, phi)
    # Riemann states in +x order: (left = interior right trace,
    # right = neighbour's left trace).  Swapping them only affects
    # discontinuous traces -- which is exactly where an upwind flux is
    # supposed to act -- so the order is part of the contract.
    star_r = acoustic_riemann_flux_x(u_r, torch.roll(u_l, shifts=-1, dims=0), medium)
    g_r = star_r - acoustic_flux_x(u_r, medium)
    proj_r = torch.einsum("lb,ijbm->ijlm", svec, g_r)
    rhs = rhs - face_x * mass * pr * proj_r.view(proj_r.shape[0], proj_r.shape[1], 1,
                                                 proj_r.shape[2], proj_r.shape[3])

    star_l = acoustic_riemann_flux_x(torch.roll(u_r, shifts=1, dims=0), u_l, medium)
    g_l = (-star_l) + acoustic_flux_x(u_l, medium)          # outward normal -e_x
    proj_l = torch.einsum("lb,ijbm->ijlm", svec, g_l)
    rhs = rhs - face_x * mass * pl * proj_l.view(proj_l.shape[0], proj_l.shape[1], 1,
                                                 proj_l.shape[2], proj_l.shape[3])

    # ---- y faces -----------------------------------------------------------
    if bc_y != "periodic":  # pragma: no cover - only periodic y is supported
        raise ValueError(f"unsupported y boundary condition {bc_y!r}")
    u_b = _trace(coeff, phi_left, 1, phi)      # (nx, ny, nq_xi, 3)
    u_t = _trace(coeff, phi_right, 1, phi)
    star_t = acoustic_riemann_flux_y(u_t, torch.roll(u_b, shifts=-1, dims=1), medium)
    g_t = star_t - acoustic_flux_y(u_t, medium)
    proj_t = torch.einsum("ka,ijam->ijkm", svec, g_t)
    pr_l = phi_right.view(1, 1, 1, k, 1)
    rhs = rhs - face_y * mass * pr_l * proj_t.view(proj_t.shape[0], proj_t.shape[1],
                                                   proj_t.shape[2], 1, proj_t.shape[3])

    star_b = -acoustic_riemann_flux_y(torch.roll(u_t, shifts=1, dims=1), u_b, medium)
    g_b = star_b + acoustic_flux_y(u_b, medium)             # outward normal -e_y
    proj_b = torch.einsum("ka,ijam->ijkm", svec, g_b)
    pl_l = phi_left.view(1, 1, 1, k, 1)
    rhs = rhs - face_y * mass * pl_l * proj_b.view(proj_b.shape[0], proj_b.shape[1],
                                                   proj_b.shape[2], 1, proj_b.shape[3])
    return rhs


class Acoustic2DResult:
    """Container for a DG run of the 2-D acoustic system."""

    def __init__(self, coeff, medium, ops, geometry, time_info, bc_x, bc_y,
                 cfl, snapshots):
        self.coeff = coeff
        self.medium = medium
        self.ops = ops
        self.nx = geometry["nx"]
        self.ny = geometry["ny"]
        self.length_x = geometry["length_x"]
        self.length_y = geometry["length_y"]
        self.hx = geometry["hx"]
        self.hy = geometry["hy"]
        self.dt, self.n_steps = time_info
        self.times = sorted(snapshots)
        self.snapshots = snapshots
        self.bc_x = bc_x
        self.bc_y = bc_y
        self.cfl = cfl
        self.degree = ops["degree"]

    def fields(self, coeff=None) -> torch.Tensor:
        """State at the quadrature points, shape ``(nx, ny, p+1, p+1, 3)``."""
        c = self.coeff if coeff is None else coeff
        return modal_to_quadrature(c, self.ops["phi"])

    def pressure(self, coeff=None) -> torch.Tensor:
        """Pressure at the quadrature points, shape ``(nx, ny, p+1, p+1)``."""
        return self.fields(coeff)[..., _P]

    def quadrature_grid(self):
        """Physical quadrature coordinates ``(x, y)`` of shape ``(nx, ny, p+1, p+1)``."""
        return quadrature_grid(self.nx, self.ny, self.hx, self.hy, self.ops["nodes"])

    def energy(self, coeff=None) -> torch.Tensor:
        c = self.coeff if coeff is None else coeff
        return acoustic_energy(c, self.ops, self.medium, self.hx, self.hy)

    def relative_l2_error(self, exact: torch.Tensor, component: Optional[int] = _P,
                          coeff=None) -> float:
        """Relative ``L2`` error at the quadrature points.

        ``component=None`` measures the whole ``(p, u, v)`` state, otherwise the
        selected component (default: pressure).
        """
        fields = self.fields(coeff)
        exact = exact.to(dtype=fields.dtype, device=fields.device)
        if component is None:
            return float(_relative_l2_error_state(fields, exact,
                                                  self.ops["weights"]))
        return float(_relative_l2_error_scalar(fields[..., component],
                                               exact[..., component],
                                               self.ops["weights"]))


def solve_acoustic_2d(initial_state: Callable, *, nx: int = 16,
                      ny: Optional[int] = None, degree: int = 2,
                      final_time: float = 0.3, length_x: float = 2.0,
                      length_y: Optional[float] = None,
                      medium: Optional[AcousticMedium] = None,
                      cfl: float = 0.2, bc_x: str = "periodic",
                      bc_y: str = "periodic",
                      snapshot_times: Sequence[float] = (),
                      dtype=torch.float64, device="cpu") -> Acoustic2DResult:
    """March the 2-D linear acoustic system with DG + SSP-RK3.

    ``initial_state(x, y)`` must return a ``(..., 3)`` array ``(p, u, v)``.  The
    computational domain is ``[0, length_x] x [0, length_y]`` (default
    ``[0,2] x [0,2]``, i.e. centred on ``(1, 1)``; shift the coordinates by
    ``-length/2`` when initialising a centred pulse).  ``final_time`` and every
    entry of ``snapshot_times`` are present in ``result.snapshots``.
    """
    if bc_x not in BC_X:
        raise ValueError(f"bc_x must be one of {BC_X}")
    if bc_y not in BC_Y:
        raise ValueError(f"bc_y must be one of {BC_Y}")
    ny = nx if ny is None else ny
    length_y = length_x if length_y is None else length_y
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if cfl <= 0:
        raise ValueError("cfl must be positive")
    medium = medium or AcousticMedium()

    ops = legendre_operators_2d(degree, device=device, dtype=dtype)
    hx, hy = length_x / nx, length_y / ny

    x, y = quadrature_grid(nx, ny, hx, hy, ops["nodes"])
    u0 = initial_state(x, y)
    if not torch.is_tensor(u0):
        u0 = torch.as_tensor(np.asarray(u0), dtype=dtype, device=device)
    u0 = u0.to(dtype=dtype, device=device)
    if u0.shape != (nx, ny, degree + 1, degree + 1, 3):
        u0 = u0.expand(nx, ny, degree + 1, degree + 1, 3)
    coeff = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], u0)

    # standard DG stability bound on the acoustic CFL number
    dt_max = cfl * min(hx, hy) / ((2.0 * degree + 1.0) * medium.c)
    n_steps = max(1, int(math.ceil(final_time / dt_max)))
    dt = final_time / n_steps

    def rhs(c):
        return _acoustic_rhs(c, ops, medium, hx, hy, bc_x, bc_y)

    wanted = sorted({float(t) for t in snapshot_times} | {float(final_time)})
    snapshots = {}
    t = 0.0
    idx = 0
    if wanted and wanted[0] <= 1e-15:
        snapshots[wanted[0]] = coeff.clone()
        idx = 1
    for step in range(n_steps):
        # SSP-RK3 (Shu & Osher 1988)
        u1 = coeff + dt * rhs(coeff)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * rhs(u1))
        coeff = coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(u2))
        t = (step + 1) * dt
        while idx < len(wanted) and t >= wanted[idx] - 1e-12:
            snapshots[wanted[idx]] = coeff.clone()
            idx += 1

    geometry = {"nx": nx, "ny": ny, "length_x": length_x, "length_y": length_y,
                "hx": hx, "hy": hy}
    return Acoustic2DResult(coeff, medium, ops, geometry, (dt, n_steps), bc_x,
                            bc_y, cfl, snapshots)
