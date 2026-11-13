"""One-dimensional linear elastodynamics with a modal DG discretisation.

The bar is described by the first-order *velocity--stress* system

.. math::

   \\frac{\\partial v}{\\partial t} - \\frac{1}{\\rho}\\frac{\\partial \\sigma}{\\partial x} = 0,
   \\qquad
   \\frac{\\partial \\sigma}{\\partial t} - E\\frac{\\partial v}{\\partial x} = 0,

with constant density :math:`\\rho` and Young modulus :math:`E`, i.e.

.. math::

   U_t + F(U)_x = 0, \\qquad U = \\begin{pmatrix} v \\\\ \\sigma \\end{pmatrix},
   \\qquad F(U) = \\begin{pmatrix} -\\sigma/\\rho \\\\ -E v \\end{pmatrix}.

The flux Jacobian has the eigenvalues :math:`\\lambda^\\pm = \\pm c` with
:math:`c = \\sqrt{E/\\rho}`; the Riemann invariants are

.. math::

   w^+ = \\sigma - \\rho c v \\;(\\text{advected with } +c), \\qquad
   w^- = \\sigma + \\rho c v \\;(\\text{advected with } -c).

Features
--------
* Modal Legendre DG of degree ``p`` (``0 <= p <= 8``), tensor/numpy-free
  operators exactly integrated for polynomial products of the selected degree.
* Exact upwind (characteristic) interface flux: :math:`F^* = A^+U_L + A^-U_R`,
  i.e. ``w+`` from the left state and ``w-`` from the right state.
* Characteristic boundary conditions, reflecting (``v = 0``, fixed end) and
  free (``\\sigma = 0``) surfaces, plus periodic ends.
* SSP-RK3 (default) or classical RK4 time integration.

Verification cases
------------------
* ``dalembert_pulse`` -- periodic bar released with a Gaussian velocity
  pulse.  The exact solution is the d'Alembert superposition of two
  counter-propagating pulses,
  :math:`v = \\tfrac12[v_0(x-ct) + v_0(x+ct)]` and
  :math:`\\sigma = \\tfrac12\\rho c [v_0(x+ct) - v_0(x-ct)]`.
* ``standing_wave`` -- bar fixed at :math:`x=0` and free at :math:`x=L`,
  released from the first longitudinal eigenmode
  :math:`u = A \\sin(kx)\\cos(\\omega t)`, :math:`k = \\pi/(2L)`,
  :math:`\\omega = ck`.  The exact stress is
  :math:`\\sigma = E A k \\cos(kx)\\cos(\\omega t)`.

References
----------
* Graff, K. F. (1975), *Wave Motion in Elastic Solids*, §2 (longitudinal
  waves in a bar, d'Alembert solution).
* LeVeque, R. J. (2002), *Finite Volume Methods for Hyperbolic Problems*,
  Ch. 3 (characteristic boundary conditions for linear systems).
* Cockburn & Shu (1998), *Runge--Kutta discontinuous Galerkin methods ...*,
  J. Comput. Phys. 141 (upwind flux and SSP time stepping).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .advection_1d import MAX_DEGREE, _operators

__all__ = [
    "solve_elastic_1d",
    "elastic_wave_speed",
    "upwind_flux",
    "gaussian_pulse",
    "dalembert_pulse",
    "standing_wave",
    "initial_pulse_velocity",
    "initial_pulse_stress",
    "initial_standing_velocity",
    "initial_standing_stress",
    "stress_profile",
    "relative_error",
    "elastic_energy",
    "peak_stress",
]

SUPPORTED_BC = ("periodic", "fixed-free", "free-fixed")


# ---------------------------------------------------------------------------
# exact solutions
# ---------------------------------------------------------------------------
def elastic_wave_speed(rho: float = 1.0, young: float = 1.0) -> float:
    """Longitudinal bar wave speed ``c = sqrt(E/rho)``."""
    if rho <= 0 or young <= 0:
        raise ValueError("rho and young must be positive")
    return math.sqrt(young / rho)


def gaussian_pulse(x: torch.Tensor, center: float = 0.5,
                   width: float = 0.05) -> torch.Tensor:
    """Unit-amplitude Gaussian ``exp(-((x-center)/width)**2)``."""
    return torch.exp(-((x - center) / width) ** 2)


def initial_pulse_velocity(x: torch.Tensor, center: float = 0.5,
                           width: float = 0.05) -> torch.Tensor:
    """Initial velocity of the periodic d'Alembert pulse case."""
    return gaussian_pulse(x, center, width)


def initial_pulse_stress(x: torch.Tensor) -> torch.Tensor:
    """Initial stress of the periodic d'Alembert pulse case (stress-free)."""
    return torch.zeros_like(x)


def _seam_limit(length: float = 1.0, center: float = 0.5) -> float:
    """Largest ``c t`` for which :func:`dalembert_pulse` is still exact."""
    return min(center, length - center)


def dalembert_pulse(x: torch.Tensor, t: float, rho: float = 1.0,
                    young: float = 1.0, center: float = 0.5,
                    width: float = 0.05, length: float = 1.0,
                    strict: bool = True):
    """Exact two-wave (d'Alembert) solution of the periodic pulse case.

    Returns ``(v, sigma)``.  Valid while the two pulses do not meet again
    through the periodic seam, i.e. for ``c t <= min(center, length-center)``.

    The local form below is a *single* image: it silently degenerates to
    ``v ~ 0`` once the packets wrap around the seam, so past that limit it is a
    wrong reference rather than an approximate one.  ``strict`` therefore
    raises ``ValueError`` instead of returning a plausible-looking field; use
    :func:`_dalembert_periodic_images` for any ``t`` (it sums the images) or
    pass ``strict=False`` to reproduce the old, unguarded behaviour.
    """
    c = elastic_wave_speed(rho, young)
    limit = _seam_limit(length, center)
    if strict and c * t > limit * (1.0 + 1e-12):
        raise ValueError(
            f"dalembert_pulse is only exact for c*t <= {limit:g} "
            f"(got c*t = {c * t:g}); the packets have wrapped through the "
            "periodic seam. Use _dalembert_periodic_images(x, t, ...) for the "
            "image-summed exact solution, or strict=False for the bare "
            "single-image form.")
    g_minus = gaussian_pulse(x - c * t, center, width)
    g_plus = gaussian_pulse(x + c * t, center, width)
    v = 0.5 * (g_minus + g_plus)
    sigma = 0.5 * rho * c * (g_plus - g_minus)
    return v, sigma


# ---------------------------------------------------------------------------
# exact solutions past the first seam crossing: image (reflection) series
# ---------------------------------------------------------------------------
def _periodic_image_sum(y: torch.Tensor, length: float = 1.0,
                        center: float = 0.5, width: float = 0.05,
                        n_images: int = 6) -> torch.Tensor:
    r"""``length``-periodic continuation ``\tilde V(y) = sum_n V(y - n L)``."""
    total = torch.zeros_like(y)
    for n in range(-n_images, n_images + 1):
        total = total + gaussian_pulse(y - n * length, center, width)
    return total


def _fixed_free_image_sum(y: torch.Tensor, length: float = 1.0,
                          center: float = 0.5, width: float = 0.05
                          ) -> torch.Tensor:
    r"""Image sum for a bar clamped at ``x=0`` and free at ``x=L``.

    ``v=0`` at the clamped end needs the potential continued *oddly* about
    ``x=0``; ``sigma=0`` at the free end needs it continued *evenly* about
    ``x=L``.  Together the two symmetries give ``V(y+2L) = -V(y)``, i.e. a
    ``4L``-periodic continuation -- each transit adds one reflection of the
    correct sign, which is exactly what the mixed clamped/free ends do.
    """
    period = 4.0 * length
    r = torch.remainder(y, period)
    flip = r > 2.0 * length
    r = torch.where(flip, period - r, r)
    sign = torch.where(flip, -torch.ones_like(r), torch.ones_like(r))
    folded = torch.where(r > length, 2.0 * length - r, r)
    return sign * gaussian_pulse(folded, center, width)


def _dalembert_from_potential(potential, x: torch.Tensor, t: float,
                              rho: float, young: float):
    """``v = (V(x-ct)+V(x+ct))/2``, ``sigma = rho c (V(x+ct)-V(x-ct))/2``."""
    c = elastic_wave_speed(rho, young)
    g_plus = potential(x + c * t)
    g_minus = potential(x - c * t)
    return 0.5 * (g_minus + g_plus), 0.5 * rho * c * (g_plus - g_minus)


def _dalembert_periodic_images(x: torch.Tensor, t: float, length: float = 1.0,
                               rho: float = 1.0, young: float = 1.0,
                               center: float = 0.5, width: float = 0.05,
                               n_images: int = 6):
    """Exact d'Alembert solution of the periodic bar for *any* ``t``."""
    return _dalembert_from_potential(
        lambda y: _periodic_image_sum(y, length, center, width, n_images),
        x, t, rho, young)


def _dalembert_fixed_free_images(x: torch.Tensor, t: float,
                                 length: float = 1.0, rho: float = 1.0,
                                 young: float = 1.0, center: float = 0.5,
                                 width: float = 0.05):
    """Exact solution of the clamped-free bar for *any* ``t`` (image series)."""
    return _dalembert_from_potential(
        lambda y: _fixed_free_image_sum(y, length, center, width),
        x, t, rho, young)


def standing_wave(x: torch.Tensor, t: float, length: float = 1.0,
                  rho: float = 1.0, young: float = 1.0,
                  amplitude: float = 1.0):
    """Exact first eigenmode of a bar fixed at ``x=0`` and free at ``x=L``.

    Returns ``(v, sigma, u)`` with ``u = A sin(kx) cos(omega t)``,
    ``k = pi/(2L)``, ``omega = c k``.
    """
    c = elastic_wave_speed(rho, young)
    k = math.pi / (2.0 * length)
    omega = c * k
    s, cs = torch.sin(k * x), torch.cos(k * x)
    u = amplitude * s * math.cos(omega * t)
    v = -amplitude * omega * s * math.sin(omega * t)
    sigma = young * amplitude * k * cs * math.cos(omega * t)
    return v, sigma, u


def initial_standing_velocity(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """``v(x,0) = 0`` for the eigenmode release test."""
    return torch.zeros_like(x)


def initial_standing_stress(x: torch.Tensor, length: float = 1.0,
                            young: float = 1.0,
                            amplitude: float = 1.0) -> torch.Tensor:
    """``sigma(x,0) = E A k cos(kx)`` for the eigenmode release test."""
    k = math.pi / (2.0 * length)
    return young * amplitude * k * torch.cos(k * x)


# ---------------------------------------------------------------------------
# numerical fluxes
# ---------------------------------------------------------------------------
def upwind_flux(v_l: torch.Tensor, s_l: torch.Tensor, v_r: torch.Tensor,
                s_r: torch.Tensor, rho: float, young: float) -> torch.Tensor:
    """Exact upwind flux ``A^+U_L + A^-U_R`` in terms of Riemann invariants.

    ``w+ = sigma - rho c v`` travels with ``+c`` (taken from the left state),
    ``w- = sigma + rho c v`` with ``-c`` (taken from the right state).
    Returns the flux vector ``(-sigma*/rho, -E v*)``.
    """
    c = math.sqrt(young / rho)
    w_plus_l = s_l - rho * c * v_l
    w_minus_r = s_r + rho * c * v_r
    sigma_star = 0.5 * (w_plus_l + w_minus_r)
    v_star = (w_minus_r - w_plus_l) / (2.0 * rho * c)
    return torch.stack((-sigma_star / rho, -young * v_star), dim=-1)


def _boundary_flux(v_int: torch.Tensor, s_int: torch.Tensor, side: str,
                   rho: float, young: float, bc: str) -> torch.Tensor:
    """Characteristic boundary flux built from the interior trace.

    At ``x=0`` the entering characteristic is ``w+`` (from the exterior) and
    the interior supplies ``w-``; at ``x=L`` it is the other way round.  Fixed
    end (``v=0``): the reflected invariant equals the outgoing one.  Free end
    (``sigma=0``): the entering invariant is negated.
    """
    c = math.sqrt(young / rho)
    w_plus = s_int - rho * c * v_int
    w_minus = s_int + rho * c * v_int
    zero = torch.zeros_like(v_int)
    if side == "left":
        # x = 0: the characteristic entering the domain is w+ (lambda = +c);
        # w- is the outgoing one taken from the interior trace.
        if bc == "fixed-free":          # v = 0  ->  w+_ext = w-_int
            sigma_star, v_star = w_minus, zero
        else:                           # sigma = 0  ->  w+_ext = -w-_int
            sigma_star, v_star = zero, w_minus / (rho * c)
    else:
        # x = L: w- enters the domain, w+ leaves through the interior trace.
        if bc == "free-fixed":          # v = 0  ->  w-_ext = w+_int
            sigma_star, v_star = w_plus, zero
        else:                           # sigma = 0  ->  w-_ext = -w+_int
            sigma_star, v_star = zero, -w_plus / (rho * c)
    return torch.stack((-sigma_star / rho, -young * v_star), dim=-1)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def _project(profile: Callable[[torch.Tensor], torch.Tensor],
             centers: torch.Tensor, h: float, xq: torch.Tensor,
             wq: torch.Tensor, phi: torch.Tensor, degree: int) -> torch.Tensor:
    """L2 projection of ``profile(x)`` onto the modal Legendre basis."""
    x_phys = centers[:, None] + 0.5 * h * xq[None, :]
    f = profile(x_phys)
    coeff = torch.empty((centers.numel(), degree + 1), device=centers.device,
                        dtype=centers.dtype)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            wq[None, :] * f * phi[:, k][None, :], dim=1
        )
    return coeff


def solve_elastic_1d(ncells: int = 100, degree: int = 2,
                     final_time: float = 0.2, length: float = 1.0,
                     rho: float = 1.0, young: float = 1.0, cfl: float = 0.3,
                     bc: str = "periodic",
                     initial_velocity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                     initial_stress: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                     device: str = "cpu", dtype: torch.dtype = torch.float64,
                     time_integrator: str = "ssp_rk3"):
    """Solve the 1-D velocity--stress elastodynamics system with DG.

    ``initial_velocity`` / ``initial_stress`` are callables ``f(x) -> tensor``
    evaluated on the physical coordinates.  Defaults are the periodic Gaussian
    velocity pulse of :func:`dalembert_pulse` (``bc='periodic'``), or the
    eigenmode release of :func:`standing_wave` for ``bc in ('fixed-free',
    'free-fixed')``.

    Returns ``(centers, coeff, meta)`` with ``coeff`` of shape
    ``[ncells, degree+1, 2]``; the last axis is ``(v, sigma)`` and the first
    modal coefficient of each variable is its cell average.
    """
    if ncells < 2 or not 0 <= degree <= MAX_DEGREE:
        raise ValueError(f"ncells must be >= 2 and degree in [0, {MAX_DEGREE}]")
    if final_time < 0 or cfl <= 0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    if bc not in SUPPORTED_BC:
        raise ValueError(f"bc must be one of {SUPPORTED_BC}")
    if time_integrator not in ("ssp_rk3", "rk4"):
        raise ValueError("time_integrator must be 'ssp_rk3' or 'rk4'")
    if length <= 0:
        raise ValueError("length must be positive")
    c = elastic_wave_speed(rho, young)
    dev = torch.device(device)
    h = length / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    xq, wq, phi, _volume, phi_l, phi_r = _operators(degree, dev, dtype)
    # derivative matrix dP_k/dxi at the quadrature nodes
    dphi = torch.stack([
        torch.as_tensor(
            np.polynomial.legendre.legval(
                xq.detach().cpu().numpy(),
                np.polynomial.legendre.legder(np.eye(degree + 1)[k])),
            device=dev, dtype=dtype)
        for k in range(degree + 1)
    ], dim=1)

    if initial_velocity is None:
        if bc == "periodic":
            initial_velocity = initial_pulse_velocity
        else:
            initial_velocity = initial_standing_velocity
    if initial_stress is None:
        if bc == "periodic":
            initial_stress = initial_pulse_stress
        else:
            initial_stress = initial_standing_stress
    v0 = _project(initial_velocity, centers, h, xq, wq, phi, degree)
    s0 = _project(initial_stress, centers, h, xq, wq, phi, degree)
    coeff = torch.stack((v0, s0), dim=-1)

    if final_time == 0 or c == 0:
        steps = 0
        dt = 0.0
    else:
        dt_nominal = cfl * h / (c * (2 * degree + 1))
        steps = max(1, math.ceil(final_time / dt_nominal))
        dt = final_time / steps

    for _ in range(steps):
        k1 = _rhs(coeff, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
        if time_integrator == "ssp_rk3":
            u1 = coeff + dt * k1
            k2 = _rhs(u1, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
            u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
            k3 = _rhs(u2, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
            coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
            continue
        u2 = coeff + 0.5 * dt * k1
        k2 = _rhs(u2, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
        u3 = coeff + 0.5 * dt * k2
        k3 = _rhs(u3, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
        u4 = coeff + dt * k3
        k4 = _rhs(u4, h, rho, young, c, phi, dphi, phi_l, phi_r, bc, wq)
        coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    meta = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "length": length, "rho": rho, "young": young, "c": c,
        "cfl": cfl, "bc": bc, "dt": dt, "steps": steps,
        "device": str(dev), "time_integrator": time_integrator,
    }
    return centers, coeff, meta


def _rhs(coeff: torch.Tensor, h: float, rho: float, young: float, c: float,
         phi: torch.Tensor, dphi: torch.Tensor, phi_l: torch.Tensor,
         phi_r: torch.Tensor, bc: str, wq: torch.Tensor) -> torch.Tensor:
    """Semi-discrete DG residual ``U_t`` for ``[ncells, nmodes, 2]`` modes."""
    ncells, nmodes, _ = coeff.shape
    # physical flux at the quadrature nodes
    u_q = torch.einsum("ckv,qk->cqv", coeff, phi)      # [cell, quad, var]
    flux_q = torch.stack((-u_q[..., 1] / rho, -young * u_q[..., 0]), dim=-1)
    # integral of F * dphi_k/dxi (the 2/h of dphi/dx and h/2 of dx cancel)
    volume = torch.einsum("q,cqv,qk->ckv", wq, flux_q, dphi)

    left = torch.einsum("ckv,k->cv", coeff, phi_l)      # [cell, var]
    right = torch.einsum("ckv,k->cv", coeff, phi_r)
    v_l, s_l = left[..., 0], left[..., 1]
    v_r, s_r = right[..., 0], right[..., 1]

    if bc == "periodic":
        flux_at = upwind_flux(torch.roll(v_r, 1), torch.roll(s_r, 1),
                              v_l, s_l, rho, young)
        flux_l = flux_at
        flux_r = torch.roll(flux_at, -1, dims=0)
    else:
        interior = upwind_flux(v_r[:-1], s_r[:-1], v_l[1:], s_l[1:], rho, young)
        b_left = _boundary_flux(v_l[0], s_l[0], "left", rho, young, bc)
        b_right = _boundary_flux(v_r[-1], s_r[-1], "right", rho, young, bc)
        flux_at = torch.cat((b_left[None, :], interior, b_right[None, :]), dim=0)
        flux_l = flux_at[:-1]
        flux_r = flux_at[1:]

    factor = (2.0 * torch.arange(nmodes, device=coeff.device,
                                 dtype=coeff.dtype) + 1.0) / h
    out = factor[None, :, None] * (
        volume
        - flux_r[:, None, :] * phi_r[None, :, None]
        + flux_l[:, None, :] * phi_l[None, :, None]
    )
    return out


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def stress_profile(centers: torch.Tensor, coeff: torch.Tensor,
                   nsample: int = 4):
    """Cell-wise sampled ``(x, v, sigma)`` at ``nsample`` points per cell."""
    degree = coeff.shape[1] - 1
    xi, _w = np.polynomial.legendre.leggauss(nsample)
    xi_t = torch.as_tensor(xi, device=centers.device, dtype=centers.dtype)
    phi = torch.as_tensor(
        np.polynomial.legendre.legval(xi, np.eye(degree + 1)).T,
        device=centers.device, dtype=centers.dtype)      # [nsample, modes]
    h = 1.0 if centers.numel() < 2 else float(centers[1] - centers[0])
    x = centers[:, None] + 0.5 * h * xi_t[None, :]
    u = torch.einsum("ckv,qk->cqv", coeff, phi)
    return x.reshape(-1), u[..., 0].reshape(-1), u[..., 1].reshape(-1)


def relative_error(x: torch.Tensor, numerical: torch.Tensor,
                   exact: torch.Tensor, weights: Optional[torch.Tensor] = None
                   ) -> float:
    """Relative (weighted) L2 error ``||num-exact|| / ||exact||``."""
    diff = numerical - exact
    if weights is None:
        num = float(torch.sum(diff * diff))
        den = float(torch.sum(exact * exact))
    else:
        num = float(torch.sum(weights * diff * diff))
        den = float(torch.sum(weights * exact * exact))
    if den <= 0.0:
        raise ValueError("exact field has zero norm; relative error undefined")
    return math.sqrt(num / den)


def elastic_energy(centers: torch.Tensor, coeff: torch.Tensor,
                   rho: float = 1.0, young: float = 1.0) -> float:
    """Discrete elastic energy ``∫ (rho v^2 + sigma^2/E)/2 dx``."""
    degree = coeff.shape[1] - 1
    nq = max(2 * degree + 3, 4)
    xi, w = np.polynomial.legendre.leggauss(nq)
    xi_t = torch.as_tensor(xi, device=centers.device, dtype=centers.dtype)
    w_t = torch.as_tensor(w, device=centers.device, dtype=centers.dtype)
    phi = torch.as_tensor(
        np.polynomial.legendre.legval(xi, np.eye(degree + 1)).T,
        device=centers.device, dtype=centers.dtype)
    u = torch.einsum("ckv,qk->cqv", coeff, phi)
    density = 0.5 * (rho * u[..., 0] ** 2 + u[..., 1] ** 2 / young)
    h = 1.0 if centers.numel() < 2 else float(centers[1] - centers[0])
    return float(0.5 * h * torch.sum(w_t[None, :] * density))


def peak_stress(x: torch.Tensor, sigma: torch.Tensor,
                lo: Optional[float] = None, hi: Optional[float] = None):
    """Signed peak (largest absolute value) of ``sigma`` on ``[lo, hi]``.

    Returns ``(x_peak, sigma_peak)``; the sign of the peak is preserved, which
    lets a test compare the compressive and tensile pulse separately.
    """
    mask = torch.ones_like(x, dtype=torch.bool)
    if lo is not None:
        mask &= x >= lo
    if hi is not None:
        mask &= x <= hi
    if not bool(mask.any()):
        raise ValueError("empty window")
    idx = torch.argmax(torch.abs(sigma[mask]))
    xs = x[mask][idx]
    return float(xs), float(sigma[mask][idx])