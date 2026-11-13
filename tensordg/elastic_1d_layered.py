"""One-dimensional linear elastodynamics in a *layered* bar (modal DG).

Motivation
----------
:mod:`tensordg.elastic_1d` solves the velocity--stress system

.. math::

   v_t - \\frac{1}{\\rho}\\frac{\\partial \\sigma}{\\partial x} = 0,
   \\qquad
   \\sigma_t - E\\frac{\\partial v}{\\partial x} = 0

for **constant** :math:`\\rho, E`.  Physical interest in a bar usually starts
where that assumption ends: at an impedance jump.  The repository has explicit
d'Alembert end-reflection cases (``CASE-DG-DALEMBERT-REFLECTIONS``) and a
homogeneous dispersion analysis (``docs/elastic-dispersion-analysis.md``), but
**no** verification of a material interface: there the wave is partly reflected
and partly transmitted with the closed-form coefficients

.. math::

   R = \\frac{Z_2 - Z_1}{Z_1 + Z_2}, \\qquad
   T = \\frac{2 Z_2}{Z_1 + Z_2}, \\qquad Z_i = \\rho_i c_i = \\sqrt{\\rho_i E_i},

which are *exact* for the linear system, so they form a sharp (<2%) acceptance
criterion and, unlike an end reflection (a pure sign flip), exercise the
interface Riemann solver itself.

What this module adds
---------------------
* A per-cell material description ``(rho, young)`` of length ``ncells``
  (piecewise constant, arbitrary number of layers, jumps only on cell faces).
* An **exact interface Riemann solver** for the heterogeneous system.  With
  :math:`Z_L, Z_R` the impedances of the two neighbouring cells and the
  invariants :math:`a = \\sigma_L - Z_L v_L` (incoming from the left,
  ``+c_L``) and :math:`b = \\sigma_R + Z_R v_R` (incoming from the right,
  ``-c_R``),

  .. math::

     v^* = \\frac{b-a}{Z_L + Z_R}, \\qquad
     \\sigma^* = \\frac{Z_R a + Z_L b}{Z_L + Z_R}.

  ``v`` and ``sigma`` are continuous across the material face, so the *state*
  is single valued, while the flux ``(-sigma/rho, -E v)`` is **side
  dependent** -- exactly the point a homogeneous socket cannot express.
  For :math:`Z_L = Z_R = \\rho c` the formula collapses algebraically to
  :func:`tensordg.elastic_1d.upwind_flux`, which is verified in the tests.
* Closed-form scattering solution (:func:`exact_rightgoing_scattering`) of a
  right-going pulse hitting one interface, valid up to the first boundary
  return.
* Piecewise energy functional (:func:`layered_energy`) so conservation can be
  checked with a discontinuous :math:`\\rho, E`.

Boundary conditions reuse the characteristic closures of
:mod:`tensordg.elastic_1d` with the *local* material of the boundary cell.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .advection_1d import MAX_DEGREE, _operators
from .elastic_1d import (
    _boundary_flux,
    _project,
    gaussian_pulse,
)

__all__ = [
    "impedance",
    "wave_speed",
    "scattering_coefficients",
    "interface_state",
    "layered_interface_flux",
    "solve_elastic_1d_layered",
    "layered_energy",
    "layered_stress_profile",
    "exact_rightgoing_scattering",
    "scattering_valid_time",
    "rightgoing_pulse_initial_state",
]

SUPPORTED_BC = ("periodic", "fixed-free", "free-fixed")


# ---------------------------------------------------------------------------
# material helpers
# ---------------------------------------------------------------------------
def wave_speed(rho: float, young: float) -> float:
    """Bar wave speed ``c = sqrt(E/rho)``."""
    if rho <= 0 or young <= 0:
        raise ValueError("rho and young must be positive")
    return math.sqrt(young / rho)


def impedance(rho: float, young: float) -> float:
    """Bar impedance ``Z = rho c = sqrt(rho E)``."""
    return rho * wave_speed(rho, young)


def scattering_coefficients(rho_left: float, young_left: float,
                            rho_right: float, young_right: float):
    """Exact ``(R, T)`` stress coefficients at a material interface.

    ``sigma_reflected = R * sigma_incident`` and
    ``sigma_transmitted = T * sigma_incident`` for a wave travelling from the
    left material into the right one.  ``1 + R == T`` always holds; ``R -> 0``
    when the impedances match even if ``rho, E`` both change.
    """
    z_l = impedance(rho_left, young_left)
    z_r = impedance(rho_right, young_right)
    return (z_r - z_l) / (z_l + z_r), 2.0 * z_r / (z_l + z_r)


# ---------------------------------------------------------------------------
# heterogeneous interface Riemann solver
# ---------------------------------------------------------------------------
def interface_state(v_l: torch.Tensor, s_l: torch.Tensor,
                    v_r: torch.Tensor, s_r: torch.Tensor,
                    z_l: torch.Tensor, z_r: torch.Tensor):
    """Riemann state ``(v*, sigma*)`` at a face between two materials.

    ``z_l`` / ``z_r`` are impedance tensors broadcastable against the trace
    tensors.  Incoming invariants: ``w+ = sigma - Z v`` from the left,
    ``w- = sigma + Z v`` from the right.
    """
    a = s_l - z_l * v_l
    b = s_r + z_r * v_r
    denom = z_l + z_r
    v_star = (b - a) / denom
    sigma_star = (z_r * a + z_l * b) / denom
    return v_star, sigma_star


def layered_interface_flux(v_l: torch.Tensor, s_l: torch.Tensor,
                           v_r: torch.Tensor, s_r: torch.Tensor,
                           rho_l: torch.Tensor, e_l: torch.Tensor,
                           rho_r: torch.Tensor, e_r: torch.Tensor):
    """Side-dependent numerical flux ``(F_left_cell, F_right_cell)``.

    Both sides see the *same* state ``(v*, sigma*)`` but their own material in
    ``F = (-sigma/rho, -E v)``.  This is the exact Godunov flux for the
    heterogeneous linear system (no artificial reflection is introduced).
    """
    z_l = rho_l * torch.sqrt(e_l / rho_l)
    z_r = rho_r * torch.sqrt(e_r / rho_r)
    v_star, sigma_star = interface_state(v_l, s_l, v_r, s_r, z_l, z_r)
    flux_left = torch.stack((-sigma_star / rho_l, -e_l * v_star), dim=-1)
    flux_right = torch.stack((-sigma_star / rho_r, -e_r * v_star), dim=-1)
    return flux_left, flux_right


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def _as_material_array(values, ncells: int, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        arr = values
    elif isinstance(values, (int, float)):
        arr = torch.full((ncells,), float(values))
    else:
        arr = torch.as_tensor(list(values), dtype=torch.float64)
    if arr.numel() == 1:
        arr = arr.expand(ncells).clone()
    if arr.numel() != ncells:
        raise ValueError(
            f"{name} must be a scalar or have one entry per cell "
            f"({ncells}); got {arr.numel()}")
    if bool(torch.any(arr <= 0)):
        raise ValueError(f"{name} must be strictly positive in every cell")
    return arr


def solve_elastic_1d_layered(ncells: int = 200, degree: int = 2,
                             final_time: float = 0.3, length: float = 1.0,
                             rho=(1.0,), young=(1.0,), cfl: float = 0.3,
                             bc: str = "periodic",
                             initial_velocity: Optional[Callable] = None,
                             initial_stress: Optional[Callable] = None,
                             device: str = "cpu",
                             dtype: torch.dtype = torch.float64,
                             time_integrator: str = "ssp_rk3"):
    """DG solve of the layered 1-D velocity--stress system.

    ``rho`` / ``young`` are piecewise-constant per cell (scalar, sequence of
    per-cell values, or tensor).  Returns ``(centers, coeff, meta)`` with
    ``coeff`` of shape ``[ncells, degree+1, 2]`` (last axis = ``(v, sigma)``),
    ``meta`` additionally carrying the per-cell materials, wave speeds,
    impedances, ``dt`` and the CFL-safe step count.
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

    dev = torch.device(device)
    rho_c = _as_material_array(rho, ncells, "rho").to(dev, dtype)
    e_c = _as_material_array(young, ncells, "young").to(dev, dtype)
    c_c = torch.sqrt(e_c / rho_c)
    z_c = rho_c * c_c
    c_max = float(torch.max(c_c))

    h = length / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    xq, wq, phi, _volume, phi_l, phi_r = _operators(degree, dev, dtype)
    dphi = torch.stack([
        torch.as_tensor(
            np.polynomial.legendre.legval(
                xq.detach().cpu().numpy(),
                np.polynomial.legendre.legder(np.eye(degree + 1)[k])),
            device=dev, dtype=dtype)
        for k in range(degree + 1)
    ], dim=1)

    if initial_velocity is None:
        initial_velocity = lambda x: gaussian_pulse(x)
    if initial_stress is None:
        initial_stress = lambda x: torch.zeros_like(x)
    v0 = _project(initial_velocity, centers, h, xq, wq, phi, degree)
    s0 = _project(initial_stress, centers, h, xq, wq, phi, degree)
    coeff = torch.stack((v0, s0), dim=-1)

    if final_time == 0 or c_max == 0:
        steps, dt = 0, 0.0
    else:
        dt_nominal = cfl * h / (c_max * (2 * degree + 1))
        steps = max(1, math.ceil(final_time / dt_nominal))
        dt = final_time / steps

    args = (h, rho_c, e_c, c_c, z_c, phi, dphi, phi_l, phi_r, bc, wq)
    for _ in range(steps):
        k1 = _rhs_layered(coeff, *args)
        if time_integrator == "ssp_rk3":
            u1 = coeff + dt * k1
            k2 = _rhs_layered(u1, *args)
            u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
            k3 = _rhs_layered(u2, *args)
            coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
            continue
        u2 = coeff + 0.5 * dt * k1
        k2 = _rhs_layered(u2, *args)
        u3 = coeff + 0.5 * dt * k2
        k3 = _rhs_layered(u3, *args)
        u4 = coeff + dt * k3
        k4 = _rhs_layered(u4, *args)
        coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    meta = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "length": length, "cfl": cfl, "bc": bc, "dt": dt, "steps": steps,
        "device": str(dev), "time_integrator": time_integrator,
        "rho": rho_c.detach().cpu().tolist(),
        "young": e_c.detach().cpu().tolist(),
        "c": c_c.detach().cpu().tolist(),
        "impedance": z_c.detach().cpu().tolist(),
        "c_max": c_max,
        "n_interfaces": int(sum(
            1 for i in range(1, ncells) if float(rho_c[i] * c_c[i]) != float(rho_c[i - 1] * c_c[i - 1])
        )),
    }
    return centers, coeff, meta


def _rhs_layered(coeff, h, rho_c, e_c, c_c, z_c, phi, dphi, phi_l, phi_r,
                 bc, wq) -> torch.Tensor:
    """Semi-discrete residual for per-cell materials."""
    ncells, nmodes, _ = coeff.shape
    u_q = torch.einsum("ckv,qk->cqv", coeff, phi)                 # [cell, quad, 2]
    flux_q = torch.stack(
        (-u_q[..., 1] / rho_c[:, None], -e_c[:, None] * u_q[..., 0]), dim=-1)
    volume = torch.einsum("q,cqv,qk->ckv", wq, flux_q, dphi)

    left = torch.einsum("ckv,k->cv", coeff, phi_l)
    right = torch.einsum("ckv,k->cv", coeff, phi_r)
    v_l, s_l = left[..., 0], left[..., 1]
    v_r, s_r = right[..., 0], right[..., 1]

    if bc == "periodic":
        # faces: (ncells-1) interior + one wrap face (cell ncells-1 -> cell 0)
        f_left_int, f_right_int = layered_interface_flux(
            v_r[:-1], s_r[:-1], v_l[1:], s_l[1:],
            rho_c[:-1], e_c[:-1], rho_c[1:], e_c[1:])
        f_wrap_l, f_wrap_r = layered_interface_flux(
            v_r[-1:], s_r[-1:], v_l[:1], s_l[:1],
            rho_c[-1:], e_c[-1:], rho_c[:1], e_c[:1])
        # f_left_int[i] is the flux the *left* cell i sees at its right face;
        # f_right_int[i] is the flux the *right* cell i+1 sees at its left
        # face.  For a homogeneous medium the two coincide, which is why this
        # pairing must be verified on a material jump (see tests).
        flux_l = torch.cat((f_wrap_r, f_left_int), dim=0)
        flux_r = torch.cat((f_right_int, f_wrap_l), dim=0)
    else:
        f_left_int, f_right_int = layered_interface_flux(
            v_r[:-1], s_r[:-1], v_l[1:], s_l[1:],
            rho_c[:-1], e_c[:-1], rho_c[1:], e_c[1:])
        b_left = _boundary_flux(v_l[0], s_l[0], "left",
                                float(rho_c[0]), float(e_c[0]), bc)
        b_right = _boundary_flux(v_r[-1], s_r[-1], "right",
                                 float(rho_c[-1]), float(e_c[-1]), bc)
        flux_l = torch.cat((b_left[None, :], f_right_int), dim=0)
        flux_r = torch.cat((f_left_int, b_right[None, :]), dim=0)

    factor = (2.0 * torch.arange(nmodes, device=coeff.device,
                                 dtype=coeff.dtype) + 1.0) / h
    return factor[None, :, None] * (
        volume
        - flux_r[:, None, :] * phi_r[None, :, None]
        + flux_l[:, None, :] * phi_l[None, :, None]
    )


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def layered_energy(coeff: torch.Tensor, h: float, rho_c, e_c,
                   degree: Optional[int] = None) -> float:
    """Total elastic energy ``int (rho v^2 / 2 + sigma^2 / (2 E)) dx``."""
    degree = coeff.shape[1] - 1 if degree is None else degree
    dev, dtype = coeff.device, coeff.dtype
    xq, wq, phi, _v, _l, _r = _operators(degree, dev, dtype)
    u = torch.einsum("ckv,qk->cqv", coeff, phi)
    rho_t = torch.as_tensor(rho_c, device=dev, dtype=dtype).reshape(-1, 1)
    e_t = torch.as_tensor(e_c, device=dev, dtype=dtype).reshape(-1, 1)
    dens = 0.5 * rho_t * u[..., 0] ** 2 + 0.5 * u[..., 1] ** 2 / e_t
    return float(0.5 * h * torch.sum(wq[None, :] * dens))


def layered_stress_profile(centers: torch.Tensor, coeff: torch.Tensor,
                           nsample: int = 8):
    """Sampled ``(x, v, sigma)`` inside every cell (``nsample`` per cell)."""
    degree = coeff.shape[1] - 1
    xi, _w = np.polynomial.legendre.leggauss(nsample)
    dev, dtype = centers.device, centers.dtype
    phi = torch.as_tensor(
        np.polynomial.legendre.legval(xi, np.eye(degree + 1)).T,
        device=dev, dtype=dtype)
    h = 1.0 if centers.numel() < 2 else float(centers[1] - centers[0])
    x = centers[:, None] + 0.5 * h * torch.as_tensor(xi, device=dev,
                                                     dtype=dtype)[None, :]
    u = torch.einsum("ckv,qk->cqv", coeff, phi)
    return x.reshape(-1), u[..., 0].reshape(-1), u[..., 1].reshape(-1)


# ---------------------------------------------------------------------------
# closed-form reference: single interface, right-going pulse
# ---------------------------------------------------------------------------
def rightgoing_pulse_initial_state(interface: float, rho_left: float = 1.0,
                                   young_left: float = 1.0,
                                   center: float = 0.2,
                                   width: float = 0.03):
    """Initial ``(v0, sigma0)`` of a pure right-going pulse in layer 1.

    ``sigma0 = f(x)`` for ``x < interface`` and ``0`` beyond, with
    ``v0 = -sigma0 / Z1``.

    Sign convention (checked against the mother case
    :func:`tensordg.elastic_1d.dalembert_pulse`): for this velocity--stress
    system a wave travelling in ``+x`` carries ``sigma = -Z v`` while one
    travelling in ``-x`` carries ``sigma = +Z v``.  Hence ``v0 = -sigma0/Z1``
    makes ``w- = sigma + Z v = 0`` initially, i.e. nothing propagates to the
    left until a boundary return -- which is what makes the closed form below a
    single-scattering formula.
    """
    z1 = impedance(rho_left, young_left)

    def sigma0(x):
        inside = x < interface
        return torch.where(inside, gaussian_pulse(x, center, width),
                           torch.zeros_like(x))

    def v0(x):
        return -sigma0(x) / z1

    return v0, sigma0


def scattering_valid_time(interface: float, length: float = 1.0,
                          rho_left: float = 1.0, young_left: float = 1.0,
                          rho_right: float = 1.0, young_right: float = 1.0
                          ) -> float:
    """Largest ``t`` for which :func:`exact_rightgoing_scattering` is exact.

    Set by the first boundary return: the reflected packet needs
    ``interface / c_1`` to reach ``x = 0`` and the transmitted one
    ``(length - interface) / c_2`` to reach ``x = L`` (periodic ends included,
    since the wrap-around is the same distance as the far end for the
    transmitted packet only when ``bc='periodic'``; the bound below is the
    conservative one for both boundary families).
    """
    c1 = wave_speed(rho_left, young_left)
    c2 = wave_speed(rho_right, young_right)
    return min(interface / c1, (length - interface) / c2)


def exact_rightgoing_scattering(x: torch.Tensor, t: float,
                                interface: float, length: float = 1.0,
                                rho_left: float = 1.0,
                                young_left: float = 1.0,
                                rho_right: float = 1.0,
                                young_right: float = 1.0,
                                center: float = 0.2, width: float = 0.03,
                                strict: bool = True):
    """Exact ``(v, sigma)`` of a right-going pulse crossing one interface.

    Left layer (``x < interface``):
    ``sigma = f(x - c1 t) + R f(2 x_i - x - c1 t)``,
    ``v = [R f(2 x_i - x - c1 t) - f(x - c1 t)] / Z1`` (the minus sign is the
    ``+x`` branch ``sigma = -Z v``, see
    :func:`rightgoing_pulse_initial_state`).
    Right layer: the transmitted wave carries the *time history of the
    interface*, ``sigma = T f(x_i - c1 (t - (x - x_i)/c2))``,
    ``v = -sigma / Z2``.  Continuity of ``sigma`` and ``v`` at ``x = x_i`` is an
    identity (``1 + R = T``), which the tests check independently.
    """
    c1 = wave_speed(rho_left, young_left)
    c2 = wave_speed(rho_right, young_right)
    z1 = impedance(rho_left, young_left)
    z2 = impedance(rho_right, young_right)
    r, tcoef = scattering_coefficients(rho_left, young_left, rho_right, young_right)
    if strict:
        limit = scattering_valid_time(interface, length, rho_left, young_left,
                                      rho_right, young_right)
        if t > limit * (1.0 + 1e-12):
            raise ValueError(
                f"exact_rightgoing_scattering is exact only for t <= {limit:g} "
                f"(got {t:g}); a boundary return has re-entered the field.")
    inc = gaussian_pulse(x - c1 * t, center, width)
    refl = r * gaussian_pulse(2.0 * interface - x - c1 * t, center, width)
    tran = tcoef * gaussian_pulse(
        interface - c1 * (t - (x - interface) / c2), center, width)
    in_left = x < interface
    v = torch.where(in_left, (refl - inc) / z1, -tran / z2)
    sigma = torch.where(in_left, inc + refl, tran)
    return v, sigma