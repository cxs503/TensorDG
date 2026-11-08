"""One-dimensional shallow water equations with DG(P1/P2).

Solves the 1-D SWE system with bed slope source term:

    ∂h/∂t + ∂(hu)/∂x = 0
    ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = −g h ∂z_b/∂x

Conservative variables U = [h, hu], physical flux F = [hu, hu² + ½gh²].

Features
--------
* Modal Legendre DG of degree 1 or 2 (P1/P2).
* HLL or Lax-Friedrichs numerical flux at interfaces.
* Well-balanced treatment: source term projected consistently with flux.
* Minmod/TVB limiter for shock capturing.
* SSP-RK2 / SSP-RK3 time integration.
* Positivity-preserving filter (h ≥ 0).

Analytical benchmarks
---------------------
* Lake-at-rest: η = h + z_b = const, u = 0 → zero drift to machine precision.
* Ritter dam break (flat bed, dry downstream): exact self-similar solution.
* Thacker (1981) parabolic basin oscillation: periodic analytical solution.

References
----------
* Audusse et al. (2004), "A fast and stable well-balanced scheme..."
* Ritter (1892), "Über die Fortpflanzung von Wasserwellen."
* Thacker (1981), "Some exact solutions of the nonlinear shallow-water equations."
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

from .limiters import apply_limiter

GRAVITY = 9.81  # m/s²


def physical_flux(u: torch.Tensor) -> torch.Tensor:
    """Physical flux F(U) for U = [h, hu]."""
    h = u[..., 0].clamp_min(0.0)
    hu = u[..., 1]
    return torch.stack((hu, hu ** 2 / h.clamp_min(1e-12) + 0.5 * GRAVITY * h ** 2), dim=-1)


def sound_speed(u: torch.Tensor) -> torch.Tensor:
    """Linear wave speed c = sqrt(g h)."""
    return torch.sqrt(GRAVITY * u[..., 0].clamp_min(1e-12))


def hll_flux(left: torch.Tensor, right: torch.Tensor, flux_kind: str = "hll") -> torch.Tensor:
    """HLL numerical flux for the SWE system.

    Uses Davis wave-speed estimates. Falls back to Lax-Friedrichs if
    ``flux_kind == "lf"``.
    """
    if flux_kind == "lf":
        fl = physical_flux(left)
        fr = physical_flux(right)
        h_l = left[..., 0].clamp_min(1e-12)
        h_r = right[..., 0].clamp_min(1e-12)
        u_l = left[..., 1] / h_l
        u_r = right[..., 1] / h_r
        c_l = sound_speed(left)
        c_r = sound_speed(right)
        alpha = torch.maximum(torch.abs(u_l) + c_l, torch.abs(u_r) + c_r)
        return 0.5 * (fl + fr) - 0.5 * alpha[..., None] * (right - left)

    h_l = left[..., 0].clamp_min(1e-12)
    h_r = right[..., 0].clamp_min(1e-12)
    u_l = left[..., 1] / h_l
    u_r = right[..., 1] / h_r
    c_l = sound_speed(left)
    c_r = sound_speed(right)

    s_l = torch.minimum(u_l - c_l, u_r - c_r)  # [ncells]
    s_r = torch.maximum(u_l + c_l, u_r + c_r)  # [ncells]

    fl = physical_flux(left)   # [ncells, 2]
    fr = physical_flux(right)  # [ncells, 2]

    denom = s_r - s_l  # [ncells]
    safe_denom = torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)

    # HLL flux: (s_r * F_L - s_l * F_R + s_l * s_r * (U_R - U_L)) / (s_r - s_l)
    # Need [..., None] to broadcast [ncells] with [ncells, 2]
    flux = (s_r[..., None] * fl - s_l[..., None] * fr
            + s_l[..., None] * s_r[..., None] * (right - left)) / safe_denom[..., None]

    flux = torch.where((s_l >= 0.0)[..., None], fl, flux)
    flux = torch.where((s_r <= 0.0)[..., None], fr, flux)
    return flux


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    """Build Legendre basis, derivatives, and boundary values."""
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    values, derivatives = [], []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        deriv = np.polynomial.legendre.legder(poly)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, poly), device=device, dtype=dtype,
        ))
        deriv_vals = (np.polynomial.legendre.legval(nodes_np, deriv)
                      if len(deriv) else np.zeros_like(nodes_np))
        derivatives.append(torch.as_tensor(deriv_vals, device=device, dtype=dtype))

    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _rhs(coeff: torch.Tensor, dx: float, weights: torch.Tensor,
         phi: torch.Tensor, dphi: torch.Tensor,
         phi_left: torch.Tensor, phi_right: torch.Tensor,
         zb_jump: torch.Tensor,
         flux_kind: str = "hll", periodic: bool = False,
         time: float = 0.0,
         bc_left_flux=None, bc_right_flux=None) -> torch.Tensor:
    """Semi-discrete DG residual for SWE with bed slope source term.

    The source term −g h ∂z_b/∂x is treated by modal projection.
    For piecewise-linear bed, ∂z_b/∂x = (z_b,R − z_b,L)/dx is constant per cell.
    The weak-form source for mode k in the momentum equation is:
        S_k = −g (∂z_b/∂x) · c_k^h     (the (2k+1)/dx weak-form prefactor
        cancels the 2/(2k+1) modal moment, so the source is weight-free)
    where c_k^h is the k-th modal coefficient of h.

    For lake-at-rest (h + z_b = const, hu = 0), the flux gradient and source
    cancel exactly at the discrete level, giving machine-precision well-balancedness.
    """
    at_quad = torch.einsum("cmv,qm->cqv", coeff, phi)
    flux_quad = physical_flux(at_quad)
    volume = torch.einsum("cqv,q,qk->ckv", flux_quad, weights, dphi)

    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)

    right_neighbour = torch.roll(left_trace, shifts=-1, dims=0)
    if not periodic:
        right_neighbour[-1] = right_trace[-1]

    interface_flux = hll_flux(right_trace, right_neighbour, flux_kind)
    left_interface_flux = torch.roll(interface_flux, shifts=1, dims=0)
    if not periodic:
        left_interface_flux[0] = hll_flux(left_trace[:1], left_trace[:1], flux_kind)

    # Optional exact boundary data (verification against analytic solutions).
    if bc_left_flux is not None:
        left_interface_flux[0] = torch.as_tensor(
            bc_left_flux(time), dtype=coeff.dtype, device=coeff.device)
    if bc_right_flux is not None:
        interface_flux[-1] = torch.as_tensor(
            bc_right_flux(time), dtype=coeff.dtype, device=coeff.device)

    dbdx = zb_jump / dx  # [ncells]

    out = torch.empty_like(coeff)
    for k in range(coeff.shape[1]):
        # Flux divergence term (both equations)
        rhs_k = ((2 * k + 1) / dx) * (
            volume[:, k]
            - interface_flux * phi_right[k]
            + left_interface_flux * phi_left[k]
        )
        # Source term: only in momentum equation (component 1)
        # Weak form of -g h dz_b/dx on mode k:
        #   ((2k+1)/dx) * int_cell S phi_k dx with S = -g dbdx * sum_m c_m^h phi_m
        #   = -g dbdx * c_k^h  ((2k+1)/2 and 2/(2k+1) cancel exactly)
        src_k = -GRAVITY * dbdx * coeff[:, k, 0]
        rhs_k[:, 1] += src_k
        out[:, k] = rhs_k

    return out


def _positivity_filter(coeff: torch.Tensor, check_points: torch.Tensor,
                       h_min: float = 1e-12) -> torch.Tensor:
    """Ensure h ≥ h_min at all quadrature and interface points."""
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    h_vals = values[..., 0]
    valid = (h_vals > h_min) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=1)

    average_only = torch.zeros_like(coeff)
    average_only[:, 0] = coeff[:, 0]

    return torch.where(valid_cell[:, None, None], coeff, average_only)


def solve_swe_1d(
    initial_h: Callable[[torch.Tensor], torch.Tensor],
    initial_u: Callable[[torch.Tensor], torch.Tensor],
    zb: Callable[[torch.Tensor], torch.Tensor],
    ncells: int = 100,
    degree: int = 1,
    final_time: float = 1.0,
    domain: tuple = (0.0, 1.0),
    cfl: float = 0.15,
    flux_kind: str = "hll",
    limiter: Optional[str] = None,
    tvb_constant: float = 1.0,
    limiter_stage: str = "final",
    rk_order: int = 3,
    h_min: float = 1e-12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    periodic: bool = False,
    bc_left_flux: Optional[Callable[[float], torch.Tensor]] = None,
    bc_right_flux: Optional[Callable[[float], torch.Tensor]] = None,
) -> tuple:
    """Solve the 1-D shallow water equations with DG.

    Parameters
    ----------
    initial_h : callable  h(x) at t=0
    initial_u : callable  u(x) at t=0
    zb : callable  bed elevation z_b(x)
    ncells : number of cells
    degree : polynomial degree (1 or 2)
    final_time : simulation end time
    domain : (x_left, x_right)
    cfl : CFL number
    flux_kind : "hll" or "lf"
    limiter : None, "tvb", or "weno"
    tvb_constant : TVB limiter constant M
    limiter_stage : "every" or "final"
    rk_order : SSP-RK order (2 or 3)
    h_min : minimum water depth for positivity
    periodic : periodic boundary conditions
    bc_left_flux : callable t -> (hu, hu^2 + g h^2 / 2) at x = x_left
        Exact boundary flux (verification mode). Overrides the default
        transmissive closure on the left boundary.
    bc_right_flux : callable t -> (hu, hu^2 + g h^2 / 2) at x = x_right
        Idem on the right boundary.

    Returns
    -------
    x : (ncells,) cell centres
    coeff : (ncells, degree+1, 2) modal coefficients [h, hu]
    zb_faces : (ncells, 2) bed elevation at left/right faces
    metadata : dict
    """
    if ncells < 4 or degree not in (1, 2):
        raise ValueError("ncells >= 4 and degree in {1, 2}")
    if rk_order not in (2, 3):
        raise ValueError("rk_order must be 2 or 3")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    if flux_kind not in ("hll", "lf"):
        raise ValueError("flux_kind must be 'hll' or 'lf'")

    lower, upper = domain
    if not upper > lower:
        raise ValueError("domain must be increasing")

    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx

    # Bed elevation at cell faces
    zb_left_faces = lower + torch.arange(ncells, device=dev, dtype=dtype) * dx
    zb_right_faces = zb_left_faces + dx
    zb_l = zb(zb_left_faces)
    zb_r = zb(zb_right_faces)
    zb_jump = zb_r - zb_l  # [ncells]

    # Initial condition at quadrature points
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    node_x = x[:, None] + 0.5 * dx * nodes[None, :]
    h0 = initial_h(node_x)
    u0 = initial_u(node_x)
    hu0 = h0 * u0

    # L2 projection onto modal basis
    coeff = torch.zeros((ncells, degree + 1, 2), device=dev, dtype=dtype)
    for k in range(degree + 1):
        coeff[:, k, 0] = ((2 * k + 1) / 2.0) * torch.einsum(
            "cq,q,q->c", h0, weights, phi[:, k])
        coeff[:, k, 1] = ((2 * k + 1) / 2.0) * torch.einsum(
            "cq,q,q->c", hu0, weights, phi[:, k])

    coeff = _positivity_filter(coeff, check_points, h_min)

    limit_each_stage = limiter_stage == "every"
    time, steps, dt = 0.0, 0, 0.0

    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        h_vals = values[..., 0].clamp_min(1e-12)
        u_vals = values[..., 1] / h_vals
        c_vals = torch.sqrt(GRAVITY * h_vals)
        max_speed = float(torch.max(torch.abs(u_vals) + c_vals).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        if dt < 1e-15:
            break

        if rk_order == 3:
            k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time,
                      bc_left_flux, bc_right_flux)
            u1 = coeff + dt * k1
            if limit_each_stage:
                u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant)
            u1 = _positivity_filter(u1, check_points, h_min)

            k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time + dt,
                      bc_left_flux, bc_right_flux)
            u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
            if limit_each_stage:
                u2 = apply_limiter(u2, dx, kind=limiter, tvb_constant=tvb_constant)
            u2 = _positivity_filter(u2, check_points, h_min)

            k3 = _rhs(u2, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time + 0.5 * dt,
                      bc_left_flux, bc_right_flux)
            coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        else:
            k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time,
                      bc_left_flux, bc_right_flux)
            u1 = coeff + dt * k1
            if limit_each_stage:
                u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant)
            u1 = _positivity_filter(u1, check_points, h_min)

            k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic)
            coeff = 0.5 * coeff + 0.5 * (u1 + dt * k2)

        if limiter_stage == "final" and limiter is not None:
            coeff = apply_limiter(coeff, dx, kind=limiter, tvb_constant=tvb_constant)
        coeff = _positivity_filter(coeff, check_points, h_min)

        h_avg = coeff[:, 0, 0]
        if not torch.isfinite(coeff).all() or (h_avg < -h_min).any():
            raise FloatingPointError("Nonphysical SWE state; reduce CFL or check IC")

        time += dt
        steps += 1

    zb_faces = torch.stack((zb_l, zb_r), dim=1)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "domain": (lower, upper), "cfl": cfl,
        "flux": flux_kind, "limiter": limiter, "rk_order": rk_order,
        "device": str(dev), "limiter_stage": limiter_stage,
        "exact_boundary_flux": bc_left_flux is not None or bc_right_flux is not None,
    }
    return x, coeff, zb_faces, metadata


def reconstruct_fields(coeff: torch.Tensor, zb_faces: torch.Tensor) -> tuple:
    """Reconstruct h, u, η at cell centres from modal coefficients.

    Returns (h_center, u_center, eta_center) each of shape (ncells,).
    """
    h_avg = coeff[:, 0, 0]
    hu_avg = coeff[:, 0, 1]
    zb_center = 0.5 * (zb_faces[:, 0] + zb_faces[:, 1])
    u_center = hu_avg / h_avg.clamp_min(1e-12)
    eta_center = h_avg + zb_center
    return h_avg, u_center, eta_center
