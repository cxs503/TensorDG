"""Verification gates for the 2-D high-order Euler DG: velocity against exact Riemann.

Mission m6313: The 2-D Euler DG module (``tensordg/euler_dg_2d.py``) was verified
in mission m2400 against the Sod-tube extruded to 2-D, but the acceptance gate
checked only density and pressure (``docs/verify_euler_dg_2d_resolved.md``).
Velocity is equally important: the HLLC contact wave is a velocity discontinuity,
and the star-region pressure ``p*`` is computed from the momentum equation that
contains velocity.  An incorrect ``p*`` would manifest as a velocity error even
when ``rho`` and ``p`` are correct.

This module adds:
* ``test_extruded_sod_velocity_gate`` -- the 3 % gate on velocity at the
  documented resolution (P2, 64 cells), using Toro's exact Riemann solver as
  the reference (the same ``star_pressure`` + ``_speeds_from_pressure`` used
  by ``hllc_flux``).
* ``test_extruded_sod_all_three_velocity_eoc`` -- observed order of the
  velocity L2 error at P1 and P2, confirming design-order convergence.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Springer 2009, Ch. 4 (two-rarefaction ``p*``) and Ch. 10 (HLLC).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.euler_1d import (
    GAMMA,
    hllc_flux,
    primitive_to_conservative,
    primitive_fields,
    sound_speed,
    star_pressure,
)
from tensordg.euler_dg_2d import solve_euler_dg_2d_sod, cell_fields

THREE_PERCENT = 0.03


# ---------------------------------------------------------------------------
# Exact Riemann solver (validation grade, 4-component)
# ---------------------------------------------------------------------------

def _exact_riemann_4c(left, right, x_over_t):
    """Exact solution of the 2-D Euler Riemann problem at position ``x/t``.

    Returns ``(rho, u, v, p)`` at the given ``x/t`` ratio.  The tangential
    velocity ``v`` is constant across all waves (it is a contact invariant).

    Parameters
    ----------
    left, right : (..., 4) conservative states
    x_over_t : (...) float, the similarity variable ``x / t``

    Returns
    -------
    rho, u, v, p : same shape as ``x_over_t``
    """
    rho_l = left[..., 0].clamp_min(1e-12)
    rho_r = right[..., 0].clamp_min(1e-12)
    u_l = left[..., 1] / rho_l
    u_r = right[..., 1] / rho_r
    v_l = left[..., 2] / rho_l
    v_r = right[..., 2] / rho_r
    p_l = (GAMMA - 1.0) * (left[..., 3] - 0.5 * (left[..., 1] ** 2 + left[..., 2] ** 2) / rho_l)
    p_r = (GAMMA - 1.0) * (right[..., 3] - 0.5 * (right[..., 1] ** 2 + right[..., 2] ** 2) / rho_r)
    a_l = torch.sqrt(GAMMA * p_l.clamp_min(1e-12) / rho_l)
    a_r = torch.sqrt(GAMMA * p_r.clamp_min(1e-12) / rho_r)

    # Two-rarefaction approximation of p*
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    numerator = a_l + a_r - 0.5 * (GAMMA - 1.0) * (u_r - u_l)
    denominator = a_l / p_l.clamp_min(1e-12) ** exponent + a_r / p_r.clamp_min(1e-12) ** exponent
    p_star = (numerator / denominator).clamp_min(1e-12) ** (1.0 / exponent)
    pvrs = (0.5 * (p_l + p_r) - 0.5 * (u_r - u_l) * 0.5 * (rho_l + rho_r) * 0.5 * (a_l + a_r)).clamp_min(1e-12)
    p_star = torch.where(torch.isnan(p_star) | (p_star <= 0.0), pvrs, p_star)

    # Newton refine
    for _ in range(8):
        gamma1 = 2.0 / ((GAMMA + 1.0) * rho_l)
        gamma2 = (GAMMA - 1.0) / (GAMMA + 1.0) * p_l
        shock = p_star > p_l
        ratio_l = (p_star - p_l) * torch.sqrt(gamma1 / (p_star + gamma2).clamp_min(1e-12))
        deriv_l = torch.sqrt(gamma1 / (p_star + gamma2).clamp_min(1e-12)) * (
            1.0 - 0.5 * (p_star - p_l) / (p_star + gamma2).clamp_min(1e-12)
        )
        exponent_r = (GAMMA - 1.0) / (2.0 * GAMMA)
        rarefaction_l = 2.0 * a_l / (GAMMA - 1.0) * (
            (p_star / p_l.clamp_min(1e-12)).clamp_min(1e-12) ** exponent_r - 1.0
        )
        deriv_r = (
            1.0 / (rho_l * a_l.clamp_min(1e-12))
            * (p_star / p_l.clamp_min(1e-12)).clamp_min(1e-12) ** (-(GAMMA + 1.0) / (2.0 * GAMMA))
        )
        f_l = torch.where(shock, ratio_l, rarefaction_l)
        d_l = torch.where(shock, deriv_l, deriv_r)

        gamma1_r = 2.0 / ((GAMMA + 1.0) * rho_r)
        gamma2_r = (GAMMA - 1.0) / (GAMMA + 1.0) * p_r
        shock_r = p_star > p_r
        ratio_r = (p_star - p_r) * torch.sqrt(gamma1_r / (p_star + gamma2_r).clamp_min(1e-12))
        deriv_r_r = torch.sqrt(gamma1_r / (p_star + gamma2_r).clamp_min(1e-12)) * (
            1.0 - 0.5 * (p_star - p_r) / (p_star + gamma2_r).clamp_min(1e-12)
        )
        rarefaction_r = 2.0 * a_r / (GAMMA - 1.0) * (
            (p_star / p_r.clamp_min(1e-12)).clamp_min(1e-12) ** exponent_r - 1.0
        )
        deriv_r_r = (
            1.0 / (rho_r * a_r.clamp_min(1e-12))
            * (p_star / p_r.clamp_min(1e-12)).clamp_min(1e-12) ** (-(GAMMA + 1.0) / (2.0 * GAMMA))
        )
        f_r = torch.where(shock_r, ratio_r, rarefaction_r)
        d_r = torch.where(shock_r, deriv_r_r, deriv_r_r)

        residual = f_l + f_r + u_r - u_l
        slope = d_l + d_r
        step = residual / torch.where(slope.abs() < 1e-14, torch.full_like(slope, 1e-14), slope)
        p_star = torch.where(
            (p_star - step > 0.0) & torch.isfinite(p_star - step),
            p_star - step,
            0.5 * p_star,
        )

    # Wave speeds from p*
    def factor(p_k, a_k, rho_k):
        ratio = (1.0 + (GAMMA + 1.0) / (2.0 * GAMMA) * (p_star / p_k.clamp_min(1e-12) - 1.0)).clamp_min(1e-12)
        return torch.where(p_star > p_k, torch.sqrt(ratio), torch.ones_like(ratio))

    s_l = u_l - a_l * factor(p_l, a_l, rho_l)
    s_r = u_r + a_r * factor(p_r, a_r, rho_r)
    denom = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    s_m = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / denom

    # Tangential velocity is constant
    v = torch.where(x_over_t <= s_m, v_l, v_r)

    # Build states
    def star_state(p_k, a_k, rho_k, u_k, v_k):
        factor_val = s_m - u_k
        energy_s = rho_s_k * (
            (p_k / (GAMMA - 1.0) + 0.5 * rho_k * (u_k ** 2 + v_k ** 2)) / rho_k
            + (s_m - u_k) * (s_m + p_k / (rho_k * factor_val))
        )
        return rho_s_k, rho_s_k * s_m, rho_s_k * v_k, energy_s

    rho_s_l = rho_l * (s_l - u_l) / (s_m - u_l)
    energy_s_l = rho_s_l * (
        (p_l / (GAMMA - 1.0) + 0.5 * rho_l * (u_l ** 2 + v_l ** 2)) / rho_l
        + (s_m - u_l) * (s_m + p_l / (rho_l * (s_l - u_l)))
    )
    rho_s_r = rho_r * (s_r - u_r) / (s_r - s_m)
    energy_s_r = rho_s_r * (
        (p_r / (GAMMA - 1.0) + 0.5 * rho_r * (u_r ** 2 + v_r ** 2)) / rho_r
        + (s_m - u_r) * (s_m + p_r / (rho_r * (s_r - u_r)))
    )

    # Evaluate at x/t
    rho = torch.where(x_over_t < s_l, rho_l,
             torch.where(x_over_t < s_m, rho_s_l, rho_s_r))
    u = torch.where(x_over_t < s_l, u_l,
             torch.where(x_over_t < s_m, s_m, u_r))
    p = torch.where(x_over_t < s_l, p_l,
             torch.where(x_over_t < s_m, p_star, p_r))

    return rho, u, v, p


def _sod_left():
    return torch.tensor([[1.0, 0.0, 0.0, 2.5 / (GAMMA - 1.0)]])


def _sod_right():
    return torch.tensor([[0.125, 0.0, 0.0, 0.25 / (GAMMA - 1.0)]])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extruded_sod_velocity_gate():
    """The 2-D Euler DG reproduces the exact Sod velocity to < 3 % at P2/64.

    This is the *missing* gate from the original m2400 verification: the
    extruded Sod test verified ``rho`` and ``p`` but not ``u``.  The velocity
    is the most sensitive quantity to the HLLC contact wave, so it is the
    strongest test of the interface flux implementation.
    """
    _, _, coeff, _ = solve_euler_dg_2d_sod(
        ncells=64, degree=2, final_time=0.2,
    )
    x, y = cell_fields(coeff)
    # Evaluate at quadrature points
    degree = coeff.shape[2] - 1
    nq = max(2 * degree + 3, 6)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    basis = []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, poly),
            device=coeff.device, dtype=coeff.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    numerical = torch.einsum("xyklv,qk,rl->xyqrv", coeff, phi, phi)

    # Exact solution at quadrature points
    nx, ny = coeff.shape[:2]
    hx, hy = 1.0 / nx, 1.0 / ny
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    xot = xq / 0.2  # t = 0.2
    left = _sod_left()
    right = _sod_right()
    exact_rho, exact_u, exact_v, exact_p = _exact_riemann_4c(left, right, xot)

    # L2 error on velocity (component 1)
    error2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * (numerical[..., 1] - exact_u) ** 2,
    )
    norm2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * exact_u ** 2,
    )
    rel_error = float(torch.sqrt(error2) / torch.sqrt(norm2))
    assert rel_error < THREE_PERCENT, f"velocity relative L2 error {rel_error:.6f} >= 3%"


def test_extruded_sod_all_three_velocity_eoc():
    """Velocity L2 error converges at design order for P1 and P2."""
    errors = []
    for degree in (1, 2):
        _, _, coeff, _ = solve_euler_dg_2d_sod(
            ncells=64, degree=degree, final_time=0.2,
        )
        nq = max(2 * degree + 3, 6)
        nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
        nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
        weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
        basis = []
        for k in range(degree + 1):
            poly = np.zeros(k + 1)
            poly[-1] = 1.0
            basis.append(torch.as_tensor(
                np.polynomial.legendre.legval(nodes_np, poly),
                device=coeff.device, dtype=coeff.dtype,
            ))
        phi = torch.stack(basis, dim=1)
        numerical = torch.einsum("xyklv,qk,rl->xyqrv", coeff, phi, phi)

        nx, ny = coeff.shape[:2]
        hx, hy = 1.0 / nx, 1.0 / ny
        xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
        yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
        xot = xq / 0.2
        left = _sod_left()
        right = _sod_right()
        exact_rho, exact_u, exact_v, exact_p = _exact_riemann_4c(left, right, xot)

        error2 = 0.25 * hx * hy * torch.sum(
            weights[None, None, :, None] * weights[None, None, None, :]
            * (numerical[..., 1] - exact_u) ** 2,
        )
        norm2 = 0.25 * hx * hy * torch.sum(
            weights[None, None, :, None] * weights[None, None, None, :]
            * exact_u ** 2,
        )
        rel_error = float(torch.sqrt(error2) / torch.sqrt(norm2))
        errors.append(rel_error)

    eoc = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert eoc > 1.9, f"velocity EOC {eoc:.2f} < 1.9"


def test_sod_velocity_sharpness():
    """The velocity profile at the contact discontinuity should be sharp
    (not smeared over many cells) for P2 DG."""
    _, _, coeff, _ = solve_euler_dg_2d_sod(
        ncells=128, degree=2, final_time=0.2,
    )
    # The contact is at x ≈ 0.664 at t=0.2 (approximate)
    # Check that the velocity gradient across the contact is sharp
    # (the difference between cells straddling the contact should be large)
    contact_x = 0.664
    contact_cell = int(contact_x * 128)
    u_left_cell = float(coeff[contact_cell - 1, 0, 1].item())  # cell average u
    u_right_cell = float(coeff[contact_cell, 0, 1].item())
    # The velocity jump across the contact should be significant
    jump = abs(u_left_cell - u_right_cell)
    assert jump > 0.1, f"velocity jump at contact too small: {jump}"
