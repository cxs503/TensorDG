"""High-order one-dimensional Euler DG with HLLC and local positivity fallback."""
from __future__ import annotations

import numpy as np
import torch

from .euler_1d import GAMMA, hllc_flux, physical_flux, primitive_to_conservative, sound_speed


def _operators(degree, device, dtype):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(degree + 2, 4))
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _positivity_filter(coeff, check_points):
    """Revert only inadmissible cells to their positive cell-average state."""
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    rho = values[..., 0]
    pressure = (GAMMA - 1.0) * (values[..., 2] - 0.5 * values[..., 1] ** 2 / rho.clamp_min(1e-12))
    valid = (rho > 0.0) & (pressure > 0.0) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=1)
    average_only = torch.zeros_like(coeff)
    average_only[:, 0] = coeff[:, 0]
    return torch.where(valid_cell[:, None, None], coeff, average_only)


def _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right):
    at_quadrature = torch.einsum("cmv,qm->cqv", coeff, phi)
    flux_quadrature = physical_flux(at_quadrature)
    volume = torch.einsum("cqv,q,qk->ckv", flux_quadrature, weights, dphi)
    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)
    flux_right = hllc_flux(right_trace, torch.roll(left_trace, shifts=-1, dims=0))
    flux_left = torch.roll(flux_right, shifts=1, dims=0)
    flux_right[-1] = physical_flux(right_trace[-1:])
    flux_left[0] = physical_flux(left_trace[:1])
    out = torch.empty_like(coeff)
    for k in range(coeff.shape[1]):
        out[:, k] = ((2 * k + 1) / dx) * (
            volume[:, k] - flux_right * phi_right[k] + flux_left * phi_left[k]
        )
    return out


def solve_euler_dg_1d(
    left_state,
    right_state,
    ncells=200,
    degree=2,
    final_time=0.13,
    diaphragm=0.5,
    domain=(0.0, 1.0),
    cfl=0.12,
    device="cpu",
    dtype=torch.float64,
):
    """Solve a Riemann problem with modal DG/HLLC/SSP-RK3.

    The local filter is positivity-based, not a TVD proof. It is intended for
    verification and moderate shock strengths, not as a general shock-capturing
    guarantee.
    """
    if ncells < 20 or degree not in (1, 2, 3) or final_time < 0.0 or not 0 < cfl <= 0.5:
        raise ValueError("invalid cells, degree, final_time, or CFL")
    lower, upper = domain
    if not upper > lower or not lower < diaphragm < upper:
        raise ValueError("domain must be increasing and contain the diaphragm")
    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    mask = x < diaphragm
    rho = torch.where(mask, torch.full_like(x, left_state[0]), torch.full_like(x, right_state[0]))
    velocity = torch.where(mask, torch.full_like(x, left_state[1]), torch.full_like(x, right_state[1]))
    pressure = torch.where(mask, torch.full_like(x, left_state[2]), torch.full_like(x, right_state[2]))
    average = primitive_to_conservative(rho, velocity, pressure)
    coeff = torch.zeros((ncells, degree + 1, 3), device=dev, dtype=dtype)
    coeff[:, 0] = average
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        speeds = torch.abs(values[..., 1] / values[..., 0].clamp_min(1e-12)) + sound_speed(values)
        max_speed = float(torch.max(speeds).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right)
        u1 = _positivity_filter(coeff + dt * k1, check_points)
        k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right)
        u2 = _positivity_filter(0.75 * coeff + 0.25 * (u1 + dt * k2), check_points)
        k3 = _rhs(u2, dx, weights, phi, dphi, phi_left, phi_right)
        coeff = _positivity_filter((1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3), check_points)
        average = coeff[:, 0]
        rho_now = average[:, 0]
        pressure_now = (GAMMA - 1.0) * (average[:, 2] - 0.5 * average[:, 1] ** 2 / rho_now.clamp_min(1e-12))
        if not torch.isfinite(coeff).all() or (rho_now <= 0).any() or (pressure_now <= 0).any():
            raise FloatingPointError("Nonphysical high-order Euler state; reduce CFL")
        time += dt
        steps += 1
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time, "steps": steps,
        "dt": dt, "diaphragm": diaphragm, "domain": (lower, upper), "cfl": cfl,
        "left_state": tuple(left_state), "right_state": tuple(right_state),
        "flux": "hllc", "limiter": "local positivity fallback", "device": str(dev),
    }
    return x, coeff, metadata
