"""Smooth periodic inviscid Burgers benchmark with high-order modal DG.

The documented case remains smooth (before shock formation), allowing the
nonlinear flux and design-order accuracy to be verified without a limiter.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MEAN = 0.2
AMPLITUDE = 0.8


def initial_profile(x: torch.Tensor) -> torch.Tensor:
    return MEAN + AMPLITUDE * torch.sin(2.0 * math.pi * x)


def exact_profile(x: torch.Tensor, final_time: float) -> torch.Tensor:
    """Solve u=u0(x-u*t) by Newton iteration in the smooth regime."""
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if final_time == 0.0:
        return initial_profile(x)
    u = initial_profile(x)
    for _ in range(40):
        argument = 2.0 * math.pi * (x - final_time * u)
        residual = u - MEAN - AMPLITUDE * torch.sin(argument)
        derivative = 1.0 + 2.0 * math.pi * final_time * AMPLITUDE * torch.cos(argument)
        updated = u - residual / derivative
        if float(torch.max(torch.abs(updated - u)).item()) < 1e-14:
            u = updated
            break
        u = updated
    else:
        raise RuntimeError("Burgers characteristic iteration did not converge; use a pre-shock time")
    return u


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor(
        [(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype,
    )
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _rhs(
    u: torch.Tensor,
    h: float,
    weights: torch.Tensor,
    phi: torch.Tensor,
    dphi: torch.Tensor,
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
) -> torch.Tensor:
    """Nonlinear weak DG operator with a Rusanov interface flux."""
    at_quadrature = u @ phi.T
    physical_flux = 0.5 * at_quadrature ** 2
    volume = torch.einsum("cq,q,qk->ck", physical_flux, weights, dphi)

    right_trace = u @ phi_right
    left_trace = u @ phi_left
    neighbor_left = torch.roll(left_trace, shifts=-1, dims=0)
    alpha = torch.maximum(torch.abs(right_trace), torch.abs(neighbor_left))
    flux_right = (
        0.25 * (right_trace ** 2 + neighbor_left ** 2)
        - 0.5 * alpha * (neighbor_left - right_trace)
    )
    flux_left = torch.roll(flux_right, shifts=1, dims=0)
    out = torch.empty_like(u)
    for k in range(u.shape[1]):
        out[:, k] = ((2 * k + 1) / h) * (
            volume[:, k] - flux_right * phi_right[k] + flux_left * phi_left[k]
        )
    return out


def solve_burgers(
    ncells: int = 64,
    degree: int = 2,
    final_time: float = 0.1,
    cfl: float = 0.1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve u_t+(u^2/2)_x=0 with a smooth periodic sine initial profile."""
    if ncells < 2 or degree not in (0, 1, 2, 3):
        raise ValueError("ncells must be >= 2 and degree must be in [0, 3]")
    if final_time < 0.0 or cfl <= 0.0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    xq = centers[:, None] + 0.5 * h * nodes[None, :]
    initial = initial_profile(xq)
    coeff = torch.empty((ncells, degree + 1), device=dev, dtype=dtype)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            weights[None, :] * initial * phi[:, k][None, :], dim=1,
        )

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quadrature = coeff @ phi.T
        max_speed = max(float(torch.max(torch.abs(at_quadrature)).item()), 1e-12)
        dt = min(cfl * h / ((2 * degree + 1) * max_speed), final_time - time)
        k1 = _rhs(coeff, h, weights, phi, dphi, phi_left, phi_right)
        u1 = coeff + dt * k1
        k2 = _rhs(u1, h, weights, phi, dphi, phi_left, phi_right)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        k3 = _rhs(u2, h, weights, phi, dphi, phi_left, phi_right)
        coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        if not torch.isfinite(coeff).all():
            raise FloatingPointError("Non-finite Burgers state; reduce CFL or use a pre-shock time")
        time += dt
        steps += 1
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "cfl": cfl, "device": str(dev),
    }
    return centers, coeff, metadata


def burgers_error(
    centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
) -> dict[str, float]:
    """Return absolute and relative L2 errors against the characteristic solution."""
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=coeff.device, dtype=coeff.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = exact_profile(x, final_time)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm}
