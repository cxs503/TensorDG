"""1-D periodic linear-advection DG benchmark using PyTorch tensors.

Modal Legendre basis, upwind numerical flux, and classical RK4 time stepping.
The spatial operator and solution state are evaluated with torch; NumPy is used
only to generate fixed Gauss-Legendre quadrature nodes during setup.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    # Exact for all polynomial products needed by the selected DG degree.
    nq = max(2 * degree + 3, 4)
    x_np, w_np = np.polynomial.legendre.leggauss(nq)
    x = torch.as_tensor(x_np, device=device, dtype=dtype)
    w = torch.as_tensor(w_np, device=device, dtype=dtype)
    # Legendre basis P_0 ... P_p and derivatives with respect to reference xi.
    vals, derivs = [], []
    for k in range(degree + 1):
        coeff = np.zeros(k + 1); coeff[-1] = 1.0
        dcoeff = np.polynomial.legendre.legder(coeff)
        vals.append(torch.as_tensor(np.polynomial.legendre.legval(x_np, coeff), device=device, dtype=dtype))
        derivs.append(torch.as_tensor(np.polynomial.legendre.legval(x_np, dcoeff) if len(dcoeff) else np.zeros_like(x_np), device=device, dtype=dtype))
    phi = torch.stack(vals, dim=1)       # [quadrature, modes]
    dphi = torch.stack(derivs, dim=1)
    phi_l = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_r = torch.ones(degree + 1, device=device, dtype=dtype)
    # V[i,j] = integral P_j(xi) * d(P_i)/dxi dxi
    volume = torch.einsum('q,qj,qi->ij', w, phi, dphi)
    return x, w, phi, volume, phi_l, phi_r


def _rhs(u: torch.Tensor, h: float, speed: float, volume: torch.Tensor,
         phi_l: torch.Tensor, phi_r: torch.Tensor) -> torch.Tensor:
    # u[cell, mode]. Positive and negative speeds use the corresponding upwind trace.
    right_trace = u @ phi_r
    left_trace = u @ phi_l
    if speed >= 0.0:
        flux_l = torch.roll(right_trace, shifts=1, dims=0)
        flux_r = right_trace
    else:
        flux_l = left_trace
        flux_r = torch.roll(left_trace, shifts=-1, dims=0)
    out = torch.empty_like(u)
    for k in range(u.shape[1]):
        out[:, k] = ((2 * k + 1) / h) * speed * (
            u @ volume[k, :] - flux_r * phi_r[k] + flux_l * phi_l[k]
        )
    return out


def solve_advection(ncells: int = 40, degree: int = 1, final_time: float = 1.0,
                    speed: float = 1.0, cfl: float = 0.08,
                    device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Solve u_t + speed*u_x=0 on [0,1] with periodic BC and u0=sin(2*pi*x).

    Returns (cell_centers, modal_coefficients, metadata). The first modal
    coefficient is the cell-average because P0=1 and higher Legendre modes
    have zero mean.
    """
    if ncells < 2 or degree not in (0, 1, 2, 3):
        raise ValueError("ncells must be >= 2 and degree must be in [0, 3]")
    if final_time < 0 or cfl <= 0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    xq, wq, phi, volume, phi_l, phi_r = _operators(degree, dev, dtype)
    # L2 projection of the initial condition into each element's modal space.
    x_phys = centers[:, None] + 0.5 * h * xq[None, :]
    u0 = torch.sin(2.0 * math.pi * x_phys)
    coeff = torch.empty((ncells, degree + 1), device=dev, dtype=dtype)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            wq[None, :] * u0 * phi[:, k][None, :], dim=1
        )
    if final_time == 0 or speed == 0:
        dt_nominal = float("inf")
        steps = 0
    else:
        dt_nominal = cfl * h / (abs(speed) * (2 * degree + 1))
        steps = math.ceil(final_time / dt_nominal)
    dt = final_time / steps if steps else 0.0
    for _ in range(steps):
        k1 = _rhs(coeff, h, speed, volume, phi_l, phi_r)
        k2 = _rhs(coeff + 0.5 * dt * k1, h, speed, volume, phi_l, phi_r)
        k3 = _rhs(coeff + 0.5 * dt * k2, h, speed, volume, phi_l, phi_r)
        k4 = _rhs(coeff + dt * k3, h, speed, volume, phi_l, phi_r)
        coeff = coeff + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    meta = {"ncells": ncells, "degree": degree, "final_time": final_time,
            "speed": speed, "dt": dt, "steps": steps, "device": str(dev)}
    return centers, coeff, meta


def l2_error(centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
             speed: float = 1.0) -> float:
    """Compute quadrature-based L2 error against the exact translated sine wave."""
    degree = coeff.shape[1] - 1
    x_np, w_np = np.polynomial.legendre.leggauss(max(2 * degree + 4, 6))
    xq = torch.as_tensor(x_np, device=coeff.device, dtype=coeff.dtype)
    wq = torch.as_tensor(w_np, device=coeff.device, dtype=coeff.dtype)
    h = 1.0 / centers.numel()
    basis = []
    for k in range(degree + 1):
        c = np.zeros(k + 1); c[-1] = 1.0
        basis.append(torch.as_tensor(np.polynomial.legendre.legval(x_np, c), device=coeff.device, dtype=coeff.dtype))
    phi = torch.stack(basis, dim=1)
    uh = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * xq[None, :]
    exact = torch.sin(2.0 * math.pi * (x - speed * final_time))
    err2 = 0.5 * h * torch.sum(wq[None, :] * (uh - exact) ** 2)
    return float(torch.sqrt(err2).item())
