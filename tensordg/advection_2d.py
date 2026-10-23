"""Two-dimensional periodic linear advection with tensor-product DG.

The discretization uses modal Legendre polynomials on a uniform Cartesian
mesh, an upwind numerical flux, Gauss--Legendre projection, and classical RK4.
The exact solution is a periodically translated smooth trigonometric profile.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    """Return one-dimensional reference operators reused in both directions."""
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
        derivative_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(
            derivative_values, device=device, dtype=dtype,
        ))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    # V[k,i] = integral P_i(xi) d(P_k)/dxi dxi.
    volume = torch.einsum("q,qi,qk->ki", weights, phi, dphi)
    phi_left = torch.tensor(
        [(-1.0) ** k for k in range(degree + 1)],
        device=device, dtype=dtype,
    )
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, volume, phi_left, phi_right


def _rhs(
    u: torch.Tensor,
    hx: float,
    hy: float,
    velocity: tuple[float, float],
    volume: torch.Tensor,
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the semi-discrete DG operator for ``u[cell_x,cell_y,k,l]``."""
    ax, ay = velocity

    # x-directed volume and upwind face terms. Y modes are spectators.
    volume_x = torch.einsum("xyil,ki->xykl", u, volume)
    right_x = torch.einsum("xyil,i->xyl", u, phi_right)
    left_x = torch.einsum("xyil,i->xyl", u, phi_left)
    if ax >= 0.0:
        flux_left_x = torch.roll(right_x, shifts=1, dims=0)
        flux_right_x = right_x
    else:
        flux_left_x = left_x
        flux_right_x = torch.roll(left_x, shifts=-1, dims=0)

    # y-directed volume and upwind face terms. X modes are spectators.
    volume_y = torch.einsum("xykj,lj->xykl", u, volume)
    top_y = torch.einsum("xykj,j->xyk", u, phi_right)
    bottom_y = torch.einsum("xykj,j->xyk", u, phi_left)
    if ay >= 0.0:
        # The upwind state is the upper trace of the cell below each face.
        flux_bottom_y = torch.roll(top_y, shifts=1, dims=1)
        flux_top_y = top_y
    else:
        flux_bottom_y = bottom_y
        flux_top_y = torch.roll(bottom_y, shifts=-1, dims=1)

    out = torch.zeros_like(u)
    for k in range(u.shape[2]):
        out[:, :, k, :] += ((2 * k + 1) * ax / hx) * (
            volume_x[:, :, k, :]
            - flux_right_x * phi_right[k]
            + flux_left_x * phi_left[k]
        )
    for l in range(u.shape[3]):
        out[:, :, :, l] += ((2 * l + 1) * ay / hy) * (
            volume_y[:, :, :, l]
            - flux_top_y * phi_right[l]
            + flux_bottom_y * phi_left[l]
        )
    return out


def solve_advection_2d(
    ncells: int | tuple[int, int] = 24,
    degree: int = 1,
    final_time: float = 1.0,
    velocity: tuple[float, float] = (1.0, 0.5),
    cfl: float = 0.08,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve ``u_t + ax*u_x + ay*u_y = 0`` on the periodic unit square.

    The initial/exact profile is ``1 + sin(2*pi*x) * sin(2*pi*y)`` translated
    by ``velocity * final_time``. ``ncells`` can be one integer or ``(nx,ny)``.

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)``. The
    coefficient array has shape ``(nx,ny,degree+1,degree+1)``; its ``[0,0]``
    mode is the cell average.
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if nx < 2 or ny < 2 or degree not in (0, 1, 2, 3):
        raise ValueError("ncells must be >= 2 and degree must be in [0, 3]")
    if final_time < 0.0 or cfl <= 0.0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    if len(velocity) != 2:
        raise ValueError("velocity must contain exactly two components")

    dev = torch.device(device)
    hx, hy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy
    nodes, weights, phi, volume, phi_left, phi_right = _operators(
        degree, dev, dtype,
    )

    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    initial = 1.0 + torch.sin(2.0 * math.pi * xq) * torch.sin(2.0 * math.pi * yq)
    coeff = torch.einsum(
        "xyqr,q,r,qk,rl->xykl", initial, weights, weights, phi, phi,
    )
    norm = torch.tensor(
        [(2 * k + 1) * (2 * l + 1) / 4.0
         for k in range(degree + 1) for l in range(degree + 1)],
        device=dev, dtype=dtype,
    ).reshape(degree + 1, degree + 1)
    coeff = coeff * norm[None, None, :, :]

    ax, ay = float(velocity[0]), float(velocity[1])
    wave_rate = abs(ax) / hx + abs(ay) / hy
    if final_time == 0.0 or wave_rate == 0.0:
        steps = 0
        dt_nominal = float("inf")
    else:
        dt_nominal = cfl / ((2 * degree + 1) * wave_rate)
        steps = math.ceil(final_time / dt_nominal)
    dt = final_time / steps if steps else 0.0

    for _ in range(steps):
        k1 = _rhs(coeff, hx, hy, (ax, ay), volume, phi_left, phi_right)
        k2 = _rhs(
            coeff + 0.5 * dt * k1, hx, hy, (ax, ay),
            volume, phi_left, phi_right,
        )
        k3 = _rhs(
            coeff + 0.5 * dt * k2, hx, hy, (ax, ay),
            volume, phi_left, phi_right,
        )
        k4 = _rhs(
            coeff + dt * k3, hx, hy, (ax, ay),
            volume, phi_left, phi_right,
        )
        coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    metadata = {
        "ncells": (nx, ny), "degree": degree, "final_time": final_time,
        "velocity": (ax, ay), "dt": dt,
        "steps": steps, "device": str(dev), "cfl": cfl,
    }
    return x, y, coeff, metadata


def error_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    coeff: torch.Tensor,
    final_time: float,
    velocity: tuple[float, float] = (1.0, 0.5),
) -> dict[str, float]:
    """Return quadrature-based absolute/relative L2 errors and exact norm."""
    degree = coeff.shape[-1] - 1
    nx, ny = coeff.shape[:2]
    hx, hy = 1.0 / nx, 1.0 / ny
    nodes_np, weights_np = np.polynomial.legendre.leggauss(
        max(2 * degree + 4, 6),
    )
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
    numerical = torch.einsum("xykl,qk,rl->xyqr", coeff, phi, phi)
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    ax, ay = velocity
    exact = 1.0 + torch.sin(
        2.0 * math.pi * (xq - ax * final_time),
    ) * torch.sin(2.0 * math.pi * (yq - ay * final_time))
    error2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * (numerical - exact) ** 2,
    )
    norm2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * exact ** 2,
    )
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
    }
