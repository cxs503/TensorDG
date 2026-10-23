"""Two-dimensional Euler isentropic-vortex benchmark on a periodic Cartesian mesh.

The current spatial baseline is P0 DG (the conservative finite-volume limit)
with Rusanov fluxes and SSP-RK3. The smooth analytical vortex makes it useful
for verifying two-dimensional fluxes, conservation, and mesh convergence.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .euler_1d import GAMMA


def vortex_primitive(x, y, time=0.0, strength=5.0, background_velocity=(1.0, 1.0)):
    """Analytical isentropic vortex with unit far-field density and pressure."""
    x_shift = x - background_velocity[0] * time
    y_shift = y - background_velocity[1] * time
    radius2 = x_shift ** 2 + y_shift ** 2
    exponential = torch.exp(0.5 * (1.0 - radius2))
    temperature = 1.0 - (GAMMA - 1.0) * strength ** 2 / (
        8.0 * GAMMA * math.pi ** 2
    ) * exponential ** 2
    rho = temperature ** (1.0 / (GAMMA - 1.0))
    velocity_x = background_velocity[0] - strength / (2.0 * math.pi) * exponential * y_shift
    velocity_y = background_velocity[1] + strength / (2.0 * math.pi) * exponential * x_shift
    pressure = rho ** GAMMA
    return rho, velocity_x, velocity_y, pressure


def _conservative(rho, velocity_x, velocity_y, pressure):
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (velocity_x ** 2 + velocity_y ** 2)
    return torch.stack((rho, rho * velocity_x, rho * velocity_y, energy), dim=-1)


def _primitive(u):
    rho = u[..., 0]
    velocity_x = u[..., 1] / rho
    velocity_y = u[..., 2] / rho
    pressure = (GAMMA - 1.0) * (
        u[..., 3] - 0.5 * (u[..., 1] ** 2 + u[..., 2] ** 2) / rho
    )
    return rho, velocity_x, velocity_y, pressure


def _sound_speed(u):
    rho = u[..., 0].clamp_min(1e-12)
    _, _, _, pressure = _primitive(u)
    return torch.sqrt((GAMMA * pressure.clamp_min(1e-12) / rho).clamp_min(0.0))


def _flux_x(u):
    rho, velocity_x, velocity_y, pressure = _primitive(u)
    return torch.stack((
        u[..., 1], u[..., 1] * velocity_x + pressure,
        u[..., 1] * velocity_y, velocity_x * (u[..., 3] + pressure),
    ), dim=-1)


def _flux_y(u):
    rho, velocity_x, velocity_y, pressure = _primitive(u)
    return torch.stack((
        u[..., 2], u[..., 2] * velocity_x,
        u[..., 2] * velocity_y + pressure, velocity_y * (u[..., 3] + pressure),
    ), dim=-1)


def _interface_flux(left, right, direction):
    flux_left = _flux_x(left) if direction == "x" else _flux_y(left)
    flux_right = _flux_x(right) if direction == "x" else _flux_y(right)
    component = 1 if direction == "x" else 2
    speed_left = torch.abs(left[..., component] / left[..., 0].clamp_min(1e-12)) + _sound_speed(left)
    speed_right = torch.abs(right[..., component] / right[..., 0].clamp_min(1e-12)) + _sound_speed(right)
    alpha = torch.maximum(speed_left, speed_right)
    return 0.5 * (flux_left + flux_right) - 0.5 * alpha[..., None] * (right - left)


def _rhs(u, dx, dy):
    right_x = torch.roll(u, shifts=-1, dims=0)
    right_y = torch.roll(u, shifts=-1, dims=1)
    flux_right_x = _interface_flux(u, right_x, "x")
    flux_right_y = _interface_flux(u, right_y, "y")
    flux_left_x = torch.roll(flux_right_x, shifts=1, dims=0)
    flux_left_y = torch.roll(flux_right_y, shifts=1, dims=1)
    return -(flux_right_x - flux_left_x) / dx - (flux_right_y - flux_left_y) / dy


def solve_isentropic_vortex(
    ncells=256,
    final_time=1.0,
    cfl=0.4,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    strength=5.0,
    background_velocity=(1.0, 1.0),
    device="cpu",
    dtype=torch.float64,
):
    """Solve the smooth Euler vortex with a periodic P0/Rusanov baseline."""
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if nx < 8 or ny < 8 or final_time < 0.0 or not 0 < cfl <= 0.5:
        raise ValueError("cells >= 8, final_time >= 0, and 0 < cfl <= 0.5 are required")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmax > xmin or not ymax > ymin:
        raise ValueError("each domain interval must be increasing")
    dev = torch.device(device)
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    # Four-point-per-direction Gauss integration gives accurate cell averages.
    nodes_np, weights_np = np.polynomial.legendre.leggauss(4)
    nodes = torch.as_tensor(nodes_np, device=dev, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=dev, dtype=dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    initial = _conservative(*vortex_primitive(
        xq, yq, 0.0, strength, background_velocity,
    ))
    u = 0.25 * torch.sum(
        initial * weights[None, None, :, None, None] * weights[None, None, None, :, None],
        dim=(2, 3),
    )

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        _, velocity_x, velocity_y, _ = _primitive(u)
        sound = _sound_speed(u)
        rate = float(torch.max(torch.abs(velocity_x) + sound).item()) / dx
        rate += float(torch.max(torch.abs(velocity_y) + sound).item()) / dy
        dt = min(cfl / max(rate, 1e-12), final_time - time)
        u1 = u + dt * _rhs(u, dx, dy)
        u2 = 0.75 * u + 0.25 * (u1 + dt * _rhs(u1, dx, dy))
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * _rhs(u2, dx, dy))
        rho, _, _, pressure = _primitive(u)
        if not torch.isfinite(u).all() or (rho <= 0).any() or (pressure <= 0).any():
            raise FloatingPointError("Nonphysical 2D Euler state; reduce CFL")
        time += dt
        steps += 1
    rho, velocity_x, velocity_y, pressure = _primitive(u)
    fields = {
        "density": rho, "velocity_x": velocity_x, "velocity_y": velocity_y,
        "pressure": pressure, "conservative": u,
    }
    metadata = {
        "ncells": (nx, ny), "degree": 0, "final_time": final_time,
        "steps": steps, "dt": dt, "cfl": cfl, "domain": domain,
        "strength": strength, "background_velocity": tuple(background_velocity),
        "flux": "rusanov", "device": str(dev),
    }
    return x, y, fields, metadata


def vortex_error(x, y, conservative, final_time, strength=5.0, background_velocity=(1.0, 1.0)):
    """Compare P0 states with four-point-per-direction exact cell averages."""
    nx, ny = conservative.shape[:2]
    dx = float((x[-1] - x[0]).item()) / (nx - 1)
    dy = float((y[-1] - y[0]).item()) / (ny - 1)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(4)
    nodes = torch.as_tensor(nodes_np, device=conservative.device, dtype=conservative.dtype)
    weights = torch.as_tensor(weights_np, device=conservative.device, dtype=conservative.dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact_q = vortex_primitive(xq, yq, final_time, strength, background_velocity)
    reference = tuple(
        0.25 * torch.sum(
            field * weights[None, None, :, None] * weights[None, None, None, :],
            dim=(2, 3),
        ) for field in exact_q
    )
    numerical = _primitive(conservative)
    result = {}
    for key, computed, exact in zip(
        ("density", "velocity_x", "velocity_y", "pressure"), numerical, reference,
    ):
        absolute = float(torch.sqrt(torch.mean((computed - exact) ** 2)).item())
        norm = float(torch.sqrt(torch.mean(exact ** 2)).item())
        result[key] = {
            "absolute_l2": absolute, "relative_l2": absolute / norm,
            "relative_l2_percent": 100.0 * absolute / norm,
        }
    return result
