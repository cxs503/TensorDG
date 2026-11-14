"""Two-dimensional Euler on Cartesian meshes with a P0 / Rusanov / SSP-RK3 baseline.

The spatial baseline is P0 DG (the conservative finite-volume limit) with
Rusanov fluxes and SSP-RK3.  Two public entry points are provided:

* :func:`solve_euler_2d` -- a generic driver for an arbitrary initial
  condition ``initial(x, y) -> (rho, u, v, p)`` on a Cartesian mesh, with
  per-axis periodicity control.  Non-periodic faces use the zero-gradient
  (outflow / transmissive) closure, i.e. the boundary face state is the
  adjacent interior cell, so the physical flux passes through the face
  unchanged and the scheme stays conservative.
* :func:`solve_isentropic_vortex` -- the smooth analytical vortex benchmark on
  a periodic mesh, which makes the module useful for verifying two-dimensional
  fluxes, conservation, and mesh convergence.

The boundary closure is what allows discontinuous, non-periodic problems
(e.g. a shock tube extruded to 2-D) to be run through this path at all; the
verification harness ``benchmarks/verify_euler_2d_p0.py`` pins those cases.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .stepping import step_metadata
from .euler_1d import GAMMA

__all__ = [
    "vortex_primitive",
    "solve_euler_2d",
    "solve_isentropic_vortex",
    "vortex_error",
]


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
    return torch.stack((
        rho,
        rho * velocity_x,
        rho * velocity_y,
        pressure / (GAMMA - 1.0)
        + 0.5 * rho * (velocity_x ** 2 + velocity_y ** 2),
    ), dim=-1)


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


def _pad(u, axis, periodic):
    """Add one ghost layer along ``axis``: wrap-around or zero-gradient."""
    first = u.select(axis, 0).unsqueeze(axis)
    last = u.select(axis, -1).unsqueeze(axis)
    if periodic:
        return torch.cat((last, u, first), dim=axis)
    return torch.cat((first, u, last), dim=axis)


def _face_fluxes(u, direction, periodic):
    """Rusanov fluxes on every face normal to ``direction`` (``n + 1`` faces)."""
    axis = 0 if direction == "x" else 1
    padded = _pad(u, axis, periodic)
    left = padded.narrow(axis, 0, padded.shape[axis] - 1)
    right = padded.narrow(axis, 1, padded.shape[axis] - 1)
    return _interface_flux(left, right, direction)


def _rhs(u, dx, dy, periodic=(True, True)):
    flux_x = _face_fluxes(u, "x", bool(periodic[0]))
    flux_y = _face_fluxes(u, "y", bool(periodic[1]))
    right_x = flux_x.narrow(0, 1, flux_x.shape[0] - 1)
    left_x = flux_x.narrow(0, 0, flux_x.shape[0] - 1)
    right_y = flux_y.narrow(1, 1, flux_y.shape[1] - 1)
    left_y = flux_y.narrow(1, 0, flux_y.shape[1] - 1)
    return -(right_x - left_x) / dx - (right_y - left_y) / dy


def _cell_averages(initial, x, y, dx, dy, quadrature, dev, dtype):
    """Gauss quadrature cell averages of ``initial(x, y)``."""
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    nodes = torch.as_tensor(nodes_np, device=dev, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=dev, dtype=dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    sampled = _conservative(*initial(xq, yq))
    return 0.25 * torch.sum(
        sampled * weights[None, None, :, None, None] * weights[None, None, None, :, None],
        dim=(2, 3),
    )


def solve_euler_2d(
    initial,
    ncells=256,
    final_time=1.0,
    cfl=0.4,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    periodic=(True, True),
    quadrature=4,
    device="cpu",
    dtype=torch.float64,
    observer=None,
):
    """Advance a 2-D Euler initial condition (P0 finite volume, Rusanov, SSP-RK3).

    Parameters
    ----------
    initial : callable
        ``initial(x, y) -> (rho, u, v, p)`` evaluated on tensors of any shape.
    ncells : int or (int, int)
        Cell counts in ``x`` and ``y``.
    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``; non-periodic faces are transmissive
        (zero-gradient) outflow, where the boundary face flux is the physical
        flux of the adjacent interior cell.
    quadrature : int
        Gauss-Legendre points per direction for the cell-average projection of
        ``initial`` (2 is exact for the P0 average of piecewise-smooth data).
    observer : callable or None
        Verification hook called as ``observer(time, steps, u)`` with the
        conservative state ``u`` before the first time step (``time = 0``,
        ``steps = 0``) and after every accepted step.  It exists so a
        conservation audit can sample the discrete invariants *once per step*
        (per-step statistics, not just an end-state difference) without
        re-implementing the RK stages around the solver.  ``u`` is the live
        tensor and must not be modified; the callback's return value is
        ignored.

    Returns
    -------
    (x, y, fields, metadata)
        Cell-centre coordinates, primitive and conservative fields, and run
        metadata (step count, ``dt``, mesh, periodicity, flux).
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if nx < 2 or ny < 2 or final_time < 0.0 or not 0 < cfl <= 0.5:
        raise ValueError("cells >= 2, final_time >= 0, and 0 < cfl <= 0.5 are required")
    if quadrature < 1:
        raise ValueError("quadrature must be >= 1")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmax > xmin or not ymax > ymin:
        raise ValueError("each domain interval must be increasing")
    periodic_flags = (bool(periodic[0]), bool(periodic[1]))
    dev = torch.device(device)
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy
    u = _cell_averages(initial, x, y, dx, dy, quadrature, dev, dtype)

    time, steps, dt = 0.0, 0, 0.0
    if observer is not None:
        observer(time, steps, u)
    while time < final_time:
        _, velocity_x, velocity_y, _ = _primitive(u)
        sound = _sound_speed(u)
        rate = float(torch.max(torch.abs(velocity_x) + sound).item()) / dx
        rate += float(torch.max(torch.abs(velocity_y) + sound).item()) / dy
        dt = min(cfl / max(rate, 1e-12), final_time - time)
        u1 = u + dt * _rhs(u, dx, dy, periodic_flags)
        u2 = 0.75 * u + 0.25 * (u1 + dt * _rhs(u1, dx, dy, periodic_flags))
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * _rhs(u2, dx, dy, periodic_flags))
        rho, _, _, pressure = _primitive(u)
        if not torch.isfinite(u).all() or (rho <= 0).any() or (pressure <= 0).any():
            raise FloatingPointError("Nonphysical 2D Euler state; reduce CFL")
        time += dt
        steps += 1
        if observer is not None:
            observer(time, steps, u)

    rho, velocity_x, velocity_y, pressure = _primitive(u)
    fields = {
        "density": rho, "velocity_x": velocity_x, "velocity_y": velocity_y,
        "pressure": pressure, "conservative": u,
    }
    metadata = {
        "ncells": (nx, ny), "degree": 0, "final_time": final_time,
        "steps": steps, **step_metadata(final_time, steps, dt),
        "cfl": cfl, "domain": domain,
        "periodic": periodic_flags, "quadrature": quadrature,
        "flux": "rusanov", "device": str(dev),
    }
    return x, y, fields, metadata


def solve_isentropic_vortex(
    ncells=256,
    final_time=1.0,
    cfl=0.4,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    strength=5.0,
    background_velocity=(1.0, 1.0),
    device="cpu",
    dtype=torch.float64,
    observer=None,
):
    """Solve the smooth Euler vortex with the periodic P0/Rusanov baseline.

    observer is forwarded to :func: for per-step
    conservation auditing.
    """
    if isinstance(ncells, int) and ncells < 8:
        raise ValueError("cells >= 8 are required")
    if not isinstance(ncells, int) and min(ncells) < 8:
        raise ValueError("cells >= 8 are required")
    def initial(x, y):
        return vortex_primitive(x, y, 0.0, strength, background_velocity)

    x, y, fields, metadata = solve_euler_2d(
        initial,
        ncells=ncells,
        final_time=final_time,
        cfl=cfl,
        domain=domain,
        periodic=(True, True),
        device=device,
        dtype=dtype,
        observer=observer,
    )
    metadata["strength"] = strength
    metadata["background_velocity"] = tuple(background_velocity)
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