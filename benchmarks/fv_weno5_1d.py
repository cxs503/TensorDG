"""Independent finite-volume WENO5 reference solver for 1-D Euler.

This module exists for one reason: the Shu--Osher shock/entropy-wave problem has
no closed-form solution, so *any* tabulated reference has to be trusted before it
can be used as a benchmark anchor.  Trust here is built from three independent
legs:

1. self-convergence of the reference solver (two successive refinements),
2. agreement with a second, structurally different discretisation family,
3. analytic Rankine--Hugoniot anchors for the transmitted shock.

The second leg is what this module provides.  It shares only the *physics*
(the HLLC flux of :mod:`tensordg.euler_1d`) with the discontinuous-Galerkin
path in :mod:`tensordg.euler_dg_1d`; the spatial discretisation is completely
different: cell-centred finite volumes with component-wise Jiang--Shu (1996)
WENO5 reconstruction, SSP-RK3 in time and transmissive ghost cells.  A DG
Bug would therefore not automatically be shared by this solver, which is
exactly what makes it useful as a cross-check.
"""
from __future__ import annotations

import time

import torch

from tensordg.euler_1d import GAMMA, hllc_flux, physical_flux, sound_speed

__all__ = ["weno5_reconstruct", "solve_fv_weno5", "GAMMA"]


def _weno5_left(a, b, c, d, e):
    """Jiang--Shu WENO5 left-biased reconstruction from five cell values.

    ``a .. e`` are the cell averages at ``i-2, i-1, i, i+1, i+2``; the returned
    value is the two-candidate polynomial blend evaluated at the ``i+1/2`` face.
    """
    eps = 1e-6
    beta0 = (13.0 / 12.0) * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    beta1 = (13.0 / 12.0) * (b - 2.0 * c + d) ** 2 + 0.25 * (b - d) ** 2
    beta2 = (13.0 / 12.0) * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2
    alpha0 = 0.1 / (eps + beta0) ** 2
    alpha1 = 0.6 / (eps + beta1) ** 2
    alpha2 = 0.3 / (eps + beta2) ** 2
    total = alpha0 + alpha1 + alpha2
    p0 = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0
    p1 = (-b + 5.0 * c + 2.0 * d) / 6.0
    p2 = (2.0 * c + 5.0 * d - e) / 6.0
    return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / total


def weno5_reconstruct(u: torch.Tensor):
    """Face states ``(u_left, u_right)`` of shape ``(ncells + 1, 3)``.

    Transmissive ends: three ghost cells per side copy the boundary cell.
    """
    ncells = u.shape[0]
    padded = torch.cat((u[:1].expand(3, -1), u, u[-1:].expand(3, -1)), dim=0)
    full = padded.shape[0]  # ncells + 6
    s = [padded[k:full - 5 + k] for k in range(6)]  # s[k] = padded[k : ncells + 1 + k]
    # face j sits between cell j-1 and cell j; the left-biased stencil is u[j-3..j+1]
    # (padded[j..j+4]) and the right-biased one its mirror image.
    u_left = _weno5_left(s[0], s[1], s[2], s[3], s[4])
    u_right = _weno5_left(s[5], s[4], s[3], s[2], s[1])
    return u_left, u_right


def _rhs(u: torch.Tensor, dx: float, wave_speed: str = "davis") -> torch.Tensor:
    u_left, u_right = weno5_reconstruct(u)
    flux = hllc_flux(u_left, u_right, wave_speed=wave_speed)
    flux = torch.cat((physical_flux(u[:1]), flux[1:-1], physical_flux(u[-1:])), dim=0)
    return -(flux[1:] - flux[:-1]) / dx


def _admissible(u: torch.Tensor) -> bool:
    rho = u[:, 0]
    if not torch.isfinite(u).all() or float(rho.min()) <= 0.0:
        return False
    pressure = (GAMMA - 1.0) * (u[:, 2] - 0.5 * u[:, 1] ** 2 / rho)
    return float(pressure.min()) > 0.0


def solve_fv_weno5(initial_state, ncells: int = 3200, final_time: float = 1.8,
                   domain=(-5.0, 5.0), cfl: float = 0.3, device: str = "cpu",
                   dtype=torch.float64, wave_speed: str = "davis"):
    """Advance ``initial_state`` to ``final_time`` with FV-WENO5/HLLC/SSP-RK3.

    ``initial_state(x)`` must accept a ``(ncells, 1)`` coordinate tensor and
    return ``(rho, velocity, pressure)`` of the same shape (the interface is the
    one used by :func:`tensordg.euler_dg_1d.solve_periodic_euler_dg_1d`, so the
    two solvers can be fed the same callable).
    """
    dev, dt_type = torch.device(device), dtype
    lower, upper = domain
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dt_type) + 0.5) * dx
    rho, velocity, pressure = (a.reshape(-1) for a in initial_state(x[:, None]))
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity ** 2
    u = torch.stack((rho, rho * velocity, energy), dim=-1)
    start, steps, t_now, dt = time.perf_counter(), 0, 0.0, 0.0
    while t_now < final_time:
        k1 = _rhs(u, dx, wave_speed)
        speed = float(torch.max(torch.abs(u[:, 1] / u[:, 0]) + sound_speed(u)).item())
        dt = min(cfl * dx / max(speed, 1e-12), final_time - t_now)
        u1 = u + dt * k1
        if not _admissible(u1):
            raise FloatingPointError(f"FV-WENO5 inadmissible state at step {steps}")
        u2 = 0.75 * u + 0.25 * (u1 + dt * _rhs(u1, dx, wave_speed))
        if not _admissible(u2):
            raise FloatingPointError(f"FV-WENO5 inadmissible state at step {steps}")
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * _rhs(u2, dx, wave_speed))
        if not _admissible(u):
            raise FloatingPointError(f"FV-WENO5 inadmissible state at step {steps}")
        t_now += dt
        steps += 1
    meta = {
        "scheme": "FV-WENO5-JS/HLLC/SSP-RK3", "ncells": ncells, "domain": [lower, upper],
        "final_time": final_time, "cfl": cfl, "steps": steps, "dt": float(dt),
        "wave_speed": wave_speed, "wall_time_s": time.perf_counter() - start,
        "density_min": float(u[:, 0].min()), "density_max": float(u[:, 0].max()),
    }
    return x.detach().cpu().numpy(), u.detach().cpu().numpy(), meta