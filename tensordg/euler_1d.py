"""One-dimensional Euler P0 DG / finite-volume Riemann benchmarks."""
from __future__ import annotations

import torch

GAMMA = 1.4


def primitive_to_conservative(rho, velocity, pressure):
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity ** 2
    return torch.stack((rho, rho * velocity, energy), dim=-1)


def primitive_fields(u):
    rho = u[..., 0]
    velocity = u[..., 1] / rho
    pressure = (GAMMA - 1.0) * (u[..., 2] - 0.5 * u[..., 1] ** 2 / rho)
    return {"density": rho, "velocity": velocity, "pressure": pressure}


def physical_flux(u):
    rho = u[..., 0].clamp_min(1e-12)
    velocity = u[..., 1] / rho
    pressure = (GAMMA - 1.0) * (u[..., 2] - 0.5 * u[..., 1] ** 2 / rho)
    return torch.stack((
        u[..., 1], u[..., 1] * velocity + pressure,
        velocity * (u[..., 2] + pressure),
    ), dim=-1)


def sound_speed(u):
    rho = u[..., 0].clamp_min(1e-12)
    pressure = (GAMMA - 1.0) * (u[..., 2] - 0.5 * u[..., 1] ** 2 / rho)
    return torch.sqrt((GAMMA * pressure.clamp_min(1e-12) / rho).clamp_min(0.0))


def rusanov_rhs(u, dx):
    """P0 semi-discrete operator with transmissive exterior states."""
    left = torch.cat((u[:1], u[:-1]), dim=0)
    right = torch.cat((u[1:], u[-1:]), dim=0)
    center_flux = physical_flux(u)
    flux_left_states, flux_right_states = physical_flux(left), physical_flux(right)
    speed_left = torch.abs(left[:, 1] / left[:, 0].clamp_min(1e-12)) + sound_speed(left)
    speed_center = torch.abs(u[:, 1] / u[:, 0].clamp_min(1e-12)) + sound_speed(u)
    speed_right = torch.abs(right[:, 1] / right[:, 0].clamp_min(1e-12)) + sound_speed(right)
    alpha_left = torch.maximum(speed_left, speed_center)
    alpha_right = torch.maximum(speed_center, speed_right)
    flux_left = 0.5 * (flux_left_states + center_flux) - 0.5 * alpha_left[:, None] * (u - left)
    flux_right = 0.5 * (center_flux + flux_right_states) - 0.5 * alpha_right[:, None] * (right - u)
    flux_left[0] = center_flux[0]
    flux_right[-1] = center_flux[-1]
    return -(flux_right - flux_left) / dx


def hllc_flux(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Three-wave HLLC flux, including a less dissipative contact wave."""
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    velocity_l = left[:, 1] / rho_l
    velocity_r = right[:, 1] / rho_r
    pressure_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    pressure_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    sound_l = torch.sqrt(GAMMA * pressure_l.clamp_min(1e-12) / rho_l)
    sound_r = torch.sqrt(GAMMA * pressure_r.clamp_min(1e-12) / rho_r)
    speed_l = torch.minimum(velocity_l - sound_l, velocity_r - sound_r)
    speed_r = torch.maximum(velocity_l + sound_l, velocity_r + sound_r)
    denominator = rho_l * (speed_l - velocity_l) - rho_r * (speed_r - velocity_r)
    speed_middle = (
        pressure_r - pressure_l
        + rho_l * velocity_l * (speed_l - velocity_l)
        - rho_r * velocity_r * (speed_r - velocity_r)
    ) / denominator

    def star_state(state, rho, velocity, pressure, speed):
        factor = speed - velocity
        rho_star = rho * factor / (speed - speed_middle)
        energy_star = rho_star * (
            state[:, 2] / rho
            + (speed_middle - velocity)
            * (speed_middle + pressure / (rho * factor))
        )
        return torch.stack((rho_star, rho_star * speed_middle, energy_star), dim=1)

    star_l = star_state(left, rho_l, velocity_l, pressure_l, speed_l)
    star_r = star_state(right, rho_r, velocity_r, pressure_r, speed_r)
    flux_l = physical_flux(left)
    flux_r = physical_flux(right)
    star_flux_l = flux_l + speed_l[:, None] * (star_l - left)
    star_flux_r = flux_r + speed_r[:, None] * (star_r - right)
    flux = torch.where((speed_l >= 0.0)[:, None], flux_l, star_flux_l)
    flux = torch.where((speed_middle < 0.0)[:, None], star_flux_r, flux)
    flux = torch.where((speed_r <= 0.0)[:, None], flux_r, flux)
    return flux


def hllc_rhs(u, dx):
    right = torch.cat((u[1:], u[-1:]), dim=0)
    # Interface i+1/2 has cell i on its left and cell i+1 on its right.
    right_flux = hllc_flux(u, right)
    left_flux = torch.roll(right_flux, shifts=1, dims=0)
    # Transmissive boundaries have equal exterior/interior states.
    right_flux[-1] = physical_flux(u[-1:])
    left_flux[0] = physical_flux(u[:1])
    return -(right_flux - left_flux) / dx


def solve_riemann(
    left_state: tuple[float, float, float],
    right_state: tuple[float, float, float],
    ncells: int = 1600,
    final_time: float = 0.2,
    diaphragm: float = 0.5,
    domain: tuple[float, float] = (0.0, 1.0),
    cfl: float = 0.4,
    flux: str = "hllc",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve a two-state Euler Riemann problem with P0/Rusanov or HLLC/SSP-RK3."""
    if ncells < 20 or final_time < 0.0 or not 0.0 < cfl <= 0.5:
        raise ValueError("ncells >= 20, final_time >= 0, and 0 < cfl <= 0.5 are required")
    if flux not in ("hllc", "rusanov"):
        raise ValueError("flux must be 'hllc' or 'rusanov'")
    rhs = hllc_rhs if flux == "hllc" else rusanov_rhs
    lower, upper = domain
    if not upper > lower or not lower < diaphragm < upper:
        raise ValueError("domain must be increasing and contain the diaphragm")
    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    left_mask = x < diaphragm
    rho = torch.where(left_mask, torch.full_like(x, left_state[0]), torch.full_like(x, right_state[0]))
    velocity = torch.where(left_mask, torch.full_like(x, left_state[1]), torch.full_like(x, right_state[1]))
    pressure = torch.where(left_mask, torch.full_like(x, left_state[2]), torch.full_like(x, right_state[2]))
    u = primitive_to_conservative(rho, velocity, pressure)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        max_speed = float((torch.abs(u[:, 1] / u[:, 0].clamp_min(1e-12)) + sound_speed(u)).max().item())
        dt = min(cfl * dx / max(max_speed, 1e-12), final_time - time)
        u1 = u + dt * rhs(u, dx)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, dx))
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2, dx))
        rho_now = u[:, 0]
        pressure_now = (GAMMA - 1.0) * (
            u[:, 2] - 0.5 * u[:, 1] ** 2 / rho_now.clamp_min(1e-12)
        )
        if not torch.isfinite(u).all() or (rho_now <= 0).any() or (pressure_now <= 0).any():
            raise FloatingPointError("Nonphysical Euler state; reduce CFL or refine the mesh")
        time += dt
        steps += 1
    fields = primitive_fields(u)
    fields["conservative"] = u
    metadata = {
        "ncells": ncells, "final_time": final_time, "steps": steps, "dt": dt,
        "diaphragm": diaphragm, "domain": (lower, upper), "cfl": cfl,
        "left_state": tuple(left_state), "right_state": tuple(right_state),
        "flux": flux, "device": str(dev),
    }
    return x, fields, metadata
