"""One-dimensional Euler P0 DG / finite-volume Riemann benchmarks."""
from __future__ import annotations

import torch
from .stepping import step_metadata

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


def davis_wave_speeds(left: torch.Tensor, right: torch.Tensor):
    """Davis (1981) outer wave-speed estimates ``S_L``, ``S_R``.

    Uses the min/max of the acoustic characteristics of the two states.  Robust
    but over-diffusive: it over-estimates the outer speeds for strong shocks, so
    the HLLC flux departs from the exact Riemann flux (see
    ``docs/euler_flux_kernel_verification.md``).
    """
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    velocity_l = left[:, 1] / rho_l
    velocity_r = right[:, 1] / rho_r
    sound_l = sound_speed(left)
    sound_r = sound_speed(right)
    speed_l = torch.minimum(velocity_l - sound_l, velocity_r - sound_r)
    speed_r = torch.maximum(velocity_l + sound_l, velocity_r + sound_r)
    return speed_l, speed_r


def einfeldt_wave_speeds(left: torch.Tensor, right: torch.Tensor):
    """Einfeldt (1988) / Roe-average outer wave-speed estimates ``S_L``, ``S_R``.

    ``S_L = min(u_L - a_L, u~ - a~)``, ``S_R = max(u_R + a_R, u~ + a~)`` with the
    Roe averages ``u~``, ``H~``, ``a~``.  Unlike Davis' estimates these bound the
    star region tightly, so the three-wave HLLC flux reproduces the exact
    Riemann flux much more closely (Toro 2009, sec. 10.5-10.6).
    """
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    velocity_l = left[:, 1] / rho_l
    velocity_r = right[:, 1] / rho_r
    pressure_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    pressure_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    enthalpy_l = (left[:, 2] + pressure_l) / rho_l
    enthalpy_r = (right[:, 2] + pressure_r) / rho_r
    root_l = torch.sqrt(rho_l)
    root_r = torch.sqrt(rho_r)
    weight = 1.0 / (root_l + root_r)
    velocity_roe = (root_l * velocity_l + root_r * velocity_r) * weight
    enthalpy_roe = (root_l * enthalpy_l + root_r * enthalpy_r) * weight
    sound_roe = torch.sqrt(
        ((GAMMA - 1.0) * (enthalpy_roe - 0.5 * velocity_roe ** 2)).clamp_min(1e-12)
    )
    sound_l = sound_speed(left)
    sound_r = sound_speed(right)
    speed_l = torch.minimum(velocity_l - sound_l, velocity_roe - sound_roe)
    speed_r = torch.maximum(velocity_r + sound_r, velocity_roe + sound_roe)
    return speed_l, speed_r


def _two_rarefaction_pressure(rho_l, u_l, p_l, a_l, rho_r, u_r, p_r, a_r):
    """Toro (2009) two-rarefaction approximation of the star pressure ``p*``."""
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    numerator = a_l + a_r - 0.5 * (GAMMA - 1.0) * (u_r - u_l)
    denominator = a_l / p_l.clamp_min(1e-12) ** exponent + a_r / p_r.clamp_min(1e-12) ** exponent
    estimate = (numerator / denominator).clamp_min(1e-12) ** (1.0 / exponent)
    pvrs = (
        0.5 * (p_l + p_r)
        - 0.5 * (u_r - u_l) * 0.5 * (rho_l + rho_r) * 0.5 * (a_l + a_r)
    ).clamp_min(1e-12)
    invalid = torch.isnan(estimate) | (estimate <= 0.0) | ~torch.isfinite(estimate)
    return torch.where(invalid, pvrs, estimate), pvrs


def _pressure_function_and_derivative(p_star, p_k, a_k, rho_k):
    """Toro (2009) eq. (4.6): ``f_K(p*)`` and ``f'_K(p*)`` for one side."""
    gamma1 = 2.0 / ((GAMMA + 1.0) * rho_k)
    gamma2 = (GAMMA - 1.0) / (GAMMA + 1.0) * p_k
    shock = p_star > p_k
    ratio = (p_star - p_k) * torch.sqrt(gamma1 / (p_star + gamma2).clamp_min(1e-12))
    derivative_shock = torch.sqrt(gamma1 / (p_star + gamma2).clamp_min(1e-12)) * (
        1.0 - 0.5 * (p_star - p_k) / (p_star + gamma2).clamp_min(1e-12)
    )
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    rarefaction = 2.0 * a_k / (GAMMA - 1.0) * (
        (p_star / p_k.clamp_min(1e-12)).clamp_min(1e-12) ** exponent - 1.0
    )
    derivative_rarefaction = (
        1.0 / (rho_k * a_k.clamp_min(1e-12))
        * (p_star / p_k.clamp_min(1e-12)).clamp_min(1e-12) ** (-(GAMMA + 1.0) / (2.0 * GAMMA))
    )
    return (torch.where(shock, ratio, rarefaction),
            torch.where(shock, derivative_shock, derivative_rarefaction))


def star_pressure(left: torch.Tensor, right: torch.Tensor, iterations: int = 8):
    """Newton-refined ``p*`` of the two-state Riemann problem (validation grade).

    Starts from the two-rarefaction approximation and applies Toro's guarded
    Newton iteration on ``f_L(p*) + f_R(p*) + u_R - u_L = 0``.  Unlike TRRS this
    is accurate for very strong shocks (TRRS is +37% there, see
    ``docs/hllc_wave_speed_verification.md``), at the cost of ``iterations``
    extra vectorised passes.
    """
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    u_l = left[:, 1] / rho_l
    u_r = right[:, 1] / rho_r
    p_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    p_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    a_l = sound_speed(left)
    a_r = sound_speed(right)
    estimate, fallback = _two_rarefaction_pressure(rho_l, u_l, p_l, a_l, rho_r, u_r, p_r, a_r)
    pressure = estimate
    for _ in range(iterations):
        f_l, d_l = _pressure_function_and_derivative(pressure, p_l, a_l, rho_l)
        f_r, d_r = _pressure_function_and_derivative(pressure, p_r, a_r, rho_r)
        residual = f_l + f_r + u_r - u_l
        slope = d_l + d_r
        step = residual / torch.where(slope.abs() < 1e-14, torch.full_like(slope, 1e-14), slope)
        candidate = pressure - step
        # Guard: keep p* positive and damp oversized Newton steps.
        candidate = torch.where(candidate > 0.0, candidate, 0.5 * pressure)
        candidate = torch.where(torch.isfinite(candidate), candidate, fallback)
        pressure = candidate
    return torch.where(torch.isfinite(pressure) & (pressure > 0.0), pressure, fallback)


def _speeds_from_pressure(left, right, p_star):
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    u_l = left[:, 1] / rho_l
    u_r = right[:, 1] / rho_r
    p_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    p_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    a_l = sound_speed(left)
    a_r = sound_speed(right)

    def factor(p_k):
        ratio = (1.0 + (GAMMA + 1.0) / (2.0 * GAMMA)
                 * (p_star / p_k.clamp_min(1e-12) - 1.0)).clamp_min(1e-12)
        return torch.where(p_star > p_k, torch.sqrt(ratio), torch.ones_like(ratio))

    return u_l - a_l * factor(p_l), u_r + a_r * factor(p_r)


def toro_wave_speeds(left: torch.Tensor, right: torch.Tensor):
    """Pressure-based outer wave speeds from the two-rarefaction ``p*`` (no iteration)."""
    pressure = _two_rarefaction_pressure(
        left[:, 0].clamp_min(1e-12), left[:, 1] / left[:, 0].clamp_min(1e-12),
        (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / left[:, 0].clamp_min(1e-12)),
        sound_speed(left),
        right[:, 0].clamp_min(1e-12), right[:, 1] / right[:, 0].clamp_min(1e-12),
        (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / right[:, 0].clamp_min(1e-12)),
        sound_speed(right),
    )[0]
    return _speeds_from_pressure(left, right, pressure)


def toro_iterative_wave_speeds(left: torch.Tensor, right: torch.Tensor):
    """Pressure-based outer wave speeds from the Newton-refined ``p*``."""
    return _speeds_from_pressure(left, right, star_pressure(left, right))


WAVE_SPEEDS = {
    "davis": davis_wave_speeds,
    "einfeldt": einfeldt_wave_speeds,
    "toro": toro_wave_speeds,
    "toro_iter": toro_iterative_wave_speeds,
}


def hllc_flux(left: torch.Tensor, right: torch.Tensor, wave_speed: str = "davis") -> torch.Tensor:
    """Three-wave HLLC flux, including a less dissipative contact wave.

    ``wave_speed`` selects the outer wave-speed estimates: ``"davis"`` (default,
    historical behaviour), ``"einfeldt"`` (Roe averages), ``"toro"``
    (two-rarefaction ``p*``, no iteration) or ``"toro_iter"`` (Newton-refined
    ``p*``, validation grade).  See ``docs/hllc_wave_speed_verification.md``.
    """
    if wave_speed not in WAVE_SPEEDS:
        raise ValueError(f"wave_speed must be one of {sorted(WAVE_SPEEDS)}")
    rho_l = left[:, 0].clamp_min(1e-12)
    rho_r = right[:, 0].clamp_min(1e-12)
    velocity_l = left[:, 1] / rho_l
    velocity_r = right[:, 1] / rho_r
    pressure_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    pressure_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    speed_l, speed_r = WAVE_SPEEDS[wave_speed](left, right)
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


def hllc_rhs(u, dx, wave_speed: str = "davis"):
    right = torch.cat((u[1:], u[-1:]), dim=0)
    # Interface i+1/2 has cell i on its left and cell i+1 on its right.
    right_flux = hllc_flux(u, right, wave_speed=wave_speed)
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
    hllc_wave_speed: str = "davis",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve a two-state Euler Riemann problem with P0/Rusanov or HLLC/SSP-RK3."""
    if ncells < 20 or final_time < 0.0 or not 0.0 < cfl <= 0.5:
        raise ValueError("ncells >= 20, final_time >= 0, and 0 < cfl <= 0.5 are required")
    if flux not in ("hllc", "rusanov"):
        raise ValueError("flux must be 'hllc' or 'rusanov'")
    if hllc_wave_speed not in WAVE_SPEEDS:
        raise ValueError(f"hllc_wave_speed must be one of {sorted(WAVE_SPEEDS)}")
    if flux == "hllc":
        rhs = lambda state, spacing: hllc_rhs(state, spacing, wave_speed=hllc_wave_speed)
    else:
        rhs = rusanov_rhs
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
        "ncells": ncells, "final_time": final_time, "steps": steps,
        **step_metadata(final_time, steps, dt),
        "diaphragm": diaphragm, "domain": (lower, upper), "cfl": cfl,
        "left_state": tuple(left_state), "right_state": tuple(right_state),
        "flux": flux, "hllc_wave_speed": hllc_wave_speed, "device": str(dev),
    }
    return x, fields, metadata
