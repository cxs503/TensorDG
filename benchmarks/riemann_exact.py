"""Exact one-dimensional Euler Riemann solver for two constant initial states."""
from __future__ import annotations

import numpy as np

GAMMA = 1.4


def _f_and_derivative(p, state, gamma):
    rho, _, pressure = state
    sound = np.sqrt(gamma * pressure / rho)
    if p > pressure:
        a = 2.0 / ((gamma + 1.0) * rho)
        b = (gamma - 1.0) / (gamma + 1.0) * pressure
        root = np.sqrt(a / (p + b))
        value = (p - pressure) * root
        derivative = root * (1.0 - 0.5 * (p - pressure) / (p + b))
    else:
        ratio = p / pressure
        value = 2.0 * sound / (gamma - 1.0) * (
            ratio ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0
        )
        derivative = 1.0 / (rho * sound) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
    return value, derivative


def star_state(left, right, gamma: float = GAMMA):
    """Return star pressure and velocity after a guarded Newton iteration."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    pressure = max(
        1e-12,
        0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r),
    )
    for _ in range(60):
        f_l, df_l = _f_and_derivative(pressure, left, gamma)
        f_r, df_r = _f_and_derivative(pressure, right, gamma)
        updated = pressure - (f_l + f_r + u_r - u_l) / (df_l + df_r)
        if updated <= 0.0:
            updated = 0.5 * pressure
        if abs(updated - pressure) <= 1e-14 * max(1.0, pressure):
            pressure = updated
            break
        pressure = updated
    else:
        raise RuntimeError("Riemann star-pressure iteration did not converge")
    f_l, _ = _f_and_derivative(pressure, left, gamma)
    f_r, _ = _f_and_derivative(pressure, right, gamma)
    velocity = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    return float(pressure), float(velocity)


def sample(x, final_time, left, right, diaphragm: float = 0.5, gamma: float = GAMMA):
    """Sample all shock/rarefaction branches of the exact Riemann solution."""
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    x = np.asarray(x, dtype=np.float64)
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    p_star, u_star = star_state(left, right, gamma)
    s = (x - diaphragm) / final_time
    rho = np.empty_like(x)
    velocity = np.empty_like(x)
    pressure = np.empty_like(x)

    left_region = s <= u_star
    if p_star > p_l:
        ratio = p_star / p_l
        shock_speed = u_l - a_l * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)
        )
        initial = left_region & (s <= shock_speed)
        star = left_region & ~initial
        rho_star = rho_l * (
            (ratio + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0)
        )
        rho[star] = rho_star
        velocity[star] = u_star
        pressure[star] = p_star
    else:
        a_star = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        initial = left_region & (s <= u_l - a_l)
        star = left_region & (s > u_star - a_star)
        fan = left_region & ~initial & ~star
        rho[star] = rho_l * (p_star / p_l) ** (1.0 / gamma)
        velocity[star] = u_star
        pressure[star] = p_star
        base = 2.0 / (gamma + 1.0) + (gamma - 1.0) / ((gamma + 1.0) * a_l) * (u_l - s[fan])
        rho[fan] = rho_l * base ** (2.0 / (gamma - 1.0))
        pressure[fan] = p_l * base ** (2.0 * gamma / (gamma - 1.0))
        velocity[fan] = 2.0 / (gamma + 1.0) * (a_l + 0.5 * (gamma - 1.0) * u_l + s[fan])
    rho[initial] = rho_l
    velocity[initial] = u_l
    pressure[initial] = p_l

    right_region = ~left_region
    if p_star > p_r:
        ratio = p_star / p_r
        shock_speed = u_r + a_r * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)
        )
        initial = right_region & (s >= shock_speed)
        star = right_region & ~initial
        rho_star = rho_r * (
            (ratio + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0)
        )
        rho[star] = rho_star
        velocity[star] = u_star
        pressure[star] = p_star
    else:
        a_star = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        initial = right_region & (s >= u_r + a_r)
        star = right_region & (s <= u_star + a_star)
        fan = right_region & ~initial & ~star
        rho[star] = rho_r * (p_star / p_r) ** (1.0 / gamma)
        velocity[star] = u_star
        pressure[star] = p_star
        base = 2.0 / (gamma + 1.0) - (gamma - 1.0) / ((gamma + 1.0) * a_r) * (u_r - s[fan])
        rho[fan] = rho_r * base ** (2.0 / (gamma - 1.0))
        pressure[fan] = p_r * base ** (2.0 * gamma / (gamma - 1.0))
        velocity[fan] = 2.0 / (gamma + 1.0) * (-a_r + 0.5 * (gamma - 1.0) * u_r + s[fan])
    rho[initial] = rho_r
    velocity[initial] = u_r
    pressure[initial] = p_r
    return {"density": rho, "velocity": velocity, "pressure": pressure}


def cell_averages(
    ncells, final_time, left, right, diaphragm=0.5,
    domain=(0.0, 1.0), quadrature_order=8, gamma=GAMMA,
):
    lower, upper = domain
    dx = (upper - lower) / ncells
    centers = lower + (np.arange(ncells) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    points = centers[:, None] + 0.5 * dx * nodes[None, :]
    fields = sample(points, final_time, left, right, diaphragm, gamma)
    return centers, {
        key: 0.5 * np.sum(value * weights[None, :], axis=1)
        for key, value in fields.items()
    }
