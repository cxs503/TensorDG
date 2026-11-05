"""Analytic-baseline tools for a discrete mathematical-entropy budget (1-D Euler).

Physical/mathematical setting
-----------------------------
For the ideal-gas Euler equations

    d_t rho + d_x (rho u)       = 0
    d_t (rho u) + d_x (rho u^2 + p) = 0
    d_t E + d_x (u (E + p))     = 0,   E = p/(gamma - 1) + rho u^2 / 2,

the (physical) entropy ``s = ln(p / rho**gamma)`` satisfies, for *smooth*
solutions, the balance law

    d_t (rho s) + d_x (rho u s) = 0                                  (1)

and, for *discontinuous* (shock-bearing) solutions, the entropy inequality

    d_t (rho s) + d_x (rho u s) = P >= 0,                            (2)

with ``P`` supported on shocks.  ``rho s`` is the standard (negated Harten)
mathematical entropy density; (1)--(2) are exactly the statements a shock
capturing scheme has to reproduce in the weak sense.  This module supplies

* the discrete entropy density / flux evaluated on the *full* DG polynomial
  (not on cell averages), so that the measured budget is not polluted by an
  O(h^2) averaging error;
* the analytic entropy production of a single shock, obtained purely from
  Rankine-Hugoniot states.  For a shock of speed ``sigma`` separating states
  ``a`` (upstream) and ``b`` (downstream),

      P = sigma * [rho s] - [rho u s],   [.] = (.)_b - (.)_a,        (3)

  and the total produced up to time ``T`` is ``P * T`` as long as no wave
  reaches the domain ends (the production rate of a self-similar shock is
  constant in time).

Equation (3) is the single number every shock-capturing run below is judged
against; nothing in this module is fitted to a numerical result.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4


# --------------------------------------------------------------------------- #
# continuum entropy pair
# --------------------------------------------------------------------------- #
def gas_entropy(density, pressure, gamma: float = GAMMA):
    """``s = ln(p / rho**gamma)`` (defined up to an additive constant)."""
    density = np.asarray(density, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    return np.log(pressure) - gamma * np.log(density)


def entropy_density(density, pressure, gamma: float = GAMMA):
    """``rho s`` -- the quantity whose integral grows across a shock."""
    density = np.asarray(density, dtype=float)
    return density * gas_entropy(density, pressure, gamma)


def entropy_flux(density, velocity, pressure, gamma: float = GAMMA):
    """``rho u s`` -- the entropy flux of (1)--(2)."""
    density = np.asarray(density, dtype=float)
    return density * velocity * gas_entropy(density, pressure, gamma)


def primitive_from_conservative(states, gamma: float = GAMMA):
    """``(rho, rho u, E)`` -> ``(rho, u, p)`` for the last axis of ``states``."""
    states = np.asarray(states, dtype=float)
    density = states[..., 0]
    velocity = states[..., 1] / density
    pressure = (gamma - 1.0) * (states[..., 2] - 0.5 * states[..., 1] ** 2 / density)
    return density, velocity, pressure


# --------------------------------------------------------------------------- #
# discrete entropy of a modal DG solution
# --------------------------------------------------------------------------- #
def legendre_modes(nodes, degree: int):
    """Legendre basis ``phi_k(xi)`` on ``[-1, 1]`` at ``nodes`` -> shape (q, m).

    Same (unnormalised) basis and coefficient convention as
    :func:`tensordg.euler_dg_1d._operators`, i.e. ``c_k = (2k+1)/2 * int u phi_k``.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        values.append(np.polynomial.legendre.legval(nodes, polynomial))
    return np.stack(values, axis=1)


def total_entropy_from_coefficients(coeff, dx: float, degree: int, quadrature_order: int | None = None,
                                    gamma: float = GAMMA) -> float:
    """``int rho s dx`` evaluated on the reconstructed DG solution.

    ``rho s`` is not polynomial in the conserved variables, so the quadrature
    order is deliberately generous (``max(degree + 2, 8)`` Gauss points);
    :func:`total_entropy_from_coefficients_convergence` shows the value is
    quadrature independent at that order.
    """
    order = quadrature_order or max(degree + 2, 8)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = legendre_modes(nodes, degree)
    values = np.einsum("cmv,qm->cqv", np.asarray(coeff, dtype=float), phi)
    density, velocity, pressure = primitive_from_conservative(values, gamma)
    del velocity
    integrand = entropy_density(density, pressure, gamma)
    return 0.5 * dx * float(np.einsum("q,cq->", weights, integrand))


def total_entropy_from_coefficients_convergence(coeff, dx: float, degree: int,
                                                quadrature_orders=(4, 6, 8, 10, 12),
                                                gamma: float = GAMMA):
    """Entropy integral for a ladder of quadrature orders (diagnostic self-test)."""
    return [
        total_entropy_from_coefficients(coeff, dx, degree, order, gamma)
        for order in quadrature_orders
    ]


def total_entropy_from_cell_averages(states, dx: float, gamma: float = GAMMA) -> float:
    """``int rho s dx`` from piecewise-constant (P0 / finite-volume) states."""
    density, _, pressure = primitive_from_conservative(states, gamma)
    return float(np.sum(entropy_density(density, pressure, gamma)) * dx)


def total_entropy_from_profiles(density, pressure, x, gamma: float = GAMMA) -> float:
    """``int rho s dx`` from pointwise profiles (exact/sampled solutions)."""
    return float(np.trapezoid(entropy_density(density, pressure, gamma), x))


# --------------------------------------------------------------------------- #
# analytic Rankine-Hugoniot budget
# --------------------------------------------------------------------------- #
def shock_speed_and_production(upstream, downstream, gamma: float = GAMMA):
    """Shock speed and entropy production rate (3) for a single shock.

    Parameters
    ----------
    upstream, downstream : (rho, u, p)
        States on the two sides of the shock.

    Returns
    -------
    dict with ``sigma`` (shock speed), ``production`` (rate of ``int rho s``
    per unit time) and the two jumps used.
    """
    rho_a, u_a, p_a = (float(value) for value in upstream)
    rho_b, u_b, p_b = (float(value) for value in downstream)
    if rho_b == rho_a:
        raise ValueError("a shock needs a density jump")
    sigma = (rho_b * u_b - rho_a * u_a) / (rho_b - rho_a)
    jump_density = rho_b * float(gas_entropy(rho_b, p_b, gamma)) - rho_a * float(gas_entropy(rho_a, p_a, gamma))
    jump_flux = (rho_b * u_b * float(gas_entropy(rho_b, p_b, gamma))
                 - rho_a * u_a * float(gas_entropy(rho_a, p_a, gamma)))
    production = sigma * jump_density - jump_flux
    return {
        "sigma": sigma,
        "production": production,
        "jump_rho_s": jump_density,
        "jump_rho_u_s": jump_flux,
        "upstream": (rho_a, u_a, p_a),
        "downstream": (rho_b, u_b, p_b),
    }


def gamma_shock_ratio(pressure_ratio: float, gamma: float = GAMMA) -> float:
    """``rho_b / rho_a`` across a shock from the pressure ratio (RH + ideal gas)."""
    a = (gamma - 1.0) / (gamma + 1.0)
    return (pressure_ratio + a) / (a * pressure_ratio + 1.0)


def star_state(left, right, gamma: float = GAMMA):
    """Exact Riemann star pressure and velocity (delegates to the repo solver)."""
    from benchmarks.riemann_exact import star_state as _star_state

    pressure, velocity = _star_state(left, right, gamma)
    return float(pressure), float(velocity)


def analytic_right_shock_budget(left, right, final_time: float, gamma: float = GAMMA):
    """Analytic ``int rho s`` production for a Riemann problem with one shock.

    Only the ``p* > p_right`` configuration (Sod, Lax) is implemented: the
    downstream state of the right shock follows from the exact star state via
    Rankine-Hugoniot, so no numerical solution enters the baseline.
    """
    rho_r, u_r, p_r = (float(value) for value in right)
    pressure_star, velocity_star = star_state(left, right, gamma)
    if not pressure_star > p_r:
        raise ValueError("this helper only covers a right-going shock (p* > p_right)")
    density_star = rho_r * gamma_shock_ratio(pressure_star / p_r, gamma)
    downstream = (density_star, velocity_star, pressure_star)
    budget = shock_speed_and_production(right, downstream, gamma)
    budget["star_state"] = (pressure_star, velocity_star)
    budget["production_total"] = budget["production"] * final_time
    budget["final_time"] = final_time
    budget["shock_position"] = 0.5 + budget["sigma"] * final_time
    return budget


# --------------------------------------------------------------------------- #
# helpers used by the study and the tests
# --------------------------------------------------------------------------- #
def relative_error(reference: float, candidate: float) -> float:
    """``|candidate - reference| / |reference|`` (dimensionless)."""
    reference = float(reference)
    candidate = float(candidate)
    return abs(candidate - reference) / max(abs(reference), 1e-300)


def pointwise_entropy_error(numerical, exact, x) -> float:
    """Relative ``L2`` error of the entropy density profile ``rho s``.

    Deprecated for reporting: ``rho s = rho ln(p / rho**gamma)`` is *signed* and
    crosses zero inside shock-tube data, so ``||e|| / ||exact||`` divides an
    ``O(1)`` absolute error by a nearly cancelling norm.  On the Sod case it
    returns ``163068 %`` and carries no information about the solver.  Use
    :func:`entropy_profile_error` instead, which normalises by the intrinsic
    entropy-jump scale of the shock.
    """
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), 1e-300))


def entropy_jump_scale(upstream, downstream, gamma: float = GAMMA) -> float:
    """``|[s]| = |s_downstream - s_upstream|`` of a single shock.

    This is the intrinsic scale of the ``rho s`` profile: the entropy plateau
    jump the profile has to reproduce, and the only sensible denominator for a
    profile error.
    """
    rho_a, _, p_a = (float(value) for value in upstream)
    rho_b, _, p_b = (float(value) for value in downstream)
    return abs(float(gas_entropy(rho_b, p_b, gamma)) - float(gas_entropy(rho_a, p_a, gamma)))


def entropy_profile_error(numerical, exact, x, entropy_scale: float | None = None) -> dict:
    """Profile error of the signed entropy density, in interpretable units.

    Parameters
    ----------
    numerical, exact : array_like
        Pointwise ``rho s`` on the sample abscissae ``x``.
    entropy_scale : float, optional
        ``|[s]|`` of the shock (:func:`entropy_jump_scale`).  When given, the
        RMS error is also expressed as a fraction of the entropy jump.

    Returns
    -------
    dict with

    ``rms``
        ``||num - exact||_2 / sqrt(L)`` -- an RMS error in entropy-density units
        (units of ``rho s``, i.e. mass per volume; ``s`` itself is dimensionless).
    ``rms_over_entropy_jump``
        ``rms / |[s]|`` when ``entropy_scale`` is supplied: "how large is the
        profile error compared with the entropy step the shock has to produce".
    ``rms_over_mean_abs``
        ``rms / mean(|exact|)`` -- a scale-free fallback that does not rely on
        the shock data.
    ``legacy_relative_l2``
        The deprecated ``||e|| / ||exact||`` for provenance only.
    """
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    x = np.asarray(x, dtype=float)
    length = max(float(x[-1] - x[0]), 1e-300)
    residual = numerical - exact
    # sqrt(mean(e^2)) == sqrt( int e^2 dx / L ) for uniformly spaced samples; the
    # plain Euclidean norm would grow like sqrt(#samples) and is not a field error.
    integral = getattr(np, "trapezoid", None) or np.trapz
    rms = float(np.sqrt(max(integral(residual ** 2, x), 0.0) / length))
    out = {
        "rms": rms,
        "rms_over_mean_abs": rms / max(float(np.mean(np.abs(exact))), 1e-300),
        "legacy_relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(exact), 1e-300)),
    }
    if entropy_scale is not None:
        out["rms_over_entropy_jump"] = rms / max(float(entropy_scale), 1e-300)
    return out


def observed_order(cells, values):
    """Slopes of ``log(values)`` against ``log(1/cells)`` (i.e. an EOC ladder)."""
    cells = np.asarray(cells, dtype=float)
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.log(values[1:] / values[:-1]) / np.log(cells[:-1] / cells[1:])
    return [None if not np.isfinite(slope) else float(slope) for slope in slopes]