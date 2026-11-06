"""Measured conservation-law diagnostics for the TensorDG verification suite.

This module contains *measurement* utilities only -- nothing here knows the
answer in advance.  Every routine reads a numerical solution (or its exact
reference) and returns a number that a separate criterion can judge:

* discrete invariants of the modal DG state (cell averages of the ``P_0`` mode),
* round-off level drift of those invariants,
* analytic boundary-flux budgets for the transmissive Euler boundary,
* Rankine--Hugoniot shock speed and a numerical shock-location sensor,
* temporal refinement (``dt`` only) self-convergence order estimators.

Rationale: a scheme written in flux-difference form conserves a quantity
*exactly* (up to floating-point round-off) whenever its numerical fluxes cancel
in a telescope.  That statement is falsifiable without any reference solution,
which makes it a strong and cheap verification criterion.  The complementary
falsifiable statement -- that a conservative scheme converges to the correct
*weak* solution -- is tested with the Rankine--Hugoniot jump speed, which is an
independent analytic quantity derived from the exact Riemann solver.

All functions accept Python sequences, NumPy arrays or torch tensors and return
plain Python floats.  Placement note: this module lives under ``tests/`` because
the current task's edit whitelist (``src/``, ``tests/``, ``docs/``) does not
cover the ``tensordg/`` package; promotion into ``tensordg/conservation.py`` is
a one-file move once the package tree is writable.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "cell_averages_1d",
    "cell_averages_2d",
    "integral_1d",
    "scaled_drift",
    "conserved_totals",
    "conserved_totals_2d",
    "drift_statistics",
    "physical_flux",
    "boundary_increment",
    "rankine_hugoniot_speed",
    "shock_position",
    "linear_slope",
    "fit_order",
    "observed_orders",
    "relative_difference",
]


# ---------------------------------------------------------------------------
# 1. Discrete invariants of a modal DG state
# ---------------------------------------------------------------------------
def _to_numpy(data) -> np.ndarray:
    """Return a float64 NumPy view of ``data`` (torch tensors, lists, arrays)."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float64)


def cell_averages_1d(coefficients) -> np.ndarray:
    """Return the cell averages of a 1-D modal state.

    With the ``P_0`` Legendre mode normalised to one, the ``k = 0`` modal
    coefficient *is* the cell average; every higher mode integrates to zero over
    the element.  ``coefficients`` has shape ``(ncells, degree+1)``.
    """
    array = _to_numpy(coefficients)
    if array.ndim != 2:
        raise ValueError("1-D modal coefficients must have shape (ncells, degree+1)")
    return np.array(array[:, 0], dtype=np.float64, copy=True)


def cell_averages_2d(coefficients) -> np.ndarray:
    """Return cell averages of a 2-D tensor-product state ``(nx,ny,px+1,py+1)``."""
    array = _to_numpy(coefficients)
    if array.ndim != 4:
        raise ValueError("2-D modal coefficients must have shape (nx,ny,px+1,py+1)")
    return np.array(array[:, :, 0, 0], dtype=np.float64, copy=True)


def integral_1d(averages, length: float = 1.0) -> float:
    """Return the discrete integral ``h * sum(u_bar)`` over a uniform mesh."""
    values = _to_numpy(averages).ravel()
    if values.size == 0:
        raise ValueError("averages must be non-empty")
    return float(values.sum() * (length / values.size))


def scaled_drift(history: Sequence[float], scale: float | None = None) -> float:
    """Return ``|s(t_last)-s(t_first)| / scale`` for a conserved quantity.

    A conserved quantity drifts only at the level of floating-point round-off
    (``~1e-15`` for float64); a non-conservative flux leaves a drift that
    survives mesh refinement.

    ``scale`` defaults to ``max|s|`` over the history.  That default is
    misleading when the invariant is itself machine-small (e.g. the mass of a
    zero-mean sine wave), so callers pass an explicit magnitude such as the
    discrete ``L1`` norm of the solution instead.
    """
    values = _to_numpy(history).ravel()
    if values.size < 2:
        raise ValueError("need at least two samples")
    if scale is None:
        scale = float(np.max(np.abs(values)))
    scale = float(scale)
    if scale == 0.0:
        return 0.0
    return float(abs(values[-1] - values[0]) / scale)


# ---------------------------------------------------------------------------
# 2. Global budgets for the system of conservation laws
# ---------------------------------------------------------------------------
def conserved_totals(conservative_states, cell_width: float) -> dict[str, float]:
    """Integrate the conserved variables of a ``P_0`` Euler state.

    ``conservative_states`` has shape ``(ncells, 3)`` holding
    ``(rho, rho*u, E)`` per cell.  Returns mass, momentum and total energy.
    """
    array = _to_numpy(conservative_states)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("Euler conservative states must have shape (ncells, >=3)")
    sums = array[:, :3].sum(axis=0) * cell_width
    return {
        "mass": float(sums[0]),
        "momentum": float(sums[1]),
        "energy": float(sums[2]),
    }


def conserved_totals_2d(
    conservative_states, dx: float, dy: float
) -> dict[str, float]:
    """Integrate the four conserved variables of a 2-D ``P_0`` Euler state.

    ``conservative_states`` has shape ``(nx, ny, 4)`` holding
    ``(rho, rho*u, rho*v, E)`` per cell; ``dx``/``dy`` are the (possibly
    different) cell widths.  The integral is ``dx*dy*sum``, which is the
    discrete counterpart of :math:`\\int_\\Omega U \\,d\\Omega` on a
    Cartesian mesh.  Only bilinear flux-difference schemes whose face fluxes
    telescope in *both* directions leave these four numbers invariant, so the
    two weighing factors make a swapped or dropped ``dx``/``dy`` visible.
    """
    array = _to_numpy(conservative_states)
    if array.ndim != 3 or array.shape[2] < 4:
        raise ValueError("2-D Euler states must have shape (nx, ny, >=4)")
    if not (float(dx) > 0.0 and float(dy) > 0.0):
        raise ValueError("cell widths must be positive")
    sums = array[:, :, :4].sum(axis=(0, 1)) * (float(dx) * float(dy))
    names = ("mass", "momentum_x", "momentum_y", "energy")
    return {name: float(sums[i]) for i, name in enumerate(names)}


def drift_statistics(history, scale: float | None = None) -> dict[str, float]:
    """Summarise a *multi-step* history of one discrete invariant.

    ``history`` is the invariant sampled once per time step (at ``t=0`` and
    after every accepted step).  Beyond the plain drift, three statistics make
    a systematic leak distinguishable from float64 round-off:

    ``increment_mean`` / ``increment_rms``
        mean and root-mean-square of the per-step increments (divided by
        ``scale``).  Round-off increments are zero-mean noise, so
        ``|mean|/rms`` stays small; a one-sided leak gives ``|mean|/rms ~ 1``.
    ``bias_ratio``
        the ratio ``|increment_mean| / increment_rms`` (0 for pure noise).
    ``closure_residual``
        ``|sum(increments) - (final - initial)| / scale``: the ledger's own
        bookkeeping must close to round-off, otherwise a measurement bug is
        masquerading as a solver result.

    ``scale`` defaults to ``|history[0]|`` when that is non-zero; callers with
    a vanishing invariant must pass an explicit magnitude.
    """
    values = _to_numpy(history).ravel()
    if values.size < 3:
        raise ValueError("need at least three samples for step statistics")
    if scale is None:
        scale = abs(float(values[0]))
    scale = float(scale)
    if scale == 0.0:
        raise ValueError("scale must be non-zero; pass an explicit magnitude")
    increments = np.diff(values)
    tolerance = np.finfo(np.float64).eps
    mean = float(np.mean(increments)) / scale
    rms = float(np.sqrt(np.mean(increments ** 2))) / scale
    closure = abs(float(increments.sum()) - float(values[-1] - values[0])) / scale
    return {
        "initial": float(values[0]),
        "final": float(values[-1]),
        "samples": int(values.size),
        "absolute_drift": abs(float(values[-1] - values[0])),
        "scaled_drift": abs(float(values[-1] - values[0])) / scale,
        "max_step_increment": float(np.max(np.abs(increments))) / scale,
        "increment_mean": mean,
        "increment_rms": rms,
        # A zero rms means the invariant never moved at all (exact arithmetic).
        "bias_ratio": abs(mean) / rms if rms > tolerance else 0.0,
        "closure_residual": closure,
    }


def physical_flux(
    state: tuple[float, float, float], gamma: float = 1.4
) -> tuple[float, float, float]:
    """Return the analytic Euler flux ``(rho*u, rho*u^2+p, u*(E+p))`` of a state.

    ``state`` is ``(rho, u, p)`` in primitive variables.
    """
    rho, velocity, pressure = (float(x) for x in state)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity ** 2
    return (
        rho * velocity,
        rho * velocity ** 2 + pressure,
        velocity * (energy + pressure),
    )


def boundary_increment(
    left_state: tuple[float, float, float],
    right_state: tuple[float, float, float],
    final_time: float,
    gamma: float = 1.4,
) -> dict[str, float]:
    """Predict the change of the global integrals from the boundary data alone.

    When the waves never reach the mesh ends, the transmissive boundary keeps
    the initial states on both ends of the domain, so the global conservation law

        d/dt int_0^1 U dx = F(U(0,t)) - F(U(1,t))

    with time-independent end states integrates to
    ``int U(t) - int U(0) = t * (F(U_L) - F(U_R))`` exactly.  This is a property
    of the PDE evaluated from the initial data, not a re-evaluation of the
    code's own numerical flux, so it is a genuine test of the boundary-flux
    bookkeeping (a wall-like pressure imbalance shows up as linear momentum
    growth).
    """
    flux_left = physical_flux(left_state, gamma)
    flux_right = physical_flux(right_state, gamma)
    names = ("mass", "momentum", "energy")
    return {
        name: float(final_time * (flux_left[i] - flux_right[i]))
        for i, name in enumerate(names)
    }


# ---------------------------------------------------------------------------
# 3. Rankine--Hugoniot shock diagnostics
# ---------------------------------------------------------------------------
def rankine_hugoniot_speed(
    upstream: tuple[float, float, float],
    downstream: tuple[float, float, float],
) -> float:
    """Return the shock speed ``s = (rho2 u2 - rho1 u1) / (rho2 - rho1)``.

    The mass jump condition fixes the speed of any discontinuity a conservative
    scheme may converge to (Lax--Wendroff theorem), so the exact Riemann star
    state turns it into an independent analytic reference for a measured shock
    motion.
    """
    rho_1, velocity_1, _ = (float(x) for x in upstream)
    rho_2, velocity_2, _ = (float(x) for x in downstream)
    if abs(rho_2 - rho_1) < 1e-14:
        raise ValueError("shock speed is undefined for equal densities")
    return float((rho_2 * velocity_2 - rho_1 * velocity_1) / (rho_2 - rho_1))


def shock_position(x: Iterable[float], values: Iterable[float], level: float) -> float:
    """Locate the first left-to-right crossing of ``level`` by linear interpolation.

    Returns ``nan`` when ``values`` never drops through ``level``; the caller
    must decide whether that is a failure of the test case or of the solution.
    """
    nodes = _to_numpy(x).ravel()
    data = _to_numpy(values).ravel()
    if nodes.size != data.size:
        raise ValueError("x and values must have the same length")
    above = data >= level
    for index in range(data.size - 1):
        if above[index] and not above[index + 1]:
            slope = data[index + 1] - data[index]
            if slope == 0.0:
                return float(nodes[index])
            weight = (level - data[index]) / slope
            return float(nodes[index] + weight * (nodes[index + 1] - nodes[index]))
    return float("nan")


def linear_slope(abscissae: Iterable[float], ordinates: Iterable[float]) -> tuple[float, float]:
    """Least-squares fit ``y = a + b x``; returns ``(b, a)``."""
    x = _to_numpy(abscissae).ravel()
    y = _to_numpy(ordinates).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("need at least two paired samples")
    design = np.stack([np.ones_like(x), x], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1]), float(coefficients[0])


# ---------------------------------------------------------------------------
# 4. Refinement-order estimators
# ---------------------------------------------------------------------------
def fit_order(parameters: Iterable[float], errors: Iterable[float]) -> float:
    """Least-squares slope of ``log(error)`` against ``log(parameter)``.

    Used for both mesh refinement (``parameter = h``) and time refinement
    (``parameter = dt``).  Returns ``nan`` if fewer than two usable pairs exist.
    """
    p = _to_numpy(parameters).ravel()
    e = _to_numpy(errors).ravel()
    if p.size != e.size:
        raise ValueError("parameters and errors must have the same length")
    mask = (p > 0.0) & (e > 0.0) & np.isfinite(e)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(p[mask]), np.log(e[mask]), 1)[0])


def observed_orders(parameters: Iterable[float], errors: Iterable[float]) -> list[float]:
    """Successive two-point rates ``log(e_i/e_{i+1})/log(p_i/p_{i+1})``."""
    p = _to_numpy(parameters).ravel()
    e = _to_numpy(errors).ravel()
    if p.size != e.size:
        raise ValueError("parameters and errors must have the same length")
    rates: list[float] = []
    for i in range(p.size - 1):
        if e[i] > 0.0 and e[i + 1] > 0.0 and p[i] > 0.0 and p[i + 1] > 0.0:
            rates.append(float(math.log(e[i] / e[i + 1]) / math.log(p[i] / p[i + 1])))
        else:
            rates.append(float("nan"))
    return rates


def relative_difference(reference, candidate) -> float:
    """Return ``||candidate-reference||_2 / ||reference||_2`` for cell data.

    Applied to two solutions of the *same* mesh but different time steps, this
    is a pure temporal-error measure: the spatial discretisation error cancels,
    so the resulting order is the order of the time integrator alone.
    """
    a = _to_numpy(reference).ravel()
    b = _to_numpy(candidate).ravel()
    if a.size != b.size:
        raise ValueError("solutions must live on the same mesh")
    scale = float(np.sqrt(np.mean(a ** 2)))
    if scale == 0.0:
        raise ValueError("reference solution has zero norm")
    return float(math.sqrt(np.mean((b - a) ** 2)) / scale)