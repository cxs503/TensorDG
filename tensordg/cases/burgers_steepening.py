"""CASE-DG-BURGERS-STEEPEN: gradient peak of a steepening inviscid Burgers profile.

Why this case exists
--------------------
TensorDG has verified the *smooth* inviscid Burgers solution through the relative
L2 error of the whole profile (:mod:`tensordg.burgers_1d`,
``docs/benchmarks/burgers-1d.md``) at ``t = 0.1``, where
``t / t_b`` = 0.50.  An L2 norm is a *mean* quantity: it is insensitive to the
one thing nonlinear steepening is about, namely that the solution develops a
gradient that grows without bound as ``t -> t_b``.  A scheme may hold a
0.01 % L2 error and still smear that gradient peak by tens of percent, and no
existing gate would notice.

This case therefore measures the **gradient peak**

    G(t) = max_x |du_h/dx|(x, t)

of the production modal-DG run at a *near-breaking* time.  It reuses the
production solver unchanged (``burgers_1d.solve_burgers``: modal Legendre
basis, Rusanov interface flux, SSP-RK3); the only new code here is the
reference, which is analytic.

Analytic reference (no tabulated values are needed or used)
-----------------------------------------------------------
With the production initial data ``u_0(x) = 0.2 + 0.8 sin(2 pi x)`` on the
periodic unit interval, the classical characteristic solution of
``u_t + (u^2/2)_x = 0`` is implicit,

    u = u_0(xi),      xi = x - t u_0(xi),                                 (1)

(Whitham, *Linear and Nonlinear Waves*, Wiley 1974, ch. 2, kinematic waves with
``c(u) = u``; the same relation is what ``burgers_1d.exact_profile`` iterates
numerically).  Differentiating (1) gives the exact gradient field

    du/dx = u_0'(xi) / (1 + t u_0'(xi)).                                   (2)

The map ``x -> xi`` stays one-to-one while ``1 + t u_0' > 0``, i.e. while

    t < t_b = 1 / max(-u_0') = 1 / (2 pi * 0.8) = 1/ (1.6 pi) = 0.1989437...

and on the negative-slope branch of the data, ``u_0' = -S`` with
``S = 1.6 pi``, (2) is largest at the flattest point of the initial profile:

    G(t) = S / (1 - t S),        0 <= t < t_b = 1/S.                       (3)

So the reference is a **closed form**: at the documented terminal time
``T = 0.19`` (``T/t_b = 0.955``) the exact peak is ``G(T) = 111.81...``,
22.2x the initial peak ``G(0) = S = 5.0265...``.  Nothing is hand-copied; every
number in the report is produced by this module or read back from the artifact
``artifacts/burgers_steepening_peak/metrics.json``.

Two properties make the reference falsifiable rather than tautological:

* ``exact_peak_gradient`` (3) is cross-checked in the test suite against
  ``sampled_peak_gradient``, a *numerically maximised* (2) on a 20001-point
  grid (independent code path, no closed form involved);
* the same measurement protocol is applied to the reference
  (``peak_reference_at_quadrature``): the maximum is taken over exactly the
  quadrature points the numerical solution is sampled on.  Without that step a
  point-sample max would look like a scheme error when part of it is only the
  sampling of a smooth extremum.

Robustness of the reference near breaking (an ecosystem observation)
-------------------------------------------------------------------
``burgers_1d.exact_profile`` inverts (1) with a plain Newton iteration capped at
40 steps.  Its Jacobian ``1 + t u_0'`` collapses to ``1 - t/t_b``, so the cap is
exceeded near breaking: at ``T = 0.195`` (``t/t_b = 0.980``, still a *smooth*
solution) that helper raises ``RuntimeError``.  :func:`characteristic_foot`
below solves the same equation with a *bracketed* bisection (bracket
``[x - t max u_0, x - t min u_0]`` encloses the root by monotonicity of
``xi -> xi + t u_0(xi)``) followed by a Newton polish, so it converges for every
``0 <= t < t_b``.  It is the reference used here; the production helper is left
untouched (its 40-step cap is a documented limitation, not a bug this case
needs to fix).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..burgers_1d import (
    AMPLITUDE,
    MEAN,
    _operators,
    exact_profile,
    initial_profile,
    solve_burgers,
)

__all__ = [
    "CASE_KEY",
    "DOMAIN",
    "FINAL_TIME",
    "CFL",
    "CELLS",
    "DEGREES",
    "TOLERANCE",
    "TIME_LADDER",
    "CFL_LADDER",
    "SLOPE_MAX",
    "BREAKING_TIME",
    "initial_profile_np",
    "initial_slope",
    "characteristic_foot",
    "exact_solution",
    "exact_gradient",
    "exact_peak_gradient",
    "sampled_peak_gradient",
    "solve_case",
    "burgers_steepening_parameters",
]

CASE_KEY = "DG-BURGERS-STEEPEN"

#: periodic unit interval; the production initial data live on [0, 1]
DOMAIN = (0.0, 1.0)

#: documented terminal time; ``FINAL_TIME / BREAKING_TIME = 0.955``
FINAL_TIME = 0.19

#: CFL of the documented run (matches ``docs/benchmarks/burgers-1d.md``)
CFL = 0.1

#: cell ladder for the documented p-refinement / h-refinement study
CELLS = (64, 128, 256, 512)
DEGREES = (1, 2, 3)

#: acceptance band on the *relative* gradient-peak error
TOLERANCE = 0.05

#: times used for the steepening-law sweep (all pre-breaking)
TIME_LADDER = (0.05, 0.10, 0.15, 0.18, 0.19)

#: CFL values used for the time-step sensitivity arm
CFL_LADDER = (0.2, 0.1, 0.05)

#: ``max|u_0'| = 2 pi * AMPLITUDE`` -- the initial gradient peak
SLOPE_MAX = 2.0 * math.pi * AMPLITUDE

#: breaking time of the initial data, ``t_b = 1 / max(-u_0')``
BREAKING_TIME = 1.0 / SLOPE_MAX


def burgers_steepening_parameters() -> dict:
    """Single source of truth for the documented parameter set."""
    return {
        "case_key": CASE_KEY,
        "domain": DOMAIN,
        "boundary": "periodic",
        "initial_condition": f"{MEAN} + {AMPLITUDE}*sin(2*pi*x)",
        "mean": MEAN,
        "amplitude": AMPLITUDE,
        "final_time": FINAL_TIME,
        "breaking_time": BREAKING_TIME,
        "time_fraction_of_breaking": FINAL_TIME / BREAKING_TIME,
        "cfl": CFL,
        "cells": CELLS,
        "degrees": DEGREES,
        "tolerance": TOLERANCE,
        "time_ladder": TIME_LADDER,
        "cfl_ladder": CFL_LADDER,
        "initial_gradient_peak": SLOPE_MAX,
        "flux": "Rusanov (Lax-Friedrichs)",
        "time_integrator": "SSP-RK3",
        "basis": "unnormalised modal Legendre",
    }


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------
def initial_profile_np(x):
    """numpy twin of ``burgers_1d.initial_profile`` (torch) for the reference."""
    return MEAN + AMPLITUDE * np.sin(2.0 * math.pi * np.asarray(x, dtype=float))


def initial_slope(x):
    """``u_0'(x) = 2 pi A cos(2 pi x)`` (elementwise, numpy or torch)."""
    if isinstance(x, torch.Tensor):
        return 2.0 * math.pi * AMPLITUDE * torch.cos(2.0 * math.pi * x)
    return 2.0 * math.pi * AMPLITUDE * np.cos(2.0 * math.pi * np.asarray(x, dtype=float))


def characteristic_foot(x, final_time: float, iterations: int = 90, polish: int = 3):
    """Invert the characteristic relation (1) for ``xi`` -- bracketed, robust.

    ``xi -> xi + t u_0(xi)`` is strictly increasing for ``t < t_b`` with
    derivative ``1 + t u_0' >= 1 - t/t_b > 0``, so the root of
    ``G(xi) = xi - x + t u_0(xi)`` is bracketed by
    ``[x - t max u_0, x - t min u_0]``.  Bisection (90 steps) then Newton polish
    converge everywhere in the smooth regime, unlike a capped plain Newton.
    """
    x = np.asarray(x, dtype=float)
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if final_time == 0.0:
        return x.copy()
    if final_time >= BREAKING_TIME:
        raise ValueError(
            f"final_time={final_time} is at or beyond the breaking time "
            f"{BREAKING_TIME}: the characteristic solution (1) is multi-valued"
        )
    lo = x - final_time * (MEAN + AMPLITUDE)
    hi = x - final_time * (MEAN - AMPLITUDE)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        residual = mid - x + final_time * initial_profile_np(mid)
        lo = np.where(residual <= 0.0, mid, lo)
        hi = np.where(residual > 0.0, mid, hi)
    xi = 0.5 * (lo + hi)
    for _ in range(polish):
        residual = xi - x + final_time * initial_profile_np(xi)
        xi = xi - residual / (1.0 + final_time * initial_slope(xi))
    return xi


def exact_solution(x, final_time: float = FINAL_TIME):
    """Exact characteristic solution ``u(x, t) = u_0(xi)`` of the inviscid problem."""
    return initial_profile_np(characteristic_foot(x, final_time))


def exact_gradient(x, final_time: float = FINAL_TIME):
    """Exact gradient field (2): ``u_0'(xi) / (1 + t u_0'(xi))``."""
    xi = characteristic_foot(x, final_time)
    slope = initial_slope(xi)
    return slope / (1.0 + final_time * slope)


def exact_peak_gradient(final_time: float = FINAL_TIME) -> float:
    """Closed form (3): ``G(t) = S / (1 - t S)``, ``S = max|u_0'|``."""
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if final_time >= BREAKING_TIME:
        raise ValueError(
            f"final_time={final_time} is at or beyond the breaking time {BREAKING_TIME}"
        )
    return SLOPE_MAX / (1.0 - final_time * SLOPE_MAX)


def sampled_peak_gradient(
    final_time: float = FINAL_TIME, samples: int = 20001, pad: float = 1.0,
) -> tuple[float, float]:
    """Numerically maximise the gradient field (2) on a uniform grid.

    Independent of the closed form (3): used by the tests to *falsify* it.
    ``pad`` extends the grid one period on each side so the periodic maximum is
    not clipped; the grid spacing is ``1/samples`` on the unit cell.
    """
    x = np.linspace(-pad, 1.0 + pad, samples, endpoint=False)
    gradient = np.abs(exact_gradient(x, final_time))
    index = int(np.argmax(gradient))
    return float(gradient[index]), float(x[index])


# ---------------------------------------------------------------------------
# numerical solution (production kernel, unchanged)
# ---------------------------------------------------------------------------
def _gradient_basis_extrema(degree: int, samples: int = 201):
    """Dense reference-element grid and the derivative-basis values on it."""
    xi = np.linspace(-1.0, 1.0, samples)
    columns = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        columns.append(
            np.polynomial.legendre.legval(xi, derivative) if len(derivative)
            else np.zeros_like(xi)
        )
    return xi, np.stack(columns, axis=1)


def solve_case(
    cells: int = CELLS[-1],
    degree: int = DEGREES[-1],
    final_time: float = FINAL_TIME,
    cfl: float = CFL,
    device: str = "cpu",
    protocol_samples: int = 201,
) -> dict:
    """Run the production DG kernel and measure the gradient peak against (3).

    Three peak readings are recorded so that *measurement protocol* is not
    confused with *scheme error*:

    ``peak_numeric_polynomial``
        the primary metric: inside every cell the DG gradient is a polynomial
        of degree ``p-1`` in the reference coordinate, so its extremum is found
        by dense evaluation (``protocol_samples`` points per cell) of the modal
        derivative.  This is the peak of the *numerical* gradient field itself,
        free of point-sampling bias.
    ``peak_numeric_quadrature``
        the naive reading: max over the ``max(2p+3, 4)`` Gauss points the
        solution is stored on.  Kept as a protocol sensitivity arm.
    ``peak_reference_closed`` / ``peak_reference_at_quadrature``
        the exact value (3) and the same naive reading applied to the exact
        gradient field, i.e. the sampling bias of the naive protocol.
    """
    centers, coeff, metadata = solve_burgers(
        ncells=cells, degree=degree, final_time=final_time, cfl=cfl, device=device,
    )
    h = 1.0 / cells
    nodes, weights, phi, dphi, _, _ = _operators(degree, coeff.device, coeff.dtype)
    xq = (centers[:, None] + 0.5 * h * nodes[None, :]).reshape(-1).cpu().numpy()
    values = (coeff @ phi.T).reshape(-1).cpu().numpy()
    gradient_quad = ((coeff @ dphi.T) * (2.0 / h)).reshape(-1).cpu().numpy()

    # primary metric: per-cell polynomial extremum of the DG gradient
    xi_dense, dphi_dense = _gradient_basis_extrema(degree, protocol_samples)
    gradient_dense = (
        (coeff.cpu().numpy() @ dphi_dense.T) * (2.0 / h)
    )  # (cells, samples)
    flat = np.abs(gradient_dense).reshape(-1)
    index = int(np.argmax(flat))
    cell_index, sample_index = divmod(index, protocol_samples)
    peak_numeric_poly = float(flat[index])
    peak_x_poly = float(
        (cell_index + 0.5) * h + 0.5 * h * xi_dense[sample_index]
    )

    reference_values = exact_solution(xq, final_time)
    reference_gradient_quad = exact_gradient(xq, final_time)

    peak_numeric_quad = float(np.max(np.abs(gradient_quad)))
    peak_reference_closed = exact_peak_gradient(final_time)
    peak_reference_quad = float(np.max(np.abs(reference_gradient_quad)))
    peak_reference_sampled, peak_x_reference_sampled = sampled_peak_gradient(final_time)

    error = values - reference_values
    weights_np = weights.cpu().numpy()
    l2_error = float(
        math.sqrt(0.5 * h * float(np.sum(weights_np * error.reshape(cells, -1) ** 2)))
    )
    l2_exact = float(
        math.sqrt(0.5 * h * float(np.sum(weights_np * reference_values.reshape(cells, -1) ** 2)))
    )

    # does the production Newton-based helper still invert the characteristics here?
    production_helper = "ok"
    try:
        exact_profile(torch.as_tensor(xq, dtype=coeff.dtype), final_time)
    except RuntimeError as exc:  # documented friction path
        production_helper = f"RuntimeError: {exc}"

    return {
        "cells": int(cells),
        "degree": int(degree),
        "final_time": float(final_time),
        "cfl": float(cfl),
        "steps": int(metadata["steps"]),
        "dt": float(metadata["dt"]),
        "peak_numeric_polynomial": peak_numeric_poly,
        "peak_x_numeric_polynomial": peak_x_poly,
        "peak_numeric_quadrature": peak_numeric_quad,
        "peak_reference_closed": peak_reference_closed,
        "peak_reference_sampled": peak_reference_sampled,
        "peak_x_reference": peak_x_reference_sampled % 1.0,
        "peak_reference_at_quadrature": peak_reference_quad,
        "relative_peak_error": peak_numeric_poly / peak_reference_closed - 1.0,
        "relative_peak_error_quadrature_protocol": peak_numeric_quad / peak_reference_closed - 1.0,
        "relative_peak_error_naive_ratio": peak_numeric_quad / peak_reference_quad - 1.0,
        "relative_l2_error": l2_error / l2_exact,
        "mean_conservation_error": float(np.mean(coeff[:, 0].cpu().numpy()) - MEAN),
        "production_exact_profile": production_helper,
    }
