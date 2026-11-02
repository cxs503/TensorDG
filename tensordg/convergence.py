"""Verification toolkit: observed order of accuracy, Richardson extrapolation, GCI.

Everything in this module is pure NumPy and independent of the DG solvers: the
routines operate on ``(resolution, error)`` pairs or on scalar solution
functionals (quantities of interest), never on solver internals.  That keeps the
verification chain honest -- the reference for every assertion is either an
analytic value or a Richardson/GCI estimate built from the *same* numerical
series, not from the implementation under test.

Conventions
-----------
* ``resolutions`` are mesh sizes (e.g. number of cells) or step sizes; they are
  supplied in **coarse to fine** order and must be strictly increasing.
* ``errors`` are positive error norms (absolute or relative L2, max norms, ...).
* ``degrees`` are polynomial degrees for p-refinement studies.

Key references for the GCI (grid convergence index) formulas:
Roache, P.J. (1994) "Perspective: a method for uniform reporting of grid
refinement studies", J. Fluids Eng. 116(3), 405-413; Celik et al. (2008)
"Procedure for estimation and reporting of uncertainty due to discretization in
CFD applications", J. Fluids Eng. 130(7), 078001.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "pairwise_orders",
    "observed_order",
    "richardson_extrapolate",
    "gci_study",
    "convergence_table",
    "p_decay_diagnostics",
    "modal_mean_1d",
    "modal_mean_2d",
    "modal_point_value_1d",
    "modal_point_value_2d",
    "modal_l2_norm_1d",
]

#: Roache's recommended factor of safety when the observed order is estimated
#: (and therefore itself uncertain) rather than known a priori.
DEFAULT_SAFETY_FACTOR = 1.25


def _as_positive_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} needs at least two samples")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


def _as_resolutions(values: Iterable[float]) -> np.ndarray:
    array = _as_positive_array(values, "resolutions")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(
            "resolutions must be supplied in strictly increasing (coarse to fine) order"
        )
    return array


def pairwise_orders(resolutions: Iterable[float], errors: Iterable[float]) -> list[float]:
    """Observed order between every consecutive pair of refinement levels.

    With ``h = 1 / resolution`` the order for a pair is
    ``log(e_coarse / e_fine) / log(r)`` with ``r = resolution_fine /
    resolution_coarse``.
    """
    n = _as_resolutions(resolutions)
    e = _as_positive_array(errors, "errors")
    if n.size != e.size:
        raise ValueError("resolutions and errors must have the same length")
    return [
        float(math.log(e[i - 1] / e[i]) / math.log(n[i] / n[i - 1]))
        for i in range(1, n.size)
    ]


def observed_order(resolutions: Iterable[float], errors: Iterable[float]) -> float:
    """Least-squares observed order of accuracy over all refinement levels.

    Fits ``log(e) = log(C) - p * log(n)`` (``h = 1/n``) and returns ``p``.  A
    least-squares fit over three or more levels is far less sensitive to a
    single non-asymptotic grid than a two-point estimate.
    """
    n = _as_resolutions(resolutions)
    e = _as_positive_array(errors, "errors")
    if n.size != e.size:
        raise ValueError("resolutions and errors must have the same length")
    slope = np.polyfit(np.log(n), np.log(e), 1)[0]
    return float(-slope)


def _roache_order(eps21: float, eps32: float, r21: float, r32: float) -> float | None:
    """Solve Roache's implicit relation for the observed order of accuracy.

    ``eps21 = f2 - f1`` and ``eps32 = f3 - f2`` with ``f1`` the finest and
    ``f3`` the coarsest functional value.  Returns ``None`` when the data do
    not admit an asymptotic solution (sign change without oscillation, or
    divergence of the fixed-point iteration).
    """
    if eps21 == 0.0 or eps32 == 0.0:
        return None
    ratio = abs(eps32 / eps21)
    sign = math.copysign(1.0, eps32 / eps21)
    p = abs(math.log(ratio) / math.log(r21))
    for _ in range(100):
        numerator = r21 ** p - sign
        denominator = r32 ** p - sign
        if numerator <= 0.0 or denominator <= 0.0:
            return None
        p_next = (math.log(ratio) + math.log(numerator / denominator)) / math.log(r21)
        if not math.isfinite(p_next):
            return None
        if abs(p_next - p) <= 1e-12:
            return float(p_next) if p_next > 0.0 else None
        p = p_next
    return float(p) if math.isfinite(p) and p > 0.0 else None


def gci_study(
    fine: float,
    medium: float,
    coarse: float,
    refinement: float | tuple[float, float] = 2.0,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> dict:
    """Three-level Richardson extrapolation and grid convergence index.

    ``fine``, ``medium`` and ``coarse`` are the values of a scalar quantity of
    interest obtained on three successively coarser meshes.  ``refinement`` is
    either a single refinement ratio ``r = h_medium / h_fine = h_coarse /
    h_medium`` or the pair ``(r21, r32)``.

    Returns a dictionary with

    ``order_observed``      Roache order estimate (``None`` if not asymptotic)
    ``extrapolated``        Richardson estimate of the exact functional value,
                            ``f1 - (f2 - f1) / (r21**p - 1)``
    ``gci_fine_percent``    GCI of the fine-grid value, in percent
    ``gci_medium_percent``  GCI of the medium-grid value, in percent
    ``asymptotic_ratio``    ``GCI_medium / (r21**p * GCI_fine)`` (~1 in range)
    ``in_asymptotic_range`` True when ``0.8 <= asymptotic_ratio <= 1.25``
    ``relative_change_percent``  ``|fine - medium| / |fine|`` in percent
    """
    if isinstance(refinement, (tuple, list)):
        if len(refinement) != 2:
            raise ValueError("refinement must be a scalar or a (r21, r32) pair")
        r21, r32 = float(refinement[0]), float(refinement[1])
    else:
        r21 = r32 = float(refinement)
    if r21 <= 0.0 or r32 <= 0.0:
        raise ValueError("refinement ratios must be positive")
    if not all(np.isfinite([fine, medium, coarse])) or fine == 0.0:
        raise ValueError("functional values must be finite and the fine value non-zero")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")

    eps21 = medium - fine
    eps32 = coarse - medium
    p = _roache_order(eps21, eps32, r21, r32)
    result = {
        "fine": float(fine),
        "medium": float(medium),
        "coarse": float(coarse),
        "refinement": [r21, r32],
        "safety_factor": float(safety_factor),
        "relative_change_percent": float(abs(eps21) / abs(fine) * 100.0),
        "order_observed": p,
        "extrapolated": None,
        "gci_fine_percent": None,
        "gci_medium_percent": None,
        "asymptotic_ratio": None,
        "in_asymptotic_range": False,
    }
    if p is None or p <= 0.0:
        return result
    denominator = r21 ** p - 1.0
    if denominator <= 0.0:
        return result
    extrapolated = fine - eps21 / denominator
    gci_fine = safety_factor * abs(eps21) / abs(fine) / denominator * 100.0
    gci_medium = safety_factor * abs(eps32) / abs(medium) / (r32 ** p - 1.0) * 100.0
    ratio = gci_medium / (r21 ** p * gci_fine) if gci_fine > 0.0 else None
    result.update(
        {
            "extrapolated": float(extrapolated),
            "gci_fine_percent": float(gci_fine),
            "gci_medium_percent": float(gci_medium),
            "asymptotic_ratio": None if ratio is None else float(ratio),
            "in_asymptotic_range": bool(ratio is not None and 0.8 <= ratio <= 1.25),
        }
    )
    return result


def richardson_extrapolate(
    fine: float,
    medium: float,
    refinement: float = 2.0,
    order: float = 2.0,
) -> float:
    """Richardson extrapolation ``f* = f1 + (f1 - f2) / (r**p - 1)``."""
    if refinement <= 1.0:
        raise ValueError("refinement ratio must exceed one")
    if order <= 0.0:
        raise ValueError("order must be positive")
    return float(fine + (fine - medium) / (refinement ** order - 1.0))


def convergence_table(
    resolutions: Sequence[float],
    errors: Sequence[float],
    theoretical_order: float | None = None,
    order_tolerance: float = 0.2,
) -> dict:
    """Assemble a machine-readable h-refinement table.

    The returned dictionary is JSON-serialisable and is what the benchmark
    driver and the documentation generator consume, so that reported numbers
    are never transcribed by hand.
    """
    n = _as_resolutions(resolutions)
    e = _as_positive_array(errors, "errors")
    if n.size != e.size:
        raise ValueError("resolutions and errors must have the same length")
    pairs = pairwise_orders(n, e)
    lsq = observed_order(n, e)
    table = {
        "resolutions": [float(v) for v in n],
        "step_sizes": [float(1.0 / v) for v in n],
        "errors": [float(v) for v in e],
        "loglog_slopes": [
            float(v) for v in np.diff(np.log(e)) / np.diff(np.log(n))
        ],
        "pairwise_orders": pairs,
        "observed_order_lsq": lsq,
        "theoretical_order": None if theoretical_order is None else float(theoretical_order),
        "order_deviation": None if theoretical_order is None else float(lsq - theoretical_order),
        "order_tolerance": float(order_tolerance),
    }
    if theoretical_order is not None:
        table["order_within_tolerance"] = bool(
            abs(lsq - theoretical_order) <= order_tolerance
        )
        table["pairwise_within_tolerance"] = bool(
            all(abs(p - theoretical_order) <= 2.0 * order_tolerance for p in pairs)
        )
    return table


def p_decay_diagnostics(
    degrees: Sequence[int],
    errors: Sequence[float],
) -> dict:
    """Diagnostics for a p-refinement (spectral) study on a fixed mesh.

    For a smooth solution DG errors decay faster than any algebraic rate in the
    polynomial degree, so on a semi-log plot ``log(e)`` versus ``p`` is close to
    a straight line with a large negative slope.  Reported quantities:

    ``decay_rates``      ``ln(e_i / e_{i+1}) / (p_{i+1} - p_i)`` per level pair
    ``lsq_decay_rate``   least-squares slope of ``ln(e)`` versus ``p``
    ``reduction_factors`` ``e_i / e_{i+1}``
    ``monotone_decreasing``  True iff the error decreases at every level
    ``superlinear``      True iff every reduction factor exceeds ``p_fine + 1``
    """
    p = np.asarray(list(degrees), dtype=float)
    e = _as_positive_array(errors, "errors")
    if p.size != e.size:
        raise ValueError("degrees and errors must have the same length")
    if np.any(np.diff(p) <= 0.0):
        raise ValueError("degrees must be strictly increasing")
    reduction = e[:-1] / e[1:]
    decay = np.log(reduction) / np.diff(p)
    lsq = -float(np.polyfit(p, np.log(e), 1)[0])
    superlinear = bool(np.all(reduction > (p[1:] + 1.0)))
    return {
        "degrees": [int(v) for v in p],
        "errors": [float(v) for v in e],
        "reduction_factors": [float(v) for v in reduction],
        "decay_rates": [float(v) for v in decay],
        "lsq_decay_rate": lsq,
        "monotone_decreasing": bool(np.all(reduction > 1.0)),
        "superlinear": superlinear,
    }

# ---------------------------------------------------------------------------
# Scalar quantity-of-interest extraction from modal DG coefficients.
#
# The DG modules store one Legendre mode set per cell on the reference element
# ``[-1, 1]`` with ``P_0 = 1``, so the first coefficient *is* the cell average.
# ``poisson_error_2d`` / ``heat_error`` / ``l2_error`` return norms; the helpers
# below additionally expose point values and (exact-for-polynomials) cell
# averages so that a study can feed a scalar functional into :func:`gci_study`.
# ---------------------------------------------------------------------------


def _legendre_row(xi: float, degree: int) -> np.ndarray:
    return np.array(
        [np.polynomial.legendre.legval(xi, np.eye(degree + 1)[k]) for k in range(degree + 1)]
    )


def modal_mean_1d(coeff) -> float:
    """Domain average of a 1-D modal DG solution on ``[0, 1]``."""
    values = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    return float(values[:, 0].sum() / values.shape[0])


def modal_mean_2d(coeff) -> float:
    """Domain average of a 2-D modal DG solution on ``[0, 1]^2``."""
    values = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    nx = values.shape[0]
    return float(values[..., 0].sum() / nx ** 2)


def modal_point_value_1d(centers, coeff, x0: float) -> float:
    """Evaluate a 1-D modal DG solution at the physical point ``x0``."""
    centers_np = np.asarray(centers.detach().cpu().numpy() if hasattr(centers, "detach") else centers)
    values = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    ncells, modes = values.shape
    h = 1.0 / ncells
    cell = int(np.clip(np.searchsorted(centers_np, x0) - 1, 0, ncells - 1))
    xi = 2.0 * (x0 - (cell + 0.5) * h) / h
    return float(np.dot(values[cell], _legendre_row(xi, modes - 1)))


def modal_point_value_2d(x, y, coeff, x0: float, y0: float) -> float:
    """Evaluate a 2-D tensor-product modal DG solution at ``(x0, y0)``."""
    x_np = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)
    y_np = np.asarray(y.detach().cpu().numpy() if hasattr(y, "detach") else y)
    values = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    nx, ny, modes_x, modes_y = values.shape
    hx, hy = 1.0 / nx, 1.0 / ny
    cell_x = int(np.clip(np.searchsorted(x_np, x0) - 1, 0, nx - 1))
    cell_y = int(np.clip(np.searchsorted(y_np, y0) - 1, 0, ny - 1))
    xi = 2.0 * (x0 - (cell_x + 0.5) * hx) / hx
    eta = 2.0 * (y0 - (cell_y + 0.5) * hy) / hy
    return float(
        np.einsum(
            "kl,k,l->",
            values[cell_x, cell_y],
            _legendre_row(xi, modes_x - 1),
            _legendre_row(eta, modes_y - 1),
        )
    )


def modal_l2_norm_1d(centers, coeff) -> float:
    """L2 norm of a 1-D modal DG solution on ``[0, 1]`` (exact for polynomials).

    ``nq = 2 * degree + 3`` Gauss points integrate ``u_h**2`` (degree ``2p``)
    exactly, so the returned norm carries no quadrature error of its own.
    """
    centers_np = np.asarray(centers.detach().cpu().numpy() if hasattr(centers, "detach") else centers)
    values = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    ncells, modes = values.shape
    degree = modes - 1
    h = 1.0 / ncells
    nodes, weights = np.polynomial.legendre.leggauss(2 * degree + 3)
    basis = np.stack(
        [np.polynomial.legendre.legval(nodes, np.eye(degree + 1)[k]) for k in range(modes)],
        axis=1,
    )
    uh = values @ basis.T                       # [cells, quadrature]
    integral = 0.5 * h * np.sum(weights[None, :] * uh ** 2)
    return float(np.sqrt(integral))
