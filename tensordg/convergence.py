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
from typing import Iterable, Mapping, Sequence

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


#: Default predeclared thresholds of :func:`exponential_decay_criterion`.
SPECTRAL_SIGMA_MIN = 1.0        # decay rate must exceed any algebraic p**-k law
SPECTRAL_ERROR_RATIO_MIN = 1.0e3  # at least three decades of error reduction
SPECTRAL_RATE_TOL = 1.0e-9      # relative slack on "pairwise rates non-decreasing"


def exponential_decay_criterion(
    diag: Mapping[str, object],
    *,
    sigma_min: float = SPECTRAL_SIGMA_MIN,
    error_ratio_min: float = SPECTRAL_ERROR_RATIO_MIN,
    rate_tol: float = SPECTRAL_RATE_TOL,
) -> dict:
    """Composite criterion separating spectral from algebraic p-decay.

    Takes the output of :func:`p_decay_diagnostics` and applies four
    predeclared conditions, all of which must hold:

    1. ``lsq_decay_rate > sigma_min`` -- an algebraic law ``p**-k`` has a
       *saturated* slope, so a large slope rules it out as the dominant law;
    2. pairwise ``decay_rates`` non-decreasing (they decrease monotonically for
       a pure algebraic law, which is the decisive discriminator);
    3. ``e(p_min) / e(p_max) > error_ratio_min`` -- at least three decades of
       measured reduction, so the slope is not fitted through noise;
    4. ``monotone_decreasing`` -- the error falls at every single level.

    The thresholds are predeclared constants of the library, not tuned per
    case.  Raise them (never lower them) when a case demands a stricter gate.
    """
    rate = float(diag["lsq_decay_rate"])  # type: ignore[arg-type]
    rates = [float(v) for v in diag["decay_rates"]]  # type: ignore[index]
    errors = [float(v) for v in diag["errors"]]  # type: ignore[index]
    if not errors:
        raise ValueError("diagnostics must contain at least one error level")
    ratio = errors[0] / errors[-1]
    non_decreasing = (
        all(b >= a * (1.0 - rate_tol) for a, b in zip(rates, rates[1:]))
        if len(rates) > 1
        else True
    )
    conditions = {
        "sigma_above_threshold": {
            "measured": rate,
            "threshold": float(sigma_min),
            "pass": bool(rate > sigma_min),
        },
        "decay_rates_non_decreasing": {
            "measured": rates,
            "tolerance": float(rate_tol),
            "pass": bool(non_decreasing),
        },
        "error_ratio_above_threshold": {
            "measured": ratio,
            "threshold": float(error_ratio_min),
            "pass": bool(ratio > error_ratio_min),
        },
        "monotone_decreasing": {
            "measured": bool(diag["monotone_decreasing"]),  # type: ignore[arg-type]
            "pass": bool(diag["monotone_decreasing"]),  # type: ignore[arg-type]
        },
    }
    return {
        "conditions": conditions,
        "pass": all(c["pass"] for c in conditions.values()),
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


# ---------------------------------------------------------------------------
# Verification floor: is a measured error still a *signal*?
#
# A p-refinement study pushes the discretisation error towards the roundoff
# level of the discrete solution itself.  Once the measured error approaches
# ``eps * ||u_h||`` the number stops being a discretisation measurement: it is
# a difference of two O(1) float64 numbers.  These helpers make that boundary
# explicit instead of leaving a claim like "spectral convergence continues"
# unverifiable (see docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md F5).
# ---------------------------------------------------------------------------

FLOAT64_EPS = float(np.finfo(np.float64).eps)


def roundoff_floor(field_norm: float, eps: float = FLOAT64_EPS) -> float:
    """Smallest L2 error the float64 representation of a solution can resolve.

    ``u_h - u`` is a difference of two numbers of size ``||u||``; each carries
    an absolute rounding of order ``eps * ||u||``, so an error norm much below
    that is measurement noise rather than discretisation error.
    """
    if field_norm < 0.0:
        raise ValueError("field_norm must be non-negative")
    return float(eps * field_norm)


def signal_to_noise(errors: Iterable[float], floor: float) -> list[float]:
    """Error-to-floor ratio per level (``inf`` when the floor vanishes)."""
    e = np.asarray(list(errors), dtype=float)
    if floor < 0.0:
        raise ValueError("floor must be non-negative")
    if floor == 0.0:
        return [float("inf")] * int(e.size)
    return [float(v) for v in e / floor]


def resolved_levels(
    degrees: Sequence[int],
    errors: Sequence[float],
    floor: float,
    snr_min: float = 10.0,
) -> dict:
    """Split p-levels into measurement-resolved and floor-limited ones.

    A level counts as *resolved* only if its error exceeds ``snr_min * floor``.
    Only resolved levels may be used to fit a decay rate; the returned
    ``max_resolvable_reduction`` is the largest error reduction the float64
    measurement can certify at all (``e_coarsest / (snr_min * floor)``).
    """
    p = np.asarray(list(degrees), dtype=int)
    e = _as_positive_array(errors, "errors")
    if p.size != e.size:
        raise ValueError("degrees and errors must have the same length")
    if snr_min <= 0.0:
        raise ValueError("snr_min must be positive")
    snr = signal_to_noise(e, floor)
    resolved = [int(v) for v, s in zip(p, snr) if s >= snr_min]
    limited = [int(v) for v, s in zip(p, snr) if s < snr_min]
    threshold = snr_min * floor
    return {
        "floor": float(floor),
        "snr_min": float(snr_min),
        "snr": [float(s) for s in snr],
        "resolved_degrees": resolved,
        "floor_limited_degrees": limited,
        "max_resolvable_reduction": (
            float(e[0] / threshold) if threshold > 0.0 else float("inf")
        ),
    }

# ---------------------------------------------------------------------------
# Exact-in-time integration of a *linear* semi-discrete operator.
#
# A verification study that measures a *spatial* error must not let the time
# integrator contribute to it.  For a linear semi-discrete system
# ``du/dt = L u`` the exact solution is ``u(t) = exp(L t) u(0)``, so the spatial
# error can be isolated completely -- no CFL refinement loop, no temporal error
# floor.  This matters at high polynomial degree, where the spatial error of a
# spectral DG scheme can fall below the truncation error of an explicit
# Runge-Kutta scheme: see docs/CLOSEOUT-m17067-exact-time-floor.md.
# ---------------------------------------------------------------------------


def semidiscrete_operator(apply, ndof: int, dtype=np.float64) -> np.ndarray:
    """Assemble the dense matrix of a linear operator from its action.

    ``apply`` maps a flat coefficient vector of length ``ndof`` to its time
    derivative, i.e. it *is* the shipped right-hand side (including its
    boundary/interface fluxes).  Assembling column by column keeps the measured
    operator identical to the production path instead of re-deriving it.
    """
    if ndof <= 0:
        raise ValueError("ndof must be positive")
    operator = np.empty((ndof, ndof), dtype=dtype)
    for j in range(ndof):
        e = np.zeros(ndof, dtype=dtype)
        e[j] = 1.0
        column = np.asarray(apply(e), dtype=dtype).reshape(-1)
        if column.size != ndof:
            raise ValueError(
                f"apply returned {column.size} entries, expected {ndof}"
            )
        operator[:, j] = column
    return operator


def operator_linearity_residual(apply, operator: np.ndarray, seed: int = 0) -> float:
    """Relative residual of ``apply(a + b) == L a + L b`` for random ``a, b``.

    Returns ``||apply(a+b) - L(a+b)|| / ||apply(a+b)||``; a non-zero value means
    the assembled matrix does not describe the shipped operator (nonlinearity,
    state-dependent flux, hidden limiter), and any exact-time study built on it
    would be measuring something else.
    """
    rng = np.random.default_rng(seed)
    ndof = operator.shape[0]
    a = rng.standard_normal(ndof)
    b = rng.standard_normal(ndof)
    lhs = np.asarray(apply(a + b), dtype=float).reshape(-1)
    rhs = operator @ (a + b)
    denominator = float(np.linalg.norm(lhs))
    if denominator == 0.0:
        return float("inf")
    return float(np.linalg.norm(lhs - rhs) / denominator)


def expm_scaled_squaring(matrix, t: float = 1.0, taylor_terms: int = 20,
                         threshold: float = 0.25) -> np.ndarray:
    """``exp(matrix * t)`` by scaling and squaring with an extended-precision Taylor sum.

    ``np.longdouble`` (80-bit on x86/loong64) accumulation is used because the
    whole point of an exact-time study is to resolve errors far below the
    float64 floor of an explicit integrator: ``scipy.linalg.expm`` in float64
    carries a forward error of order ``eps * ||L t||``, which for a DG operator
    (``||L t|| ~ 1e2``) is the same size as the errors being measured
    (``~1e-14``).  ``taylor_terms`` and ``threshold`` control the scaling and
    squaring schedule; varying them gives a control test of the truncation.
    """
    raw = np.asarray(matrix)
    dtype = np.clongdouble if np.iscomplexobj(raw) else np.longdouble
    a = raw.astype(dtype) * dtype(t)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    norm = float(np.max(np.sum(np.abs(a), axis=1)))
    squarings = 0
    while norm > threshold:
        norm /= 2.0
        squarings += 1
    a = a / dtype(2) ** squarings
    ident = np.eye(a.shape[0], dtype=dtype)
    term = ident.copy()
    total = ident.copy()
    for k in range(1, taylor_terms + 1):
        term = term @ a / dtype(k)
        total = total + term
    for _ in range(squarings):
        total = total @ total
    return total


def exact_time_solution(apply, u0, t: float) -> np.ndarray:
    """Solution of ``du/dt = apply(u)`` at time ``t`` for a linear ``apply``."""
    u0 = np.asarray(u0, dtype=np.longdouble).reshape(-1)
    operator = semidiscrete_operator(apply, u0.size)
    return expm_scaled_squaring(operator, t) @ u0
