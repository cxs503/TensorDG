"""Canonical case ``DG-PREFINEMENT-EXPONENTIAL-2D``: p-refinement in 2-D.

2-D companion of :mod:`tensordg.cases.prefinement_exponential` (which is 1-D).
Same PDE family and same analytic solution class, but on the unit square with a
tensor-product Legendre DG discretisation, so that the exponential-in-``p``
convergence statement of Babuska & Suri (1994) is tested where the operators are
two-dimensional (directional splitting of the volume/face terms, 2-D quadrature).

Problem (periodic unit square, ``u_t + a . grad u = 0`` with ``a = (1, 0.5)``)::

    u0(x, y) = 1 + sin(2*pi*x) * sin(2*pi*y)          (entire function)
    ue(x, y, t) = 1 + sin(2*pi*(x - ax*t)) * sin(2*pi*(y - ay*t))

Fixed mesh ``8 x 8`` cells with ``p = 1..6`` isolates p-refinement from
h-refinement.  The RK4 step is refined (CFL halving) per level until the L2
error changes by less than ``TEMPORAL_TOLERANCE`` (1 %), so the measured error is
spatial.

Note on the degree range: :func:`tensordg.advection_2d.solve_advection_2d`
historically rejected ``degree > 3`` although its operators are degree-generic
(the same arbitrary cap that used to exist in 1-D).  The cap is now the explicit
:data:`tensordg.advection_2d.MAX_DEGREE`; the degrees actually *measured* here
stop at the level where float64 round-off dominates (see the benchmark's
signal-to-noise table).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ..advection_2d import MAX_DEGREE, error_2d, solve_advection_2d
from ..convergence import exponential_decay_criterion, p_decay_diagnostics

CASE_KEY = "DG-PREFINEMENT-EXPONENTIAL-2D"
LEDGER_ANCHOR = "CASE-DG-PREFINEMENT-EXPONENTIAL"

REFERENCE = {
    "authors": "I. Babuska, M. Suri",
    "title": (
        "The p- and h-p versions of the finite element method, basic principles "
        "and properties"
    ),
    "journal": "SIAM Review 36(4), 578-632 (1994)",
    "doi": "10.21236/ada260237",
    "statement": (
        "for analytic solutions the p-version error decays exponentially in the "
        "polynomial degree, e(p) <= C exp(-sigma p)"
    ),
}

DOMAIN = (0.0, 1.0)
VELOCITY = (1.0, 0.5)
FINAL_TIME = 1.0
NCELLS = 8                     # 8 x 8 cells, held fixed while p varies
DEGREES = (1, 2, 3, 4, 5, 6)
CFL_BASE = 0.1                 # refined per level to suppress temporal error
TEMPORAL_TOLERANCE = 0.01      # < 1 % change between successive CFLs
MAX_TEMPORAL_REFINEMENTS = 6

# Predeclared gates (identical values to the 1-D case, see audit m16499 F4).
LEGACY_SLOPE_THRESHOLD = 0.5   # legacy sigma > 0.5: necessary, not sufficient
SIGMA_MIN_DISCRIMINATING = 1.0
ERROR_RATIO_MIN = 1.0e3
DECAY_RATE_TOL = 1.0e-9


def initial_condition(x, y):
    """Entire periodic initial data (accepts scalars or arrays)."""
    import math

    return 1.0 + math.sin(2.0 * math.pi * x) * math.sin(2.0 * math.pi * y)


def exact_solution(x, y, t: float = FINAL_TIME,
                   velocity: tuple[float, float] = VELOCITY):
    """Analytic solution: the initial profile advected by ``velocity * t``."""
    import math

    ax, ay = velocity
    return 1.0 + math.sin(2.0 * math.pi * (x - ax * t)) * math.sin(
        2.0 * math.pi * (y - ay * t)
    )


def prefinement_parameters() -> dict:
    """Machine-readable case definition (no fabricated numbers)."""
    return {
        "case_key": CASE_KEY,
        "ledger_anchor": LEDGER_ANCHOR,
        "reference": dict(REFERENCE),
        "domain": list(DOMAIN),
        "periodic": True,
        "pde": "u_t + ax*u_x + ay*u_y = 0",
        "initial_condition": "1 + sin(2*pi*x)*sin(2*pi*y)",
        "exact_solution": "1 + sin(2*pi*(x-ax*t))*sin(2*pi*(y-ay*t))",
        "velocity": list(VELOCITY),
        "ncells": NCELLS,
        "final_time": FINAL_TIME,
        "degrees": list(DEGREES),
        "cfl_base": CFL_BASE,
        "temporal_tolerance": TEMPORAL_TOLERANCE,
        "legacy_slope_threshold": LEGACY_SLOPE_THRESHOLD,
        "sigma_min_discriminating": SIGMA_MIN_DISCRIMINATING,
        "error_ratio_min": ERROR_RATIO_MIN,
        "max_degree": MAX_DEGREE,
        "metric": "relative L2 error vs the analytic travelling wave",
    }


def solve_level(
    degree: int,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
    velocity: tuple[float, float] = VELOCITY,
    cfl: float = CFL_BASE,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    max_refinements: int = MAX_TEMPORAL_REFINEMENTS,
    error_fn=error_2d,
) -> dict:
    """Relative L2 error of one p level with the RK4 step refined for stability.

    The CFL is halved until the relative change of the L2 error between two
    successive CFLs is ``<= temporal_tolerance``; the whole history is returned
    so a caller can audit (or reject) the level.
    """
    if degree < 1 or degree > MAX_DEGREE:
        raise ValueError(f"degree must be in 1..{MAX_DEGREE}, got {degree}")
    history: list[dict] = []
    current_cfl = cfl
    error: float | None = None
    converged = False
    for _ in range(max_refinements + 1):
        x, y, coeff, metadata = solve_advection_2d(
            ncells=ncells, degree=degree, final_time=final_time,
            velocity=velocity, cfl=current_cfl,
        )
        new_error = float(error_fn(x, y, coeff, final_time, velocity)["relative_l2"])
        change = (abs(new_error - error) / new_error) if error else float("inf")
        history.append({
            "cfl": current_cfl,
            "dt": metadata["dt"],
            "steps": metadata["steps"],
            "relative_l2": new_error,
            "change_vs_previous": change,
        })
        error = new_error
        if change <= temporal_tolerance:
            converged = True
            break
        current_cfl *= 0.5
    assert error is not None
    nx, ny = metadata["ncells"]
    return {
        "degree": degree,
        "ncells": list(metadata["ncells"]),
        "h": 1.0 / nx,
        "dofs": nx * ny * (degree + 1) ** 2,
        "relative_l2": error,
        "cfl": history[-1]["cfl"],
        "dt": history[-1]["dt"],
        "steps": history[-1]["steps"],
        "temporal_change": history[-1]["change_vs_previous"],
        "temporal_converged": converged,
        "refinements": len(history) - 1,
        "history": history,
    }


def p_ladder(
    degrees: Sequence[int] | Iterable[int] = DEGREES,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
    velocity: tuple[float, float] = VELOCITY,
    cfl: float = CFL_BASE,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    error_ratio_min: float = ERROR_RATIO_MIN,
) -> dict:
    """Run the p ladder and adjudicate both predeclared gates.

    ``pass`` is the conjunction of the legacy slope gate (``sigma > 0.5``) and the
    discriminating composite criterion (``sigma > 1``, non-decreasing pairwise
    decay rates, ``e(p_min)/e(p_max) > 1e3``), i.e. exactly the gate that the
    1-D case moved to after audit m16499/m16918.
    """
    degrees = tuple(int(p) for p in degrees)
    if len(degrees) < 2:
        raise ValueError("a p ladder needs at least two degrees")
    levels = [
        solve_level(
            p, ncells=ncells, final_time=final_time, velocity=velocity, cfl=cfl,
            temporal_tolerance=temporal_tolerance,
        )
        for p in degrees
    ]
    errors = [level["relative_l2"] for level in levels]
    diagnostics = p_decay_diagnostics(degrees, errors)
    legacy = {
        "measured": diagnostics["lsq_decay_rate"],
        "threshold": LEGACY_SLOPE_THRESHOLD,
        "pass": bool(diagnostics["lsq_decay_rate"] > LEGACY_SLOPE_THRESHOLD),
    }
    discriminating = exponential_decay_criterion(
        diagnostics,
        sigma_min=SIGMA_MIN_DISCRIMINATING,
        error_ratio_min=error_ratio_min,
        rate_tol=DECAY_RATE_TOL,
    )
    return {
        "case_key": CASE_KEY,
        "ledger_anchor": LEDGER_ANCHOR,
        "degrees": list(degrees),
        "ncells": ncells,
        "final_time": final_time,
        "velocity": list(velocity),
        "levels": levels,
        "errors": [float(e) for e in errors],
        "diagnostics": diagnostics,
        "gates": {"legacy_slope": legacy, "discriminating": discriminating},
        "temporal_all_converged": all(lv["temporal_converged"] for lv in levels),
        "pass": bool(legacy["pass"] and discriminating["pass"]),
    }


def anchor_table(report: dict | None = None) -> list[dict]:
    """Per-degree table ``p, DOFs, L2, reduction vs previous`` for reporting."""
    report = report if report is not None else p_ladder()
    rows = []
    for level, reduction in zip(
        report["levels"],
        [None] + list(report["diagnostics"]["reduction_factors"]),
    ):
        rows.append({
            "degree": level["degree"],
            "dofs": level["dofs"],
            "cfl": level["cfl"],
            "steps": level["steps"],
            "relative_l2": level["relative_l2"],
            "reduction_vs_previous": reduction,
            "temporal_converged": level["temporal_converged"],
        })
    return rows


def solve_prefinement_2d_case(
    degrees: Sequence[int] | Iterable[int] = DEGREES, **kwargs,
) -> dict:
    """Alias following the ``solve_<case>`` convention of this package."""
    return p_ladder(degrees=degrees, **kwargs)


__all__ = [
    "CASE_KEY",
    "LEDGER_ANCHOR",
    "REFERENCE",
    "DOMAIN",
    "VELOCITY",
    "NCELLS",
    "FINAL_TIME",
    "DEGREES",
    "CFL_BASE",
    "TEMPORAL_TOLERANCE",
    "LEGACY_SLOPE_THRESHOLD",
    "SIGMA_MIN_DISCRIMINATING",
    "ERROR_RATIO_MIN",
    "MAX_DEGREE",
    "anchor_table",
    "exact_solution",
    "initial_condition",
    "p_ladder",
    "prefinement_parameters",
    "solve_level",
    "solve_prefinement_2d_case",
]