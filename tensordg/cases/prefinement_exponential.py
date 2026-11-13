"""p-refinement exponential convergence on an analytic solution: canonical case.

Case pool anchor
----------------
``CASE-DG-PREFINEMENT-EXPONENTIAL`` (ledger acceptance: semi-log error-vs-p
slope ``< -0.5`` per unit ``p``, i.e. decay rate ``sigma > 0.5``; metric: L2
error versus ``p = 1..6`` on a fixed mesh).

Reference
---------
I. Babuska & M. Suri, *The p- and h-p versions of the finite element method,
basic principles and properties*, SIAM Review **36**(4) (1994) 578--632,
DOI ``10.21236/ada260237``.  Theorem 3.1 gives, for analytic solutions and a
fixed mesh, exponential decay of the discretisation error in the polynomial
degree, ``||u - u_h||_L2 <= C exp(-sigma p)``, with ``sigma`` set by the width
of the analyticity strip of the solution around the element.

Physics / problem
-----------------
One-dimensional periodic linear advection ``u_t + u_x = 0`` on ``[0, 1)`` with

    u0(x) = sin(2*pi*x)      (an entire function: analyticity strip = infinity)

integrated to ``T = 1.0``, exactly one period, so the exact solution equals the
initial condition.  The mesh is fixed at ``ncells = 16`` and only ``p`` varies,
which isolates p-refinement from h-refinement.  Because the exact solution is
known in closed form, this case needs no third-party solver to produce its
reference values.

Temporal error is removed by *measurement*, not by assumption: ``solve_level``
halves the CFL until the relative change of the L2 error between two successive
time steps falls below ``TEMPORAL_TOLERANCE`` (default 1 %).  The published
study ``docs/CLOSEOUT-m16918-prefinement-criterion.md`` shows that the relative
form of this test can be satisfied *while* the level is still temporal-error
dominated; ``docs/CLOSEOUT-m17067-exact-time-floor.md`` therefore adds an
exact-in-time (matrix exponential) measurement path in
``benchmarks/p_refinement_exact_time.py``, which is the authority for
``p >= 7``.

Acceptance gate
---------------
Two nested gates, both predeclared (never re-tuned after seeing data):

``legacy_slope``          the ledger tolerance itself, ``sigma > 0.5``;
``discriminating``        :func:`tensordg.convergence.exponential_decay_criterion`,
                          which additionally requires non-decreasing pairwise
                          decay rates, a ``> 1e3`` total error reduction and
                          level-by-level monotone decrease.  The legacy slope
                          alone is *not* sufficient: audit m16499 (F4) showed an
                          algebraic ``p**-3`` law and a genuine ``C0``-kinked
                          solution both clear ``sigma > 0.5``.

Usage
-----
>>> from tensordg.cases.prefinement_exponential import p_ladder
>>> report = p_ladder(degrees=(1, 2, 3))          # doctest: +SKIP
>>> report["pass"]                                # doctest: +SKIP
True
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import exponential_decay_criterion, p_decay_diagnostics

#: key under which this case is registered in ``tensordg.cases.CASES``
CASE_KEY = "p-refinement-exponential-smooth"
#: feature-ledger slug this case is the verification evidence for
LEDGER_ANCHOR = "CASE-DG-PREFINEMENT-EXPONENTIAL"

#: bibliographic anchor (no numbers are transcribed from it -- only the law)
REFERENCE = {
    "authors": "I. Babuska, M. Suri",
    "title": (
        "The p- and h-p versions of the finite element method, "
        "basic principles and properties"
    ),
    "journal": "SIAM Review",
    "volume": 36,
    "issue": 4,
    "pages": "578-632",
    "year": 1994,
    "doi": "10.21236/ada260237",
    "statement": "L2 error decays exponentially in p for analytic solutions: "
                 "||u - u_h|| <= C exp(-sigma p)",
}

DOMAIN = (0.0, 1.0)
NCELLS = 16               # fixed mesh: isolates p-refinement
FINAL_TIME = 1.0          # one full period -> exact solution equals u0
SPEED = 1.0
CFL_BASE = 0.08           # halved until the temporal error is measurable-negligible
DEGREES = (1, 2, 3, 4, 5, 6)
TEMPORAL_TOLERANCE = 0.01     # relative change of L2 between successive CFLs
MAX_TEMPORAL_REFINEMENTS = 6

#: ledger tolerance: semi-log slope per unit p must satisfy |slope| > 0.5
LEGACY_SLOPE_THRESHOLD = 0.5
#: library cap on the polynomial degree (``tensordg`` modal basis)
MAX_DEGREE = 8
#: above this degree the float64 operator representation floor dominates (m17067)
FLOOR_LIMITED_DEGREES = (7, 8)


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """``u0(x) = sin(2*pi*x)`` -- entire, so p-convergence is exponential."""
    return torch.sin(2.0 * math.pi * x)


def exact_solution(x: torch.Tensor, t: float = FINAL_TIME,
                   speed: float = SPEED) -> torch.Tensor:
    """Exact travelling-wave solution ``sin(2*pi*(x - speed*t))``."""
    return torch.sin(2.0 * math.pi * (x - speed * t))


def prefinement_parameters() -> dict:
    """Return the canonical case parameter dictionary (JSON-serialisable)."""
    return {
        "case_key": CASE_KEY,
        "ledger_anchor": LEDGER_ANCHOR,
        "reference": dict(REFERENCE),
        "domain": list(DOMAIN),
        "periodic": True,
        "pde": "u_t + u_x = 0",
        "initial_condition": "sin(2*pi*x)",
        "exact_solution": "sin(2*pi*(x - t))",
        "ncells": NCELLS,
        "final_time": FINAL_TIME,
        "speed": SPEED,
        "degrees": list(DEGREES),
        "cfl_base": CFL_BASE,
        "temporal_tolerance": TEMPORAL_TOLERANCE,
        "legacy_slope_threshold": LEGACY_SLOPE_THRESHOLD,
        "max_degree": MAX_DEGREE,
        "floor_limited_degrees": list(FLOOR_LIMITED_DEGREES),
        "metric": "relative L2 error vs the analytic travelling wave",
    }


def solve_level(
    degree: int,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
    speed: float = SPEED,
    cfl: float = CFL_BASE,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    max_refinements: int = MAX_TEMPORAL_REFINEMENTS,
) -> dict:
    """L2 error of one p level, with the CFL refined until time error is small.

    The loop halves the CFL and stops as soon as the relative change of the L2
    error between two successive CFLs is ``<= temporal_tolerance``.  The whole
    refinement history is returned so a caller can audit (or reject) the level.
    """
    if degree < 1 or degree > MAX_DEGREE:
        raise ValueError(f"degree must be in 1..{MAX_DEGREE}, got {degree}")
    history: list[dict] = []
    current_cfl = cfl
    error: float | None = None
    converged = False
    for _ in range(max_refinements + 1):
        centers, coefficients, metadata = solve_advection(
            ncells=ncells, degree=degree, final_time=final_time,
            speed=speed, cfl=current_cfl, time_integrator="rk4",
            initial=initial_condition,
        )
        new_error = l2_error(centers, coefficients, final_time, speed=speed,
                             exact=exact_solution)
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
    return {
        "degree": degree,
        "ncells": ncells,
        "h": (DOMAIN[1] - DOMAIN[0]) / ncells,
        "dofs": ncells * (degree + 1),
        "relative_l2": float(error),
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
    speed: float = SPEED,
    cfl: float = CFL_BASE,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    error_ratio_min: float = 1.0e3,
) -> dict:
    """Run the p ladder and adjudicate both predeclared gates.

    Returns ``levels`` (per-degree records), ``diagnostics``
    (:func:`tensordg.convergence.p_decay_diagnostics`), ``gates`` (legacy slope
    and discriminating criterion) and ``pass`` = conjunction of the two.
    """
    degrees = tuple(int(p) for p in degrees)
    if len(degrees) < 2:
        raise ValueError("a p ladder needs at least two degrees")
    levels = [
        solve_level(
            p, ncells=ncells, final_time=final_time, speed=speed, cfl=cfl,
            temporal_tolerance=temporal_tolerance,
        )
        for p in degrees
    ]
    errors = [level["relative_l2"] for level in levels]
    diagnostics = p_decay_diagnostics(degrees, errors)
    legacy = {
        "measured": diagnostics["lsq_decay_rate"],
        "threshold": LEGACY_SLOPE_THRESHOLD,
        "pass": bool(abs(diagnostics["lsq_decay_rate"]) > LEGACY_SLOPE_THRESHOLD),
    }
    discriminating = exponential_decay_criterion(
        diagnostics, error_ratio_min=error_ratio_min,
    )
    return {
        "case_key": CASE_KEY,
        "ledger_anchor": LEDGER_ANCHOR,
        "grades": list(degrees),
        "ncells": ncells,
        "final_time": final_time,
        "speed": speed,
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
        [None] + report["diagnostics"]["reduction_factors"],
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


def solve_prefinement_case(degrees: Sequence[int] | Iterable[int] = DEGREES, **kwargs) -> dict:
    """Alias following the ``solve_<case>`` convention of this package."""
    return p_ladder(degrees=degrees, **kwargs)


__all__ = [
    "CASE_KEY",
    "LEDGER_ANCHOR",
    "REFERENCE",
    "DOMAIN",
    "NCELLS",
    "FINAL_TIME",
    "SPEED",
    "CFL_BASE",
    "DEGREES",
    "TEMPORAL_TOLERANCE",
    "LEGACY_SLOPE_THRESHOLD",
    "MAX_DEGREE",
    "FLOOR_LIMITED_DEGREES",
    "anchor_table",
    "exact_solution",
    "initial_condition",
    "p_ladder",
    "prefinement_parameters",
    "solve_level",
    "solve_prefinement_case",
]