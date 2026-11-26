"""Exact cell averages of the Riemann solution: the reference instrument, fixed.

Why this module exists
----------------------
``benchmarks.riemann_exact.cell_averages`` builds its reference by integrating the
exact Riemann solution with a **fixed** ``q``-point Gauss rule over each cell
(``q = 8`` by default).  For a *discontinuous* exact solution that rule is not
exact inside the cells that contain a wave, and the error it makes is **O(jump),
independent of the mesh**: a ``q``-point Gauss rule integrates a step crossing the
cell with an error ``rho_q * jump * dx`` while the cell average itself carries a
``1/dx``, so the cell-average error is ``rho_q * jump`` with ``rho_q`` a fixed
constant of the rule (``rho_8 ~ 1e-2``).  Refinement moves the wave through fewer
cells but does not shrink the per-cell error, so the induced relative-L2 bias of
the reference decays only like ``N^{-1/2}`` (fewer affected cells), never to zero.

This matters because that reference is the *denominator of the whole 3 %-gate
family* for the Sod problem: ``tests/test_euler_sod_1d.py``,
``tests/test_euler_2d_p0_verification.py`` and ``tests/test_repro_cks1998.py`` all
compare against ``benchmarks.sod_exact.cell_averages``, and the limiting field of
the extruded-2D gate sits at 2.5 % of a 3 % bar.  A reference bias that cannot be
refined away therefore eats a fixed share of the gate margin for ever.

What this module provides
-------------------------
* :func:`wave_positions` — the analytic wave positions of a 1-D Riemann problem
  (closed form, Rankine-Hugoniot + isentropic invariants, *not* solver output).
* :func:`cell_averages` — cell averages computed by splitting every cell at the
  analytic breakpoints it contains and integrating the (smooth) pieces with
  Gauss-Legendre; converges to machine precision in ``quadrature_order``.
* :func:`fixed_rule_cell_averages` — the legacy fixed-rule reference, exposed so
  the bias can be measured instead of asserted.
* :func:`reference_bias` — per-mesh bias of the legacy reference against the
  exact one (absolute and relative), which is the quantity the gates inherit.
* :func:`relative_l2` — the norm used by the Sod gates.

The exact reference is the *unique* answer: a cell average is an integral, and
this module computes that integral, so the numbers here carry no discretisation
knob once ``quadrature_order`` has converged (asserted in the tests).
"""
from __future__ import annotations

import numpy as np

from benchmarks.riemann_exact import GAMMA, cell_averages as _fixed_rule_cell_averages
from benchmarks.riemann_exact import sample, star_state

FIELDS = ("density", "velocity", "pressure")
DEFAULT_ORDER = 40


def wave_positions(left, right, final_time: float, gamma: float = GAMMA,
                   diaphragm: float = 0.5) -> dict:
    """Analytic positions of the waves of a 1-D Riemann problem at ``final_time``.

    The four (three for two-shock configurations) positions are the closed-form
    speeds of the outer edge of each wave multiplied by time and shifted by the
    diaphragm.  They are the breakpoints at which the exact solution is
    discontinuous or loses smoothness:

    ``head``      outer edge of a left rarefaction (or the left shock),
    ``tail``      inner edge of a left rarefaction (absent if the left wave is a shock),
    ``contact``   material contact,
    ``front``     right shock (or the outer edge of a right rarefaction).

    Speeds come from the Rankine-Hugoniot relations for shocks and from the
    isentropic Riemann invariants for rarefactions, i.e. from the same closed-form
    machinery as :func:`benchmarks.riemann_exact.sample`, but derived here from the
    star state so that callers can reason about the wave *structure* rather than
    re-derive it.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    p_star, u_star = star_state(left, right, gamma)
    positions = {}
    if p_star > p_l:  # left shock
        ratio = p_star / p_l
        speed = u_l - a_l * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)
        )
        positions["head"] = diaphragm + final_time * speed
    else:  # left rarefaction
        a_star = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        positions["head"] = diaphragm + final_time * (u_l - a_l)
        positions["tail"] = diaphragm + final_time * (u_star - a_star)
    positions["contact"] = diaphragm + final_time * u_star
    if p_star > p_r:  # right shock
        ratio = p_star / p_r
        speed = u_r + a_r * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)
        )
        positions["front"] = diaphragm + final_time * speed
    else:  # right rarefaction
        a_star = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        positions["tail_right"] = diaphragm + final_time * (u_star + a_star)
        positions["front"] = diaphragm + final_time * (u_r + a_r)
    return positions


def breakpoints(left, right, final_time: float, gamma: float = GAMMA,
                diaphragm: float = 0.5, domain=(0.0, 1.0)) -> np.ndarray:
    """Sorted analytic breakpoints of the exact solution inside ``domain``."""
    lower, upper = domain
    values = sorted(wave_positions(left, right, final_time, gamma, diaphragm).values())
    return np.array([v for v in values if lower < v < upper], dtype=np.float64)


def cell_averages(ncells: int, final_time: float, left, right,
                  diaphragm: float = 0.5, domain=(0.0, 1.0),
                  gamma: float = GAMMA, quadrature_order: int = DEFAULT_ORDER,
                  *, conserved: bool = False):
    """Exact cell averages of the Riemann solution (breakpoint-split quadrature).

    With conserved=True the integrals are of density, momentum and total
    energy density, evaluated before averaging.

    Each cell is cut at the analytic breakpoints it contains; every resulting
    sub-interval is smooth, so a single Gauss-Legendre rule of order
    ``quadrature_order`` integrates it to machine precision.  The result is the
    *definition* of the cell average -- there is no mesh-dependent accuracy knob
    left, which is what makes it usable as a 3 %-gate reference.
    """
    if ncells < 1:
        raise ValueError("ncells must be positive")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be >= 2")
    lower, upper = domain
    dx = (upper - lower) / ncells
    centers = lower + (np.arange(ncells) + 0.5) * dx
    cuts = breakpoints(left, right, final_time, gamma, diaphragm, domain)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    fields = ("density", "momentum", "energy") if conserved else FIELDS
    out = {key: np.zeros(ncells, dtype=np.float64) for key in fields}
    for i, center in enumerate(centers):
        x_left, x_right = center - 0.5 * dx, center + 0.5 * dx
        inner = cuts[(cuts > x_left) & (cuts < x_right)]
        edges = np.concatenate(([x_left], inner, [x_right]))
        for a, b in zip(edges[:-1], edges[1:]):
            width = b - a
            if width <= 0.0:
                continue
            points = a + 0.5 * width * (nodes + 1.0)
            values = sample(points, final_time, left, right, diaphragm, gamma)
            if conserved:
                rho, u, p = (values[key] for key in FIELDS)
                values = dict(density=rho, momentum=rho*u,
                              energy=p/(gamma-1.0) + 0.5*rho*u**2)
            for key in fields:
                out[key][i] += 0.5 * width * float(np.dot(values[key], weights))
    for key in fields:
        out[key] /= dx
    return centers, out


def fixed_rule_cell_averages(ncells: int, final_time: float, left, right,
                             diaphragm: float = 0.5, domain=(0.0, 1.0),
                             gamma: float = GAMMA, quadrature_order: int = 8):
    """Legacy reference: ``benchmarks.riemann_exact.cell_averages`` (fixed rule).

    Kept as an explicit, named instrument so that its bias against
    :func:`cell_averages` can be measured by the tests and by the audit driver.
    """
    return _fixed_rule_cell_averages(
        ncells, final_time, left, right, diaphragm=diaphragm, domain=domain,
        quadrature_order=quadrature_order, gamma=gamma,
    )


def relative_l2(numerical, reference) -> float:
    """Relative L2 norm of the Sod gates: ``||u - u_ref|| / ||u_ref||`` (RMS form)."""
    numerical = np.asarray(numerical, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.sqrt(np.mean((numerical - reference) ** 2))
        / np.sqrt(np.mean(reference ** 2))
    )


def reference_bias(ncells: int, final_time: float, left, right,
                   diaphragm: float = 0.5, domain=(0.0, 1.0),
                   gamma: float = GAMMA, quadrature_order: int = 8) -> dict:
    """Bias of the legacy fixed-rule reference against the exact one.

    Returns per-field ``max_abs`` (largest cell-average error the legacy reference
    makes, in physical units), ``rms_rel`` (the induced relative-L2 bias, i.e. the
    share of the gate it consumes) and the affected cell count.
    """
    centers, exact = cell_averages(
        ncells, final_time, left, right, diaphragm, domain, gamma
    )
    _, legacy = fixed_rule_cell_averages(
        ncells, final_time, left, right, diaphragm, domain, gamma,
        quadrature_order=quadrature_order,
    )
    cuts = breakpoints(left, right, final_time, gamma, diaphragm, domain)
    dx = (domain[1] - domain[0]) / ncells
    cell_edges = np.concatenate(([centers[0] - 0.5 * dx], centers + 0.5 * dx))
    affected = int(np.sum((np.histogram(cuts, bins=cell_edges)[0]) > 0))
    out = {"ncells": ncells, "affected_cells": affected,
           "quadrature_order": quadrature_order,
           "breakpoints": [float(v) for v in cuts], "fields": {}}
    for key in FIELDS:
        out["fields"][key] = {
            "max_abs": float(np.max(np.abs(legacy[key] - exact[key]))),
            "rms_rel": relative_l2(legacy[key], exact[key]),
            "max_rel_of_max": float(
                np.max(np.abs(legacy[key] - exact[key])) / np.max(np.abs(exact[key]))
            ),
        }
    return out


def bias_ladder(meshes, final_time: float, left, right, **kwargs) -> list:
    """Run :func:`reference_bias` over a list of meshes."""
    return [reference_bias(n, final_time, left, right, **kwargs) for n in meshes]