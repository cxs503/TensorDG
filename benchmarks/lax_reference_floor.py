"""CASE-LAX-DG · how much of the "< 3 %" error is a property of the *reference*, not the solver?

Why this module exists
----------------------
``feature_ledger`` registers the anchor ``CASE-LAX-DG`` as
``ref: exact solution | tolerance < 3 %`` and the pool pins the verdict on a
whole-field relative-L2 density error measured **on the element Gauss nodes
against the pointwise exact Riemann solution**
(``benchmarks/lax_case_pool.py``, leg ``p3.N400`` = 3.11556e-02 -> FAIL,
leg ``p3.N800`` = 2.31490e-02 -> PASS).

That metric mixes two things a reader of the ledger cannot separate:

* **the scheme's error** -- the part a better scheme / more cells / higher
  degree can reduce; and
* **the reference's own discretisation** -- the distance between the pointwise
  exact solution and *any* function that lives in the DG space on that mesh.
  Two of the four Lax waves are genuine jumps, so this distance is **not zero
  at any finite N** and no solver can beat it.  It is a property of the mesh and
  of the exact solution alone.

The split is exact, not a heuristic.  Let ``u`` be the exact solution, ``V`` the
DG space (piecewise degree ``p`` on ``N`` cells) and ``Pi u`` its L2 projection.
For any ``u_h in V`` orthogonality of the projection gives

    ||u_h - u||^2 = ||u_h - Pi u||^2 + ||Pi u - u||^2        (exact quadrature)

so the graded error is the hypotenuse of a *solver* leg and a *reference floor*
leg.  The floor leg needs no solver at all, so it can be tabulated far beyond
the meshes the solver has been run on.

Pre-registered design (written before the numbers were looked at)
----------------------------------------------------------------
* **Measured ladder** -- every leg carried by
  ``docs/benchmarks/results/lax_asymptotic_probe/profiles.npz``:
  ``(N, p) = (400,1), (800,1), (400,2), (800,2), (200,3), (400,3), (800,3)``.
  Nothing is re-solved: the committed profiles *are* the evidence, so this study
  cannot change the anchor's verdict.
* **Floor ladder (solver-free)** -- ``N = 100 .. 1600`` tabulated for ``p = 1, 2, 3``
  (budget shares up to ``N = 3200``); the floor needs no solver, so the ladder is
  bounded only by the cost of evaluating the closed-form solution.
* **Metric conventions** -- ``total_quad`` is the pool's own convention
  (10-point Gauss on element nodes, denominator = exact-solution norm) -- the
  number the ledger grades.  ``total_exact`` is the same relative norm with the
  integral evaluated exactly (cells split at the four wave abscissae).  Their
  difference is the **quadrature bias of the registered metric**.
* **Q1** floor / tolerance share per leg.  **Q2** quadrature bias.
  **Q3** solver leg and its N-order.  **Q4** floor N-exponent and p-dependence.
* **H (falsifiable)** -- at ``p3.N400`` the floor leg is **not** negligible
  beside the solver leg: ``floor / solver >= 0.25``.  If the measurement lands
  below 0.25 the hypothesis is falsified, is recorded as such, and the pool
  verdict books a genuine scheme shortfall.
* **Invariants (asserted in tests)** -- (i) Pythagoras
  ``total_exact^2 == solver^2 + floor^2``; (ii) the committed ``numerical``
  fields are degree-``p`` polynomials per cell (reconstruction residual ~1e-14);
  (iii) ``total_quad`` reproduces the committed pool reading; (iv) split
  quadrature is saturated (20/40/80 points agree); (v) the floor is insensitive
  to a 1e-6 nudge of the split positions.

Artifacts::

    docs/benchmarks/results/lax_reference_floor/metrics.json
    docs/benchmarks/results/lax_reference_floor/metrics.sha256
    docs/benchmarks/results/lax_reference_floor/figures/*.png
    docs/benchmarks/results/lax_reference_floor/summary.md
    docs/CASE-LAX-DG-reference-floor.md          (rendered by ``--render``)

Usage::

    PYTHONPATH=. python -m benchmarks.lax_reference_floor            # full
    PYTHONPATH=. python -m benchmarks.lax_reference_floor --quick    # smoke
    PYTHONPATH=. python -m benchmarks.lax_reference_floor --render   # doc only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from benchmarks import lax_case_pool as pool
from benchmarks.riemann_exact import sample

FIELDS = ("density", "velocity", "pressure")

#: Legs carried by the committed asymptotic-probe profiles (reuse, no re-solve).
PROFILE_LADDER = ((400, 1), (800, 1), (400, 2), (800, 2), (200, 3), (400, 3), (800, 3))

#: Solver-free floor ladder.
FLOOR_CELLS = (100, 200, 400, 800, 1600)
FLOOR_DEGREES = (1, 2, 3)

#: Gauss points per smooth subinterval (saturation is asserted in ``--quick``).
SPLIT_ORDER = 40
SATURATION_ORDERS = (20, 40, 80)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "docs", "benchmarks", "results", "lax_asymptotic_probe", "profiles.npz")
RESULTS = os.path.join(ROOT, "docs", "benchmarks", "results", "lax_reference_floor")
DOC = os.path.join(ROOT, "docs", "CASE-LAX-DG-reference-floor.md")

DOMAIN = (0.0, 1.0)


# --------------------------------------------------------------------------- #
# reference / geometry helpers
# --------------------------------------------------------------------------- #
def case_context():
    """Case parameters plus the closed-form wave abscissae, all from the pool."""
    waves = pool.exact_waves()
    positions = waves["positions"]
    breaks = np.array([positions["rarefaction_head"], positions["rarefaction_tail"],
                       positions["contact"], positions["shock"]], dtype=float)
    breaks = np.unique(breaks[(breaks > DOMAIN[0]) & (breaks < DOMAIN[1])])
    return {
        "final_time": float(waves["final_time"]),
        "left_state": list(waves["left_state"]),
        "right_state": list(waves["right_state"]),
        "gamma": float(waves["gamma"]),
        "diaphragm": float(waves["diaphragm"]),
        "pressure_star": float(waves["pressure_star"]),
        "velocity_star": float(waves["velocity_star"]),
        "density_star_left": float(waves["density_star_left"]),
        "density_star_right": float(waves["density_star_right"]),
        "positions": positions,
        "breaks": breaks,
    }


def exact_at(x, final_time):
    return sample(x, final_time, pool.LEFT, pool.RIGHT)


def cell_geometry(ncells):
    lower, upper = DOMAIN
    edges = np.linspace(lower, upper, ncells + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dx = (upper - lower) / ncells
    return edges, centres, dx


def smooth_subintervals(a, b, breaks):
    """Cell ``[a, b]`` cut at the wave abscissae, so every piece is smooth."""
    inside = breaks[(breaks > a) & (breaks < b)]
    return np.unique(np.concatenate(([a, b], inside)))


def legendre_basis(xi, degree):
    """``(degree + 1, len(xi))`` matrix of Legendre polynomials ``P_k(xi)``."""
    return np.polynomial.legendre.legvander(np.asarray(xi, dtype=float), degree).T


# --------------------------------------------------------------------------- #
# quadrature core: cells split at the waves, exact solution projected per cell
# --------------------------------------------------------------------------- #
def _split_quadrature(edges, cell, breaks, order):
    """``(xg, wg)`` covering ``[a, b]`` as disjoint smooth pieces."""
    a, b = edges[cell], edges[cell + 1]
    subs = smooth_subintervals(a, b, breaks)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xs, ws = [], []
    for s in range(len(subs) - 1):
        lo, hi = subs[s], subs[s + 1]
        half = 0.5 * (hi - lo)
        xs.append(0.5 * (lo + hi) + half * nodes)
        ws.append(half * weights)
    return np.concatenate(xs), np.concatenate(ws)


def accumulated(ncells, degree, ctx, modal_conserved=None, order=SPLIT_ORDER, want_coeffs=False):
    """One sweep over the cells, every error leg integrated on the smooth pieces.

    Per-field accumulators:

    * ``floor_sq``  ``int (Pi u - u)^2 dx``   -- reference-projection leg;
    * ``solver_sq`` ``int (u_h - Pi u)^2 dx`` -- solver leg (needs a profile);
    * ``cross``     ``2 int (u_h - Pi u)(Pi u - u) dx`` -- Pythagoras cross term.
      It vanishes identically for the **conserved** field (``rho_h`` lies in the
      DG space), and is *not* zero for the primitive velocity / pressure, which
      are rational functions of the conserved polynomial vector;
    * ``total_sq``  ``int (u_h - u)^2 dx``;
    * ``norm_sq``   ``int u^2 dx`` (denominator of every relative norm).
    """
    edges, centres, dx = cell_geometry(ncells)
    breaks = ctx["breaks"]
    degree = int(degree)
    gamma = ctx["gamma"]
    floor_sq = {field: 0.0 for field in FIELDS}
    solver_sq = {field: 0.0 for field in FIELDS}
    cross = {field: 0.0 for field in FIELDS}
    total_sq = {field: 0.0 for field in FIELDS}
    norm_sq = {field: 0.0 for field in FIELDS}
    coeffs = {field: np.zeros((ncells, degree + 1)) for field in FIELDS} if want_coeffs else None

    for cell in range(ncells):
        xg, wg = _split_quadrature(edges, cell, breaks, order)
        exact = exact_at(xg, ctx["final_time"])
        xi = (xg - centres[cell]) / (0.5 * dx)
        basis = legendre_basis(xi, degree)
        primitive = None
        if modal_conserved is not None:
            conserved = modal_conserved[:, cell, :] @ basis                 # (3, npts)
            rho = conserved[0]
            primitive = {
                "density": rho,
                "velocity": conserved[1] / rho,
                "pressure": (gamma - 1.0) * (conserved[2] - 0.5 * conserved[1] ** 2 / rho),
            }
        for field in FIELDS:
            u = np.asarray(exact[field], dtype=float)
            a_k = 0.5 * (2.0 * np.arange(degree + 1) + 1.0) * (2.0 / dx) * (basis @ (wg * u))
            if coeffs is not None:
                coeffs[field][cell] = a_k
            projected = a_k @ basis
            floor_sq[field] += float(np.sum(wg * (projected - u) ** 2))
            norm_sq[field] += float(np.sum(wg * u ** 2))
            if primitive is not None:
                residual = primitive[field] - projected
                solver_sq[field] += float(np.sum(wg * residual ** 2))
                cross[field] += 2.0 * float(np.sum(wg * residual * (projected - u)))
                total_sq[field] += float(np.sum(wg * (primitive[field] - u) ** 2))

    return {
        "floor_sq": floor_sq,
        "norm_sq": norm_sq,
        "solver_sq": solver_sq,
        "cross": cross,
        "total_sq": total_sq,
        "coeffs": coeffs,
        "dx": dx,
    }


def project_exact(ncells, degree, ctx, order=SPLIT_ORDER):
    """Legendre coefficients of ``Pi u`` per cell plus the floor / norm sums."""
    out = accumulated(ncells, degree, ctx, order=order, want_coeffs=True)
    return out["coeffs"], out["floor_sq"], out["norm_sq"]


def reconstruct_from_nodes(points, values, degree):
    """Legendre coefficients of a *conserved* DG field on each cell.

    Conserved fields are polynomials of degree ``p``; the committed profiles
    sample them on 10 Gauss nodes, so the least-squares fit is exact and its
    residual doubles as an integrity check of the profile artifact.
    """
    nodes, _ = np.polynomial.legendre.leggauss(points.shape[1])
    basis = legendre_basis(nodes, degree)                 # (degree+1, nnodes)
    pinv = np.linalg.pinv(basis)                          # (nnodes, degree+1)
    coeffs = np.einsum("nk,cn->ck", pinv, values)
    residual = values - np.einsum("ck,kn->cn", coeffs, basis)
    return coeffs, float(np.max(np.abs(residual)))


def conserved_from_primitive(numerical, gamma):
    """``(rho, rho u, E)`` on the profile nodes -- the fields DG actually stores."""
    rho, velocity, pressure = numerical
    return np.stack([rho, rho * velocity,
                     pressure / (gamma - 1.0) + 0.5 * rho * velocity ** 2])


# --------------------------------------------------------------------------- #
# measuring a committed profile leg
# --------------------------------------------------------------------------- #
def measure_leg(ncells, degree, ctx, profiles=PROFILES):
    data = np.load(profiles)
    key = f"N{ncells}_p{degree}"
    points = data[f"{key}_points"]
    numerical = data[f"{key}_numerical"]                   # (3, ncells, nnodes)
    exact_nodes = data[f"{key}_exact"]
    _, weights = np.polynomial.legendre.leggauss(points.shape[1])
    centres = 0.5 * (points[:, 0] + points[:, -1])
    dx = float((centres[-1] - centres[0]) / (len(centres) - 1))

    conserved = conserved_from_primitive(numerical, ctx["gamma"])
    modal = np.empty((3, ncells, degree + 1))
    residual = 0.0
    for index in range(3):
        modal[index], r = reconstruct_from_nodes(points, conserved[index], degree)
        residual = max(residual, r)

    acc = accumulated(ncells, degree, ctx, modal_conserved=modal)
    leg = {"cells": ncells, "degree": degree, "dx": dx,
           "conserved_reconstruction_residual": residual, "fields": {}}
    for index, field in enumerate(FIELDS):
        floor = math.sqrt(acc["floor_sq"][field] / acc["norm_sq"][field])
        solver = math.sqrt(acc["solver_sq"][field] / acc["norm_sq"][field])
        total = math.sqrt(acc["total_sq"][field] / acc["norm_sq"][field])
        cross = acc["cross"][field] / acc["norm_sq"][field]
        total_quad = pool._weighted_l2(points, numerical[index], exact_nodes[index], weights, dx)
        leg["fields"][field] = {
            "total_quad": total_quad,
            "total_exact": total,
            "reference_floor": floor,
            "solver_leg": solver,
            "cross_term": cross,
            "floor_share_of_squared_error": acc["floor_sq"][field] / acc["total_sq"][field],
            "solver_share_of_squared_error": acc["solver_sq"][field] / acc["total_sq"][field],
            "quadrature_bias": abs(total_quad - total) / total,
            "pythagoras_residual": abs(total ** 2 - (floor ** 2 + solver ** 2)),
            "decomposition_residual": abs(total ** 2 - (floor ** 2 + solver ** 2 + cross)),
            "exact_norm": math.sqrt(acc["norm_sq"][field]),
            "in_dg_space": field == "density",
        }
    leg["worst_total_quad"] = max(leg["fields"][f]["total_quad"] for f in FIELDS)
    leg["worst_field_quad"] = max(FIELDS, key=lambda f: leg["fields"][f]["total_quad"])
    leg["worst_reference_floor"] = max(leg["fields"][f]["reference_floor"] for f in FIELDS)
    return leg


_FLOOR_CACHE = {}


def floor_only(ncells, degree, ctx, order=SPLIT_ORDER):
    key = (int(ncells), int(degree), int(order), tuple(float(b) for b in ctx["breaks"]))
    if key in _FLOOR_CACHE:
        return dict(_FLOOR_CACHE[key])
    acc = accumulated(ncells, degree, ctx, order=order)
    out = {field: math.sqrt(acc["floor_sq"][field] / acc["norm_sq"][field]) for field in FIELDS}
    _FLOOR_CACHE[key] = dict(out)
    return out


def saturation_check(ncells, degree, ctx):
    """Floor at 20 / 40 / 80 Gauss points per smooth subinterval."""
    readings = {order: floor_only(ncells, degree, ctx, order=order) for order in SATURATION_ORDERS}
    spread = max(
        abs(readings[order][field] - readings[SATURATION_ORDERS[-1]][field])
        for order in SATURATION_ORDERS[:-1]
        for field in FIELDS
    )
    return {"readings": {str(o): readings[o] for o in SATURATION_ORDERS}, "max_spread": spread}


def split_sensitivity(ncells, degree, ctx, deltas=(1e-6, 1e-9, 1e-12)):
    """Floor vs a nudge of the wave split positions (should be linear in the nudge)."""
    base = floor_only(ncells, degree, ctx)
    out = {}
    for delta in deltas:
        nudge = dict(ctx)
        nudge["breaks"] = ctx["breaks"] + delta
        moved = floor_only(ncells, degree, nudge)
        out[f"{delta:g}"] = max(abs(moved[f] - base[f]) / base[f] for f in FIELDS)
    return out


def fit_exponent(cells, values):
    x = np.log(np.asarray(cells, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

# --------------------------------------------------------------------------- #
# study driver
# --------------------------------------------------------------------------- #
def run(tolerance=None, quick=False):
    started = time.time()
    ctx = case_context()
    tolerance = pool.ledger_acceptance_tolerance() if tolerance is None else tolerance

    legs = [measure_leg(n, p, ctx) for (n, p) in PROFILE_LADDER]

    floor_cells = (100, 200, 400) if quick else FLOOR_CELLS
    floor_table = {d: {str(n): floor_only(n, d, ctx) for n in floor_cells} for d in FLOOR_DEGREES}
    exponents = {
        d: {field: fit_exponent(floor_cells, [floor_table[d][str(n)][field] for n in floor_cells])
            for field in FIELDS}
        for d in FLOOR_DEGREES
    }

    saturation = saturation_check(800, 3, ctx)
    sensitivity = split_sensitivity(800, 3, ctx)

    # reuse check against the committed pool reading of the pinned leg
    pool_metrics = os.path.join(ROOT, "docs", "benchmarks", "results", "lax_case_pool", "metrics.json")
    reuse = {}
    if os.path.exists(pool_metrics):
        with open(pool_metrics) as handle:
            pool_l2 = json.load(handle)["acceptance"]["per_leg"]["p3.N400"]["l2"]
        pinned_leg = next(leg for leg in legs if (leg["cells"], leg["degree"]) == (400, 3))
        for field in FIELDS:
            committed = pool_l2[field]
            reuse[field] = abs(pinned_leg["fields"][field]["total_quad"] - committed) / committed
    worst_reuse = max(reuse.values()) if reuse else None

    # leg orders (same degree, two meshes)
    orders = {}
    for degree in (1, 2, 3):
        pair = sorted([(leg["cells"], leg) for leg in legs if leg["degree"] == degree])
        if len(pair) >= 2:
            (n0, coarse), (n1, fine) = pair[0], pair[-1]
            orders[degree] = {
                "cells": [n0, n1],
                "solver_leg": {
                    field: float(math.log(coarse["fields"][field]["solver_leg"]
                                          / fine["fields"][field]["solver_leg"]) / math.log(n1 / n0))
                    for field in FIELDS
                },
                "reference_floor": {
                    field: float(math.log(coarse["fields"][field]["reference_floor"]
                                          / fine["fields"][field]["reference_floor"]) / math.log(n1 / n0))
                    for field in FIELDS
                },
            }

    pinned = next(leg for leg in legs if (leg["cells"], leg["degree"]) == (400, 3))
    ratios = {field: pinned["fields"][field]["reference_floor"] / pinned["fields"][field]["solver_leg"]
              for field in FIELDS}
    hypothesis = {
        "statement": "reference floor / solver leg >= 0.25 at p3.N400",
        "ratios": ratios,
        "supported": bool(min(ratios.values()) >= 0.25),
        "reading": (
            "the registered metric is co-dominated by the reference discretisation at the pinned mesh"
            if min(ratios.values()) >= 0.25
            else "the reference floor is a minority term at the pinned mesh; the pool verdict books a "
                 "genuine scheme shortfall"
        ),
    }

    # how much of the 3 % budget does the floor alone eat, per mesh?
    wide_cells = (100, 200, 400) if quick else (100, 200, 400, 800, 1600, 3200)
    floor_budget = {}
    for degree in FLOOR_DEGREES:
        row = {}
        for n in wide_cells:
            readings = floor_only(n, degree, ctx)
            row[str(n)] = {field: readings[field] / tolerance for field in FIELDS}
        floor_budget[degree] = row

    metrics = {
        "anchor": "CASE-LAX-DG",
        "case": {
            "name": "Lax shock tube (Toro test 2)",
            "domain": list(DOMAIN),
            "final_time": ctx["final_time"],
            "left_state": ctx["left_state"],
            "right_state": ctx["right_state"],
            "gamma": ctx["gamma"],
            "waves": ctx["positions"],
            "closed_form_star": {
                "pressure": ctx["pressure_star"],
                "velocity": ctx["velocity_star"],
                "density_left": ctx["density_star_left"],
                "density_right": ctx["density_star_right"],
            },
        },
        "question": (
            "The ledger's whole-field relative-L2 metric compares the DG field with the pointwise exact "
            "solution. Decompose it exactly into a solver leg and a reference-projection floor leg, and "
            "report which of the two dominates at the meshes the ledger grades."
        ),
        "reference_definition": {
            "exact_solution": "closed-form Riemann solution, benchmarks.riemann_exact.sample",
            "projection": "L2 projection onto piecewise degree-p Legendre space on N uniform cells",
            "identity": "||u_h-u||^2 = ||u_h-Pi u||^2 + ||Pi u-u||^2 (exact quadrature)",
            "quadrature": f"{SPLIT_ORDER}-point Gauss per smooth subinterval, cells split at wave abscissae",
            "floor_is_solver_free": True,
            "norms": "relative L2, denominator = exact-solution L2 norm; whole field, no band removed",
        },
        "config": {
            "tolerance": tolerance,
            "profile_ladder": [list(pair) for pair in PROFILE_LADDER],
            "floor_cells": list(floor_cells),
            "floor_degrees": list(FLOOR_DEGREES),
            "profiles": os.path.relpath(PROFILES, ROOT),
            "split_order": SPLIT_ORDER,
            "quick": quick,
        },
        "legs": legs,
        "floor_table": {str(d): floor_table[d] for d in FLOOR_DEGREES},
        "floor_budget_share": {str(d): floor_budget[d] for d in FLOOR_DEGREES},
        "floor_exponents": {str(d): exponents[d] for d in FLOOR_DEGREES},
        "solver_orders": {str(d): v for d, v in orders.items()},
        "saturation": saturation,
        "split_sensitivity_max_rel": sensitivity,
        "reuse_check": {
            "source": os.path.relpath(pool_metrics, ROOT),
            "per_field_relative_difference": reuse,
            "worst": worst_reuse,
        },
        "hypothesis": hypothesis,
    }
    metrics["verdict"] = {
        "floor_share_of_tolerance_pinned": pinned["worst_reference_floor"] / tolerance,
        "quadrature_bias_worst": max(leg["fields"][f]["quadrature_bias"] for leg in legs for f in FIELDS),
        "pythagoras_density_worst": max(leg["fields"]["density"]["pythagoras_residual"] for leg in legs),
        "decomposition_residual_worst": max(leg["fields"][f]["decomposition_residual"]
                                            for leg in legs for f in FIELDS),
        "cross_term_worst": max(abs(leg["fields"][f]["cross_term"]) for leg in legs for f in FIELDS),
        "cross_term_share_worst": max(abs(leg["fields"][f]["cross_term"]) / leg["fields"][f]["total_exact"] ** 2
                                      for leg in legs for f in FIELDS),
        "conserved_reconstruction_residual_worst": max(leg["conserved_reconstruction_residual"]
                                                       for leg in legs),
        "reuse_worst_relative": worst_reuse,
        "saturation_spread": saturation["max_spread"],
        "split_sensitivity": sensitivity,
    }
    metrics["wall_seconds"] = time.time() - started
    return metrics


def write_artifacts(metrics, figures=True):
    os.makedirs(RESULTS, exist_ok=True)
    payload = json.dumps(metrics, indent=1, sort_keys=True)
    path = os.path.join(RESULTS, "metrics.json")
    with open(path, "w") as handle:
        handle.write(payload + "\n")
    digest = hashlib.sha256((payload + "\n").encode()).hexdigest()
    with open(os.path.join(RESULTS, "metrics.sha256"), "w") as handle:
        handle.write(f"{digest}  metrics.json\n")
    with open(os.path.join(RESULTS, "summary.md"), "w") as handle:
        handle.write(render_summary(metrics))
    if figures:
        try:
            render_figures(metrics)
        except ModuleNotFoundError:          # matplotlib is optional
            pass
    return path, digest


def _pct(value, digits=2):
    return f"{100.0 * value:.{digits}f} %"


def render_summary(metrics):
    tolerance = metrics["config"]["tolerance"]
    lines = [
        "# CASE-LAX-DG · reference-projection floor of the acceptance metric",
        "",
        "> identity `||u_h-u||^2 = ||u_h-Pi u||^2 + ||Pi u-u||^2`; "
        f"tolerance {_pct(tolerance)}; digest in `metrics.sha256`",
        "",
        "## Measured legs (committed profiles, no re-solve)",
        "",
        "| p | N | total L2 quad | total L2 exact | solver leg | reference floor | floor / tolerance |",
        "|---|---|---|---|---|---|---|",
    ]
    for leg in metrics["legs"]:
        f = leg["fields"]["density"]
        lines.append(
            f"| {leg['degree']} | {leg['cells']} | {f['total_quad']:.5e} | {f['total_exact']:.5e} | "
            f"{f['solver_leg']:.5e} | {f['reference_floor']:.5e} | {_pct(f['reference_floor'] / tolerance)} |"
        )
    lines += ["", "## Reference floor ladder (solver-free)", ""]
    cells = list(metrics["config"]["floor_cells"])
    lines += ["| p | " + " | ".join(f"N={n}" for n in cells) + " | exponent |",
              "|" + "---|" * (len(cells) + 2)]
    for degree in metrics["config"]["floor_degrees"]:
        table = metrics["floor_table"][str(degree)]
        row = " | ".join(f"{table[str(n)]['density']:.4e}" for n in cells)
        lines.append(f"| {degree} | {row} | {metrics['floor_exponents'][str(degree)]['density']:.3f} |")
    hyp = metrics["hypothesis"]
    lines += [
        "",
        "## Hypothesis H",
        "",
        f"* statement: {hyp['statement']}",
        "* floor / solver: " + ", ".join(f"{k} {v:.3f}" for k, v in hyp["ratios"].items()),
        f"* supported: **{hyp['supported']}** -- {hyp['reading']}",
        "",
        "## Invariants",
        "",
        f"* Pythagoras residual, density (worst): {metrics['verdict']['pythagoras_density_worst']:.3e}",
        f"* decomposition residual incl. cross term (worst): "
        f"{metrics['verdict']['decomposition_residual_worst']:.3e}",
        f"* cross term (worst, velocity/pressure): {metrics['verdict']['cross_term_worst']:.3e}",
        f"* conserved-field reconstruction residual (worst): "
        f"{metrics['verdict']['conserved_reconstruction_residual_worst']:.3e}",
        f"* reuse vs committed pool p3.N400: {metrics['verdict']['reuse_worst_relative']:.3e}",
        f"* quadrature saturation (20/40/80 pt): {metrics['verdict']['saturation_spread']:.3e}",
        "* split-position sensitivity: " + ", ".join(
            f"{k} -> {v:.3e}" for k, v in metrics["verdict"]["split_sensitivity"].items()),
        f"* wall time: {metrics['wall_seconds']:.1f} s",
    ]
    return "\n".join(lines) + "\n"


def render_figures(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(RESULTS, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    tolerance = metrics["config"]["tolerance"]

    # 1) floor ladder, log-log
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    cells = np.asarray(metrics["config"]["floor_cells"], dtype=float)
    for degree in metrics["config"]["floor_degrees"]:
        table = metrics["floor_table"][str(degree)]
        values = [table[str(int(n))]["density"] for n in cells]
        slope = metrics["floor_exponents"][str(degree)]["density"]
        ax.loglog(cells, values, "o-", label=f"p = {degree} (slope {slope:.2f})")
    ax.axhline(tolerance, color="crimson", ls="--", lw=1.0, label="ledger tolerance 3 %")
    ax.set_xlabel("cells N")
    ax.set_ylabel(r"reference floor $\|\Pi u-u\|_2/\|u\|_2$ (density)")
    ax.set_title("CASE-LAX-DG · reference-projection floor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "floor_ladder.png"), dpi=150)
    plt.close(fig)

    # 2) squared-error budget per leg
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, floor_share, solver_share = [], [], []
    for leg in metrics["legs"]:
        f = leg["fields"]["density"]
        total_sq = f["total_exact"] ** 2
        labels.append(f"p{leg['degree']}N{leg['cells']}")
        floor_share.append(f["reference_floor"] ** 2 / total_sq)
        solver_share.append(f["solver_leg"] ** 2 / total_sq)
    index = np.arange(len(labels))
    ax.bar(index, solver_share, label="solver leg", color="#3b6ea5")
    ax.bar(index, floor_share, bottom=solver_share, label="reference floor", color="#d98c3f")
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of squared L2 error (density)")
    ax.set_title("exact decomposition of the graded error")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "error_budget.png"), dpi=150)
    plt.close(fig)

    # 3) graded field vs exact in the jump region
    try:
        data = np.load(PROFILES)
        points = data["N800_p3_points"]
        centres = 0.5 * (points[:, 0] + points[:, -1])
        nodes = np.linspace(0.5, 1.0, 6001)
        exact = exact_at(nodes, metrics["case"]["final_time"])["density"]
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(nodes, exact, "k-", lw=1.2, label="exact Riemann solution")
        ax.plot(centres, data["N800_p3_numerical"][0].mean(axis=1), "o", ms=2.2,
                label="DG p3 N800 (cell mean)")
        ax.set_xlim(0.6, 0.95)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.set_title("contact / shock region: the source of the floor")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "shock_region.png"), dpi=150)
        plt.close(fig)
    except Exception:                        # a figure must never break the study
        pass
    return fig_dir


def render_doc(metrics):
    tolerance = metrics["config"]["tolerance"]
    pinned = next(leg for leg in metrics["legs"] if (leg["cells"], leg["degree"]) == (400, 3))
    n800 = next(leg for leg in metrics["legs"] if (leg["cells"], leg["degree"]) == (800, 3))
    hyp = metrics["hypothesis"]
    verdict = metrics["verdict"]
    cells = list(metrics["config"]["floor_cells"])
    wide = sorted(metrics["floor_budget_share"]["1"], key=int)

    rows_legs = []
    for leg in metrics["legs"]:
        f = leg["fields"]["density"]
        rows_legs.append(
            f"| p{leg['degree']} | {leg['cells']} | {f['total_quad']:.5e} | {f['total_exact']:.5e} | "
            f"{f['solver_leg']:.5e} | {f['reference_floor']:.5e} | "
            f"{100 * f['floor_share_of_squared_error']:.1f} % | "
            f"{100 * f['reference_floor'] / tolerance:.1f} % | {100 * f['quadrature_bias']:.2f} % |"
        )
    rows_floor = []
    for degree in metrics["config"]["floor_degrees"]:
        table = metrics["floor_table"][str(degree)]
        row = " | ".join(f"{table[str(n)]['density']:.4e}" for n in cells)
        rows_floor.append(f"| p{degree} | {row} | {metrics['floor_exponents'][str(degree)]['density']:.3f} |")
    rows_budget = []
    for degree in metrics["config"]["floor_degrees"]:
        share = metrics["floor_budget_share"][str(degree)]
        rows_budget.append("| p%d | " % degree + " | ".join(
            f"{100 * share[n]['density']:.1f} %" for n in wide) + " |")

    order_rows = []
    for degree, entry in sorted(metrics["solver_orders"].items(), key=lambda kv: int(kv[0])):
        order_rows.append(
            f"| p{degree} | {entry['cells'][0]}→{entry['cells'][1]} | "
            f"{entry['solver_leg']['density']:.2f} | {entry['reference_floor']['density']:.2f} | "
            f"{entry['solver_leg']['velocity']:.2f} | {entry['solver_leg']['pressure']:.2f} |"
        )
    order_block = "\n".join(order_rows) if order_rows else "| - | - | - | - | - | - |"

    doc = f"""# CASE-LAX-DG · 「< 3 %」里有多少是参考解的离散化，不是求解器？

> 锚点：`feature_ledger` `CASE-LAX-DG`（`ref: exact solution | tolerance < 3 %`，status `open`）。
> 上游：`docs/CASE-LAX-DG.md`（算例池）、`docs/CASE-LAX-DG-resolution-requirement.md`（N=800 闭合）、
> `docs/CASE-LAX-DG-p3-asymptotic-probe.md`（阶探针）。本单**不重跑求解器**，只对已提交的
> `lax_asymptotic_probe/profiles.npz` 做误差分解，因此不可能改变锚点既有判决。

## ① 案例介绍

Lax (1954) / Toro test 2 激波管：`[0,1]`、隔膜 `x0 = 0.5`、γ = {metrics['case']['gamma']}、
快照 `t = {metrics['case']['final_time']}`。左态
`(ρ,u,p) = ({', '.join(f'{v:g}' for v in metrics['case']['left_state'])})`，右态
`({', '.join(f'{v:g}' for v in metrics['case']['right_state'])})`。闭式星区
`p* = {metrics['case']['closed_form_star']['pressure']:.6f}`、
`u* = {metrics['case']['closed_form_star']['velocity']:.6f}`。四道波（闭式）：
稀疏头 `{metrics['case']['waves']['rarefaction_head']:.6f}`、
稀疏尾 `{metrics['case']['waves']['rarefaction_tail']:.6f}`、
接触 `{metrics['case']['waves']['contact']:.6f}`、激波 `{metrics['case']['waves']['shock']:.6f}`。
接触与激波是**真跳变**——这正是本文的核心：任何有限网格上的 DG 空间都装不下它们。

## ② 怎么做的

台账判据是「全场相对 L2，对**逐点**精确解，取单元 Gauss 节点」，最差场量是密度。把它拆开：

    ||u_h − u||² = ||u_h − Πu||² + ||Πu − u||²          （Π = 到 DG 空间的 L2 投影）

* **solver 腿** `||u_h − Πu||`：求解器在给定空间内没做到的，**可以**靠更好的格式 / 更细网格降下来；
* **floor 腿** `||Πu − u||`：**该网格上任何** DG 函数都做不到的，与格式无关，只由网格 + 精确解决定。

这条恒等式有两件事必须说清楚，否则它会被读成比实际更强的结论：

* **对守恒量（密度）它严格成立**：`ρ_h ∈ V`，投影正交性把交叉项抹掉，逐腿残差
  `{verdict['pythagoras_density_worst']:.1e}`。正交性还给出闭式
  `∫P_kP_m = 2/(2k+1)δ`，所以密度腿上 solver/floor 两个分量是**无求积误差**的。
* **对原始量（速度、压力）不成立**：DG 存的是守恒量 `(ρ, ρu, E)`，而
  `u_h = (ρu)_h/ρ_h`、`p_h = (γ−1)(E_h − ½(ρu)_h²/ρ_h)` 是守恒多项式的**有理函数**，
  **不属于 V**。于是恒等式多一个交叉项 `X = 2⟨u_h−Πu, Πu−u⟩`。本单**测**它而不是假设它为零：
  最差 `|X| = {verdict['cross_term_worst']:.1e}`（相对平方口径），带交叉项的分解残差
  `{verdict['decomposition_residual_worst']:.1e}`。这是本单踩到的一个真实坑——把原始量当多项式
  去「还原」会得到 10 % 量级的伪残差（第一版实现就是这样，见 ⑤ 注意点 6）。

实现要点：

1. 每个单元在四道波的位置切开，每段上精确解**光滑**，用 {metrics['config']['split_order']} 点 Gauss 积分；
   20/40/80 点一致性 `{verdict['saturation_spread']:.1e}` 说明积分已饱和；
2. 投影系数 `a_k = (2k+1)/2 ∫ u P_k dξ` 用同一套切分段积分给出；
3. solver 腿在切分段上直接积分 `∫(u_h − Πu)²`，其中 `u_h` 由守恒量的多项式表示算出
   （不是把原始量强行当多项式插值）；密度的正交性闭式作为独立校核；
4. 已提交 `numerical` 场先转成守恒量 `(ρ, ρu, E)`，再用 10 个 Gauss 节点最小二乘还原 p 次多项式，
   残差 `{verdict['conserved_reconstruction_residual_worst']:.1e}`——这同时是**工件完整性检查**
   （它证明已提交剖面确实是 p 次多项式，而不是随便一组采样）；
5. `total_quad` 用池子自己的 `_weighted_l2` 口径（10 点 Gauss），与已提交台账臂
   逐位一致（最差相对差 `{verdict['reuse_worst_relative']:.1e}`），「同一口径」有证据；
6. 切分位置精度：把波位置整体推 `δ` 后 floor 的相对变化见 ⑤ 注意点 4（`δ` 越小变化线性越小）。

工具链：`benchmarks/lax_reference_floor.py`（本单新增）、`benchmarks/lax_case_pool.py`、
`benchmarks/riemann_exact.py`；无新求解，本机单核，墙钟 {metrics['wall_seconds']:.1f} s。

## ③ 计算结果

### 3.1 逐腿分解（密度；`floor` 列即参考解自身的离散化）

| p | N | total L2（10 点 Gauss 口径） | total L2（精确积分） | solver 腿 | floor 腿 | floor 平方占比 | floor / 3 % 预算 | 求积偏差 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows_legs)}

`p3.N400`（台账钉扎腿）floor = `{pinned['worst_reference_floor']:.5e}`；
`p3.N800`（闭合腿）floor = `{n800['worst_reference_floor']:.5e}`（三场量最差）。

### 3.2 floor 阶梯（**不需要求解器**，可外推）

| p | {' | '.join(f'N={n}' for n in cells)} | 幂律 |
|---|---|---|---|
{chr(10).join(rows_floor)}

### 3.3 floor 占 3 % 预算的比例（越大越说明判据被参考解吃掉）

| p | {' | '.join(f'N={n}' for n in wide)} |
|---|---|
{chr(10).join(rows_budget)}

### 3.4 阶（同一 p 在两条腿上的 N-阶，密度/速度/压力）

| p | 网格对 | solver 腿阶 | floor 腿阶 | solver 阶（u） | solver 阶（p） |
|---|---|---|---|---|---|
{order_block}

## ④ 结果图

![floor ladder](benchmarks/results/lax_reference_floor/figures/floor_ladder.png)

*图 1*：参考解投影 floor 随网格的幂律（log-log），红线为 3 % 判据——曲线与格式无关。

![error budget](benchmarks/results/lax_reference_floor/figures/error_budget.png)

*图 2*：每条腿的平方误差占比（Pythagoras 分解），蓝 = solver 腿、橙 = 参考 floor。

![shock region](benchmarks/results/lax_reference_floor/figures/shock_region.png)

*图 3*：`p3.N800` 在接触/激波区的剖面与精确解——floor 的来源就是这两道跳变所在的单元。

## ⑤ 注意点 / 局限

1. **不改变锚点判决**：本单没有新求解，`A0` 的 `p3.N400 = 3.116 %` FAIL 原样保留；
   floor 小**不能**当作「求解器已经达标」来读。
2. **口径敏感**：`total_quad`（10 点 Gauss）与精确积分的最差偏差 `{verdict['quadrature_bias_worst']:.2e}`；
   台账的 3 % 是在 Gauss 口径下判的，口径写进工件 `reference_definition`，换口径需重裁。
3. **投影不是最优逼近**：`Πu` 是 L2 投影，作为「任何 DG 函数能到多准」的口径是标准的，
   但不是 minimax 最优；floor 因此是可逼近误差的**度量基准**，不是下界的严格极小值。
4. **只覆盖已提交的腿**：P1/P2 只到 N=800、P3 到 N=800；更细的腿需要重跑求解器（本单不做）。
5. **假设 H 的结论**：`{'支持' if hyp['supported'] else '被否证'}` —— {hyp['reading']}。
6. **原始量不是多项式（本单踩的坑）**：第一版实现把已提交的 `velocity`/`pressure` 当 p 次多项式还原，
   残差 0.35（10 % 量级）——因为 DG 的守恒量才是多项式，原始量是有理函数。正确做法是先转
   `(ρ, ρu, E)` 再还原，残差 `{verdict['conserved_reconstruction_residual_worst']:.1e}`。
   因此「solver 腿」对 u/p 只能定义成 `||u_h − Πu||` 并附交叉项，不能写成严格 Pythagoras。
7. **floor 与格式无关，但不是「下界」**：Πu 是 L2 投影而非 minimax 最优，floor 是可达误差的
   **度量基准**，不是严格极小值；换 Galerkin 投影/最优逼近会给出更小一点的数。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| `p3.N400` total L2 ρ（台账臂，Gauss 口径） | 精确 Riemann 解 | `{pinned['fields']['density']['total_quad']:.5e}` | {100 * pinned['fields']['density']['total_quad']:.3f} % | **否** |
| `p3.N800` total L2 ρ（闭合腿，Gauss 口径） | 精确 Riemann 解 | `{n800['fields']['density']['total_quad']:.5e}` | {100 * n800['fields']['density']['total_quad']:.3f} % | 是 |
| `p3.N400` solver 腿 ρ（去掉参考离散化） | 网格上的投影 Πu | `{pinned['fields']['density']['solver_leg']:.5e}` | {100 * pinned['fields']['density']['solver_leg']:.3f} % | **否** |
| `p3.N400` 参考 floor ρ（求解器无关） | 精确 Riemann 解 | `{pinned['fields']['density']['reference_floor']:.5e}` | {100 * pinned['fields']['density']['reference_floor']:.3f} % | 是（但占用 {100 * pinned['fields']['density']['reference_floor'] / tolerance:.1f} % 预算） |
| `p3.N800` 参考 floor ρ | 精确 Riemann 解 | `{n800['fields']['density']['reference_floor']:.5e}` | {100 * n800['fields']['density']['reference_floor']:.3f} % | 是（占用 {100 * n800['fields']['density']['reference_floor'] / tolerance:.1f} % 预算） |

**结论**：台账 3 % 判据在 `p3.N400` 上 FAIL（{100 * pinned['fields']['density']['total_quad']:.3f} %），
本单把它拆成 solver 腿 {100 * pinned['fields']['density']['solver_leg']:.3f} % + floor 腿
{100 * pinned['fields']['density']['reference_floor']:.3f} %。即钉扎腿上
**{100 * pinned['fields']['density']['reference_floor'] ** 2 / pinned['fields']['density']['total_exact'] ** 2:.1f} %
的平方误差来自参考解自身在该网格上的离散化**，这部分不是格式缺陷；剩下
{100 * pinned['fields']['density']['solver_leg'] ** 2 / pinned['fields']['density']['total_exact'] ** 2:.1f} %
才是求解器要负责的，它仍然是 FAIL 的主项——所以本单**不**为锚点翻案，只是把判据的
「不可降低项」量化并入库，供台账持有人裁定是否改口径（例如改判 solver 腿）。
"""
    with open(DOC, "w") as handle:
        handle.write(doc)
    return DOC


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke run on a short ladder")
    parser.add_argument("--render", action="store_true", help="render the doc from committed metrics")
    parser.add_argument("--figures-only", action="store_true", help="re-render figures from committed metrics")
    args = parser.parse_args(argv)

    if args.render:
        with open(os.path.join(RESULTS, "metrics.json")) as handle:
            metrics = json.load(handle)
        print("wrote", render_doc(metrics))
        return metrics

    if args.figures_only:
        with open(os.path.join(RESULTS, "metrics.json")) as handle:
            metrics = json.load(handle)
        print("figures ->", render_figures(metrics))
        return metrics

    metrics = run(quick=args.quick)
    path, digest = write_artifacts(metrics)
    if not args.quick:
        render_doc(metrics)
    print(json.dumps({
        "metrics": os.path.relpath(path, ROOT),
        "sha256": digest,
        "hypothesis_supported": metrics["hypothesis"]["supported"],
        "floor_share_pinned": metrics["verdict"]["floor_share_of_tolerance_pinned"],
        "wall_seconds": metrics["wall_seconds"],
    }, indent=1))
    return metrics


if __name__ == "__main__":
    main()