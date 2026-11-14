"""Lax shock tube (Toro test 2) -- DG case-pool driver with five machine gates.

Case identity
-------------
Lax (1954) / Toro (1999) test 2: two constant states separated by a diaphragm at
``x0 = 0.5`` in ``[0, 1]`` with ``gamma = 1.4``::

    (rho, u, p)_L = (0.445, 0.698, 3.528)     (rho, u, p)_R = (0.5, 0.0, 0.571)

snapshotted at ``t = 0.13``.  The exact solution contains four waves, left to
right: a left rarefaction (head -> tail), a contact discontinuity and a right
shock.  All four positions are analytic once the star state is known, so the
case carries reference values that no other solver has to produce.

Why this driver exists
----------------------
``benchmarks/lax_1d.py`` reports one single-resolution relative L2 number
against the exact Riemann solution, and ``tests/test_lax_1d.py`` re-checks a 3 %
bar.  That is a spot check, not a verification case: it pins neither the wave
positions, nor the star plateau, nor any convergence order.  This driver turns
the case into a *pool entry* carrying the five pre-registered gates of the
ledger anchor ``CASE-LAX-DG``:

===== ======================================================================  ==================================
gate  what is measured                                                      pre-registered tolerance
===== ======================================================================  ==================================
C1    four wave positions (rarefaction head/tail, contact, shock) located     ``|dx| / |x_exact - x0| < 2 %``
      in the finest DG leg, versus the analytic positions
C2    star state ``(rho*_L, u*, p*, rho*_R)`` read off DG plateau windows     ``< 5 %`` (relative)
C3    whole-field relative L1 convergence order over the resolution ladder    ``>= 1.0`` (literal)
R3    relative L1 order restricted to the *analytic* regions (a band of       ``>= 0.85 * (p + 1)``
      ``4 dx`` around each wave is removed)
R5    smooth entropy-wave design order at the two pre-registered wave speeds  order within ``3 %`` of ``p + 1``
      ``u = 1.0`` and ``u = 2.0`` (periodic domain, constant ``p``,
      ``rho = 1 + 0.2 sin(2 pi x)``)
===== ======================================================================  ==================================

The ledger wording for R5 ("within 3 % of 1.0/2.0") is read here as: the measured
order must sit within 3 % of the design order for the two pre-registered speeds
``u = 1.0`` and ``u = 2.0`` (design orders ``p+1`` = 2 for P1, 3 for P2).  The
reading is stated in ``docs/CASE-LAX-DG.md`` next to the measured numbers so a
reviewer can re-adjudicate without re-running anything.

Every number emitted here is computed from the solver or from the exact Riemann
solution; nothing is transcribed.

Artifacts (inside the worktree)::

    runs/lax_case_pool/metrics.json      measured values + gate verdicts
    runs/lax_case_pool/profiles.npz      finest-leg primitive fields + exact
    runs/lax_case_pool/figures/*.png     profiles, convergence, entropy wave
    runs/lax_case_pool/summary.md        tables rendered from metrics.json

Usage::

    python -m benchmarks.lax_case_pool             # full pool (a few CPU minutes)
    python -m benchmarks.lax_case_pool --quick     # reduced ladder, smoke only
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.riemann_exact import sample, star_state
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

# ------------------------------------------------------------------ case data
GAMMA = 1.4
LEFT = (0.445, 0.698, 3.528)
RIGHT = (0.5, 0.0, 0.571)
DIAPHRAGM = 0.5
FINAL_TIME = 0.13
DOMAIN = (0.0, 1.0)
CFL = 0.35

#: resolution ladder for C1/C2/C3/R3 -- every entry divides the finest grid (400)
LADDER = [(100, 1), (200, 1), (400, 1),
          (100, 2), (200, 2), (400, 2),
          (100, 3), (200, 3), (400, 3)]
QUICK_LADDER = [(100, 1), (200, 2), (400, 3)]

#: entropy-wave ladder for R5 (smooth flow, periodic domain)
ENTROPY_CELLS = (16, 32, 64, 128)
ENTROPY_DEGREES = (1, 2)
ENTROPY_SPEEDS = (1.0, 2.0)
ENTROPY_AMPLITUDE = 0.2
ENTROPY_PRESSURE = 1.0
ENTROPY_TIME = 0.25
ENTROPY_CFL = 0.2

#: pre-registered tolerances (never tuned after seeing the numbers)
TOL_C1 = 0.02
TOL_C2 = 0.05
MIN_C3_ORDER = 1.0
R3_ORDER_FRACTION = 0.85
R5_ORDER_FRACTION = 0.03
#: R3 region rule -- a fixed **physical** half width around every analytic
#: wave.  A rule scaled by ``dx`` is not nested across the ladder: the
#: retained measure grows with refinement (m16606 F1: 0.680 -> 0.920,
#: +35.3 %) and the measured "order" then mixes discretisation error with
#: region growth (P1 analytic-region error *increases* from N=64 to N=128,
#: order -0.945).  The width is fixed before the run, never tuned after.
R3_BAND_WIDTH = 0.02
#: retired rule, measured side by side so the change stays traceable
R3_LEGACY_BAND_CELLS = 4.0
R3_BAND_CELLS = R3_LEGACY_BAND_CELLS  # backward-compatible alias

#: CFL refinement ladder for the temporal-error separation.  m16606 listed
#: the split of the mixed space-time order as *not run*; the mesh is held
#: fixed and only ``dt`` is refined, so ``E(dt) = E0 + C dt**q``.
TIME_LEGS = ((100, 1), (200, 2), (200, 3))
TIME_CFL_DIVISORS = (1, 2, 4)
#: pre-registered bound on the temporal share of the measured error at the
#: production CFL; above it the Lax ladder verdict would be dt-polluted.
TOL_TIME_ERROR_SHARE = 0.05
#: fraction of the pressure drop used to place the rarefaction head/tail levels
RAREFACTION_LEVEL_FRACTION = 0.1

# ``runs/`` is never carried into ``main`` by the merge gate (0 tracked
# files), so a gate pinned there is vacuous on every clean checkout (see
# tools/audit_gate_vacuity.py).  Keep the case-pool record on the tracked
# ``docs/`` path, exactly as benchmarks/lax_region_band.py already does.
OUT = Path("docs/benchmarks/results/lax_case_pool")
FIG = OUT / "figures"


# --------------------------------------------------------------- exact solution
def exact_waves(left=LEFT, right=RIGHT, final_time=FINAL_TIME,
                diaphragm=DIAPHRAGM, gamma=GAMMA):
    """Analytic wave pattern of the Lax tube, all four positions included.

    The star state comes from the Newton solve of ``benchmarks.riemann_exact``;
    the wave positions are the Rankine--Hugoniot / isentropic relations
    evaluated in closed form here, so they are not read back from the solver.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    pressure, velocity = star_state(left, right)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    if pressure <= p_l:                                     # left rarefaction
        rho_star_l = rho_l * (pressure / p_l) ** (1.0 / gamma)
        a_star_l = a_l * (pressure / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        head_speed = u_l - a_l
        tail_speed = velocity - a_star_l
        left_kind = "rarefaction"
    else:                                                   # left shock
        rho_star_l = rho_l * ((pressure / p_l + (gamma - 1.0) / (gamma + 1.0))
                              / ((gamma - 1.0) / (gamma + 1.0) * pressure / p_l + 1.0))
        head_speed = tail_speed = u_l - a_l * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * pressure / p_l
            + (gamma - 1.0) / (2.0 * gamma))
        left_kind = "shock"
    if pressure > p_r:                                      # right shock
        ratio = pressure / p_r
        rho_star_r = rho_r * ((ratio + (gamma - 1.0) / (gamma + 1.0))
                              / ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0))
        shock_speed = u_r + a_r * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma))
        right_kind = "shock"
    else:                                                   # right rarefaction
        rho_star_r = rho_r * (pressure / p_r) ** (1.0 / gamma)
        shock_speed = u_r + a_r
        right_kind = "rarefaction"
    return {
        "left_state": list(left),
        "right_state": list(right),
        "gamma": gamma,
        "diaphragm": diaphragm,
        "final_time": final_time,
        "pressure_star": pressure,
        "velocity_star": velocity,
        "density_star_left": rho_star_l,
        "density_star_right": rho_star_r,
        "left_wave_kind": left_kind,
        "right_wave_kind": right_kind,
        "positions": {
            "rarefaction_head": diaphragm + head_speed * final_time,
            "rarefaction_tail": diaphragm + tail_speed * final_time,
            "contact": diaphragm + velocity * final_time,
            "shock": diaphragm + shock_speed * final_time,
        },
        "speeds": {"head": head_speed, "tail": tail_speed,
                   "contact": velocity, "shock": shock_speed},
    }


# ------------------------------------------------------------- DG field access
def modal_fields(x, coeff, quadrature_order=10):
    """Primitive variables on the Gauss nodes of every element."""
    values = coeff.detach().cpu().numpy()
    centres = x.detach().cpu().numpy()
    cells, modes, _ = values.shape
    dx = (centres[-1] - centres[0]) / (cells - 1)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    phi = []
    for k in range(modes):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        phi.append(np.polynomial.legendre.legval(nodes, polynomial))
    phi = np.stack(phi, axis=1)
    numerical = np.einsum("cmv,qm->cqv", values, phi)
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    primitive = {
        "density": numerical[..., 0],
        "velocity": numerical[..., 1] / numerical[..., 0],
        "pressure": 0.4 * (numerical[..., 2]
                           - 0.5 * numerical[..., 1] ** 2 / numerical[..., 0]),
    }
    return points, primitive, dx, weights


def _weighted_l1(points, values, exact_values, weights, dx, mask=None):
    """Relative L1 norm over the (optionally masked) quadrature points."""
    cell_weight = 0.5 * dx * weights[None, :] * np.ones_like(values)
    if mask is not None:
        cell_weight = cell_weight * mask
    numerator = np.sum(cell_weight * np.abs(values - exact_values))
    denominator = np.sum(cell_weight * np.abs(exact_values))
    return float(numerator / denominator)


def exact_reference(points, final_time=FINAL_TIME, left=LEFT, right=RIGHT,
                    diaphragm=DIAPHRAGM):
    return sample(points, final_time, left, right, diaphragm)


def locate_level(points, values, level, window=None):
    """First crossing of ``level`` by the piecewise-linear reconstruction.

    ``points``/``values`` are quadrature samples and the crossing is linearly
    interpolated between adjacent samples inside ``window``.  The same function
    is applied to the exact solution sampled on a very fine grid, so the
    reference position carries no bias from the level choice.
    """
    p = np.asarray(points).ravel()
    v = np.asarray(values).ravel()
    order = np.argsort(p)
    p, v = p[order], v[order]
    if window is not None:
        keep = (p >= window[0]) & (p <= window[1])
        p, v = p[keep], v[keep]
    s = v - level
    hits = np.nonzero(np.sign(s[:-1]) * np.sign(s[1:]) <= 0)[0]
    if hits.size == 0:
        return None
    i = int(hits[0])
    p0, p1 = p[i], p[i + 1]
    v0, v1 = v[i], v[i + 1]
    if v1 == v0:
        return 0.5 * (p0 + p1)
    return float(p0 + (level - v0) * (p1 - p0) / (v1 - v0))


def wave_positions(points, primitive, waves, dx=None):
    """Locate the four waves of the Lax tube in a numerical field.

    Pre-registered definitions:

    * rarefaction head -- pressure crossing at ``p_L - 0.1 (p_L - p*)``;
    * rarefaction tail -- pressure crossing at ``p* + 0.1 (p_L - p*)``;
    * contact         -- density half-level of ``(rho*_L, rho*_R)``, searched
                         between the exact tail and the exact shock;
    * shock           -- density half-level of ``(rho*_R, rho_R)``, searched to
                         the right of the exact contact.

    The search windows come from the *exact* pattern: they isolate a wave, they
    do not encode its position.  Each window is pulled back from its neighbour by
    ``margin = max(4 dx, 0.02)`` so that the (smeared, a few cells wide) contact
    cannot be mistaken for the shock and vice versa.
    """
    pos = waves["positions"]
    margin = max(4.0 * dx, 0.02) if dx is not None else 0.02
    p_l = waves["left_state"][2]
    p_star = waves["pressure_star"]
    drop = p_l - p_star
    head_level = p_l - RAREFACTION_LEVEL_FRACTION * drop
    tail_level = p_star + RAREFACTION_LEVEL_FRACTION * drop
    contact_level = 0.5 * (waves["density_star_left"] + waves["density_star_right"])
    shock_level = 0.5 * (waves["density_star_right"] + waves["right_state"][0])
    lo, hi = DOMAIN
    return {
        "rarefaction_head": locate_level(
            points, primitive["pressure"], head_level,
            window=(lo, pos["rarefaction_tail"] + 0.02)),
        "rarefaction_tail": locate_level(
            points, primitive["pressure"], tail_level,
            window=(pos["rarefaction_head"] - 0.02, pos["contact"] + 0.02)),
        "contact": locate_level(
            points, primitive["density"], contact_level,
            window=(pos["rarefaction_tail"] + margin, pos["shock"] - margin)),
        "shock": locate_level(
            points, primitive["density"], shock_level,
            window=(pos["contact"] + margin, hi - 0.5 * margin)),
    }


def plateau_values(points, primitive, waves, shrink=0.2):
    """Star-state plateaus: mean of each field over a shrunken wave window."""
    pos = waves["positions"]
    windows = {"star_left": (pos["rarefaction_tail"], pos["contact"]),
               "star_right": (pos["contact"], pos["shock"])}
    out = {}
    for name, (a, b) in windows.items():
        width = b - a
        lo, hi = a + shrink * width, b - shrink * width
        mask = (points >= lo) & (points <= hi)
        if not mask.any():
            raise RuntimeError(f"empty plateau window {name}")
        out[name] = {key: float(field[mask].mean()) for key, field in primitive.items()}
    return out


def exact_positions(waves, samples=400001):
    """Positions of the four waves *located from the exact field itself*."""
    grid = np.linspace(DOMAIN[0], DOMAIN[1], samples)
    exact = exact_reference(grid[None, :])
    primitive = {key: value[0] for key, value in exact.items()}
    return wave_positions(grid, primitive, waves, dx=grid[1] - grid[0])


# ---------------------------------------------------------------------- gates
def gate_c1(numerical, exact):
    rows = {}
    for name, x_exact in exact.items():
        x_num = numerical.get(name)
        if x_num is None:
            rows[name] = {"exact": float(x_exact), "numerical": None,
                          "relative_error": None, "status": "FAIL"}
            continue
        error = abs(x_num - x_exact) / abs(x_exact - DIAPHRAGM)
        rows[name] = {"exact": float(x_exact), "numerical": float(x_num),
                      "absolute_error": float(abs(x_num - x_exact)),
                      "relative_error": float(error),
                      "status": "PASS" if error < TOL_C1 else "FAIL"}
    waves = ("rarefaction_head", "rarefaction_tail", "contact", "shock")
    present = [rows[name]["relative_error"] for name in waves]
    # a wave that the locator could not find is an automatic FAIL, never a
    # silent pass through a max() over the remaining waves
    worst = float("inf") if any(v is None for v in present) else max(present)
    return {"tolerance": TOL_C1,
            "normalisation": "|dx| / |x_exact - x0|",
            "worst_relative_error": float(worst), "waves": rows,
            "status": "PASS" if worst < TOL_C1 else "FAIL"}


def gate_c2(plateaus, waves):
    reference = {
        ("star_left", "density"): waves["density_star_left"],
        ("star_left", "velocity"): waves["velocity_star"],
        ("star_left", "pressure"): waves["pressure_star"],
        ("star_right", "density"): waves["density_star_right"],
    }
    rows, worst = {}, 0.0
    for (region, field), value in reference.items():
        got = plateaus[region][field]
        error = abs(got - value) / abs(value)
        rows[f"{region}.{field}"] = {"exact": float(value), "got": float(got),
                                     "relative_error": float(error)}
        worst = max(worst, error)
    return {"tolerance": TOL_C2, "worst_relative_error": float(worst),
            "values": rows, "status": "PASS" if worst < TOL_C2 else "FAIL"}


def ledger_acceptance_tolerance():
    """The registered ledger tolerance, read from the canonical case module.

    ``feature_ledger`` registers ``CASE-LAX-DG`` against ``ref: exact solution |
    tolerance < 3 %``.  The number lives in
    ``tensordg.cases.lax_1d.lax_parameters()["acceptance_threshold"]`` (where it
    is documented as *relative L2*).  It is imported here instead of copied: a
    copied constant is exactly how a pool gate starts grading against a number
    nobody registered any more.
    """
    from tensordg.cases.lax_1d import lax_parameters

    return float(lax_parameters()["acceptance_threshold"])


def _weighted_l2(points, values, exact_values, weights, dx, mask=None):
    """Relative L2 norm over the (optionally masked) quadrature points."""
    cell_weight = 0.5 * dx * weights[None, :] * np.ones_like(values)
    if mask is not None:
        cell_weight = cell_weight * mask
    numerator = np.sqrt(np.sum(cell_weight * (values - exact_values) ** 2))
    denominator = np.sqrt(np.sum(cell_weight * exact_values ** 2))
    return float(numerator / denominator)


def production_errors_from_profiles(path, quadrature_order=10):
    """Re-measure the production leg straight out of the committed profiles.

    ``profiles.npz`` holds the finest ladder leg's primitive fields and the
    exact solution **on the element Gauss nodes**, so both norms can be
    re-derived without re-running a solver.  This is what makes A0's ledger arm
    checkable offline: no number in the acceptance verdict has to be trusted.
    """
    data = np.load(path)
    points = data["points"]                 # (ncells, n_nodes) physical abscissae
    _, weights = np.polynomial.legendre.leggauss(quadrature_order)
    # Recover the cell width exactly as ``modal_fields`` computes it: the span
    # over ``ncells - 1``.  ``np.mean(np.diff(centres))`` averages ``ncells - 1``
    # rounded differences and lands a few ulp away from the solver's ``dx``, which
    # then shows up as a spurious last-bit mismatch between the ladder's own L2
    # and the L2 re-measured here (m20346 F1: 3.1155618811580767e-02 vs
    # 3.115561881158076e-02).
    centres = 0.5 * (points[:, 0] + points[:, -1])
    dx = float((centres[-1] - centres[0]) / (len(centres) - 1))
    fields = ("density", "velocity", "pressure")
    out = {"l1": {}, "l2": {}}
    for field in fields:
        numerical = data[f"numerical_{field}"]
        exact = data[f"exact_{field}"]
        out["l1"][field] = _weighted_l1(points, numerical, exact, weights, dx)
        out["l2"][field] = _weighted_l2(points, numerical, exact, weights, dx)
    return out


def gate_a0(entries, tolerance=None, production_leg=None, production_l2=None):
    """A0 -- ledger acceptance: relative L2 vs the exact solution, two arms.

    Reading of the ledger wording (``ref: exact solution | tolerance < 3 %``):
    the registered threshold is the one the case module documents, i.e. a
    **relative L2** error, evaluated at the case's **production leg** -- the
    resolution ``benchmarks/lax_1d.py`` and ``tests/test_lax_1d.py`` already
    use (``N = 400``, ``P3``), not a resolution picked after seeing the
    numbers.  Two pre-registered arms are reported separately:

    ``production_l2``
        relative L2 at the finest registered leg, every field, whole field.
        This is the ledger acceptance itself.
    ``ladder_l1``
        relative L1 on **every** ladder leg and field, whole field.  A stricter
        reading of the same 3 % bar (L1 is well defined at a discontinuity and
        does not reward averaging over legs); reported so a reader who takes
        the bar as a norm-free "within 3 %" still gets an answer.

    A0 is deliberately *not* an order gate.  C3/R3 ask the strictly stronger
    question "does the family converge at design order?" and are allowed to
    FAIL on a discontinuous solution; keeping the layers separate is what lets
    an honest verification FAIL coexist with a met acceptance bar.
    """
    tolerance = (ledger_acceptance_tolerance() if tolerance is None
                 else float(tolerance))
    if production_leg is None:
        if not entries:
            raise ValueError("gate_a0 needs at least one ladder leg")
        production_leg = entries[-1]  # ladder is ordered coarse -> fine
    fields = ("density", "velocity", "pressure")
    rows, worst_l2, worst_l1 = {}, None, None
    for entry in entries:
        key = f"p{entry['degree']}.N{entry['cells']}"
        stored_l2 = entry.get("l2_errors")
        rows[key] = {"l1": {f: float(entry["errors"][f]) for f in fields},
                     "l2": {f: (float(stored_l2[f]) if stored_l2 else None)
                            for f in fields}}
        known = [v for v in rows[key]["l2"].values() if v is not None]
        rows[key]["status"] = ("n/a" if not known
                               else "PASS" if max(known) < tolerance else "FAIL")
        for field in fields:
            l2v = rows[key]["l2"][field]
            l1v = rows[key]["l1"][field]
            if l2v is not None and (worst_l2 is None
                                    or l2v > worst_l2["relative_l2"]):
                worst_l2 = {"leg": key, "field": field, "relative_l2": l2v}
            if worst_l1 is None or l1v > worst_l1["relative_l1"]:
                worst_l1 = {"leg": key, "field": field, "relative_l1": l1v}
    prod_key = f"p{production_leg['degree']}.N{production_leg['cells']}"
    if production_l2 is None:
        prod_values, prod_source = dict(production_leg["l2_errors"]), "ladder"
    else:
        prod_values, prod_source = dict(production_l2), "profiles"
        for field in fields:  # keep the per-leg table consistent with the arm
            rows[prod_key]["l2"][field] = float(prod_values[field])
    prod_worst = max(prod_values.values())
    arms = {
        "production_l2": {"leg": prod_key, "metric": "relative L2",
                          "worst": float(prod_worst), "source": prod_source,
                          "status": "PASS" if prod_worst < tolerance else "FAIL"},
        "ladder_l1": {"legs": len(rows), "metric": "relative L1",
                      "worst": float(worst_l1["relative_l1"]),
                      "status": ("PASS"
                                 if worst_l1["relative_l1"] < tolerance else "FAIL")},
    }
    ok = all(arm["status"] == "PASS" for arm in arms.values())
    return {"gate": "A0", "source": "feature_ledger CASE-LAX-DG",
            "wording": "ref: exact solution | tolerance < 3 %",
            "metric": ("relative L2 at the production leg (ledger arm) and "
                       "relative L1 on every ladder leg (stricter arm), on the "
                       "element Gauss nodes, denominator = norm of the exact "
                       "solution"),
            "regions": "whole field, every field, no wave band removed, no exemption",
            "tolerance": tolerance, "production_leg": prod_key,
            "per_leg": rows, "arms": arms,
            "worst": worst_l2, "worst_l1": worst_l1,
            "status": "PASS" if ok else "FAIL"}


def convergence_order(coarse_cells, coarse_error, fine_cells, fine_error):
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    return math.log(coarse_error / fine_error) / math.log(fine_cells / coarse_cells)


def ladder_orders(entries):
    """Per-degree, per-field observed orders between consecutive resolutions."""
    table = {}
    for entry in entries:
        table.setdefault(entry["degree"], {})[entry["cells"]] = entry["errors"]
    out = {}
    for degree, by_cells in table.items():
        cells = sorted(by_cells)
        if len(cells) < 2:
            continue
        per_field = {}
        for field in ("density", "velocity", "pressure"):
            pairs = []
            for coarse, fine in zip(cells[:-1], cells[1:]):
                pairs.append({"coarse": coarse, "fine": fine,
                              "order": convergence_order(
                                  coarse, by_cells[coarse][field],
                                  fine, by_cells[fine][field])})
            per_field[field] = {"pairs": pairs,
                                "finest_order": pairs[-1]["order"]}
        out[degree] = {"cells": cells, "fields": per_field}
    return out


def gate_c3(entries):
    orders = ladder_orders(entries)
    per_leg = {}
    for degree, data in orders.items():
        for field, metrics in data["fields"].items():
            per_leg[f"p{degree}.{field}"] = metrics["finest_order"]
    density = [data["fields"]["density"]["finest_order"] for data in orders.values()]
    gated = min(density) if density else float("inf")
    return {"minimum_order": MIN_C3_ORDER,
            "gated_on": "density, finest ladder pair, every degree",
            "gated_order": float(gated), "per_leg": per_leg,
            "status": "PASS" if gated >= MIN_C3_ORDER else "FAIL"}


def gate_r3(entries, error_key="analytic_errors", band=None):
    """Analytic-region order: same ladder, wave bands masked out."""
    orders = ladder_orders([{"cells": e["cells"], "degree": e["degree"],
                             "errors": e[error_key]} for e in entries])
    per_leg, ok = {}, True
    for degree, data in orders.items():
        design = degree + 1
        for field, metrics in data["fields"].items():
            per_leg[f"p{degree}.{field}"] = metrics["finest_order"]
        density = data["fields"]["density"]["finest_order"]
        requirement = R3_ORDER_FRACTION * design
        per_leg[f"p{degree}.requirement"] = requirement
        per_leg[f"p{degree}.density_ratio"] = None if density is None else density / requirement
        ok = ok and density is not None and density >= requirement
    band = dict(band or {})
    return {"order_fraction": R3_ORDER_FRACTION,
            "band": band, "band_width": band.get("width"),
            "band_cells": band.get("legacy_cells"), "error_key": error_key,
            "gated_on": "density, finest ladder pair, every degree",
            "per_leg": per_leg, "status": "PASS" if ok else "FAIL"}


def r3_nesting_diagnostic(entries):
    """Retained-measure spread per degree: a correct region rule is nested."""
    table = {}
    for entry in entries:
        table.setdefault(entry["degree"], {"nested": [], "legacy": []})
        table[entry["degree"]]["nested"].append(entry["analytic_fraction"])
        table[entry["degree"]]["legacy"].append(entry["analytic_fraction_legacy"])
    report = {}
    for degree, data in table.items():
        nested, legacy = data["nested"], data["legacy"]
        report[f"p{degree}"] = {
            "nested_fractions": nested, "legacy_fractions": legacy,
            "nested_spread": float(max(nested) - min(nested)),
            "legacy_spread": float(max(legacy) - min(legacy)),
            "legacy_grows_with_refinement": bool(legacy[-1] > legacy[0]),
        }
    return report


def entropy_wave_exact(points, final_time, speed, amplitude=ENTROPY_AMPLITUDE):
    phase = 2.0 * np.pi * (points - speed * final_time)
    return 1.0 + amplitude * np.sin(phase)


def gate_r5(entries):
    rows, ok = {}, True
    for entry in entries:
        key = f"u={entry['speed']:.1f}.p{entry['degree']}"
        design = entry["degree"] + 1
        order = entry["order"]
        inside = (order is not None
                  and (1.0 - R5_ORDER_FRACTION) * design <= order
                  <= (1.0 + R5_ORDER_FRACTION) * design)
        rows[key] = {"cells": entry["cells"], "design_order": design,
                     "observed_order": order,
                     "relative_deviation": None if order is None else order / design - 1.0,
                     "relative_l1": entry["relative_l1"],
                     "status": "PASS" if inside else "FAIL"}
        ok = ok and inside
    return {"fraction": R5_ORDER_FRACTION,
            "reading": "observed order within 3 % of the design order p+1",
            "cases": rows, "status": "PASS" if ok else "FAIL"}


def temporal_order(errors):
    """Observed order in ``dt`` from three exact halvings.

    With ``E(dt) = E0 + C dt**q`` the three errors at ``d``, ``d/2``, ``d/4``
    give ``(E1 - E2) / (E2 - E3) = 2**q`` exactly, so ``q`` follows in closed
    form -- no least-squares fit and no assumed spatial-error model.
    """
    if len(errors) != 3 or any(e is None or e <= 0.0 for e in errors):
        return None
    e1, e2, e3 = errors
    if not (e1 > e2 > e3):
        return None
    return math.log((e1 - e2) / (e2 - e3)) / math.log(2.0)


def temporal_bound(errors):
    """Relative error change under a 4x ``dt`` refinement.

    When the error is dt-insensitive the model order is not identifiable (the
    series is flat or non-monotone); the *bound* is then the honest reading:
    a 4x smaller ``dt`` moved the answer by this fraction, so the temporal term
    cannot be larger than that.
    """
    if len(errors) != 3 or any(e is None or e <= 0.0 for e in errors):
        return None
    return float(abs(errors[0] - errors[2]) / errors[2])


def temporal_share(errors, order):
    """Fraction of the baseline-CFL error that the ``dt`` term accounts for."""
    if order is None or errors[0] <= 0.0:
        return None
    growth = 1.0 - 4.0 ** (-order)
    if growth <= 1e-12:
        return None
    return float((errors[0] - errors[2]) / growth / errors[0])


def run_time_refinement(legs=TIME_LEGS, divisors=TIME_CFL_DIVISORS):
    """Hold the mesh fixed, refine only ``dt``, split off the temporal error."""
    rows = []
    for cells, degree in legs:
        started = time.time()
        levels = []
        for divisor in divisors:
            cfl = CFL / divisor
            x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                               final_time=FINAL_TIME, cfl=cfl)
            points, primitive, dx, weights = modal_fields(x, coeff)
            exact = exact_reference(points)
            errors = {field: _weighted_l1(points, primitive[field], exact[field],
                                          weights, dx)
                      for field in ("density", "velocity", "pressure")}
            # ``meta["dt"]`` is the time-truncated *last* step, which is not the
            # step a CFL study refines (it is not monotone in ``cfl``).  The
            # quantity that actually shrinks with the CFL divisor is the mean
            # step ``final_time / steps``; the remnant is kept for traceability.
            # ``cells``/``degree`` are repeated on every level so the artifact
            # itself shows that only ``dt`` moved inside a leg (the --``cells``
            # held fixed'' invariant is otherwise unverifiable from the record).
            levels.append({"cfl": float(cfl),
                           "cells": int(cells), "degree": int(degree),
                           "dt": float(meta.get("dt_mean", FINAL_TIME / max(meta["steps"], 1))),
                           "dt_last_step": float(meta["dt"]),
                           "steps": int(meta["steps"]), "relative_l1": errors})
            print(f"[lax-time] N={cells:4d} p={degree} cfl={cfl:.4f} "
                  f"L1(rho)={errors['density']:.6e}", flush=True)
        fields = {}
        for field in ("density", "velocity", "pressure"):
            series = [level["relative_l1"][field] for level in levels]
            order = temporal_order(series)
            fields[field] = {"relative_l1": series, "temporal_order": order,
                             "temporal_share": temporal_share(series, order),
                             "temporal_bound": temporal_bound(series)}
        rows.append({"cells": cells, "degree": degree, "levels": levels,
                     "fields": fields,
                     "wall_seconds": float(time.time() - started)})
    return rows


def gate_time(rows):
    """Every leg must have a temporal share below the pre-registered bound."""
    per_leg, ok = {}, True
    for row in rows:
        key = f"p{row['degree']}.N{row['cells']}"
        shares = {field: data["temporal_share"]
                  for field, data in row["fields"].items()}
        orders = {field: data["temporal_order"]
                  for field, data in row["fields"].items()}
        bounds = {field: data["temporal_bound"]
                  for field, data in row["fields"].items()}
        # identifiable order -> the model share; otherwise the measured 4x-dt
        # sensitivity is the bound.  Both are compared to the same tolerance.
        worst = None
        for field in shares:
            candidate = shares[field] if shares[field] is not None else bounds[field]
            if candidate is not None:
                worst = candidate if worst is None else max(worst, candidate)
        per_leg[key] = {"temporal_share": shares, "temporal_order": orders,
                        "temporal_bound": bounds, "worst_share": worst}
        ok = ok and worst is not None and worst < TOL_TIME_ERROR_SHARE
    return {"tolerance": TOL_TIME_ERROR_SHARE,
            "reading": ("temporal share of the baseline-CFL L1 error, from three "
                        "exact dt halvings on a fixed mesh"),
            "per_leg": per_leg, "status": "PASS" if ok else "FAIL"}


# --------------------------------------------------------------------- runner
def run_ladder(ladder=LADDER):
    waves = exact_waves()
    exact_pos = exact_positions(waves)
    entries, finest, profiles = [], None, None
    for index, (cells, degree) in enumerate(ladder):
        started = time.time()
        x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                           final_time=FINAL_TIME, cfl=CFL)
        points, primitive, dx, weights = modal_fields(x, coeff)
        exact = exact_reference(points)
        errors = {field: _weighted_l1(points, primitive[field], exact[field],
                                      weights, dx)
                  for field in ("density", "velocity", "pressure")}
        l2_errors = {field: _weighted_l2(points, primitive[field], exact[field],
                                        weights, dx)
                     for field in ("density", "velocity", "pressure")}
        # R3 region rule: fixed *physical* band (nested across the ladder) ...
        mask = np.ones_like(primitive["density"], dtype=bool)
        # ... and the retired dx-scaled band, measured on the very same fields
        mask_legacy = np.ones_like(mask, dtype=bool)
        for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
            distance = np.abs(points - waves["positions"][name])
            mask &= distance > R3_BAND_WIDTH
            mask_legacy &= distance > R3_LEGACY_BAND_CELLS * dx
        analytic_errors = {
            field: _weighted_l1(points, primitive[field], exact[field], weights, dx,
                                mask=mask)
            for field in ("density", "velocity", "pressure")}
        analytic_errors_legacy = {
            field: _weighted_l1(points, primitive[field], exact[field], weights, dx,
                                mask=mask_legacy)
            for field in ("density", "velocity", "pressure")}
        entries.append({
            "cells": cells, "degree": degree,
            "steps": int(meta["steps"]),
            "dt": float(meta.get("dt_mean", FINAL_TIME / max(meta["steps"], 1))),
            "dt_last_step": float(meta["dt"]),
            "wall_seconds": float(time.time() - started),
            "analytic_fraction": float(mask.mean()),
            "analytic_fraction_legacy": float(mask_legacy.mean()),
            "analytic_errors_legacy_band": analytic_errors_legacy,
            "density_min": float(primitive["density"].min()),
            "errors": errors, "l2_errors": l2_errors,
            "analytic_errors": analytic_errors,
        })
        print(f"[lax] N={cells:4d} p={degree} L1(rho)={errors['density']:.6f} "
              f"analytic={analytic_errors['density']:.6f} "
              f"t={entries[-1]['wall_seconds']:.1f}s", flush=True)
        if (cells, degree) == ladder[-1]:
            finest = {"cells": cells, "degree": degree,
                      "positions": wave_positions(points, primitive, waves, dx=dx),
                      "plateaus": plateau_values(points, primitive, waves)}
            profiles = {"points": points, "numerical": primitive, "exact": exact}
    return waves, exact_pos, entries, finest, profiles


def run_entropy(ladder=ENTROPY_CELLS):
    entries = []
    for degree in ENTROPY_DEGREES:
        for speed in ENTROPY_SPEEDS:
            measured = {}
            for cells in ladder:
                def initial_state(node_x, speed=speed):
                    rho = 1.0 + ENTROPY_AMPLITUDE * torch.sin(2.0 * np.pi * node_x)
                    return (rho, torch.full_like(rho, speed),
                            torch.full_like(rho, ENTROPY_PRESSURE))

                x, coeff, _ = solve_periodic_euler_dg_1d(
                    initial_state, ncells=cells, degree=degree,
                    final_time=ENTROPY_TIME, domain=DOMAIN, cfl=ENTROPY_CFL)
                points, primitive, dx, weights = modal_fields(x, coeff)
                exact = entropy_wave_exact(points, ENTROPY_TIME, speed)
                measured[cells] = _weighted_l1(points, primitive["density"], exact,
                                               weights, dx)
            cells = sorted(measured)
            order = convergence_order(cells[-2], measured[cells[-2]],
                                      cells[-1], measured[cells[-1]])
            entries.append({"degree": degree, "speed": speed, "cells": cells,
                            "relative_l1": measured, "order": order})
            print(f"[entropy] u={speed} p={degree} order={order:.4f} "
                  f"L1={measured[cells[-1]]:.3e}", flush=True)
    return entries


def render_figures(waves, entries, finest, profiles, entropy_entries, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    points = profiles["points"]
    order = np.argsort(points.ravel())
    flat = lambda field: np.asarray(field).ravel()[order]  # noqa: E731

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    labels = {"density": r"$\rho$", "velocity": r"$u$", "pressure": r"$p$"}
    for ax, field in zip(axes, ("density", "velocity", "pressure")):
        ax.plot(flat(points), flat(profiles["numerical"][field]), "o", ms=2.0,
                mfc="none", color="tab:blue", label="DG")
        ax.plot(flat(points), flat(profiles["exact"][field]), "-", lw=1.0,
                color="k", label="exact")
        for name, position in waves["positions"].items():
            ax.axvline(position, color="tab:red", lw=0.6, ls=":")
        ax.set_ylabel(labels[field])
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("x")
    axes[0].set_title(f"Lax tube t={FINAL_TIME}, N={finest['cells']}, "
                      f"P{finest['degree']} (dotted: exact wave positions)")
    fig.tight_layout()
    path = out_dir / "lax_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, key, title in ((axes[0], "errors", "whole field"),
                           (axes[1], "analytic_errors", "analytic regions")):
        for degree in sorted({e["degree"] for e in entries}):
            legs = sorted([e for e in entries if e["degree"] == degree],
                          key=lambda e: e["cells"])
            cells = np.array([e["cells"] for e in legs], dtype=float)
            err = np.array([e[key]["density"] for e in legs])
            line, = ax.loglog(cells, err, "o-", label=f"P{degree}")
            ax.loglog(cells, err[-1] * (cells / cells[-1]) ** (-(degree + 1)), ":",
                      color=line.get_color(), lw=0.8)
        ax.set_xlabel("cells")
        ax.set_ylabel(r"relative $L_1$ (density)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lax_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for entry in entropy_entries:
        cells = np.array(entry["cells"], dtype=float)
        err = np.array([entry["relative_l1"][c] for c in entry["cells"]])
        line, = ax.loglog(cells, err, "o-",
                          label=f"u={entry['speed']:.1f}, P{entry['degree']}")
        design = entry["degree"] + 1
        ax.loglog(cells, err[-1] * (cells / cells[-1]) ** (-design), ":",
                  color=line.get_color(), lw=0.8)
    ax.set_xlabel("cells")
    ax.set_ylabel(r"relative $L_1$ ($\rho$)")
    ax.set_title("smooth entropy wave, periodic domain, "
                 r"$\rho=1+0.2\sin(2\pi x)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "entropy_wave_order.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)
    return files


def render_acceptance_figure(entries, acceptance, out_dir=FIG):
    """Per-leg relative L1/L2 against the registered acceptance tolerance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"p{e['degree']} N={e['cells']}" for e in entries]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    styles = {"density": ("o-", r"$\rho$"), "velocity": ("s-", r"$u$"),
              "pressure": ("^-", r"$p$")}
    for field, (style, label) in styles.items():
        l2 = [e.get("l2_errors", {}).get(field) for e in entries]
        known = [(i, v) for i, v in enumerate(l2) if v is not None]
        if known:
            ax.semilogy([i for i, _ in known], [v for _, v in known], style,
                        ms=4, lw=1.1, label=f"rel L2 {label}")
        ax.semilogy(x, [e["errors"][field] for e in entries], style, ms=3, lw=0.7,
                    alpha=0.45, label=f"rel L1 {label}")
    ax.axhline(acceptance["tolerance"], color="tab:red", ls="--", lw=1.2,
               label=f"ledger tolerance {acceptance['tolerance']:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("relative error vs exact")
    prod = acceptance["arms"]["production_l2"]
    ax.set_title(f"A0 ledger acceptance: worst rel L2 {prod['worst']:.3e} at "
                 f"{prod['leg']} -> {acceptance['status']}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "lax_acceptance_a0.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _pct(value, digits=2):
    return "n/a" if value is None else f"{value * 100:.{digits}f} %"


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def render_time_figure(rows, out_dir):
    """L1 error vs dt at a fixed mesh: a flat curve means spatial error dominates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    field_names = ("density", "velocity", "pressure")
    for ax, field in zip(axes, field_names):
        for row in rows:
            dts = [level["dt"] for level in row["levels"]]
            errors = row["fields"][field]["relative_l1"]
            ax.loglog(dts, errors, "o-", markersize=3,
                      label=f"p{row['degree']} N={row['cells']}")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(f"relative $L_1$ ({field})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("temporal refinement at fixed mesh (Lax tube, t=0.13)", fontsize=10)
    fig.tight_layout()
    path = out_dir / "time_refinement.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [path]


def render_summary(metrics):
    lines = ["# CASE-LAX-DG -- measured results (generated, do not edit)", ""]
    lines.append(f"Case: Lax tube, gamma={metrics['waves']['gamma']}, "
                 f"t={metrics['waves']['final_time']}, "
                 f"x0={metrics['waves']['diaphragm']}, CFL={metrics['config']['cfl']}")
    lines.append("")
    lines.append("## analytic wave pattern")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    w = metrics["waves"]
    for key in ("pressure_star", "velocity_star", "density_star_left",
                "density_star_right"):
        lines.append(f"| {key} | {_fmt(w[key], 6)} |")
    for key, value in w["positions"].items():
        lines.append(f"| x({key}) | {_fmt(value, 6)} |")
    lines.append("")
    lines.append("## resolution ladder")
    lines.append("")
    lines.append("| P | N | steps | rel L1 rho | rel L1 u | rel L1 p | "
                 "analytic-region rel L1 rho | wall [s] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in metrics["ladder"]:
        lines.append(f"| {e['degree']} | {e['cells']} | {e['steps']} | "
                     f"{e['errors']['density']:.5e} | {e['errors']['velocity']:.5e} | "
                     f"{e['errors']['pressure']:.5e} | "
                     f"{e['analytic_errors']['density']:.5e} | "
                     f"{e['wall_seconds']:.1f} |")
    lines.append("")
    acceptance = metrics.get("acceptance") or {}
    if acceptance:
        lines.append("## acceptance (ledger CASE-LAX-DG: ref = exact solution, "
                     "tolerance < 3 %)")
        lines.append("")
        lines.append(f"Gate A0 -- {acceptance['metric']}; regions: "
                     f"{acceptance['regions']}.")
        lines.append("")
        lines.append("| arm | metric | leg / legs | worst | verdict |")
        lines.append("|---|---|---|---|---|")
        for name, arm in acceptance["arms"].items():
            where = arm.get("leg") or f"{arm.get('legs')} legs"
            lines.append(f"| {name} | {arm['metric']} | {where} | "
                         f"{arm['worst']:.5e} | {arm['status']} |")
        lines.append("")
        lines.append("| leg | rel L2 rho | rel L2 u | rel L2 p | "
                     "rel L1 rho | rel L1 u | rel L1 p |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, row in acceptance["per_leg"].items():
            lines.append(
                "| " + key + " | "
                + " | ".join(_fmt(row["l2"][f], 5) if row["l2"][f] is not None
                             else "n/a"
                             for f in ("density", "velocity", "pressure")) + " | "
                + " | ".join(f"{row['l1'][f]:.5e}"
                             for f in ("density", "velocity", "pressure")) + " |")
        lines.append("")
        prod = acceptance["arms"]["production_l2"]
        lines.append(f"A0 verdict: **{acceptance['status']}** -- ledger arm "
                     f"(relative L2, {prod['leg']}) worst {prod['worst']:.5e} vs "
                     f"tolerance {acceptance['tolerance']:.2e} "
                     f"(margin x{acceptance['tolerance'] / prod['worst']:.3f}); "
                     f"stricter L1 arm over {acceptance['arms']['ladder_l1']['legs']} "
                     f"legs worst {acceptance['arms']['ladder_l1']['worst']:.5e}.")
        lines.append("")
    lines.append("## gates")
    lines.append("")
    lines.append("| gate | verdict | detail |")
    lines.append("|---|---|---|")
    for name, gate in metrics["gates"].items():
        if name == "c1":
            detail = "; ".join(f"{k}: {_fmt(v['relative_error'] * 100, 3)} %"
                               for k, v in gate["waves"].items())
        elif name == "c2":
            detail = "; ".join(f"{k}: {_fmt(v['relative_error'] * 100, 3)} %"
                               for k, v in gate["values"].items())
        elif name in ("c3", "r3"):
            detail = "; ".join(f"{k}: {_fmt(v, 3)}" for k, v in gate["per_leg"].items())
        elif name == "time":
            detail = "; ".join(f"{k}: worst share {_pct(v['worst_share'])}"
                               for k, v in gate["per_leg"].items())
        elif name == "r5":
            detail = "; ".join(
                f"{k}: order {_fmt(v['observed_order'], 3)} (dev "
                f"{_fmt(v['relative_deviation'] * 100, 2)} %)"
                for k, v in gate["cases"].items())
        else:  # pragma: no cover
            detail = ""
        lines.append(f"| {name.upper()} | {gate['status']} | {detail} |")
    lines.append("")
    time_rows = metrics.get("time_refinement") or []
    if time_rows:
        lines.append("## temporal refinement (mesh fixed, dt -> dt/2 -> dt/4)")
        lines.append("")
        lines.append("| leg | cfl | dt | steps | rel L1 rho | rel L1 u | "
                     "temporal order (rho) | temporal share (rho) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in time_rows:
            for level in row["levels"]:
                rho = row["fields"]["density"]
                lines.append(
                    f"| p{row['degree']} N={row['cells']} | {level['cfl']:.4f} | "
                    f"{level['dt']:.4e} | {level['steps']} | "
                    f"{level['relative_l1']['density']:.5e} | "
                    f"{level['relative_l1']['velocity']:.5e} | "
                    f"{_fmt(rho['temporal_order'], 3)} | "
                    f"{_pct(rho['temporal_share'])} |")
        lines.append("")
    diagnostics = metrics.get("diagnostics") or {}
    if "r3_nesting" in diagnostics:
        lines.append("## R3 region-rule nesting (retained measure per leg)")
        lines.append("")
        lines.append("| degree | nested spread | legacy spread | "
                     "legacy grows with refinement |")
        lines.append("|---|---|---|---|")
        for key, data in diagnostics["r3_nesting"].items():
            lines.append(f"| {key} | {data['nested_spread']:.2e} | "
                         f"{data['legacy_spread']:.2e} | "
                         f"{data['legacy_grows_with_refinement']} |")
        lines.append("")
        legacy = diagnostics.get("r3_legacy_band")
        if legacy:
            lines.append(f"Retired-rule R3 (4-cell band) status for the record: "
                         f"**{legacy['status']}**")
            lines.append("")
    lines.append(f"Verification verdict (supplementary order gates): "
                 f"**{metrics['verdict']}**")
    if acceptance:
        lines.append(f"Acceptance verdict (ledger A0, < 3 % vs exact solution): "
                     f"**{acceptance['status']}**")
    lines.append("")
    lines.append("The two layers answer different questions and are reported "
                 "separately: A0 asks whether the solution sits within the "
                 "registered distance of the exact reference, the order gates "
                 "ask whether the family converges at design order in the "
                 "presence of discontinuities. The verification FAIL is kept "
                 "as-is (honest FAIL), it is not re-labelled by an A0 PASS.")
    lines.append("")
    lines.append(f"Overall verdict: **{metrics['verdict']}**")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="reduced ladder (smoke test, not a verdict run)")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-time-study", action="store_true",
                        help="skip the dt-refinement study (not a verdict run)")
    args = parser.parse_args(argv)
    ladder = QUICK_LADDER if args.quick else LADDER
    out = Path(args.output_dir)
    run_time = not (args.skip_time_study or args.quick)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"

    started = time.time()
    waves, exact_pos, entries, finest, profiles = run_ladder(ladder)
    entropy_entries = run_entropy()
    time_rows = run_time_refinement() if run_time else []
    c1 = gate_c1(finest["positions"], exact_pos)
    c2 = gate_c2(finest["plateaus"], waves)
    c3 = gate_c3(entries)
    r3 = gate_r3(entries, "analytic_errors",
                 {"width": R3_BAND_WIDTH, "rule": "fixed physical band (nested)"})
    r5 = gate_r5(entropy_entries)
    gates = {"c1": c1, "c2": c2, "c3": c3, "r3": r3, "r5": r5}
    acceptance = gate_a0(entries)
    diagnostics = {}
    if entries and "analytic_errors_legacy_band" in entries[0]:
        diagnostics["r3_legacy_band"] = gate_r3(
            entries, "analytic_errors_legacy_band",
            {"legacy_cells": R3_LEGACY_BAND_CELLS,
             "rule": "retired 4-cell band (not nested, m16606 F1)"})
        diagnostics["r3_nesting"] = r3_nesting_diagnostic(entries)
    if time_rows:
        gates["time"] = gate_time(time_rows)
    verdict = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"

    figures = [] if args.skip_figures else render_figures(
        waves, entries, finest, profiles, entropy_entries, fig_dir)
    if time_rows and not args.skip_figures:
        figures.extend(render_time_figure(time_rows, fig_dir))
    if not args.skip_figures:
        figures.append(render_acceptance_figure(entries, acceptance, fig_dir))
    np.savez(out / "profiles.npz",
             points=profiles["points"],
             **{f"numerical_{k}": v for k, v in profiles["numerical"].items()},
             **{f"exact_{k}": v for k, v in profiles["exact"].items()})

    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cfl": CFL, "ladder": ladder, "quick": args.quick,
                   "quadrature_order": 10, "tolerances": {
                       "c1": TOL_C1, "c2": TOL_C2, "c3_min_order": MIN_C3_ORDER,
                       "r3_order_fraction": R3_ORDER_FRACTION,
                       "r5_order_fraction": R5_ORDER_FRACTION,
                       "r3_band_width": R3_BAND_WIDTH,
                       "r3_band_cells_legacy": R3_LEGACY_BAND_CELLS,
                       "time_error_share": TOL_TIME_ERROR_SHARE,
                       "a0_acceptance": acceptance["tolerance"]},
                   "time_study": {"legs": [list(leg) for leg in TIME_LEGS],
                                  "cfl_divisors": list(TIME_CFL_DIVISORS)},
                   "entropy": {"cells": list(ENTROPY_CELLS),
                               "degrees": list(ENTROPY_DEGREES),
                               "speeds": list(ENTROPY_SPEEDS),
                               "amplitude": ENTROPY_AMPLITUDE,
                               "pressure": ENTROPY_PRESSURE,
                               "final_time": ENTROPY_TIME, "cfl": ENTROPY_CFL}},
        "waves": waves,
        "exact_positions": exact_pos,
        "ladder": entries,
        "entropy_wave": entropy_entries,
        "time_refinement": time_rows,
        "diagnostics": diagnostics,
        "figures": [str(p) for p in figures],
        "acceptance": acceptance,
        "gates": gates,
        "verdict": verdict,
        "verdicts": {"acceptance": acceptance["status"], "verification": verdict,
                     "reading": ("acceptance = ledger CASE-LAX-DG, relative L1 vs "
                                 "exact < registered tolerance; verification = "
                                 "supplementary order gates C1/C2/C3/R3/R5/TIME")},
        "wall_seconds": float(time.time() - started),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "summary.md").write_text(render_summary(metrics))
    print(f"[lax] acceptance(A0)={acceptance['status']} "
          f"rel-L2 {acceptance['production_leg']}: "
          f"{acceptance['arms']['production_l2']['worst']:.4e}; "
          f"rel-L1 all legs: {acceptance['arms']['ladder_l1']['worst']:.4e} "
          f"| verification={verdict} -> {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()