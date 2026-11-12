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
R3_BAND_CELLS = 4.0
#: fraction of the pressure drop used to place the rarefaction head/tail levels
RAREFACTION_LEVEL_FRACTION = 0.1

OUT = Path("runs/lax_case_pool")
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


def gate_r3(entries):
    """Analytic-region order: same ladder, wave bands masked out."""
    orders = ladder_orders([{"cells": e["cells"], "degree": e["degree"],
                             "errors": e["analytic_errors"]} for e in entries])
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
    return {"order_fraction": R3_ORDER_FRACTION, "band_cells": R3_BAND_CELLS,
            "gated_on": "density, finest ladder pair, every degree",
            "per_leg": per_leg, "status": "PASS" if ok else "FAIL"}


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
        mask = np.ones_like(primitive["density"], dtype=bool)
        for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
            mask &= np.abs(points - waves["positions"][name]) > R3_BAND_CELLS * dx
        analytic_errors = {
            field: _weighted_l1(points, primitive[field], exact[field], weights, dx,
                                mask=mask)
            for field in ("density", "velocity", "pressure")}
        entries.append({
            "cells": cells, "degree": degree,
            "steps": int(meta["steps"]), "dt": float(meta["dt"]),
            "wall_seconds": float(time.time() - started),
            "analytic_fraction": float(mask.mean()),
            "density_min": float(primitive["density"].min()),
            "errors": errors, "analytic_errors": analytic_errors,
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


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


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
        elif name == "r5":
            detail = "; ".join(
                f"{k}: order {_fmt(v['observed_order'], 3)} (dev "
                f"{_fmt(v['relative_deviation'] * 100, 2)} %)"
                for k, v in gate["cases"].items())
        else:  # pragma: no cover
            detail = ""
        lines.append(f"| {name.upper()} | {gate['status']} | {detail} |")
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
    args = parser.parse_args(argv)
    ladder = QUICK_LADDER if args.quick else LADDER
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"

    started = time.time()
    waves, exact_pos, entries, finest, profiles = run_ladder(ladder)
    entropy_entries = run_entropy()
    c1 = gate_c1(finest["positions"], exact_pos)
    c2 = gate_c2(finest["plateaus"], waves)
    c3 = gate_c3(entries)
    r3 = gate_r3(entries)
    r5 = gate_r5(entropy_entries)
    gates = {"c1": c1, "c2": c2, "c3": c3, "r3": r3, "r5": r5}
    verdict = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"

    figures = [] if args.skip_figures else render_figures(
        waves, entries, finest, profiles, entropy_entries, fig_dir)
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
                       "r3_band_cells": R3_BAND_CELLS},
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
        "figures": [str(p) for p in figures],
        "gates": gates,
        "verdict": verdict,
        "wall_seconds": float(time.time() - started),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "summary.md").write_text(render_summary(metrics))
    print(f"[lax] verdict={verdict} -> {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()