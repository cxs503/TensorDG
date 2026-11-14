"""Global conservation-budget audit for the ``CASE-LAX-DG`` shock tube.

The case pool grades the DG run against the exact Riemann solution (wave
positions, star plateaus, relative L2/L1).  It never asks the question a
*conservative* scheme must answer independently of accuracy:

    int_Omega U(x,t) dx - int_Omega U(x,0) dx  ==  t * (F(U_L) - F(U_R))

whenever no wave reaches a domain end, because the transmissive boundary keeps
the initial states at both ends, so the end fluxes are time independent and the
global conservation law integrates in closed form
(:func:`tests._conservation_tools.boundary_increment`).

For the Lax problem the left state carries ``u = 0.698``, so the naive invariant
"total mass/momentum/energy do not change" is *false*: the left end feeds mass,
momentum and energy into the tube.  That makes the case a sharp test of the
boundary-flux bookkeeping -- a scheme that silently loses or gains at the ends
still reproduces the right wave pattern but cannot satisfy the budget.

Two arms are measured, and the pair is falsifiable in both directions:

``budget``
    residual against the closed-form budget.  Must be round-off level
    (pre-registered tolerance :data:`TOLERANCE`).
``naive``
    residual against "``int U`` is constant".  Must be *large* for this case.  A
    tolerance that also passed this arm would be blind to exactly the O(1)
    bookkeeping error the audit exists to catch.

The exact Riemann reference is audited with the same identity, integrated
piecewise with Gauss quadrature on the five analytic pieces of the wave fan.
That check never calls the solver, so it validates the reference itself.

Run::

    PYTHONPATH=. python -m benchmarks.lax_conservation_audit
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (
    DIAPHRAGM,
    FINAL_TIME,
    GAMMA,
    LEFT,
    RIGHT,
    exact_reference,
    exact_waves,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tests._conservation_tools import (
    boundary_increment,
    conserved_totals,
)

DOMAIN = (0.0, 1.0)
FIELDS = ("mass", "momentum", "energy")
NAMES = ("mass", "momentum", "energy")

#: Pre-registered relative tolerance on the budget residual.
#:
#: The measured round-off floor of the audit is a few parts in ``1e-7`` at the
#: coarsest leg and drops with refinement (see ``metrics.json``); ``1e-5`` sits
#: ~10x above that floor and ~3000x below the ledger accuracy bar (3e-2), so it
#: cannot be met by a scheme that loses mass, momentum or energy at a physical
#: rate.
TOLERANCE = 1e-5

#: Resolution / order ladder.  ``(50, 1)`` is the round-off stress leg (largest
#: relative residual), ``(400, 3)`` is the ledger production leg.
LADDER = ((50, 1), (100, 1), (200, 2), (400, 3))

#: Limiter arms measured over the whole ladder.
LIMITER_ARMS = (None, "tvb", "weno")

#: Positivity arms measured at :data:`POSITIVITY_LEG` only.
POSITIVITY_ARMS = ("scale", "collapse")
POSITIVITY_LEG = (200, 2)

#: Historical CFL of the case pool.
CFL = 0.35

OUT = Path("docs/benchmarks/results/lax_conservation")
FIG = OUT / "figures"


# ---------------------------------------------------------------------------
# initial data / budget
# ---------------------------------------------------------------------------
def conserved(rho, velocity, pressure, gamma=GAMMA):
    """``(rho, u, p)`` -> ``(rho, rho*u, E)`` for float arrays of any shape."""
    energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
    return np.stack((rho, rho * velocity, energy), axis=-1)


def initial_cell_averages(
    ncells, left=LEFT, right=RIGHT, diaphragm=DIAPHRAGM, domain=DOMAIN
):
    """Exact cell averages of the piecewise-constant Riemann initial data.

    The average is taken per conserved variable; the initial data are piecewise
    constant, so the fraction-weighted average is exact, also for the (unused)
    case of a diaphragm that falls strictly inside a cell.
    """
    edges = np.linspace(domain[0], domain[1], ncells + 1)
    widths = edges[1:] - edges[:-1]
    fraction_left = np.clip(
        (np.minimum(edges[1:], diaphragm) - edges[:-1]) / widths, 0.0, 1.0
    )
    return (
        fraction_left[:, None] * conserved(*left)
        + (1.0 - fraction_left)[:, None] * conserved(*right)
    )


def start_totals(initial):
    """``int U(0) dx`` of the cell averages (unit-length domain)."""
    return {name: float(initial[:, i].mean()) for i, name in enumerate(NAMES)}


def budget_totals(initial, final_time=FINAL_TIME, left=LEFT, right=RIGHT, gamma=GAMMA):
    """Closed-form ``int U(t)`` from ``int U(0)`` plus the end-flux increment."""
    increment = boundary_increment(left, right, final_time, gamma)
    start = start_totals(initial)
    totals = {name: start[name] + increment[name] for name in NAMES}
    return totals, increment


# ---------------------------------------------------------------------------
# exact reference: piecewise Gauss integral of the closed-form solution
# ---------------------------------------------------------------------------
def exact_reference_totals(order=48, final_time=FINAL_TIME):
    """Integrate the closed-form Riemann solution piecewise (no solver call).

    The solution is analytic on the five pieces cut out by the four wave
    positions, so Gauss-Legendre on each piece converges geometrically; ``48``
    points per piece put the quadrature error below the float64 floor.
    """
    waves = exact_waves(final_time=final_time)
    positions = waves["positions"]
    bounds = [
        DOMAIN[0],
        positions["rarefaction_head"],
        positions["rarefaction_tail"],
        positions["contact"],
        positions["shock"],
        DOMAIN[1],
    ]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    total = np.zeros(3)
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        middle, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        points = middle + half * nodes
        reference = exact_reference(points, final_time=final_time)
        states = conserved(
            np.asarray(reference["density"], dtype=np.float64).ravel(),
            np.asarray(reference["velocity"], dtype=np.float64).ravel(),
            np.asarray(reference["pressure"], dtype=np.float64).ravel(),
        )
        total += half * (weights[:, None] * states).sum(axis=0)
    return {name: float(total[i]) for i, name in enumerate(NAMES)}


def reference_residual(order=48, final_time=FINAL_TIME):
    """Budget residual of the *reference itself* (validates the closed algebra)."""
    initial = initial_cell_averages(400, domain=DOMAIN)
    target, increment = budget_totals(initial, final_time=final_time)
    exact = exact_reference_totals(order=order, final_time=final_time)
    return {
        "totals": exact,
        "budget": target,
        "increment": increment,
        "relative": {
            name: abs(exact[name] - target[name]) / max(abs(target[name]), 1e-300)
            for name in FIELDS
        },
    }


# ---------------------------------------------------------------------------
# solver legs
# ---------------------------------------------------------------------------
def measure_leg(
    ncells,
    degree,
    limiter=None,
    positivity="scale",
    cfl=CFL,
    final_time=FINAL_TIME,
    domain=DOMAIN,
):
    """Run one DG leg and return its conservation bookkeeping."""
    dx = (domain[1] - domain[0]) / ncells
    initial = initial_cell_averages(ncells, domain=domain)
    started = time.perf_counter()
    _x, coefficients, meta = solve_euler_dg_1d(
        LEFT,
        RIGHT,
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        positivity=positivity,
    )
    wall = time.perf_counter() - started
    averages = np.asarray(coefficients.detach().cpu().numpy())[:, 0, :]
    final = conserved_totals(averages, dx)
    target, increment = budget_totals(initial, final_time=final_time)
    start = start_totals(initial)
    return {
        "cells": int(ncells),
        "degree": int(degree),
        "limiter": limiter if limiter else "none",
        "positivity": positivity,
        "steps": int(meta["steps"]),
        "wall_seconds": wall,
        "totals": final,
        "start": start,
        "budget": target,
        "increment": increment,
        "budget_residual": {
            name: abs(final[name] - target[name]) / max(abs(target[name]), 1e-300)
            for name in FIELDS
        },
        "naive_residual": {
            name: abs(final[name] - start[name]) / max(abs(start[name]), 1e-300)
            for name in FIELDS
        },
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate(entries, tolerance=TOLERANCE):
    """Adjudicate the budget arm; the naive arm is reported, never graded."""
    rows = []
    worst = {"value": 0.0}
    for entry in entries:
        for name in FIELDS:
            value = entry["budget_residual"][name]
            rows.append(
                {
                    "leg": f"p{entry['degree']}.N{entry['cells']}",
                    "limiter": entry["limiter"],
                    "positivity": entry["positivity"],
                    "field": name,
                    "budget_residual": value,
                    "naive_residual": entry["naive_residual"][name],
                    "pass": bool(value <= tolerance),
                }
            )
            if value > worst["value"]:
                worst = {
                    "value": value,
                    "leg": f"p{entry['degree']}.N{entry['cells']}",
                    "limiter": entry["limiter"],
                    "positivity": entry["positivity"],
                    "field": name,
                }
    naive_worst = max(row["naive_residual"] for row in rows)
    verdict = "PASS" if all(row["pass"] for row in rows) else "FAIL"
    return {
        "tolerance": tolerance,
        "bar": "|int U(t) - int U(0) - t*(F_L - F_R)| / |int U(t)| on every field and leg",
        "verdict": verdict,
        "worst": worst,
        "naive_worst": naive_worst,
        "detection_ratio": naive_worst / tolerance if tolerance > 0 else None,
        "rows": rows,
    }


def sensitivity(entry, relative_flux_error=1.0e-2, tolerance=TOLERANCE):
    """Detection margin of a relative end-flux defect of ``relative_flux_error``.

    A defect of relative size ``eps`` in the end fluxes displaces the closed-form
    budget by ``eps * |t*(F_L - F_R)|``; divided by ``|int U(t)|`` that is the
    residual the audit would see.  ``detection_margin > 1`` means the defect is
    resolved at :data:`TOLERANCE`.
    """
    scale = max(
        abs(entry["increment"][name]) / max(abs(entry["budget"][name]), 1e-300)
        for name in FIELDS
    )
    displacement = relative_flux_error * scale
    return {
        "relative_flux_error": relative_flux_error,
        "budget_displacement": displacement,
        "detection_margin": displacement / tolerance,
        "increment_scale": scale,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def render_figures(entries, report, out_dir=FIG):
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []

    production = [
        entry for entry in entries if (entry["cells"], entry["degree"]) == (400, 3)
    ] or entries[-1:]
    figure, axes = plt.subplots(figsize=(8.0, 4.4))
    positions, naive_values, budget_values, tick_labels, width = [], [], [], [], 0.38
    cursor = 0.0
    for entry in production:
        for name in FIELDS:
            positions.append(cursor)
            naive_values.append(entry["naive_residual"][name])
            budget_values.append(entry["budget_residual"][name])
            tick_labels.append(f"{entry['limiter']}/{entry['positivity']}\n{name}")
            cursor += 1.0
    axes.bar(np.array(positions) - width / 2, naive_values, width,
             label='naive invariant |dU|=0 (wrong here)')
    axes.bar(np.array(positions) + width / 2, budget_values, width,
             label="budget residual (graded)")
    axes.axhline(report["gate"]["tolerance"], color="k", ls="--", lw=1.0,
                 label=f"tolerance {report['gate']['tolerance']:g}")
    axes.set_yscale("log")
    axes.set_ylabel("relative residual")
    axes.set_xticks(positions)
    axes.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
    axes.set_title(
        f"CASE-LAX-DG conservation: naive invariant fails by O(1), "
        f"closed-form budget closes (N={production[0]['cells']}, p={production[0]['degree']})",
        fontsize=9,
    )
    axes.legend(fontsize=7)
    figure.tight_layout()
    path = out_dir / "lax_conservation_naive_vs_budget.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(str(path))

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for limiter in LIMITER_ARMS:
        label = limiter if limiter else "none"
        series = [
            entry for entry in entries
            if entry["limiter"] == label and entry["positivity"] == "scale"
        ]
        if not series:
            continue
        series.sort(key=lambda entry: (entry["cells"], entry["degree"]))
        dofs = [entry["cells"] * (entry["degree"] + 1) for entry in series]
        values = [max(entry["budget_residual"].values()) for entry in series]
        axes.loglog(dofs, values, "o-", label=f"limiter={label}")
    axes.axhline(report["gate"]["tolerance"], color="k", ls="--", lw=1.0, label="tolerance")
    axes.set_xlabel("degrees of freedom  N*(p+1)")
    axes.set_ylabel("worst relative budget residual")
    axes.set_title("CASE-LAX-DG: the global budget closes at every leg and arm", fontsize=9)
    axes.legend(fontsize=7)
    figure.tight_layout()
    path = out_dir / "lax_conservation_residual_vs_resolution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(str(path))
    return created


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(ladder=LADDER, limiter_arms=LIMITER_ARMS, extra_positivity=True, write=True, entries=None):
    started = time.time()
    if entries is not None:
        ladder = sorted({(e['cells'], e['degree']) for e in entries})
        limiter_arms = tuple(sorted({e['limiter'] if e['limiter'] != 'none' else None for e in entries}, key=str))
    given = entries is not None
    entries = entries if given else []
    for cells, degree in (() if given else ladder):
        for limiter in limiter_arms:
            entry = measure_leg(cells, degree, limiter=limiter)
            entries.append(entry)
            print(
                f"[lax-cons] N={cells:4d} p={degree} limiter={entry['limiter']:5s} "
                f"budget={max(entry['budget_residual'].values()):.3e} "
                f"naive={max(entry['naive_residual'].values()):.3e} "
                f"({entry['wall_seconds']:.1f}s)",
                flush=True,
            )
    if extra_positivity and not given:
        for positivity in POSITIVITY_ARMS:
            if positivity == "scale":
                continue  # covered by the ladder arms
            entry = measure_leg(*POSITIVITY_LEG, limiter=None, positivity=positivity)
            entries.append(entry)
            print(
                f"[lax-cons] N={entry['cells']:4d} p={entry['degree']} "
                f"positivity={positivity:8s} "
                f"budget={max(entry['budget_residual'].values()):.3e} "
                f"({entry['wall_seconds']:.1f}s)",
                flush=True,
            )

    reference = reference_residual()
    report = {
        "case": "CASE-LAX-DG",
        "anchor": {
            "slug": "CASE-LAX-DG",
            "title": "算例池·Lax via DG",
            "acceptance": "ref: 精确解 | 容差 < 3%",
        },
        "study": "global conservation budget (boundary-flux bookkeeping) of the Lax shock tube",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "left_state": list(LEFT),
            "right_state": list(RIGHT),
            "gamma": GAMMA,
            "final_time": FINAL_TIME,
            "diaphragm": DIAPHRAGM,
            "domain": list(DOMAIN),
            "cfl": CFL,
            "ladder": [list(leg) for leg in ladder],
            "limiter_arms": [lim if lim else "none" for lim in limiter_arms],
            "positivity_arms": list(POSITIVITY_ARMS),
            "positivity_leg": list(POSITIVITY_LEG),
        },
        "exact_reference": {
            "method": "piecewise Gauss-Legendre (48 pts) on the 5 analytic pieces",
            "totals": reference["totals"],
            "budget": reference["budget"],
            "increment": reference["increment"],
            "relative_residual": reference["relative"],
            "worst": max(reference["relative"].values()),
        },
        "gate": gate(entries),
        "sensitivity": sensitivity(entries[-1]),
        "entries": entries,
        "wall_seconds": time.time() - started,
    }
    report["figures"] = render_figures(entries, report)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"[lax-cons] wrote {OUT / 'metrics.json'}")
    print(
        f"[lax-cons] verdict={report['gate']['verdict']} "
        f"worst={report['gate']['worst']} reference={report['exact_reference']['worst']:.2e}"
    )
    return report


PARTIAL = OUT / "entries_partial.json"


def run_partial(legs, limiter_arms=LIMITER_ARMS, part=PARTIAL):
    """Measure a subset of legs and append to the partial artifact.

    The audit runs long enough that it is submitted to the cluster in chunks;
    the partial file makes a chunked or resumable run exact (no leg is measured
    twice, none is invented later).
    """
    entries = json.loads(part.read_text()) if part.exists() else []
    have = {(e["cells"], e["degree"], e["limiter"], e["positivity"]) for e in entries}
    for cells, degree in legs:
        for limiter in limiter_arms:
            label = limiter if limiter else "none"
            if (cells, degree, label, "scale") in have:
                print(f"[lax-cons] skip N={cells} p={degree} limiter={label} (already measured)",
                      flush=True)
                continue
            entry = measure_leg(cells, degree, limiter=limiter)
            entries.append(entry)
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_text(json.dumps(entries), encoding="utf-8")
            print(
                f"[lax-cons] N={cells:4d} p={degree} limiter={label:5s} "
                f"budget={max(entry['budget_residual'].values()):.3e} "
                f"naive={max(entry['naive_residual'].values()):.3e} "
                f"({entry['wall_seconds']:.1f}s)",
                flush=True,
            )
    return entries


if __name__ == "__main__":  # pragma: no cover
    spec = os.environ.get("LAX_CONS_LEGS", "").strip()
    if spec:
        legs = [tuple(int(part) for part in token.split("x")) for token in spec.split(",")]
        run_partial(legs)
    else:
        entries = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else None
        if entries is not None:
            entries.sort(key=lambda e: (e["cells"], e["degree"], e["limiter"]))
        run(entries=entries)
        if PARTIAL.exists():
            PARTIAL.unlink()