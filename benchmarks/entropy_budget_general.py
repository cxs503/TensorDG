"""General analytic entropy budget for the 1-D Euler Riemann problem.

Motivation / gap being closed
-----------------------------
``benchmarks._entropy_tools.analytic_right_shock_budget`` only covers the
``p* > p_right`` configuration (Sod, Lax).  ``docs/entropy_verification.md``
§⑤ records this restriction as an open debt:

    "解析基准只实现「右侧激波」构型（p* > p_R，Sod/Lax 均属此类）；
     左侧激波/双稀疏波未覆盖。"

This module closes it: the analytic Rankine-Hugoniot budget is assembled for
**all four wave patterns** of a two-state Riemann problem

    left  wave :  left  shock  (p* > p_L)  or  left  rarefaction  (p* <= p_L)
    right wave :  right shock  (p* > p_R)  or  right rarefaction  (p* <= p_R)

and the result is anchored two independent ways:

1. *Convention-free anchor* — the entropy integral of the **exact sampled
   solution** ``int rho s dx`` at ``t`` minus the same at ``t -> 0``.  As long
   as no wave reaches the domain ends, no entropy leaves the domain, so this
   difference *is* the total production.  It uses no Rankine-Hugoniot formula
   at all, hence it can falsify the RH baseline.
2. *Literature anchor* — the star states ``(p*, u*)`` of Toro's four standard
   test problems (Toro, *Riemann Solvers and Numerical Methods for Fluid
   Dynamics*, 3rd ed., Table 4.1) are reproduced to the printed precision in
   ``tests/test_entropy_budget_general.py``.

Sign conventions (important, and the reason a naive "left shock" extension
gets the sign wrong) are fixed by ``shock_speed_and_production``: its two
arguments are the states on the **left** and the **right in space**, and it
returns a positive production for a physically admissible shock.  Therefore a
left-going shock is assembled as ``(star, left_state)`` and a right-going
shock as ``(right_state, star)`` — see :func:`analytic_entropy_budget`.

Usage
-----
    PYTHONPATH=. python -m benchmarks.entropy_budget_general
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks._entropy_tools import (
    GAMMA,
    gamma_shock_ratio,
    relative_error,
    shock_speed_and_production,
    total_entropy_from_coefficients,
    total_entropy_from_profiles,
)
from benchmarks.riemann_exact import sample, star_state
from tensordg.euler_dg_1d import solve_euler_dg_1d

# --------------------------------------------------------------------------- #
# the four wave patterns, and the documented test cases
# --------------------------------------------------------------------------- #
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)

# Mirror image of Sod: the shock now travels to the *left* (p* = 0.30313 >
# p_L = 0.1), so this exercises the previously uncovered left-shock branch.
MIRROR_LEFT = SOD_RIGHT
MIRROR_RIGHT = SOD_LEFT

# Symmetric double rarefaction: two fans, an entropy-carrying contact, and no
# shock at all -> the exact entropy production is identically zero.
# Density 2 (not 1) on purpose: with rho = p = 1 the specific entropy is
# identically zero and the entropy budget would be trivially satisfied.
RAREFACTION_LEFT = (2.0, -1.0, 1.0)
RAREFACTION_RIGHT = (2.0, 1.0, 1.0)

# Toro's test 2 ("123 problem"), near-vacuum double rarefaction.
NEAR_VACUUM_LEFT = (1.0, -2.0, 0.4)
NEAR_VACUUM_RIGHT = (1.0, 2.0, 0.4)

CASES = {
    "sod_right_shock": {
        "left": SOD_LEFT, "right": SOD_RIGHT, "final_time": 0.2,
        "pattern": "left_rarefaction + right_shock",
    },
    "mirror_sod_left_shock": {
        "left": MIRROR_LEFT, "right": MIRROR_RIGHT, "final_time": 0.2,
        "pattern": "left_shock + right_rarefaction",
    },
    "symmetric_two_rarefaction": {
        "left": RAREFACTION_LEFT, "right": RAREFACTION_RIGHT, "final_time": 0.15,
        "pattern": "left_rarefaction + right_rarefaction",
    },
    "near_vacuum_123": {
        "left": NEAR_VACUUM_LEFT, "right": NEAR_VACUUM_RIGHT, "final_time": 0.15,
        "pattern": "left_rarefaction + right_rarefaction (strong)",
    },
}


# --------------------------------------------------------------------------- #
# analytic budget
# --------------------------------------------------------------------------- #
def _rarefaction_speeds(state, pressure_star, gamma):
    """(head, tail) characteristic speeds of the rarefaction fan of ``state``."""
    density, velocity, pressure = state
    sound = float(np.sqrt(gamma * pressure / density))
    sound_star = sound * (pressure_star / pressure) ** ((gamma - 1.0) / (2.0 * gamma))
    return velocity - sound, velocity + sound, sound_star


def analytic_entropy_budget(left, right, final_time: float, gamma: float = GAMMA):
    """Total ``int rho s`` production up to ``final_time`` for any wave pattern.

    Parameters
    ----------
    left, right : (rho, u, p)
        The two constant initial states (``left`` on the low-``x`` side).
    final_time : float
        Time at which the budget is evaluated; the production rate of a
        self-similar shock is constant, so ``production_total = rate * t``.

    Returns
    -------
    dict
        ``star_state``, ``patterns``, per-``waves`` records, ``production_rate``,
        ``production_total``, ``isentropic`` and ``wave_speed_bounds`` (the
        slowest / fastest characteristic speeds, used to check that no wave
        has left a finite domain).
    """
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    rho_l, u_l, p_l = (float(value) for value in left)
    rho_r, u_r, p_r = (float(value) for value in right)
    pressure_star, velocity_star = star_state(left, right, gamma)

    waves = []
    patterns = []

    # ---- left-hand wave -------------------------------------------------- #
    if pressure_star > p_l:
        patterns.append("left_shock")
        density_star = rho_l * gamma_shock_ratio(pressure_star / p_l, gamma)
        # space ordering is (star, left state): the shock moves into the left state
        budget = shock_speed_and_production(
            (density_star, velocity_star, pressure_star), (rho_l, u_l, p_l), gamma
        )
        waves.append({
            "side": "left", "kind": "shock",
            "speed": budget["sigma"],
            "upstream": list(left),
            "downstream": [density_star, velocity_star, pressure_star],
            "production_rate": budget["production"],
        })
    else:
        patterns.append("left_rarefaction")
        head_l, _, sound_star_l = _rarefaction_speeds(left, pressure_star, gamma)
        waves.append({
            "side": "left", "kind": "rarefaction",
            "speed": head_l,
            "tail_speed": velocity_star - sound_star_l,
            "upstream": list(left),
            "downstream": None,
            "production_rate": 0.0,     # rarefactions are isentropic
        })

    # ---- right-hand wave ------------------------------------------------- #
    if pressure_star > p_r:
        patterns.append("right_shock")
        density_star = rho_r * gamma_shock_ratio(pressure_star / p_r, gamma)
        budget = shock_speed_and_production(
            (rho_r, u_r, p_r), (density_star, velocity_star, pressure_star), gamma
        )
        waves.append({
            "side": "right", "kind": "shock",
            "speed": budget["sigma"],
            "upstream": list(right),
            "downstream": [density_star, velocity_star, pressure_star],
            "production_rate": budget["production"],
        })
    else:
        patterns.append("right_rarefaction")
        _, head_r, sound_star_r = _rarefaction_speeds(right, pressure_star, gamma)
        waves.append({
            "side": "right", "kind": "rarefaction",
            "speed": head_r,
            "tail_speed": velocity_star + sound_star_r,
            "upstream": list(right),
            "downstream": None,
            "production_rate": 0.0,
        })

    production_rate = float(sum(wave["production_rate"] for wave in waves))
    speeds = []
    for wave in waves:
        speeds.append(wave["speed"])
        if "tail_speed" in wave:
            speeds.append(wave["tail_speed"])
    return {
        "left": list(left),
        "right": list(right),
        "final_time": float(final_time),
        "star_state": (float(pressure_star), float(velocity_star)),
        "patterns": patterns,
        "waves": waves,
        "production_rate": production_rate,
        "production_total": production_rate * float(final_time),
        "isentropic": not any(wave["kind"] == "shock" for wave in waves),
        "wave_speed_bounds": (float(min(speeds)), float(max(speeds))),
    }


def exact_solution_production(left, right, final_time, diaphragm: float = 0.5,
                              domain=(0.0, 1.0), samples: int = 200_001,
                              gamma: float = GAMMA):
    """Convention-free production from the exact solution.

    ``P T = d int rho s dx + T * (f(upper) - f(lower))`` with ``f = rho u s``.
    The boundary term matters whenever an *initial state is moving* (a double
    rarefaction necessarily has through-flow), so it is not optional.  Valid
    while no wave reaches the domain ends, which leaves the boundary states at
    their initial values for all ``t <= final_time``.
    """
    lower, upper = domain
    x = np.linspace(lower, upper, samples)
    initial = sample(x, 1e-9, left, right, diaphragm, gamma)
    final = sample(x, final_time, left, right, diaphragm, gamma)
    delta = (
        total_entropy_from_profiles(final["density"], final["pressure"], x, gamma)
        - total_entropy_from_profiles(initial["density"], initial["pressure"], x, gamma)
    )

    def entropy_flux(state):
        density, velocity, pressure = (float(value) for value in state)
        specific = float(np.log(pressure) - gamma * np.log(density))
        return density * velocity * specific

    boundary = final_time * (entropy_flux(right) - entropy_flux(left))
    return delta + boundary


def domain_is_clean(spec, diaphragm: float = 0.5, domain=(0.0, 1.0)):
    """True when no wave reaches the domain ends before ``final_time``."""
    budget = analytic_entropy_budget(spec["left"], spec["right"], spec["final_time"])
    slow, fast = budget["wave_speed_bounds"]
    t = spec["final_time"]
    return diaphragm + slow * t > domain[0] and diaphragm + fast * t < domain[1]


# --------------------------------------------------------------------------- #
# DG measurement and study driver
# --------------------------------------------------------------------------- #
def measure_dg(left, right, ncells, degree, final_time, limiter, tvb_constant,
               limiter_stage="every", cfl=0.3):
    """``d int rho s dx`` over one DG run (full polynomial, not cell averages)."""
    _, coeff_0, _ = solve_euler_dg_1d(
        left, right, ncells=ncells, degree=degree, final_time=0.0, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
    )
    _, coeff_t, meta = solve_euler_dg_1d(
        left, right, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
    )
    dx = 1.0 / ncells
    entropy_0 = total_entropy_from_coefficients(coeff_0, dx, degree)
    entropy_t = total_entropy_from_coefficients(coeff_t, dx, degree)
    return {
        "entropy_initial": entropy_0,
        "entropy_final": entropy_t,
        "production_numeric": entropy_t - entropy_0,
        "steps": meta["steps"],
        "dt": meta["dt"],
    }


def study(cells=(32, 64, 128), degree=2, limiter_settings=((None, 1.0), ("tvb", 1.0)),
          cfl=0.3, names=None):
    """Compare the DG production with the general analytic budget for each case."""
    results = {}
    for name, spec in CASES.items():
        if names is not None and name not in names:
            continue
        left, right, final_time = spec["left"], spec["right"], spec["final_time"]
        budget = analytic_entropy_budget(left, right, final_time)
        reference = exact_solution_production(left, right, final_time)
        rows = []
        for limiter, tvb_constant in limiter_settings:
            for ncells in cells:
                try:
                    run = measure_dg(left, right, ncells, degree, final_time,
                                     limiter, tvb_constant, cfl=cfl)
                except FloatingPointError as error:
                    rows.append({
                        "limiter": limiter or "none",
                        "tvb_constant": tvb_constant,
                        "cells": ncells,
                        "degree": degree,
                        "error": type(error).__name__,
                        "message": str(error),
                    })
                    continue
                produced = run["production_numeric"]
                rows.append({
                    "limiter": limiter or "none",
                    "tvb_constant": tvb_constant,
                    "cells": ncells,
                    "degree": degree,
                    "steps": run["steps"],
                    "entropy_initial": run["entropy_initial"],
                    "entropy_final": run["entropy_final"],
                    "production_numeric": produced,
                    "production_analytic": budget["production_total"],
                    "production_exact_solution": reference,
                    "relative_error_vs_analytic": (
                        None if budget["production_total"] == 0.0
                        else relative_error(budget["production_total"], produced)
                    ),
                    "second_law": bool(produced >= 0.0),
                })
        results[name] = {
            "pattern": spec["pattern"],
            "final_time": final_time,
            "domain_clean": domain_is_clean(spec),
            "analytic": {
                "star_state": list(budget["star_state"]),
                "patterns": budget["patterns"],
                "waves": budget["waves"],
                "production_rate": budget["production_rate"],
                "production_total": budget["production_total"],
                "isentropic": budget["isentropic"],
                "wave_speed_bounds": list(budget["wave_speed_bounds"]),
            },
            "baseline_selfcheck": {
                "production_exact_solution": reference,
                "relative_difference": (
                    None if budget["production_total"] == 0.0
                    else relative_error(budget["production_total"], reference)
                ),
            },
            "rows": rows,
        }
    return results


def figure(path: Path, results, degree=2, ncells_figure=128):
    """Density/pressure of the DG solution vs the exact solution, four patterns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, (name, spec) in zip(axes.ravel(), CASES.items()):
        left, right, final_time = spec["left"], spec["right"], spec["final_time"]
        budget = analytic_entropy_budget(left, right, final_time)
        _, coeff, _ = solve_euler_dg_1d(
            left, right, ncells=ncells_figure, degree=degree, final_time=final_time,
            cfl=0.3, limiter="tvb", tvb_constant=1.0, limiter_stage="every",
        )
        nodes, _ = np.polynomial.legendre.leggauss(degree + 1)
        dx = 1.0 / ncells_figure
        centers = (np.arange(ncells_figure) + 0.5) * dx
        points = (centers[:, None] + 0.5 * dx * nodes[None, :]).ravel()

        def legendre(nodes_in, degree_in):
            basis = []
            for k in range(degree_in + 1):
                polynomial = np.zeros(k + 1)
                polynomial[-1] = 1.0
                basis.append(np.polynomial.legendre.legval(nodes_in, polynomial))
            return np.stack(basis, axis=1)

        phi = legendre(nodes, degree)
        values = np.einsum("cmv,qm->cqv", np.asarray(coeff), phi)
        density = values.reshape(-1, 3)[:, 0]
        momentum = values.reshape(-1, 3)[:, 1]
        energy = values.reshape(-1, 3)[:, 2]
        velocity = momentum / density
        pressure = (GAMMA - 1.0) * (energy - 0.5 * momentum ** 2 / density)

        order = np.argsort(points)
        exact = sample(points[order], final_time, left, right, 0.5)
        axis.plot(points[order], exact["density"], "k-", lw=1.4, label="exact $\\rho$")
        axis.plot(points, density, "C0.", ms=2.0, label="DG $P_2$ + TVB, %d cells" % ncells_figure)
        twin = axis.twinx()
        twin.plot(points[order], exact["pressure"], "k--", lw=1.0, alpha=0.6)
        twin.plot(points, pressure, "C1.", ms=1.6, alpha=0.7)
        twin.set_ylabel("pressure (dashed/right)", fontsize=8)
        twin.tick_params(labelsize=8)
        axis.set_title(
            "%s\n$p^*$=%.5f, $u^*$=%+.5f, $P t$=%.4e"
            % (name, budget["star_state"][0], budget["star_state"][1],
               budget["production_total"]),
            fontsize=9,
        )
        axis.set_xlabel("$x$", fontsize=8)
        axis.set_ylabel("density", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("TensorDG: four Riemann wave patterns vs exact solution "
                 "(previously uncovered: left shock, double rarefaction)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--out", type=str,
                        default="validation/entropy_1d_general/summary.json")
    parser.add_argument("--figure", type=str,
                        default="docs/figures/entropy_general_patterns.png")
    args = parser.parse_args()
    results = study(cells=tuple(args.cells), degree=args.degree, cfl=args.cfl,
                    names=args.cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(Path(args.figure), results, degree=args.degree)
    payload = {"degree": args.degree, "cells": args.cells, "cfl": args.cfl,
               "cases": results}
    out.write_text(json.dumps(payload, indent=2))
    for name, block in results.items():
        analytic = block["analytic"]
        print(f"{name}: {block['pattern']}  P*t={analytic['production_total']:+.6e}"
              f"  clean_domain={block['domain_clean']}")
        for row in block["rows"]:
            if "error" in row:
                print(f"   {row['limiter']:4s} n={row['cells']:4d}  FAILED {row['message']}")
                continue
            print(f"   {row['limiter']:4s} n={row['cells']:4d}  produced={row['production_numeric']:+.6e}"
                  f"  rel_err={'n/a' if row['relative_error_vs_analytic'] is None else format(row['relative_error_vs_analytic'], '.3e')}"
                  f"  second_law={row['second_law']}")
    print(f"wrote {out} and {args.figure}")


if __name__ == "__main__":
    main()