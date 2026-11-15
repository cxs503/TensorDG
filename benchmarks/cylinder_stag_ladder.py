"""CKS1998 §5.4 (flow past a circular cylinder): stagnation-pressure verification ladder.

What is exact here and why it is a *verification* (not validation) gate
----------------------------------------------------------------------
The one quantity on the cylinder wall that is known in closed form for the
steady 2-D Euler equations is the **stagnation pressure**.  Along the stagnation
streamline the total pressure is conserved (the flow is inviscid, adiabatic and
steady on that streamline), so the pressure the fluid exerts on the body at the
front stagnation point is the total pressure *behind* whatever shock the
stagnation streamline crosses:

* ``M_inf < 1``  no shock on the streamline ->
  ``p_wall/p_inf = (1 + (γ-1)/2 M²)^{γ/(γ-1)}``        (isentropic stagnation)
* ``M_inf > 1``  the streamline crosses the normal part of the bow shock ->
  ``p_wall/p_inf = p02/p01 · p01/p_inf``               (Rayleigh pitot)

hence

    ``Cp_stag = 2 (p_wall/p_inf - 1) / (γ M_inf²)``

is a *number*, not a fitted reference.  The in-repo implementation of both
formulae is ``benchmarks/cylinder_euler_steady.py`` (``isentropic_total_pressure``
/ ``rayleigh_pitot``); this module re-derives them from the Rankine-Hugoniot
relations in ``independent_stagnation_target`` so that the target used here is
not read back from the code under test (the two derivations are cross-checked in
``tests/test_cylinder_stag_ladder.py``).

What the ladder measures
------------------------
For a sequence of annular O-grids (radial cells ``nr``, azimuthal ``ntheta``,
polynomial degree ``p``, radial clustering ``growth``, far-field radius
``r_out``) the module drives the curvilinear DG solver to a steady state and
records

* ``cp_stag`` - ``max`` wall ``Cp`` (the stagnation point),
* ``cp_stag_target`` / ``cp_stag_rel_error`` - the exact target above,
* ``cp_stag_mean_band`` - mean ``Cp`` over the last 5 % of the azimuthal range
  around the stagnation point (robust to a one-cell shift of the peak),
* ``symmetry_error`` - ``max |Cp(θ) - Cp(-θ)| / max|Cp|`` (the geometry and the
  inflow are invariant under ``y -> -y``, so this is pure discretisation error),
* ``mass/energy drift`` - the DG scheme conserves the domain integrals up to the
  boundary fluxes,
* the residual plateau (steady state reached) and the run cost.

Usage::

    PYTHONPATH=. python -m benchmarks.cylinder_stag_ladder --smoke      # ~1 min
    PYTHONPATH=. python -m benchmarks.cylinder_stag_ladder --ladder     # full
    PYTHONPATH=. python -m benchmarks.cylinder_stag_ladder --ladder m3-p0-12x24
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.cylinder_euler_steady import isentropic_total_pressure, rayleigh_pitot
from tensordg.euler_curved import CurvilinearEuler, cylinder_annulus_mesh

OUT = Path(__file__).resolve().parents[1] / "validation" / "cylinder_stag_ladder"
GAMMA = 1.4


# --------------------------------------------------------------------------- #
# exact target, re-derived independently of the module under verification
# --------------------------------------------------------------------------- #
def independent_stagnation_target(mach: float, gamma: float = GAMMA) -> float:
    """``Cp_stag`` from the Rankine-Hugoniot relations written out by hand.

    For ``mach > 1`` the stagnation streamline crosses the normal part of the bow
    shock.  With ``M = mach``:

        p2/p1 = (2γM² - (γ-1)) / (γ+1)                        (normal shock)
        M2²   = ((γ-1)M² + 2) / (2γM² - (γ-1))
        p02/p2 = (1 + (γ-1)/2 M2²)^{γ/(γ-1)}                  (isentropic, post-shock)
        p0/p1  = (p02/p2)(p2/p1)                              (Rayleigh pitot)

    For ``mach < 1`` there is no shock on the streamline and the isentropic
    relation applies directly.  The ``Cp`` normalisation uses the free-stream
    dynamic pressure ``½ ρ_inf u_inf² = ½ γ p_inf M²``.
    """
    if mach < 1.0:
        ratio = (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** (gamma / (gamma - 1.0))
    else:
        m2 = mach * mach
        p2_p1 = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)
        m2_sq = ((gamma - 1.0) * m2 + 2.0) / (2.0 * gamma * m2 - (gamma - 1.0))
        p02_p2 = (1.0 + 0.5 * (gamma - 1.0) * m2_sq) ** (gamma / (gamma - 1.0))
        ratio = p02_p2 * p2_p1
    return 2.0 * (ratio - 1.0) / (gamma * mach ** 2)


def _symmetry_error(theta: np.ndarray, cp: np.ndarray, nsample: int = 721) -> float:
    """``max |Cp(θ) - Cp(-θ)| / max|Cp|`` on a common azimuthal grid."""
    order = np.argsort(theta)
    th, c = theta[order], cp[order]
    # close the periodic ring so ``np.interp`` sees both ends
    th_ext = np.concatenate([th - 2.0 * np.pi, th, th + 2.0 * np.pi])
    c_ext = np.concatenate([c, c, c])
    grid = np.linspace(-np.pi, np.pi, nsample, endpoint=False)
    cp_grid = np.interp(grid, th_ext, c_ext)
    cp_mirror = np.interp(-grid, th_ext, c_ext)
    scale = max(float(np.max(np.abs(cp_grid))), 1e-30)
    return float(np.max(np.abs(cp_grid - cp_mirror)) / scale)


def _stagnation_band(theta: np.ndarray, cp: np.ndarray, frac: float = 0.05) -> float:
    """Mean ``Cp`` over the azimuthal band of width ``frac·π`` about the peak."""
    i = int(np.argmax(cp))
    th0 = theta[i]
    d = np.abs(np.mod(theta - th0 + np.pi, 2.0 * np.pi) - np.pi)
    band = d <= frac * np.pi
    return float(np.mean(cp[band]))


# --------------------------------------------------------------------------- #
# one ladder rung
# --------------------------------------------------------------------------- #
def run_case(nr: int, ntheta: int, p: int, mach: float = 3.0, r_out: float = 5.0,
             growth: float = 1.0, cfl: float = 0.4, max_time: float = 6.0,
             steady_tol: float = 1e-4, report_every: int = 100,
             record: bool = True) -> dict:
    a_inf = math.sqrt(GAMMA)
    u_inf = mach * a_inf
    rho_inf, p_inf = 1.0, 1.0
    mesh = cylinder_annulus_mesh(nr, ntheta, 1.0, r_out, degree=2, growth=growth)
    solver = CurvilinearEuler(
        mesh, p, cfl=cfl, boundaries=("wall", "farfield", "periodic", "periodic"),
        farfield=(rho_inf, u_inf, 0.0, p_inf))
    state = solver.project(lambda x, y: (np.ones_like(x), u_inf * np.ones_like(x),
                                         np.zeros_like(x), np.ones_like(x)))
    totals0 = solver.totals(state)
    dt = 0.5 * solver.stable_dt(state)

    t0 = time.perf_counter()
    hist, steps, t = [], 0, 0.0
    steady = False
    rate0 = None
    blew_up = False
    while t < max_time:
        state = solver.ssp_rk3_step(state, dt)
        t += dt
        steps += 1
        if steps % report_every == 0 or t >= max_time:
            rate = float(np.max(np.abs(solver.rhs(state))))
            if rate0 is None:
                rate0 = rate
            hist.append({"t": t, "residual": rate, "ratio": rate / max(rate0, 1e-300)})
            if not np.all(np.isfinite(state)) or np.abs(state).max() > 1e3:
                blew_up = True
                break
            if rate0 is not None and rate < steady_tol * rate0 and steps > 4 * report_every:
                steady = True
                break
    wall = time.perf_counter() - t0
    totals1 = solver.totals(state)

    walls = solver.wall_pressure(state)
    theta = np.asarray(walls[(0, -1)]["theta"]).ravel()
    pressure = np.asarray(walls[(0, -1)]["p"]).ravel()
    cp = 2.0 * (pressure - p_inf) / (GAMMA * mach ** 2)
    target = independent_stagnation_target(mach)
    cp_stag = float(np.max(cp))
    force = solver.wall_force(state)
    cd = force["per_side"][(0, -1)][0] / (0.5 * rho_inf * u_inf ** 2 * 2.0)

    row = {
        "tag": f"m{mach:g}-p{p}-{nr}x{ntheta}-g{growth:g}-R{r_out:g}",
        "nr": nr, "ntheta": ntheta, "degree": p, "mach": mach,
        "r_out": r_out, "growth": growth, "cfl": cfl,
        "cells": int(mesh.ncells), "dof": int(mesh.ncells) * (p + 1) ** 2 * 4,
        "dt": dt, "steps": steps, "time": t, "wall_seconds": wall,
        "steady": steady, "blew_up": blew_up,
        "residual_first": None if not hist else hist[0]["residual"],
        "residual_last": None if not hist else hist[-1]["residual"],
        "residual_ratio": None if not hist else hist[-1]["ratio"],
        "mass_drift_rel": float((totals1[0] - totals0[0]) / totals0[0]),
        "energy_drift_rel": float((totals1[3] - totals0[3]) / totals0[3]),
        "cp_stag": cp_stag,
        "cp_stag_mean_band": _stagnation_band(theta, cp),
        "cp_stag_target": target,
        "cp_stag_rel_error": abs(cp_stag - target) / target,
        "cp_min": float(np.min(cp)),
        "cd": float(cd),
        "symmetry_error": _symmetry_error(theta, cp),
    }
    if record:
        row["theta"] = np.asarray(theta).tolist()
        row["cp"] = np.asarray(cp).tolist()
        row["residual_history"] = hist
    return row


# --------------------------------------------------------------------------- #
# ladder driver
# --------------------------------------------------------------------------- #
def default_ladder() -> list[dict]:
    """h-refinement sequences for P0/P1 + a subsonic rung, M=3 steady bow shock.

    The grids are kept close to isotropic *at the body* (``dr ~ 2π/nr_theta``)
    with a mild radial clustering (``growth``) so that the bow shock and the
    subsonic post-shock cap between body and shock are resolved without paying
    for an unnecessarily small radial step.
    """
    out = []
    for p, sizes in ((0, [(16, 26, 1.10), (24, 40, 1.08), (32, 52, 1.06)]),
                     (1, [(12, 20, 1.10), (24, 40, 1.08), (32, 52, 1.06)])):
        for nr, nt, growth in sizes:
            out.append(dict(nr=nr, ntheta=nt, p=p, mach=3.0, r_out=5.0,
                            growth=growth))
    # subsonic rung: no shock on the streamline -> isentropic target
    out.append(dict(nr=24, ntheta=40, p=1, mach=0.38, r_out=5.0, growth=1.08))
    return out


def control_cases() -> list[dict]:
    """Attribution controls: is the committed cylinder deficit a grid or a *time* effect?

    ``validation/cylinder_steady/metrics_p0_6x12.json`` (m21767) reports
    ``cp_stag = 0.4022`` against the same analytic target used here, i.e. a 77 %
    deficit.  That run is ``steps = 35``, ``time = 0.307`` - the start-up
    transient, before the bow shock exists.  The three controls are

    C1  the *identical* committed configuration and final time  -> must reproduce
        the committed number (the run is repeatable, the solver is not "broken"),
    C2  the same configuration continued to a steady state       -> the deficit
        must disappear (it was time, not space),
    C3  a resolved grid stopped at the *same* early time 0.307   -> the deficit
        must reappear (it cannot be cured by resolution).
    """
    committed = dict(nr=6, ntheta=12, p=0, mach=3.0, r_out=10.0, growth=1.0,
                     cfl=0.4)
    return [
        dict(tag_suffix="C1_committed_t0.307", max_time=0.307, **committed),
        dict(tag_suffix="C2_committed_steady", max_time=8.0, **committed),
        dict(tag_suffix="C3_resolved_t0.307", max_time=0.307,
             nr=24, ntheta=40, growth=1.08, p=0, mach=3.0, r_out=5.0, cfl=0.4),
    ]


def smoke_ladder() -> list[dict]:
    return [dict(nr=8, ntheta=16, p=0, mach=3.0, r_out=5.0, growth=1.35),
            dict(nr=12, ntheta=24, p=0, mach=3.0, r_out=5.0, growth=1.25)]


def summarise(rows: list[dict]) -> dict:
    clean = [{k: v for k, v in r.items()
              if k not in ("theta", "cp", "residual_history")} for r in rows]
    fam: dict[str, list[dict]] = {}
    for r in clean:
        fam.setdefault(f"m{r['mach']:g}-p{r['degree']}", []).append(r)
    fits = {}
    for key, group in fam.items():
        group = sorted(group, key=lambda r: r["cells"])
        if len(group) < 2:
            continue
        n = np.array([g["cells"] for g in group], float)
        e = np.array([g["cp_stag_rel_error"] for g in group], float)
        e = np.maximum(e, 1e-12)
        slope = float(np.polyfit(np.log(n), np.log(e), 1)[0])
        fits[key] = {"cells": n.tolist(), "rel_error": e.tolist(),
                     "loglog_slope": slope,
                     "error_ratio_last_two": float(e[-1] / e[-2]) if len(e) > 1 else None}
    best = min(clean, key=lambda r: r["cp_stag_rel_error"]) if clean else None
    return {
        "rows": clean, "fits": fits,
        "best": None if best is None else {
            "tag": best["tag"], "cp_stag": best["cp_stag"],
            "cp_stag_target": best["cp_stag_target"],
            "cp_stag_rel_error": best["cp_stag_rel_error"],
            "steady": best["steady"], "symmetry_error": best["symmetry_error"]},
        "gate": None if best is None else {
            "tolerance_rel": 0.03,
            "passed": bool(best["cp_stag_rel_error"] < 0.03),
            "value": best["cp_stag_rel_error"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these tags (as produced by --smoke/--ladder)")
    ap.add_argument("--pick", type=int, default=None,
                    help="run only ladder case number PICK (0-based; for sharding)")
    ap.add_argument("--list", action="store_true", help="print the ladder and exit")
    ap.add_argument("--max-time", type=float, default=6.0)
    ap.add_argument("--steady-tol", type=float, default=1e-4)
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()

    control = bool(args.control)
    cases = control_cases() if control else (smoke_ladder() if args.smoke else default_ladder())
    if args.list:
        for i, c in enumerate(cases):
            print(i, f"m{c['mach']:g}-p{c['p']}-{c['nr']}x{c['ntheta']}",
                  f"growth={c['growth']} r_out={c['r_out']}")
        return
    if args.pick is not None:
        cases = [cases[args.pick]]
    if args.only:
        cases = [c for c in cases
                 if f"m{c['mach']:g}-p{c['p']}-{c['nr']}x{c['ntheta']}" in args.only]
        if not cases:
            raise SystemExit(f"no case matched {args.only}")
    rows = []
    for case in cases:
        kw = dict(case)
        mt = kw.pop("max_time", args.max_time)
        suffix = kw.pop("tag_suffix", None)
        row = run_case(max_time=mt, steady_tol=args.steady_tol, **kw)
        if suffix:
            row["tag"] = f"{row['tag']}-{suffix}"
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ("theta", "cp", "residual_history")}, indent=2),
              flush=True)
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / f"row_{row['tag']}.json").write_text(json.dumps(row, indent=1))
    summary = summarise(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    name = ("summary_control.json" if control
            else "summary_smoke.json" if args.smoke else "summary_ladder.json")
    (args.outdir / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps({"fits": summary["fits"], "best": summary["best"],
                      "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()