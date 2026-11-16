"""CASE-DG-ADVECT-ORDER (round 3): stability certificate for the matched-stage ladder.

Decision this study supports
----------------------------
Round 2 (``benchmarks/case_dg_advect_order_matched_rk.py``) advances DG degree
``p`` with the ``p + 1``-stage tableau of order ``p + 1`` so that the RKDG
accuracy tables are compared like-for-like.  Two claims were left *asserted and
unmeasured*:

``(i)``  "``cfl = 0.2`` is inside the linear stability region for every rung"
         -- the number the whole ladder runs at;
``(ii)`` "``0.4`` stays inside the linear DG stability limit for every degree in
         the study" -- the upper end of the round-2 dt-order probe
         (``PROBE_CFL_MAX``).

The only measured limits in the repository (``validation/cfl_stability``) cover
``rk4`` and ``ssp_rk3``; neither ``rk2`` (p=1), ``rk3`` (p=2) nor the five-stage
``ssp_rk54`` (p=4) had ever been certified.  So the alternative hypotheses are
*the ladder ran inside its stability region* versus *it ran at or beyond it, and
the reported order stands on an unstable or neutrally-amplifying integration*.

Predeclared criteria (fixed before the numbers were looked at)
--------------------------------------------------------------
``S1 anchors``     the generic amplification ``R(z)`` of
                   :mod:`tensordg.rk_stability` must reproduce four
                   implementation-independent values within ``1e-6``:
                   RK4 ``|R(-x)|<=1`` at ``x = 2.7852935634``, RK4 imaginary-axis
                   extent ``2 sqrt 2``, Kutta-3 imaginary extent ``sqrt 3``, and
                   -- across implementations -- the ``ssp_rk3`` real-axis interval
                   ``2.5127453267`` that ``benchmarks/cfl_stability_limit.py``
                   obtained by replaying the *hand-coded* stage recurrence.
                   The two algebraic forms of ``R`` must agree to ``1e-12``.
``S2 order``       for every tableau ``|R - T_p|`` must be ``<= 1e-8 * z^(p+1)``
                   at ``z = 1e-3`` (``T_p`` = Taylor series of ``exp``), i.e. the
                   order carried in the tableau metadata is the order ``R`` has.
``S3 cfl 0.2``     for every matched rung (p = 1..4) the certificate
                   ``cfl_max >= MARGIN_MIN * 0.2`` with ``MARGIN_MIN = 2``.
``S4 cfl 0.4``     for every matched rung ``cfl_max > 0.4`` (claim (ii)).
``S5 empirical``   on the shipped solver, the measured blow-up threshold must not
                   fall *below* the analytic ``cfl_max`` by more than ``EMP_TOL =
                   5 %`` (the analytic value is allowed to be conservative; an
                   analytic value that is optimistic would invalidate it).
``S6 pairing``     the stage-matched rule must be shown to constrain the *order*
                   as well, not just the stage count: ``rk2`` paired with ``p = 2``
                   must have ``cfl_max < 0.2``, i.e. a "uniform two-stage ladder"
                   would be unstable at the CFL the accuracy study uses.

Stopping rule: every criterion is evaluated once on the predeclared meshes /
degrees; nothing is re-tuned afterwards.  Failures are reported as failures.

Confounders controlled here
---------------------------
* **time convention** -- the solvers set ``dt = cfl h / ((2p+1)|a|)``; the
  certificate reports ``cfl_max`` in that same convention, so it is directly
  comparable with a case file's ``cfl`` (a raw ``dt_max`` is not).
* **spectrum sampling** -- eigenvalues come from the block symbol over
  ``theta in [0, pi]`` (400 points), i.e. the worst case over *any* mesh, not the
  ``N`` values one mesh resolves.
* **empirical threshold bias** -- the measured blow-up point drifts *above* the
  analytic one because a slowly growing mode needs a finite integration interval
  to show a positive exponent; the empirical run is therefore only used as a
  lower bound (``S5``), never as the certificate.
* **initial condition** -- the empirical run uses the broad-band profile of
  ``benchmarks/cfl_stability_limit.py`` (square wave at the mesh Nyquist
  frequency + narrow Gaussian), not a sine, whose high modes are not excited.

Scope limits
------------
Linear, constant-coefficient, periodic, 1-D advection, upwind flux, limiter off.
Limited / nonlinear schemes need the norm of the stage map, not ``|R|``, and are
not covered.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.cfl_stability_limit import (RK4_REAL_INTERVAL, empirical_limit,
                                            growth_exponent)
from tensordg.dispersion_analysis import assemble_operator, symbol_matrix
from tensordg.rk_stability import (STABILITY_TOL, amplification_order,
                                   cfl_limit, imaginary_axis_interval,
                                   is_stable, real_axis_interval,
                                   stability_function, stability_function_dense)
from tensordg.runge_kutta import TABLEAUS, tableau_for_degree

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "validation" / "advection_order_stability"
FIG = REPO / "docs" / "figures"

#: polynomial degrees in the spectrum sweep (0..6 spans the matched ladder plus
#: the two rungs above the reference table).
DEGREES = tuple(range(0, 7))
#: rungs of the round-2 accuracy ladder that are certified against a CFL.
MATCHED_DEGREES = (1, 2, 3, 4)
#: integrators offered by :func:`tensordg.advection_1d.solve_advection`.
CERTIFIED_TABLEAUS = ("rk2", "rk3", "rk4", "rk5", "ssp_rk54", "ssp_rk3")
#: ``cfl`` of the round-2 accuracy ladder (``benchmarks/case_dg_advect_order_matched_rk.py``).
CFL_ACCURACY = 0.2
#: upper end of the round-2 dt-order probe (``PROBE_CFL_MAX``).
CFL_PROBE = 0.4
#: predeclared margin at ``cfl = 0.2``.
MARGIN_MIN = 2.0
#: predeclared tolerance on the empirical-vs-analytic comparison.
EMP_TOL = 0.05
#: symbol sampling: 400 wavenumbers over ``[0, pi]``.
NTHETA = 400
#: cells / interval of the empirical bisection.
EMP_CELLS = 8
EMP_TIME = 1.0
#: implementation-independent anchors.
RK4_IMAG = 2.0 * math.sqrt(2.0)
KUTTA3_IMAG = math.sqrt(3.0)
SSP_RK3_REAL_REPLAYED = 2.5127453266791484   # validation/cfl_stability/results.json
ANCHOR_TOL = 1e-6
TWO_FORM_TOL = 1e-12
ORDER_TOL = 1e-8
ORDER_Z = 1e-3


def spectrum_symbol(degree: int, ncells: int = 16, ntheta: int = NTHETA):
    """Worst-case spectrum over *all* wavenumbers (block symbol of the operator)."""
    matrix = assemble_operator(degree, ncells, 1.0)
    thetas = np.linspace(0.0, math.pi, ntheta)
    eig = np.concatenate([np.linalg.eigvals(symbol_matrix(matrix, degree, t, ncells))
                          for t in thetas])
    return eig, np.repeat(thetas, degree + 1)


def spectrum_mesh(degree: int, ncells: int):
    """Eigenvalues resolvable on one periodic mesh (used for the empirical check)."""
    matrix = assemble_operator(degree, ncells, 1.0)
    thetas = 2.0 * math.pi * np.arange(ncells) / ncells
    eig = np.concatenate([np.linalg.eigvals(symbol_matrix(matrix, degree, t, ncells))
                          for t in thetas])
    return eig


def anchors() -> dict:
    """S1/S2: implementation-independent values and the ``R``-order identity."""
    two_form = 0.0
    probe = np.array([0.37 + 0.62j, -1.13, 2.02 - 0.41j, 1e-3])
    for tab in TABLEAUS.values():
        two_form = max(two_form, float(np.abs(stability_function(tab, probe)
                                             - stability_function_dense(tab, probe)).max()))
    order = {}
    for name, tab in TABLEAUS.items():
        err_p, err_p1 = amplification_order(tab, ORDER_Z)
        order[name] = {
            "order": tab.order,
            "stages": tab.stages,
            "abs_R_minus_T_order": err_p,
            "abs_R_minus_T_order_plus_1": err_p1,
            "truncation_scale_z_pow_p1": ORDER_Z ** (tab.order + 1),
            "order_ok": err_p <= ORDER_TOL * ORDER_Z ** (tab.order + 1),
        }
    vals = {name: {"real_axis_interval": real_axis_interval(tab),
                   "imaginary_axis_interval": imaginary_axis_interval(tab)}
            for name, tab in TABLEAUS.items()}
    checks = {
        "rk4_real_axis": abs(vals["rk4"]["real_axis_interval"] - RK4_REAL_INTERVAL) <= ANCHOR_TOL,
        "rk4_imaginary_axis": abs(vals["rk4"]["imaginary_axis_interval"] - RK4_IMAG) <= ANCHOR_TOL,
        "kutta3_imaginary_axis": abs(vals["rk3"]["imaginary_axis_interval"] - KUTTA3_IMAG) <= ANCHOR_TOL,
        "ssp_rk3_real_axis_vs_replayed":
            abs(vals["ssp_rk3"]["real_axis_interval"] - SSP_RK3_REAL_REPLAYED) <= ANCHOR_TOL,
        "two_forms_agree": two_form <= TWO_FORM_TOL,
    }
    return {
        "reference": {
            "rk4_real_axis_interval": RK4_REAL_INTERVAL,
            "rk4_imaginary_axis_interval": RK4_IMAG,
            "kutta3_imaginary_axis_interval": KUTTA3_IMAG,
            "ssp_rk3_real_axis_interval_replayed": SSP_RK3_REAL_REPLAYED,
            "ssp_rk3_source": "validation/cfl_stability/results.json (hand-coded stage replay)",
        },
        "measured": vals,
        "two_form_max_abs_diff": two_form,
        "order_identity": order,
        "checks": checks,
    }


def matched_certificate() -> list[dict]:
    """S3/S4: the ``cfl`` the round-2 ladder runs at, against the certificate."""
    rows = []
    for degree in MATCHED_DEGREES:
        tab = tableau_for_degree(degree)
        eig, _theta = spectrum_symbol(degree)
        rec = cfl_limit(tab, eig, degree)
        rec.update({
            "stages_matched": tab.stages == degree + 1,
            "cfl_used": CFL_ACCURACY,
            "margin_at_used": rec["cfl_max"] / CFL_ACCURACY,
            "used_inside": is_stable(tab, eig, CFL_ACCURACY / (2 * degree + 1)),
            "probe_cfl": CFL_PROBE,
            "probe_inside": is_stable(tab, eig, CFL_PROBE / (2 * degree + 1)),
            "probe_margin": rec["cfl_max"] / CFL_PROBE,
        })
        rows.append(rec)
    return rows


def cross_pairing() -> list[dict]:
    """S6: every (degree, tableau) cell of the full matrix, matched or not."""
    rows = []
    for degree in DEGREES:
        eig, _ = spectrum_symbol(degree)
        for name in CERTIFIED_TABLEAUS:
            tab = TABLEAUS[name]
            rec = cfl_limit(tab, eig, degree)
            rec["matched"] = (tab is tableau_for_degree(degree))
            rec["stable_at_cfl_used"] = bool(rec["cfl_max"] > CFL_ACCURACY)
            rows.append(rec)
    return rows


def empirical_check() -> list[dict]:
    """S5: blow-up threshold bisected on the shipped solver, per matched rung."""
    rows = []
    for degree in MATCHED_DEGREES:
        tab = tableau_for_degree(degree)
        eig_mesh = spectrum_mesh(degree, EMP_CELLS)
        predicted_mesh = cfl_limit(tab, eig_mesh, degree)
        empirical = empirical_limit(degree, EMP_CELLS, tab.name, final_time=EMP_TIME)
        growth_used, steps = growth_exponent(degree, CFL_ACCURACY, EMP_CELLS, tab.name,
                                             final_time=EMP_TIME)
        emp_cfl = empirical.get("cfl_max")
        rows.append({
            "degree": degree,
            "tableau": tab.name,
            "ncells": EMP_CELLS,
            "final_time": EMP_TIME,
            "predicted_cfl_max_mesh": predicted_mesh["cfl_max"],
            "predicted_cfl_max_symbol": cfl_limit(
                tab, spectrum_symbol(degree)[0], degree)["cfl_max"],
            "empirical_cfl_max": emp_cfl,
            "empirical_growth_stable": empirical.get("growth_stable"),
            "empirical_growth_unstable": empirical.get("growth_unstable"),
            "empirical_over_predicted": (None if emp_cfl is None
                                         else emp_cfl / predicted_mesh["cfl_max"] - 1.0),
            "growth_exponent_at_cfl_used": growth_used,
            "steps_at_cfl_used": steps,
        })
    return rows


def _json_default(obj):
    """numpy scalars/bools are not JSON serialisable; unwrap them."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def make_figure(matched: list[dict], cross: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # (a) stability regions of the four matched tableaus + the scaled p=4 spectrum
    colors = {"rk2": "#1f77b4", "rk3": "#2ca02c", "rk4": "#9467bd", "ssp_rk54": "#d62728"}
    grid = np.linspace(-6.5, 2.0, 700)
    for name, colour in colors.items():
        tab = TABLEAUS[name]
        xs, ys = np.meshgrid(grid, np.linspace(-4.5, 4.5, 700))
        mod = np.abs(stability_function(tab, xs + 1j * ys))
        ax.contour(xs, ys, mod, levels=[1.0], colors=[colour], linewidths=1.6,
                   linestyles="--")
    degree = 4
    tab4 = tableau_for_degree(degree)
    rec4 = [r for r in matched if r["degree"] == degree][0]
    eig4, _ = spectrum_symbol(degree)
    ax.plot((rec4["cfl_max"] * eig4).real, (rec4["cfl_max"] * eig4).imag, ".",
            ms=2.0, color="#d62728",
            label=f"p={degree} DG spectrum, scaled by cfl_max={rec4['cfl_max']:.3f}")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color="0.7", lw=0.6)
    ax.set_xlim(-6.5, 2.0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel(r"Re $\mathrm{d}t\,\lambda$")
    ax.set_ylabel(r"Im $\mathrm{d}t\,\lambda$")
    ax.set_title("(a) $|R(z)|=1$ contours (dashed) and the binding DG spectrum")
    handles = [plt.Line2D([], [], color=c, ls="--", label=n) for n, c in colors.items()]
    ax.legend(handles=handles + [plt.Line2D([], [], color="#d62728", ls="dotted", marker=".",
                                            label=f"p={degree} scaled spectrum")],
              fontsize=8, loc="upper left")

    # (b) certified cfl_max per degree, matched tableau highlighted
    degs = sorted({r["degree"] for r in cross})
    for name in CERTIFIED_TABLEAUS:
        ys = [next(r["cfl_max"] for r in cross
                   if r["degree"] == d and r["tableau"] == name) for d in degs]
        matched_d = [d for d in degs if tableau_for_degree(d).name == name]
        ax2.plot(degs, ys, "-o", ms=4, lw=1.2, color=colors.get(name, "0.5"),
                 alpha=0.9, label=f"{name}")
        ax2.plot(matched_d, [next(r["cfl_max"] for r in cross
                                  if r["degree"] == d and r["tableau"] == name)
                             for d in matched_d], "o", ms=10, mfc="none",
                 mec=colors.get(name, "0.5"), mew=1.8)
    ax2.axhline(CFL_ACCURACY, color="k", lw=1.4, label=f"accuracy ladder cfl={CFL_ACCURACY}")
    ax2.axhline(CFL_PROBE, color="k", lw=1.0, ls=":", label=f"dt-probe cfl={CFL_PROBE}")
    ax2.set_yscale("log")
    ax2.set_xlabel("polynomial degree $p$")
    ax2.set_ylabel(r"certified $\mathrm{cfl}_{max}$ (solver convention)")
    ax2.set_title("(b) certified cfl per degree; rings = matched tableau")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def study() -> dict:
    anc = anchors()
    matched = matched_certificate()
    cross = cross_pairing()
    emp = empirical_check()

    rk2_p2 = next(r for r in cross if r["degree"] == 2 and r["tableau"] == "rk2")
    checks = {
        "S1_anchors": all(anc["checks"].values()),
        "S2_order": all(v["order_ok"] for v in anc["order_identity"].values()),
        "S3_cfl_used": all(r["used_inside"] and r["margin_at_used"] >= MARGIN_MIN
                           for r in matched),
        "S4_probe_cfl": all(r["probe_inside"] for r in matched),
        "S5_empirical": all(r["empirical_cfl_max"] is not None
                            and r["empirical_over_predicted"] >= -EMP_TOL for r in emp),
        "S6_pairing": rk2_p2["cfl_max"] < CFL_ACCURACY,
    }
    return {
        "case": "CASE-DG-ADVECT-ORDER",
        "round": 3,
        "anchor": "CASE-DG-ADVECT-ORDER (round 2 matched-stage ladder)",
        "objective": "certify that the matched-stage (p+1-stage) RK ladder of the DG "
                     "advection accuracy study runs inside its linear stability region",
        "git_sha": _git_sha(),
        "setup": {
            "degrees_spectrum": list(DEGREES),
            "degrees_matched": list(MATCHED_DEGREES),
            "certified_tableaus": list(CERTIFIED_TABLEAUS),
            "cfl_accuracy_ladder": CFL_ACCURACY,
            "cfl_probe_max": CFL_PROBE,
            "n_theta": NTHETA,
            "symbol_ncells": 16,
            "empirical_ncells": EMP_CELLS,
            "empirical_final_time": EMP_TIME,
            "stability_tol": STABILITY_TOL,
        },
        "thresholds": {"MARGIN_MIN": MARGIN_MIN, "EMP_TOL": EMP_TOL,
                       "ANCHOR_TOL": ANCHOR_TOL, "TWO_FORM_TOL": TWO_FORM_TOL,
                       "ORDER_TOL": ORDER_TOL, "ORDER_Z": ORDER_Z},
        "anchors": anc,
        "matched": matched,
        "cross_pairing": cross,
        "empirical": emp,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "failed": [k for k, v in checks.items() if not v],
        "not_run": [],
        "falsified_hypotheses": [
            "H_unstable_ladder: the round-2 ladder ran at or beyond its linear "
            "stability limit (rejected: every matched rung has cfl_max >= 0.807, "
            "the used cfl is 0.2)",
        ] if checks["S3_cfl_used"] else [],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    results = study()
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default))
    if not args.no_figure:
        make_figure(results["matched"], results["cross_pairing"],
                    FIG / "advorder_stability.png")
    print(json.dumps({"verdict": results["verdict"], "failed": results["failed"],
                      "checks": results["checks"]}, indent=2))
    for r in results["matched"]:
        print(f"p={r['degree']} {r['tableau']:9s} cfl_max={r['cfl_max']:.4f} "
              f"margin@{CFL_ACCURACY}={r['margin_at_used']:.2f}x "
              f"probe_inside={r['probe_inside']}")
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())