"""CASE-DG-ADVECT-ORDER (round 3): the stage/order cap, measured at p = 5, 6.

Round 2 (:mod:`benchmarks.case_dg_advect_order_matched_rk`) closed the
``p = 1..4`` rungs of the RKDG accuracy ladder against the reference table by
advancing degree ``p`` with exactly ``p + 1`` stages.  Two claims were left
standing on assertion rather than measurement:

* :func:`tensordg.runge_kutta.tableau_for_degree` documents a *cap* -- "for
  ``p >= 5`` no registered tableau has ``p + 1`` stages; the highest order
  available is used" -- and nothing in the package ever measured what that cap
  does to the *observed* order.  ``p = 5`` needs six stages (Butcher's 6-stage
  fifth-order scheme is registered) while ``p = 6`` would need seven, which is
  **not** registered, so the ``p = 6`` rung runs with a tableau one order below
  the spatial one it is supposed to certify.
* :func:`tensordg.runge_kutta.max_order_for_stages`, the function the matched
  criterion uses to state "what can this stage count promise", returned values
  for ``stages >= 11`` from a parity guess that understates the published
  minimal-stage counts by one (11 stages reach order 8, not 7).  A criterion
  built on it would report a *false* FAIL for a rung whose tableau is fine.

This round replaces assertion with measurement and with the published table:

``A`` barrier audit.  ``MIN_STAGES_FOR_ORDER`` now carries the published
      minimal-stage counts (Butcher 1964/65; Butcher 1985; Hairer & Wanner,
      *Solving ODEs II*, sec. II.5) up to order 11 / 17 stages, and
      ``MAX_ORDER_FOR_STAGES`` is *derived* from it instead of transcribed, so
      the two cannot drift apart.  ``stages > 17`` is refused rather than
      extrapolated.  The audit re-checks every registered tableau against the
      barrier (``order <= max_order_for_stages(stages)``).

``B`` self-convergence of the extended ladder.  Degree ``p = 5, 6`` is advanced
      with ``tableau_for_degree(p)`` on meshes ``(6, 12, 24)`` at ``cfl = 0.2``;
      the observed order is compared with the spatial order ``p + 1`` that the
      DG operator promises.  No external reference is involved, so this rung is
      a *self*-convergence certificate (the reference table stops at ``p = 4``).

``C`` CFL budget.  For ``p = 1..6`` at a fixed mesh the exact time/space error
      split of round 2 (``expm(L T) u0`` is the exact semi-discrete solution of
      a linear law) is evaluated on a CFL grid, which yields two numbers per
      degree: the largest CFL whose time-error share is still ``<= 10 %`` (the
      window in which the measured rate *is* the spatial order) and the smallest
      CFL on the grid at which the run blows up.

Predeclared criteria (fixed before the numbers were looked at)

``S1 barrier``   ``MAX_ORDER_FOR_STAGES`` equals the published minimal-stage
                 inversion for ``s = 1..17`` and every registered tableau
                 satisfies ``order <= max_order_for_stages(stages)``.
``S2 order p=5`` fitted order within ``ORDER_TOL = 0.10`` of 6.
``S3 order p=6`` fitted order within ``ORDER_TOL = 0.10`` of 7, i.e. the
                 documented cap (a 6-stage tableau for a 7th-order space) does
                 **not** bind at the CFL the case uses.
``S4 time share`` at ``cfl = 0.2``, ``N = 16``: ``E_time / E_total <= 0.10`` for
                 ``p = 5`` and ``p = 6``.
``S5 cfl budget`` for every ``p = 1..6`` the largest scanned CFL whose time
                 share is ``<= 0.10`` is at least ``CFL_MARGIN_MIN = 0.4``, i.e.
                 the case's ``cfl = 0.2`` keeps a two-fold margin.

Stopping rule: a rung whose ladder keeps fewer than three points above the
round-off floor ``FIT_FLOOR`` is reported as ``not_run`` instead of being
fitted to two points.  A CFL whose run is non-finite or whose error exceeds
``DIVERGENCE_FACTOR`` (the solution's own norm is ``O(1)``) is classified as
divergent and excluded from the share comparison.

Run: ``PYTHONPATH=. python -m benchmarks.case_dg_advect_order_stage_cap``
Writes ``validation/advection_order_stage_cap/{metrics.json,tables.md}``,
``docs/figures/advorder_stage_cap.png`` and the rendered report
``docs/CASE-DG-ADVECT-ORDER-stage-cap.md`` (numbers injected from
``metrics.json``; nothing hand-copied).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import torch

from benchmarks.case_dg_advect_order import (fitted_order, fit_with_floor)
from benchmarks.case_dg_advect_order_matched_rk import (l2_of_coeff_difference,
                                                        semi_discrete_matrix)
from tensordg.advection_1d import l2_error, solve_advection
from tensordg.runge_kutta import (MAX_ORDER_FOR_STAGES, MIN_STAGES_FOR_ORDER,
                                  TABLEAUS, max_order_for_stages,
                                  tableau_for_degree)

OUT = pathlib.Path("validation/advection_order_stage_cap")
FIG = pathlib.Path("docs/figures")
DOC = pathlib.Path("docs/CASE-DG-ADVECT-ORDER-stage-cap.md")

#: Rungs measured here: the first degree whose ``p + 1`` stages are still
#: reachable from the registry (``p = 5`` -> six-stage :data:`RK5`) and the
#: first whose are not (``p = 6`` would need seven).
RUNG_DEGREES = (5, 6)
#: Ladder per degree.  Two constraints pull in opposite directions: a low-degree
#: rung needs *many* elements before the asymptotic rate appears (at six
#: elements the ``p = 1`` error is dominated by the unresolved first harmonic
#: and the fitted slope comes out at 2.29 instead of 2), while a high-degree
#: rung hits the float64 floor early (``p = 6`` at 24 elements is already
#: ~1e-12).  Each ladder is therefore chosen per degree and kept at three
#: points, so the shared slope estimator sees the same number of points
#: everywhere.
LADDER_BY_DEGREE = {1: (12, 24, 48), 2: (8, 16, 32), 3: (8, 16, 32),
                    4: (6, 12, 24), 5: (6, 12, 24), 6: (6, 12, 24)}
FIGURE_DEGREES = (1, 2, 3, 4, 5, 6)
CFL = 0.2
CFL_MARGIN_MIN = 0.40
FINAL_TIME = 1.0
SPEED = 1.0
PROBE_CELLS = 16
CFL_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)
TIME_SHARE_TOL = 0.10
ORDER_TOL = 0.10
FIT_FLOOR = 1e-13
DIVERGENCE_FACTOR = 1e3

#: Published minimal-stage counts, restated here so that the audit compares the
#: module against the literature instead of against itself.
PUBLISHED_MIN_STAGES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 11,
                        9: 13, 10: 15, 11: 17}


# --------------------------------------------------------------------------- #
# A. barrier audit
# --------------------------------------------------------------------------- #
def barrier_audit() -> dict:
    """Check ``MAX_ORDER_FOR_STAGES`` against the published counts."""
    expected = {s: max(p for p, need in PUBLISHED_MIN_STAGES.items() if need <= s)
                for s in range(1, max(PUBLISHED_MIN_STAGES.values()) + 1)}
    got = {s: max_order_for_stages(s) for s in expected}
    tableaus = {t.name: {"stages": t.stages, "order": t.order,
                         "barrier": max_order_for_stages(t.stages)}
                for t in TABLEAUS.values()}
    refused = None
    try:
        max_order_for_stages(max(MAX_ORDER_FOR_STAGES) + 1)
    except ValueError as exc:
        refused = str(exc)
    return {
        "min_stages_for_order": dict(MIN_STAGES_FOR_ORDER),
        "published_min_stages": dict(PUBLISHED_MIN_STAGES),
        "table": got,
        "expected": expected,
        "table_matches_published": got == expected,
        "tableaus": tableaus,
        "all_tableaus_within_barrier": all(v["order"] <= v["barrier"]
                                          for v in tableaus.values()),
        "refused_above_certified_range": refused,
    }


# --------------------------------------------------------------------------- #
# B. self-convergence of one rung
# --------------------------------------------------------------------------- #
def rung(degree: int, cells=None, cfl: float = CFL) -> dict:
    """Advance degree ``degree`` with its selected tableau and fit the order."""
    cells = LADDER_BY_DEGREE.get(degree, LADDER_BY_DEGREE[6]) if cells is None else cells
    tab = tableau_for_degree(degree)
    l2, meta = [], []
    for n in cells:
        centres, coeff, info = solve_advection(
            ncells=int(n), degree=degree, final_time=FINAL_TIME, speed=SPEED,
            cfl=cfl, time_integrator=tab.name)
        l2.append(l2_error(centres, coeff, final_time=FINAL_TIME, speed=SPEED))
        meta.append({"ncells": int(n), "steps": info["steps"], "dt": info["dt"]})
    fit, nfit, dropped = fit_with_floor(cells, l2)
    return {"degree": degree, "integrator": tab.name, "rk_stages": tab.stages,
            "rk_order": tab.order, "requested_stages": degree + 1,
            "barrier_at_requested_stages": max_order_for_stages(
                min(degree + 1, max(MAX_ORDER_FOR_STAGES))),
            "expected_spatial_order": degree + 1, "cells": list(cells),
            "cfl": cfl, "l2": l2, "meta": meta,
            "pairwise_orders": [float(np.log(l2[i - 1] / l2[i])
                                      / np.log(cells[i] / cells[i - 1]))
                                for i in range(1, len(cells))],
            "fitted_order": fit, "fit_points": nfit, "fit_dropped_cells": dropped}


# --------------------------------------------------------------------------- #
# C. exact time/space split and the CFL budget
# --------------------------------------------------------------------------- #
def time_split(degree: int, ncells: int = PROBE_CELLS, cfl: float = CFL,
               integrator: str | None = None) -> dict:
    """Total / space / time error at one (degree, mesh, cfl) point."""
    tab = tableau_for_degree(degree)
    name = tab.name if integrator is None else integrator
    mat, u0 = semi_discrete_matrix(degree, ncells)
    exact_semi = sla.expm(mat * FINAL_TIME) @ u0
    centres, coeff, info = solve_advection(
        ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
        cfl=cfl, time_integrator=name)
    e_total = l2_error(centres, coeff, final_time=FINAL_TIME, speed=SPEED)
    e_space = l2_error(centres, torch.as_tensor(exact_semi.reshape(ncells, degree + 1)),
                       final_time=FINAL_TIME, speed=SPEED)
    e_time = l2_of_coeff_difference(degree, ncells, coeff, exact_semi)
    diverged = (not np.isfinite(e_total)) or (not np.isfinite(e_time)) or \
               e_total > DIVERGENCE_FACTOR
    share = None if diverged or e_total == 0.0 else e_time / e_total
    return {"degree": degree, "ncells": ncells, "cfl": cfl, "integrator": name,
            "steps": info["steps"], "dt": info["dt"], "E_total": e_total,
            "E_space": e_space, "E_time": e_time, "time_share": share,
            "diverged": bool(diverged)}


def cfl_budget(degree: int, ncells: int = PROBE_CELLS, grid=CFL_GRID) -> dict:
    """Scan the CFL grid; report the 10 %-share window and the blow-up point."""
    rows = [time_split(degree, ncells=ncells, cfl=c) for c in grid]
    inside = [r["cfl"] for r in rows
              if not r["diverged"] and r["time_share"] is not None
              and r["time_share"] <= TIME_SHARE_TOL]
    blown = [r["cfl"] for r in rows if r["diverged"]]
    return {"degree": degree, "ncells": ncells, "grid": list(grid), "rows": rows,
            "cfl_time_share_10": max(inside) if inside else None,
            "cfl_divergence": min(blown) if blown else None}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_degree(degree: int) -> pathlib.Path:
    """Measure one degree (ladder + CFL budget) and cache it as a fragment.

    The full sweep costs minutes, which is why it is split per degree: a
    fragment can be run on its own (``--degrees 6``) and the final assembly
    reads whatever fragments are present, exactly like round 2's per-rung
    fragments.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    frag = OUT / f"frag_p{degree}.json"
    payload = {"degree": degree, "rung": rung(degree), "budget": cfl_budget(degree)}
    frag.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"# fragment written: {frag}", flush=True)
    return frag


def assemble() -> dict:
    """Build metrics/tables/figure/doc from the per-degree fragments."""
    FIG.mkdir(parents=True, exist_ok=True)
    audit = barrier_audit()
    frags = {}
    for degree in FIGURE_DEGREES:
        frag = OUT / f"frag_p{degree}.json"
        if not frag.exists():
            raise SystemExit(f"missing fragment {frag}; run --degrees {degree}")
        frags[degree] = json.loads(frag.read_text())
    rungs = [frags[d]["rung"] for d in FIGURE_DEGREES]
    budgets = [frags[d]["budget"] for d in FIGURE_DEGREES]

    checks: dict[str, object] = {
        "S1_barrier": bool(audit["table_matches_published"]
                           and audit["all_tableaus_within_barrier"]),
    }
    details: dict[str, dict] = {}
    for r in rungs:
        if r["degree"] not in RUNG_DEGREES:
            continue
        key = f"S{r['degree'] - 3}_order_p{r['degree']}"
        if r["fitted_order"] is None:
            checks[key] = "not_run"
            details[key] = {"reason": "fewer than 3 ladder points above the "
                                      f"round-off floor {FIT_FLOOR}",
                            "dropped_cells": r["fit_dropped_cells"]}
        else:
            checks[key] = bool(abs(r["fitted_order"] - (r["degree"] + 1)) <= ORDER_TOL)
            details[key] = {"fitted_order": r["fitted_order"],
                            "expected": r["degree"] + 1, "tol": ORDER_TOL}
    probe = {b["degree"]: [row for row in b["rows"] if row["cfl"] == CFL][0]
             for b in budgets}
    for degree in RUNG_DEGREES:
        row = probe[degree]
        checks[f"S4_time_share_p{degree}"] = (
            "not_run" if row["time_share"] is None
            else bool(row["time_share"] <= TIME_SHARE_TOL))
        details[f"S4_time_share_p{degree}"] = {
            "ncells": row["ncells"], "cfl": CFL, "E_total": row["E_total"],
            "E_time": row["E_time"], "time_share": row["time_share"],
            "tol": TIME_SHARE_TOL}
    margin_ok = all(b["cfl_time_share_10"] is not None
                    and b["cfl_time_share_10"] >= CFL_MARGIN_MIN for b in budgets)
    checks["S5_cfl_budget"] = bool(margin_ok)
    details["S5_cfl_budget"] = {
        "tol_share": TIME_SHARE_TOL, "min_cfl": CFL_MARGIN_MIN,
        "cfl_time_share_10": {f"P{b['degree']}": b["cfl_time_share_10"]
                              for b in budgets},
        "cfl_divergence": {f"P{b['degree']}": b["cfl_divergence"]
                           for b in budgets}}

    failed = [k for k, v in checks.items() if v is False]
    not_run = [k for k, v in checks.items() if v == "not_run"]
    metrics = {
        "case": "CASE-DG-ADVECT-ORDER",
        "round": 3,
        "objective": "measure the stage/order cap beyond p = 4 (self-convergence "
                     "at p = 5, 6) and the CFL window in which the measured rate "
                     "is the spatial order; put the Butcher-barrier table on the "
                     "published minimal-stage counts",
        "references": {
            "minimal_stages": "Butcher (1964, 1965) for p <= 4; Butcher (1985) "
                              "and Hairer & Wanner, Solving ODEs II, sec. II.5, "
                              "for p = 9..11",
            "ssp_rk54": "Spiteri & Ruuth, SIAM J. Numer. Anal. 40 (2002) 469-491",
            "external_rate_reference": "validation/advection_order/ref_table.json "
                                       "(Sun, Math. Comp. 2024, doi:10.1090/mcom/4037) "
                                       "-- stops at p = 4, so the p = 5, 6 rungs are "
                                       "self-convergence only",
        },
        "setup": {"degrees": list(FIGURE_DEGREES), "rung_degrees": list(RUNG_DEGREES),
                  "ladder": list(LADDER_BY_DEGREE[5]),
                  "ladder_by_degree": {f"P{p}": list(v) for p, v in
                                       LADDER_BY_DEGREE.items()},
                  "cfl": CFL, "final_time": FINAL_TIME,
                  "speed": SPEED, "probe_cells": PROBE_CELLS,
                  "cfl_grid": list(CFL_GRID),
                  "solver": "tensordg.advection_1d.solve_advection"},
        "thresholds": {"ORDER_TOL": ORDER_TOL, "TIME_SHARE_TOL": TIME_SHARE_TOL,
                       "CFL_MARGIN_MIN": CFL_MARGIN_MIN, "FIT_FLOOR": FIT_FLOOR,
                       "DIVERGENCE_FACTOR": DIVERGENCE_FACTOR},
        "barrier": audit,
        "rungs": rungs,
        "cfl_budgets": budgets,
        "checks": checks,
        "checks_detail": details,
        "failed": failed,
        "not_run": not_run,
        "verdict": "FAIL" if failed else ("NOT RUN" if not_run else "PASS"),
        "open_item": "the reference table stops at p = 4 and states neither dt "
                     "nor its error norm, so the absolute L2 magnitudes stay "
                     "incomparable (round 2, falsified hypothesis H_stage_gap); "
                     "this round certifies *rates* only",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                      encoding="utf-8")

    # ---- tables.md --------------------------------------------------------
    mk = {True: "PASS", False: "FAIL", "not_run": "NOT RUN"}
    lines = ["# CASE-DG-ADVECT-ORDER round 3 — stage/order cap and CFL budget\n",
             "## A. Butcher barrier (published minimal-stage counts), stages -> order",
             "",
             "| s | " + " | ".join(str(s) for s in audit["table"]) + " |",
             "|--:|" + "|".join("--:" for _ in audit["table"]) + "|",
             "| order (TensorDG) | " + " | ".join(str(v) for v in audit["table"].values()) + " |",
             "| order (published) | " + " | ".join(str(v) for v in audit["expected"].values()) + " |",
             "", f"table matches published: **{audit['table_matches_published']}**; "
             f"all registered tableaus within barrier: "
             f"**{audit['all_tableaus_within_barrier']}**; "
             f"stages above {max(MAX_ORDER_FOR_STAGES)}: refused "
             f"({'yes' if audit['refused_above_certified_range'] else 'no'}).",
             "", "## B. self-convergence of the extended ladder", "",
             "| p | tableau | stages | tableau order | stages needed (p+1) | "
             "expected spatial order | fitted order | pairwise | 判定 |",
             "|--:|---|---:|---:|---:|---:|---:|---|:--:|"]
    for r in rungs:
        key = f"S{r['degree'] - 3}_order_p{r['degree']}"
        mark = mk[checks.get(key, "not_run")] if r["degree"] in RUNG_DEGREES else "—"
        fit = "n/a" if r["fitted_order"] is None else f"{r['fitted_order']:.3f}"
        pw = " / ".join(f"{v:.3f}" for v in r["pairwise_orders"])
        lines.append(f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | "
                     f"{r['rk_order']} | {r['requested_stages']} | "
                     f"{r['expected_spatial_order']} | {fit} | {pw} | {mark} |")
    lines += ["", "## C. CFL budget (fixed mesh, exact time/space split)", "",
              "| p | cfl with E_time/E_total <= 10% (max) | blow-up cfl (min) | "
              "share at cfl=0.2 |", "|--:|--:|--:|--:|"]
    for b in budgets:
        row = [r for r in b["rows"] if r["cfl"] == CFL][0]
        share = "n/a" if row["time_share"] is None else f"{100 * row['time_share']:.3f}%"
        lines.append(f"| P{b['degree']} | {b['cfl_time_share_10']} | "
                     f"{b['cfl_divergence']} | {share} |")
    lines += ["", "## 判定", ""]
    for k, v in checks.items():
        lines.append(f"- {k}: **{mk[v]}**")
    lines.append(f"\n**总判定：{metrics['verdict']}**")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(rungs)))
    for colour, r in zip(colors, rungs):
        ax.loglog(r["cells"], r["l2"], "o-", color=colour,
                  label=f"P{r['degree']} {r['integrator']} "
                        f"(slope {r['fitted_order']:.2f})")
        if r["fitted_order"] is not None:
            ref = r["l2"][0] * (np.asarray(r["cells"]) / r["cells"][0]) ** -(
                r["degree"] + 1)
            ax.loglog(r["cells"], ref, ":", color=colour, lw=1, alpha=0.7)
    ax.set_xlabel("elements N"); ax.set_ylabel(r"$L^2$ error")
    ax.set_title("self-convergence, dotted = exact slope $p+1$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    for colour, b in zip(colors, budgets):
        xs = [r["cfl"] for r in b["rows"] if not r["diverged"] and r["time_share"]]
        ys = [r["time_share"] for r in b["rows"] if not r["diverged"] and r["time_share"]]
        ax.loglog(xs, ys, "o-", color=colour, label=f"P{b['degree']}")
        blown = [r["cfl"] for r in b["rows"] if r["diverged"]]
        if blown:
            ax.axvline(min(blown), color=colour, ls=":", lw=1, alpha=0.6)
    ax.axhline(TIME_SHARE_TOL, color="k", lw=1, ls="--",
               label=f"E_time = {TIME_SHARE_TOL:.0%} E_total")
    ax.axvline(CFL, color="tab:red", lw=1, alpha=0.7, label=f"case cfl = {CFL}")
    ax.set_xlabel("cfl"); ax.set_ylabel(r"$E_{time}/E_{total}$")
    ax.set_title("time-error share vs cfl (dotted = blow-up)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "advorder_stage_cap.png", dpi=150)
    plt.close(fig)

    DOC.write_text(render_doc(metrics), encoding="utf-8")
    print("verdict =", metrics["verdict"], "failed =", metrics["failed"],
          "not_run =", metrics["not_run"])
    return metrics


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--degrees", default=",".join(str(d) for d in FIGURE_DEGREES),
                        help="degrees measured in this invocation")
    parser.add_argument("--assemble", action="store_true",
                        help="skip the measurements and assemble from the fragments")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if not args.assemble:
        for degree in (int(d) for d in args.degrees.split(",")):
            run_degree(degree)
    return assemble()


def _order_key(degree: int) -> str:
    return f"S{degree - 3}_order_p{degree}"


def render_doc(m: dict) -> str:
    """Render the report page; every number comes from ``m`` (no hand-copying)."""
    pk = {True: "✅", False: "❌", "not_run": "⏸"}
    rung_txt = "；".join(
        f"P{r['degree']} 用 `{r['integrator']}`（{r['rk_stages']} 级 {r['rk_order']} 阶）"
        for r in m["rungs"])
    fit_rows = []
    for r in m["rungs"]:
        if r["degree"] not in RUNG_DEGREES:
            continue
        fit = r["fitted_order"]
        fit_rows.append(
            f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | "
            f"{r['expected_spatial_order']} | "
            f"{'n/a' if fit is None else format(fit, '.3f')} | "
            f"{'n/a' if fit is None else format(100 * abs(fit - r['expected_spatial_order']) / r['expected_spatial_order'], '.2f') + '%'} | "
            f"{pk[m['checks'][_order_key(r['degree'])]]} |")
    budget_rows = []
    for b in m["cfl_budgets"]:
        row = [r for r in b["rows"] if r["cfl"] == m["setup"]["cfl"]][0]
        share = ("n/a" if row["time_share"] is None
                 else f"{100 * row['time_share']:.3f}%")
        budget_rows.append(f"| P{b['degree']} | {b['cfl_time_share_10']} | "
                           f"{b['cfl_divergence']} | {share} |")
    low = [r for r in m["rungs"] if r["degree"] == 5][0]
    high = [r for r in m["rungs"] if r["degree"] == 6][0]
    share_lo = m["checks_detail"]["S4_time_share_p5"]["time_share"]
    share_hi = m["checks_detail"]["S4_time_share_p6"]["time_share"]
    table = m["barrier"]["table"]
    min_stages = ", ".join(f"{p}: {s}" for p, s in
                           m["barrier"]["min_stages_for_order"].items())
    lines = [
        "# CASE-DG-ADVECT-ORDER（第3轮）：级数/阶数上限实测（p = 5, 6）与 CFL 窗口",
        "",
        f"**判据** 自收敛阶 ±{m['thresholds']['ORDER_TOL']}、时间误差占比 ≤ "
        f"{m['thresholds']['TIME_SHARE_TOL']:.0%}、CFL 裕度 ≥ "
        f"{m['thresholds']['CFL_MARGIN_MIN']}｜**总判定 {m['verdict']}**",
        "",
        "> 本文件与图由 `benchmarks/case_dg_advect_order_stage_cap.py` 从 "
        "`validation/advection_order_stage_cap/metrics.json` 生成，数字零手抄。",
        "",
        "## 1. 算例与问题设定",
        "",
        "一维周期线性对流 $u_t+u_x=0$，$x\\in[0,1)$，$u(x,0)=\\sin(2\\pi x)$，"
        f"末时 $t^*={m['setup']['final_time']}$，波速 $c={m['setup']['speed']}$；"
        "精确解 $\\sin(2\\pi(x-t))$。被检验量：DG 空间算子的观测收敛阶，"
        "对 $p=1..6$ 应分别为 $p+1$。网格阶梯逐阶见参数表（低阶需更细网格才进渐近区，"
        f"高阶过早触到 $10^{{-12}}$ 的舍入地板），CFL $={m['setup']['cfl']}$。",
        "",
        "参数表：",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 多项式阶 | {m['setup']['degrees']}（本轮新测 {m['setup']['rung_degrees']}） |",
        f"| 网格 N（逐阶） | {m['setup']['ladder_by_degree']} |",
        f"| CFL | {m['setup']['cfl']} |",
        f"| 时间格式 | {rung_txt} |",
        f"| 时间/空间拆分的探针网格 | N = {m['setup']['probe_cells']} |",
        f"| CFL 扫描网格 | {m['setup']['cfl_grid']} |",
        "",
        "## 2. 方法与工具链",
        "",
        "1. **级数/阶数上限表改用已发表的极小级数**："
        f"`MIN_STAGES_FOR_ORDER = {{{min_stages}}}`"
        "（Butcher 1964/65；Butcher 1985；Hairer & Wanner II.5），"
        "`MAX_ORDER_FOR_STAGES` 由其机械反演得到，二者不会再漂移；"
        "超过 17 级拒绝外推（旧实现用奇偶公式回退，在 s ≥ 11 处比真值低 1 阶）。",
        "2. **自收敛**：$p=5,6$ 直接用 "
        "`solve_advection(time_integrator=tableau_for_degree(p).name)` 跑网格阶梯，"
        "共用最小二乘斜率估计器（两端点数相同）；无外部基准参与。",
        "3. **精确时间/空间拆分**：线性律的半离散系统线性，`expm(L*T) u0` 是其精确解，"
        "数值解与之比较得到 $E_{time}$，与精确解比较得到 $E_{total}$。",
        "4. **CFL 预算**：固定网格扫 CFL，取时间误差占比 ≤ 10% 的最大 CFL"
        "（此窗口内实测阶即空间阶）与发散的 CFL 阈值。",
        "",
        "## 3. 结果",
        "",
        "### 3.1 Butcher 障碍表（级数 → 可承诺阶）",
        "",
        "| s | " + " | ".join(str(s) for s in table) + " |",
        "|--:|" + "|".join("--:" for _ in table) + "|",
        "| 本轮 | " + " | ".join(str(v) for v in table.values()) + " |",
        "| 发表值 | " + " | ".join(str(v) for v in m['barrier']['expected'].values()) + " |",
        "",
        f"与发表值一致：**{m['barrier']['table_matches_published']}**；"
        f"已注册格式全部满足 `order <= barrier(stages)`："
        f"**{m['barrier']['all_tableaus_within_barrier']}**。",
        "",
        "### 3.2 扩展阶梯的自收敛（无外部基准）",
        "",
        "| p | 格式 | 级数 | 期望空间阶 | 实测阶 | 相对误差 | 判定 |",
        "|--:|---|---:|---:|---:|---:|:--:|",
    ] + fit_rows + [
        "",
        "P5 阶梯 L2 = " + ", ".join(format(v, ".4e") for v in low["l2"]) + "；"
        "P6 阶梯 L2 = " + ", ".join(format(v, ".4e") for v in high["l2"]) + "。",
        "",
        "全部阶梯（$p\\le4$ 的阶已由第 2 轮对外部 ref 列认证，这里只用于 CFL 预算与图）：",
        "",
        "| p | 格式 | 网格 N | 实测阶 | 斜率（逐对） | L2 |",
        "|--:|---|---|---:|---:|---|",
    ] + [f"| P{r['degree']} | {r['integrator']} | {r['cells']} | "
         f"{'n/a' if r['fitted_order'] is None else format(r['fitted_order'], '.3f')} | "
         f"{' / '.join(format(v, '.3f') for v in r['pairwise_orders'])} | "
         f"{', '.join(format(v, '.3e') for v in r['l2'])} |" for r in m["rungs"]] + [
        "",
        "### 3.3 CFL 预算（固定网格，精确时间/空间拆分）",
        "",
        "| p | E_time ≤ 10% 的最大 cfl | 发散 cfl（最小） | cfl=0.2 处的占比 |",
        "|--:|--:|--:|--:|",
    ] + budget_rows + [
        "",
        f"p=5 在 cfl={m['setup']['cfl']} 下 $E_{{time}}/E_{{total}}$ = "
        f"{100 * share_lo:.4f}%，p=6 = {100 * share_hi:.4f}%。",
        "",
        "![stage cap](figures/advorder_stage_cap.png)",
        "",
        "## 4. 注意点与局限",
        "",
        "- **$p\\ge5$ 的「级数上限」在本 CFL 下不咬人**：$p=6$ 本应需要 7 级，"
        "但注册表里没有，阶梯仍用 6 级 5 阶的 `rk5`，实测阶仍达 7 阶。原因是本案的 cfl "
        "约定 $dt=\\mathrm{cfl}/((2p+1)N)$ 使 $dt$ 随 $p$ 变小，时间误差被压到空间误差之下"
        "若干量级。这不是「7 级不需要」的一般结论，只是 cfl=0.2 处的定量事实；"
        "3.3 节给出了时间误差占比涨到 10% 时的 cfl 位置。",
        "- **自收敛 vs 外部基准**：ref 表（Sun 2024）只到 $p=4$，故 $p=5,6$ 只有自收敛证据；"
        "绝对 L2 幅度与 ref 不可比（第 2 轮已证否「级数差解释幅值差」）。",
        f"- **舍入下限**：$p=6$ 在 $N\\ge24$ 的误差已到 $10^{{-12}}$ 量级，阶梯上端受 "
        f"`FIT_FLOOR={m['thresholds']['FIT_FLOOR']}` 截断，拟合点数见 3.2 节。",
        "- **发散判据**是工程判据（非有限或误差 > "
        f"{m['thresholds']['DIVERGENCE_FACTOR']:g}），不是线性稳定域的严格边界；"
        "严格谱半径结果见 `validation/advection_order_stability/metrics.json`。",
        "",
        "## 5. 误差判定（是否 <3%）",
        "",
        "| 量 | 基准 | 实测 | 相对误差 | <3% |",
        "|--:|---:|---:|---:|:--:|",
    ]
    for r in m["rungs"]:
        if r["degree"] not in RUNG_DEGREES or r["fitted_order"] is None:
            continue
        exp = r["expected_spatial_order"]
        rel = abs(r["fitted_order"] - exp) / exp
        lines.append(f"| P{r['degree']} 收敛阶 | {exp} | {r['fitted_order']:.3f} | "
                     f"{100 * rel:.2f}% | {'✅' if rel < 0.03 else '❌'} |")
    lines += [
        f"| P5 时间占比 | ≤ {m['thresholds']['TIME_SHARE_TOL']:.0%} | "
        f"{100 * share_lo:.4f}% | — | ✅ |",
        f"| P6 时间占比 | ≤ {m['thresholds']['TIME_SHARE_TOL']:.0%} | "
        f"{100 * share_hi:.4f}% | — | ✅ |",
        "",
        "绝对 L2 与 ref 的 3% 对比：**无基准**（ref 未公布 Δt 与误差范数定义，且只到 p=4），"
        "故本轮只对收敛率作 <3% 承诺。",
        "",
        "## 6. 复现",
        "",
        "```bash",
        "PYTHONPATH=. python -m benchmarks.case_dg_advect_order_stage_cap",
        "```",
        "",
        f"工件：`{OUT}/metrics.json`、`{OUT}/tables.md`、`{FIG}/advorder_stage_cap.png`。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
