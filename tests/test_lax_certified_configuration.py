"""Contract tests for the CASE-LAX-DG certified-configuration artifact.

Three layers:

* the certification **rule** is a pure function, so it is exercised on synthetic
  rows here (no solver run): PASS/FAIL per leg, minimal-DOF choice, the
  grid-scope minimality flag, and the tie-break;
* the **committed artifact** must be re-adjudicable offline -- every verdict and
  the certified configuration are rebuilt from the stored per-leg errors, the
  tolerance is the registered case's, and the reuse rows must have passed;
* the **doc** must be a pure function of the artifact (numbers injected, not
  transcribed), and the pre-registration guard must refuse a grid that raises
  the DOF budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.lax_case_pool import ledger_acceptance_tolerance
from benchmarks.lax_certified_configuration import (
    CANDIDATES, CONSISTENCY_TOL, DOF_CAP, FIELDS, METRICS, RULE, certification,
    dof, render_doc, render_summary, reuse_check)

ROOT = Path(__file__).resolve().parents[1]


def _metrics():
    # ``skip`` (not ``pass``): a missing artifact must read as "not run here".
    if not METRICS.exists():
        pytest.skip("certified-configuration artifact not committed yet "
                    f"({METRICS.relative_to(ROOT)}); reproduce with "
                    "`python -m benchmarks.lax_certified_configuration`")
    return json.loads(METRICS.read_text())


def _row(leg, cells, degree, density, velocity, pressure, wall=1.0):
    return {"leg": leg, "cells": cells, "degree": degree,
            "l2_errors": {"density": density, "velocity": velocity,
                          "pressure": pressure},
            "wall_seconds": wall}


# --------------------------------------------------------------------------- #
# rule
# --------------------------------------------------------------------------- #
def test_rule_is_an_and_over_all_fields_not_a_max_of_the_means():
    rows = [
        _row("p1.N400", 400, 1, 0.050, 0.020, 0.010),
        _row("p2.N400", 400, 2, 0.029, 0.031, 0.010),  # velocity breaks the bar
        _row("p4.N400", 400, 4, 0.020, 0.010, 0.010),
    ]
    res = certification(rows, 0.03)
    status = {v["leg"]: v["status"] for v in res["per_leg"]}
    assert status == {"p1.N400": "FAIL", "p2.N400": "FAIL", "p4.N400": "PASS"}
    assert res["certified"]["leg"] == "p4.N400"
    assert res["certified"]["worst_field"] == "density"
    assert res["certified"]["margin_of_tolerance"] == pytest.approx(0.02 / 0.03)


def test_rule_certifies_the_smallest_dof_and_reports_grid_minimality():
    rows = [
        _row("p1.N200", 200, 1, 0.060, 0.040, 0.030),       # 400 dof, FAIL
        _row("p3.N200", 200, 3, 0.025, 0.020, 0.010),       # 800 dof, PASS
        _row("p3.N400", 400, 3, 0.031, 0.022, 0.013),       # 1600 dof, FAIL
        _row("p3.N800", 800, 3, 0.023, 0.014, 0.009),       # 3200 dof, PASS
    ]
    res = certification(rows, 0.03)
    assert res["certified"]["leg"] == "p3.N200"
    # p1.N200 is strictly cheaper and fails -> minimality holds on this grid
    assert res["minimality"]["cheaper_legs"] == ["p1.N200"]
    assert res["minimality"]["cheaper_legs_all_fail"] is True
    assert res["status"] == "PASS"
    assert res["cheapest_by_wall_clock"] in {"p3.N200", "p3.N800"}


def test_rule_is_honest_when_nothing_passes_and_ties_break_on_wall_clock():
    rows = [_row("p1.N400", 400, 1, 0.05, 0.02, 0.01),
            _row("p2.N400", 400, 2, 0.04, 0.02, 0.01)]
    res = certification(rows, 0.03)
    assert res["status"] == "FAIL" and res["certified"] is None
    assert res["cheapest_by_wall_clock"] is None
    assert res["minimality"]["cheaper_legs_all_fail"] is False  # vacuous, not green

    # p2.N800 and p3.N600 both carry 2400 DOF: the cheaper wall clock wins
    tie = [_row("p2.N800", 800, 2, 0.020, 0.010, 0.010, wall=9.0),
           _row("p3.N600", 600, 3, 0.020, 0.010, 0.010, wall=3.0)]
    res = certification(tie, 0.03)
    assert res["certified"]["leg"] == "p3.N600"  # equal DOF, cheaper wall clock


def test_grid_respects_the_declared_dof_cap():
    assert isinstance(RULE, str) and "(p+1)*N" in RULE
    assert dof(800, 3) == DOF_CAP
    for cells, degree in CANDIDATES:
        assert dof(cells, degree) <= DOF_CAP, (cells, degree)


# --------------------------------------------------------------------------- #
# committed artifact
# --------------------------------------------------------------------------- #
def test_artifact_recomputes_its_own_verdicts_and_certificate():
    metrics = _metrics()
    tolerance = ledger_acceptance_tolerance()
    assert metrics["certification"]["tolerance"] == pytest.approx(tolerance)
    assert metrics["config"]["tolerance"] == pytest.approx(tolerance)
    rebuilt = certification(metrics["legs"], tolerance)
    assert rebuilt == metrics["certification"]
    certified = rebuilt["certified"]
    assert certified is not None, "artifact records no certified configuration"
    leg = next(r for r in metrics["legs"] if r["leg"] == certified["leg"])
    for field in FIELDS:  # every field, no exemption
        assert leg["l2_errors"][field] < tolerance
    assert certified["worst_l2"] == pytest.approx(
        max(next(r for r in metrics["legs"]
                if r["leg"] == certified["leg"])["l2_errors"].values()))
    assert metrics["verdict"] == "PASS"


def test_artifact_reuse_rows_reproduce_the_committed_artifacts():
    metrics = _metrics()
    check = metrics["reuse_check"]
    assert check["compared"] is True
    assert check["tolerance"] == pytest.approx(CONSISTENCY_TOL)
    worst = max(row["relative_difference"] for row in check["rows"])
    assert worst == pytest.approx(check["worst_relative_difference"])
    assert check["status"] == "PASS", (
        "a reference leg no longer reproduces its committed artifact: "
        f"worst relative difference {worst:.3e}")
    sources = {row["source"] for row in check["rows"]}
    assert sources == {"lax_case_pool", "lax_resolution_requirement"}


def test_artifact_dofs_steps_and_wall_clock_are_measured_values():
    metrics = _metrics()
    by_leg = {v["leg"]: v for v in metrics["certification"]["per_leg"]}
    for leg in metrics["legs"]:
        assert leg["steps"] > 0
        assert leg["wall_seconds"] > 0.0
        # the DOF accounting is the declared (p+1)*N, not a stored label
        assert by_leg[leg["leg"]]["dof"] == dof(leg["cells"], leg["degree"])


# --------------------------------------------------------------------------- #
# doc purity
# --------------------------------------------------------------------------- #
def test_rendered_doc_and_summary_are_pure_functions_of_the_artifact():
    metrics = _metrics()
    doc = render_doc(metrics)
    assert isinstance(doc, str) and doc.strip()
    assert render_doc(metrics) == doc
    assert render_summary(metrics) == render_summary(metrics)
    cert = metrics["certification"]["certified"]
    if cert is not None:
        assert cert["leg"] in doc
        assert f"{cert['worst_l2']:.5e}" in doc
    assert "3 %" in doc or "3%" in doc  # the bar is stated in the doc