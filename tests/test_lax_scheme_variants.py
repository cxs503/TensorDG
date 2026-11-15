"""CASE-LAX-DG -- machine checks on the pinned-leg configuration sweep.

The anchor ``CASE-LAX-DG`` is pinned at ``p3.N400`` and misses the registered
3 % bar by a hair (3.1156e-02).  ``benchmarks/lax_scheme_variants.py`` sweeps
every knob of the *existing* scheme that leaves the degrees of freedom alone and
records the answer.  These tests keep that study honest without re-running a
solver:

* the control arm still reproduces the committed ledger arm (no silent drift
  between the case pool and the sweep);
* the arm set is closed and pre-registered -- no arm can be dropped because it
  lost, and no winning arm can be added after the fact;
* each non-control arm changes exactly one knob;
* the verdict is a pure function of the recorded rows (rule, not narrative);
* the attainability budget is the projection-floor identity, computed from the
  committed floor artifact rather than hard-coded;
* the admonition doc agrees with ``metrics.json`` (drift gate).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import lax_scheme_variants as study
from benchmarks.lax_case_pool import ledger_acceptance_tolerance

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "metrics.json"
LEGS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "legs"
DOC = ROOT / "docs" / "CASE-LAX-DG-scheme-variants.md"
FLOOR = ROOT / "docs" / "benchmarks" / "results" / "lax_reference_floor" / "metrics.json"


@pytest.fixture(scope="module")
def metrics():
    assert METRICS.exists(), (
        "the sweep artifact is missing; run "
        "`python -m benchmarks.lax_scheme_variants` for every arm, then --aggregate")
    return json.loads(METRICS.read_text())


def _graded(metrics):
    """Rows that produced a finite measurement (unknown/errored arms excluded)."""
    return [row for row in metrics["rows"] if not study._failed(row)]


# ----------------------------------------------------------------- agreement
def test_control_arm_reproduces_committed_ledger_arm(metrics):
    """The sweep's control arm must be the committed ledger arm, not a cousin."""
    control = metrics["control_check"]
    assert control["reproduced"] is True
    assert control["worst_relative_difference"] <= metrics["config"]["control_tolerance"]
    # the committed arm must be the one the case pool registered as production
    pool = json.loads((ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool"
                       / "metrics.json").read_text())
    assert pool["acceptance"]["production_leg"] == "p3.N400"
    assert math.isclose(control["committed_l2"]["density"],
                        pool["acceptance"]["per_leg"]["p3.N400"]["l2"]["density"],
                        rel_tol=0.0, abs_tol=0.0)


def test_the_measured_control_value_is_the_recorded_one(metrics):
    base = json.loads((LEGS / "base.json").read_text())
    control = metrics["control_check"]
    assert base["l2"]["density"] == control["measured_l2"]["density"]
    assert base["steps"] > 0 and base["dof"] == (study.PINNED_DEGREE + 1) * study.PINNED_CELLS


def test_tolerance_is_read_from_the_ledger(metrics):
    assert metrics["config"]["tolerance"] == float(ledger_acceptance_tolerance())


# ------------------------------------------------------------- pre-registration
def test_arm_set_is_closed_and_pre_registered(metrics):
    assert metrics["config"]["closed_arm_set"] is True
    assert tuple(metrics["config"]["variant_order"]) == study.VARIANT_ORDER
    recorded = [row["variant"] for row in metrics["rows"]]
    assert recorded == list(study.VARIANT_ORDER), "arms were added or dropped"
    for row in metrics["rows"]:
        if study._failed(row):
            # an arm that did not run has no leg record; it is recorded as unknown
            assert (LEGS / f"{row['variant']}.json").exists() or row.get("not_run")


def test_each_non_control_arm_changes_exactly_one_knob(metrics):
    for row in metrics["rows"]:
        if row["variant"] == "base":
            assert row["knob"] == {}
            continue
        assert len(row["knob"]) == 1, f"{row['variant']} changes {row['knob']}"


def test_arms_share_the_discretisation_budget(metrics):
    for row in metrics["rows"]:
        assert row["cells"] == study.PINNED_CELLS
        assert row["degree"] == study.PINNED_DEGREE
        assert row["dof"] == metrics["config"]["dof"]


# --------------------------------------------------------------------- verdict
def test_worst_field_is_the_max_over_fields(metrics):
    for row in _graded(metrics):
        worst = max(study.FIELDS, key=lambda f: row["l2"][f])
        assert row["worst_field"] == worst
        assert row["worst_l2"] == row["l2"][worst]


def test_verdict_follows_the_registered_rule(metrics):
    graded = _graded(metrics)
    admissible = [row["variant"] for row in graded
                  if row["decision"]["admissible"]]
    for row in graded:
        passes = all(row["l2"][f] < metrics["config"]["tolerance"] for f in study.FIELDS)
        assert row["decision"]["passes_tolerance"] is passes
        assert row["decision"]["admissible"] is (passes and row["decision"][
            "same_discretisation_budget"])
    assert metrics["verdict"]["admissible_arms"] == admissible
    expected = "PASS" if admissible and metrics["control_check"]["reproduced"] else "FAIL"
    assert metrics["verdict"]["verdict"] == expected


def test_unknown_arms_are_recorded_and_never_admissible(metrics):
    """A crashed or unrun arm is unknown -- it can never be read as a pass."""
    for row in metrics["rows"]:
        if study._failed(row):
            assert row["decision"]["admissible"] is False
            assert "l2" not in row
            assert row["decision"]["margin"] is None
    if not metrics["verdict"]["scope_complete"]:
        assert metrics["verdict"]["arms_not_run"] or metrics["verdict"]["arms_errored"]


# ---------------------------------------------------------------- attainability
def test_attainability_is_the_projection_floor_identity(metrics):
    """``solver_leg_required`` must be sqrt(tol^2 - floor^2) from the floor artifact."""
    bound = metrics["attainability"]
    tolerance = metrics["config"]["tolerance"]
    data = json.loads(FLOOR.read_text())
    leg = next(item for item in data["legs"]
               if item["cells"] == study.PINNED_CELLS
               and item["degree"] == study.PINNED_DEGREE)
    field = leg["fields"]["density"]
    assert bound["reference_floor"] == field["reference_floor"]
    assert bound["solver_leg_measured"] == field["solver_leg"]
    expected = math.sqrt(tolerance ** 2 - field["reference_floor"] ** 2)
    assert math.isclose(bound["solver_leg_required"], expected, rel_tol=1e-12)
    assert math.isclose(bound["floor_share_of_tolerance"],
                        field["reference_floor"] / tolerance, rel_tol=1e-12)
    assert math.isclose(bound["measured_over_required"],
                        field["solver_leg"] / expected, rel_tol=1e-12)


def test_floor_alone_cannot_explain_a_pass_or_a_fail(metrics):
    """The floor must be below the tolerance, else the arm set would be vacuous."""
    bound = metrics["attainability"]
    assert bound["feasible"] is True
    assert bound["reference_floor"] < metrics["config"]["tolerance"]
    # the sweep must not be tuned to the boundary: the required reduction is small
    # but positive, i.e. the bar is neither already met nor hopeless
    assert bound["relative_reduction_needed"] >= 0.0


# ------------------------------------------------------------------- artifacts
def test_doc_agrees_with_metrics(metrics):
    assert DOC.exists(), "run `python -m benchmarks.lax_scheme_variants --report`"
    study.check()


def test_rule_is_pure_and_rejects_an_off_budget_arm():
    """The admissibility rule must be a pure function, testable without a solver."""
    tolerance = float(ledger_acceptance_tolerance())
    good = {"cells": study.PINNED_CELLS, "degree": study.PINNED_DEGREE,
            "l2": {"density": 0.5 * tolerance, "velocity": 0.5 * tolerance,
                   "pressure": 0.5 * tolerance}}
    good["worst_l2"] = max(good["l2"].values())
    assert study.admissibility(good, tolerance)["admissible"] is True
    # one field over the line is enough to fail (no field exemption)
    bad = json.loads(json.dumps(good))
    bad["l2"]["density"] = 1.5 * tolerance
    bad["worst_l2"] = 1.5 * tolerance
    assert study.admissibility(bad, tolerance)["admissible"] is False
    # a coarser/finer mesh changes the budget, so it is not an admissible arm
    off = json.loads(json.dumps(good))
    off["cells"] = study.PINNED_CELLS // 2
    assert study.admissibility(off, tolerance)["admissible"] is False
    # an arm that did not run is not admissible by omission
    crashed = {"cells": study.PINNED_CELLS, "degree": study.PINNED_DEGREE,
               "error": "FloatingPointError: Nonphysical high-order Euler state"}
    decision = study.admissibility(crashed, tolerance)
    assert decision["admissible"] is False and decision["margin"] is None
    # an arm that never ran is unknown, not a pass by omission
    unrun = {"cells": study.PINNED_CELLS, "degree": study.PINNED_DEGREE, "not_run": True}
    decision = study.admissibility(unrun, tolerance)
    assert decision["admissible"] is False
    assert decision["unknown"] is True