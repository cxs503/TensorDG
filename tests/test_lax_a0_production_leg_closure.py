"""Machine checks for the CASE-LAX-DG A0 production-leg closure.

These assert *invariants of the closure*, not that the closure "succeeded":

* the re-adjudicated A0 verdict is regenerated from the committed artifacts
  (case-pool ladder + certified configuration), so the file is not hand-written;
* the old production leg (p3.N400) is recorded as an honest FAIL and the new
  production leg (p2.N800) as PASS, with the honest-FAIL preserved;
* the minimal-DOF passing leg over the enlarged set is p2.N800;
* every number is consistent with the tolerance imported from the registered case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.lax_a0_production_leg_closure import (
    OUT, adjudicate, dof, load_case_pool, load_certified, tolerance,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def metrics():
    if not (OUT / "metrics.json").exists():
        pytest.skip("closure artifact not present")
    return json.loads((OUT / "metrics.json").read_text())


def test_closure_regenerates_from_committed_artifacts():
    """Re-adjudication from the committed artifacts reproduces the verdict."""
    m = adjudicate(load_case_pool(), load_certified())
    assert m["old_production_leg"]["leg"] == "p3.N400"
    assert m["new_production_leg"]["leg"] == "p2.N800"


def test_old_leg_is_honest_fail():
    m = adjudicate(load_case_pool(), load_certified())
    assert m["old_production_leg"]["verdict"] == "FAIL"
    assert m["old_production_leg"]["worst_l2"] > tolerance()


def test_new_leg_passes_bar():
    m = adjudicate(load_case_pool(), load_certified())
    assert m["new_production_leg"]["verdict"] == "PASS"
    assert m["new_production_leg"]["worst_l2"] < tolerance()


def test_minimal_dof_passing_is_p2_n800():
    m = adjudicate(load_case_pool(), load_certified())
    assert m["minimal_dof_passing"]["leg"] == "p2.N800"
    assert m["minimal_dof_passing"]["dof"] == 2400


def test_honest_fail_preserved():
    m = adjudicate(load_case_pool(), load_certified())
    assert m["honest_fail_preserved"]["leg"] == "p3.N400"
    assert "not re-labelled" in m["honest_fail_preserved"]["note"]


def test_committed_artifact_matches_regeneration(metrics):
    """The committed metrics.json is the one the arithmetic gives."""
    m = adjudicate(load_case_pool(), load_certified())
    assert metrics["old_production_leg"]["worst_l2"] == \
        pytest.approx(m["old_production_leg"]["worst_l2"], rel=1e-12)
    assert metrics["new_production_leg"]["worst_l2"] == \
        pytest.approx(m["new_production_leg"]["worst_l2"], rel=1e-12)
    assert metrics["closure"]["a0_verdict"] == "PASS"
