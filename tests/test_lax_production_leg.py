"""Gates on the *registered* production leg of CASE-LAX-DG (ledger A0).

These tests exist because the pool used to read the ledger bar on "the finest
ladder leg" (``p3.N400`` -> 3.116 %, FAIL) while the certified-configuration
closure published ``p2.N800`` (2.570 %, PASS).  Two artifacts, one ledger, no
single source of truth: a consumer that read ``metrics.json`` got the opposite
verdict from a consumer that read the closure document.  The gates below make
that class of drift fail loudly instead of silently:

* the registered leg lives in ``tensordg.cases.lax_1d.lax_parameters()`` and is
  the only place the pool may learn it;
* the artifact's ledger arm must be **recomputable** from the committed
  production-leg record, and the record must be re-measurable from its committed
  profiles (no hand-edited number reaches the verdict);
* the ladder's own ``p3.N400`` row keeps its honest sub-bar verdict.

The tamper tests are mutation controls: they assert that the recomputation gate
actually *has* discriminating power (a gate that passes on a forged artifact is
not a gate).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import numpy as np

from benchmarks.lax_case_pool import (  # noqa: E402
    gate_a0, ledger_acceptance_tolerance, production_errors_from_profiles,
    registered_production_leg)
from tensordg.cases.lax_1d import lax_parameters

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool"
METRICS = POOL / "metrics.json"
RECORD = POOL / "production_leg.json"
PROFILES = POOL / "profiles_production.npz"
FIELDS = ("density", "velocity", "pressure")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text())


def test_registry_is_the_single_source_of_the_production_leg():
    params = lax_parameters()
    leg = params["production_leg"]
    assert isinstance(leg, dict)
    assert int(leg["cells"]) == 800 and int(leg["degree"]) == 2, (
        "the registered leg must stay the certified-configuration leg; "
        "changing it is a registration decision, not a refactor")
    assert registered_production_leg() == {"cells": 800, "degree": 2}
    assert "p3.N400" in params["production_leg_basis"]
    # the threshold is imported by the gate, never re-typed there
    assert ledger_acceptance_tolerance() == pytest.approx(
        params["acceptance_threshold"])


def test_artifact_ledger_arm_is_read_on_the_registered_leg(metrics):
    acceptance = metrics["acceptance"]
    registered = registered_production_leg()
    expected = f"p{registered['degree']}.N{registered['cells']}"
    assert acceptance["production_leg"] == expected
    assert acceptance["production_leg_source"] == "registered (case module)"
    assert acceptance["arms"]["production_l2"]["leg"] == expected
    assert metrics["production_leg"]["leg"] == expected
    assert metrics["config"]["production_leg"] == registered


def test_record_is_bound_to_the_registry_and_to_the_artifact(metrics, record):
    registered = registered_production_leg()
    assert (record["cells"], record["degree"]) == (
        registered["cells"], registered["degree"])
    assert record["registration"]["leg"] == registered
    stored = metrics["production_leg"]
    for field in FIELDS:
        assert stored["l2_errors"][field] == pytest.approx(
            record["l2_errors"][field], rel=0, abs=0)
        assert stored["errors"][field] == pytest.approx(
            record["errors"][field], rel=0, abs=0)
    assert stored["leg"] == record["leg"]


def test_record_is_re_measurable_from_its_committed_profiles(record):
    """No number in the ledger arm has to be trusted: re-derive it."""
    assert PROFILES.exists(), "the registered leg must ship its profiles"
    remeasured = production_errors_from_profiles(PROFILES)
    for field in FIELDS:
        assert remeasured["l2"][field] == pytest.approx(
            record["l2_errors"][field], rel=1e-9)
        assert remeasured["l1"][field] == pytest.approx(
            record["errors"][field], rel=1e-9)


def test_verdict_is_recomputable_from_the_artifact_alone(metrics):
    acceptance = metrics["acceptance"]
    entry = {"cells": metrics["production_leg"]["cells"],
             "degree": metrics["production_leg"]["degree"],
             "errors": metrics["production_leg"]["errors"],
             "l2_errors": metrics["production_leg"]["l2_errors"]}
    rebuilt = gate_a0(metrics["ladder"],
                      tolerance=ledger_acceptance_tolerance(),
                      production_leg=entry,
                      production_l2=entry["l2_errors"])
    assert rebuilt["production_leg"] == acceptance["production_leg"]
    assert rebuilt["tolerance"] == pytest.approx(acceptance["tolerance"])
    for arm in ("production_l2", "ladder_l1"):
        assert rebuilt["arms"][arm]["worst"] == pytest.approx(
            acceptance["arms"][arm]["worst"], rel=1e-12)
        assert rebuilt["arms"][arm]["status"] == acceptance["arms"][arm]["status"]
    assert rebuilt["status"] == acceptance["status"]
    assert metrics["verdicts"]["acceptance"] == acceptance["status"]


def test_superseded_ladder_leg_keeps_its_honest_verdict(metrics):
    """Re-pointing the ledger leg must not erase the sub-bar measurement."""
    per_leg = metrics["acceptance"]["per_leg"]
    assert "p3.N400" in per_leg, "the 400-cell P3 point must stay recorded"
    row = per_leg["p3.N400"]
    tolerance = metrics["acceptance"]["tolerance"]
    if max(row["l2"].values()) >= tolerance:
        assert row["status"] == "FAIL"
    assert row["l2"]["density"] == pytest.approx(0.031155618811580767, rel=1e-9)


def test_forged_ledger_arm_is_refused(metrics):
    """Mutation control: the gate must flip when the arm is inflated."""
    entry = {"cells": metrics["production_leg"]["cells"],
             "degree": metrics["production_leg"]["degree"],
             "errors": metrics["production_leg"]["errors"],
             "l2_errors": dict(metrics["production_leg"]["l2_errors"])}
    entry["l2_errors"]["density"] *= 2.0
    rebuilt = gate_a0(metrics["ladder"],
                      tolerance=ledger_acceptance_tolerance(),
                      production_leg=entry,
                      production_l2=entry["l2_errors"])
    assert rebuilt["arms"]["production_l2"]["status"] == "FAIL"
    assert rebuilt["status"] == "FAIL"


def test_unregistered_leg_is_refused(record):
    """Mutation control: a record for a leg nobody registered must not attach."""
    forged = dict(record)
    forged["cells"] = 400
    forged["degree"] = 3
    registered = registered_production_leg()
    assert (forged["cells"], forged["degree"]) != (
        registered["cells"], registered["degree"])