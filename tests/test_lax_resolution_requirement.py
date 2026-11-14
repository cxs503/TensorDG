"""Contract tests for the CASE-LAX-DG resolution-requirement artifact.

The artifact must be re-adjudicable offline: every verdict is recomputed here
from the stored per-leg errors, the pinned A0 leg must still be the case pool's,
and the rendered doc/summary must be a pure function of the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.lax_case_pool import ledger_acceptance_tolerance
from benchmarks.lax_resolution_requirement import (
    CONSISTENCY_TOL, DOC, FIELDS, METRICS, PINNED_LEG, POOL,
    rate, render_doc, requirements)

ROOT = Path(__file__).resolve().parents[1]


def _metrics():
    # ``skip`` (not ``pass``): the driver has been executed end to end on the
    # cluster (cj2647, rc = 0) but the fetch channel returns the *staged input*
    # snapshot, not the job's writes, so the artifact is not in this worktree
    # yet.  A missing artifact must read as "not run here", never as green.
    if not METRICS.exists():
        pytest.skip("resolution-requirement artifact not committed yet "
                    f"({METRICS.relative_to(ROOT)}); reproduce with "
                    "`python -m benchmarks.lax_resolution_requirement` -- "
                    "see docs/CASE-LAX-DG-residual-gap.md")
    return json.loads(METRICS.read_text())


def test_requirement_is_recomputed_from_the_stored_errors():
    metrics = _metrics()
    rebuilt = requirements(metrics["legs"], metrics["requirement"]["tolerance"])
    assert rebuilt == metrics["requirement"]
    assert metrics["requirement"]["tolerance"] == pytest.approx(
        ledger_acceptance_tolerance())
    assert metrics["config"]["tolerance"] == pytest.approx(
        ledger_acceptance_tolerance())
    # the verdict is the rule, not a label: worst field vs tolerance per leg
    for row, leg in zip(rebuilt["per_leg"], metrics["legs"]):
        assert row["worst_l2"] == pytest.approx(max(leg["l2_errors"].values()))
        assert row["status"] == ("PASS" if row["worst_l2"]
                                 < rebuilt["tolerance"] else "FAIL")
    assert metrics["verdict"] == rebuilt["status"]


def test_orders_are_the_measured_pair_orders():
    metrics = _metrics()
    legs = metrics["legs"]
    assert [leg["cells"] for leg in legs] == list(metrics["orders"]["pair"])
    for field in FIELDS:
        assert metrics["orders"]["l1"][field] == pytest.approx(
            rate(legs, field, "l1"))
        assert metrics["orders"]["l2"][field] == pytest.approx(
            rate(legs, field, "l2"))


def test_pinned_acceptance_leg_is_not_moved():
    """The additive study must not relabel the ledger acceptance verdict."""
    metrics = _metrics()
    pool = json.loads(POOL.read_text())
    acceptance = pool["acceptance"]
    pinned = next(e for e in pool["ladder"]
                  if (e["cells"], e["degree"]) == PINNED_LEG)
    assert acceptance["production_leg"] == f"p{PINNED_LEG[1]}.N{PINNED_LEG[0]}"
    assert acceptance["arms"]["production_l2"]["leg"] == (
        f"p{PINNED_LEG[1]}.N{PINNED_LEG[0]}")
    assert metrics["pinned"]["leg"] == f"p{PINNED_LEG[1]}.N{PINNED_LEG[0]}"
    assert metrics["pinned"]["worst_l2"] == pytest.approx(
        max(pinned["l2_errors"].values()))
    # a study that moves the leg would have to disagree with the pool here
    assert acceptance["arms"]["production_l2"]["worst"] == pytest.approx(
        metrics["pinned"]["worst_l2"])


def test_reuse_check_reproduces_the_committed_artifacts():
    metrics = _metrics()
    reuse = metrics["reuse_check"]
    assert reuse["compared"], "the reuse check must actually compare something"
    assert reuse["tolerance"] == pytest.approx(CONSISTENCY_TOL)
    assert reuse["worst_relative_difference"] <= CONSISTENCY_TOL
    assert reuse["status"] == "PASS"
    sources = {row["source"] for row in reuse["rows"]}
    assert "lax_case_pool" in sources


def test_doc_and_summary_are_rendered_from_the_numbers():
    metrics = _metrics()
    assert DOC.exists(), f"missing rendered doc {DOC.relative_to(ROOT)}"
    assert render_doc(metrics) == DOC.read_text()
    summary = (METRICS.parent / "summary.md").read_text()
    assert f"R2 verdict **{metrics['requirement']['status']}**" in summary
    for leg in metrics["legs"]:
        assert f"{leg['l2_errors']['density']:.5e}" in summary