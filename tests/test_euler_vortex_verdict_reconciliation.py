"""Gate on the single-source-of-truth verdict for CASE-DG-EULER-VORTEX-ERR.

These tests pin the two defects that ``tools/verdict_reconcile.py`` exists to
prevent.  They assert *invariants over the committed evidence*, not just that a
function runs:

D1  two artefacts at one commit published opposite verdicts for one slug, with
    identical provenance strings, so nothing distinguished them.  The anchor is
    now declared, and the stale artefact's number is required to reproduce the
    free-space reference certificate -- if that ever stops holding, the
    classification (and this test) must fail rather than silently pass.

D2  ``all(gate.passed for gate in gates)`` publishes a *coverage* fact
    (``nan``/``None`` = not evaluated) as a *physics* failure.  The tri-state
    reduction must never turn a not-run gate into FAIL.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import verdict_reconcile as vr  # noqa: E402

SLUG = vr.DEFAULT_SLUG


@pytest.fixture(scope="module")
def report():
    return vr.adjudicate(SLUG)


# --------------------------------------------------------------------------
# D1 -- one slug, one verdict
# --------------------------------------------------------------------------
def test_anchor_is_declared_exactly_once(report):
    anchors = [a for a in report["artefacts"] if a["scope"] == "anchor"]
    assert len(anchors) == 1, [a["path"] for a in anchors]
    assert report["authoritative_path"] == "case_euler_vortex_yee/metrics.json"


def test_anchor_passes_its_tolerance(report):
    assert report["authoritative_verdict"] == vr.PASS
    assert report["authoritative_percent"] < report["tolerance"] * 100.0
    assert report["adjudicated_pass"] is True


def test_no_blocking_reconciliation_problem(report):
    assert report["blocking"] == []


def test_every_verdict_bearing_artefact_declares_its_scope(report):
    undeclared = [a["path"] for a in report["artefacts"] if a["scope"] is None]
    assert undeclared == [], (
        "a verdict-bearing artefact appeared without a scope declaration in "
        f"{Path(vr.__file__).name} SCOPE_REGISTRY: {undeclared}")


def test_check_cli_exit_code_is_clean():
    assert vr.main(["--check", "--quiet"]) == 0


# --------------------------------------------------------------------------
# D1 root cause -- the stale number *is* the reference model mismatch
# --------------------------------------------------------------------------
def test_superseded_artefact_is_certified_by_the_reference_mismatch(report):
    stale = next(a for a in report["artefacts"]
                 if a["path"] == "case_euler_vortex/metrics_yee.json")
    assert stale["classification"] == "SUPERSEDED"
    assert stale["verdict"] == vr.FAIL
    certificate = stale["certificate"]
    assert certificate["matched"] is True
    assert certificate["certificate_time"] == pytest.approx(10.0)
    # 1 % band is far tighter than any genuine scheme error and far wider than
    # the 2e-5 spread between the two ways of evaluating the same discrepancy.
    assert certificate["relative_difference"] < 0.01


def test_superseded_number_is_orders_away_from_the_measured_error(report):
    stale = next(a for a in report["artefacts"]
                 if a["path"] == "case_euler_vortex/metrics_yee.json")
    ratio = stale["gate_percent"] / report["authoritative_percent"]
    assert ratio > 100.0, (
        "the stale artefact is no longer dominated by the reference defect; "
        f"ratio to the measured error is only {ratio:.3g}")


def test_certificate_is_reference_model_discriminating():
    """The certificate must be large where the models differ and ~0 where not."""
    cert = vr.certificate(times=(1.0, 10.0))
    at_one = cert["records"]["1"]["max_percent"]
    at_ten = cert["records"]["10"]["max_percent"]
    assert at_ten > 15.0, at_ten            # one full period: core is outside the box
    assert at_one < 0.05, at_one            # still inside the box: models agree
    assert at_ten / max(at_one, 1e-12) > 1000.0


def test_superseded_artefact_has_no_live_producer(report):
    stale = next(a for a in report["artefacts"]
                 if a["path"] == "case_euler_vortex/metrics_yee.json")
    assert stale["producers"] == [], stale["producers"]


def test_superseded_marker_points_at_the_authority(report):
    marker_path = ROOT / "case_euler_vortex" / "SUPERSEDED.json"
    assert marker_path.exists(), "run tools/verdict_reconcile.py --mark"
    marker = json.loads(marker_path.read_text())
    assert marker["current_artefact"] == report["authoritative_path"]
    assert marker["superseded_artefact"] == "metrics_yee.json"
    assert marker["certificate"]["matched"] is True
    assert "NOT the current verdict" in marker["note"]


# --------------------------------------------------------------------------
# D2 -- tri-state never coerces "not run" into "failed"
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gate,expected", [
    ({"passed": True, "measured": 0.01}, vr.PASS),
    ({"passed": True, "measured": None}, vr.PASS),
    ({"passed": False, "measured": 0.14}, vr.FAIL),
    ({"passed": False, "measured": None}, vr.INCOMPLETE),
    ({"passed": False, "measured": float("nan")}, vr.INCOMPLETE),
    ({"passed": None, "measured": None}, vr.INCOMPLETE),
    # counterexample found by reviewing this very rule: a count-valued criterion
    # really failed, with no numeric reading to prove it
    ({"passed": False, "measured": None, "status": "failed"}, vr.FAIL),
    ({"passed": False, "measured": None, "status": "not_evaluated"}, vr.INCOMPLETE),
    ({"passed": True, "measured": None, "status": "passed"}, vr.PASS),
])
def test_gate_state(gate, expected):
    assert vr.gate_state(gate) == expected


def test_tri_state_precedence():
    assert vr.tri_state([vr.PASS, vr.INCOMPLETE]) == vr.INCOMPLETE
    assert vr.tri_state([vr.PASS, vr.FAIL, vr.INCOMPLETE]) == vr.FAIL
    assert vr.tri_state([]) == vr.INCOMPLETE


def test_incomplete_study_is_not_published_as_fail(report):
    """The mechanism study's raw boolean is False; the honest verdict is INCOMPLETE."""
    payload = json.loads((ROOT / "case_euler_vortex_errmechanism" / "metrics.json").read_text())
    raw_boolean = all(g.get("passed") for g in payload["gates"].values() if "passed" in g)
    assert raw_boolean is False, "the defect under test changed shape; re-derive it"
    record = next(a for a in report["artefacts"]
                  if a["path"] == "case_euler_vortex_errmechanism/metrics.json")
    assert record["verdict"] == vr.INCOMPLETE
    assert record["not_run_gates"], record
    assert record["failed_gates"] == [], record["failed_gates"]


def test_count_valued_gate_failure_is_not_demoted_to_not_run(report):
    """C6 compares records present vs declared: it fails without a numeric reading."""
    record = next(a for a in report["artefacts"]
                  if a["path"] == "case_euler_vortex_errconserve_lattice/metrics.json")
    assert "C6_ladder_completeness" in record["failed_gates"], record["gate_states"]
    assert record["verdict"] == vr.FAIL
    # its not-run gates are still reported separately, so neither fact hides the other
    assert record["not_run_gates"]


def test_study_verdicts_are_all_tri_state(report):
    for record in report["artefacts"]:
        assert record["verdict"] in {vr.PASS, vr.FAIL, vr.INCOMPLETE}, record["path"]


# --------------------------------------------------------------------------
# drift guard -- an undeclared artefact is blocking, not ignored
# --------------------------------------------------------------------------
def test_undeclared_scope_is_blocking(monkeypatch):
    monkeypatch.delitem(vr.SCOPE_REGISTRY, "case_euler_vortex_yee/metrics.json")
    broken = vr.adjudicate(SLUG)
    kinds = {p["kind"] for p in broken["blocking"]}
    assert "undeclared_scope" in kinds
    assert "missing_authority" in kinds
    assert broken["adjudicated_pass"] is False
    assert vr.main(["--check", "--quiet"]) == 1


def test_reconciliation_report_is_committed_and_current(report):
    path = ROOT / "artifacts" / f"verdict_reconciliation_{SLUG}.json"
    assert path.exists(), ("run tools/verdict_reconcile.py --report "
                           "artifacts/verdict_reconciliation_<slug>.json")
    committed = json.loads(path.read_text())
    assert committed["authoritative_path"] == report["authoritative_path"]
    assert committed["adjudicated_pass"] == report["adjudicated_pass"]
    assert committed["blocking"] == []