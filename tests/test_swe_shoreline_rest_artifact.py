"""Artifact guards for the dry-bank shoreline rest-state validation JSON.

Companion to ``tests/test_swe_hydrostatic_results_artifact.py`` (which guards
the *wet-everywhere* Thacker basin artifact).  This file does not re-solve the
SWE system — that is done by
``validation/swe_thacker/run_shoreline_rest.py`` — it only checks that the
committed result file

* carries a provenance block (revision, versions, host, command),
* covers both shorelines geometries and every declared mesh/degree,
* still meets its own pass criteria (``max|eta - eta0| < 1e-8 m`` and
  ``max|u| < 1e-8 m/s``, the exact solution being the rest state),
* records the pre-fix filter as *failing* those criteria, i.e. the defect the
  change fixed was real and is measurable in the same artifact (a before/after
  contrast, not a remembered number).

Regenerate with::

    PYTHONPATH=. python validation/swe_thacker/run_shoreline_rest.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

RESULTS = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "swe_thacker"
    / "shoreline_rest_state_results.json"
)

ETA_TOL = 1e-8   # m, from the file's own pass_criteria
U_TOL = 1e-8     # m/s
CASES = {"parabolic_hump", "tent_on_face"}
MESH = {(1, 32), (1, 64), (1, 128), (2, 32), (2, 64), (2, 128)}


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.is_file(), f"missing validation artifact: {RESULTS}"
    with open(RESULTS) as f:
        return json.load(f)


def test_provenance_block_is_recorded(results):
    prov = results.get("provenance")
    assert isinstance(prov, dict), "results JSON has no provenance block"
    for key in ("generated_at_utc", "git_commit_short", "python", "torch",
                "numpy", "platform", "machine", "command", "generator"):
        assert key in prov, f"provenance missing key {key!r}"
        assert prov[key] not in ("", None), f"provenance key {key!r} is empty"
    datetime.fromisoformat(prov["generated_at_utc"])
    assert prov["generator"].endswith("run_shoreline_rest.py")
    assert "run_shoreline_rest.py" in prov["command"]


def test_exact_solution_is_the_rest_state(results):
    assert results["exact_solution"] == {"eta": 1.0, "u": 0.0}
    # the error metric must exclude dry cells (they have no water surface)
    assert "dry" in results["error_definition"].lower()
    assert results["dry_depth"] > 0


def test_cases_cover_both_shoreline_geometries(results):
    cases = {case["name"]: case for case in results["cases"]}
    assert set(cases) == CASES
    for case in cases.values():
        seen = sorted((run["degree"], run["ncells"]) for run in case["runs"])
        assert set(seen) == MESH, case["name"]
        for run in case["runs"]:
            assert run["reached_final_time"] is True
            assert run["diverged"] is False
            # the bank must really be dry, otherwise the case tests nothing
            assert run["dry_cell_count"] > 0, run
            assert run["h_min"] >= -1e-15, run


def test_shoreline_positions_are_recorded(results):
    """The "on a face" case must actually put the shoreline on a cell face."""
    cases = {case["name"]: case for case in results["cases"]}
    assert cases["tent_on_face"]["shoreline_x"] == pytest.approx(1.5)
    assert cases["tent_on_face"]["shoreline_on_face"]["64"] is True
    assert cases["parabolic_hump"]["shoreline_x"] == pytest.approx(2.0 ** 0.5)
    assert cases["parabolic_hump"]["shoreline_on_face"]["64"] is False


def test_every_run_meets_the_pass_criteria(results):
    for case in results["cases"]:
        for run in case["runs"]:
            assert run["eta_error_max"] < ETA_TOL, (case["name"], run)
            assert run["u_max"] < U_TOL, (case["name"], run)


def test_legacy_filter_contrast_records_the_defect(results):
    """The pre-fix filter must still fail badly on the recorded scenarios."""
    legacy = results["legacy_filter_contrast"]
    assert legacy, "no before/after contrast recorded"
    seen = {(block["case"], block["degree"], block["ncells"]) for block in legacy}
    assert seen == {(name, degree, 64) for name in CASES for degree in (1, 2)}
    for block in legacy:
        assert block["filter"] == "legacy_flatten_to_cell_average"
        # the old filter produced at least an O(1e-2 m) surface error (and, on
        # one case, an outright blow-up): never a rest state
        assert block["diverged"] or max(block["eta_error_max"],
                                        block["u_max"]) > 1e-3, block


def test_summary_matches_the_runs(results):
    summary = results["summary"]
    assert set(summary["pass_criteria"]) == {"eta_error_max", "u_max"}
    runs = [r for case in results["cases"] for r in case["runs"]]
    assert summary["actual_results"]["eta_error_max"] == pytest.approx(
        max(r["eta_error_max"] for r in runs), rel=0, abs=0)
    assert summary["actual_results"]["u_max"] == pytest.approx(
        max(r["u_max"] for r in runs), rel=0, abs=0)
    legacy = results["legacy_filter_contrast"]
    assert summary["legacy_filter_results"]["eta_error_max"] == pytest.approx(
        max(r["eta_error_max"] for r in legacy), rel=0, abs=0)
    converged = [r for r in legacy if not r["diverged"]]
    assert converged, "every pre-fix run diverged; the contrast is not usable"
    assert summary["legacy_filter_converged"]["u_max"] == pytest.approx(
        max(r["u_max"] for r in converged), rel=0, abs=0)
    assert summary["pass"] is True
    # the fix must buy at least three orders of magnitude, not a few percent
    assert summary["improvement_factor"]["eta_error_max"] > 1e3
    assert summary["improvement_factor"]["u_max"] > 1e3
    # a diverged pre-fix run must be recorded as such, not hidden in the max
    assert set(summary["legacy_filter_results"]["diverged_runs"]) == {
        f"{r['case']} P{r['degree']} N={r['ncells']}"
        for r in legacy if r["diverged"]}