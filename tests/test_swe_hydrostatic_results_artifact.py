"""Artifact guards for the hydrostatic well-balanced validation JSON.

These tests do **not** re-solve the SWE system (that is done by
``tests/test_swe_wellbalanced_hydrostatic.py`` and
``validation/swe_thacker/run_hydrostatic_wb.py``); they check that the
committed result file is the product of the current script and still
satisfies its own acceptance criteria.

Motivation: the pre-``zb_center`` snapshot of
``validation/swe_thacker/hydrostatic_wb_results.json`` was committed by hand
carrying an O(dx^2) post-processing drift in the P2 entries and no record of
which code revision produced it, so its numbers could not be traced back to a
run.  The guards below make that failure mode visible without a 5-minute CPU
solve:

* provenance block (code revision, versions, architecture, command),
* the reconstruction recipe actually used (must be the ``zb_center`` one),
* the internal consistency of ``summary`` vs ``single_run``, and
* the declared pass criteria on every convergence entry.

A regeneration (``PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb.py``)
rewrites the file with a fresh provenance block; the tests only require that
the block exists and is self-consistent.
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
    / "hydrostatic_wb_results.json"
)

ETA_TOL = 1e-8   # m, from the file's own pass_criteria
U_TOL = 1e-8     # m/s


@pytest.fixture(scope="module")
def results() -> dict:
    with open(RESULTS) as f:
        return json.load(f)


def test_results_file_exists():
    assert RESULTS.is_file(), f"missing validation artifact: {RESULTS}"


def test_provenance_block_is_recorded(results):
    prov = results.get("provenance")
    assert isinstance(prov, dict), "results JSON has no provenance block"
    for key in ("generated_at_utc", "git_commit_short", "python", "torch",
                "numpy", "platform", "machine", "command", "generator"):
        assert key in prov, f"provenance missing key {key!r}"
        assert prov[key] not in ("", None), f"provenance key {key!r} is empty"
    # Timestamp must parse (ISO-8601 UTC) so staleness can be assessed.
    datetime.fromisoformat(prov["generated_at_utc"])
    assert prov["generator"].endswith("run_hydrostatic_wb.py")
    assert "run_hydrostatic_wb.py" in prov["command"]


def test_reconstruction_uses_polynomial_bed_centre(results):
    """The P2 fix: eta must be reconstructed with zb_center = zb_modes[:, 0]."""
    single = results["single_run"]
    assert "zb_center" in single.get("reconstruction", ""), (
        "single_run.reconstruction does not record the zb_center recipe; "
        "the file is probably the pre-fix snapshot"
    )
    for entry in results["convergence_study"]["runs"]:
        for run in entry["results"]:
            assert "zb_center" in run.get("reconstruction", ""), (
                f"P{run['degree']} N={run['ncells']} entry lacks the "
                "zb_center reconstruction recipe"
            )


def test_single_run_meets_pass_criteria(results):
    single = results["single_run"]
    assert single["well_balanced"] is True
    assert single["reached_final_time"] is True
    assert single["steps"] > 0
    assert single["eta_error_max"] < ETA_TOL
    assert single["u_max"] < U_TOL
    # A lake at rest must stay at rest: the velocity scale is round-off level,
    # not merely "small" (< 1e-8).  Loosen to 1e-9 for slower-architecure
    # round-off (see docs: loong64 ~5x the x86 residual).
    assert single["u_max"] < 1e-9


def test_summary_matches_single_run(results):
    summary = results["summary"]
    assert set(summary["pass_criteria"]) == {"eta_error_max", "u_max"}
    assert summary["actual_results"]["eta_error_max"] == pytest.approx(
        results["single_run"]["eta_error_max"], rel=0, abs=0)
    assert summary["actual_results"]["u_max"] == pytest.approx(
        results["single_run"]["u_max"], rel=0, abs=0)
    assert summary["pass"] is True


def test_convergence_study_covers_degrees_and_meshes(results):
    study = results["convergence_study"]
    assert study["degrees"] == [1, 2]
    assert study["ncells_list"] == [50, 100, 200]
    assert len(study["runs"]) == 2
    seen = []
    for block in study["runs"]:
        assert block["degree"] in (1, 2)
        for run in block["results"]:
            seen.append((run["degree"], run["ncells"]))
            assert run["degree"] == block["degree"]
            assert run["well_balanced"] is True
            assert run["reached_final_time"] is True
            assert run["eta_error_max"] < ETA_TOL, run
            assert run["u_max"] < U_TOL, run
    assert sorted(seen) == [(1, 50), (1, 100), (1, 200),
                            (2, 50), (2, 100), (2, 200)]