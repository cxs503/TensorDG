"""Regression gates for the near-vacuum double-rarefaction verification (Toro test 2).

These tests lock in the evidence recorded in
``validation/riemann_vacuum/vacuum_sweep.json`` and
``docs/riemann_vacuum_verification.md``:

* the shipped local positivity fallback keeps the near-vacuum problem physical;
* the fallback is load bearing - disabling it aborts the same run;
* the fallback stays local (a small fraction of cells at most);
* the scheme is accurate against the *exact* Riemann solution (< 3 %).

They use P1 and the coarsest meshes so the whole module stays cheap
(~30 s single-threaded) and can run in the default suite.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from benchmarks.riemann_vacuum_1d import CASES, run_configuration

ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "validation", "riemann_vacuum", "vacuum_sweep.json",
)

NEAR_VACUUM = "toro2_near_vacuum"
CONTROL = "toro1_modified_sod"


@pytest.fixture(scope="module")
def coarse_with_fallback():
    torch.set_num_threads(1)
    return run_configuration(NEAR_VACUUM, 100, 1, fallback=True)


@pytest.fixture(scope="module")
def coarse_without_fallback():
    torch.set_num_threads(1)
    return run_configuration(NEAR_VACUUM, 100, 1, fallback=False)


@pytest.fixture(scope="module")
def refined_with_fallback():
    torch.set_num_threads(1)
    return run_configuration(NEAR_VACUUM, 200, 1, fallback=True)


def test_case_definition_is_the_standard_near_vacuum_problem():
    case = CASES[NEAR_VACUUM]
    assert case["left"] == (1.0, -2.0, 0.4)
    assert case["right"] == (1.0, 2.0, 0.4)
    assert case["final_time"] == 0.15


def test_fallback_keeps_near_vacuum_state_physical(coarse_with_fallback):
    record = coarse_with_fallback
    assert record["status"] == "ok"
    assert record["finite"] is True
    assert record["min_density"] > 0.0
    assert record["min_pressure"] > 0.0
    # The exact solution's pressure minimum is ~200x below the initial 0.4.
    assert record["exact_min_pressure"] < 0.01


def test_fallback_is_load_bearing(coarse_without_fallback, coarse_with_fallback):
    record = coarse_without_fallback
    assert record["status"] == "aborted"
    assert "Nonphysical" in record["abort_reason"]
    # It fails in the opening part of the run, i.e. the fallback is doing real
    # work from the start rather than at a late boundary event.
    total = coarse_with_fallback["steps"]
    assert record["steps_reached"] < max(50, 0.05 * total)
    assert record["steps_reached"] < total


def test_fallback_stays_local(coarse_with_fallback):
    record = coarse_with_fallback
    assert record["max_reverted_fraction_per_call"] < 0.10
    # It must actually act, otherwise this test proves nothing.
    assert record["cells_reverted_total"] > 0


def test_near_vacuum_accuracy_against_exact_solution(refined_with_fallback):
    record = refined_with_fallback
    assert record["status"] == "ok"
    for field in ("density", "velocity", "pressure"):
        error = record[f"rel_l2_{field}"]
        assert error < 0.03, f"{field} relative L2 {error:.4%} exceeds the 3% gate"


def test_refinement_does_not_degrade_accuracy(coarse_with_fallback, refined_with_fallback):
    assert refined_with_fallback["max_field_error"] < coarse_with_fallback["max_field_error"]


def test_control_case_still_matches_published_sod_baseline():
    """Toro test 1 (modified Sod) is an already-verified case; keep it honest."""
    torch.set_num_threads(1)
    record = run_configuration(CONTROL, 200, 1, fallback=True)
    assert record["status"] == "ok"
    assert record["max_field_error"] < 0.03


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="sweep artifact not generated")
def test_recorded_sweep_verdicts_are_consistent():
    with open(ARTIFACT, encoding="utf-8") as handle:
        payload = json.load(handle)
    verdicts = payload["verdicts"]
    assert verdicts["C1_admissibility_with_fallback"] is True
    assert verdicts["C2_near_vacuum_finest_mesh_error_below_3pct"] is True
    assert verdicts["C3_near_vacuum_fallback_disabled_aborts"] is True
    assert verdicts["C4_fallback_locality_below_10pct_cells"] is True
    assert verdicts["all_criteria_pass"] is True
    # The modified-Sod control intentionally stays out of the gate: unlimited
    # P3 over its shock overshoots (honest FAIL recorded in the report).
    assert "C2b_control_finest_mesh_error_below_3pct" in verdicts