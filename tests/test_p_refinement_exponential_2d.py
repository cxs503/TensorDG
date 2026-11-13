"""Live verification gates for the 2-D p-refinement case (CASE-DG-PREFINEMENT-EXPONENTIAL-2D).

Design rule taken from audits m16499/m16918: a gate that only re-reads the
published JSON proves nothing about the code.  Every test below *runs the
solver* on a small configuration; the artifact consistency test additionally
re-runs one published level and compares it against the file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.p_refinement_exponential import falsification_control
from benchmarks.p_refinement_exponential_2d import (
    independent_relative_l2,
)
from tensordg.advection_2d import MAX_DEGREE, error_2d, solve_advection_2d
from tensordg.cases import CASES, PREFINEMENT_2D_CASE_KEY
from tensordg.cases.prefinement_exponential_2d import (
    CASE_KEY,
    DEGREES,
    VELOCITY,
    anchor_table,
    exact_solution,
    initial_condition,
    p_ladder,
    prefinement_parameters,
    solve_level,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "docs" / "benchmarks" / "results" / "p-refinement-exponential-2d"
)
RESULTS_JSON = ARTIFACT_DIR / "p_refinement_2d_results.json"

SMALL_N = 6
SMALL_CFL = 0.2


def test_case_is_registered_in_case_pool():
    assert CASES["prefinement_exponential_2d"] == CASE_KEY
    params = prefinement_parameters()
    assert params["case_key"] == CASE_KEY
    assert params["ledger_anchor"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"
    assert "Babuska" in params["reference"]["authors"]
    assert params["degrees"] == list(DEGREES)
    assert PREFINEMENT_2D_CASE_KEY == CASE_KEY


def test_analytic_data_consistency():
    """At t=0 the exact solution must reduce to the initial condition."""
    for x, y in [(0.1, 0.3), (0.5, 0.5), (0.9, 0.05)]:
        assert exact_solution(x, y, 0.0) == pytest.approx(initial_condition(x, y))


def test_degree_cap_lifted_and_documented_bound_enforced():
    """The historical ``degree in (0,1,2,3)`` cap must be gone (audit gap)."""
    for degree in (4, 5, MAX_DEGREE):
        _, _, coeff, md = solve_advection_2d(
            ncells=4, degree=degree, final_time=0.05, velocity=VELOCITY, cfl=0.2,
        )
        assert md["degree"] == degree
        assert coeff.shape == (4, 4, degree + 1, degree + 1)
    for degree in (-1, MAX_DEGREE + 1):
        with pytest.raises(ValueError):
            solve_advection_2d(ncells=4, degree=degree, final_time=0.05)


def test_error_decreases_with_degree_on_fixed_mesh():
    """Fixed mesh, rising p: the error must drop fast (spectral, not algebraic)."""
    errors = []
    for degree in (1, 2, 3):
        x, y, coeff, _ = solve_advection_2d(
            ncells=SMALL_N, degree=degree, final_time=1.0,
            velocity=VELOCITY, cfl=SMALL_CFL,
        )
        errors.append(float(error_2d(x, y, coeff, 1.0, VELOCITY)["relative_l2"]))
    assert errors[0] > errors[1] > errors[2]
    # each p increment must buy at least 5x (algebraic p^-1 cannot: 1.5x)
    assert errors[0] / errors[1] > 5.0
    assert errors[1] / errors[2] > 5.0
    # sigma over p=1..3: > 2.5 rules out every algebraic law tested in the
    # falsification suite (p^-1..p^-5 give 0.55..1.71)
    p = np.array([1.0, 2.0, 3.0])
    sigma = -np.polyfit(p, np.log(np.array(errors)), 1)[0]
    assert sigma > 2.5, sigma


def test_independent_quadrature_path_agrees_with_production_error():
    """Second measurement path (numpy, higher Gauss order) must reproduce L2."""
    x, y, coeff, _ = solve_advection_2d(
        ncells=SMALL_N, degree=2, final_time=1.0, velocity=VELOCITY, cfl=0.2,
    )
    prod = float(error_2d(x, y, coeff, 1.0, VELOCITY)["relative_l2"])
    indep = independent_relative_l2(x, y, coeff, 1.0, VELOCITY)
    assert indep["quadrature_order"] > 2 * 2 + 4  # genuinely a different order
    assert abs(indep["relative_l2"] - prod) / prod < 1e-10


def test_solve_level_reports_temporal_convergence():
    level = solve_level(2, ncells=SMALL_N, cfl=0.2, temporal_tolerance=0.01)
    assert level["temporal_converged"] is True
    assert level["refinements"] >= 1
    assert level["dofs"] == SMALL_N * SMALL_N * 9
    assert set(level["ncells"]) == {SMALL_N}
    # refining the CFL may only lower the measured error (equal when the first
    # attempt was already temporally converged)
    assert 0.0 < level["relative_l2"] <= level["history"][0]["relative_l2"] * (1 + 1e-5)


def test_p_ladder_diagnostics_and_anchor_table_are_consistent():
    ladder = p_ladder(degrees=(1, 2, 3), ncells=SMALL_N, cfl=0.2)
    assert len(ladder["levels"]) == 3
    assert ladder["diagnostics"]["monotone_decreasing"] is True
    rows = anchor_table(ladder)
    assert [r["degree"] for r in rows] == [1, 2, 3]
    assert rows[0]["reduction_vs_previous"] is None
    for prev, cur in zip(rows, rows[1:]):
        assert cur["relative_l2"] < prev["relative_l2"]


def test_falsification_suite_still_rejects_algebraic_laws():
    """The predeclared criterion (shared with the 1-D case) must stay sharp."""
    control = falsification_control(tuple(DEGREES))
    assert control["gate_pass"] is True
    rejected = [
        name for name, row in control["laws"].items()
        if not row["passes_discriminating"]
    ]
    for name in ("constant", "algebraic_p^-1", "algebraic_p^-2",
                 "algebraic_p^-3", "algebraic_p^-4", "algebraic_p^-5"):
        assert name in rejected


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="artifact not generated yet")
def test_published_artifact_is_live_and_self_consistent():
    """Bind the published artifact to the current code with one live level."""
    report = json.loads(RESULTS_JSON.read_text())
    assert report["schema"] == "tensordg-p-refinement-exponential-2d/1"
    assert report["case"] == CASE_KEY
    levels = report["results"]["levels"]
    assert [lv["degree"] for lv in levels] == list(DEGREES)
    errors = report["results"]["errors"]
    # stored diagnostics must be reproducible from the stored errors alone
    p = np.array(report["results"]["degrees"], dtype=float)
    sigma = -np.polyfit(p, np.log(np.array(errors)), 1)[0]
    assert sigma == pytest.approx(
        report["results"]["diagnostics"]["lsq_decay_rate"], rel=1e-9,
    )
    # every stored level must be temporally converged and monotone
    assert all(lv["temporal_converged"] for lv in levels)
    assert all(a > b for a, b in zip(errors, errors[1:]))
    # live recompute of the cheapest published level (p=1, 8x8, its own cfl)
    first = levels[0]
    x, y, coeff, md = solve_advection_2d(
        ncells=tuple(first["ncells"]), degree=first["degree"],
        final_time=report["problem"]["final_time"], velocity=VELOCITY,
        cfl=first["cfl"],
    )
    live = float(error_2d(x, y, coeff, report["problem"]["final_time"], VELOCITY)[
        "relative_l2"
    ])
    assert live == pytest.approx(first["relative_l2"], rel=1e-9)
    assert md["steps"] == first["steps"]
    # Honest FAIL lockdown: the 2-D ladder passes the anchor criterion and every
    # composite clause except "pairwise decay rates non-decreasing" (a 1.9 %
    # pre-asymptotic dip at p=2->3). If someone silently re-tunes the criterion
    # this test fails, which is the point.
    conditions = report["checks"]["criterion"]["discriminating"]["conditions"]
    assert report["checks"]["criterion"]["decay_rate"]["pass"] is True
    assert conditions["sigma_above_threshold"]["pass"] is True
    assert conditions["error_ratio_above_threshold"]["pass"] is True
    assert conditions["monotone_decreasing"]["pass"] is True
    assert conditions["decay_rates_non_decreasing"]["pass"] is False
    assert report["verdict"] == "FAIL"
    assert report["checks"]["degree_guard"]["pass"] is True
    assert report["checks"]["max_path_dispersion"] < 1e-6


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "transfer_diagnostics.json").exists(),
    reason="transfer diagnostics not generated yet",
)
def test_transfer_diagnostics_support_the_fail_reasoning():
    """The FAIL is attributed: symmetric solver, exponential beats algebraic fit."""
    data = json.loads((ARTIFACT_DIR / "transfer_diagnostics.json").read_text())
    assert data["predeclared_gate"]["verdict"] == "FAIL"
    assert data["predeclared_gate"]["failing_clause"] == "decay_rates_non_decreasing"
    assert data["rotational_symmetry"]["pass"] is True
    assert data["rotational_symmetry"]["max_relative_difference"] < 1e-12
    assert data["fit_comparison"]["exponential_preferred"] is True
    assert data["rate_trend"]["last_over_first"] > 1.0
    assert data["portable_clause"]["pass"] is True
    assert data["portable_falsification"]["gate_pass"] is True


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="artifact not generated yet")
def test_published_provenance_hashes_the_artifact():
    import hashlib

    provenance = json.loads((ARTIFACT_DIR / "provenance.json").read_text())
    digest = hashlib.sha256(RESULTS_JSON.read_bytes()).hexdigest()
    assert provenance["artifact_sha256"] == digest
    assert provenance["case"] == CASE_KEY
    assert provenance["degrees"] == list(DEGREES)
    assert provenance["git_commit"], "provenance must record the producing commit"


def test_exponential_fit_matches_declared_reference_slope():
    """Babuska-Suri statement sanity: sigma must be finite and positive."""
    ladder = p_ladder(degrees=(1, 2, 3), ncells=SMALL_N, cfl=0.2)
    sigma = ladder["diagnostics"]["lsq_decay_rate"]
    assert math.isfinite(sigma) and sigma > 0.0
    assert ladder["gates"]["legacy_slope"]["pass"] is True