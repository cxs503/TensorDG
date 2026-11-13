"""Case-pool entry: ``CASE-DG-PREFINEMENT-EXPONENTIAL`` (p-type exponential decay).

Every assertion compares a *computed* field against the analytic travelling wave
``sin(2*pi*(x - t))`` or against a predeclared threshold -- no stored regression
baselines, no transcribed numbers.

Runtime: the module-scoped ``quick_report`` fixture runs the canonical p=1..3
ladder once (≈20 s on one core); tests that need p=1..6 read the published
artifact ``docs/benchmarks/results/p-refinement-exponential/case_pool_entry.json``
and are skipped when it is absent.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from tensordg.cases import CASES
from tensordg.cases.prefinement_exponential import (
    CASE_KEY,
    DEGREES,
    FLOOR_LIMITED_DEGREES,
    LEDGER_ANCHOR,
    LEGACY_SLOPE_THRESHOLD,
    MAX_DEGREE,
    NCELLS,
    REFERENCE,
    anchor_table,
    exact_solution,
    initial_condition,
    p_ladder,
    prefinement_parameters,
    solve_level,
)
from tensordg.convergence import exponential_decay_criterion, p_decay_diagnostics

ARTIFACT = (Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
            / "results" / "p-refinement-exponential" / "case_pool_entry.json")
QUICK_DEGREES = (1, 2, 3)


@pytest.fixture(scope="module")
def quick_report():
    """Canonical p=1..3 ladder, computed once per test module."""
    return p_ladder(degrees=QUICK_DEGREES)


# --------------------------------------------------------------- registration
def test_case_key_is_registered():
    assert "prefinement_exponential" in CASES, (
        "prefinement_exponential missing from tensordg.cases.CASES")
    assert CASES["prefinement_exponential"] == CASE_KEY


def test_parameters_dict_is_complete():
    params = prefinement_parameters()
    required = {
        "case_key", "ledger_anchor", "reference", "domain", "periodic", "pde",
        "initial_condition", "exact_solution", "ncells", "final_time", "speed",
        "degrees", "cfl_base", "temporal_tolerance", "legacy_slope_threshold",
        "max_degree", "floor_limited_degrees", "metric",
    }
    missing = required - set(params)
    assert not missing, f"parameters missing {sorted(missing)}"
    # JSON-serialisable (reported into artifacts)
    json.dumps(params)
    assert params["periodic"] is True
    assert params["domain"] == [0.0, 1.0]
    assert params["ncells"] == NCELLS
    assert params["degrees"] == list(DEGREES)
    assert params["ledger_anchor"] == LEDGER_ANCHOR


def test_reference_is_citable():
    """The ledger anchor names Babuska & Suri 1994 with a DOI -- pin it exactly."""
    assert REFERENCE["year"] == 1994
    assert "Babuska" in REFERENCE["authors"] and "Suri" in REFERENCE["authors"]
    assert REFERENCE["journal"] == "SIAM Review"
    assert REFERENCE["volume"] == 36 and REFERENCE["issue"] == 4
    assert REFERENCE["pages"] == "578-632"
    assert REFERENCE["doi"] == "10.21236/ada260237"
    assert "exp(-sigma p)" in REFERENCE["statement"]


def test_library_cap_and_floor_limited_degrees_declared():
    assert MAX_DEGREE == 8
    assert all(p > max(DEGREES) for p in FLOOR_LIMITED_DEGREES)
    assert all(p <= MAX_DEGREE for p in FLOOR_LIMITED_DEGREES)


# ------------------------------------------------------------------- analysis
def test_exact_solution_reduces_to_initial_condition_at_t0():
    x = torch.linspace(0.0, 1.0, 57, dtype=torch.float64)
    assert torch.allclose(exact_solution(x, 0.0), initial_condition(x),
                          atol=1e-14, rtol=0.0)


def test_exact_solution_is_one_period_periodic():
    """T=1 with speed 1 shifts by exactly one period -> exact solution == u0."""
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    assert torch.allclose(exact_solution(x, 1.0), initial_condition(x),
                          atol=1e-13, rtol=0.0)


def test_single_level_matches_the_analytic_wave():
    """A P2 solve must sit far below the ledger's 3 % bar against the analytic wave."""
    level = solve_level(2)
    assert math.isfinite(level["relative_l2"])
    assert level["relative_l2"] < 3.0e-2
    assert level["temporal_converged"] is True
    assert level["dofs"] == NCELLS * 3
    assert level["history"], "temporal refinement history must be recorded"


def test_quick_ladder_passes_both_predeclared_gates(quick_report):
    assert quick_report["temporal_all_converged"] is True
    legacy = quick_report["gates"]["legacy_slope"]
    discriminating = quick_report["gates"]["discriminating"]
    # ledger tolerance: semi-log slope < -0.5 per unit p
    assert abs(legacy["measured"]) > LEGACY_SLOPE_THRESHOLD
    assert legacy["pass"] is True
    # composite discriminating gate
    assert discriminating["pass"] is True
    assert quick_report["pass"] is True
    diag = quick_report["diagnostics"]
    assert diag["monotone_decreasing"] is True
    # non-decreasing pairwise decay rates: the algebraic-law fingerprint is absent
    rates = diag["decay_rates"]
    assert all(b >= a * (1.0 - 1e-9) for a, b in zip(rates, rates[1:]))


def test_error_reduction_is_exponential_not_algebraic(quick_report):
    """Design order p+1 would cap the reduction at ~(p+2)/(p+1); spectral beats it."""
    diag = quick_report["diagnostics"]
    for p_fine, reduction in zip(QUICK_DEGREES[1:], diag["reduction_factors"]):
        assert reduction > (p_fine + 1) ** 2, (
            f"reduction {reduction:.1f}x at p={p_fine} is not clearly super-algebraic")


def test_anchor_table_is_consistent_with_the_ladder(quick_report):
    rows = anchor_table(quick_report)
    assert [r["degree"] for r in rows] == list(QUICK_DEGREES)
    assert [r["relative_l2"] for r in rows] == quick_report["errors"]
    assert rows[0]["reduction_vs_previous"] is None
    assert all(r["temporal_converged"] for r in rows)


# ------------------------------------------------------- negative controls
def test_discriminating_gate_rejects_algebraic_law():
    degrees = DEGREES
    errors = [1.0e-2 / (p + 1.0) ** 4 for p in range(len(degrees))]
    diag = p_decay_diagnostics(degrees, errors)
    assert abs(diag["lsq_decay_rate"]) > LEGACY_SLOPE_THRESHOLD  # legacy gate passes
    assert exponential_decay_criterion(diag)["pass"] is False     # composite rejects


def test_discriminating_gate_accepts_pure_exponential():
    degrees = DEGREES
    errors = [1.0e-2 * math.exp(-3.927 * p) for p in range(len(degrees))]
    diag = p_decay_diagnostics(degrees, errors)
    assert exponential_decay_criterion(diag)["pass"] is True


def test_benchmark_wrapper_delegates_to_library_gate():
    """One implementation of the gate: benchmark wrapper == library function."""
    from benchmarks.p_refinement_exponential import check_discriminating_criterion

    degrees = DEGREES
    for errors in ([1.0e-2 / (p + 1.0) ** 3 for p in range(len(degrees))],
                   [1.0e-2 * math.exp(-3.927 * p) for p in range(len(degrees))]):
        diag = p_decay_diagnostics(degrees, errors)
        assert (check_discriminating_criterion(diag)
                == exponential_decay_criterion(diag))


# -------------------------------------------------------------- input guards
@pytest.mark.parametrize("degree", [0, -1, MAX_DEGREE + 1])
def test_out_of_range_degree_is_rejected(degree):
    with pytest.raises(ValueError):
        solve_level(degree)


def test_ladder_needs_at_least_two_degrees():
    with pytest.raises(ValueError):
        p_ladder(degrees=(3,))


# ------------------------------------------------------------ artifact gate
@pytest.mark.skipif(not ARTIFACT.exists(), reason="case_pool_entry.json not generated")
def test_published_case_pool_entry_passes_all_gates():
    data = json.loads(ARTIFACT.read_text())
    assert data["case_key"] == CASE_KEY
    assert data["ledger_anchor"] == LEDGER_ANCHOR
    assert data["ladder"]["grades"] == list(DEGREES)
    assert data["ladder"]["temporal_all_converged"] is True
    assert data["gates"]["legacy_slope"]["pass"] is True
    assert data["gates"]["discriminating"]["pass"] is True
    assert data["negative_controls"]["gate_pass"] is True
    assert data["negative_controls"]["kinked"]["rejected_by_discriminating"] is True
    assert data["reproduction"]["applicable"] is True
    assert data["reproduction"]["all_pass"] is True, (
        "case-pool entry must reproduce the published p_refinement_results.json")
    assert data["pass"] is True
    # provenance: the artifact records the commit it was produced from
    assert data["provenance"]["git_commit"]
    assert len(data["artifact_sha256"]) == 64