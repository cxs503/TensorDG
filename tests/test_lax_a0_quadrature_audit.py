"""Invariants for the A0 quadrature audit (offline, zero solver runs).

These tests pin the *measurement*, not the solver:

* the interpolation used to move the DG field off its cached nodes is exact;
* the breakpoint-aligned quadrature is order-invariant (the Q1 convergence gate);
* the committed pool artifact is reproduced bit-for-bit at the committed rule
  (the Q0 reuse gate);
* the recorded finding -- a fixed-order Gauss rule is *not* converged and the
  repo's 8-point rule lands below the 3 % bar while the converged value does not
  -- is asserted, so any silent change of the measure turns a test red.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lax_a0_quadrature_audit import (
    BASE_ORDER,
    CONVERGED_ORDER,
    FIELDS,
    OUT,
    POOL_METRICS,
    POOL_PROFILES,
    PRODUCTION_LEG,
    barycentric_matrix,
    load_fields,
    measure,
)
from benchmarks.lax_case_pool import ledger_acceptance_tolerance

METRICS = OUT / "metrics.json"


@pytest.fixture(scope="module")
def fields():
    return load_fields(POOL_PROFILES)


def test_interpolation_matrix_is_exact_for_the_polynomial_degree(fields):
    nodes = np.polynomial.legendre.leggauss(BASE_ORDER)[0]
    for degree in range(0, 10):                     # 10 nodes: exact up to degree 9
        coefficients = np.zeros(degree + 1)
        coefficients[-1] = 1.0
        targets = np.linspace(-0.999, 0.999, 37)
        matrix = barycentric_matrix(targets, nodes)
        interpolated = matrix @ np.polynomial.legendre.legval(nodes, coefficients)
        expected = np.polynomial.legendre.legval(targets, coefficients)
        assert np.max(np.abs(interpolated - expected)) < 1e-12


def test_interpolation_hits_cached_nodes_exactly(fields):
    nodes = np.polynomial.legendre.leggauss(BASE_ORDER)[0]
    matrix = barycentric_matrix(nodes, nodes)
    assert np.allclose(matrix, np.eye(BASE_ORDER), atol=0.0, rtol=0.0)


def test_committed_pool_value_is_reproduced_by_the_committed_rule(fields):
    pool = json.loads(POOL_METRICS.read_text())
    committed = pool["acceptance"]["per_leg"][PRODUCTION_LEG]["l2"]
    measured = measure(fields, "uniform", BASE_ORDER)
    for field in FIELDS:
        reference = committed[field]
        assert abs(measured["l2"][field] - reference) <= 1e-12 * abs(reference)


def test_aligned_quadrature_is_order_invariant(fields):
    """Q1: the aligned rule is converged by order 16 (the pre-registered gate)."""
    coarse = measure(fields, "aligned", 16)
    fine = measure(fields, "aligned", CONVERGED_ORDER)
    for field in FIELDS:
        assert abs(fine["l2"][field] - coarse["l2"][field]) <= 1e-10 * abs(fine["l2"][field])
    # order 10 is already within 1e-6; the mis-converged regime is *low* order only
    mid = measure(fields, "aligned", 10)
    for field in FIELDS:
        assert abs(fine["l2"][field] - mid["l2"][field]) <= 1e-6 * abs(fine["l2"][field])


def test_aligned_quadrature_differs_from_mass_subdivision(fields):
    """Naive equal-mass subdivision is *not* a converged rule (the audit's F2)."""
    converged = measure(fields, "aligned", CONVERGED_ORDER)["l2"]["density"]
    subdivided = measure(fields, "subdivision", 10, subdivisions=8)["l2"]["density"]
    assert abs(subdivided - converged) / converged > 1e-4


def test_gauss_rule_leaves_a_band_of_plausible_values():
    """A fixed-order rule scatters by several percent -- the FAIL margin is smaller."""
    committed = json.loads(METRICS.read_text())
    values = [committed["uniform_sweep"][key]["l2"]["density"]
              for key in committed["uniform_sweep"]]
    spread = (max(values) - min(values)) / min(values)
    assert spread > 0.05, "order sweep must expose quadrature sensitivity"
    converged = committed["ledger"]["fields"]["density"]["converged_aligned"]
    assert abs(committed["uniform_sweep"]["10"]["l2"]["density"] - converged) / converged > 1e-3


def test_repo_eight_point_rule_is_below_the_bar_while_converged_is_not():
    """The green `tests/test_lax_1d.py` is a quadrature artifact (recorded finding)."""
    committed = json.loads(METRICS.read_text())
    tolerance = ledger_acceptance_tolerance()
    order8 = committed["uniform_sweep"]["8"]["l2"]["density"]
    converged = committed["ledger"]["fields"]["density"]["converged_aligned"]
    assert order8 < tolerance, "8-point rule must land under 3 % (the artifact)"
    assert converged >= tolerance, "converged value must stay above 3 % (honest FAIL)"


def test_ledger_verdict_matches_the_pre_registered_rule():
    committed = json.loads(METRICS.read_text())
    tolerance = ledger_acceptance_tolerance()
    converged = committed["ledger"]["fields"]
    verdicts = {field: converged[field]["status"] for field in FIELDS}
    expected = {field: ("PASS" if converged[field]["converged_aligned"] < tolerance else "FAIL")
                for field in FIELDS}
    assert verdicts == expected
    assert committed["ledger"]["status"] == (
        "PASS" if all(v == "PASS" for v in expected.values()) else "FAIL")


def test_artifact_is_json_strict_and_carries_its_pre_registration():
    committed = json.loads(METRICS.read_text())
    assert committed["pre_registration"]["Q0"].startswith("uniform/order10")
    assert committed["provenance"]["solver_rerun"] is False
    assert committed["config"]["tolerance"] == ledger_acceptance_tolerance()
    assert committed["config"]["production_leg"] == PRODUCTION_LEG