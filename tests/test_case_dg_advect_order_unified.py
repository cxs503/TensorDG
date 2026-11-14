"""Integration test for the CASE-DG-ADVECT-ORDER unified report.

Verifies that the assembled metrics are self-consistent and all checks pass.

    PYTHONPATH=. pytest -q tests/test_case_dg_advect_order_unified.py
"""
from __future__ import annotations

import json
import pathlib

import pytest

METRICS = pathlib.Path("validation/advection_order_unified/metrics.json")
SUMMARY = pathlib.Path("validation/advection_order_unified/summary.md")
CSV = pathlib.Path("validation/advection_order_unified/convergence.csv")
FIG_DIR = pathlib.Path("docs/figures")


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip("unified metrics not yet assembled")
    return json.loads(METRICS.read_text())


def test_metrics_file_exists():
    assert METRICS.exists(), "unified metrics.json not found"


def test_summary_file_exists():
    assert SUMMARY.exists(), "unified summary.md not found"


def test_csv_file_exists():
    assert CSV.exists(), "unified convergence.csv not found"


def test_figures_exist():
    expected = [
        "advorder_unified_convergence.png",
        "advorder_unified_error_split.png",
        "advorder_unified_p0_constant.png",
    ]
    for name in expected:
        p = FIG_DIR / name
        assert p.exists(), f"figure {name} not found"
        assert p.stat().st_size > 1000, f"figure {name} is too small"


def test_verdict_is_pass(metrics):
    assert metrics["verdict"] == "PASS", f"expected PASS, got {metrics['verdict']}"


def test_all_checks_pass(metrics):
    for name, value in metrics["checks"].items():
        assert value is True, f"check {name} failed: {value}"


def test_u1_rate_check(metrics):
    """U1: P1-P4 observed order vs ref within ±0.10."""
    comparison = metrics["p1p4"]["comparison"]
    tol = metrics["thresholds"]["RATE_TOL"]
    for row in comparison:
        gap = row["order_gap_vs_ref"]
        assert gap is not None, f"P{row['degree']}: order_gap_vs_ref is None"
        assert abs(gap) <= tol, (
            f"P{row['degree']}: order gap {gap:+.4f} exceeds tolerance ±{tol}"
        )


def test_u2_p0_order(metrics):
    """U2: P0 fine ladder order recovers 1.00 within 0.01."""
    p0 = metrics["p0"]
    fine_order = p0["derived"]["order_analytic_fine"]
    tol = metrics["thresholds"]["P0_FINE_ORDER_TOL"]
    assert abs(fine_order - 1.0) <= tol, (
        f"P0 fine order {fine_order:.4f} deviates from 1.0 by more than {tol}"
    )


def test_u3_p0_constant(metrics):
    """U3: P0 asymptotic constant matches closed form within 0.1%."""
    p0 = metrics["p0"]
    const_err = p0["derived"]["constant_rel_error"]
    tol = metrics["thresholds"]["P0_CONST_TOL"]
    assert const_err <= tol, (
        f"P0 constant relative error {const_err:.4%} exceeds tolerance {tol:.1%}"
    )


def test_u4_time_share(metrics):
    """U4: P1-P4 time error share <= 10% at cfl=0.2."""
    error_split = metrics["p1p4"]["error_split"]
    tol = metrics["thresholds"]["TIME_SHARE_TOL"]
    for s in error_split:
        assert s["time_share"] <= tol, (
            f"P{s['degree']} N={s['ncells']}: time share {s['time_share']:.4f} "
            f"exceeds tolerance {tol}"
        )


def test_u5_solver_exact(metrics):
    """U5: P0 solver matches semidiscrete exact within 0.001%."""
    p0 = metrics["p0"]
    solver_exact = p0["derived"]["solver_vs_exact_max_rel"]
    tol = metrics["thresholds"]["SOLVER_EXACT_TOL"]
    assert solver_exact <= tol, (
        f"P0 solver vs exact relative error {solver_exact:.4e} "
        f"exceeds tolerance {tol:.3e}"
    )


def test_p0_verdict_pass(metrics):
    assert metrics["p0"]["verdict"] == "PASS"


def test_p1p4_rate_all_pass(metrics):
    """All P1-P4 rate checks pass (M1_rate in round 2)."""
    for row in metrics["p1p4"]["comparison"]:
        assert row["rate_ok"] is True, (
            f"P{row['degree']}: rate_ok is False "
            f"(gap={row['order_gap_vs_ref']:+.4f})"
        )


def test_summary_contains_verdict():
    content = SUMMARY.read_text()
    assert "PASS" in content, "summary.md does not contain PASS verdict"


def test_summary_contains_reference():
    content = SUMMARY.read_text()
    assert "Sun" in content, "summary.md does not cite the reference"
    assert "10.1090/mcom/4037" in content, "summary.md does not include DOI"


def test_csv_has_correct_columns():
    content = CSV.read_text()
    header = content.split("\n")[0]
    expected = ["degree", "integrator", "rk_stages", "ref_column",
                "ref_fitted_order", "got_fitted_order", "order_gap", "rate_ok"]
    for col in expected:
        assert col in header, f"CSV missing column: {col}"


def test_csv_has_four_rows():
    lines = CSV.read_text().strip().split("\n")
    assert len(lines) == 5, f"expected 5 lines (header + 4 data), got {len(lines)}"
