"""Pytest gates for the p-refinement exponential convergence benchmark.

Case: CASE-DG-PREFINEMENT-EXPONENTIAL
Reference: Babuska & Suri (1994) SIAM Review 36:578

Run::
    pytest -q tests/test_p_refinement_exponential.py
"""
import json
import math
from pathlib import Path

import pytest

ARTIFACT = (Path(__file__).resolve().parent.parent
            / "docs" / "benchmarks" / "results" / "p-refinement-exponential"
            / "p_refinement_results.json")


def _artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("p_refinement_results.json missing; run python -m benchmarks.p_refinement_exponential")
    return json.loads(ARTIFACT.read_text())


def test_artifact_exists():
    assert ARTIFACT.exists(), "Run: python -m benchmarks.p_refinement_exponential"


def test_artifact_schema():
    art = _artifact()
    assert art["schema"] == "tensordg-p-refinement-exponential/1"
    assert art["case"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"


def test_verdict_passes():
    art = _artifact()
    assert art["verdict"] == "PASS"


def test_all_degrees_present():
    art = _artifact()
    assert art["results"]["degrees"] == [1, 2, 3, 4, 5, 6]


def test_errors_monotonically_decrease():
    art = _artifact()
    errors = art["results"]["errors"]
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], f"Error increased at p={art['results']['degrees'][i]}"


def test_decay_rate_exceeds_threshold():
    art = _artifact()
    sigma = art["results"]["diagnostics"]["lsq_decay_rate"]
    assert sigma > 0.5, f"Decay rate {sigma:.4f} below threshold 0.5"


def test_decay_rate_reasonable():
    """For sin(2*pi*x) on 16 cells, sigma should be in a plausible range."""
    art = _artifact()
    sigma = art["results"]["diagnostics"]["lsq_decay_rate"]
    # The sine function is entire, so convergence should be fast but not infinite
    # (limited by quadrature and floating point).
    assert 1.0 < sigma < 10.0, f"Decay rate {sigma:.4f} outside plausible range"


def test_p6_error_near_machine_precision():
    """At p=6 on 16 cells, the error should be very small but not zero."""
    art = _artifact()
    errors = art["results"]["errors"]
    p6_error = errors[-1]
    assert p6_error < 1e-9, f"p=6 error {p6_error:.2e} too large"
    assert p6_error > 1e-15, f"p=6 error {p6_error:.2e} suspiciously small (possible underflow)"


def test_reduction_factors_increase_with_p():
    """Higher p should give larger error reduction factors (exponential convergence)."""
    art = _artifact()
    factors = art["results"]["diagnostics"]["reduction_factors"]
    for i in range(1, len(factors)):
        assert factors[i] > factors[i - 1], (
            f"Reduction factor decreased: {factors[i-1]:.1f} -> {factors[i]:.1f}"
        )


def test_superlinear_convergence():
    art = _artifact()
    assert art["results"]["diagnostics"]["superlinear"] is True


def test_figures_generated():
    outdir = ARTIFACT.parent
    assert (outdir / "p_refinement_semilog.png").exists()
    assert (outdir / "p_refinement_log10.png").exists()


def test_csv_table_exists():
    outdir = ARTIFACT.parent
    assert (outdir / "p_refinement_table.csv").exists()
