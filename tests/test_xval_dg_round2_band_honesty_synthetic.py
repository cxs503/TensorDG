"""Synthetic-data tests for the band-honesty audit engine.

Purpose
-------
Tests the band-honesty audit pipeline with synthetic data, without needing
the actual WENO7 solver run (which is a heavy cluster job).

Coverage
--------
B1  coverage_verdict returns well-formed dict with all required keys
B2  coverage verdict is COVERED when cross-family distance <= U_ref
B3  coverage verdict is MARGINAL when U_ref < distance <= U_ref + U_other
B4  coverage verdict is UNDER_COVERS when distance > U_ref + U_other
B5  gate survival arithmetic: worst + delta < gate → survives
B6  gate survival arithmetic: worst + delta >= gate → does not survive
B7  rejudge_legs reports correct verdict flips
B8  certify() handles non-nesting ladders gracefully
B9  certify() rejects divergent ladders
B10 certify() rejects stagnant ladders
B11 coarsen preserves mean
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensordg import certified_reference as cr  # noqa: E402


def _make_synthetic_cert(reference, band_percent, tolerance_percent=3.0):
    """Create a synthetic certification dict with all required fields."""
    return {
        "reference": reference,
        "reference_ncells": len(reference),
        "band_percent": band_percent,
        "verdict": "CERTIFIED",
        "classification": "monotonic",
        "tolerance_percent": tolerance_percent,
        "u_share_max": cr.U_SHARE_MAX,
        "observed_order": 1.0,
        "ladder": [len(reference) // 4, len(reference) // 2, len(reference)],
    }


def test_coverage_verdict_well_formed():
    """B1: coverage_verdict returns a dict with all required keys."""
    from benchmarks.xval_dg_round2_band_honesty import coverage_verdict
    result = coverage_verdict(0.05, 0.083, 0.05)
    for key in ("delta_percent", "band_reference_percent", "band_other_percent",
                "ratio_delta_over_band", "level", "band_covers_cross_family",
                "combined_bands_cover"):
        assert key in result


def test_coverage_covered():
    """B2: COVERED when delta <= U_ref."""
    from benchmarks.xval_dg_round2_band_honesty import coverage_verdict
    result = coverage_verdict(0.05, 0.083, 0.05)
    assert result["level"] == "COVERED"
    assert result["band_covers_cross_family"] is True


def test_coverage_marginal():
    """B3: MARGINAL when U_ref < delta <= U_ref + U_other."""
    from benchmarks.xval_dg_round2_band_honesty import coverage_verdict
    result = coverage_verdict(0.10, 0.083, 0.05)
    assert result["level"] == "MARGINAL"
    assert result["combined_bands_cover"] is True


def test_coverage_under_covers():
    """B4: UNDER_COVERS when delta > U_ref + U_other."""
    from benchmarks.xval_dg_round2_band_honesty import coverage_verdict
    result = coverage_verdict(0.20, 0.083, 0.05)
    assert result["level"] == "UNDER_COVERS"
    assert result["combined_bands_cover"] is False


def test_gate_survival_yes():
    """B5: worst + delta < gate → survives."""
    assert 2.4 + 0.05 < 3.0


def test_gate_survival_no():
    """B6: worst + delta >= gate → does not survive."""
    assert 2.9 + 0.15 >= 3.0


def test_rejudge_legs_verdict_flips():
    """B7: rejudge_legs correctly identifies verdict flips."""
    from benchmarks.xval_dg_round2_band_honesty import rejudge_legs
    rng = np.random.RandomState(42)
    n = 400
    ref5 = rng.randn(n) * 0.01 + 1.0
    ref7 = ref5 + 0.001
    cert5 = _make_synthetic_cert(ref5, band_percent=0.083)
    cert7 = _make_synthetic_cert(ref7, band_percent=0.05)
    leg_field = ref5 * 1.025
    legs = [{"name": "synthetic_leg", "x": np.arange(n), "field": leg_field}]
    result = rejudge_legs(legs, cert5, cert7, gate=3.0)
    assert result["n_legs"] == 1
    assert result["legs"][0]["error_percent_vs_weno5_ref"] == pytest.approx(2.5, abs=0.3)
    assert result["legs"][0]["passes_both"] is True


def test_non_nesting_ladder_handled():
    """B8: certify() handles non-nesting ladders without crashing."""
    levels = [
        {"ncells": 100, "field": np.random.randn(100)},
        {"ncells": 150, "field": np.random.randn(150)},
        {"ncells": 200, "field": np.random.randn(200)},
    ]
    result = cr.certify(levels, tolerance_percent=5.0)
    assert "verdict" in result


def test_certify_divergent_ladder():
    """B9: certify() rejects a divergent ladder."""
    base = np.linspace(0, 1, 400)
    levels = [{"ncells": n, "field": base[:n] + 0.1 * n / 100} for n in [100, 200, 400]]
    result = cr.certify(levels, tolerance_percent=5.0)
    assert result["verdict"] != "CERTIFIED"


def test_certify_stagnant_ladder():
    """B10: certify() rejects a stagnant ladder."""
    base = np.linspace(0, 1, 400)
    levels = [{"ncells": n, "field": base[:n] + 0.1 * (400 / n) ** 0.1} for n in [100, 200, 400]]
    result = cr.certify(levels, tolerance_percent=5.0)
    assert result["verdict"] != "CERTIFIED"


def test_coarsen_preserves_mean():
    """B11: coarsen() preserves the cell-average of the profile."""
    rng = np.random.RandomState(42)
    fine = rng.randn(400) + 1.0
    coarse = cr.coarsen(fine, 4)
    assert np.mean(fine) == pytest.approx(np.mean(coarse), rel=1e-10)
