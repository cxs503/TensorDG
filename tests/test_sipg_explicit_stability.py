"""Regression gates for the measured SIPG explicit-RK stability limit.

These tests have teeth in the sense that the two production heuristics
(``advection_diffusion.solve`` diffusion step, ``ns_dg_2d`` viscous step group)
are checked against an independent measurement rather than against themselves.
If a future change raises the polynomial degree range, weakens the penalty, or
edits the hard-coded constants, the safety factors move and the gate fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import sipg_explicit_stability as bench

ART = Path(__file__).resolve().parents[1] / "validation" / "sipg_explicit_stability"


def _results() -> dict:
    return json.loads((ART / "results.json").read_text())


# --------------------------------------------------------------------------- #
# integrator anchors
# --------------------------------------------------------------------------- #
def test_rk4_negative_real_axis_interval_matches_literature():
    z = bench.stability_interval("rk4")
    assert z == pytest.approx(bench.RK4_REAL_AXIS_LITERATURE, rel=1e-12)


def test_ssp_rk3_interval_matches_its_stability_polynomial():
    z = bench.stability_interval("ssp_rk3")
    # 1 + z + z^2/2 + z^3/6 has its negative-real-axis crossing at 2.5127...
    assert 2.51 < z < 2.52
    R = bench.ssp_rk3_amplification(complex(-z, 0.0))
    assert abs(R) == pytest.approx(1.0, abs=1e-12)
    assert abs(bench.ssp_rk3_amplification(complex(-(z + 1e-3), 0.0))) > 1.0


# --------------------------------------------------------------------------- #
# spatial operator: exact spectra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("penalty", [2.0, 20.0])
def test_one_dimensional_sipg_operator_is_positive_definite(penalty):
    """``K`` SPD => every eigenvalue of ``M^{-1}K`` is real and non-negative."""
    for degree in (1, 2, 3):
        row = bench.spectrum_1d(degree, 8, penalty)
        assert row["lam_min"] > 0.0
        assert row["lam_max"] > 0.0
        assert row["c"] > 0.0


def test_one_dimensional_spectral_constant_is_mesh_independent():
    coarse = bench.spectrum_1d(2, 16, 2.0)
    fine = bench.spectrum_1d(2, 32, 2.0)
    rel = abs(coarse["c"] - fine["c"]) / fine["c"]
    assert rel < 1e-2, rel


def test_two_dimensional_spectral_constant_is_twice_the_one_dimensional_one():
    """A product of two decoupled directions doubles the spectral radius."""
    one = bench.spectrum_1d(2, 32, 2.0)["c"]
    two = bench.spectrum_2d(12, 12, 2, 2.0)["c"]
    assert two / one == pytest.approx(2.0, rel=3e-2)


# --------------------------------------------------------------------------- #
# production operator (advection_diffusion.rhs) is dissipative
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_production_diffusion_operator_is_not_growing(degree):
    """No eigenvalue of the production operator may have a growing real part."""
    row = bench.production_dense_2d(4, degree, 2.0)
    assert row["re_max"] <= 1e-3 * row["lam_abs_max"], row
    assert row["im_max"] <= 1e-2 * row["lam_abs_max"], row


def test_dense_and_lanczos_channels_agree_on_the_largest_eigenvalue():
    dense = bench.production_dense_2d(4, 2, 2.0)
    lanczos = bench.production_lanczos_2d(4, 4, 2, 2.0)
    rel = abs(dense["c"] - lanczos["c"]) / lanczos["c"]
    assert rel < 2e-2, rel


# --------------------------------------------------------------------------- #
# the heuristics themselves
# --------------------------------------------------------------------------- #
def test_advection_diffusion_step_rule_stays_inside_the_measured_limit():
    """``dt_diff`` must be strictly below ``z_RK4/|lambda|max`` for every supported p."""
    z = bench.stability_interval("rk4")
    for degree in bench.AD_DEGREES:
        row = bench.production_lanczos_2d(16, 16, degree, 2.0)
        dt_true = z / row["lam_abs_max"]
        dt_code = bench.ad_dt_diff(degree, 2.0, row["h"])
        assert dt_code < dt_true, (degree, dt_code, dt_true)
        # documented thin margin: p=3 must still keep >= 1.25x
        assert dt_true / dt_code > 1.25, (degree, dt_true / dt_code)


def test_artifact_safety_factors_match_a_fresh_measurement():
    """The committed artifact must be reproducible (guards against stale numbers)."""
    results = _results()
    for row in results["consequences"]:
        degree = row["degree"]
        fresh = bench.production_lanczos_2d(16, 16, degree, 2.0)
        rel = abs(fresh["c"] - row["c"]) / row["c"]
        assert rel < 2e-2, (degree, fresh["c"], row["c"])


def test_ns_viscous_group_keeps_a_margin_above_one():
    """``S_max`` must exceed the hard-coded ``VISCOUS_STABILITY`` for every p."""
    results = _results()
    z = bench.stability_interval("ssp_rk3")
    for row in results["consequences"]:
        s_max = bench.ns_S_max(row["degree"], 2.0, row["c"], z)
        assert s_max > bench.CODE_NS_VISCOUS_STABILITY, (row["degree"], s_max)
        # the code comment claims 0.3-0.5 measured; record the actual finding
        assert s_max < 0.5, (row["degree"], s_max)


def test_artifact_provenance_hashes_the_committed_artifacts():
    provenance = json.loads((ART / "provenance.json").read_text())
    digest = hashlib.sha256((ART / "results.json").read_bytes()).hexdigest()
    assert provenance["results_sha256"] == digest
    assert (ART / provenance["figure"]).is_file()


# --------------------------------------------------------------------------- #
# end-to-end empirical closure on the production residual
# --------------------------------------------------------------------------- #
def test_empirical_bisection_reproduces_the_spectral_prediction():
    z = bench.stability_interval("rk4")
    empirical = bench.empirical_dt_max_2d(0, 4, 2.0)
    predicted_row = bench.production_lanczos_2d(4, 4, 0, 2.0)
    predicted = z / predicted_row["lam_abs_max"]
    rel = abs(empirical["dt_max_empirical"] - predicted) / predicted
    assert rel < 5e-2, rel
    assert empirical["growth_below"] <= 0.0 < empirical["growth_above"]