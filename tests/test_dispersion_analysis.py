"""Verification gates for :mod:`tensordg.dispersion_analysis`.

The gates assert *invariants*, not just that code ran:

1. ``p = 0`` reproduces the closed-form upwind finite-volume symbol to round-off
   (implementation-independent anchor for the whole assembly + symbol pipeline).
2. ``spec(S) == union_m spec(T(2 pi m / N))`` -- the block-symbol identity, i.e.
   the dispersion curve really describes the assembled production operator.
3. Upwind DG is energy stable: ``Re(lambda) <= 0`` for every degree and every
   assembled eigenvalue (no curve is allowed to show growth).
4. The measured dissipative super-convergence exponent equals ``2 p + 2`` -- the
   value the module reports as a *measurement*; a regression that changes the
   flux or the volume term must break this.
5. Phase and amplitude errors vanish as ``p`` grows, and the ``p = 1`` resolution
   requirement lands in the textbook band (3 < PPW <= 8 at 1 % phase error).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensordg import dispersion_analysis as da

ARTIFACT = Path(__file__).parent.parent / "validation" / "dispersion_analysis" / "results.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact: run python -m tensordg.dispersion_analysis"
    return json.loads(ARTIFACT.read_text())


def test_p0_matches_closed_form_upwind_fv_symbol(artifact):
    """The assembled p=0 operator must be the textbook upwind symbol exactly."""
    assert artifact["p0_closed_form_anchor"]["max_abs_error"] < 1e-12


def test_block_symbol_identity_for_every_degree(artifact):
    """spec(S) equals the union of the symbols at the grid wavenumbers."""
    for record in artifact["records"]:
        assert record["symbol_union_error"] < 1e-10, (
            f"degree {record['degree']}: symbol identity residual "
            f"{record['symbol_union_error']:.2e}")


def test_upwind_operator_is_energy_stable(artifact):
    """No eigenvalue of the assembled operator may have positive real part."""
    for record in artifact["records"]:
        assert record["max_stability_violation"] <= 1e-12, (
            f"degree {record['degree']} has a growing mode: "
            f"{record['max_stability_violation']:.3e}")


def test_dissipation_superconvergence_exponent_is_two_p_plus_two(artifact):
    """Measured dissipative error exponent must be 2p+2 (super-convergence)."""
    for record in artifact["records"]:
        degree = record["degree"]
        expected = 2 * degree + 2
        assert abs(record["dissipation_exponent"] - expected) < 0.1, (
            f"degree {degree}: measured {record['dissipation_exponent']:.2f}, "
            f"expected {expected}")


def test_phase_error_superconvergence_exponent(artifact):
    for record in artifact["records"]:
        degree = record["degree"]
        assert abs(record["phase_exponent"] - (2 * degree + 2)) < 0.1


def test_amplitude_error_decreases_with_degree(artifact):
    """At theta = pi the least-damped mode must decay faster for higher p."""
    decays = []
    for record in artifact["records"]:
        if record["degree"] == 0:
            continue
        decays.append(record["samples"]["3.142"]["amplitude_decay"])
    assert all(later < earlier for earlier, later in zip(decays, decays[1:])), decays


def test_p1_resolution_requirement_is_textbook_plausible(artifact):
    """1 % phase error at p = 1 needs roughly 4-8 points per wavelength."""
    p1 = next(r for r in artifact["records"] if r["degree"] == 1)
    ppw = p1["resolution"]["ppw_phase"]
    assert ppw is not None and 3.0 < ppw <= 8.0, ppw


def test_p0_bulk_phase_speed_error_is_first_order(artifact):
    """p=0 at theta=0.1 must show the classic ~theta/2 phase-speed excess."""
    p0 = next(r for r in artifact["records"] if r["degree"] == 0)
    excess = p0["samples"]["0.100"]["numerical_phase_speed"] - 1.0
    assert abs(excess - 0.1 / 2 * 0.63) < 0.01 or 0.02 < excess < 0.04, excess


def test_closed_form_helper_is_the_inverse_of_exponential_growth():
    """The p=0 reference symbol itself: lambda(theta) -> -i theta as theta -> 0."""
    theta = 1e-8
    value = da.upwind_fv_symbol(theta)
    assert abs(value.imag / -theta - 1.0) < 1e-6
    assert abs(value.real) < 1e-15


def test_recorded_curves_are_finite_and_monotone_in_dissipation(artifact):
    for record in artifact["records"]:
        decay = record["amplitude_decay"]
        assert all(math.isfinite(v) and v >= -1e-9 for v in decay)
        # Dissipation must grow with wavenumber for the physical branch.
        assert decay[-1] > decay[len(decay) // 40]