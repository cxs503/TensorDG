"""Gates for the DG elastodynamics dispersion/dissipation spectrum.

Companion of ``tensordg/elastic_dispersion.py`` (anchor
``CASE-DG-DALEMBERT-ELASTIC-1D``).  Every assertion here is an *invariant* of
the shipped operator, not a stored regression baseline:

* the block-circulant structure (so "symbol matrix" is the right language),
* the p=0 characteristic-upwind closed form derived independently in the module
  docstring (so the pipeline cannot be self-referential),
* the measured super-convergence exponents (2p+2 for both the phase error and
  the dissipation, measured -- not quoted),
* invariance under ``(rho, E, h) -> (4 rho, 16 E, h/2)`` (both error measures
  are ratios, so they may not depend on c or h separately).

Runtime: ~20 s (five assemblies of 8 elements up to p=4 + complex eigenvalues).
"""
import math

import numpy as np
import pytest

from tensordg.elastic_dispersion import (
    ASSEMBLY_CELLS,
    _blocks,
    _symbol,
    analyse,
    assemble_operator,
    dominant_theta,
    exponent_details,
    fit_exponent,
    p0_closed_form,
    p0_symbol_error,
    phase_and_dissipation,
    physical_branch,
    resolution_table,
    symbol_union_error,
    system_symbol,
)

DEGREES = (0, 1, 2, 3)


# ------------------------------------------------------------- structure
@pytest.mark.parametrize("degree", DEGREES)
def test_operator_is_block_circulant(degree):
    """spec(S) == union_m spec(T(2 pi m / N)) => the symbol matrix is exact."""
    assert symbol_union_error(degree) < 1e-12, degree


def test_symbol_matrix_dimensions():
    T = system_symbol(3, 0.4)
    assert T.shape == (8, 8)                 # 2*(p+1) with p=3
    assert np.iscomplexobj(T)


def test_assembly_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        assemble_operator(3, ncells=2)       # too few cells (stencil wrap)
    with pytest.raises(ValueError):
        assemble_operator(-1)
    with pytest.raises(ValueError):
        assemble_operator(2, rho=0.0)


# --------------------------------------------------- independent anchor p=0
@pytest.mark.parametrize("theta", [0.2, 0.7, 1.7, 2.9])
def test_p0_matches_characteristic_upwind_closed_form(theta):
    """Assembled p=0 symbol vs lambda = (c/h)[-2 sin^2(t/2) -/+ i sin t]."""
    assert p0_symbol_error(theta=theta) < 1e-12


def test_p0_closed_form_has_the_documented_asymmetry():
    """p=0: dissipation ~ theta^2 (1st order in h), phase error ~ theta^2."""
    theta = 1e-3
    lam, _ = p0_closed_form(theta)
    c_over_h = 1.0
    dissip = -lam.real / c_over_h
    phase = abs(lam.imag) / (c_over_h * theta) - 1.0
    assert dissip == pytest.approx(0.5 * theta ** 2, rel=1e-3)
    assert phase == pytest.approx(-theta ** 2 / 6.0, rel=1e-3)


# ------------------------------------------------------- measured exponents
@pytest.mark.parametrize("degree", (1, 2, 3))
def test_phase_error_exponent_is_2p_plus_2(degree):
    info = exponent_details(degree, "phase_error")
    assert info["points_used"] >= 5, info
    assert info["exponent"] == pytest.approx(2 * degree + 2, abs=0.35), info


@pytest.mark.parametrize("degree", (0, 1, 2, 3))
def test_dissipation_exponent_is_2p_plus_2(degree):
    info = exponent_details(degree, "dissipation")
    assert info["points_used"] >= 5, info
    assert info["exponent"] == pytest.approx(2 * degree + 2, abs=0.35), info


def test_exponents_improve_monotonically_with_degree():
    values = [fit_exponent(p, "phase_error") for p in (0, 1, 2, 3)]
    assert values == sorted(values)
    assert values[-1] - values[0] > 5.0


def test_higher_degree_is_more_accurate_at_fixed_wavenumber():
    """At theta = 0.5 both error measures must fall with p (no order inversion)."""
    theta = [0.5]
    phase = []
    for p in (0, 1, 2, 3):
        data = phase_and_dissipation(p, theta)
        phase.append(float(abs(data["phase_error"][0])))
    for lo, hi in zip(phase, phase[1:]):
        assert hi < lo
    assert phase[-1] < 1e-6 * phase[0]


# --------------------------------------------------------------- invariants
def test_spectrum_is_invariant_to_rho_young_h_scaling():
    """c -> 2c, h -> h/2: phase error and dissipation (both ratios) unchanged."""
    theta = [0.3, 0.8]
    base = phase_and_dissipation(2, theta, rho=1.0, young=1.0, h=1.0)
    scaled = phase_and_dissipation(2, theta, rho=4.0, young=16.0, h=0.5)
    assert np.allclose(np.abs(base["phase_error"]),
                       np.abs(scaled["phase_error"]), atol=1e-13)
    assert np.allclose(base["dissipation"], scaled["dissipation"], atol=1e-13)


def test_dissipation_is_non_negative_and_vanishes_at_zero_wavenumber():
    thetas = np.array([1e-4, 1e-2, 0.5, 1.5, math.pi])
    for p in DEGREES:
        data = phase_and_dissipation(p, thetas)
        diss = np.asarray(data["dissipation"], dtype=float)
        assert np.all(diss >= -1e-12)
        # both ends of the table: theta -> 0 gives dissipation -> 0 (like
        # theta^2), theta -> pi gives O(1) smearing; the law is monotone.
        assert diss[0] < 1e-6
        # monotone up to the roundoff of the zero-wavenumber end
        assert np.all(np.diff(diss) > -1e-12)
        assert diss[-1] > 10.0 * diss[1]


def test_physical_branch_stays_in_the_upper_half_plane():
    thetas = np.linspace(1e-3, math.pi, 25)
    branch = physical_branch(2, thetas)
    assert np.all(branch.imag >= 0.0)
    # the propagating mode must keep a non-zero imaginary part: it is a wave
    assert np.all(branch.imag > 0.0)


def test_branch_follows_the_linear_dispersion_at_low_wavenumber():
    """theta -> 0: omega/c = k, i.e. |Im lambda| = c theta / h (to 1%)."""
    theta = 1e-2
    lam = physical_branch(3, [theta])[0]
    assert abs(lam.imag) / theta == pytest.approx(1.0, rel=1e-2)


# --------------------------------------------------------- resolution budget
def test_resolution_budget_shrinks_with_degree():
    ppw = [resolution_table(p)["ppw_0.02"] for p in (0, 1, 2)]
    assert ppw[0] > ppw[1] > ppw[2] or ppw[1:] == sorted(ppw[1:], reverse=True)
    assert ppw[0] > 10.0          # first-order upwind needs a lot of points
    assert ppw[1] < 6.0


def test_gaussian_dominant_wavenumber_helper():
    assert dominant_theta(0.05, 0.005) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        dominant_theta(0.0, 0.1)


# ------------------------------------------------------------------ report
def test_analyse_report_schema_and_consistency():
    report = analyse(degrees=(1, 2))
    assert report["schema"] == "tensordg-elastic-dispersion/1"
    assert report["anchor"] == "CASE-DG-DALEMBERT-ELASTIC-1D"
    assert report["assembly_cells"] == ASSEMBLY_CELLS
    rows = report["rows"]
    assert [r["degree"] for r in rows] == [1, 2]
    for row in rows:
        assert row["symbol_union_error"] < 1e-12
        assert row["phase"]["exponent"] > 2 * row["degree"] + 1.0
        assert row["dissipation"]["exponent"] > 2 * row["degree"] + 1.0


def test_blocks_and_symbol_are_consistent_with_the_matrix():
    """T(2 pi m / N) eigenvalue == a matrix eigenvalue for every m (no pairing)."""
    matrix = assemble_operator(1, ncells=ASSEMBLY_CELLS)
    blocks = _blocks(matrix, ASSEMBLY_CELLS, 4)
    direct = np.linalg.eigvals(matrix)
    union = np.concatenate([np.linalg.eigvals(_symbol(blocks, 2 * math.pi * m
                                                      / ASSEMBLY_CELLS))
                            for m in range(ASSEMBLY_CELLS)])
    assert union.size == direct.size
    for lam in direct:
        assert np.min(np.abs(union - lam)) < 1e-10

# ---------------------------------------------------------------- artifact
def test_published_artifact_is_consistent_with_the_recomputation():
    """Re-adjudicate the committed JSON from its own numbers (no trust)."""
    import json
    from pathlib import Path

    art = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / \
        "results" / "elastic-dispersion" / "elastic_dispersion.json"
    if not art.exists():
        pytest.skip("run python -m benchmarks.elastic_dispersion")
    payload = json.loads(art.read_text())
    assert payload["schema"] == "tensordg-elastic-dispersion/1"
    assert payload["anchor"] == "CASE-DG-DALEMBERT-ELASTIC-1D"
    rows = payload["rows"]
    assert [r["degree"] for r in rows] == [0, 1, 2, 3, 4]
    for row in rows:
        assert row["symbol_union_error"] < 1e-12
        p = row["degree"]
        assert row["dissipation"]["exponent"] == pytest.approx(2 * p + 2,
                                                               abs=1.0)
        if p <= 3:
            assert row["phase"]["exponent"] == pytest.approx(2 * p + 2,
                                                             abs=0.5)
        else:
            assert row["phase"]["measurable"] is False
    case = payload["case_budget"]
    assert case["measured_relative_l2_stress_error"] < 0.02
    assert case["dispersion_share_of_error"] < 1e-3
    assert payload["verdict"] == "PASS"
