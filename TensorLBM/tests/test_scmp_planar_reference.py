"""Tests for the independent two-component SCMP planar-interface reference.

These tests pin the *reference*, not the solver: they are the instrument against
which ``benchmarks/scmp_two_component_planar_interface.py`` scores TensorLBM's
MCMP path.  If the reference drifts, the anchor's verdict drifts with it.
"""

import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import scmp_planar_reference as REF  # noqa: E402
from tensorlbm.cross_force import pseudopotential_rho  # noqa: E402


def test_critical_coupling_is_inverse_density():
    """|G_c| = 1/rho_bar for psi = rho (mixture Hessian loses positivity)."""
    assert REF.critical_coupling(0.5) == pytest.approx(2.0, rel=1e-14)
    assert REF.critical_coupling(0.25) == pytest.approx(4.0, rel=1e-14)


def test_subcritical_and_repulsive_couplings_have_no_tieline():
    for G in (-1.99, -0.5, +1.0, +3.0, 0.0):
        assert REF.coexistence_tieline(G) == (0.5, 0.5)


def test_tieline_agrees_between_two_independent_numerical_routes():
    """Route A (transcendental root) vs route B (free-energy minimisation)."""
    for A in (2.05, 2.2, 2.5, 3.0, 5.0):
        a = REF.coexistence_rho_hi(A)
        b = REF.coexistence_rho_hi_by_energy(-A, n_grid=20001, n_golden=120)
        assert abs(a - b) < 1e-6, (A, a, b)


def test_tieline_satisfies_equal_chemical_potential_and_pressure():
    G = -3.0
    rhi, rlo = REF.coexistence_tieline(G)
    mu_a = REF.chemical_potentials(rhi, rlo, G)
    mu_b = REF.chemical_potentials(rlo, rhi, G)
    assert mu_a[0] == pytest.approx(mu_b[0], abs=1e-12)
    assert mu_a[1] == pytest.approx(mu_b[1], abs=1e-12)
    assert REF.mechanical_pressure(rhi, rlo, G) == pytest.approx(
        REF.mechanical_pressure(rlo, rhi, G), abs=1e-14)


def test_tieline_conserves_mass_and_deepens_with_coupling():
    prev = 0.5
    for A in (2.05, 2.2, 2.5, 3.0, 4.0):
        rhi, rlo = REF.coexistence_tieline(-A)
        assert rhi + rlo == pytest.approx(1.0, abs=1e-12)
        assert rlo < rhi
        assert rhi > prev, "demixing amplitude must grow with |G|"
        prev = rhi


def test_free_energy_reproduces_the_model_pressure():
    """p = rho1 mu1 + rho2 mu2 - f must equal cs^2 (rho1+rho2-G rho1 rho2)."""
    G = -2.3
    for r1, r2 in ((0.8, 0.2), (0.5, 0.5), (0.63, 0.37)):
        f = REF.free_energy_density(r1, r2, G)
        mu1, mu2 = REF.chemical_potentials(r1, r2, G)
        p_from_f = r1 * mu1 + r2 * mu2 - f
        assert p_from_f == pytest.approx(REF.mechanical_pressure(r1, r2, G), rel=1e-12)


def test_interface_dip_makes_the_pressure_flat():
    """The predicted interface density must equalise (1) across the interface."""
    G = -2.5
    rhi, rlo = REF.coexistence_tieline(G)
    s = REF.interface_density_dip(G)
    assert s < rhi + rlo, "Shan-Chen interfaces must be less dense than the bulks"
    assert REF.mechanical_pressure(0.5 * s, 0.5 * s, G) == pytest.approx(
        REF.mechanical_pressure(rhi, rlo, G), rel=1e-10)


def test_pressure_flatness_flags_a_wrong_coupling_scaling():
    """A factor-2 error in G must show up as a large pressure spread."""
    G = -2.5
    rhi, rlo = REF.coexistence_tieline(G)
    s = REF.interface_density_dip(G)
    prof1 = [rhi] * 4 + [0.5 * s] * 4 + [rlo] * 4
    prof2 = [rlo] * 4 + [0.5 * s] * 4 + [rhi] * 4
    good = REF.profile_pressure_flatness(prof1, prof2, G)
    bad = REF.profile_pressure_flatness(prof1, prof2, 2.0 * G)
    assert good["pressure_spread_relative"] < 1e-10
    assert bad["pressure_spread_relative"] > 0.05


def test_nonlinear_pseudopotential_reference_is_withheld():
    with pytest.raises(NotImplementedError):
        REF.coexistence_tieline(-6.0, psi_kind="exp")


def test_tie_line_error_metric_is_zero_on_the_reference_state():
    G = -2.2
    rhi, rlo = REF.coexistence_tieline(G)
    err = REF.tie_line_relative_error(rhi, rlo, G)
    assert err["err_max"] < 1e-14
    shifted = REF.tie_line_relative_error(rhi + 0.05, rlo - 0.05, G)
    assert shifted["err_max"] > 0.05


def test_solver_pseudopotential_matches_the_reference_convention():
    """The reference assumes psi = rho; the shipped linear psi must agree."""
    import torch
    rho = torch.tensor([0.3, 0.7], dtype=torch.float64)
    assert torch.allclose(pseudopotential_rho(rho), rho)
    assert REF.pseudopotential(0.42) == pytest.approx(0.42)