"""Unit tests for the GCI (Grid Convergence Index) calculator.

Tests cover:
  * Basic GCI computation with known values
  * Three-grid GCI analysis
  * Apparent order computation
  * Edge cases and error handling
  * Asymptotic range verification
"""
from __future__ import annotations

import math

import pytest

from tensordg.converge import (
    GCIResult,
    compute_gci,
    compute_apparent_order,
    gci_three_grid,
)


# ---------------------------------------------------------------------------
# compute_apparent_order
# ---------------------------------------------------------------------------

class TestApparentOrder:
    def test_exact_p2(self):
        """For a second-order method with r=2, errors should halve by 4x."""
        p = compute_apparent_order(0.01, 0.04, 0.16, r=2.0)
        assert p == pytest.approx(2.0, abs=1e-10)

    def test_exact_p3(self):
        """Third-order: errors reduce by 8x each refinement."""
        p = compute_apparent_order(0.001, 0.008, 0.064, r=2.0)
        assert p == pytest.approx(3.0, abs=1e-10)

    def test_exact_p1(self):
        """First-order: errors halve each refinement."""
        p = compute_apparent_order(0.01, 0.02, 0.04, r=2.0)
        assert p == pytest.approx(1.0, abs=1e-10)

    def test_non_unit_exact(self):
        """Non-zero exact solution."""
        exact = 1.0
        f1 = exact + 0.01
        f2 = exact + 0.04
        f3 = exact + 0.16
        p = compute_apparent_order(f1, f2, f3, r=2.0)
        assert p == pytest.approx(2.0, abs=1e-10)

    def test_zero_error(self):
        """Zero error on finest grid → inf order."""
        p = compute_apparent_order(1.0, 1.0, 1.04, r=2.0)
        assert math.isinf(p)

    def test_refinement_ratio_4(self):
        """Non-standard refinement ratio."""
        p = compute_apparent_order(0.01, 0.16, 2.56, r=4.0)
        assert p == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_gci
# ---------------------------------------------------------------------------

class TestComputeGCI:
    def test_basic_gci(self):
        """Standard GCI computation."""
        result = compute_gci(
            fine_value=0.01, coarse_value=0.04,
            refinement_ratio=2.0, theoretical_order=2.0,
        )
        assert result.gci_fine == pytest.approx(1.25, rel=1e-10)
        assert result.gci_coarse == pytest.approx(5.0, rel=1e-10)

    def test_gci_third_order(self):
        """Third-order method."""
        result = compute_gci(
            fine_value=0.001, coarse_value=0.008,
            refinement_ratio=2.0, theoretical_order=3.0,
        )
        eps = abs(0.001 - 0.008) / 0.001
        gci_fine = 1.25 * eps / (8 - 1)
        assert result.gci_fine == pytest.approx(gci_fine, rel=1e-10)

    def test_gci_with_apparent_order(self):
        """Using apparent order instead of theoretical."""
        result = compute_gci(
            fine_value=0.012, coarse_value=0.045,
            refinement_ratio=2.0, apparent_order=1.9,
        )
        assert result.apparent_order == pytest.approx(1.9)

    def test_gci_extrapolated_value(self):
        """Richardson extrapolation."""
        result = compute_gci(
            fine_value=0.01, coarse_value=0.04,
            refinement_ratio=2.0, theoretical_order=2.0,
        )
        assert result.extrapolated_value == pytest.approx(0.0, abs=1e-12)

    def test_gci_asymptotic_ratio(self):
        """Asymptotic ratio for exact power-law with f_exact=0."""
        result = compute_gci(
            fine_value=0.01, coarse_value=0.04,
            refinement_ratio=2.0, theoretical_order=2.0,
        )
        assert result.asymptotic_ratio == pytest.approx(1.0, rel=1e-10)

    def test_gci_invalid_ratio(self):
        """Refinement ratio must be > 1."""
        with pytest.raises(ValueError, match="refinement_ratio"):
            compute_gci(0.01, 0.04, refinement_ratio=1.0, theoretical_order=2.0)

    def test_gci_no_order(self):
        """Must provide either apparent or theoretical order."""
        with pytest.raises(ValueError, match="Must provide"):
            compute_gci(0.01, 0.04, refinement_ratio=2.0)

    def test_gci_safety_factor(self):
        """Custom safety factor."""
        result = compute_gci(
            fine_value=0.01, coarse_value=0.04,
            refinement_ratio=2.0, theoretical_order=2.0,
            safety_factor=3.0,
        )
        assert result.gci_fine == pytest.approx(3.0, rel=1e-10)


# ---------------------------------------------------------------------------
# gci_three_grid
# ---------------------------------------------------------------------------

class TestGCIThreeGrid:
    def test_second_order_asymptotic(self):
        """Three-grid GCI for a second-order method in asymptotic range."""
        result = gci_three_grid(0.01, 0.04, 0.16, r=2.0)
        assert result["apparent_order"] == pytest.approx(2.0, abs=1e-10)
        assert result["expected_asymptotic_ratio"] == pytest.approx(0.25, rel=1e-10)
        assert result["in_asymptotic_range"] is True

    def test_third_order_asymptotic(self):
        """Three-grid GCI for a third-order method."""
        result = gci_three_grid(0.001, 0.008, 0.064, r=2.0)
        assert result["apparent_order"] == pytest.approx(3.0, abs=1e-10)
        assert result["expected_asymptotic_ratio"] == pytest.approx(0.125, rel=1e-10)
        assert result["in_asymptotic_range"] is True

    def test_first_order_asymptotic(self):
        """Three-grid GCI for a first-order method."""
        result = gci_three_grid(0.01, 0.02, 0.04, r=2.0)
        assert result["apparent_order"] == pytest.approx(1.0, abs=1e-10)
        assert result["expected_asymptotic_ratio"] == pytest.approx(0.5, rel=1e-10)
        assert result["in_asymptotic_range"] is True

    def test_non_asymptotic(self):
        """Pre-asymptotic data should flag in_asymptotic_range=False."""
        # Perturbed values that don't follow exact power law
        result = gci_three_grid(0.015, 0.04, 0.16, r=2.0)
        # The order won't be exactly 2, and asymptotic ratio will deviate
        assert abs(result["apparent_order"] - 2.0) > 0.1

    def test_gci_fine_values(self):
        """Check GCI values for known second-order case."""
        result = gci_three_grid(0.01, 0.04, 0.16, r=2.0)
        assert result["gci_fine_21"] == pytest.approx(1.25, rel=1e-10)
        assert result["gci_fine_32"] == pytest.approx(1.25, rel=1e-10)


# ---------------------------------------------------------------------------
# ConvergenceTable integration with GCI
# ---------------------------------------------------------------------------

class TestConvergenceTableGCI:
    def test_richardson_extrapolation(self):
        """ConvergenceTable should compute Richardson extrapolation."""
        from tensordg.converge import ConvergenceTable

        def mock_solver(ncells, degree):
            h = 1.0 / ncells
            err = 0.5 * h ** 2
            return {"absolute_l2": err, "dofs": ncells * 3}

        tbl = ConvergenceTable(
            problem="mock", degree=2, solver=mock_solver,
            base_ncells=4, n_levels=5, theoretical_order=2,
        )
        tbl.run()
        assert tbl.richardson_extrapolated is not None
        # Richardson extrapolated error estimate for exact p=2:
        # |e_fine - e_medium| / (r^p - 1) = |0.5/64^2 - 0.5/32^2| / 3
        # = 0.5 * (1/4096 - 1/1024) / 3 = 0.5 * (-3/4096) / 3 = 0.5/4096 = 1/8192
        expected = 0.5 / (64 ** 2)  # = error on finest grid / (r^p - 1) * (r^p - 1)/r^p
        # Actually: |e3 - e2| / (r^p - 1) where e3=0.5/64^2, e2=0.5/32^2
        # = 0.5 * |1/4096 - 1/1024| / 3 = 0.5 * 3/4096 / 3 = 0.5/4096
        assert tbl.richardson_extrapolated == pytest.approx(0.5 / 4096, rel=1e-10)

    def test_assert_order_passes(self):
        """Assertion should fail when observed order is below theoretical."""
        from tensordg.converge import ConvergenceTable

        def mock_solver(ncells, degree):
            h = 1.0 / ncells
            err = 0.5 * h ** 3  # third-order
            return {"absolute_l2": err, "dofs": ncells * 4}

        tbl = ConvergenceTable(
            problem="mock", degree=3, solver=mock_solver,
            base_ncells=4, n_levels=5, theoretical_order=4,
        )
        tbl.run()
        with pytest.raises(AssertionError):
            tbl.assert_all_orders(tolerance=0.3)

    def test_assert_order_correct_theory(self):
        """Assertion passes when theoretical order matches."""
        from tensordg.converge import ConvergenceTable

        def mock_solver(ncells, degree):
            h = 1.0 / ncells
            err = 0.5 * h ** (degree + 1)
            return {"absolute_l2": err, "dofs": ncells * (degree + 1)}

        for deg in [1, 2, 3]:
            tbl = ConvergenceTable(
                problem=f"mock_p{deg}", degree=deg, solver=mock_solver,
                base_ncells=4, n_levels=5, theoretical_order=deg + 1,
            )
            tbl.run()
            tbl.assert_all_orders(tolerance=0.3)
