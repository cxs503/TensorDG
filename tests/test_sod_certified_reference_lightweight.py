"""Lightweight integration test: SOD case audit structure verification.

This test verifies that the SOD certified-reference audit produces the expected
structure and that key invariants hold, *without* running the full FV-WENO5
solver (which is too heavy for CI).  It uses the exact Riemann solution and
synthetic FV ladders to exercise the engine's decision logic.

Why this test exists
--------------------
The full SOD audit (`benchmarks.certified_reference_sod.audit`) runs the
FV-WENO5 solver on three grids, which takes minutes.  This test provides a
fast CI gate that the audit structure, criteria, and engine decisions are
correct, using the exact solution as a synthetic FV ladder.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import certified_reference_sod as sod
from tensordg import certified_reference as cr


class TestSodAuditStructure:
    """Verify the SOD audit produces the expected structure."""

    def test_star_state_values(self):
        """Toro's Sod star state values."""
        st = sod.star_state()
        assert st["p_star"] == pytest.approx(0.303130, abs=1e-5)
        assert st["u_star"] == pytest.approx(0.927453, abs=1e-5)
        assert st["rho_star_left"] == pytest.approx(0.426319, abs=1e-5)
        assert st["rho_star_right"] == pytest.approx(0.265574, abs=1e-5)
        assert st["shock_speed"] == pytest.approx(1.752156, abs=1e-5)

    def test_exact_profiles_positive(self):
        """Exact profiles must be positive everywhere."""
        for n in [100, 200, 400, 800]:
            ex = sod.exact_profiles(n)
            assert np.all(ex["density"] > 0.0)
            assert np.all(ex["pressure"] > 0.0)
            assert np.all(np.isfinite(ex["density"]))
            assert np.all(np.isfinite(ex["pressure"]))
            assert np.all(np.isfinite(ex["velocity"]))

    def test_exact_profiles_converge(self):
        """Finer exact profiles should be closer to the continuum limit."""
        ex200 = sod.exact_profiles(200)
        ex400 = sod.exact_profiles(400)
        ex800 = sod.exact_profiles(800)
        # Coarsen ex400 and ex800 to 200 grid for comparison
        d200_400 = cr.rel_l1(ex200["density"], cr.coarsen(ex400["density"], 2))
        d200_800 = cr.rel_l1(ex200["density"], cr.coarsen(ex800["density"], 4))
        assert d200_800 < d200_400, "Finer profile should be closer to coarse"

    def test_exact_mass_conservation(self):
        """Exact mass at t=0.2 should approximately equal initial mass.

        The cell-average quadrature has O(1/N) error at discontinuities
        (shock and contact), so we use a generous tolerance.
        """
        for n in [100, 200, 400, 800]:
            ex = sod.exact_profiles(n)
            dx = float(ex["x"][1] - ex["x"][0])
            measured = float(np.sum(ex["density"]) * dx)
            # O(1/N) error at discontinuities: ~0.1% at N=100, ~0.01% at N=1000
            assert abs(measured - sod.INITIAL_MASS) < 0.001  # < 0.1% absolute


class TestSodEngineDecisions:
    """Verify the engine's decision logic on SOD-like data."""

    def _synthetic_fv_ladder(self, ncells_list):
        """Build a synthetic FV ladder from the exact solution."""
        levels = []
        for n in ncells_list:
            ex = sod.exact_profiles(n)
            levels.append({"ncells": n, "field": ex["density"].copy()})
        return levels

    def test_exact_ladder_certifies(self):
        """The exact solution ladder should certify (it's the continuum limit)."""
        levels = self._synthetic_fv_ladder([100, 200, 400])
        cert = cr.certify(levels, tolerance_percent=5.0)
        # The exact solution converges perfectly, so the band should be tiny
        assert cert["verdict"] == cr.CERTIFIED
        assert cert["band_percent"] is not None
        assert cert["band_percent"] < 1.0  # Should be very small

    def test_exact_ladder_band_covers_true_error(self):
        """The certified band should cover the true error vs the finest exact."""
        levels = self._synthetic_fv_ladder([100, 200, 400])
        cert = cr.certify(levels, tolerance_percent=5.0)
        # True error of the certified reference vs the 400-cell exact
        ex400 = sod.exact_profiles(400)
        ref_on_400 = cr.coarsen(cert["reference"], 1)  # Already on 200 grid
        # Coarsen 400 to 200 for comparison
        ex400_on_200 = cr.coarsen(ex400["density"], 2)
        true_err = cr.rel_l1(cert["reference"], ex400_on_200)
        true_err_pct = 100.0 * true_err
        # The band should cover the true error (or be very close)
        assert true_err_pct <= cert["band_percent"] + 0.01  # Small tolerance for numerical

    def test_stagnating_ladder_refused(self):
        """A stagnating ladder (identical levels) must be refused."""
        ex = sod.exact_profiles(100)
        flat = [{"ncells": 100, "field": ex["density"].copy()} for _ in range(3)]
        cert = cr.certify(flat, tolerance_percent=5.0)
        assert cert["verdict"] != cr.CERTIFIED

    def test_two_level_ladder_refused(self):
        """Two levels are insufficient for GCI."""
        levels = self._synthetic_fv_ladder([100, 200])
        cert = cr.certify(levels, tolerance_percent=5.0)
        assert cert["verdict"] == cr.WITHHELD_TOO_FEW_LEVELS


class TestSodOracleComparisons:
    """Verify oracle comparisons for the SOD case."""

    def test_plateau_density_left(self):
        """Left state density should be exactly 1.0."""
        ex = sod.exact_profiles(2000)
        x, rho = ex["x"], ex["density"]
        left_mean = float(np.mean(rho[x < 0.25]))
        assert left_mean == pytest.approx(1.0, abs=1e-8)

    def test_plateau_density_right(self):
        """Right state density should be exactly 0.125."""
        ex = sod.exact_profiles(2000)
        x, rho = ex["x"], ex["density"]
        st = sod.star_state()
        right_mean = float(np.mean(rho[x > st["shock_position"] + 0.02]))
        assert right_mean == pytest.approx(0.125, abs=1e-8)

    def test_shock_position(self):
        """Shock position should match the analytic value."""
        st = sod.star_state()
        ex = sod.exact_profiles(4000)
        found = sod._shock_position(ex["x"], ex["density"],
                                     x_min=st["contact_position"] + 0.01)
        assert abs(found - st["shock_position"]) < 2.0e-3

    def test_contact_position(self):
        """Contact position should be between rarefaction tail and shock."""
        st = sod.star_state()
        assert st["contact_position"] > 0.5  # Moved right from initial
        assert st["contact_position"] < st["shock_position"]
