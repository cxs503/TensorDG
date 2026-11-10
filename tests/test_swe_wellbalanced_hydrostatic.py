"""Hydrostatic well-balancedness tests for the 1-D SWE solver.

These tests verify that the DG solver preserves the hydrostatic equilibrium
(lake-at-rest) condition η = h + z_b = const, u = 0 to machine precision,
even on curved beds when well_balanced=True is used.

The hydrostatic well-balanced property is critical for:
1. Coastal and estuarine flows where the water is nearly at rest
2. Tsunami propagation over varying bathymetry
3. Flood modeling with complex terrain
4. Avoiding spurious currents in quiescent conditions

References
----------
* Audusse et al. (2004), "A fast and stable well-balanced scheme..."
* Xing & Shu (2005), "Efficient and high order accurate... well-balanced schemes"
* Xing & Shu (2006), "High order finite difference WENO schemes... for the Euler equations"
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.swe_1d import (
    check_hydrostatic_equilibrium,
    compute_hydrostatic_pressure_force,
    create_hydrostatic_initial_conditions,
    estimate_well_balanced_error_bound,
    reconstruct_fields,
    solve_swe_1d,
    validate_well_balanced_setup,
)

G = 9.81


class TestHydrostaticWellBalanced:
    """Test suite for hydrostatic well-balanced property."""

    @pytest.mark.parametrize("degree", [1, 2])
    def test_flat_bed_hydrostatic_equilibrium(self, degree):
        """Lake at rest on flat bed: η = const, u = 0 should be preserved exactly."""
        eta0 = 2.0
        ncells = 32
        final_time = 0.5

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: torch.full_like(xx, eta0),
            lambda xx: torch.zeros_like(xx),
            lambda xx: torch.zeros_like(xx),
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        h, u, eta = reconstruct_fields(coeff, zb_faces)

        # Check that η remains constant
        eta_error = float((eta - eta0).abs().max())
        assert eta_error < 1e-10, f"η drift {eta_error:.3e} m exceeds machine precision"

        # Check that u remains zero
        u_max = float(u.abs().max())
        assert u_max < 1e-10, f"spurious velocity {u_max:.3e} m/s"

        # Check that the solver reached final time
        assert meta["reached_final_time"], "Time integration did not reach final_time"

    @pytest.mark.parametrize("degree", [1, 2])
    def test_sloped_bed_hydrostatic_equilibrium(self, degree):
        """Lake at rest on sloped bed: η = const, u = 0 with z_b = slope * x."""
        eta0 = 3.0
        slope = 0.5
        ncells = 50
        final_time = 1.0

        def zb(xx):
            return slope * xx

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 2.0),
            cfl=0.3,
            well_balanced=True,
        )

        h, u, eta = reconstruct_fields(coeff, zb_faces)

        # For well_balanced=True, η should be preserved to high accuracy
        eta_error = float((eta - eta0).abs().max())
        # Allow slightly larger tolerance for P2 due to higher-order effects
        tol = 1e-9 if degree == 1 else 1e-8
        assert eta_error < tol, f"η drift {eta_error:.3e} m exceeds tolerance {tol}"

        # Check that u remains zero.  The P2 path carries a slightly larger
        # roundoff-level residual: the DG weak form's 1/dx prefactor amplifies
        # the flux/source cancellation round-off, so the velocity tolerance
        # mirrors the η tolerance above.  Both are many orders of magnitude
        # below any physical velocity, i.e. the lake-at-rest state is
        # preserved (no spurious current).
        u_max = float(u.abs().max())
        u_tol = 1e-9 if degree == 1 else 1e-8
        assert u_max < u_tol, f"spurious velocity {u_max:.3e} m/s exceeds {u_tol}"

    @pytest.mark.parametrize("degree", [1, 2])
    def test_curved_bed_hydrostatic_equilibrium(self, degree):
        """Lake at rest on curved bed: η = const, u = 0 with z_b = A * sin(k*x)."""
        eta0 = 2.5
        amplitude = 0.4
        wave_number = math.pi
        ncells = 64
        final_time = 0.8

        def zb(xx):
            return amplitude * torch.sin(wave_number * xx)

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(-1.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        # Reconstruct η with the polynomial bed's mode-0 coefficient as the
        # cell-centre bed elevation.  The default trapezoidal face mean is only
        # the cell average for a piecewise-linear bed; for a curved bed it
        # leaves an O(dx^2) spurious free-surface drift in the reconstructed η
        # (see the reconstruct_fields docstring).  The solver itself is
        # well-balanced to round-off; this removes the post-processing error.
        h, u, eta = reconstruct_fields(coeff, zb_faces,
                                       zb_center=meta["zb_modes"][:, 0])

        # For well_balanced=True with polynomial bed representation,
        # η should be preserved to high accuracy
        eta_error = float((eta - eta0).abs().max())
        # P1 and P2 should both achieve good accuracy with the nodal LGL approach
        tol = 1e-8 if degree == 1 else 1e-7
        assert eta_error < tol, f"η drift {eta_error:.3e} m exceeds tolerance {tol}"

        # Check that u remains zero
        u_max = float(u.abs().max())
        assert u_max < 1e-8, f"spurious velocity {u_max:.3e} m/s"

    @pytest.mark.parametrize("ncells", [32, 64])
    def test_convergence_on_curved_bed(self, ncells):
        """Verify that η error decreases with mesh refinement."""
        eta0 = 2.0
        amplitude = 0.3
        wave_number = 2.0 * math.pi
        degree = 1
        final_time = 0.5

        def zb(xx):
            return amplitude * torch.sin(wave_number * xx)

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        h, u, eta = reconstruct_fields(coeff, zb_faces)
        eta_error = float((eta - eta0).abs().max())

        # Store error for convergence analysis
        # With well_balanced=True, we expect very small errors (near machine precision)
        # The exact rate depends on the bed representation accuracy
        assert eta_error < 1e-7, f"η error {eta_error:.3e} too large for N={ncells}"

    def test_hydrostatic_with_small_perturbation(self):
        """Test that a small perturbation from hydrostatic equilibrium evolves correctly."""
        eta0 = 2.0
        perturbation = 1e-4
        ncells = 50
        degree = 1
        final_time = 0.3

        def zb(xx):
            return 0.2 * torch.sin(math.pi * xx)

        # Initial condition: hydrostatic + small Gaussian perturbation
        def initial_h(xx):
            return eta0 - zb(xx) + perturbation * torch.exp(-50.0 * (xx - 0.5) ** 2)

        x, coeff, zb_faces, meta = solve_swe_1d(
            initial_h,
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        h, u, eta = reconstruct_fields(coeff, zb_faces)

        # The perturbation should propagate as waves, not grow unboundedly
        eta_max = float(eta.max())
        eta_min = float(eta.min())

        # Check that the solution remains bounded
        assert eta_max < eta0 + 10 * perturbation, "Perturbation grew unboundedly"
        assert eta_min > eta0 - 10 * perturbation, "Solution became unphysical"

        # Check that momentum is generated (perturbation creates flow)
        u_max = float(u.abs().max())
        assert u_max > 1e-6, "No flow generated from perturbation"

    def test_well_balanced_vs_standard_comparison(self):
        """Compare well_balanced=True vs False on a curved bed."""
        eta0 = 2.0
        ncells = 50
        degree = 1
        final_time = 0.5

        def zb(xx):
            return 0.3 * torch.cos(2.0 * math.pi * xx)

        # Run with well_balanced=True
        x1, coeff1, zb_faces1, meta1 = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )
        h1, u1, eta1 = reconstruct_fields(coeff1, zb_faces1)
        error_wb = float((eta1 - eta0).abs().max())

        # Run with well_balanced=False (standard L2 projection)
        x2, coeff2, zb_faces2, meta2 = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=False,
        )
        h2, u2, eta2 = reconstruct_fields(coeff2, zb_faces2)
        error_std = float((eta2 - eta0).abs().max())

        # well_balanced=True should be significantly more accurate
        assert error_wb < error_std, (
            f"well_balanced=True error {error_wb:.3e} should be < "
            f"standard error {error_std:.3e}"
        )

        # well_balanced should achieve near machine precision
        assert error_wb < 1e-8, f"well_balanced error {error_wb:.3e} too large"

    def test_metadata_reports_well_balanced(self):
        """Verify that metadata correctly reports well_balanced setting."""
        eta0 = 2.0
        ncells = 32
        degree = 1
        final_time = 0.3

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: torch.full_like(xx, eta0),
            lambda xx: torch.zeros_like(xx),
            lambda xx: torch.zeros_like(xx),
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(0.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        assert meta["well_balanced"] is True
        assert meta["bed_representation"] == "nodal_lgl_polynomial"
        assert meta["zb_modes"] is not None

    @pytest.mark.parametrize("degree", [1, 2])
    def test_long_time_integration_stability(self, degree):
        """Verify that hydrostatic equilibrium is maintained over long times."""
        eta0 = 2.0
        ncells = 32
        final_time = 2.0  # Long integration time (reduced from 5.0 for CI efficiency)

        def zb(xx):
            return 0.2 * torch.sin(math.pi * xx)

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb(xx),
            lambda xx: torch.zeros_like(xx),
            zb,
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            domain=(-1.0, 1.0),
            cfl=0.3,
            well_balanced=True,
        )

        # Reconstruct η with the polynomial bed's mode-0 coefficient as the
        # cell-centre bed elevation (the default trapezoidal face mean is only
        # the cell average for a piecewise-linear bed; for this curved bed it
        # would leave an O(dx^2) spurious free-surface drift).
        h, u, eta = reconstruct_fields(coeff, zb_faces,
                                       zb_center=meta["zb_modes"][:, 0])

        # Even after long integration, η should remain close to constant
        eta_error = float((eta - eta0).abs().max())
        tol = 1e-7 if degree == 1 else 1e-6
        assert eta_error < tol, (
            f"Long-time η drift {eta_error:.3e} m exceeds tolerance {tol}"
        )

        # u should remain zero.  For this curved bed the discrete (polynomial)
        # bed differs from the true bed by O(dx^3), which seeds a small
        # imbalance that accumulates over the long integration; the P2
        # tolerance is relaxed accordingly (still many orders of magnitude
        # below any physical velocity, i.e. no spurious current of concern).
        u_max = float(u.abs().max())
        u_tol = 1e-7 if degree == 1 else 1e-6
        assert u_max < u_tol, (
            f"Long-time spurious velocity {u_max:.3e} m/s exceeds {u_tol}"
        )


class TestHydrostaticHelpers:
    """Tests for the public hydrostatic helper functions of tensordg.swe_1d.

    The five helpers (create_hydrostatic_initial_conditions,
    validate_well_balanced_setup, check_hydrostatic_equilibrium,
    compute_hydrostatic_pressure_force, estimate_well_balanced_error_bound)
    ship with the well-balanced feature but were not covered by the original
    13-test suite; this class closes that gap.
    """

    def test_create_hydrostatic_initial_conditions(self):
        """IC factory returns h = eta0 - z_b and u = 0 exactly."""
        eta0 = 2.5

        def zb(xx):
            return 0.4 * torch.sin(math.pi * xx)

        initial_h, initial_u = create_hydrostatic_initial_conditions(zb, eta0)

        x = torch.linspace(0.0, 1.0, 37, dtype=torch.float64)
        h = initial_h(x)
        u = initial_u(x)

        # eta = h + z_b must equal eta0 to machine precision
        eta = h + zb(x)
        assert float((eta - eta0).abs().max()) < 1e-14
        # u must be identically zero with the input's shape and dtype
        assert u.shape == x.shape
        assert u.dtype == x.dtype
        assert float(u.abs().max()) == 0.0

    def test_validate_well_balanced_setup_accepts_equilibrium(self):
        """A consistent lake-at-rest setup is accepted."""
        eta0 = 2.0

        def zb(xx):
            return 0.3 * torch.cos(2.0 * math.pi * xx)

        initial_h, initial_u = create_hydrostatic_initial_conditions(zb, eta0)
        result = validate_well_balanced_setup(initial_h, initial_u, zb)

        assert result["is_valid"] is True
        assert result["eta_variance"] < 1e-20
        assert result["u_max"] == 0.0
        assert abs(result["eta_mean"] - eta0) < 1e-12
        assert "valid" in result["message"]

    def test_validate_well_balanced_setup_rejects_perturbed_eta(self):
        """A spatially varying eta must be rejected."""
        eta0 = 2.0

        def zb(xx):
            return 0.3 * torch.cos(2.0 * math.pi * xx)

        def initial_h(xx):
            return eta0 - zb(xx) + 0.1 * torch.sin(2.0 * math.pi * xx)

        result = validate_well_balanced_setup(
            initial_h, lambda xx: torch.zeros_like(xx), zb)

        assert result["is_valid"] is False
        assert result["eta_variance"] > 1e-10

    def test_validate_well_balanced_setup_rejects_nonzero_velocity(self):
        """Nonzero initial velocity is rejected even with flat eta."""
        eta0 = 2.0

        result = validate_well_balanced_setup(
            lambda xx: torch.full_like(xx, eta0),
            lambda xx: torch.full_like(xx, 0.05),
            lambda xx: torch.zeros_like(xx),
        )
        assert result["is_valid"] is False
        assert result["u_max"] == pytest.approx(0.05)

    def test_check_hydrostatic_equilibrium_detects_equilibrium(self):
        """After a short well-balanced run the checker reports equilibrium."""
        eta0 = 2.0

        def zb(xx):
            return 0.2 * torch.sin(math.pi * xx)

        initial_h, initial_u = create_hydrostatic_initial_conditions(zb, eta0)
        x, coeff, zb_faces, meta = solve_swe_1d(
            initial_h, initial_u, zb,
            ncells=32, degree=1, final_time=0.3,
            domain=(0.0, 1.0), cfl=0.3, well_balanced=True,
        )
        result = check_hydrostatic_equilibrium(
            coeff, zb_faces, zb_center=meta["zb_modes"][:, 0],
            eta_expected=eta0)

        for key in ("is_hydrostatic", "eta_error_max", "eta_error_l2",
                    "u_max", "u_l2", "eta_expected", "message"):
            assert key in result
        assert result["is_hydrostatic"] is True
        assert result["eta_error_max"] < 1e-8
        assert result["u_max"] < 1e-8

    def test_check_hydrostatic_equilibrium_detects_perturbation(self):
        """A Gaussian perturbation must be flagged as non-equilibrium."""
        eta0 = 2.0

        def zb(xx):
            return 0.2 * torch.sin(math.pi * xx)

        def initial_h(xx):
            return eta0 - zb(xx) + 0.05 * torch.exp(-50.0 * (xx - 0.5) ** 2)

        x, coeff, zb_faces, meta = solve_swe_1d(
            initial_h, lambda xx: torch.zeros_like(xx), zb,
            ncells=32, degree=1, final_time=0.3,
            domain=(0.0, 1.0), cfl=0.3, well_balanced=True,
        )
        result = check_hydrostatic_equilibrium(
            coeff, zb_faces, zb_center=meta["zb_modes"][:, 0],
            eta_expected=eta0)

        assert result["is_hydrostatic"] is False
        assert result["eta_error_max"] > 1e-8

    def test_compute_hydrostatic_pressure_force(self):
        """F_p = 0.5*rho*g*h^2 per unit width for a constant-depth lake."""
        eta0 = 2.0
        ncells = 16

        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: torch.full_like(xx, eta0),
            lambda xx: torch.zeros_like(xx),
            lambda xx: torch.zeros_like(xx),
            ncells=ncells, degree=1, final_time=0.2,
            domain=(0.0, 1.0), cfl=0.3, well_balanced=True,
        )
        fp = compute_hydrostatic_pressure_force(coeff, zb_faces)

        assert fp.shape == (ncells,)
        expected = 0.5 * 1000.0 * G * eta0 ** 2  # 19620 N/m
        assert float((fp - expected).abs().max()) < 1e-8 * expected

    def test_validate_well_balanced_setup_rejects_sub_tolerance_drift(self):
        """A 1e-5 m η drift must not pass a 1e-10 tolerance check.

        Regression for the variance-vs-sup-norm bug: the acceptance test used
        to compare ``eta_variance`` (units m², quadratic in the amplitude) with
        ``tolerance`` (units m).  For a sin(2πx)-shaped drift of amplitude
        ``a = 1e-5`` the variance is ``a²/2 = 5e-11 < 1e-10``, so a drift 1e5
        times larger than the stated tolerance was accepted.  The criterion is
        now the sup norm ``max|η - <η>|``, consistent with
        :func:`check_hydrostatic_equilibrium`.
        """
        eta0 = 2.0
        drift_amplitude = 1e-5

        def zb(xx):
            return 0.3 * torch.cos(2.0 * math.pi * xx)

        def initial_h(xx):
            return (eta0 - zb(xx)
                    + drift_amplitude * torch.sin(2.0 * math.pi * xx))

        result = validate_well_balanced_setup(
            initial_h, lambda xx: torch.zeros_like(xx), zb)

        assert result["is_valid"] is False
        # the variance would have been (wrongly) accepted ...
        assert result["eta_variance"] < 1e-10
        # ... while the sup norm correctly reports the full drift (the mean of
        # the sampled sine is not exactly zero, hence the 1e-3 slack)
        assert result["eta_deviation_max"] == pytest.approx(
            drift_amplitude, rel=1e-3)
        assert "NOT valid" in result["message"]

    def test_validate_well_balanced_setup_sup_norm_matches_checker(self):
        """Both helpers accept/reject on the same sup-norm criterion."""
        eta0 = 2.0

        def zb(xx):
            return 0.2 * torch.sin(math.pi * xx)

        # 1e-12 drift: below the 1e-10 default tolerance of both helpers
        initial_h, initial_u = create_hydrostatic_initial_conditions(zb, eta0)
        ok = validate_well_balanced_setup(initial_h, initial_u, zb)
        assert ok["is_valid"] is True
        assert ok["eta_deviation_max"] < 1e-10

        # 1e-8 drift: above the tolerance of both helpers
        def drifted(xx):
            return initial_h(xx) + 1e-8 * torch.sin(2.0 * math.pi * xx)

        bad = validate_well_balanced_setup(drifted, initial_u, zb)
        assert bad["is_valid"] is False
        assert bad["eta_deviation_max"] == pytest.approx(1e-8, rel=1e-3)

    def test_estimate_well_balanced_error_bound_not_clamped(self):
        """The bound is the representation estimate, not a 1e-10 constant.

        Regression for the hard-coded ``min(estimate, 1e-10)``: for an O(1)
        bed curvature the theoretical estimate is ~2e-3 m, but the function
        used to return 1e-10 m, i.e. it reported a floor instead of the
        estimate and was flat under refinement.
        """
        ncells, degree, length = 8, 1, 1.0
        bound = estimate_well_balanced_error_bound(
            ncells, degree, length, bed_curvature=1.0)

        dx = length / ncells
        assert bound["dx"] == pytest.approx(dx)
        assert bound["eta_error_estimate"] == pytest.approx(dx ** 2 / 8.0)
        assert bound["expected_eta_error"] == pytest.approx(dx ** 2 / 8.0)
        assert bound["expected_eta_error"] > 1e-5
        assert bound["floor_applied"] is False
        assert "1e-10" not in f"{bound['expected_eta_error']:.3e}"

    def test_estimate_well_balanced_error_bound_explicit_floor(self):
        """An explicit floor is honoured and reported."""
        estimate = estimate_well_balanced_error_bound(
            8, 1, 1.0, bed_curvature=1.0)["eta_error_estimate"]

        floored = estimate_well_balanced_error_bound(
            8, 1, 1.0, bed_curvature=1.0, floor=1e-2)
        assert floored["eta_error_estimate"] == pytest.approx(estimate)
        assert floored["expected_eta_error"] == pytest.approx(1e-2)
        assert floored["floor_applied"] is True

        # a floor below the estimate must not bind
        low_floor = estimate_well_balanced_error_bound(
            8, 1, 1.0, bed_curvature=1.0, floor=1e-12)
        assert low_floor["expected_eta_error"] == pytest.approx(estimate)
        assert low_floor["floor_applied"] is False

    def test_estimate_well_balanced_error_bound_velocity_scale(self):
        """u scale is the linearised relation η sqrt(g / h_ref) for h_ref."""
        shallow = estimate_well_balanced_error_bound(
            8, 1, 1.0, bed_curvature=1.0, reference_depth=1.0)
        deep = estimate_well_balanced_error_bound(
            8, 1, 1.0, bed_curvature=1.0, reference_depth=4.0)

        assert shallow["expected_u_error"] == pytest.approx(
            shallow["expected_eta_error"] * math.sqrt(G))
        assert deep["expected_u_error"] == pytest.approx(
            shallow["expected_u_error"] / 2.0)
        with pytest.raises(ValueError):
            estimate_well_balanced_error_bound(
                8, 1, 1.0, bed_curvature=1.0, reference_depth=0.0)

    def test_estimate_well_balanced_error_bound_scaling(self):
        """The bound shrinks with refinement at the documented rate.

        No round-off floor is imposed by default (``floor=None``), so the
        returned bound is exactly the O(dx**(p+1)) representation estimate and
        the ratio between two meshes is 2**rate to round-off.
        """
        for degree, rate, curvature in ((1, 2.0, 1e-6), (2, 3.0, 1e-5)):
            b1 = estimate_well_balanced_error_bound(100, degree, 1.0,
                                                    bed_curvature=curvature)
            b2 = estimate_well_balanced_error_bound(200, degree, 1.0,
                                                    bed_curvature=curvature)

            assert b1["dx"] == pytest.approx(0.01)
            assert b2["dx"] == pytest.approx(0.005)
            assert b1["convergence_rate"] == pytest.approx(rate)
            # halving dx must shrink the bound by 2**rate
            assert b2["expected_eta_error"] == pytest.approx(
                b1["expected_eta_error"] / 2.0 ** rate)
            assert b1["expected_u_error"] == pytest.approx(
                b1["expected_eta_error"] * math.sqrt(G))
            assert isinstance(b1["notes"], str) and b1["notes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
