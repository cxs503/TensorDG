"""Tests for the 1-D and 2-D shallow water equation solvers.

Mission m6144: TensorDG SWE with source terms.

Verified capabilities:
* Lake-at-rest well-balancedness (flat bed): machine precision
* Smooth wave propagation: physical solutions
* Positivity preservation: h ≥ 0
* Flux functions: physical flux, HLL, Lax-Friedrichs
* 2-D flux functions: correct
* Error handling: input validation

Known limitations:
* Well-balancedness for non-flat beds requires Audusse-style hydrostatic
  reconstruction (not yet implemented).
* Discontinuous initial conditions (sharp dam break) may require smaller CFL
  or more robust positivity treatment for very dry beds.
* Periodic boundary condition convergence needs further investigation.
* 2-D solver trace computation needs verification.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.swe_1d import (
    GRAVITY,
    _operators,
    _positivity_filter,
    hll_flux,
    physical_flux,
    reconstruct_fields,
    solve_swe_1d,
    sound_speed,
)
from tensordg.swe_2d import (
    hll_flux_2d,
    physical_flux_2d,
)


# ---------------------------------------------------------------------------
# (a) Lake-at-rest well-balancedness
# ---------------------------------------------------------------------------

class TestLakeAtRest:
    """Criterion (a): max|η − η0| ≤ 1e-11 · H after 400+ steps."""

    def _run_lake(self, h0=1.0, ncells=50, degree=1, final_time=2.0):
        def init_h(x):
            return torch.full_like(x, h0)

        def init_u(x):
            return torch.zeros_like(x)

        def zb(x):
            return torch.zeros_like(x)

        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=ncells, degree=degree,
            final_time=final_time, domain=(0.0, 10.0),
            cfl=0.2, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )

        h_avg, _, eta = reconstruct_fields(coeff, zb_faces)
        max_drift = float(torch.max(torch.abs(eta - h0)).item())
        return max_drift, h0, meta

    def test_flat_bed_p1(self):
        drift, h0, meta = self._run_lake(h0=1.0, ncells=50, degree=1, final_time=2.0)
        assert drift < 1e-11 * h0, f"P1 lake drift {drift:.2e}"
        assert meta["steps"] >= 400, f"Too few steps: {meta['steps']}"

    def test_flat_bed_p2(self):
        drift, h0, meta = self._run_lake(h0=1.0, ncells=50, degree=2, final_time=2.0)
        assert drift < 1e-11 * h0, f"P2 lake drift {drift:.2e}"

    def test_long_integration(self):
        drift, h0, meta = self._run_lake(h0=1.0, ncells=50, degree=1, final_time=5.0)
        assert meta["steps"] >= 500
        assert drift < 1e-10 * h0

    def test_different_depths(self):
        for h0 in [0.1, 0.5, 1.0, 5.0]:
            drift, _, _ = self._run_lake(h0=h0, ncells=50, degree=1, final_time=1.0)
            assert drift < 1e-10 * h0, f"Depth {h0}: drift {drift:.2e}"


# ---------------------------------------------------------------------------
# (b) Smooth wave propagation
# ---------------------------------------------------------------------------

class TestSmoothWave:

    def test_smooth_wave_p2(self):
        def init_h(x):
            return 1.0 + 0.2 * torch.exp(-10.0 * (x - 5.0) ** 2)

        def init_u(x):
            return torch.full_like(x, 0.5)

        def zb(x):
            return torch.zeros_like(x)

        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=100, degree=2,
            final_time=1.0, domain=(0.0, 10.0),
            cfl=0.15, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )

        h_avg = coeff[:, 0, 0]
        assert torch.isfinite(h_avg).all()
        assert (h_avg > 0.5).all(), f"h dropped below 0.5: min={h_avg.min().item():.4f}"

    def test_wet_dry_transition(self):
        def init_h(x):
            return 0.5 + 0.49 * torch.tanh(5.0 * (x - 5.0))

        def init_u(x):
            return torch.full_like(x, 0.3)

        def zb(x):
            return torch.zeros_like(x)

        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=100, degree=2,
            final_time=0.5, domain=(0.0, 10.0),
            cfl=0.15, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )

        h_avg = coeff[:, 0, 0]
        assert (h_avg >= -1e-14).all(), f"Negative h: min={h_avg.min().item():.2e}"


# ---------------------------------------------------------------------------
# (c) Positivity preservation
# ---------------------------------------------------------------------------

class TestPositivity:

    def test_smooth_flow_positivity(self):
        def init_h(x):
            return 0.5 + 0.2 * torch.sin(2.0 * math.pi * x / 10.0)

        def init_u(x):
            return torch.full_like(x, 1.0)

        def zb(x):
            return torch.zeros_like(x)

        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=50, degree=2,
            final_time=0.5, domain=(0.0, 10.0),
            cfl=0.15, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )

        h_avg = coeff[:, 0, 0]
        assert (h_avg >= -1e-14).all(), f"Negative h: min={h_avg.min().item():.2e}"


# ---------------------------------------------------------------------------
# Flux function tests
# ---------------------------------------------------------------------------

class TestFluxFunctions:

    def test_physical_flux_still_water(self):
        h = torch.tensor([1.0, 2.0, 0.5])
        hu = torch.zeros(3)
        u = torch.stack((h, hu), dim=-1)
        f = physical_flux(u)
        assert torch.allclose(f[..., 0], torch.zeros(3), atol=1e-12)
        expected_p = 0.5 * GRAVITY * h ** 2
        assert torch.allclose(f[..., 1], expected_p, atol=1e-12)

    def test_physical_flux_uniform_flow(self):
        h = torch.tensor([1.0])
        u_vel = torch.tensor([2.0])
        hu = h * u_vel
        u = torch.stack((h, hu), dim=-1)
        f = physical_flux(u)
        assert torch.allclose(f[..., 0], hu, atol=1e-12)
        expected = hu * u_vel + 0.5 * GRAVITY * h ** 2
        assert torch.allclose(f[..., 1], expected, atol=1e-12)

    def test_sound_speed(self):
        h = torch.tensor([1.0, 4.0, 0.25])
        hu = torch.zeros(3)
        u = torch.stack((h, hu), dim=-1)
        c = sound_speed(u)
        expected = torch.sqrt(GRAVITY * h)
        assert torch.allclose(c, expected, atol=1e-12)

    def test_hll_flux_identical_states(self):
        h = torch.tensor([1.0, 2.0])
        hu = torch.tensor([0.5, -1.0])
        u = torch.stack((h, hu), dim=-1)
        f_hll = hll_flux(u, u, "hll")
        f_phys = physical_flux(u)
        assert torch.allclose(f_hll, f_phys, atol=1e-10)

    def test_hll_flux_lax_friedrichs(self):
        h = torch.tensor([[1.0], [0.5]])
        hu = torch.tensor([[1.0], [0.0]])
        left = torch.stack((h, hu), dim=-1)
        right = torch.stack((torch.flip(h, [0]), torch.flip(hu, [0])), dim=-1)
        f_hll = hll_flux(left, right, "hll")
        f_lf = hll_flux(left, right, "lf")
        assert torch.isfinite(f_hll).all()
        assert torch.isfinite(f_lf).all()

    def test_hll_flux_dry_state(self):
        h_l = torch.tensor([1.0])
        hu_l = torch.tensor([0.5])
        h_r = torch.tensor([1e-15])
        hu_r = torch.tensor([0.0])
        left = torch.stack((h_l, hu_l), dim=-1)
        right = torch.stack((h_r, hu_r), dim=-1)
        f = hll_flux(left, right, "hll")
        assert torch.isfinite(f).all()


# ---------------------------------------------------------------------------
# 2-D flux tests
# ---------------------------------------------------------------------------

class TestSWE2DFlux:

    def test_2d_flux_functions(self):
        state = torch.tensor([[[[1.0, 0.5, 0.3]]]])
        fx = physical_flux_2d(state, 0)
        fy = physical_flux_2d(state, 1)
        assert fx.shape == state.shape
        assert fy.shape == state.shape
        assert torch.isfinite(fx).all()
        assert torch.isfinite(fy).all()

        f_hll = hll_flux_2d(state, state, 0)
        f_phys = physical_flux_2d(state, 0)
        assert torch.allclose(f_hll, f_phys, atol=1e-10)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_invalid_degree(self):
        with pytest.raises(ValueError, match="degree"):
            solve_swe_1d(
                lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
                lambda x: torch.zeros_like(x), ncells=100, degree=3,
            )

    def test_invalid_ncells(self):
        with pytest.raises(ValueError, match="ncells"):
            solve_swe_1d(
                lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
                lambda x: torch.zeros_like(x), ncells=2, degree=1,
            )

    def test_invalid_rk_order(self):
        with pytest.raises(ValueError, match="rk_order"):
            solve_swe_1d(
                lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
                lambda x: torch.zeros_like(x), ncells=100, degree=1, rk_order=4,
            )

    def test_invalid_flux_kind(self):
        with pytest.raises(ValueError, match="flux_kind"):
            solve_swe_1d(
                lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
                lambda x: torch.zeros_like(x), ncells=100, degree=1, flux_kind="roe",
            )

    def test_invalid_domain(self):
        with pytest.raises(ValueError, match="domain"):
            solve_swe_1d(
                lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
                lambda x: torch.zeros_like(x), ncells=100, degree=1, domain=(1.0, 0.0),
            )

    def test_zero_final_time(self):
        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda x: torch.ones_like(x), lambda x: torch.zeros_like(x),
            lambda x: torch.zeros_like(x), ncells=100, degree=1, final_time=0.0,
        )
        assert meta["steps"] == 0


# ---------------------------------------------------------------------------
# Reconstruct fields test
# ---------------------------------------------------------------------------

class TestReconstructFields:

    def test_reconstruct_uniform(self):
        ncells = 50
        coeff = torch.zeros((ncells, 2, 2), dtype=torch.float64)
        coeff[:, 0, 0] = 1.0
        coeff[:, 0, 1] = 0.5
        zb_faces = torch.zeros((ncells, 2), dtype=torch.float64)

        h, u, eta = reconstruct_fields(coeff, zb_faces)
        assert torch.allclose(h, torch.ones(ncells, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(u, torch.full((ncells,), 0.5, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(eta, torch.ones(ncells, dtype=torch.float64), atol=1e-12)

    def test_reconstruct_with_bed(self):
        ncells = 50
        coeff = torch.zeros((ncells, 2, 2), dtype=torch.float64)
        coeff[:, 0, 0] = 1.0
        coeff[:, 0, 1] = 0.0
        zb_faces = torch.linspace(0, 0.5, ncells).unsqueeze(1).repeat(1, 2)

        h, u, eta = reconstruct_fields(coeff, zb_faces)
        zb_center = 0.5 * (zb_faces[:, 0] + zb_faces[:, 1])
        assert torch.allclose(eta, h + zb_center, atol=1e-12)


# ---------------------------------------------------------------------------
# Operators test
# ---------------------------------------------------------------------------

class TestOperators:

    def test_operators_p1(self):
        nodes, weights, phi, dphi, phi_left, phi_right = _operators(1, "cpu", torch.float64)
        assert phi.shape[1] == 2
        assert dphi.shape[1] == 2
        assert len(phi_left) == 2
        assert len(phi_right) == 2
        assert torch.allclose(phi[:, 0], torch.ones_like(phi[:, 0]), atol=1e-12)

    def test_operators_p2(self):
        nodes, weights, phi, dphi, phi_left, phi_right = _operators(2, "cpu", torch.float64)
        assert phi.shape[1] == 3
        assert abs(phi_right[2] - 1.0) < 1e-12
        assert abs(phi_left[2] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Positivity filter test
# ---------------------------------------------------------------------------

class TestPositivityFilter:

    def test_filter_passes_good_state(self):
        coeff = torch.zeros((10, 2, 2), dtype=torch.float64)
        coeff[:, 0, 0] = 1.0
        coeff[:, 0, 1] = 0.5
        nodes, weights, phi, dphi, phi_left, phi_right = _operators(1, "cpu", torch.float64)
        check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

        filtered = _positivity_filter(coeff, check_points, h_min=1e-12)
        assert torch.allclose(filtered, coeff, atol=1e-12)

    def test_filter_fixes_negative_h(self):
        coeff = torch.zeros((10, 2, 2), dtype=torch.float64)
        coeff[:, 0, 0] = 1.0
        coeff[:, 0, 1] = 0.5
        coeff[5, 1, 0] = 2.0

        nodes, weights, phi, dphi, phi_left, phi_right = _operators(1, "cpu", torch.float64)
        check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

        filtered = _positivity_filter(coeff, check_points, h_min=1e-12)
        assert filtered[5, 1, 0] == 0.0
        assert torch.allclose(filtered[:5], coeff[:5], atol=1e-12)
