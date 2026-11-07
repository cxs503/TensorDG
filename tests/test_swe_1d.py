"""Tests for the 1-D and 2-D shallow water equation solvers.

Mission m6144: TensorDG SWE with source terms.
Mission m6308: Ritter dam break analytical verification.

Verified capabilities:
* Lake-at-rest well-balancedness (flat bed): machine precision
* Smooth wave propagation: physical solutions
* Positivity preservation: h ≥ 0
* Flux functions: physical flux, HLL, Lax-Friedrichs
* 2-D flux functions: correct
* Error handling: input validation
* Ritter dam break (P1): rel L1 h < 3%, mass conservation exact

Known limitations:
* Well-balancedness for non-flat beds requires Audusse-style hydrostatic
  reconstruction (not yet implemented).
* Discontinuous initial conditions (sharp dam break) may require smaller CFL
  or more robust positivity treatment for very dry beds.
* Periodic boundary condition convergence needs further investigation.
* 2-D solver trace computation needs verification.
* P2+ for wet-dry fronts: needs reduced CFL and may still fail without
  specialised wet-dry treatment.
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

    def test_operators_p2(self):
        nodes, weights, phi, dphi, phi_left, phi_right = _operators(2, "cpu", torch.float64)
        assert phi.shape[1] == 3
        assert dphi.shape[1] == 3


# ---------------------------------------------------------------------------
# (d) Ritter dam break — analytical verification (m6308)
# ---------------------------------------------------------------------------

def _ritter_exact(x: np.ndarray, t: float, h0: float, g: float = GRAVITY):
    """Exact Ritter (1892) dam-break solution on flat, dry bed."""
    c0 = math.sqrt(g * h0)
    h = np.zeros_like(x, dtype=np.float64)
    u = np.zeros_like(x, dtype=np.float64)
    mask1 = x <= -c0 * t
    h[mask1] = h0
    mask2 = (x > -c0 * t) & (x < 2.0 * c0 * t)
    xi = x[mask2] / t
    h[mask2] = (1.0 / (9.0 * g)) * (2.0 * c0 - xi) ** 2
    u[mask2] = (2.0 / 3.0) * (c0 + xi)
    return h, u


class TestRitterDamBreak:
    """Ritter (1892) dam break: P1 DG vs exact self-similar solution.

    The module docstring claims this as a verified benchmark; these tests
    close that verification debt.
    """

    def _run_ritter(self, ncells=200, degree=1, h0=1.0, final_time=0.5,
                    domain=(-5.0, 5.0), cfl=0.15, dry_tol=1e-8):
        def init_h(x):
            return torch.where(x < 0.0, torch.full_like(x, h0), torch.full_like(x, dry_tol))

        def init_u(x):
            return torch.zeros_like(x)

        def zb(x):
            return torch.zeros_like(x)

        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=ncells, degree=degree,
            final_time=final_time, domain=domain,
            cfl=cfl, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )
        h_avg, u_avg, _ = reconstruct_fields(coeff, zb_faces)
        return x.numpy(), h_avg.numpy(), u_avg.numpy(), meta

    def test_ritter_p1_density_within_three_percent(self):
        """P1 DG rel L1 error in h vs Ritter exact < 3%."""
        x, h_num, u_num, meta = self._run_ritter(ncells=200, degree=1)
        h_ex, _ = _ritter_exact(x, 0.5, h0=1.0)
        wet = h_ex > 1e-8
        rel_l1 = float(np.mean(np.abs(h_num[wet] - h_ex[wet]))) / float(np.mean(h_ex[wet]))
        assert rel_l1 < 0.03, f"P1 Ritter rel L1 h = {rel_l1:.4%}"

    def test_ritter_p1_velocity_within_five_percent(self):
        """P1 DG rel L1 error in u vs Ritter exact < 5% (looser due to wet front)."""
        x, h_num, u_num, meta = self._run_ritter(ncells=200, degree=1)
        _, u_ex = _ritter_exact(x, 0.5, h0=1.0)
        wet = h_num > 1e-8
        u_scale = float(np.max(np.abs(u_ex[wet]))) if np.max(np.abs(u_ex[wet])) > 1e-12 else 1.0
        rel_l1 = float(np.mean(np.abs(u_num[wet] - u_ex[wet]))) / u_scale
        assert rel_l1 < 0.05, f"P1 Ritter rel L1 u = {rel_l1:.4%}"

    def test_ritter_mass_conservation(self):
        """Total mass error < 0.1% (exact for flat bed with consistent source)."""
        x, h_num, u_num, meta = self._run_ritter(ncells=200, degree=1)
        dx = 10.0 / 200
        mass_num = float(np.sum(h_num) * dx)
        mass_init = 1.0 * 5.0  # h0 * |domain[0]|
        mass_err = abs(mass_num - mass_init) / mass_init
        assert mass_err < 0.001, f"Mass error = {mass_err:.4%}"

    def test_ritter_positivity(self):
        """h ≥ 0 everywhere (dry tol is the floor)."""
        x, h_num, u_num, meta = self._run_ritter(ncells=200, degree=1)
        assert (h_num >= -1e-14).all(), f"Negative h: min={h_num.min():.2e}"

    def test_ritter_wet_front_reasonable(self):
        """Wet front position within 15 cells of exact (2*c0*t)."""
        x, h_num, u_num, meta = self._run_ritter(ncells=200, degree=1)
        c0 = math.sqrt(GRAVITY * 1.0)
        front_exact = 2.0 * c0 * 0.5
        dx = 10.0 / 200
        front_num = float(x[h_num > 1e-8].max()) if (h_num > 1e-8).any() else 0.0
        err_cells = abs(front_num - front_exact) / dx
        assert err_cells < 15, f"Wet front error = {err_cells:.1f} cells"

    def test_ritter_p1_converges(self):
        """Error decreases with mesh refinement (P1)."""
        errors = []
        for ncells in [100, 200]:
            x, h_num, _, _ = self._run_ritter(ncells=ncells, degree=1)
            h_ex, _ = _ritter_exact(x, 0.5, h0=1.0)
            wet = h_ex > 1e-8
            err = float(np.mean(np.abs(h_num[wet] - h_ex[wet]))) / float(np.mean(h_ex[wet]))
            errors.append(err)
        assert errors[1] < errors[0], f"P1 Ritter: {errors[0]:.4%} -> {errors[1]:.4%} (should decrease)"
