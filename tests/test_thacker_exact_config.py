"""Regression test: Thacker benchmark with exact slope + exact BC configuration.

This test locks in the finding from m7536 that the Thacker ~12% error is fixed
by using consistent source term and boundary condition treatment.
"""
import pytest
import torch
import numpy as np
import math

from tensordg import swe_exact
from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

# Thacker parameters
H0 = 1.0
AMP = 0.5
L = 1.0
G = 9.81
OMEGA = math.sqrt(2 * G * H0) / L


def thacker_bed(x):
    return -H0 * (1 - (x / L) ** 2)


def thacker_bed_slope(x):
    return 2 * H0 * x / L ** 2


def thacker_initial_h(x):
    h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)
    return torch.as_tensor(h, dtype=torch.float64)


def thacker_initial_u(x):
    return torch.zeros_like(x)


def _run_thacker(ncells, degree, t_end, use_exact_slope, use_exact_bc):
    """Helper to run Thacker case."""
    zb_slope_fn = thacker_bed_slope if use_exact_slope else None
    
    if use_exact_bc:
        def bc_left(t):
            fl, _ = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            return fl
        def bc_right(t):
            _, fr = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            return fr
    else:
        bc_left = None
        bc_right = None
    
    x, coeff, zb_faces, meta = solve_swe_1d(
        thacker_initial_h, thacker_initial_u, thacker_bed,
        ncells=ncells, degree=degree, final_time=t_end, domain=(-L, L),
        cfl=0.3, flux_kind="hll", limiter=None, rk_order=3, dtype=torch.float64,
        bc_left_flux=bc_left, bc_right_flux=bc_right, zb_slope=zb_slope_fn,
    )
    
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(x.numpy(), t_end, H0, AMP, L, G)
    zb = thacker_bed(x.numpy())
    eta_e = torch.as_tensor(h_e + zb, dtype=torch.float64)
    u_e_t = torch.as_tensor(u_e, dtype=torch.float64)
    
    err_eta = float(torch.sqrt(torch.mean((eta - eta_e) ** 2)) / torch.sqrt(torch.mean(eta_e ** 2))) * 100
    err_u = float(torch.sqrt(torch.mean((u - u_e_t) ** 2)) / torch.sqrt(torch.mean(u_e_t ** 2))) * 100
    
    dx = 2 * L / ncells
    vol = float(h.sum()) * dx
    eta_c = H0 + (AMP ** 2 / (4 * H0)) * math.sin(OMEGA * t_end) ** 2
    vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * L
    vol_err = abs(vol - vol_exact) / vol_exact * 100
    
    return err_eta, err_u, vol_err


class TestThackerExactConfiguration:
    """Tests for Thacker benchmark with exact slope + exact BC (m7536 fix)."""
    
    def test_exact_config_achieves_subpercent_accuracy(self):
        """Exact slope + exact BC achieves <0.1% error at N=80, t=0.3s."""
        err_eta, err_u, vol_err = _run_thacker(80, 1, 0.3, True, True)
        assert err_eta < 0.1, f"η error {err_eta:.4f}% should be <0.1%"
        assert err_u < 0.01, f"u error {err_u:.4f}% should be <0.01%"
        assert vol_err < 0.001, f"Volume error {vol_err:.4f}% should be <0.001%"
    
    def test_exact_config_volume_conservation(self):
        """Exact BC conserves volume to machine precision."""
        err_eta, err_u, vol_err = _run_thacker(40, 1, 0.3, True, True)
        assert vol_err < 1e-10, f"Volume error {vol_err:.6e}% should be ~0"
    
    def test_exact_config_convergence_rate(self):
        """Exact slope + exact BC achieves EOC=2 for P1 (theoretical optimal)."""
        err_20, _, _ = _run_thacker(20, 1, 0.3, True, True)
        err_40, _, _ = _run_thacker(40, 1, 0.3, True, True)
        err_80, _, _ = _run_thacker(80, 1, 0.3, True, True)
        
        eoc_1 = math.log(err_20 / err_40) / math.log(40 / 20)
        eoc_2 = math.log(err_40 / err_80) / math.log(80 / 40)
        
        assert abs(eoc_1 - 2.0) < 0.1, f"EOC {eoc_1:.2f} should be ~2.0"
        assert abs(eoc_2 - 2.0) < 0.1, f"EOC {eoc_2:.2f} should be ~2.0"
    
    def test_legacy_config_is_suboptimal_but_usable(self):
        """Legacy slope + transmissive BC has ~3% error (documented limitation)."""
        err_eta, err_u, vol_err = _run_thacker(40, 1, 0.3, False, False)
        # This is the "known defect" from m6377, but it's usable
        assert err_eta < 5.0, f"Legacy config η error {err_eta:.2f}% should be <5%"
        assert vol_err < 2.0, f"Legacy config volume error {vol_err:.2f}% should be <2%"
    
    def test_exact_slope_without_exact_bc_fails(self):
        """Exact slope + transmissive BC is catastrophically wrong (consistency check)."""
        err_eta, err_u, vol_err = _run_thacker(40, 1, 0.3, True, False)
        # This should be very bad (>50% error)
        assert err_eta > 30.0, f"Exact slope + transmissive BC should fail badly, got {err_eta:.2f}%"


class TestThackerShortTimeAccuracy:
    """Short-time accuracy tests (already passing in existing tests)."""
    
    def test_short_time_eta_within_three_percent(self):
        """At t=0.1s, even legacy config is <3% accurate."""
        err_eta, _, _ = _run_thacker(40, 1, 0.1, False, False)
        assert err_eta < 3.0, f"η error {err_eta:.2f}% should be <3% at t=0.1s"
    
    def test_short_time_velocity_within_three_percent(self):
        """At t=0.1s, velocity is <3% accurate."""
        _, err_u, _ = _run_thacker(40, 1, 0.1, False, False)
        assert err_u < 3.0, f"u error {err_u:.2f}% should be <3% at t=0.1s"
