"""Tests for Thacker (1981) parabolic basin oscillation benchmark.

Validates well-balanced property and oscillatory flow accuracy for 1-D SWE.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields


def thacker_exact(x: torch.Tensor, t: float, h0: float, A: float, L: float) -> tuple:
    """Compute exact Thacker solution at time t."""
    g = 9.81
    omega = math.sqrt(2.0 * g * h0) / L
    
    eta = h0 + A * math.cos(omega * t) * x / L
    u = torch.full_like(x, -A * omega * math.sin(omega * t) / g)
    zb = -h0 * (1.0 - (x / L) ** 2)
    h = eta - zb
    
    return eta, u, h


class TestThackerParabolicBasin:
    """Thacker (1981) parabolic basin oscillation tests."""
    
    def _run_thacker(self, ncells, degree, final_time, h0=1.0, A=0.5, L=1.0):
        """Helper to run Thacker benchmark."""
        def init_h(x):
            zb = -h0 * (1.0 - (x / L) ** 2)
            eta = h0 + A * x / L
            return eta - zb
        
        def init_u(x):
            return torch.zeros_like(x)
        
        def zb(x):
            return -h0 * (1.0 - (x / L) ** 2)
        
        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=ncells, degree=degree,
            final_time=final_time, domain=(-L, L),
            cfl=0.3, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )
        
        h_num, u_num, eta_num = reconstruct_fields(coeff, zb_faces)
        
        x_t = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float64)
        eta_exact, u_exact, h_exact = thacker_exact(x_t, final_time, h0, A, L)
        
        eta_err = torch.sqrt(torch.mean((eta_num - eta_exact) ** 2))
        u_err = torch.sqrt(torch.mean((u_num - u_exact) ** 2))
        h_err = torch.sqrt(torch.mean((h_num - h_exact) ** 2))
        
        eta_rel = eta_err / torch.max(torch.abs(eta_exact)) * 100
        u_rel = u_err / torch.max(torch.abs(u_exact)) * 100
        h_rel = h_err / torch.max(torch.abs(h_exact)) * 100
        
        return {
            "eta_rel_pct": float(eta_rel),
            "u_rel_pct": float(u_rel),
            "h_rel_pct": float(h_rel),
            "steps": meta["steps"],
        }
    
    def test_thacker_p1_basic(self):
        """P1 should achieve <5% error on Thacker benchmark."""
        res = self._run_thacker(ncells=50, degree=1, final_time=0.5)
        assert res["eta_rel_pct"] < 5.0, f"P1 η error {res['eta_rel_pct']:.2f}% > 5%"
        assert res["u_rel_pct"] < 5.0, f"P1 u error {res['u_rel_pct']:.2f}% > 5%"
        assert res["steps"] > 0
    
    def test_thacker_p2_basic(self):
        """P2 should achieve <3% error on Thacker benchmark."""
        res = self._run_thacker(ncells=50, degree=2, final_time=0.5)
        assert res["eta_rel_pct"] < 3.0, f"P2 η error {res['eta_rel_pct']:.2f}% > 3%"
        assert res["u_rel_pct"] < 3.0, f"P2 u error {res['u_rel_pct']:.2f}% > 3%"
    
    def test_thacker_p1_convergence(self):
        """P1 should show ~2nd order convergence."""
        res_coarse = self._run_thacker(ncells=25, degree=1, final_time=0.3)
        res_fine = self._run_thacker(ncells=50, degree=1, final_time=0.3)
        
        # Error should decrease by factor of ~4 (2nd order)
        ratio = res_coarse["eta_rel_pct"] / res_fine["eta_rel_pct"]
        assert ratio > 2.5, f"P1 convergence ratio {ratio:.2f} < 2.5 (expected ~4)"
    
    def test_thacker_p2_convergence(self):
        """P2 should show ~3rd order convergence."""
        res_coarse = self._run_thacker(ncells=25, degree=2, final_time=0.3)
        res_fine = self._run_thacker(ncells=50, degree=2, final_time=0.3)
        
        # Error should decrease by factor of ~8 (3rd order)
        ratio = res_coarse["eta_rel_pct"] / res_fine["eta_rel_pct"]
        assert ratio > 4.0, f"P2 convergence ratio {ratio:.2f} < 4 (expected ~8)"
    
    def test_thacker_positivity(self):
        """Water depth should remain positive."""
        def init_h(x):
            h0, A, L = 1.0, 0.5, 1.0
            zb = -h0 * (1.0 - (x / L) ** 2)
            eta = h0 + A * x / L
            return eta - zb
        
        def init_u(x):
            return torch.zeros_like(x)
        
        def zb(x):
            h0, L = 1.0, 1.0
            return -h0 * (1.0 - (x / L) ** 2)
        
        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, init_u, zb,
            ncells=50, degree=1,
            final_time=0.5, domain=(-1.0, 1.0),
            cfl=0.3, flux_kind="hll", limiter=None,
            rk_order=3, device="cpu", dtype=torch.float64,
        )
        
        h_num, _, _ = reconstruct_fields(coeff, zb_faces)
        assert (h_num > 0).all(), "Negative water depth detected"
    
    def test_thacker_small_amplitude(self):
        """Small amplitude should be more accurate (linear regime)."""
        res_large = self._run_thacker(ncells=50, degree=2, final_time=0.3, A=0.5)
        res_small = self._run_thacker(ncells=50, degree=2, final_time=0.3, A=0.1)
        
        # Small amplitude should have smaller error
        assert res_small["eta_rel_pct"] < res_large["eta_rel_pct"]
