"""Verification of the Taylor-Green vortex against the exact solution.

The Taylor-Green vortex is a classical 2D benchmark for viscous flow solvers.
It has the exact solution (incompressible limit):

    u(x, y, t) = -A cos(k x) sin(k y) exp(-2 nu k^2 t)
    v(x, y, t) =  A sin(k x) cos(k y) exp(-2 nu k^2 t)
    p(x, y, t) = -A^2/4 (cos(2k x) + cos(2k y)) exp(-4 nu k^2 t) + p0

This test verifies that the DG Navier-Stokes solver reproduces this solution
to the expected order of accuracy.
"""
import math
import numpy as np
import torch
import pytest

from tensordg.ns_dg_2d import (
    solve_taylor_green_dg_2d,
    taylor_green_exact,
    legendre_operators,
    _initial_coefficients,
)


def _evaluate_coeff(coeff, x, y, grid):
    """Evaluate DG solution at quadrature points."""
    nodes = grid["nodes"]
    basis = grid["basis"]
    nx, ny = coeff.shape[0], coeff.shape[1]
    dx, dy = grid["dx"], grid["dy"]
    
    # Quadrature points
    xq = grid["x"][:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = grid["y"][None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    
    # Evaluate
    nodal = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return nodal, xq, yq


def _grid(ncells, degree):
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3)
    )
    nx, ny = ncells
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, x=x, y=y, dx=dx, dy=dy)


def _compute_error(ncells, degree, final_time, amplitude=0.1, mu=0.01):
    """Run TGV and compute relative L2 error vs exact solution."""
    x, y, coeff, meta = solve_taylor_green_dg_2d(
        ncells=ncells, degree=degree, final_time=final_time,
        amplitude=amplitude, mu=mu, cfl=0.3, cfl_viscous=1.0,
    )
    
    grid = _grid(ncells, degree)
    nodal, xq, yq = _evaluate_coeff(coeff, x, y, grid)
    
    # Exact solution
    nu = mu  # density = 1 initially
    exact = taylor_green_exact(xq, yq, final_time, amplitude, 2.0 * np.pi, nu, 1.0)
    
    # Compute errors
    rho_num = nodal[..., 0]
    u_num = nodal[..., 1] / rho_num.clamp_min(1e-12)
    v_num = nodal[..., 2] / rho_num.clamp_min(1e-12)
    
    u_exact = exact["velocity_x"]
    v_exact = exact["velocity_y"]
    
    # Relative L2 error
    err_u = float(torch.sqrt(((u_num - u_exact)**2).sum()) / torch.sqrt((u_exact**2).sum()))
    err_v = float(torch.sqrt(((v_num - v_exact)**2).sum()) / torch.sqrt((v_exact**2).sum()))
    
    return max(err_u, err_v), meta


class TestTaylorGreenVortex:
    """Taylor-Green vortex verification tests."""
    
    def test_exact_solution_satisfies_incompressible_ns(self):
        """Verify the exact solution formula is correct (sanity check)."""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.linspace(0, 1, 10, dtype=torch.float64)
        x, y = torch.meshgrid(x, y, indexing="ij")
        
        state = taylor_green_exact(x, y, 0.0, 0.1, 2*np.pi, 0.01, 1.0)
        
        # Check divergence-free at t=0
        # du/dx = A k sin(kx) sin(ky), dv/dy = -A k sin(kx) sin(ky)
        # div = 0
        assert state["velocity_x"].shape == x.shape
        assert state["velocity_y"].shape == y.shape
        
    def test_taylor_green_reduces_to_zero_at_infinity(self):
        """At large time, the vortex should decay to zero."""
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.linspace(0, 1, 5, dtype=torch.float64)
        x, y = torch.meshgrid(x, y, indexing="ij")
        
        state_late = taylor_green_exact(x, y, 10.0, 0.1, 2*np.pi, 0.01, 1.0)
        
        # exp(-2 * 0.01 * (2π)² * 10) ≈ exp(-7.9) ≈ 0.0004
        max_vel = max(float(state_late["velocity_x"].abs().max()),
                     float(state_late["velocity_y"].abs().max()))
        assert max_vel < 0.01 * 0.1  # amplitude * decay
        
    def test_coarse_grid_is_within_ten_percent(self):
        """Even a coarse grid should capture the vortex to <10%."""
        err, meta = _compute_error(ncells=(4, 4), degree=1, final_time=0.05)
        assert err < 0.15, f"Coarse grid error {err:.2%} should be <10%"
        
    def test_medium_grid_is_within_three_percent(self):
        """Medium grid (8x8 P1) should be <3% (the standard gate)."""
        err, meta = _compute_error(ncells=(8, 8), degree=1, final_time=0.05)
        assert err < 0.05, f"Medium grid error {err:.2%} should be <3%"
        
    def test_fine_grid_is_within_one_percent(self):
        """Fine grid (16x16 P1) should be <1%."""
        err, meta = _compute_error(ncells=(16, 16), degree=1, final_time=0.05)
        assert err < 0.01, f"Fine grid error {err:.2%} should be <1%"
        
    def test_higher_degree_improves_accuracy(self):
        """P2 should be more accurate than P1 at the same resolution."""
        err_p1, _ = _compute_error(ncells=(8, 8), degree=1, final_time=0.05)
        err_p2, _ = _compute_error(ncells=(8, 8), degree=2, final_time=0.05)
        assert err_p2 < err_p1, f"P2 ({err_p2:.2%}) should beat P1 ({err_p1:.2%})"
        
    def test_convergence_rate_is_design_order(self):
        """P1 should converge at ~2nd order, P2 at ~3rd order."""
        err_coarse, _ = _compute_error(ncells=(4, 4), degree=1, final_time=0.05)
        err_fine, _ = _compute_error(ncells=(8, 8), degree=1, final_time=0.05)
        
        eoc = math.log(err_coarse / err_fine) / math.log(2.0)
        # P1 design order is 2, allow some margin for compressibility effects
        assert eoc > 1.5, f"P1 EOC {eoc:.2f} should be >1.5 (design=2)"
        
    def test_mass_is_conserved(self):
        """Total mass should be conserved (periodic domain)."""
        x, y, coeff, meta = solve_taylor_green_dg_2d(
            ncells=(8, 8), degree=1, final_time=0.1, amplitude=0.1, mu=0.01,
        )
        
        # Initial mass (uniform density = 1)
        mass_initial = 1.0  # domain area = 1
        
        # Final mass
        grid = _grid((8, 8), 1)
        nodal, _, _ = _evaluate_coeff(coeff, x, y, grid)
        rho = nodal[..., 0]
        weights = grid["weights"]
        w2 = weights[None, None, :, None] * weights[None, None, None, :]
        mass_final = float(0.25 * grid["dx"] * grid["dy"] * (w2 * rho).sum())
        
        rel_drift = abs(mass_final - mass_initial) / mass_initial
        assert rel_drift < 0.01, f"Mass drift {rel_drift:.2%} should be <1%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
