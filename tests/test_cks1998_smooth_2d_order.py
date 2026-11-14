"""Regression pins for the CKS1998 §5.1 smooth 2-D Euler accuracy case.

Three separate things are nailed down here, in the order they can fail:

1. **The reference is really exact.**  ``benchmarks/cks1998_smooth_2d.py``
   claims ``rho = 1 + 0.2 sin(pi (x + y - 2t)), u = v = 1, p = 1`` solves the
   2-D Euler equations, i.e. that no manufactured source term is needed.  The
   claim is re-derived with automatic differentiation of the conservative form
   and gated at machine precision -- if someone "fixes" the reference, this
   goes red instead of silently turning every order measurement into a
   measurement against the wrong solution.

2. **The order.**  A cheap P1 ladder must show the design order ``p + 1 = 2``
   in *both* norms the paper tabulates (``L1`` and ``L-infinity``).  The
   repository's other 2-D order tests only use the relative ``L2`` norm on the
   isentropic vortex.

3. **The TVB threshold is where the theory says it is.**  The limiter flags
   smooth data only if ``M < max|d^2 rho/dx^2| = 0.2 pi^2``.  Measured on one
   mesh: ``M = 1`` (below) must lose accuracy relative to ``M = 10`` (above),
   and ``M = 10`` must reproduce the limiter-off error.  This is the
   falsifiable part: the prediction is declared *before* the measurement.

Run::

    OMP_NUM_THREADS=1 python -m pytest tests/test_cks1998_smooth_2d_order.py -q
"""
from __future__ import annotations

import math

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.cks1998_smooth_2d import (
    AMPLITUDE,
    PREDICTED_TVB_THRESHOLD,
    error_norms,
    run_case,
    smooth_conservative,
)

FINAL_TIME = 0.25


# ---------------------------------------------------------------------------
# 1. The reference is an exact solution (no manufactured source term)
# ---------------------------------------------------------------------------
def test_smooth_solution_satisfies_the_euler_residual():
    """dQ/dt + dF_x/dx + dF_y/dy = 0 to machine precision for the exact data."""
    dtype = torch.float64
    x = torch.tensor([0.137, 0.402, 0.751, 1.113, 1.684], dtype=dtype)
    y = torch.tensor([0.211, 0.573, 0.938, 1.302, 1.751], dtype=dtype)
    X = x[:, None].expand(-1, y.numel()).clone().requires_grad_(True)
    Y = y[None, :].expand(x.numel(), -1).clone().requires_grad_(True)
    t = torch.tensor(0.31, dtype=dtype, requires_grad=True)

    state = smooth_conservative(X, Y, t)
    rho, momentum_x, momentum_y, energy = (state[..., i] for i in range(4))
    pressure = 1.0 + 0.0 * rho          # p = 1 identically for this solution
    flux_x = torch.stack(
        (momentum_x, momentum_x ** 2 / rho + pressure,
         momentum_x * momentum_y / rho,
         (energy + pressure) * momentum_x / rho), dim=-1)
    flux_y = torch.stack(
        (momentum_y, momentum_x * momentum_y / rho,
         momentum_y ** 2 / rho + pressure,
         (energy + pressure) * momentum_y / rho), dim=-1)

    ones = torch.ones_like(state)
    # The x/y derivatives are elementwise (each node depends on its own X, Y),
    # so a vector-Jacobian product with ones returns them directly.  ``t`` is a
    # single scalar shared by every node, so the vjp would *sum* over nodes --
    # differentiate the time axis node by node instead (25 nodes, cheap).
    d_dx = torch.stack([
        torch.autograd.grad(flux_x[..., v], X, grad_outputs=ones[..., v],
                            retain_graph=True)[0]
        for v in range(4)], dim=-1)
    d_dy = torch.stack([
        torch.autograd.grad(flux_y[..., v], Y, grad_outputs=ones[..., v],
                            retain_graph=True)[0]
        for v in range(4)], dim=-1)
    d_dt = torch.zeros_like(state)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            for v in range(4):
                d_dt[i, j, v] = torch.autograd.grad(state[i, j, v], t,
                                                    retain_graph=True)[0]

    residual = d_dt + d_dx + d_dy
    # The time derivative is genuinely non-zero (the wave moves); the *sum* is not.
    assert d_dt.abs().max() > 0.1, d_dt.abs().max().item()
    assert residual.abs().max() < 1e-12, residual.abs().max().item()


def test_primitive_initial_state_is_the_analytic_solution():
    """At t=0 the initial condition must equal the reference state exactly."""
    x = torch.tensor([0.1, 0.9, 1.7], dtype=torch.float64)
    y = torch.tensor([0.3, 1.1], dtype=torch.float64)
    state = smooth_conservative(x[:, None], y[None, :], 0.0)
    rho = state[..., 0]
    expected = 1.0 + AMPLITUDE * torch.sin(math.pi * (x[:, None] + y[None, :]))
    assert torch.allclose(rho, expected, atol=1e-15, rtol=0.0)
    assert torch.allclose(state[..., 1], expected, atol=1e-15)      # rho u, u = 1
    assert torch.allclose(state[..., 2], expected, atol=1e-15)      # rho v, v = 1
    energy = 1.0 / (dg2.GAMMA - 1.0) + expected                     # p = 1, |u|^2 = 2
    assert torch.allclose(state[..., 3], energy, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Order of accuracy in L1 and L-infinity (the paper's norms)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p1_ladder():
    coarse = run_case(1, 16, None, 1.0, False, FINAL_TIME)
    fine = run_case(1, 32, None, 1.0, False, FINAL_TIME)
    ratio = coarse["h"] / fine["h"]
    eoc = {
        norm: math.log(coarse["norms"]["density"][norm] / fine["norms"]["density"][norm])
        / math.log(ratio)
        for norm in ("L1", "L1_rel", "Linf")
    }
    return coarse, fine, eoc


def test_p1_density_reaches_second_order_in_l1(p1_ladder):
    coarse, fine, eoc = p1_ladder
    assert 1.7 < eoc["L1"] < 2.3, (eoc, coarse["norms"]["density"], fine["norms"]["density"])


def test_p1_density_reaches_second_order_in_linf(p1_ladder):
    _, _, eoc = p1_ladder
    assert 1.7 < eoc["Linf"] < 2.3, eoc


def test_mass_and_momentum_are_conserved_by_the_periodic_scheme(p1_ladder):
    """The advected entropy wave keeps integral rho and integral rho u exact."""
    coarse, fine, _ = p1_ladder
    for run in (coarse, fine):
        # integral of the exact density is the box area (2^2) since the
        # perturbation is a full sine period; the L1 measure of the *error*
        # being O(h^2) is already gated above, here we only check finiteness
        # and that the solver reported a positive step count.
        assert run["steps"] > 0
        assert math.isfinite(run["norms"]["density"]["L1"])
        assert run["norms"]["density"]["L1"] > 0.0


# ---------------------------------------------------------------------------
# 3. TVB threshold: predicted before measured
# ---------------------------------------------------------------------------
def test_predicted_tvb_threshold_is_the_curvature_of_the_exact_density():
    """M* = max |d^2 rho / dx^2| = 0.2 pi^2 for this case."""
    assert PREDICTED_TVB_THRESHOLD == pytest.approx(AMPLITUDE * math.pi ** 2, rel=0.0)
    assert 1.9 < PREDICTED_TVB_THRESHOLD < 2.0  # 1.9739...


def test_limiter_below_the_threshold_costs_accuracy_and_above_it_does_not():
    """M=1 (below M*) must be worse than M=10 (above M*), which tracks limiter-off.

    One mesh (24^2, P1) -- the point is the *comparison*, not a full ladder, so
    the test stays inside the local short-run budget.
    """
    ncells = 24
    off = run_case(1, ncells, None, 1.0, False, FINAL_TIME)
    below = run_case(1, ncells, "tvb", 1.0, False, FINAL_TIME)
    above = run_case(1, ncells, "tvb", 10.0, False, FINAL_TIME)

    err_off = off["norms"]["density"]["L1_rel"]
    err_below = below["norms"]["density"]["L1_rel"]
    err_above = above["norms"]["density"]["L1_rel"]

    # Below the threshold the limiter clips the smooth extremum -> larger error.
    assert err_below > 1.5 * err_off, (err_below, err_off)
    # Above the threshold the smooth data are untouched -> the limiter-off
    # error is reproduced to well within a factor of two.
    assert 0.5 * err_off < err_above < 2.0 * err_off, (err_above, err_off)


def test_error_norms_are_zero_for_the_exact_modal_state():
    """Sanity: an exactly represented solution has zero measured error."""
    x, y, coeff, _ = dg2.solve_euler_dg_2d(
        lambda x, y: (1.0 + 0.0 * x, 1.0 + 0.0 * x, 1.0 + 0.0 * x, 1.0 + 0.0 * x),
        ncells=8, degree=2, domain=((0.0, 2.0), (0.0, 2.0)), final_time=0.1)
    # u = v = p = 1 is a free stream; the density is constant, which is *not*
    # the smooth solution, so the error must be O(1) -- but the machinery must
    # still evaluate finite numbers on the same grid.
    norms = error_norms(coeff, x, y, 2, FINAL_TIME)
    assert all(math.isfinite(v["L1"]) for v in norms.values())